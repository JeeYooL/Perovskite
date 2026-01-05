import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

# 머신러닝 & 설명 가능한 AI(XAI) 관련 라이브러리
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import shap

# -------------------------------------------------------------------
# 페이지 설정
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Perovskite AI Lab (XGBoost + SHAP)",
    page_icon="🧪",
    layout="wide"
)

# 스타일 커스텀
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1, h2, h3 { color: #003366; font-family: 'Arial', sans-serif; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 함수 정의
# -------------------------------------------------------------------

def load_data(uploaded_files):
    """업로드된 파일들을 하나의 데이터프레임으로 병합"""
    all_dfs = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                # encoding 관련 에러 방지를 위한 시도
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='cp949')
                all_dfs.append(df)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                all_dfs.append(df)
        except Exception as e:
            st.error(f"파일 로드 중 오류 발생 ({uploaded_file.name}): {e}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None

def preprocess_data(df):
    """데이터 전처리: 유연한 타겟 컬럼 찾기, 결측치 제거, MLB, 컬럼명 정제"""
    
    # 1. 컬럼명 공백 제거 (실수 방지)
    df.columns = df.columns.str.strip()
    
    # 2. 타겟 컬럼('PCE') 자동 탐색
    # 사용자가 말한 '특수문자 이슈'를 피하기 위해 정확한 이름 대신 'PCE'가 포함된 컬럼을 찾습니다.
    target_column = 'PCE (%)'
    
    if target_column not in df.columns:
        # 'PCE'라는 단어가 포함된 컬럼을 후보로 찾음 (대소문자 구분 없이)
        pce_candidates = [c for c in df.columns if 'PCE' in c.upper()]
        
        if len(pce_candidates) > 0:
            target_column = pce_candidates[0] # 첫 번째 후보를 타겟으로 설정
            st.info(f"ℹ️ '{target_column}' 컬럼을 타겟 값(효율)으로 자동 인식했습니다.")
        else:
            st.error("데이터에서 'PCE' 관련 컬럼(예: PCE (%), PCE)을 찾을 수 없습니다. CSV 파일을 확인해주세요.")
            return None, None, None, None

    # 타겟 값이 없는 행 제거
    df_cleaned = df.dropna(subset=[target_column]).copy()
    
    # Data Leakage 방지: 결과값 컬럼 제외
    # PCE 외에 전압, 전류 등 결과 지표들도 제거해야 합니다.
    drop_keywords = ['PCE', 'Voc', 'Jsc', 'FF', 'Rs', 'Rsh', 'Scan Direction', 'Sample', 'File']
    
    # 삭제할 컬럼 찾기
    cols_to_drop = []
    for col in df_cleaned.columns:
        # 타겟 컬럼은 당연히 y로 가므로 X에서는 제외
        if col == target_column:
            cols_to_drop.append(col)
            continue
            
        # 결과 지표들 포함 여부 확인
        for kw in drop_keywords:
            if kw in col:
                cols_to_drop.append(col)
                break
                
    # Unnamed 컬럼 제거
    cols_to_drop += [c for c in df_cleaned.columns if 'Unnamed' in c]
    
    X_raw = df_cleaned.drop(columns=cols_to_drop, errors='ignore')
    y = df_cleaned[target_column]
    
    # 수치형/범주형 분리
    X_numeric = X_raw.select_dtypes(exclude=['object'])
    X_categorical = X_raw.select_dtypes(include=['object'])
    
    all_processed_dfs = [X_numeric]

    for col in X_categorical.columns:
        # 'FAI + MACl' 같은 복합 조성을 개별 성분으로 분리
        binarized = X_categorical[col].fillna('').astype(str).str.get_dummies(sep=' + ')
        binarized = binarized.add_prefix(f"{col}_")
        all_processed_dfs.append(binarized)
        
    X_processed = pd.concat(all_processed_dfs, axis=1).fillna(0)
    
    # [중요] XGBoost 호환성을 위해 **모든 컬럼명**에서 특수문자 제거
    # 이 작업은 데이터를 다 만든 후(X_processed)에 수행하므로,
    # 초기에 'PCE (%)' 컬럼을 찾는 과정에는 영향을 주지 않습니다.
    X_processed.columns = X_processed.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(r'\s+', '_', regex=True)
    
    return X_processed, y, df_cleaned, X_raw

# -------------------------------------------------------------------
# 메인 UI
# -------------------------------------------------------------------

st.title("🧪 Perovskite AI Lab: XGBoost & SHAP Analysis")
st.markdown("""
최신 연구 트렌드(Science, Nature Energy 등)를 반영하여 **XGBoost(고성능 부스팅)** 모델과 **SHAP(설명 가능한 AI)** 기법을 적용했습니다.
""")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.header("1. Data Upload")
    uploaded_files = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], accept_multiple_files=True)
    
    st.header("2. Model Settings")
    test_size = st.slider("Test Set Ratio", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("CV Folds", 2, 10, 5)
    
    st.markdown("---")
    st.info("💡 **XGBoost**는 페로브스카이트 공정 데이터와 같은 정형 데이터(Tabular Data)에서 최고의 성능을 보입니다.")

if uploaded_files:
    raw_df = load_data(uploaded_files)
    
    if raw_df is not None:
        st.write(f"✅ Loaded **{len(raw_df)}** samples.")
        
        # 간단한 데이터 미리보기 (디버깅용)
        with st.expander("데이터 미리보기 (상위 5행)"):
            st.dataframe(raw_df.head())
        
        if st.button("🚀 Run AI Analysis (Train & Explain)"):
            with st.spinner('Preprocessing data & optimizing XGBoost model...'):
                
                # 1. 전처리 (유연한 컬럼 탐색 적용됨)
                X, y, df_clean, X_raw_origin = preprocess_data(raw_df)
                
                if X is not None:
                    # 2. 데이터 분할
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    # 3. XGBoost 모델 설정
                    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
                    
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0]
                    }
                    
                    grid_search = GridSearchCV(
                        xgb_model, 
                        param_grid, 
                        cv=cv_folds, 
                        scoring='neg_mean_absolute_error',
                        verbose=1,
                        error_score='raise'
                    )
                    
                    try:
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        
                        # [Tab 1: 성능]
                        st.subheader("1. Model Performance")
                        col1, col2, col3 = st.columns(3)
                        
                        y_pred = best_model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        cv_r2 = cross_val_score(best_model, X, y, cv=cv_folds, scoring='r2').mean()
                        
                        col1.metric("Test R² Score", f"{r2:.4f}")
                        col2.metric("Mean Absolute Error", f"{mae:.4f} %")
                        col3.metric("Cross-Validation R²", f"{cv_r2:.4f}")
                        
                        # 예측 그래프
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.scatter(y_test, y_pred, alpha=0.6, color='#2c3e50', edgecolors='w')
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                        ax.set_xlabel("Experimental PCE (%)")
                        ax.set_ylabel("Predicted PCE (%)")
                        ax.set_title("Prediction Accuracy")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # [Tab 2: SHAP 분석 (XAI)]
                        st.markdown("---")
                        st.subheader("2. Explainable AI (SHAP Analysis)")
                        st.markdown("""
                        **SHAP Summary Plot**은 각 공정 변수가 효율에 미치는 영향을 보여줍니다.
                        * **점의 색상**: 변수의 값 (빨강=높음, 파랑=낮음)
                        * **X축 위치**: 효율에 미치는 영향 (오른쪽=효율 증가, 왼쪽=효율 감소)
                        """)
                        
                        with st.spinner("Calculating SHAP values..."):
                            explainer = shap.Explainer(best_model, X_train)
                            shap_values = explainer(X_test)
                            
                            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values, X_test, show=False)
                            st.pyplot(fig_shap)
                            
                            st.markdown("**Feature Importance Ranking (SHAP based)**")
                            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                            st.pyplot(fig_bar)

                        # [Tab 3: 최적화 제안]
                        st.markdown("---")
                        st.subheader("3. Optimization Suggestions")
                        
                        best_idx = y.idxmax()
                        st.success(f"현재 데이터셋 최고 효율: **{y.max():.2f}%** (Sample ID: {best_idx})")
                        
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': np.abs(shap_values.values).mean(axis=0)
                        }).sort_values('importance', ascending=False)
                        
                        top_features = feature_importance['feature'].head(5).tolist()
                        
                        st.markdown("#### 🔬 핵심 제어 변수 (Top 5)")
                        best_recipe = df_clean.loc[best_idx]
                        suggestions = []
                        
                        for feat in top_features:
                            # 원본 컬럼 찾기 (정제된 이름 -> 원본 이름 매핑)
                            original_col = feat
                            # 1. 완벽 매칭 시도
                            if feat in X_raw_origin.columns:
                                original_col = feat
                            else:
                                # 2. 부분 매칭 (특수문자 처리 전후 비교)
                                for raw_col in X_raw_origin.columns:
                                    cleaned_raw = re.sub(r'[^\w\s]', '_', str(raw_col))
                                    cleaned_raw = re.sub(r'\s+', '_', cleaned_raw)
                                    if cleaned_raw == feat:
                                        original_col = raw_col
                                        break
                            
                            current_val = best_recipe.get(original_col, "N/A")
                            
                            suggestions.append({
                                "Rank": top_features.index(feat) + 1,
                                "Feature (AI Name)": feat,
                                "Original Column": original_col,
                                "Best Value": current_val,
                                "Action": "SHAP 그래프 참조하여 미세 조정"
                            })
                        
                        st.table(pd.DataFrame(suggestions))

                    except Exception as e:
                        st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
                        st.write("힌트: 데이터의 양이 너무 적거나(최소 20개 이상 권장), XGBoost가 처리할 수 없는 데이터 형식이 포함되었을 수 있습니다.")

else:
    st.info("👈 Please upload your data file to start.")

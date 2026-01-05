import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------------------------------------------
# 페이지 설정
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Perovskite ML Optimizer V2.0",
    page_icon="⚗️",
    layout="wide"
)

# 스타일 커스텀
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
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
                df = pd.read_csv(uploaded_file)
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
    """데이터 전처리: 결측치 제거, 타겟 분리, MLB(Multi-Label Binarization)"""
    target_column = 'PCE (%)'
    
    # 타겟 값이 없는 행 제거
    if target_column not in df.columns:
        st.error(f"데이터에 '{target_column}' 컬럼이 없습니다.")
        return None, None, None, None

    df_cleaned = df.dropna(subset=[target_column]).copy()
    
    # Data Leakage 방지를 위한 결과값 컬럼 제외
    drop_cols = [
        'PCE (%)', 'Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'Rs (Ω·cm²)', 'Rsh (Ω·cm²)',
        'Sample', 'File', 'Scan Direction', 'Unnamed: 0'
    ]
    # 실제 데이터셋에 존재하는 컬럼만 drop
    cols_to_drop = [c for c in drop_cols if c in df_cleaned.columns]
    
    X_raw = df_cleaned.drop(columns=cols_to_drop, errors='ignore')
    y = df_cleaned[target_column]
    
    # 문자열/수치형 분리
    X_numeric = X_raw.select_dtypes(exclude=['object'])
    X_categorical = X_raw.select_dtypes(include=['object'])
    
    # 범주형 데이터 처리 (MLB 방식: 'A + B' -> A, B 각각 1)
    all_processed_dfs = [X_numeric]
    
    # 전처리 과정 로그용
    processed_cols_log = []

    for col in X_categorical.columns:
        # 결측치는 빈 문자열로 처리 후 분리
        binarized = X_categorical[col].fillna('').astype(str).str.get_dummies(sep=' + ')
        
        # 컬럼명에 원래 변수명 접두사 추가 (예: Solvent_DMF)
        binarized = binarized.add_prefix(f"{col}_")
        
        # 특수문자 정제 (컬럼명 깨짐 방지)
        binarized.columns = binarized.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(r'\s+', '_', regex=True)
        
        all_processed_dfs.append(binarized)
        processed_cols_log.append(col)
        
    X_processed = pd.concat(all_processed_dfs, axis=1).fillna(0)
    
    return X_processed, y, df_cleaned, X_raw

# -------------------------------------------------------------------
# UI 구성
# -------------------------------------------------------------------

st.title("⚗️ Perovskite 공정 최적화 및 성능 예측 AI (V2.0)")
st.markdown("---")

# 사이드바: 데이터 업로드 및 설정
with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded_files = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요 (여러 개 가능)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.header("2. 모델 설정")
    test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("교차 검증 (K-Fold) 횟수", 2, 10, 5)
    
    st.markdown("---")
    st.info("💡 **Tip**: 'A + B' 형태의 텍스트 데이터는 자동으로 분리되어 학습됩니다.")

if uploaded_files:
    # 1. 데이터 로드
    raw_df = load_data(uploaded_files)
    
    if raw_df is not None:
        st.write(f"✅ 총 **{len(raw_df)}**개의 샘플이 로드되었습니다.")
        
        # 데이터 미리보기
        with st.expander("원본 데이터 미리보기"):
            st.dataframe(raw_df.head())

        # 2. 전처리 및 학습 버튼
        if st.button("🚀 AI 모델 학습 및 최적화 시작"):
            with st.spinner('데이터 전처리 및 모델 최적화 중입니다... (시간이 소요될 수 있습니다)'):
                
                # 전처리 실행
                X, y, df_clean, X_raw_origin = preprocess_data(raw_df)
                
                if X is not None:
                    st.success(f"전처리 완료! 학습에 사용될 피처 수: **{X.shape[1]}개**")
                    
                    # 3. 데이터 분할
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # 4. GridSearchCV (하이퍼파라미터 튜닝)
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                    
                    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
                    grid_search = GridSearchCV(
                        rf, 
                        param_grid, 
                        cv=cv_folds, 
                        scoring='neg_mean_absolute_error',
                        verbose=0
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    
                    st.markdown("---")
                    
                    # 5. 결과 리포트 섹션 (2단 레이아웃)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 모델 성능 평가")
                        
                        # 교차 검증 점수
                        cv_scores = cross_val_score(best_model, X, y, cv=cv_folds, scoring='r2')
                        st.metric("5-Fold CV 평균 R²", f"{cv_scores.mean():.4f}")
                        
                        # 테스트 셋 점수
                        y_pred = best_model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        st.write(f"**테스트 세트 R²:** {r2:.4f}")
                        st.write(f"**평균 오차 (MAE):** {mae:.4f} %PCE")
                        st.caption(f"최적 파라미터: {grid_search.best_params_}")
                        
                        # 실제값 vs 예측값 그래프
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', color='#2980b9')
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                        ax.set_xlabel("Actual PCE (%)")
                        ax.set_ylabel("Predicted PCE (%)")
                        ax.set_title("Actual vs Predicted")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                    with col2:
                        st.subheader("🔑 중요 공정 변수 (Top 20)")
                        
                        # 중요도 추출
                        importances = best_model.feature_importances_
                        feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
                        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)
                        
                        # 중요도 그래프
                        fig2, ax2 = plt.subplots(figsize=(6, 8))
                        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax2)
                        ax2.set_title("Feature Importance")
                        st.pyplot(fig2)

                    st.markdown("---")
                    
                    # 6. 실험 방향 제안
                    st.header("💡 AI 기반 실험 제안")
                    st.write("현재 데이터셋 내 **최고 효율 장치**의 레시피와 **중요 변수**를 기반으로 분석한 결과입니다.")
                    
                    best_idx = y.idxmax()
                    best_val = y.max()
                    
                    st.success(f"🏆 현재 최고 효율: **{best_val:.2f}%** (Sample ID: {best_idx})")
                    
                    # 최고 효율 레시피 추출
                    best_recipe = df_clean.loc[best_idx]
                    
                    # 중요 변수 상위 5개에 대한 제안 생성
                    suggestions = []
                    for feat in feat_imp_df['Feature'].head(5):
                        # 원본 컬럼 찾기 (MLB 전의 이름 추적)
                        # 예: Solvent_DMF -> Solvent
                        original_col = next((c for c in X_raw_origin.columns if feat.startswith(c)), None)
                        
                        if original_col:
                            val = best_recipe.get(original_col, "N/A")
                            suggestions.append({
                                "중요 변수 (Feature)": feat,
                                "원인 변수": original_col,
                                "최고 효율 조건 값": val,
                                "제안": "이 변수는 성능에 매우 중요합니다. 위 값을 중심으로 미세 조정(Fine-tuning) 하세요."
                            })
                        else:
                            # 수치형 변수일 경우
                            val = best_recipe.get(feat, "N/A")
                            suggestions.append({
                                "중요 변수 (Feature)": feat,
                                "원인 변수": feat,
                                "최고 효율 조건 값": val,
                                "제안": "수치형 중요 변수입니다. 이 값 주변으로 범위를 좁혀 최적화하세요."
                            })
                    
                    st.table(pd.DataFrame(suggestions))

else:
    st.info("왼쪽 사이드바에서 데이터 파일을 업로드해주세요.")
    st.markdown("""
    ### 👋 환영합니다!
    이 앱은 페로브스카이트 태양전지 공정 데이터를 분석하여 최적의 레시피를 제안합니다.
    
    **데이터 파일 형식:**
    - `.csv` 또는 `.xlsx`
    - 필수 컬럼: `PCE (%)`
    - 그 외 공정 변수들 (예: `Temp`, `Solvent`, `Additive` 등)
    """)

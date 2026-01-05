import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

# 머신러닝 라이브러리
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 모델
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel

# 설명 가능한 AI
import shap

# -------------------------------------------------------------------
# 페이지 설정
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Perovskite AI Lab V5",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1, h2, h3 { color: #003366; font-family: 'Arial', sans-serif; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; }
    .stAlert { padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 함수 정의
# -------------------------------------------------------------------

def load_data(uploaded_files):
    """파일 로드 및 병합"""
    all_dfs = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='cp949')
                all_dfs.append(df)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                all_dfs.append(df)
        except Exception as e:
            st.error(f"파일 로드 오류 ({uploaded_file.name}): {e}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None

def clean_column_names(df):
    """컬럼명 특수문자 제거 (XGBoost 등 호환성 확보)"""
    df.columns = df.columns.str.strip()
    return df

def detect_target_column(df):
    """타겟 컬럼(PCE) 자동 감지"""
    candidates = [c for c in df.columns if 'PCE' in c.upper()]
    if candidates:
        return candidates[0]
    return df.columns[-1] if not df.empty else None

def preprocess_data(df, target_column):
    """전처리: 타겟 분리, 결측치 제거, 인코딩, 형변환"""
    
    # 1. 타겟값 결측치 제거
    df_cleaned = df.dropna(subset=[target_column]).copy()
    
    if len(df_cleaned) == 0:
        return None, None, None, None

    # 2. 결과 지표 제거 (Data Leakage 방지)
    drop_keywords = ['PCE', 'Voc', 'Jsc', 'FF', 'Rs', 'Rsh', 'Scan', 'Sample', 'File', 'Unnamed']
    cols_to_drop = []
    for col in df_cleaned.columns:
        if col == target_column: continue
        for kw in drop_keywords:
            if kw in col:
                cols_to_drop.append(col)
                break
    
    X_raw = df_cleaned.drop(columns=cols_to_drop, errors='ignore')
    y = df_cleaned[target_column]
    
    # 3. MLB / One-Hot Encoding
    X_numeric = X_raw.select_dtypes(exclude=['object'])
    X_categorical = X_raw.select_dtypes(include=['object'])
    
    all_processed = [X_numeric]
    for col in X_categorical.columns:
        # 'A + B' 형태 분리
        binarized = X_categorical[col].fillna('').astype(str).str.get_dummies(sep=' + ')
        binarized = binarized.add_prefix(f"{col}_")
        all_processed.append(binarized)
        
    X_processed = pd.concat(all_processed, axis=1).fillna(0)
    
    # 4. 특수문자 제거 (컬럼명)
    X_processed.columns = X_processed.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(r'\s+', '_', regex=True)
    
    # 5. [중요] 모든 데이터를 float형으로 강제 변환 (에러 방지)
    try:
        X_processed = X_processed.astype(float)
    except ValueError:
        # 변환 실패 시 (혹시 모를 문자열 잔재) 강제 변환
        for col in X_processed.columns:
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)

    return X_processed, y, df_cleaned, X_raw

# -------------------------------------------------------------------
# 메인 UI
# -------------------------------------------------------------------

st.title("🧪 Perovskite AI Lab V5")
st.write("재료 탐색 및 공정 최적화를 위한 지능형 분석 플랫폼")
st.markdown("---")

# 1. 사이드바: 데이터 업로드
with st.sidebar:
    st.header("📂 1. Data Input")
    uploaded_files = st.file_uploader("CSV/Excel 업로드", type=['csv', 'xlsx'], accept_multiple_files=True)
    
    st.markdown("---")
    st.caption("Developed based on recent PV ML studies (Nature Energy, 2024)")

if uploaded_files:
    raw_df = load_data(uploaded_files)
    
    if raw_df is not None:
        raw_df = clean_column_names(raw_df)
        st.write(f"✅ **{len(raw_df)}**개의 샘플 데이터가 로드되었습니다.")
        
        with st.expander("데이터 미리보기"):
            st.dataframe(raw_df.head())
        
        st.markdown("---")
        
        # ----------------------------------------------------------------
        # 2. 사용자 설정 (타겟 & 모델 선택)
        # ----------------------------------------------------------------
        st.header("⚙️ 2. Analysis Settings")
        
        col_set1, col_set2, col_set3 = st.columns(3)
        
        # Step 1: 타겟 변수 선택
        with col_set1:
            default_target = detect_target_column(raw_df)
            try:
                default_idx = list(raw_df.columns).index(default_target) if default_target else 0
            except:
                default_idx = 0
                
            target_col = st.selectbox(
                "목표 타겟 (Target Variable)", 
                options=raw_df.columns, 
                index=default_idx,
                help="예측하고자 하는 값 (보통 효율 PCE)"
            )

        # Step 2: 모델 선택
        with col_set2:
            model_options = [
                "XGBoost (Recommended)",
                "Random Forest (Robust)",
                "Gaussian Process (Bayesian Opt.)"
            ]
            model_choice = st.selectbox(
                "사용할 ML 모델", 
                options=model_options,
                help="데이터가 적다면 Gaussian Process나 Random Forest를 권장합니다."
            )
        
        # Step 3: 테스트 비율
        with col_set3:
            test_ratio = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05)

        # 경고 및 가이드 메시지
        data_len = len(raw_df)
        if data_len < 20:
            st.warning(f"⚠️ 데이터가 **{data_len}개**로 매우 적습니다.")
            if "XGBoost" in model_choice:
                st.error("🛑 XGBoost는 데이터가 너무 적을 때(20개 미만) 작동하지 않거나 과적합될 수 있습니다. **Gaussian Process** 또는 **Random Forest**를 선택하세요.")
            else:
                st.info("💡 적은 데이터셋(Small Data)에 강한 모델을 선택하셨군요. 분석을 진행합니다.")

        # ----------------------------------------------------------------
        # 3. 분석 실행
        # ----------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 AI 분석 및 최적화 시작 (Run Analysis)", type="primary"):
            
            # 모델 실행 조건 체크 (데이터 너무 적고 XGBoost면 중단 가능하나, 일단 진행하되 try-catch)
            with st.spinner(f"데이터 전처리 및 {model_choice.split()[0]} 최적화 중..."):
                
                # 전처리
                X, y, df_clean, X_raw_origin = preprocess_data(raw_df, target_col)
                
                if X is None:
                    st.error("전처리 실패: 타겟 컬럼에 유효한 데이터가 없습니다.")
                else:
                    # 데이터 분할
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
                    
                    # 모델 초기화 및 학습
                    model = None
                    is_tree_model = False
                    
                    try:
                        # -----------------------
                        # A. XGBoost
                        # -----------------------
                        if "XGBoost" in model_choice:
                            is_tree_model = True
                            xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
                            param_grid = {
                                'n_estimators': [100, 200] if len(X) < 50 else [100, 300, 500],
                                'max_depth': [3, 5],
                                'learning_rate': [0.05, 0.1]
                            }
                            search = GridSearchCV(xgb_reg, param_grid, cv=3, scoring='neg_mean_absolute_error')
                            search.fit(X_train, y_train)
                            model = search.best_estimator_
                            st.caption(f"Best Params: {search.best_params_}")

                        # -----------------------
                        # B. Random Forest
                        # -----------------------
                        elif "Random Forest" in model_choice:
                            is_tree_model = True
                            rf_reg = RandomForestRegressor(random_state=42, n_jobs=-1)
                            param_grid = {
                                'n_estimators': [100, 200],
                                'max_depth': [None, 10],
                                'min_samples_leaf': [1, 2]
                            }
                            search = GridSearchCV(rf_reg, param_grid, cv=3, scoring='neg_mean_absolute_error')
                            search.fit(X_train, y_train)
                            model = search.best_estimator_
                            st.caption(f"Best Params: {search.best_params_}")

                        # -----------------------
                        # C. Gaussian Process (Bayesian Opt Logic)
                        # -----------------------
                        elif "Gaussian Process" in model_choice:
                            # 데이터 스케일링 (GP는 스케일에 민감)
                            scaler_X = StandardScaler()
                            X_train_scaled = scaler_X.fit_transform(X_train)
                            X_test_scaled = scaler_X.transform(X_test)
                            
                            # 커널 정의 (RBF + WhiteKernel for noise)
                            kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
                            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
                            gp.fit(X_train_scaled, y_train)
                            model = gp
                            
                            # 예측 함수 오버라이딩 (스케일링 포함)
                            def gp_predict(X_input):
                                X_input_scaled = scaler_X.transform(X_input)
                                return gp.predict(X_input_scaled, return_std=False)
                            
                            model.predict = gp_predict # 덕 타이핑
                            
                            # GP는 Test set 예측 시 std도 반환받아서 불확실성 시각화 가능
                            y_pred, y_std = gp.predict(X_test_scaled, return_std=True)

                        # 공통 예측 및 평가
                        if "Gaussian Process" not in model_choice:
                            y_pred = model.predict(X_test)
                        
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        # ----------------------------------------------------------------
                        # 4. 결과 리포트
                        # ----------------------------------------------------------------
                        st.success("✅ 분석 완료!")
                        
                        # Tab 구성
                        tab1, tab2, tab3 = st.tabs(["📊 성능 평가", "🔍 중요도 분석 (XAI)", "💡 최적화 제안"])
                        
                        with tab1:
                            c1, c2 = st.columns(2)
                            c1.metric("R² Score (정확도)", f"{r2:.4f}")
                            c2.metric("MAE (평균 오차)", f"{mae:.4f}")
                            
                            fig, ax = plt.subplots(figsize=(6, 5))
                            ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Data')
                            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal')
                            if "Gaussian Process" in model_choice:
                                ax.errorbar(y_test, y_pred, yerr=y_std, fmt='none', alpha=0.2, ecolor='gray', label='Uncertainty')
                            
                            ax.set_xlabel(f"Actual {target_col}")
                            ax.set_ylabel(f"Predicted {target_col}")
                            ax.set_title(f"{model_choice.split()[0]} Regression Result")
                            ax.legend()
                            st.pyplot(fig)

                        with tab2:
                            st.subheader("Feature Analysis")
                            if is_tree_model:
                                st.write("**SHAP (SHapley Additive exPlanations)** 분석 결과")
                                explainer = shap.Explainer(model, X_train)
                                shap_values = explainer(X_test)
                                
                                fig_shap, ax_shap = plt.subplots()
                                shap.summary_plot(shap_values, X_test, show=False)
                                st.pyplot(fig_shap)
                                
                                # 중요도 추출
                                importances = np.abs(shap_values.values).mean(axis=0)
                            else:
                                st.info("Gaussian Process는 SHAP 대신 ARD(Automatic Relevance Determination) 또는 상관계수를 기반으로 중요도를 추정할 수 있습니다. (현재 버전은 상관계수 표시)")
                                # 간단한 상관계수 히트맵
                                corr = X.copy()
                                corr['Target'] = y
                                corr_matrix = corr.corr()[['Target']].sort_values(by='Target', key=abs, ascending=False).head(10)
                                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
                                importances = np.abs(corr.corr()[target_col].drop(target_col).values)

                        with tab3:
                            st.subheader("실험 조건 최적화 제안")
                            best_idx = y.idxmax()
                            st.success(f"현재 최고 성능: **{y.max():.4f}** (Sample ID: {best_idx})")
                            
                            # 중요 변수 Top 5
                            if is_tree_model:
                                feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Imp': importances})
                            else:
                                feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Imp': list(importances)[:len(X.columns)]}) # 단순 매핑
                                
                            top_feats = feat_imp_df.sort_values('Imp', ascending=False).head(5)['Feature'].tolist()
                            
                            best_recipe = df_clean.loc[best_idx]
                            suggestions = []
                            for feat in top_feats:
                                # 원본 컬럼 찾기
                                orig = feat
                                for raw_c in X_raw_origin.columns:
                                    # 전처리된 이름과 매칭되는지 확인 (단순 포함 관계)
                                    if re.sub(r'[^\w]', '_', str(raw_c)) in feat:
                                        orig = raw_c
                                        break
                                
                                val = best_recipe.get(orig, best_recipe.get(feat, "N/A"))
                                suggestions.append({
                                    "중요 변수": feat,
                                    "현재 최고값": val,
                                    "제안": "이 변수의 주변 값을 탐색(Exploration) 하세요."
                                })
                            
                            st.table(pd.DataFrame(suggestions))
                            
                    except Exception as e:
                        st.error(f"모델 학습 중 치명적인 오류 발생: {e}")
                        st.write("해결책: 데이터 샘플 수를 늘리거나, 범주형 변수의 종류를 줄여보세요.")

else:
    st.info("👈 왼쪽 사이드바에서 데이터 파일을 업로드해주세요.")

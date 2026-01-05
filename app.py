import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# --- [1] 페이지 설정 ---
st.set_page_config(page_title="SolarCell ML Optimizer", layout="wide", page_icon="🤖")

st.title("🤖 Solar Cell ML Optimizer")
st.markdown("""
이 앱은 **실험 데이터(CSV)**를 기반으로 머신러닝 모델을 학습시켜, 
공정 변수와 소자 효율(PCE) 간의 상관관계를 분석하고 최적의 조건을 탐색합니다.
""")

# --- [2] 사이드바: 데이터 업로드 ---
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("ML용 CSV 파일 업로드", type=["csv", "xlsx"])

# --- [3] 메인 로직 ---
if uploaded_file:
    # 데이터 로드
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"데이터 로드 성공! 총 {len(df)}개의 샘플이 있습니다.")
        
        # 데이터 미리보기 (접기 가능)
        with st.expander("데이터 미리보기 (상위 5행)", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        # --- [4] 데이터 전처리 ---
        st.subheader("2. Data Preprocessing & Modeling")
        
        # 1. 타겟 변수 선택 (PCE)
        target_col = st.selectbox("타겟 변수 (예측 목표) 선택", df.columns, index=df.columns.get_loc("PCE (%)") if "PCE (%)" in df.columns else 0)
        
        # 2. 입력 변수(Feature) 선택
        # 기본적으로 숫자형 컬럼이나 특정 패턴이 있는 컬럼을 추천할 수 있지만, 여기선 전체 컬럼 중 선택하게 함
        # 불필요한 컬럼 (Sample, File 등 식별자) 제외
        exclude_cols = ['Sample', 'File', 'Scan Direction', target_col]
        feature_candidates = [c for c in df.columns if c not in exclude_cols]
        
        selected_features = st.multiselect("학습에 사용할 변수(Feature) 선택", feature_candidates, default=[c for c in feature_candidates if c.startswith('HTL') or c.startswith('Perovskite')][:5])
        
        if not selected_features:
            st.warning("최소 1개 이상의 변수를 선택해주세요.")
            st.stop()

        # 3. 모델 학습 버튼
        if st.button("🚀 Run Machine Learning (Random Forest)"):
            
            # --- 데이터 준비 ---
            X = df[selected_features].copy()
            y = df[target_col].copy()
            
            # 결측치 처리 (숫자형: 평균, 범주형: 최빈값)
            num_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(exclude=np.number).columns
            
            # 숫자형 Imputer
            if len(num_cols) > 0:
                imputer_num = SimpleImputer(strategy='mean')
                X[num_cols] = imputer_num.fit_transform(X[num_cols])
            
            # 범주형 인코딩 (Label Encoding)
            label_encoders = {}
            if len(cat_cols) > 0:
                for col in cat_cols:
                    le = LabelEncoder()
                    # 결측치는 'Missing'으로 채움
                    X[col] = X[col].fillna('Missing').astype(str)
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le
            
            # 타겟 결측치 제거
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 모델 학습
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            # --- 결과 분석 ---
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            st.markdown("---")
            st.subheader("3. Analysis Results")
            
            # 성능 지표
            col1, col2 = st.columns(2)
            col1.metric("Model R² Score", f"{r2:.3f}", help="1에 가까울수록 모델이 데이터를 잘 설명합니다.")
            col2.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
            
            # 1. Feature Importance Plot
            st.markdown("#### 🌟 Feature Importance (중요 변수 순위)")
            importances = rf.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances}).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h', title="Top Influential Factors on PCE")
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # 2. Actual vs Predicted Plot
            st.markdown("#### 🎯 Prediction Accuracy (실제값 vs 예측값)")
            fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual PCE', 'y': 'Predicted PCE'}, title="Actual vs Predicted")
            fig_pred.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # 3. Correlation Scatter Plot (Top Feature)
            top_feature = feature_imp_df.iloc[-1]['Feature']
            st.markdown(f"#### 🔍 Top Factor Analysis: {top_feature} vs {target_col}")
            
            # 원본 데이터(df)를 사용하여 시각화 (인코딩 전 값 사용)
            fig_scatter = px.scatter(df, x=top_feature, y=target_col, color=target_col, title=f"Correlation: {top_feature} vs {target_col}")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # --- [5] 최적화 시뮬레이터 (Optional) ---
            st.markdown("---")
            st.subheader("🧪 Virtual Experiment (Simulator)")
            st.info("아래 변수들을 조절하여 예상 PCE를 예측해보세요.")
            
            input_data = {}
            cols = st.columns(3)
            
            for i, col_name in enumerate(selected_features):
                col_obj = cols[i % 3]
                
                # 범주형인 경우
                if col_name in cat_cols:
                    # 원본 데이터의 unique 값들 가져오기
                    options = list(label_encoders[col_name].classes_)
                    val = col_obj.selectbox(f"{col_name}", options)
                    # 인코딩해서 저장
                    input_data[col_name] = label_encoders[col_name].transform([val])[0]
                
                # 숫자형인 경우
                else:
                    min_val = float(df[col_name].min())
                    max_val = float(df[col_name].max())
                    mean_val = float(df[col_name].mean())
                    val = col_obj.slider(f"{col_name}", min_val, max_val, mean_val)
                    input_data[col_name] = val
            
            if st.button("Predict PCE for these conditions"):
                input_df = pd.DataFrame([input_data])
                pred_pce = rf.predict(input_df)[0]
                st.success(f"🧪 예측된 PCE: **{pred_pce:.2f}%**")

else:
    st.info("👈 왼쪽 사이드바에서 데이터 파일을 업로드해주세요.")

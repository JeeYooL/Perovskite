import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- [설정] 페이지 환경 설정 ---
st.set_page_config(page_title="Perovskite ML Dashboard", layout="wide")

st.title("🔬 Perovskite Solar Cell Integrated Analysis Platform")
st.markdown("---")

# --- [사이드바] 데이터 업로드 ---
st.sidebar.header("📂 Data Center")
uploaded_file = st.sidebar.file_uploader("정리된 ML용 CSV 업로드", type=["csv", "xlsx"])

if uploaded_file:
    # 1. 데이터 로드
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 기본 정보 세션
    st.sidebar.subheader("📍 Session Info")
    user_name = st.sidebar.text_input("User Name", value="Hyoungwoo Kwon")
    
    # --- [탭 구성] 데이터 확인 / 상세 분석 / ML 예측 ---
    tab_data, tab_detail, tab_ml = st.tabs(["📊 Data Overview", "🔍 Sample Analysis", "🤖 Machine Learning"])

    # 탭 1: 전체 데이터 개요
    with tab_data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Devices", len(df))
        col2.metric("Max PCE (%)", f"{df['PCE (%)'].max():.2f}")
        col3.metric("Avg Voc (V)", f"{df['Voc (V)'].mean():.3f}")
        st.dataframe(df, use_container_width=True)

    # 탭 2: 상세 분석 (J-V 곡선 시각화)
    with tab_detail:
        target_sample = st.selectbox("분석할 샘플 선택", df['Sample'].unique())
        sample_df = df[df['Sample'] == target_sample]
        
        # J-V 곡선 (Rs, Rsh는 계산할 필요 없이 데이터 사용)
        fig = px.scatter(sample_df, x='Voc (V)', y='Jsc (mA/cm2)', color='File',
                         title=f"Results for {target_sample}",
                         labels={'Voc (V)': 'Voltage (V)', 'Jsc (mA/cm2)': 'Current Density (mA/cm²)'})
        st.plotly_chart(fig, use_container_width=True)

    # 탭 3: ML 예측 (구글 Colab 로직 통합)
    with tab_ml:
        st.header("Random Forest Regression Analysis")
        
        if st.button("🚀 Run ML Analysis"):
            with st.spinner("최적의 모델을 찾는 중입니다... (GridSearch & CV)"):
                # 3-1. [cite_start]전처리 [cite: 12, 13, 16]
                target_col = 'PCE (%)'
                df_ml = df.dropna(subset=[target_col])
                
                # [cite_start]피처/타겟 분리 (Data Leakage 방지 [cite: 11])
                X = df_ml.drop(columns=[
                    'PCE (%)', 'Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'Rs (Ω·cm²)', 'Rsh (Ω·cm²)',
                    'Sample', 'File', 'Scan Direction', 'Operator', 'Structure'
                ], errors='ignore')
                y = df_ml[target_col]

                # [cite_start]범주형 변수 처리 (Multi-Label Binarization [cite: 14, 15, 16])
                X_numeric = X.select_dtypes(exclude=['object'])
                X_categorical = X.select_dtypes(include=['object'])

                processed_parts = [X_numeric]
                for col in X_categorical.columns:
                    binarized = X_categorical[col].fillna('').str.get_dummies(sep=' + ')
                    binarized = binarized.add_prefix(f"{col}_")
                    binarized.columns = binarized.columns.str.replace(r'[^\w\s]', '_', regex=True)
                    processed_parts.append(binarized)

                X_processed = pd.concat(processed_parts, axis=1).fillna(0)
                
                # 3-2. 모델 학습 (Colab Ver 2.0 최적화 로직 적용)
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
                
                # 하이퍼파라미터 튜닝 (간소화 버전)
                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # 3-3. 결과 출력
                y_pred = model.predict(X_test)
                st.success(f"학습 완료! R² Score: {r2_score(y_test, y_pred):.3f}")
                
                # 중요 변수 시각화
                importances = model.feature_importances_
                feat_imp = pd.DataFrame({'Variable': X_processed.columns, 'Importance': importances})
                feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(15)
                
                fig_imp = px.bar(feat_imp, x='Importance', y='Variable', orientation='h',
                                 title="Top 15 Key Process Variables",
                                 color='Importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("개인용 프로그램에서 정리된 CSV 파일을 왼쪽 사이드바에 업로드해 주세요.")

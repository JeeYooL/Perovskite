import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from PIL import Image
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from skimage import feature, measurement, segmentation
from scipy import ndimage

# --- [설정] 페이지 환경 설정 ---
st.set_page_config(page_title="SolarCell Data Hub", layout="wide")

# --- [사이드바] 공통 제어 영역 ---
st.sidebar.header("📁 Global Data Center")
main_csv = st.sidebar.file_uploader("1. ML 데이터셋 업로드 (CSV/XLSX)", type=["csv", "xlsx"]) #

# --- [기능 함수: ML 분석] ---
def run_rf_analysis(df):
    """ 기반 분석"""
    target_col = 'PCE (%)'
    df_ml = df.dropna(subset=[target_col])
    
    # 결과값 제외 (Data Leakage 방지)
    X = df_ml.drop(columns=[
        'PCE (%)', 'Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'Rs (Ω·cm²)', 'Rsh (Ω·cm²)',
        'Sample', 'File', 'Scan Direction', 'Operator', 'Structure'
    ], errors='ignore')
    y = df_ml[target_col]

    # 범주형 변수 처리
    X_numeric = X.select_dtypes(exclude=['object'])
    X_categorical = X.select_dtypes(include=['object'])
    processed_parts = [X_numeric]
    for col in X_categorical.columns:
        binarized = X_categorical[col].fillna('').str.get_dummies(sep=' + ')
        binarized = binarized.add_prefix(f"{col}_")
        processed_parts.append(binarized)

    X_processed = pd.concat(processed_parts, axis=1).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X_processed

# --- [기능 함수: SEM 결정립 분석] ---
def analyze_grain_size(img_array, bar_nm, bar_pixel_width):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    nm_per_pixel = bar_nm / bar_pixel_width
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    labels = measurement.label(thresh)
    props = measurement.regionprops(labels)
    diameters = [p.equivalent_diameter * nm_per_pixel for p in props if p.area > 50]
    return diameters, thresh

# --- [메인 탭 구성] ---
# CSV 업로드 여부와 상관없이 탭이 항상 보이도록 배치
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 J-V Dashboard", 
    "📈 XRD & PL Analysis", 
    "🔬 SEM Grain Analysis", 
    "🤖 Machine Learning"
])

# 탭 1: J-V 데이터 대시보드
with tab1:
    if main_csv:
        df = pd.read_csv(main_csv) if main_csv.name.endswith('.csv') else pd.read_excel(main_csv)
        st.header("Master Database Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Devices", len(df)) #
        c2.metric("Best PCE (%)", df['PCE (%)'].max())
        c3.metric("Avg Voc (V)", f"{df['Voc (V)'].mean():.3f}")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("사이드바에서 CSV 파일을 업로드하면 데이터 요약이 표시됩니다.")

# 탭 2: XRD & PL (텍스트 파일 업로드)
with tab2:
    st.header("XRD & PL Spectrum Plotter")
    st.markdown("`.txt` 또는 `.csv` 형태의 원본 데이터를 업로드하세요.")
    
    char_files = st.file_uploader("XRD/PL 데이터 파일 업로드 (Multi-select 가능)", type=["txt", "csv"], accept_multiple_files=True)
    
    if char_files:
        for f in char_files:
            try:
                # 데이터 파싱 (1열: X, 2열: Intensity 가정)
                char_df = pd.read_csv(f, sep=r'\s+', header=None, names=['X', 'Intensity'])
                fig = px.line(char_df, x='X', y='Intensity', title=f"File: {f.name}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"{f.name} 처리 중 오류 발생: {e}")

# 탭 3: SEM 이미지 분석
with tab3:
    st.header("SEM Grain Size Analyzer")
    sem_file = st.file_uploader("SEM 이미지 파일 업로드 (JPG, PNG, TIF)", type=["jpg", "png", "tif"])
    
    if sem_file:
        c_img, c_res = st.columns(2)
        img = Image.open(sem_file)
        img_array = np.array(img)
        c_img.image(img, caption="Original SEM Image", use_container_width=True)
        
        # 분석 설정 (이미지 하단 배율바 기준)
        st.divider()
        col_set1, col_set2 = st.columns(2)
        bar_nm = col_set1.number_input("Scale Bar 실제 길이 (nm)", value=500, step=100)
        bar_px = col_set2.number_input("Scale Bar의 픽셀 길이 (측정값)", value=100, step=1)
        
        if st.button("🚀 자동 결정립 분석 실행"):
            diameters, processed_img = analyze_grain_size(img_array, bar_nm, bar_px)
            c_res.image(processed_img, caption="Detected Grain Boundaries", use_container_width=True)
            
            # 분석 결과 텍스트화
            st.subheader("📝 SEM 분석 리포트")
            res1, res2, res3 = st.columns(3)
            res1.write(f"**검출된 그레인 수:** {len(diameters)} 개")
            res2.write(f"**평균 크기:** {np.mean(diameters):.2f} nm")
            res3.write(f"**표준 편차:** {np.std(diameters):.2f} nm")
            
            # 분포도 그래프
            fig_hist = px.histogram(diameters, nbins=20, title="Grain Size Distribution",
                                    labels={'value': 'Size (nm)'}, color_discrete_sequence=['indianred'])
            st.plotly_chart(fig_hist, use_container_width=True)

# 탭 4: 머신러닝 분석
with tab4:
    st.header("AI-Driven Research Insight")
    if main_csv:
        if st.button("Run Random Forest Prediction"):
            model, X_test, y_test, X_processed = run_rf_analysis(df)
            y_pred = model.predict(X_test)
            
            st.success(f"모델 학습 완료! R²: {r2_score(y_test, y_pred):.3f}")
            
            # 중요 변수 시각화
            importances = pd.DataFrame({'Variable': X_processed.columns, 'Importance': model.feature_importances_})
            importances = importances.sort_values(by='Importance', ascending=False).head(15)
            fig_imp = px.bar(importances, x='Importance', y='Variable', orientation='h', 
                             title="Top 15 Critical Variables", color='Importance')
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("머신러닝 분석을 위해 사이드바에서 메인 CSV 파일을 먼저 업로드해 주세요.")

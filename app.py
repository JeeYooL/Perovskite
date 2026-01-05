import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
st.set_page_config(page_title="SolarCell Data Lab", layout="wide")

# --- [기능 함수: ML 분석] ---
def run_rf_analysis(df):
    """ 로직 기반 ML 학습"""
    target_col = 'PCE (%)'
    df_ml = df.dropna(subset=[target_col])
    
    # 데이터 누수 방지 처리
    X = df_ml.drop(columns=[
        'PCE (%)', 'Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'Rs (Ω·cm²)', 'Rsh (Ω·cm²)',
        'Sample', 'File', 'Scan Direction', 'Operator', 'Structure'
    ], errors='ignore')
    y = df_ml[target_col]

    # 범주형 변수 Multi-Label Binarization
    X_numeric = X.select_dtypes(exclude=['object'])
    X_categorical = X.select_dtypes(include=['object'])
    processed_parts = [X_numeric]
    
    for col in X_categorical.columns:
        binarized = X_categorical[col].fillna('').str.get_dummies(sep=' + ')
        binarized = binarized.add_prefix(f"{col}_")
        binarized.columns = binarized.columns.str.replace(r'[^\w\s]', '_', regex=True)
        processed_parts.append(binarized)

    X_processed = pd.concat(processed_parts, axis=1).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    importances = pd.DataFrame({'Variable': X_processed.columns, 'Importance': model.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False).head(15)
    
    return r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), importances

# --- [기능 함수: SEM 결정립 분석] ---
def analyze_grain_size(img_array, bar_nm, bar_pixel_width):
    """이미지 분석을 통한 결정립 크기 계산 및 텍스트화"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    nm_per_pixel = bar_nm / bar_pixel_width
    
    # 전처리 및 세그멘테이션
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 결정립 라벨링
    labels = measurement.label(thresh)
    props = measurement.regionprops(labels)
    
    # 직경 계산 (um 단위 변환 포함)
    diameters = [p.equivalent_diameter * nm_per_pixel for p in props if p.area > 50]
    
    report = {
        "count": len(diameters),
        "mean": np.mean(diameters),
        "std": np.std(diameters),
        "dist": diameters
    }
    return report, thresh

# --- [메인 UI] ---
st.title("☀️ Perovskite Solar Cell Research Data Hub")
st.markdown("---")

# 사이드바 데이터 업로드
st.sidebar.header("📁 Data Upload")
main_csv = st.sidebar.file_uploader("ML 데이터셋 (CSV)", type=["csv"])

if main_csv:
    df = pd.read_csv(main_csv)
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Characterization", "🔬 SEM Analysis", "🤖 AI Insight"])

    with tab1:
        st.header("Master Database Overview")
        # PCE, Voc, Jsc 통계 표시
        c1, c2, c3 = st.columns(3)
        c1.metric("Devices", len(df))
        c2.metric("Best PCE", f"{df['PCE (%)'].max()}%")
        c3.metric("Avg FF", f"{df['FF (%)'].mean():.1f}%")
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.header("XRD & PL Plotter")
        c_file = st.file_uploader("XRD/PL .txt 파일 업로드", type=["txt"], accept_multiple_files=True)
        if c_file:
            for f in c_file:
                txt_df = pd.read_csv(f, sep=r'\s+', header=None, names=['X', 'Intensity'])
                fig = px.line(txt_df, x='X', y='Intensity', title=f"Spectrum: {f.name}")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Automated SEM Grain Analysis")
        sem_file = st.file_uploader("SEM 이미지 업로드", type=["jpg", "png", "tif"])
        
        if sem_file:
            col_img, col_res = st.columns(2)
            img = Image.open(sem_file)
            img_array = np.array(img)
            col_img.image(img, caption="Original SEM", use_container_width=True)
            
            # 분석 설정
            bar_nm = st.number_input("Scale Bar 실제 길이 (nm)", value=500)
            bar_px = st.number_input("Scale Bar 픽셀 길이 (이미지에서 측정값)", value=100)
            
            if st.button("실행: 결정립 크기 분석"):
                report, processed = analyze_grain_size(img_array, bar_nm, bar_px)
                col_res.image(processed, caption="Detected Boundaries", use_container_width=True)
                
                st.subheader("📝 Analysis Report")
                st.write(f"- **검출된 결정립 수:** {report['count']} 개")
                st.write(f"- **평균 결정립 크기:** {report['mean']:.2f} nm")
                st.write(f"- **표준 편차:** {report['std']:.2f} nm")
                
                # 분포도 그래프
                fig_dist = px.histogram(report['dist'], nbins=30, title="Grain Size Distribution",
                                        labels={'value': 'Size (nm)'})
                st.plotly_chart(fig_dist)

    with tab4:
        st.header("Machine Learning Insight")
        if st.button("🚀 ML 분석 실행 (Random Forest)"):
            r2, mae, imp = run_rf_analysis(df)
            st.success(f"분석 완료! R² Score: {r2:.3f} | MAE: {mae:.3f}")
            
            fig_imp = px.bar(imp, x='Importance', y='Variable', orientation='h',
                             title="Top 15 Critical Variables",
                             color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.warning("먼저 CSV 데이터를 업로드해 주세요.")

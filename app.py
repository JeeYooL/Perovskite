import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# --- [설정] 페이지 환경 설정 ---
st.set_page_config(page_title="Perovskite Data Hub", layout="wide")

st.title("🔬 Perovskite Solar Cell Research Dashboard")
st.markdown("---")

# --- [사이드바] 데이터 업로드 및 세션 정보 ---
st.sidebar.header("📂 Data Control")
uploaded_file = st.sidebar.file_uploader("ML용 CSV 또는 Excel 업로드", type=["csv", "xlsx"])

if uploaded_file:
    # 데이터 로드
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 사이드바 세션 정보 (박사님 요청 사항 반영)
    st.sidebar.subheader("📍 Session Information")
    user_name = st.sidebar.text_input("User Name", value="Hyoungwoo Kwon")
    structure = st.sidebar.selectbox("Structure", ["p-i-n", "n-i-p", "Unknown"])
    
    # --- [메인 화면] 1. 데이터 개요 ---
    st.header("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Devices", len(df))
    col2.metric("Max PCE (%)", f"{df['PCE (%)'].max():.2f}")
    col3.metric("Avg Voc (V)", f"{df['Voc (V)'].mean():.3f}")

    # 데이터 테이블 출력 (검색 및 필터링 가능)
    with st.expander("원본 데이터 보기"):
        st.dataframe(df, use_container_width=True)

    # --- [메인 화면] 2. 상세 분석 (Sample별 데이터 연동) ---
    st.markdown("---")
    st.header("🔍 Sample Detail Analysis")
    
    target_sample = st.selectbox("분석할 샘플을 선택하세요", df['Sample'].unique())
    sample_df = df[df['Sample'] == target_sample]

    # 레이아웃 분할: 왼쪽(J-V) / 오른쪽(XRD, SEM, PL)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📈 J-V Curves")
        # Plotly를 이용한 인터랙티브 그래프
        fig = px.line(sample_df, x='Voc (V)', y='Jsc (mA/cm2)', color='File',
                      title=f"J-V Curves for {target_sample}")
        fig.update_layout(xaxis_title="Voltage (V)", yaxis_title="Current Density (mA/cm²)")
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("🖼️ Characterization Data")
        tab1, tab2, tab3 = st.tabs(["XRD", "SEM", "PL"])
        
        with tab1:
            st.info("XRD Raw 데이터 파일을 업로드하면 그래프가 여기에 표시됩니다.")
            # 예시: st.line_chart(xrd_df)
            
        with tab2:
            st.info("해당 샘플 ID와 매칭되는 SEM 이미지를 표시합니다.")
            # 예시: image = Image.open(f'path/{target_sample}.jpg')
            # st.image(image, caption=f"SEM Image of {target_sample}")
            
        with tab3:
            st.info("PL/TRPL 스펙트럼 분석 영역입니다.")

    # --- [메인 화면] 3. ML 예측 모델 연동 ---
    st.markdown("---")
    st.header("🤖 Machine Learning Insights")
    if st.button("Run PCE Prediction Model"):
        st.success("모델 분석 중... (Random Forest 최적화 적용)")
        # 기존 Colab 코드를 함수로 묶어 여기서 호출
        # st.write(feat_imp_df) # 중요 변수 그래프 출력
else:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하여 시작하세요.")
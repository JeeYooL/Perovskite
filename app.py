import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from PIL import Image
import io
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- [설정] 페이지 환경 설정 ---
st.set_page_config(page_title="SolarCell Data Hub", layout="wide")

# --- [사이드바: 모든 업로드 버튼을 이곳에 집중] ---
st.sidebar.header("📁 Data Center (Drag & Drop)")

# 1. 메인 실험 데이터 업로드
main_csv = st.sidebar.file_uploader("1. ML 데이터셋 (CSV/XLSX)", type=["csv", "xlsx"])

# 2. XRD/PL 데이터 업로드 (텍스트 파일)
spectra_files = st.sidebar.file_uploader("2. XRD/PL 데이터 (.txt)", type=["txt"], accept_multiple_files=True)

# 3. SEM 이미지 업로드
sem_files = st.sidebar.file_uploader("3. SEM 이미지 (.jpg/png)", type=["jpg", "png"], accept_multiple_files=True)

# --- [데이터 처리 로직] ---
# 세션 상태를 사용하여 업로드된 캐릭터리제이션 데이터를 관리합니다.
if 'spectra_data' not in st.session_state:
    st.session_state.spectra_data = {}

# --- [메인 화면 구성] ---
st.title("☀️ Perovskite Research: Integrated Data Hub")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 J-V & ML Analysis", "📈 Spectra Linking", "🔬 SEM Analysis"])

# --- Tab 1: J-V & ML (CSV 기반) ---
with tab1:
    if main_csv:
        df = pd.read_csv(main_csv) if main_csv.name.endswith('.csv') else pd.read_excel(main_csv)
        st.header("Experiment Database Overview")
        st.dataframe(df.head(), use_container_width=True)
        
        # ML 실행 버튼
        if st.button("🚀 Run Machine Learning"):
            st.info("데이터 분석 중... (Random Forest 적용)")
            # (이전의 ML 학습 로직 수행)
    else:
        st.warning("먼저 사이드바에서 실험 결과 CSV 파일을 업로드해 주세요.")

# --- Tab 2: Spectra Linking (PL/XRD 데이터를 실험 결과와 연결) ---
with tab2:
    st.header("Spectra to Sample Linking")
    
    
    if spectra_files and main_csv:
        df = pd.read_csv(main_csv) if main_csv.name.endswith('.csv') else pd.read_excel(main_csv)
        sample_list = df['Sample'].unique().tolist()
        
        for f in spectra_files:
            st.markdown(f"#### 📄 File: {f.name}")
            
            # 파일명에서 샘플명 자동 추출 시도 (예: 'PL_Sample1.txt'에서 'Sample1' 탐색)
            suggested_sample = next((s for s in sample_list if s in f.name), sample_list[0])
            
            # 드롭다운으로 연결할 샘플 확인/수정
            linked_sample = st.selectbox(f"이 데이터({f.name})와 연결할 샘플 ID 선택", 
                                         sample_list, 
                                         index=sample_list.index(suggested_sample),
                                         key=f"link_{f.name}")
            
            # 연결된 샘플의 정보 요약 표시
            sample_info = df[df['Sample'] == linked_sample].iloc[0]
            st.caption(f"✅ 연결됨: {linked_sample} (PCE: {sample_info['PCE (%)']}%, Voc: {sample_info['Voc (V)']}V)")
            
            # 그래프 출력
            try:
                txt_df = pd.read_csv(f, sep=r'\s+', header=None, names=['X', 'Intensity'])
                fig = px.line(txt_df, x='X', y='Intensity', title=f"{f.name} ({linked_sample})")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.error("데이터 파싱에 실패했습니다. 파일 형식을 확인하세요.")
    else:
        st.info("사이드바에서 CSV와 .txt 파일을 모두 업로드하면 연결 분석이 시작됩니다.")

# --- Tab 3: SEM 분석 ---
with tab3:
    st.header("SEM Image Grain Analysis")
    if sem_files:
        for sem in sem_files:
            st.subheader(f"Image: {sem.name}")
            img = Image.open(sem)
            st.image(img, use_container_width=True)
            # (이전의 Grain Size 분석 로직 추가 가능)
    else:
        st.info("사이드바에서 SEM 이미지를 업로드하세요.")

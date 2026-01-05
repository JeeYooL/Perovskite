import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from scipy.signal import savgol_filter

# --- [1] 핵심 분석 로직 (analyze.py에서 이식) ---

def parse_value(value_str):
    if value_str is None: return None
    value_str = str(value_str).strip()
    try:
        if value_str.endswith('m'): return float(value_str[:-1]) / 1000.0
        elif value_str.endswith('u'): return float(value_str[:-1]) / 1000000.0
        elif value_str.endswith('k'): return float(value_str[:-1]) * 1000.0
        else: return float(value_str)
    except (ValueError, TypeError):
        return np.nan

def calculate_resistances(df_jv):
    """J-V 데이터에서 Rs, Rsh를 계산하는 함수"""
    Rs, Rsh = None, None
    try:
        # 컬럼 이름이 다를 수 있으므로 표준화 시도
        v_col = next((c for c in df_jv.columns if c.startswith('V') or 'Voltage' in c), None)
        j_col = next((c for c in df_jv.columns if c.startswith('J') or 'Current' in c), None)
        
        if not v_col or not j_col: return None, None

        V_raw = df_jv[v_col].values
        J_raw = df_jv[j_col].values

        if len(V_raw) < 10: return None, None

        # Smoothing
        window_length = 5
        polyorder = 3
        if len(V_raw) < window_length: window_length = len(V_raw) - 1
        J_smooth = savgol_filter(J_raw, window_length, polyorder)

        # Rsh calculation (near V=0)
        mask_rsh = np.abs(V_raw) < 0.1
        if np.sum(mask_rsh) > 2:
            coeffs_rsh = np.polyfit(V_raw[mask_rsh], J_smooth[mask_rsh], 1)
            if coeffs_rsh[0] != 0: Rsh = np.abs(1.0 / coeffs_rsh[0])

        # Rs calculation (near Voc)
        voc_index = np.argmin(np.abs(J_smooth))
        voc_value = V_raw[voc_index]
        mask_rs = (V_raw > voc_value - 0.05) & (V_raw < voc_value + 0.05)
        if np.sum(mask_rs) > 2:
            coeffs_rs = np.polyfit(V_raw[mask_rs], J_smooth[mask_rs], 1)
            if coeffs_rs[0] != 0: Rs = np.abs(1.0 / coeffs_rs[0])

        return Rs, Rsh
    except:
        return None, None

def detect_scan_direction(filename, df_jv):
    """파일명 또는 데이터로 스캔 방향 감지"""
    filename = filename.lower()
    if 'rev' in filename or 'reverse' in filename: return 'Reverse'
    elif 'fwd' in filename or 'forward' in filename: return 'Forward'
    
    try:
        v_col = next((c for c in df_jv.columns if c.startswith('V') or 'Voltage' in c), None)
        if v_col:
            V = df_jv[v_col].values
            if len(V) > 1:
                if V[0] < V[-1]: return 'Forward'
                elif V[0] > V[-1]: return 'Reverse'
    except: pass
    return 'Unknown'

# --- [2] Streamlit 페이지 설정 ---
st.set_page_config(page_title="PCE Analyzer Web", layout="wide", page_icon="☀️")

st.title("☀️ Perovskite J-V Analyzer (Web Version)")
st.markdown("""
기존 로컬 분석 프로그램의 기능을 웹으로 옮겼습니다. 
**.txt 파일을 드래그 앤 드롭**하여 J-V 곡선을 분석하고 시각화하세요.
""")

# --- [3] 사이드바: 데이터 업로드 및 필터 ---
st.sidebar.header("1. Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "J-V txt 파일들을 업로드하세요 (다중 선택 가능)", 
    type=["txt", "csv"], 
    accept_multiple_files=True
)

st.sidebar.header("2. Settings")
process_vars = ["TCO", "HTL", "Perovskite", "ETL", "Contact"]
selected_vars = st.sidebar.multiselect("활성화할 변수 컬럼", process_vars, default=["HTL", "Perovskite"])

# --- [4] 메인 로직 ---

if uploaded_files:
    # 데이터 처리 (캐싱을 위해 함수로 분리 가능하나 간단하게 구현)
    all_data = []
    
    # 진행률 표시
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # 파일 읽기 (Bytes -> String)
            # 인코딩 문제 해결을 위해 여러 시도
            content = None
            for enc in ['cp949', 'utf-8', 'latin-1']:
                try:
                    content = uploaded_file.getvalue().decode(enc)
                    break
                except: continue
            
            if not content: continue

            lines = content.splitlines()
            
            # 헤더 파싱 (기존 로직 활용)
            header_line = lines[0].strip()
            data_lines = []
            parameters = {}
            
            # 데이터와 파라미터 분리 로직
            line_iter = iter(lines[1:])
            for line in line_iter:
                line = line.strip()
                if not line: continue
                if line == 'end': break
                data_lines.append(line)
            
            # 파라미터 추출
            for line in line_iter:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2: parameters[parts[0].strip()] = parts[1].strip()

            # DataFrame 생성
            clean_header = '\t'.join(header_line.split('\t')[:3])
            full_data_string = clean_header + '\n' + '\n'.join(data_lines)
            jv_data = pd.read_csv(io.StringIO(full_data_string), sep='\t')
            
            # Jsc 단위 변환 확인 (A -> mA)
            if 'Jsc(A/cm2)' in parameters:
                jsc_val = parse_value(parameters['Jsc(A/cm2)'])
                jsc_ma = jsc_val * 1000 if jsc_val is not None else None
            else:
                jsc_ma = None

            # Rs, Rsh 계산
            Rs, Rsh = calculate_resistances(jv_data)
            scan_dir = detect_scan_direction(uploaded_file.name, jv_data)

            # 결과 저장
            all_data.append({
                'Filename': uploaded_file.name,
                'Scan': scan_dir,
                'Voc (V)': parse_value(parameters.get('Voc (V)')),
                'Jsc (mA/cm2)': jsc_ma,
                'FF (%)': parse_value(parameters.get('Fill factor (%)')),
                'PCE (%)': parse_value(parameters.get('Efficiency (%)')),
                'Rs': round(Rs, 2) if Rs else None,
                'Rsh': round(Rsh, 1) if Rsh else None,
                '_raw_df': jv_data  # 그래프 그리기 위해 원본 데이터 저장 (숨김 컬럼)
            })
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()

    if all_data:
        df = pd.DataFrame(all_data)
        
        # --- 탭 구성 ---
        tab1, tab2, tab3 = st.tabs(["📊 Data Table & Filter", "📈 J-V Curves", "📦 Statistics (Box Plot)"])

        # --- Tab 1: 데이터 테이블 및 변수 입력 ---
        with tab1:
            st.subheader("Processed Data Table")
            
            # 변수 입력을 위한 빈 컬럼 추가
            for var in selected_vars:
                if var not in df.columns:
                    df[var] = ""  # 초기값 빈 문자열

            # 보여줄 컬럼 선택
            display_cols = ['Filename', 'Scan', 'Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'PCE (%)', 'Rs', 'Rsh'] + selected_vars
            
            # [기능] 데이터 에디터 (엑셀처럼 수정 가능)
            edited_df = st.data_editor(
                df[display_cols],
                column_config={
                    "Filename": st.column_config.TextColumn("Filename", disabled=True),
                },
                use_container_width=True,
                height=400,
                key="data_editor"
            )
            
            # 필터링 기능 (Best Device 추출 등)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Processed Data (CSV)",
                    edited_df.to_csv(index=False).encode('utf-8-sig'),
                    "processed_jv_data.csv",
                    "text/csv"
                )

        # --- Tab 2: J-V 곡선 (Plotly 사용) ---
        with tab2:
            st.subheader("Interactive J-V Curves")
            
            # 그래프 옵션
            c1, c2 = st.columns([1, 3])
            with c1:
                scan_filter = st.radio("Scan Direction Filter", ["All", "Reverse", "Forward"])
                show_best_only = st.checkbox("Show Best PCE Only")
            
            # 그래프 그리기 로직
            fig = go.Figure()
            
            # 필터링
            filtered_df = df.copy()
            if scan_filter != "All":
                filtered_df = filtered_df[filtered_df['Scan'] == scan_filter]
            
            if show_best_only:
                # 파일명에서 샘플 그룹을 추출하는 로직이 필요하지만, 여기선 단순히 전체 중 최고 효율 1개만 예시로
                best_idx = filtered_df['PCE (%)'].idxmax()
                plot_target = filtered_df.loc[[best_idx]]
            else:
                plot_target = filtered_df

            # 선택된 데이터 루프
            # 너무 많으면 느려지므로 제한
            if len(plot_target) > 50 and not show_best_only:
                st.warning(f"데이터가 너무 많습니다 ({len(plot_target)}개). 상위 50개만 표시합니다.")
                plot_target = plot_target.head(50)

            for idx, row in plot_target.iterrows():
                raw_df = row['_raw_df'] # 저장해둔 원본 데이터
                # 컬럼명 찾기
                v_col = next((c for c in raw_df.columns if c.startswith('V')), None)
                j_col = next((c for c in raw_df.columns if c.startswith('J')), None)
                
                if v_col and j_col:
                    fig.add_trace(go.Scatter(
                        x=raw_df[v_col],
                        y=raw_df[j_col] * 1000, # A -> mA 변환
                        mode='lines+markers',
                        name=f"{row['Filename']} ({row['PCE (%)']}%)"
                    ))

            fig.update_layout(
                xaxis_title="Voltage (V)",
                yaxis_title="Current Density (mA/cm²)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.add_hline(y=0, line_width=1, line_color="gray")
            fig.add_vline(x=0, line_width=1, line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 3: 통계 (Box Plot) ---
        with tab3:
            st.subheader("Statistical Distribution")
            
            # 그룹화 기준 선택 (파일명 또는 입력한 변수)
            group_by = st.selectbox("Group By", ["Scan"] + selected_vars)
            
            # 파라미터 선택
            stat_param = st.selectbox("Parameter", ["PCE (%)", "Voc (V)", "Jsc (mA/cm2)", "FF (%)", "Rs", "Rsh"])
            
            if group_by:
                # 사용자가 편집한 데이터프레임(edited_df) 사용 (변수 입력 반영)
                # 원본 df와 edited_df를 매칭해야 함. (여기선 간략히 edited_df만 사용)
                # 데이터 에디터는 원본 데이터프레임의 인덱스를 보존하므로 매핑 가능
                
                # Plotly Box Plot
                fig_box = px.box(edited_df, x=group_by, y=stat_param, points="all", color=group_by)
                st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.warning("유효한 데이터가 없습니다.")

else:
    st.info("왼쪽 사이드바에서 파일을 업로드해주세요.")

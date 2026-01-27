import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# [설정] Matplotlib 백엔드 (스레드 충돌 방지)
# -------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg') # 서버 전용(GUI 없음) 모드로 설정
import matplotlib.pyplot as plt

import seaborn as sns
import io
import re

# 이미지 처리 라이브러리 (SEM 분석용)
import cv2

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
    page_title="Perovskite AI Lab V7.1 (Bandgap)",
    page_icon="⚗️",
    layout="wide"
)

# CSS 스타일 커스텀
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1, h2, h3 { color: #003366; font-family: 'Arial', sans-serif; }
    
    /* 파일 업로더 컴팩트하게 만들기 */
    [data-testid='stFileUploader'] {
        padding-top: 0px;
        padding-bottom: 0px;
        margin-bottom: 0px;
    }
    [data-testid='stFileUploader'] section {
        padding: 0px;
        min-height: 40px; /* 높이 최소화 */
        background-color: #f8f9fa;
        border: 1px dashed #ced4da;
    }
    /* 업로드된 파일 이름 폰트 줄이기 */
    [data-testid='stFileUploader'] section > div {
        padding: 2px;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem;
    }
    
    /* 테이블 헤더 스타일 */
    .upload-header {
        font-weight: bold;
        text-align: center;
        background-color: #e9ecef;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-size: 0.9rem;
    }
    
    /* 샘플 ID 셀 스타일 */
    .sample-id-cell {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 42px; /* 업로더 높이와 맞춤 */
        font-weight: bold;
        color: #2c3e50;
        background-color: #f1f3f5;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    .bottom-spacer { height: 100px; }
    </style>
""", unsafe_allow_html=True)

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# -------------------------------------------------------------------
# 함수 정의
# -------------------------------------------------------------------

def load_data(uploaded_files):
    """메인 데이터 파일 로드"""
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

def extract_features_from_spectra(file, data_type):
    """
    XRD, PL, TRPL 등 스펙트럼 데이터(X, Y)에서 핵심 Feature 추출
    + [신규] PL 데이터인 경우 Bandgap 자동 계산 추가
    """
    try:
        # 파일 내용 읽기 (파싱 로직)
        file.seek(0)
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            content = file.read().decode('cp949', errors='ignore')
            
        lines = content.splitlines()
        
        # 데이터 시작 라인 찾기
        data_start_idx = 0
        is_data_found = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            parts = re.split(r'[,\t\s]+', line)
            parts = [p for p in parts if p] 
            
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    data_start_idx = i
                    is_data_found = True
                    break
                except ValueError:
                    continue
        
        if not is_data_found:
            return None
            
        from io import StringIO
        data_str = "\n".join(lines[data_start_idx:])
        df = pd.read_csv(StringIO(data_str), sep=None, engine='python', header=None)

        if df.shape[1] < 2:
            return None

        x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
        
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values

        features = {}
        
        # 1. Max Intensity & Peak Position
        max_idx = np.argmax(y)
        max_y = y[max_idx]
        max_x = x[max_idx] # Peak Position (nm or degree)

        features[f"{data_type}_Peak_Pos"] = max_x
        features[f"{data_type}_Max_Int"] = max_y

        # [신규 기능] PL 데이터일 경우 Bandgap(eV) 계산
        # 공식: Energy (eV) = 1240 / Wavelength (nm)
        if data_type == "PL" and max_x > 0:
            features[f"{data_type}_Bandgap_eV"] = 1240.0 / max_x

        # 2. FWHM (반치폭)
        half_max = max_y / 2.0
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # 피크가 여러 개일 수 있으므로 단순화된 로직 사용 (최대 피크 기준)
        # 1. 최대값보다 왼쪽/오른쪽 데이터 분리
        #    (x_sorted에서 max_x의 위치를 찾음)
        #    Note: searchsorted는 정렬된 배열에서만 작동
        try:
            # 실제 max_x와 정확히 일치하는 인덱스가 없을 수 있으므로 가장 가까운 값 찾기
            max_pos_idx = np.argmin(np.abs(x_sorted - max_x))
            
            left_x = x_sorted[:max_pos_idx]
            left_y = y_sorted[:max_pos_idx]
            right_x = x_sorted[max_pos_idx:]
            right_y = y_sorted[max_pos_idx:]

            fwhm = 0
            if len(left_y) > 0 and len(right_y) > 0:
                idx_l = np.argmin(np.abs(left_y - half_max))
                idx_r = np.argmin(np.abs(right_y - half_max))
                fwhm = right_x[idx_r] - left_x[idx_l]
            features[f"{data_type}_FWHM"] = fwhm
        except:
            features[f"{data_type}_FWHM"] = 0

        # 3. Area
        area = np.trapz(y, x)
        features[f"{data_type}_Area"] = area

        return features

    except Exception as e:
        return None

def extract_features_from_sem(file):
    """
    SEM 이미지에서 Grain Size 분석 (OpenCV 활용)
    """
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None

        # 전처리
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        blurred = cv2.GaussianBlur(cl1, (5, 5), 0)

        # 이진화 및 윤곽선
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        grain_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10: 
                grain_areas.append(area)
        
        features = {}
        if len(grain_areas) > 0:
            avg_area = np.mean(grain_areas)
            avg_diameter = np.sqrt(4 * avg_area / np.pi)
            
            features["SEM_Grain_Count"] = len(grain_areas)
            features["SEM_Avg_Size_px"] = avg_diameter
        else:
            features["SEM_Grain_Count"] = 0
            features["SEM_Avg_Size_px"] = 0
            
        return features

    except Exception as e:
        return None

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df

def detect_target_column(df):
    candidates = [c for c in df.columns if 'PCE' in c.upper()]
    if candidates:
        return candidates[0]
    return df.columns[-1] if not df.empty else None

def preprocess_data(df, target_column):
    df_cleaned = df.dropna(subset=[target_column]).copy()
    if len(df_cleaned) == 0: return None, None, None, None

    drop_keywords = ['PCE', 'Voc', 'Jsc', 'FF', 'Rs', 'Rsh', 'Scan', 'Sample', 'File', 'Unnamed']
    cols_to_drop = [target_column]
    for col in df_cleaned.columns:
        if col == target_column: continue
        for kw in drop_keywords:
            if kw in col:
                cols_to_drop.append(col)
                break
    
    X_raw = df_cleaned.drop(columns=cols_to_drop, errors='ignore')
    y = df_cleaned[target_column]
    
    X_numeric = X_raw.select_dtypes(exclude=['object'])
    X_categorical = X_raw.select_dtypes(include=['object'])
    
    all_processed = [X_numeric]
    for col in X_categorical.columns:
        binarized = X_categorical[col].fillna('').astype(str).str.get_dummies(sep=' + ')
        binarized = binarized.add_prefix(f"{col}_")
        all_processed.append(binarized)
        
    X_processed = pd.concat(all_processed, axis=1).fillna(0)
    X_processed.columns = X_processed.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(r'\s+', '_', regex=True)
    
    # 중복 컬럼 처리
    if X_processed.columns.duplicated().any():
        new_columns = []
        seen = {}
        for col in X_processed.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        X_processed.columns = new_columns
    
    try:
        X_processed = X_processed.astype(float)
    except ValueError:
        for col in X_processed.columns:
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)

    return X_processed, y, df_cleaned, X_raw

# -------------------------------------------------------------------
# 메인 UI
# -------------------------------------------------------------------

st.title("⚗️ Perovskite AI Lab V7.1 (Physics-Informed)")
st.write("공정-물성 통합 분석 (PL 밴드갭 자동 계산 포함)")
st.markdown("---")

# 사이드바: 메인 데이터 업로드 (항상 표시)
with st.sidebar:
    st.header("📂 1. Main Recipe Data")
    uploaded_files = st.file_uploader("메인 CSV/Excel (Sample ID 필수)", type=['csv', 'xlsx'], accept_multiple_files=True, key="main")
    
    st.markdown("---")
    if st.button("🔄 결과 초기화"):
        st.session_state.analysis_results = None
        st.rerun()

# 메인 로직
if uploaded_files:
    raw_df = load_data(uploaded_files)
    
    if raw_df is not None:
        raw_df = clean_column_names(raw_df)
        
        # --------------------------------------------------------------------
        # 화면 분할 레이아웃 (50:50)
        # --------------------------------------------------------------------
        col_left, col_right = st.columns([1, 1], gap="medium")

        # ====================================================================
        # [왼쪽] 추가 데이터 업로드 (테이블 형식)
        # ====================================================================
        with col_left:
            st.subheader("🔬 2. Characterization Data Upload")
            st.info("샘플별 XRD, PL, TRPL, SEM 데이터를 업로드하세요. (PL 업로드 시 밴드갭 자동 계산)")
            
            if 'Sample' in raw_df.columns:
                try:
                    sample_ids = sorted(raw_df['Sample'].unique(), key=lambda x: float(x) if str(x).replace('.','',1).isdigit() else str(x))
                except:
                    sample_ids = sorted(raw_df['Sample'].astype(str).unique())
                
                # 검색창
                search_term = st.text_input("🔍 Sample ID 검색", placeholder="샘플 번호 입력...")
                filtered_samples = sample_ids
                if search_term:
                    filtered_samples = [s for s in sample_ids if search_term.lower() in str(s).lower()]

                st.markdown("<br>", unsafe_allow_html=True)

                # 1. 테이블 헤더 (5열)
                h_c1, h_c2, h_c3, h_c4, h_c5 = st.columns([1, 2, 2, 2, 2])
                h_c1.markdown("<div class='upload-header'>ID</div>", unsafe_allow_html=True)
                h_c2.markdown("<div class='upload-header'>XRD</div>", unsafe_allow_html=True)
                h_c3.markdown("<div class='upload-header'>PL</div>", unsafe_allow_html=True)
                h_c4.markdown("<div class='upload-header'>TRPL</div>", unsafe_allow_html=True)
                h_c5.markdown("<div class='upload-header'>SEM</div>", unsafe_allow_html=True)
                
                # 2. 테이블 행 반복 생성
                additional_features_list = []
                
                for s_id in filtered_samples:
                    row_c1, row_c2, row_c3, row_c4, row_c5 = st.columns([1, 2, 2, 2, 2])
                    
                    with row_c1:
                        st.markdown(f"<div class='sample-id-cell'>{s_id}</div>", unsafe_allow_html=True)
                    
                    f_xrd = row_c2.file_uploader("XRD", key=f"xrd_{s_id}", type=['csv', 'txt', 'dat'], label_visibility="collapsed")
                    f_pl = row_c3.file_uploader("PL", key=f"pl_{s_id}", type=['csv', 'txt', 'dat'], label_visibility="collapsed")
                    f_trpl = row_c4.file_uploader("TRPL", key=f"trpl_{s_id}", type=['csv', 'txt', 'dat'], label_visibility="collapsed")
                    f_sem = row_c5.file_uploader("SEM", key=f"sem_{s_id}", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], label_visibility="collapsed")
                    
                    current_feats = {'Sample': s_id}
                    
                    if f_xrd:
                        feats = extract_features_from_spectra(f_xrd, "XRD")
                        if feats: current_feats.update(feats)
                    if f_pl:
                        feats = extract_features_from_spectra(f_pl, "PL")
                        if feats: current_feats.update(feats)
                    if f_trpl:
                        feats = extract_features_from_spectra(f_trpl, "TRPL")
                        if feats: current_feats.update(feats)
                    if f_sem:
                        feats = extract_features_from_sem(f_sem)
                        if feats: current_feats.update(feats)
                    
                    if len(current_feats) > 1:
                        additional_features_list.append(current_feats)
                
                # 병합 로직
                if additional_features_list:
                    add_df = pd.DataFrame(additional_features_list)
                    try:
                        raw_df['Sample'] = raw_df['Sample'].astype(int)
                        add_df['Sample'] = add_df['Sample'].astype(int)
                    except:
                        raw_df['Sample'] = raw_df['Sample'].astype(str)
                        add_df['Sample'] = add_df['Sample'].astype(str)
                    
                    raw_df = pd.merge(raw_df, add_df, on='Sample', how='left')
                    st.success(f"✅ 총 {len(additional_features_list)}개 샘플의 외부 데이터 병합 완료")

            else:
                st.error("메인 데이터에 'Sample' 컬럼이 없어 테이블을 생성할 수 없습니다.")

        # ====================================================================
        # [오른쪽] 분석 설정 및 결과
        # ====================================================================
        with col_right:
            st.subheader("⚙️ Analysis & Results")
            st.write(f"✅ 총 **{len(raw_df)}**개 샘플 데이터 준비됨")
            
            # 외부 변수 확인
            ext_cols = [c for c in raw_df.columns if c.startswith(('XRD_', 'PL_', 'TRPL_', 'SEM_'))]
            if ext_cols:
                st.caption(f"✨ 추출된 변수: {', '.join(ext_cols)}")
            
            # 밴드갭 계산 확인 메시지
            if 'PL_Bandgap_eV' in raw_df.columns:
                st.info("💡 **Physics-Informed:** PL 데이터를 기반으로 **Bandgap (eV)**이 자동 계산되었습니다!")

            with st.expander("통합 데이터 미리보기", expanded=False):
                st.dataframe(raw_df.head())
            
            st.markdown("---")

            # 분석 설정 UI
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                target_col = st.selectbox("타겟 변수", options=raw_df.columns, index=list(raw_df.columns).index(detect_target_column(raw_df)) if detect_target_column(raw_df) else 0)
            with lc2:
                model_choice = st.selectbox("ML 모델", ["XGBoost (Recommended)", "Random Forest", "Gaussian Process"])
            with lc3:
                test_ratio = st.slider("테스트 비율", 0.1, 0.5, 0.2)

            # 분석 실행 버튼
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
                with st.spinner(f"{model_choice} 최적화 모델 구동 중..."):
                    try:
                        X, y, df_clean, X_raw_origin = preprocess_data(raw_df, target_col)
                        
                        if X is None:
                            st.error("데이터 전처리 실패")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
                            
                            model = None
                            is_tree_model = False
                            
                            if "XGBoost" in model_choice:
                                is_tree_model = True
                                xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
                                search = GridSearchCV(xgb_reg, {'n_estimators':[100,200], 'max_depth':[3,5], 'learning_rate':[0.05,0.1]}, cv=3, scoring='neg_mean_absolute_error', error_score='raise')
                                search.fit(X_train, y_train)
                                model = search.best_estimator_
                            elif "Random Forest" in model_choice:
                                is_tree_model = True
                                rf_reg = RandomForestRegressor(random_state=42, n_jobs=-1)
                                search = GridSearchCV(rf_reg, {'n_estimators':[100,200], 'max_depth':[10,None]}, cv=3, scoring='neg_mean_absolute_error')
                                search.fit(X_train, y_train)
                                model = search.best_estimator_
                            elif "Gaussian Process" in model_choice:
                                scaler_X = StandardScaler()
                                X_train_scaled = scaler_X.fit_transform(X_train)
                                X_test_scaled = scaler_X.transform(X_test)
                                kernel = 1.0 * RBF(1.0) + WhiteKernel(1.0)
                                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
                                gp.fit(X_train_scaled, y_train)
                                model = gp
                                model.custom_predict_std = lambda X_in: gp.predict(scaler_X.transform(X_in), return_std=True)

                            if "Gaussian Process" in model_choice:
                                y_pred, y_std = model.custom_predict_std(X_test)
                            else:
                                y_pred = model.predict(X_test)
                                y_std = None
                                
                            st.session_state.analysis_results = {
                                "model": model, "r2": r2_score(y_test, y_pred), "mae": mean_absolute_error(y_test, y_pred),
                                "y_test": y_test, "y_pred": y_pred, "y_std": y_std, "X_test": X_test, "X_train": X_train,
                                "X": X, "y": y, "target_col": target_col, "df_clean": df_clean,
                                "X_raw_origin": X_raw_origin, "model_choice": model_choice, "is_tree_model": is_tree_model
                            }
                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {e}")

            # 결과 리포트
            if st.session_state.analysis_results:
                res = st.session_state.analysis_results
                st.markdown("---")
                
                t1, t2, t3 = st.tabs(["📊 성능 평가", "🔍 중요도 분석", "💡 최적화 제안"])
                
                with t1:
                    col1, col2 = st.columns(2)
                    col1.metric("결정계수 (R²)", f"{res['r2']:.4f}")
                    col2.metric("오차 (MAE)", f"{res['mae']:.4f}")
                    fig, ax = plt.subplots(figsize=(6,5))
                    ax.scatter(res['y_test'], res['y_pred'], alpha=0.7, edgecolors='k')
                    ax.plot([res['y'].min(), res['y'].max()], [res['y'].min(), res['y'].max()], 'r--', lw=2)
                    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                    st.pyplot(fig)

                with t2:
                    importances = None
                    if res['is_tree_model']:
                        try:
                            explainer = shap.Explainer(res['model'], res['X_train'])
                            shap_values = explainer(res['X_test'])
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values, res['X_test'], show=False)
                            st.pyplot(fig)
                            importances = np.abs(shap_values.values).mean(axis=0)
                        except:
                            importances = res['model'].feature_importances_
                    else:
                        full_corr = res['X'].copy()
                        full_corr['Target'] = res['y'].values
                        importances = np.abs(full_corr.corr()['Target'].drop('Target').values)
                    
                with t3:
                    best_idx = res['y'].idxmax()
                    st.info(f"🏆 Best Sample: **ID {best_idx}** ({res['y'].max():.2f})")
                    
                    feat_imp_df = pd.DataFrame({'Feature': res['X'].columns, 'Imp': list(importances)})
                    top_feats = feat_imp_df.sort_values('Imp', ascending=False).head(5)['Feature'].tolist()
                    
                    best_recipe = res['df_clean'].loc[best_idx]
                    suggestions = []
                    for feat in top_feats:
                        orig = feat
                        for raw_c in res['X_raw_origin'].columns:
                            if re.sub(r'[^\w]', '_', str(raw_c)) in feat:
                                orig = raw_c
                                break
                        val = best_recipe.get(orig, "N/A")
                        
                        # Context
                        parts = str(orig).split('_')
                        prefix = "_".join(parts[:2]) if len(parts)>=2 else parts[0]
                        ctx = [f"{c.replace(prefix,'').strip('_')}:{best_recipe[c]}" for c in best_recipe.index if c!=orig and str(c).startswith(prefix) and pd.notna(best_recipe[c])]
                        
                        suggestions.append({"순위": top_feats.index(feat)+1, "중요 변수": feat, "최고 효율 조건": val, "세부 조건": " | ".join(ctx) if ctx else "-"})
                    st.table(pd.DataFrame(suggestions))
                
                st.markdown('<div class="bottom-spacer"></div>', unsafe_allow_html=True)
else:
    st.info("👈 왼쪽 사이드바에서 메인 데이터 파일을 업로드하세요.")

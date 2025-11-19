"""
BC-Viz: 유방암 데이터 분석 대시보드
병원 기업용 전문 데이터 분석 및 시각화 도구
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import base64
from io import BytesIO
import os
import platform

# 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance

# UMAP (선택적)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# SHAP (선택적)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# LIME (선택적)
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# 스타일 설정
st.set_page_config(
    page_title="BC-Viz 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 찾기 함수
def get_korean_font():
    """한글 폰트 찾기 및 반환"""
    font_list = [font.name for font in fm.fontManager.ttflist]
    korean_fonts = [
        'Malgun Gothic', '맑은 고딕', 'MalgunGothic',
        'NanumGothic', '나눔고딕', 'Nanum Gothic',
        'NanumBarunGothic', '나눔바른고딕',
        'AppleGothic', 'Apple Gothic',
        'Gulim', '굴림'
    ]
    
    for font in korean_fonts:
        if font in font_list:
            return font
    
    # 최후 수단: 시스템 폰트 경로에서 찾기
    if platform.system() == 'Windows':
        font_paths = [
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                # 맑은 고딕 찾기
                malgun_files = [f for f in os.listdir(font_path) if 'malgun' in f.lower() or '맑은' in f]
                if malgun_files:
                    return 'Malgun Gothic'
    
    return 'Arial Unicode MS'  # 기본값

# 한글 폰트 설정
KOREAN_FONT = get_korean_font()

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = KOREAN_FONT
plt.rcParams['axes.unicode_minus'] = False

# Plotly 폰트 설정 함수
def update_plotly_font(fig, dark_mode=False):
    """Plotly 차트에 한글 폰트 적용 및 다크 모드 지원"""
    # Plotly는 웹 환경에서 작동하므로, 시스템 폰트 이름을 일반적인 이름으로 변환
    plotly_font = KOREAN_FONT
    if 'Malgun' in KOREAN_FONT or '맑은' in KOREAN_FONT:
        plotly_font = 'Malgun Gothic'
    elif 'Nanum' in KOREAN_FONT or '나눔' in KOREAN_FONT:
        plotly_font = 'Nanum Gothic'
    
    # 다크 모드에 맞는 색상 설정
    if dark_mode:
        fig.update_layout(
            font=dict(
                family=plotly_font,
                size=12,
                color='#FFFFFF'
            ),
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            xaxis=dict(gridcolor='#333333'),
            yaxis=dict(gridcolor='#333333')
        )
    else:
        fig.update_layout(
            font=dict(
                family=plotly_font,
                size=12
            )
        )
    
    # 모든 텍스트 요소에 폰트 적용
    fig.update_xaxes(title_font=dict(family=plotly_font))
    fig.update_yaxes(title_font=dict(family=plotly_font))
    return fig

# 테마 변경 JavaScript 함수
def get_theme_script(dark_mode=False):
    """Streamlit 테마를 동적으로 변경하는 JavaScript"""
    theme = "dark" if dark_mode else "light"
    return f"""
    <script>
        (function() {{
            const theme = "{theme}";
            
            // 즉시 실행
            function applyTheme() {{
                const htmlElement = document.documentElement;
                const bodyElement = document.body;
                const appElement = document.querySelector('.stApp');
                const appViewContainer = document.querySelector('[data-testid="stAppViewContainer"]');
                const sidebar = document.querySelector('[data-testid="stSidebar"]');
                const header = document.querySelector('[data-testid="stHeader"]');
                
                if (theme === "dark") {{
                    // 다크 모드 설정
                    htmlElement.setAttribute('data-theme', 'dark');
                    if (appElement) {{
                        appElement.setAttribute('data-theme', 'dark');
                        appElement.style.backgroundColor = '#0E1117';
                    }}
                    if (appViewContainer) {{
                        appViewContainer.style.backgroundColor = '#0E1117';
                    }}
                    if (sidebar) {{
                        sidebar.style.backgroundColor = '#1E1E1E';
                    }}
                    if (header) {{
                        header.style.backgroundColor = '#1E1E1E';
                    }}
                    if (bodyElement) {{
                        bodyElement.style.backgroundColor = '#0E1117';
                        bodyElement.style.color = '#FFFFFF';
                    }}
                }} else {{
                    // 라이트 모드 설정
                    htmlElement.setAttribute('data-theme', 'light');
                    if (appElement) {{
                        appElement.setAttribute('data-theme', 'light');
                        appElement.style.backgroundColor = '#FFFFFF';
                    }}
                    if (appViewContainer) {{
                        appViewContainer.style.backgroundColor = '#FFFFFF';
                    }}
                    if (sidebar) {{
                        sidebar.style.backgroundColor = '#FFFFFF';
                    }}
                    if (header) {{
                        header.style.backgroundColor = '#FFFFFF';
                    }}
                    if (bodyElement) {{
                        bodyElement.style.backgroundColor = '#FFFFFF';
                        bodyElement.style.color = '#262730';
                    }}
                }}
            }}
            
            // 즉시 실행
            applyTheme();
            
            // DOM이 로드된 후에도 실행
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', applyTheme);
            }} else {{
                applyTheme();
            }}
            
            // MutationObserver로 동적 변경 감지
            const observer = new MutationObserver(function(mutations) {{
                applyTheme();
            }});
            
            observer.observe(document.body, {{
                childList: true,
                subtree: true
            }});
        }})();
    </script>
    """

# 다크 모드 CSS 함수
def get_css_style(dark_mode=False):
    """다크 모드에 따른 CSS 스타일 반환"""
    font_name = KOREAN_FONT
    
    if dark_mode:
        css = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

* {{
    font-family: '{font_name}', 'Noto Sans KR', 'Malgun Gothic', 'NanumGothic', sans-serif !important;
}}

html[data-theme="dark"], html[data-theme="dark"] body {{
    background-color: #0E1117 !important;
    color: #FFFFFF !important;
}}

.stApp[data-theme="dark"] {{
    background-color: #0E1117 !important;
}}

[data-testid="stAppViewContainer"] {{
    background-color: #0E1117 !important;
}}

.main {{
    background-color: #0E1117 !important;
    color: #FFFFFF !important;
}}

.main-header {{
    font-size: 2.5rem;
    font-weight: bold;
    color: #4ECDC4;
    text-align: center;
    padding: 1rem 0;
    font-family: '{font_name}', 'Noto Sans KR', sans-serif !important;
}}

.metric-card {{
    background-color: #1E1E1E;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4ECDC4;
    color: #FFFFFF;
}}

.stMetric {{
    background-color: #1E1E1E !important;
    padding: 0.5rem;
    border-radius: 0.25rem;
    color: #FFFFFF !important;
}}

.stMetric label {{
    color: #FFFFFF !important;
}}

.stMetric [data-testid="stMetricValue"] {{
    color: #E0E0E0 !important;
}}

.stDataFrame {{
    background-color: #1E1E1E !important;
}}

.stMarkdown, .stText, .stHeader, .stSubheader {{
    font-family: '{font_name}', 'Noto Sans KR', sans-serif !important;
    color: #FFFFFF !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background-color: #1E1E1E !important;
}}

.stTabs [data-baseweb="tab"] {{
    color: #FFFFFF !important;
}}

.element-container {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] {{
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] .stMarkdown {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] label {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] p {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] div {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] span {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
}}

[data-testid="stToggle"] {{
    color: #FFFFFF !important;
}}

[data-testid="stToggle"] label {{
    color: #FFFFFF !important;
}}

[data-testid="stToggle"] span {{
    color: #FFFFFF !important;
}}

[data-testid="stToggle"] * {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] label {{
    color: #FFFFFF !important;
}}

[data-testid="stHeader"] {{
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
}}
</style>"""
    else:
        css = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

* {{
    font-family: '{font_name}', 'Noto Sans KR', 'Malgun Gothic', 'NanumGothic', sans-serif !important;
}}

html[data-theme="light"], html[data-theme="light"] body {{
    background-color: #FFFFFF !important;
    color: #262730 !important;
}}

.stApp[data-theme="light"] {{
    background-color: #FFFFFF !important;
    color: #262730 !important;
}}

[data-testid="stAppViewContainer"] {{
    background-color: #FFFFFF !important;
    color: #262730 !important;
}}

.main-header {{
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
    font-family: '{font_name}', 'Noto Sans KR', sans-serif !important;
}}

.metric-card {{
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}}

.stMetric {{
    background-color: white;
    padding: 0.5rem;
    border-radius: 0.25rem;
    color: #262730 !important;
}}

.stMetric label {{
    color: #262730 !important;
}}

.stMetric [data-testid="stMetricValue"] {{
    color: #262730 !important;
}}

.stMarkdown, .stText, .stHeader, .stSubheader {{
    font-family: '{font_name}', 'Noto Sans KR', sans-serif !important;
    color: #262730 !important;
}}

.stMarkdown p, .stMarkdown div, .stMarkdown span {{
    color: #262730 !important;
}}

.stHeader, h1, h2, h3, h4, h5, h6 {{
    color: #262730 !important;
}}

.element-container {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] {{
    background-color: #FFFFFF !important;
    color: #262730 !important;
}}

[data-testid="stSidebar"] * {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] .stMarkdown {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] label {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] p {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] div {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] span {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #262730 !important;
}}

[data-testid="stToggle"] {{
    color: #262730 !important;
}}

[data-testid="stToggle"] label {{
    color: #262730 !important;
}}

[data-testid="stToggle"] span {{
    color: #262730 !important;
}}

[data-testid="stToggle"] * {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] {{
    color: #262730 !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] label {{
    color: #262730 !important;
}}

[data-testid="stHeader"] {{
    background-color: #FFFFFF !important;
    color: #262730 !important;
}}

.stSelectbox label, .stMultiselect label, .stSlider label {{
    color: #262730 !important;
}}

[data-testid="stSelectbox"] label {{
    color: #262730 !important;
}}

.stDataFrame {{
    background-color: #FFFFFF !important;
    color: #262730 !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background-color: #FFFFFF !important;
}}

.stTabs [data-baseweb="tab"] {{
    color: #262730 !important;
}}
</style>"""
    
    return css


# 로고 로드 함수
@st.cache_data
def load_logo():
    """로고 이미지 로드"""
    try:
        logo = Image.open("team_logo.png")
        return logo
    except FileNotFoundError:
        st.error("⚠️ team_logo.png 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"⚠️ 로고 로드 오류: {e}")
        return None

# 데이터 로드 함수
@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    try:
        # 데이터 파일 찾기
        if os.path.exists("kr_data.csv"):
            df = pd.read_csv("kr_data.csv")
        elif os.path.exists("data.csv"):
            df = pd.read_csv("data.csv")
        else:
            st.error("❌ 데이터 파일을 찾을 수 없습니다. (kr_data.csv 또는 data.csv)")
            return None, None, None
        
        # 타겟 컬럼 확인
        if "진단" in df.columns:
            target_col = "진단"
        elif "diagnosis" in df.columns:
            target_col = "diagnosis"
            df = df.rename(columns={"diagnosis": "진단"})
        else:
            st.error("❌ '진단' 또는 'diagnosis' 컬럼을 찾을 수 없습니다.")
            return None, None, None
        
        # 진단 인코딩
        if df[target_col].dtype == "object":
            df[target_col] = df[target_col].map({
                "M": 1, "B": 0,
                "악성(M)": 1, "양성(B)": 0,
                "악성": 1, "양성": 0
            })
        
        # Feature 컬럼 선택
        feature_cols = [c for c in df.columns 
                       if c not in ["id", "ID", target_col, "Unnamed: 32"]]
        feature_cols = [c for c in feature_cols 
                       if df[c].dtype in [np.int64, np.float64]]
        
        # 데이터 전처리
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # NaN 처리
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        return df, X_scaled, y, feature_cols
    
    except Exception as e:
        st.error(f"❌ 데이터 로드 오류: {e}")
        return None, None, None, None

# 모델 학습 함수
@st.cache_data
def train_models(X_scaled, y):
    """머신러닝 모델 학습"""
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # Decision Tree
    dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_clf.fit(X_train, y_train)
    dt_pred = dt_clf.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    return {
        "rf_clf": rf_clf,
        "dt_clf": dt_clf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "rf_accuracy": rf_accuracy,
        "dt_accuracy": dt_accuracy,
        "rf_pred": rf_pred,
        "dt_pred": dt_pred
    }

# 메인 함수
def main():
    # 다크 모드 상태 초기화
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # 다크 모드 토글 버튼
    with st.sidebar:
        st.markdown("---")
        dark_mode = st.toggle("🌙 다크 모드", value=st.session_state.dark_mode)
        st.session_state.dark_mode = dark_mode
    
    # CSS 스타일 적용
    css_content = get_css_style(st.session_state.dark_mode)
    st.markdown(css_content, unsafe_allow_html=True)
    
    # 테마 변경 JavaScript 적용
    js_content = get_theme_script(st.session_state.dark_mode)
    st.markdown(js_content, unsafe_allow_html=True)
    
    # 헤더에 로고 표시
    logo = load_logo()
    if logo:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo, use_container_width=True)
    
    st.markdown('<h1 class="main-header">📊 BC-Viz 데이터 분석 대시보드</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 데이터 로드
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        st.stop()
    
    # 사이드바
    st.sidebar.title("📋 메뉴")
    page = st.sidebar.selectbox(
        "분석 섹션 선택",
        [
            "🏠 데이터 개요",
            "📈 데이터 시각화",
            "🤖 머신러닝 모델",
            "🔍 XAI 분석",
            "🗺️ 차원 축소",
            "📊 상관관계 분석"
        ]
    )
    
    # 페이지별 내용
    if page == "🏠 데이터 개요":
        show_overview(df, y, st.session_state.dark_mode)
    
    elif page == "📈 데이터 시각화":
        show_visualizations(df, X_scaled, y, feature_cols, st.session_state.dark_mode)
    
    elif page == "🤖 머신러닝 모델":
        show_ml_models(df, X_scaled, y, feature_cols, st.session_state.dark_mode)
    
    elif page == "🔍 XAI 분석":
        show_xai_analysis(df, X_scaled, y, feature_cols, st.session_state.dark_mode)
    
    elif page == "🗺️ 차원 축소":
        show_dimension_reduction(X_scaled, y, feature_cols, st.session_state.dark_mode)
    
    elif page == "📊 상관관계 분석":
        show_correlation_analysis(X_scaled, feature_cols, st.session_state.dark_mode)

# 데이터 개요 페이지
def show_overview(df, y, dark_mode=False):
    st.header("🏠 데이터 개요")
    
    # 통계 카드 - 개선된 스타일
    st.markdown("### 데이터 개요")
    
    col1, col2, col3, col4 = st.columns(4)
    
    benign_count = int((y == 0).sum())
    malignant_count = int((y == 1).sum())
    benign_pct = round((y == 0).sum() / len(y) * 100, 2)
    
    # 카드 스타일 적용
    card_style = """
    <div style="background-color: %s; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin-bottom: 1rem;">
        <div style="color: %s; font-size: 0.9rem; margin-bottom: 0.5rem;">%s</div>
        <div style="color: %s; font-size: 2rem; font-weight: bold;">%s</div>
    </div>
    """
    
    bg_color = "#1E1E1E" if dark_mode else "#FFFFFF"
    text_color = "#FFFFFF" if dark_mode else "#1E1E1E"
    value_color = "#E0E0E0" if dark_mode else "#666666"
    
    with col1:
        st.markdown(card_style % (bg_color, text_color, "총 샘플 수", value_color, len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown(card_style % (bg_color, text_color, "양성 (B) 샘플", value_color, benign_count), unsafe_allow_html=True)
    
    with col3:
        st.markdown(card_style % (bg_color, text_color, "악성 (M) 샘플", value_color, malignant_count), unsafe_allow_html=True)
    
    with col4:
        st.markdown(card_style % (bg_color, text_color, "양성 비율", value_color, f"{benign_pct}%"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 데이터 분포 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        # 진단 분포 파이 차트
        diagnosis_counts = pd.Series({
            "양성 (B)": benign_count,
            "악성 (M)": malignant_count
        })
        
        fig_pie = px.pie(
            values=diagnosis_counts.values,
            names=diagnosis_counts.index,
            title="진단 분포",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie = update_plotly_font(fig_pie, dark_mode)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 진단 분포 바 차트
        fig_bar = px.bar(
            x=diagnosis_counts.index,
            y=diagnosis_counts.values,
            title="진단별 샘플 수",
            labels={"x": "진단", "y": "샘플 수"},
            color=diagnosis_counts.index,
            color_discrete_sequence=['#4ECDC4', '#FF6B9D']
        )
        fig_bar = update_plotly_font(fig_bar, dark_mode)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # 숫자형 컬럼 미리 계산
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 데이터 미리보기와 통계를 탭으로 구분
    tab1, tab2, tab3 = st.tabs(["📋 데이터 미리보기", "📊 데이터 통계", "📈 특징 요약"])
    
    with tab1:
        st.subheader("📋 데이터 미리보기")
        st.info(f"총 {len(df)}개의 샘플 중 처음 10개를 표시합니다.")
        # 숫자 형식 지정
        st.dataframe(
            df.head(10).style.format(precision=2),
            use_container_width=True,
            height=400
        )
        
        # 컬럼 정보
        st.subheader("📌 컬럼 정보")
        col_info = pd.DataFrame({
            '컬럼명': df.columns,
            '데이터 타입': df.dtypes,
            '결측치 수': df.isnull().sum(),
            '결측치 비율 (%)': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True, height=400)
    
    with tab2:
        st.subheader("📊 데이터 통계 요약")
        st.info("각 특징별 기술 통계량입니다.")
        
        if len(numeric_cols) > 0:
            # 스타일링 적용된 통계 테이블
            stats_df = df[numeric_cols].describe().T
            stats_df = stats_df.round(2)
            stats_df.columns = ['개수', '평균', '표준편차', '최솟값', '25%', '중앙값', '75%', '최댓값']
            
            # 색상 스타일 적용
            styled_stats = stats_df.style.background_gradient(
                subset=['평균', '표준편차'],
                cmap='YlOrRd'
            ).format(precision=2)
            
            st.dataframe(
                styled_stats,
                use_container_width=True,
                height=600
            )
            
            # 통계 카드로 주요 통계 표시
            st.subheader("📈 주요 특징 통계")
            if len(numeric_cols) >= 3:
                selected_stats_cols = st.multiselect(
                    "통계를 확인할 특징 선택",
                    options=numeric_cols,
                    default=numeric_cols[:5]
                )
                
                if selected_stats_cols:
                    col1, col2, col3 = st.columns(3)
                    for idx, col_name in enumerate(selected_stats_cols):
                        if idx % 3 == 0:
                            with col1:
                                st.metric(
                                    f"{col_name} 평균",
                                    f"{df[col_name].mean():.2f}",
                                    delta=f"±{df[col_name].std():.2f}"
                                )
                        elif idx % 3 == 1:
                            with col2:
                                st.metric(
                                    f"{col_name} 평균",
                                    f"{df[col_name].mean():.2f}",
                                    delta=f"±{df[col_name].std():.2f}"
                                )
                        else:
                            with col3:
                                st.metric(
                                    f"{col_name} 평균",
                                    f"{df[col_name].mean():.2f}",
                                    delta=f"±{df[col_name].std():.2f}"
                                )
        else:
            st.warning("숫자형 데이터가 없습니다.")
    
    with tab3:
        st.subheader("📈 특징 요약 정보")
        
        # 특징별 기본 정보
        summary_data = {
            '특징': numeric_cols if len(numeric_cols) > 0 else [],
            '평균': [df[col].mean() for col in numeric_cols] if len(numeric_cols) > 0 else [],
            '중앙값': [df[col].median() for col in numeric_cols] if len(numeric_cols) > 0 else [],
            '표준편차': [df[col].std() for col in numeric_cols] if len(numeric_cols) > 0 else [],
            '최솟값': [df[col].min() for col in numeric_cols] if len(numeric_cols) > 0 else [],
            '최댓값': [df[col].max() for col in numeric_cols] if len(numeric_cols) > 0 else []
        }
        
        if summary_data['특징']:
            summary_df = pd.DataFrame(summary_data).round(2)
            
            # 시각화로 표시
            st.dataframe(
                summary_df.style.background_gradient(subset=['평균', '표준편차'], cmap='Blues'),
                use_container_width=True,
                height=600
            )
            
            # 특징 개수 표시
            st.info(f"📊 총 {len(numeric_cols)}개의 숫자형 특징이 있습니다.")

# 데이터 시각화 페이지
def show_visualizations(df, X_scaled, y, feature_cols, dark_mode=False):
    st.header("📈 데이터 시각화")
    
    # Boxplot
    st.subheader("📦 Boxplot - 특징별 분포 및 이상치 확인")
    selected_features = st.multiselect(
        "시각화할 특징 선택",
        options=feature_cols[:10],  # 처음 10개 특징
        default=feature_cols[:5]
    )
    
    if selected_features:
        fig = make_subplots(
            rows=len(selected_features),
            cols=1,
            subplot_titles=selected_features,
            vertical_spacing=0.05
        )
        
        for i, feature in enumerate(selected_features, 1):
            if feature in X_scaled.columns:
                fig.add_trace(
                    go.Box(
                        y=X_scaled[feature],
                        name=feature,
                        boxmean='sd'
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            height=300 * len(selected_features),
            showlegend=False,
            title_text="특징별 Boxplot"
        )
        fig = update_plotly_font(fig, dark_mode)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Pairplot
    st.subheader("🔗 Pairplot - 특징 간 관계 분석")
    
    # Pairplot용 특징 목록 (data_analysis.ipynb와 동일)
    pairplot_features = [
        '반경 평균',
        '조직 평균',
        '면적 평균',
        '치밀도 평균',
        '좌우 대칭도 평균',
        '오목한 포인트 개수 평균',
        '둘레 길이 평균',
        '매끄러움 평균',
        '오목함 평균'
    ]
    
    # 존재하는 특징만 선택
    available_pairplot_features = [f for f in pairplot_features if f in feature_cols]
    
    if len(available_pairplot_features) >= 3:
        # 특징을 3개씩 묶어서 그룹 생성
        features_per_plot = 3
        n_plots = min(3, (len(available_pairplot_features) + features_per_plot - 1) // features_per_plot)
        
        feature_groups = []
        for i in range(n_plots):
            start_idx = i * features_per_plot
            end_idx = min(start_idx + features_per_plot, len(available_pairplot_features))
            if start_idx < len(available_pairplot_features):
                feature_groups.append(available_pairplot_features[start_idx:end_idx])
        
        st.info(f"총 {len(feature_groups)}개의 Pairplot 그래프를 생성합니다. (각 그래프당 3개 특징)")
        
        # Pairplot용 데이터 준비
        df_pairplot = X_scaled[available_pairplot_features].copy()
        df_pairplot['진단'] = y
        df_pairplot['진단_라벨'] = df_pairplot['진단'].map({0: '양성(B)', 1: '악성(M)'})
        
        # 각 그룹별로 Pairplot 생성
        for plot_idx, feature_group in enumerate(feature_groups, 1):
            st.write(f"**Pairplot {plot_idx}: {' & '.join(feature_group)}**")
            
            try:
                # matplotlib 한글 폰트 재설정
                plt.rcParams['font.family'] = KOREAN_FONT
                plt.rcParams['axes.unicode_minus'] = False
                
                # 다크 모드에 맞는 스타일 설정
                if dark_mode:
                    plt.style.use('dark_background')
                    bg_color = '#1E1E1E'
                    text_color = '#FFFFFF'
                else:
                    plt.style.use('whitegrid')
                    bg_color = '#FFFFFF'
                    text_color = '#262730'
                
                # Pairplot 생성
                g = sns.pairplot(
                    df_pairplot,
                    vars=feature_group,
                    hue='진단_라벨',
                    palette={'양성(B)': '#00CED1', '악성(M)': '#FF1493'},
                    diag_kind='kde',
                    plot_kws={'s': 60, 'alpha': 0.75, 'edgecolors': 'white', 'linewidth': 0.8},
                    diag_kws={'alpha': 0.85, 'linewidth': 3, 'fill': True},
                    height=4.5,
                    aspect=1
                )
                
                # 배경색 설정
                g.fig.patch.set_facecolor(bg_color)
                
                # 모든 subplot의 폰트 크기 및 색상 조정
                for ax in g.axes.flat:
                    if ax is not None:
                        ax.tick_params(labelsize=14, colors=text_color)
                        ax.set_facecolor(bg_color)
                        
                        # xlabel과 ylabel 설정
                        xlabel_text = ax.get_xlabel()
                        ylabel_text = ax.get_ylabel()
                        
                        if xlabel_text:
                            ax.set_xlabel(xlabel_text, fontsize=16, fontfamily=KOREAN_FONT, 
                                        fontweight='bold', color=text_color)
                        
                        if ylabel_text:
                            ax.set_ylabel(ylabel_text, fontsize=16, fontfamily=KOREAN_FONT, 
                                        fontweight='bold', color=text_color)
                        
                        # 제목 색상 설정
                        if ax.get_title():
                            ax.set_title(ax.get_title(), fontsize=16, fontfamily=KOREAN_FONT, 
                                        fontweight='bold', color=text_color)
                
                # 범례 색상 설정
                if g._legend:
                    for text in g._legend.get_texts():
                        text.set_color(text_color)
                        text.set_fontfamily(KOREAN_FONT)
                
                # 제목 설정
                g.fig.suptitle(
                    f'Pairplot {plot_idx}: {" & ".join(feature_group)}',
                    fontsize=18,
                    fontfamily=KOREAN_FONT,
                    fontweight='bold',
                    color=text_color,
                    y=1.02
                )
                
                st.pyplot(g.fig)
                plt.close(g.fig)
                
            except Exception as e:
                st.error(f"Pairplot 생성 오류: {e}")
                st.write(f"특징 그룹: {feature_group}")
    else:
        st.warning(f"⚠️ Pairplot을 생성하려면 최소 3개의 특징이 필요합니다. (현재: {len(available_pairplot_features)}개)")
        if available_pairplot_features:
            st.write(f"사용 가능한 특징: {available_pairplot_features}")
    
    st.markdown("---")
    
    # 레이더 차트
    st.subheader("🎯 레이더 차트 - 진단별 특징 비교")
    
    radar_features = st.multiselect(
        "레이더 차트에 사용할 특징 선택",
        options=feature_cols,
        default=['반경 평균', '조직 평균', '면적 평균', '치밀도 평균', '좌우 대칭도 평균', '오목한 포인트 개수 평균']
    )
    
    if radar_features:
        # 진단별 평균 계산
        X_with_diag = X_scaled.copy()
        X_with_diag['진단'] = y
        
        benign_avg = X_with_diag[X_with_diag['진단'] == 0][radar_features].mean()
        malignant_avg = X_with_diag[X_with_diag['진단'] == 1][radar_features].mean()
        
        # 레이더 차트
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=benign_avg.values,
            theta=radar_features,
            fill='toself',
            name='양성 (B)',
            line_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=malignant_avg.values,
            theta=radar_features,
            fill='toself',
            name='악성 (M)',
            line_color='#FF6B9D'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )),
            showlegend=True,
            title="진단별 평균 특징값 비교"
        )
        fig = update_plotly_font(fig, dark_mode)
        st.plotly_chart(fig, use_container_width=True)

# 머신러닝 모델 페이지
def show_ml_models(df, X_scaled, y, feature_cols, dark_mode=False):
    st.header("🤖 머신러닝 모델")
    
    with st.spinner("모델 학습 중..."):
        model_results = train_models(X_scaled, y)
    
    # 모델 성능 비교
    st.subheader("📊 모델 성능 비교")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Random Forest 정확도",
            f"{model_results['rf_accuracy']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Decision Tree 정확도",
            f"{model_results['dt_accuracy']*100:.2f}%"
        )
    
    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Random Forest Confusion Matrix
        cm_rf = confusion_matrix(model_results['y_test'], model_results['rf_pred'])
        fig_cm_rf = px.imshow(
            cm_rf,
            labels=dict(x="예측", y="실제", color="개수"),
            x=["양성 (B)", "악성 (M)"],
            y=["양성 (B)", "악성 (M)"],
            title="Random Forest Confusion Matrix",
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm_rf = update_plotly_font(fig_cm_rf, dark_mode)
        st.plotly_chart(fig_cm_rf, use_container_width=True)
    
    with col2:
        # Decision Tree Confusion Matrix
        cm_dt = confusion_matrix(model_results['y_test'], model_results['dt_pred'])
        fig_cm_dt = px.imshow(
            cm_dt,
            labels=dict(x="예측", y="실제", color="개수"),
            x=["양성 (B)", "악성 (M)"],
            y=["양성 (B)", "악성 (M)"],
            title="Decision Tree Confusion Matrix",
            color_continuous_scale='Oranges',
            text_auto=True
        )
        fig_cm_dt = update_plotly_font(fig_cm_dt, dark_mode)
        st.plotly_chart(fig_cm_dt, use_container_width=True)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance (Random Forest)")
    
    rf_importance = model_results['rf_clf'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=False)
    
    # 상위 15개 특징
    top_features = importance_df.head(15)
    
    fig_importance = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title="상위 15개 중요 특징",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig_importance = update_plotly_font(fig_importance, dark_mode)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # 분류 보고서
    st.subheader("📋 분류 보고서")
    
    tab1, tab2 = st.tabs(["Random Forest", "Decision Tree"])
    
    with tab1:
        st.text(classification_report(
            model_results['y_test'],
            model_results['rf_pred'],
            target_names=['양성 (B)', '악성 (M)']
        ))
    
    with tab2:
        st.text(classification_report(
            model_results['y_test'],
            model_results['dt_pred'],
            target_names=['양성 (B)', '악성 (M)']
        ))

# XAI 분석 페이지
def show_xai_analysis(df, X_scaled, y, feature_cols, dark_mode=False):
    st.header("🔍 XAI 분석")
    
    model_results = train_models(X_scaled, y)
    rf_clf = model_results['rf_clf']
    X_test = model_results['X_test']
    
    # SHAP 분석
    if SHAP_AVAILABLE:
        st.subheader("📊 SHAP 분석")
        
        if st.button("SHAP 분석 실행"):
            with st.spinner("SHAP 값 계산 중... (시간이 걸릴 수 있습니다)"):
                try:
                    # 샘플 추출 (너무 많으면 줄이기)
                    if len(X_test) > 100:
                        X_test_sample = X_test.sample(100, random_state=42)
                    else:
                        X_test_sample = X_test
                    
                    explainer = shap.TreeExplainer(rf_clf)
                    shap_values = explainer.shap_values(X_test_sample)
                    
                    # 이진분류에서 악성 클래스 SHAP 값
                    if isinstance(shap_values, list):
                        shap_values_class1 = shap_values[1]
                    else:
                        shap_values_class1 = shap_values
                    
                    # SHAP Summary Plot
                    st.write("**SHAP Summary Plot**")
                    fig_summary, ax = plt.subplots(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values_class1,
                        X_test_sample,
                        feature_names=feature_cols,
                        show=False
                    )
                    # 한글 폰트 재설정 (SHAP이 폰트를 변경할 수 있음)
                    plt.rcParams['font.family'] = KOREAN_FONT
                    plt.rcParams['axes.unicode_minus'] = False
                    st.pyplot(fig_summary)
                    
                    # SHAP Bar Plot
                    st.write("**상위 특징의 평균 절대 SHAP 값**")
                    mean_shap = np.abs(shap_values_class1).mean(axis=0)
                    top_indices = np.argsort(mean_shap)[::-1][:15]
                    
                    fig_bar, ax = plt.subplots(figsize=(10, 8))
                    ax.barh(
                        np.array(feature_cols)[top_indices][::-1],
                        mean_shap[top_indices][::-1]
                    )
                    ax.set_xlabel("Mean |SHAP value|", fontfamily=KOREAN_FONT)
                    ax.set_title("상위 15개 특징의 평균 절대 SHAP 값", fontfamily=KOREAN_FONT)
                    # y축 라벨 폰트 설정
                    for label in ax.get_yticklabels():
                        label.set_fontfamily(KOREAN_FONT)
                    st.pyplot(fig_bar)
                    
                except Exception as e:
                    st.error(f"SHAP 분석 오류: {e}")
    else:
        st.warning("⚠️ SHAP 라이브러리가 설치되어 있지 않습니다. `pip install shap`로 설치하세요.")
    
    # Permutation Importance
    st.subheader("🔄 Permutation Importance")
    
    if st.button("Permutation Importance 계산"):
        with st.spinner("Permutation Importance 계산 중..."):
            try:
                perm_importance = permutation_importance(
                    rf_clf,
                    X_test[:50],  # 샘플 크기 제한
                    model_results['y_test'][:50],
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                perm_importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': perm_importance.importances_mean
                }).sort_values('Importance', ascending=False)
                
                top_perm = perm_importance_df.head(15)
                
                fig_perm = px.bar(
                    top_perm,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Permutation Importance (상위 15개)",
                    color='Importance',
                    color_continuous_scale='Plasma'
                )
                fig_perm.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_perm = update_plotly_font(fig_perm, dark_mode)
                st.plotly_chart(fig_perm, use_container_width=True)
                
            except Exception as e:
                st.error(f"Permutation Importance 계산 오류: {e}")

# 차원 축소 페이지
def show_dimension_reduction(X_scaled, y, feature_cols, dark_mode=False):
    st.header("🗺️ 차원 축소")
    
    method = st.selectbox(
        "차원 축소 방법 선택",
        ["PCA", "t-SNE", "UMAP"] if UMAP_AVAILABLE else ["PCA", "t-SNE"]
    )
    
    if method == "PCA":
        st.subheader("📊 PCA (Principal Component Analysis)")
        
        n_components = st.slider("주성분 개수", 2, 3, 3)
        
        if st.button("PCA 실행"):
            with st.spinner("PCA 계산 중..."):
                pca = PCA(n_components=n_components, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                # 설명된 분산 비율
                explained_var = pca.explained_variance_ratio_
                st.write(f"**설명된 분산 비율**: {explained_var}")
                st.write(f"**총 설명된 분산**: {sum(explained_var):.2%}")
                
                # 2D 또는 3D 시각화
                if n_components == 2:
                    fig = px.scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        color=y,
                        color_discrete_map={0: '#4ECDC4', 1: '#FF6B9D'},
                        labels={'x': f'PC1 ({explained_var[0]:.2%})',
                               'y': f'PC2 ({explained_var[1]:.2%})',
                               'color': '진단'},
                        title="PCA 2D 시각화"
                    )
                    fig = update_plotly_font(fig, dark_mode)
                    st.plotly_chart(fig, use_container_width=True)
                else:  # 3D
                    fig = px.scatter_3d(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        z=X_pca[:, 2],
                        color=y,
                        color_discrete_map={0: '#4ECDC4', 1: '#FF6B9D'},
                        labels={'x': f'PC1 ({explained_var[0]:.2%})',
                               'y': f'PC2 ({explained_var[1]:.2%})',
                               'z': f'PC3 ({explained_var[2]:.2%})',
                               'color': '진단'},
                        title="PCA 3D 시각화"
                    )
                    fig = update_plotly_font(fig, dark_mode)
                    st.plotly_chart(fig, use_container_width=True)
    
    elif method == "t-SNE":
        st.subheader("📊 t-SNE (t-distributed Stochastic Neighbor Embedding)")
        
        perplexity = st.slider("Perplexity", 5, 50, 30)
        n_components = st.slider("차원 수", 2, 3, 3)
        
        if st.button("t-SNE 실행"):
            with st.spinner("t-SNE 계산 중... (시간이 걸릴 수 있습니다)"):
                tsne = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=perplexity
                )
                X_tsne = tsne.fit_transform(X_scaled)
                
                # 2D 또는 3D 시각화
                if n_components == 2:
                    fig = px.scatter(
                        x=X_tsne[:, 0],
                        y=X_tsne[:, 1],
                        color=y,
                        color_discrete_map={0: '#4ECDC4', 1: '#FF6B9D'},
                        labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': '진단'},
                        title="t-SNE 2D 시각화"
                    )
                    fig = update_plotly_font(fig, dark_mode)
                    st.plotly_chart(fig, use_container_width=True)
                else:  # 3D
                    fig = px.scatter_3d(
                        x=X_tsne[:, 0],
                        y=X_tsne[:, 1],
                        z=X_tsne[:, 2],
                        color=y,
                        color_discrete_map={0: '#4ECDC4', 1: '#FF6B9D'},
                        labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'z': 't-SNE 3', 'color': '진단'},
                        title="t-SNE 3D 시각화"
                    )
                    fig = update_plotly_font(fig, dark_mode)
                    st.plotly_chart(fig, use_container_width=True)
    
    elif method == "UMAP" and UMAP_AVAILABLE:
        st.subheader("📊 UMAP (Uniform Manifold Approximation and Projection)")
        
        n_neighbors = st.slider("Neighbors", 5, 50, 15)
        min_dist = st.slider("Min Distance", 0.0, 1.0, 0.1)
        n_components = st.slider("차원 수", 2, 3, 3)
        
        if st.button("UMAP 실행"):
            with st.spinner("UMAP 계산 중... (시간이 걸릴 수 있습니다)"):
                umap_reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist
                )
                X_umap = umap_reducer.fit_transform(X_scaled)
                
                # 2D 또는 3D 시각화
                if n_components == 2:
                    fig = px.scatter(
                        x=X_umap[:, 0],
                        y=X_umap[:, 1],
                        color=y,
                        color_discrete_map={0: '#4ECDC4', 1: '#FF6B9D'},
                        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'color': '진단'},
                        title="UMAP 2D 시각화"
                    )
                    fig = update_plotly_font(fig, dark_mode)
                    st.plotly_chart(fig, use_container_width=True)
                else:  # 3D
                    fig = px.scatter_3d(
                        x=X_umap[:, 0],
                        y=X_umap[:, 1],
                        z=X_umap[:, 2],
                        color=y,
                        color_discrete_map={0: '#4ECDC4', 1: '#FF6B9D'},
                        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3', 'color': '진단'},
                        title="UMAP 3D 시각화"
                    )
                    fig = update_plotly_font(fig, dark_mode)
                    st.plotly_chart(fig, use_container_width=True)

# 상관관계 분석 페이지
def show_correlation_analysis(X_scaled, feature_cols, dark_mode=False):
    st.header("📊 상관관계 분석")
    
    # 상관관계 행렬
    corr_matrix = X_scaled.corr()
    
    # 전체 상관관계 히트맵
    st.subheader("🔥 상관관계 히트맵 (전체)")
    
    # 샘플링 (너무 크면 일부만 표시)
    if len(feature_cols) > 20:
        st.info("⚠️ 특징이 많아 처음 20개만 표시합니다.")
        selected_features = feature_cols[:20]
        corr_subset = corr_matrix.loc[selected_features, selected_features]
    else:
        corr_subset = corr_matrix
        selected_features = feature_cols
    
    fig_corr = px.imshow(
        corr_subset,
        labels=dict(x="특징", y="특징", color="상관계수"),
        x=selected_features,
        y=selected_features,
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    fig_corr.update_layout(height=800)
    fig_corr = update_plotly_font(fig_corr, dark_mode)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 높은 상관관계 특징 쌍 찾기
    st.subheader("🔗 높은 상관관계 특징 쌍")
    
    threshold = st.slider("상관계수 임계값", 0.5, 0.99, 0.8, 0.01)
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(high_corr_df, use_container_width=True)
    else:
        st.info(f"상관계수가 {threshold} 이상인 특징 쌍이 없습니다.")

if __name__ == "__main__":
    import os
    main()


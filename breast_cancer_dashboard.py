"""
유방암 데이터 분석 대시보드
- 반응형 인터랙티브 대시보드
- 라이트/다크 모드 전환 지원
- 모든 분석 내용 포함
- 병원/회사용 전문 디자인
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 데이터 로드 및 전처리 (안정화)
# ============================================
print("데이터 로드 중...")
try:
    df = pd.read_csv("data.csv")
    if df.empty:
        raise ValueError("데이터 파일이 비어있습니다.")
except FileNotFoundError:
    raise FileNotFoundError("data.csv 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")
except Exception as e:
    raise Exception(f"데이터 로드 중 오류 발생: {str(e)}")

# 데이터 검증
if 'diagnosis' not in df.columns:
    raise ValueError("'diagnosis' 컬럼이 데이터에 없습니다.")

# diagnosis 인코딩 (M=1, B=0)
df['diagnosis_encoded'] = df['diagnosis'].map({'M': 1, 'B': 0})
df['diagnosis_label'] = df['diagnosis'].map({'M': '악성(M)', 'B': '양성(B)'})

# NaN 값 처리
df['diagnosis_encoded'] = df['diagnosis_encoded'].fillna(-1)
if (df['diagnosis_encoded'] == -1).any():
    print("⚠️ 경고: 일부 진단 값이 인식되지 않았습니다.")

# 특징 컬럼 선택
exclude_cols = ['id', 'diagnosis', 'diagnosis_encoded', 'diagnosis_label', 'Unnamed: 32']
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]

if len(feature_cols) == 0:
    raise ValueError("사용 가능한 특징 컬럼이 없습니다.")

# NaN 값이 있는 컬럼 제거
feature_cols = [col for col in feature_cols if not df[col].isnull().all()]

# 스케일링
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].fillna(df[feature_cols].median()))
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    df_scaled['diagnosis'] = df['diagnosis_encoded'].values
    df_scaled['diagnosis_label'] = df['diagnosis_label'].values
    
    # 유효한 진단 값만 유지
    valid_mask = df_scaled['diagnosis'] != -1
    df_scaled = df_scaled[valid_mask].copy()
    df = df[valid_mask].copy()
except Exception as e:
    raise Exception(f"데이터 스케일링 중 오류 발생: {str(e)}")

# 학습/테스트 분할
try:
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled[feature_cols],
        df_scaled['diagnosis'],
        test_size=0.2,
        random_state=42,
        stratify=df_scaled['diagnosis']
    )
except Exception as e:
    print(f"⚠️ 계층 분할 실패, 일반 분할 사용: {str(e)}")
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled[feature_cols],
        df_scaled['diagnosis'],
        test_size=0.2,
        random_state=42
    )

# 모델 학습
print("모델 학습 중...")
try:
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)

    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_clf.fit(X_train, y_train)
except Exception as e:
    raise Exception(f"모델 학습 중 오류 발생: {str(e)}")

# SHAP 설명자 준비 (샘플만 사용, 안정화)
shap_explainer = None
shap_values_class1 = None
try:
    print("SHAP 계산 중...")
    sample_size = min(100, len(X_test))
    if sample_size > 0:
        X_test_sample = X_test.sample(sample_size, random_state=42)
        shap_explainer = shap.TreeExplainer(rf_clf)
        shap_values = shap_explainer.shap_values(X_test_sample)
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_class1 = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
    else:
        print("⚠️ SHAP 계산을 위한 충분한 테스트 데이터가 없습니다.")
except Exception as e:
    print(f"⚠️ SHAP 계산 중 오류 발생 (계속 진행): {str(e)}")

# LIME 설명자 준비 (안정화)
lime_explainer = None
try:
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=['양성(B)', '악성(M)'],
        mode='classification',
        discretize_continuous=True
    )
except Exception as e:
    print(f"⚠️ LIME 설명자 생성 중 오류 발생 (계속 진행): {str(e)}")

print("✅ 데이터 준비 완료!")

# ============================================
# Dash 앱 초기화
# ============================================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "유방암 데이터 분석 대시보드"
server = app.server

# 커스텀 CSS 추가
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
            
            .stat-card {
                animation: fadeInUp 0.6s ease-out;
            }
            
            .stat-card:hover {
                transform: translateY(-8px) scale(1.02);
            }
            
            .content-card {
                animation: fadeInUp 0.8s ease-out;
            }
            
            .content-card:hover {
                transform: translateY(-4px);
            }
            
            .gradient-text {
                background: linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: shimmer 3s infinite linear;
                background-size: 200% auto;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================
# 고급 스타일 정의 (프리미엄 디자인)
# ============================================
LIGHT_THEME = {
    'bg': '#F8FAFC',
    'bg_pattern': 'radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.03) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(236, 72, 153, 0.03) 0%, transparent 50%)',
    'card_bg': '#FFFFFF',
    'card_bg_gradient': 'linear-gradient(145deg, #FFFFFF 0%, #F8F9FA 100%)',
    'text': '#0F172A',
    'text_secondary': '#64748B',
    'border': '#E2E8F0',
    'shadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    'shadow_hover': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    'shadow_large': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    'primary': '#6366F1',
    'primary_gradient': 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%)',
    'primary_glow': '0 0 20px rgba(99, 102, 241, 0.3)',
    'success': '#10B981',
    'success_gradient': 'linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%)',
    'success_glow': '0 0 20px rgba(16, 185, 129, 0.3)',
    'danger': '#EF4444',
    'danger_gradient': 'linear-gradient(135deg, #F87171 0%, #EF4444 50%, #DC2626 100%)',
    'danger_glow': '0 0 20px rgba(239, 68, 68, 0.3)',
    'warning': '#F59E0B',
    'warning_gradient': 'linear-gradient(135deg, #FBBF24 0%, #F59E0B 50%, #D97706 100%)',
    'warning_glow': '0 0 20px rgba(245, 158, 11, 0.3)',
    'info': '#06B6D4',
    'header_bg': 'linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%)',
    'header_shadow': '0 10px 30px rgba(102, 126, 234, 0.3)',
}

DARK_THEME = {
    'bg': '#0A0E27',
    'bg_pattern': 'radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(236, 72, 153, 0.1) 0%, transparent 50%)',
    'card_bg': '#1E293B',
    'card_bg_gradient': 'linear-gradient(145deg, #1E293B 0%, #0F172A 100%)',
    'text': '#F1F5F9',
    'text_secondary': '#94A3B8',
    'border': '#334155',
    'shadow': '0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3)',
    'shadow_hover': '0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 10px 10px -5px rgba(0, 0, 0, 0.4)',
    'shadow_large': '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
    'primary': '#818CF8',
    'primary_gradient': 'linear-gradient(135deg, #818CF8 0%, #A78BFA 50%, #C084FC 100%)',
    'primary_glow': '0 0 30px rgba(129, 140, 248, 0.5)',
    'success': '#34D399',
    'success_gradient': 'linear-gradient(135deg, #34D399 0%, #10B981 50%, #059669 100%)',
    'success_glow': '0 0 30px rgba(52, 211, 153, 0.5)',
    'danger': '#F87171',
    'danger_gradient': 'linear-gradient(135deg, #FCA5A5 0%, #F87171 50%, #EF4444 100%)',
    'danger_glow': '0 0 30px rgba(248, 113, 113, 0.5)',
    'warning': '#FBBF24',
    'warning_gradient': 'linear-gradient(135deg, #FCD34D 0%, #FBBF24 50%, #F59E0B 100%)',
    'warning_glow': '0 0 30px rgba(251, 191, 36, 0.5)',
    'info': '#22D3EE',
    'header_bg': 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #A855F7 100%)',
    'header_shadow': '0 10px 30px rgba(79, 70, 229, 0.5)',
}

# ============================================
# 헬퍼 함수 (프리미엄 스타일)
# ============================================
def get_card_style(theme, hover=False):
    return {
        'background': theme['card_bg_gradient'],
        'border': f"1px solid {theme['border']}",
        'borderRadius': '24px',
        'padding': '2rem',
        'boxShadow': theme['shadow_hover'] if hover else theme['shadow'],
        'height': '100%',
        'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        'backdropFilter': 'blur(20px)',
        'position': 'relative',
        'overflow': 'hidden',
    }

def get_stat_card_style(theme, color_type='primary'):
    color_map = {
        'primary': (theme['primary_gradient'], theme.get('primary_glow', '')),
        'success': (theme['success_gradient'], theme.get('success_glow', '')),
        'danger': (theme['danger_gradient'], theme.get('danger_glow', '')),
        'warning': (theme['warning_gradient'], theme.get('warning_glow', '')),
    }
    gradient, glow = color_map.get(color_type, (theme['primary_gradient'], ''))
    
    return {
        'background': gradient,
        'borderRadius': '28px',
        'padding': '2.5rem',
        'boxShadow': f"{theme['shadow_large']}, {glow}",
        'height': '100%',
        'color': '#FFFFFF',
        'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        'position': 'relative',
        'overflow': 'hidden',
        'cursor': 'pointer',
    }

def get_figure_layout(theme, title=""):
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {
            'color': theme['text'], 
            'family': 'Malgun Gothic, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif', 
            'size': 13
        },
        'title': {
            'text': title, 
            'x': 0.5, 
            'font': {
                'size': 22, 
                'color': theme['text'], 
                'family': 'Malgun Gothic, sans-serif'
            }, 
            'pad': {'t': 25, 'b': 25}
        },
        'margin': {'l': 70, 'r': 70, 't': 90, 'b': 70},
        'hovermode': 'closest',
    }

# ============================================
# 레이아웃 구성 (프리미엄 디자인)
# ============================================
app.layout = dbc.Container([
    dcc.Store(id='theme-store', data='dark'),
    
    # 프리미엄 헤더
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("유방암 데이터 분석 대시보드", className="mb-3 gradient-text", style={
                    'fontWeight': '800',
                    'fontSize': '3rem',
                    'letterSpacing': '-0.02em',
                    'lineHeight': '1.2',
                }),
                html.P("종합적인 데이터 분석 및 머신러닝 모델 성능 평가 시스템", className="mb-0", style={
                    'fontSize': '1.2rem',
                    'color': '#94A3B8',
                    'fontWeight': '400',
                    'letterSpacing': '0.01em',
                }),
            ])
        ], md=9),
        dbc.Col([
            html.Div([
                dbc.Button(
                    [html.I(className="bi bi-moon-stars-fill me-2"), " 다크 모드"],
                    id="theme-toggle",
                    color="secondary",
                    className="float-end",
                    style={
                        'marginTop': '10px',
                        'borderRadius': '16px',
                        'padding': '0.875rem 1.75rem',
                        'fontWeight': '600',
                        'fontSize': '1rem',
                        'boxShadow': '0 8px 16px rgba(0, 0, 0, 0.2)',
                        'border': 'none',
                        'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                        'background': 'linear-gradient(135deg, #475569 0%, #334155 100%)',
                        'color': '#FFFFFF',
                    }
                ),
            ], className="text-end"),
        ], md=3),
    ], className="mb-5", id="header-row", style={
        'padding': '2rem',
        'borderRadius': '24px',
        'background': DARK_THEME['card_bg_gradient'],
        'boxShadow': DARK_THEME['header_shadow'],
        'marginBottom': '3rem',
        'border': f"1px solid {DARK_THEME['border']}",
    }),
    
    # 통계 카드 (프리미엄 디자인)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-people-fill", style={
                            'fontSize': '3rem', 
                            'opacity': '0.9',
                            'filter': 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2))',
                        }),
                    ], className="mb-4"),
                    html.H6("전체 샘플", className="mb-2", style={
                        'opacity': '0.95', 
                        'fontSize': '1rem', 
                        'fontWeight': '600',
                        'letterSpacing': '0.05em',
                        'textTransform': 'uppercase',
                    }),
                    html.H2(f"{len(df):,}", className="mb-0", style={
                        'fontWeight': '800', 
                        'fontSize': '3rem',
                        'textShadow': '0 2px 4px rgba(0, 0, 0, 0.2)',
                        'letterSpacing': '-0.02em',
                    }),
                ], style={'textAlign': 'center', 'position': 'relative', 'zIndex': 1})
            ], id="card-total", className="h-100 stat-card", style=get_stat_card_style(DARK_THEME, 'primary')),
        ], md=3, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-exclamation-triangle-fill", style={
                            'fontSize': '3rem', 
                            'opacity': '0.9',
                            'filter': 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2))',
                        }),
                    ], className="mb-4"),
                    html.H6("악성 (M)", className="mb-2", style={
                        'opacity': '0.95', 
                        'fontSize': '1rem', 
                        'fontWeight': '600',
                        'letterSpacing': '0.05em',
                        'textTransform': 'uppercase',
                    }),
                    html.H2(f"{(df['diagnosis']=='M').sum():,}", className="mb-0", style={
                        'fontWeight': '800', 
                        'fontSize': '3rem',
                        'textShadow': '0 2px 4px rgba(0, 0, 0, 0.2)',
                        'letterSpacing': '-0.02em',
                    }),
                    html.P(f"{(df['diagnosis']=='M').sum()/len(df)*100:.1f}%", className="mb-0 mt-3", style={
                        'opacity': '0.95', 
                        'fontSize': '1.1rem',
                        'fontWeight': '500',
                    }),
                ], style={'textAlign': 'center', 'position': 'relative', 'zIndex': 1})
            ], id="card-malignant", className="h-100 stat-card", style=get_stat_card_style(DARK_THEME, 'danger')),
        ], md=3, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-check-circle-fill", style={
                            'fontSize': '3rem', 
                            'opacity': '0.9',
                            'filter': 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2))',
                        }),
                    ], className="mb-4"),
                    html.H6("양성 (B)", className="mb-2", style={
                        'opacity': '0.95', 
                        'fontSize': '1rem', 
                        'fontWeight': '600',
                        'letterSpacing': '0.05em',
                        'textTransform': 'uppercase',
                    }),
                    html.H2(f"{(df['diagnosis']=='B').sum():,}", className="mb-0", style={
                        'fontWeight': '800', 
                        'fontSize': '3rem',
                        'textShadow': '0 2px 4px rgba(0, 0, 0, 0.2)',
                        'letterSpacing': '-0.02em',
                    }),
                    html.P(f"{(df['diagnosis']=='B').sum()/len(df)*100:.1f}%", className="mb-0 mt-3", style={
                        'opacity': '0.95', 
                        'fontSize': '1.1rem',
                        'fontWeight': '500',
                    }),
                ], style={'textAlign': 'center', 'position': 'relative', 'zIndex': 1})
            ], id="card-benign", className="h-100 stat-card", style=get_stat_card_style(DARK_THEME, 'success')),
        ], md=3, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-graph-up-arrow", style={
                            'fontSize': '3rem', 
                            'opacity': '0.9',
                            'filter': 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2))',
                        }),
                    ], className="mb-4"),
                    html.H6("모델 정확도", className="mb-2", style={
                        'opacity': '0.95', 
                        'fontSize': '1rem', 
                        'fontWeight': '600',
                        'letterSpacing': '0.05em',
                        'textTransform': 'uppercase',
                    }),
                    html.H2(f"{accuracy_score(y_test, rf_clf.predict(X_test))*100:.1f}%", className="mb-0", style={
                        'fontWeight': '800', 
                        'fontSize': '3rem',
                        'textShadow': '0 2px 4px rgba(0, 0, 0, 0.2)',
                        'letterSpacing': '-0.02em',
                    }, id="accuracy-display"),
                    html.P("Random Forest", className="mb-0 mt-3", style={
                        'opacity': '0.95', 
                        'fontSize': '1.1rem',
                        'fontWeight': '500',
                    }),
                ], style={'textAlign': 'center', 'position': 'relative', 'zIndex': 1})
            ], id="card-accuracy", className="h-100 stat-card", style=get_stat_card_style(DARK_THEME, 'warning')),
        ], md=3, className="mb-4"),
    ], className="mb-5"),
    
    # 탭 (프리미엄 스타일)
    dbc.Tabs([
        dbc.Tab(
            label="📊 데이터 탐색", 
            tab_id="tab-explore", 
            tab_style={'marginRight': '12px', 'padding': '0.75rem 1.5rem'},
            label_style={
                'fontWeight': '700', 
                'fontSize': '1.1rem',
                'letterSpacing': '0.02em',
                'transition': 'all 0.3s ease',
            }
        ),
        dbc.Tab(
            label="🔬 차원 축소", 
            tab_id="tab-dimension", 
            tab_style={'marginRight': '12px', 'padding': '0.75rem 1.5rem'},
            label_style={
                'fontWeight': '700', 
                'fontSize': '1.1rem',
                'letterSpacing': '0.02em',
                'transition': 'all 0.3s ease',
            }
        ),
        dbc.Tab(
            label="🤖 모델 성능", 
            tab_id="tab-model", 
            tab_style={'marginRight': '12px', 'padding': '0.75rem 1.5rem'},
            label_style={
                'fontWeight': '700', 
                'fontSize': '1.1rem',
                'letterSpacing': '0.02em',
                'transition': 'all 0.3s ease',
            }
        ),
        dbc.Tab(
            label="🧠 XAI 분석", 
            tab_id="tab-xai", 
            tab_style={'padding': '0.75rem 1.5rem'},
            label_style={
                'fontWeight': '700', 
                'fontSize': '1.1rem',
                'letterSpacing': '0.02em',
                'transition': 'all 0.3s ease',
            }
        ),
    ], id="main-tabs", active_tab="tab-explore", className="mb-4", style={
        'borderBottom': f"3px solid {DARK_THEME['border']}",
        'paddingBottom': '0.5rem',
    }),
    
    html.Div(id="tab-content", className="mt-4"),
    
], fluid=True, id="main-container", style={
    'backgroundColor': DARK_THEME['bg'],
    'backgroundImage': DARK_THEME['bg_pattern'],
    'minHeight': '100vh',
    'padding': '3rem',
    'color': DARK_THEME['text'],
    'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
})

# ============================================
# 콜백: 테마 전환 (수정)
# ============================================
@app.callback(
    [Output('theme-store', 'data'),
     Output('theme-toggle', 'children'),
     Output('main-container', 'style'),
     Output('card-total', 'style'),
     Output('card-malignant', 'style'),
     Output('card-benign', 'style'),
     Output('card-accuracy', 'style'),
     Output('header-row', 'style')],
    [Input('theme-toggle', 'n_clicks')],
    [State('theme-store', 'data')],
    prevent_initial_call=False
)
def toggle_theme(n_clicks, current_theme):
    # 초기 로드 시 또는 클릭 시 처리
    if n_clicks is None or n_clicks == 0:
        # 초기 로드: 다크 모드로 시작
        new_theme = 'dark'
        theme = DARK_THEME
        button_text = [html.I(className="bi bi-sun-fill me-2"), " 라이트 모드"]
    else:
        # 클릭 시: 토글
        if current_theme == 'dark':
            new_theme = 'light'
            theme = LIGHT_THEME
            button_text = [html.I(className="bi bi-moon-stars-fill me-2"), " 다크 모드"]
        else:
            new_theme = 'dark'
            theme = DARK_THEME
            button_text = [html.I(className="bi bi-sun-fill me-2"), " 라이트 모드"]
    
    container_style = {
        'backgroundColor': theme['bg'],
        'backgroundImage': theme.get('bg_pattern', ''),
        'minHeight': '100vh',
        'padding': '3rem',
        'color': theme['text'],
        'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
    }
    
    header_style = {
        'padding': '2rem',
        'borderRadius': '24px',
        'background': theme['card_bg_gradient'],
        'boxShadow': theme.get('header_shadow', theme['shadow']),
        'marginBottom': '3rem',
        'border': f"1px solid {theme['border']}",
    }
    
    card_style_total = get_stat_card_style(theme, 'primary')
    card_style_malignant = get_stat_card_style(theme, 'danger')
    card_style_benign = get_stat_card_style(theme, 'success')
    card_style_accuracy = get_stat_card_style(theme, 'warning')
    
    return new_theme, button_text, container_style, card_style_total, card_style_malignant, card_style_benign, card_style_accuracy, header_style

# ============================================
# 콜백: 탭 내용
# ============================================
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab'),
     Input('theme-store', 'data')]
)
def render_tab_content(active_tab, theme):
    try:
        if theme is None:
            theme = 'dark'
        
        if active_tab is None:
            active_tab = "tab-explore"
        
        if theme == 'dark':
            theme_colors = DARK_THEME
        else:
            theme_colors = LIGHT_THEME
        
        if active_tab == "tab-explore":
            return render_explore_tab(theme_colors)
        elif active_tab == "tab-dimension":
            return render_dimension_tab(theme_colors)
        elif active_tab == "tab-model":
            return render_model_tab(theme_colors)
        elif active_tab == "tab-xai":
            return render_xai_tab(theme_colors)
        return html.Div("탭을 선택하세요", style={'padding': '2rem', 'textAlign': 'center'})
    except Exception as e:
        return html.Div([
            html.H5("오류 발생", style={'color': 'red'}),
            html.P(str(e), style={'color': 'red', 'padding': '1rem'})
        ], style={'padding': '2rem'})

# ============================================
# 탭 렌더링 함수들
# ============================================
def render_explore_tab(theme):
    return dbc.Row([
        # 상관관계 히트맵
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("상관관계 히트맵", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="correlation-heatmap",
                        figure=create_correlation_heatmap(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=12, className="mb-4"),
        
        # Boxplot
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("특징별 Boxplot (이상치 탐색)", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Dropdown(
                        id="boxplot-feature",
                        options=[{'label': col, 'value': col} for col in (feature_cols[:10] if len(feature_cols) > 0 else [])],
                        value=feature_cols[0] if len(feature_cols) > 0 else None,
                        clearable=False,
                        style={
                            'backgroundColor': theme['card_bg'], 
                            'color': theme['text'],
                            'borderRadius': '12px',
                            'border': f"1px solid {theme['border']}",
                        }
                    ),
                    html.Div(id="boxplot-graph-container"),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # 바이올린 차트
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("바이올린 차트", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Dropdown(
                        id="violin-feature",
                        options=[{'label': col, 'value': col} for col in (feature_cols[:10] if len(feature_cols) > 0 else [])],
                        value=feature_cols[0] if len(feature_cols) > 0 else None,
                        clearable=False,
                        style={
                            'backgroundColor': theme['card_bg'], 
                            'color': theme['text'],
                            'borderRadius': '12px',
                            'border': f"1px solid {theme['border']}",
                        }
                    ),
                    html.Div(id="violin-graph-container"),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # 버블 차트
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("버블 차트", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="bubble-chart",
                        figure=create_bubble_chart(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=12, className="mb-4"),
        
        # 레이더 차트
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("레이더 차트: 진단별 평균 특징값", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="radar-chart",
                        figure=create_radar_chart(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=12, className="mb-4"),
    ])

def render_dimension_tab(theme):
    return dbc.Row([
        # PCA 2D
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("PCA 2D 투영", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="pca-2d",
                        figure=create_pca_2d(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # PCA 3D
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("PCA 3D 투영", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="pca-3d",
                        figure=create_pca_3d(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # t-SNE 2D
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("t-SNE 2D", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="tsne-2d",
                        figure=create_tsne_2d(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # t-SNE 3D
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("t-SNE 3D", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="tsne-3d",
                        figure=create_tsne_3d(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
    ])

def render_model_tab(theme):
    # 모델 성능 메트릭
    rf_pred = rf_clf.predict(X_test)
    dt_pred = dt_clf.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_prec = precision_score(y_test, dt_pred)
    dt_rec = recall_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    
    return dbc.Row([
        # 모델 성능 비교
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("모델 성능 비교", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="model-comparison",
                        figure=create_model_comparison(theme, rf_acc, rf_prec, rf_rec, rf_f1, dt_acc, dt_prec, dt_rec, dt_f1),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=12, className="mb-4"),
        
        # 혼동 행렬
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Random Forest 혼동 행렬", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="confusion-matrix-rf",
                        figure=create_confusion_matrix(theme, y_test, rf_pred, "Random Forest"),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Decision Tree 혼동 행렬", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="confusion-matrix-dt",
                        figure=create_confusion_matrix(theme, y_test, dt_pred, "Decision Tree"),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # 특징 중요도
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Random Forest 특징 중요도", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="feature-importance",
                        figure=create_feature_importance(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=12, className="mb-4"),
    ])

def render_xai_tab(theme):
    return dbc.Row([
        # SHAP Summary
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("SHAP Summary Plot", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="shap-summary",
                        figure=create_shap_summary(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=12, className="mb-4"),
        
        # SHAP Bar
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("SHAP Bar Plot (상위 15개 특징)", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="shap-bar",
                        figure=create_shap_bar(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # Permutation Importance
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Permutation Importance", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Graph(
                        id="permutation-importance",
                        figure=create_permutation_importance(theme),
                        config={'displayModeBar': True, 'displaylogo': False}
                    ),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=6, className="mb-4"),
        
        # Partial Dependence Plot
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Partial Dependence Plot", className="mb-4", style={
                        'fontWeight': '700', 
                        'fontSize': '1.5rem',
                        'color': theme['text'],
                        'letterSpacing': '0.02em',
                    }),
                    dcc.Dropdown(
                        id="pdp-feature",
                        options=[{'label': col, 'value': col} for col in (feature_cols[:10] if len(feature_cols) > 0 else [])],
                        value=feature_cols[0] if len(feature_cols) > 0 else None,
                        clearable=False,
                        style={
                            'backgroundColor': theme['card_bg'], 
                            'color': theme['text'],
                            'borderRadius': '12px',
                            'border': f"1px solid {theme['border']}",
                        }
                    ),
                    html.Div(id="pdp-graph-container"),
                ])
            ], className="content-card", style=get_card_style(theme)),
        ], md=12, className="mb-4"),
    ])

# ============================================
# 그래프 생성 함수들
# ============================================
def create_correlation_heatmap(theme):
    corr_matrix = df_scaled[feature_cols[:15]].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 9, "color": theme['text']},
        hovertemplate='%{x} vs %{y}<br>상관계수: %{z:.3f}<extra></extra>',
    ))
    
    fig.update_layout(**get_figure_layout(theme, "특징 간 상관관계 히트맵"))
    return fig

def create_bubble_chart(theme):
    bubble_size = (df_scaled['area_mean'] - df_scaled['area_mean'].min()) / (
        df_scaled['area_mean'].max() - df_scaled['area_mean'].min()) * 50 + 10
    
    fig = go.Figure()
    
    for diagnosis_val, label, color in [(0, '양성(B)', theme['success']), (1, '악성(M)', theme['danger'])]:
        mask = df_scaled['diagnosis'] == diagnosis_val
        fig.add_trace(go.Scatter(
            x=df_scaled.loc[mask, 'radius_mean'],
            y=df_scaled.loc[mask, 'concave points_mean'],
            mode='markers',
            name=label,
            marker=dict(
                size=bubble_size[mask],
                color=color,
                opacity=0.7,
                line=dict(width=1.5, color=theme['text'])
            ),
            hovertemplate=f'<b>{label}</b><br>반경: %{{x:.2f}}<br>오목한 점: %{{y:.2f}}<extra></extra>',
        ))
    
    fig.update_layout(
        **get_figure_layout(theme, "버블 차트: 반경 vs 오목한 점 (크기=면적)"),
        xaxis_title="반경 평균",
        yaxis_title="오목한 점 평균",
    )
    return fig

def create_radar_chart(theme):
    radar_features = ['radius_mean', 'texture_mean', 'area_mean', 'compactness_mean', 'symmetry_mean', 'concave points_mean']
    radar_data = df_scaled.groupby('diagnosis')[radar_features].mean()
    
    fig = go.Figure()
    
    for diagnosis_val, label, color in [(0, '양성(B)', theme['success']), (1, '악성(M)', theme['danger'])]:
        values = radar_data.loc[diagnosis_val].values.tolist()
        values.append(values[0])  # 닫힌 형태
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_features + [radar_features[0]],
            fill='toself',
            name=label,
            line_color=color,
            fillcolor=color,
            opacity=0.3,
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2, 2], gridcolor=theme['border'], linecolor=theme['border']),
            angularaxis=dict(gridcolor=theme['border'], linecolor=theme['border'])
        ),
        **get_figure_layout(theme, "레이더 차트: 진단별 평균 특징값 비교"),
    )
    return fig

def create_pca_2d(theme):
    pca = PCA(n_components=2, random_state=42)
    X_pca_2d = pca.fit_transform(df_scaled[feature_cols])
    explained_var = pca.explained_variance_ratio_
    
    fig = go.Figure()
    
    for diagnosis_val, label, color in [(0, '양성(B)', theme['success']), (1, '악성(M)', theme['danger'])]:
        mask = df_scaled['diagnosis'] == diagnosis_val
        fig.add_trace(go.Scatter(
            x=X_pca_2d[mask, 0],
            y=X_pca_2d[mask, 1],
            mode='markers',
            name=label,
            marker=dict(color=color, opacity=0.7, size=10, line=dict(width=1, color=theme['text'])),
            hovertemplate=f'<b>{label}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>',
        ))
    
    fig.update_layout(
        **get_figure_layout(theme, f"PCA 2D 투영 (설명 분산: {explained_var.sum()*100:.1f}%)"),
        xaxis_title=f"주성분 1 ({explained_var[0]*100:.1f}%)",
        yaxis_title=f"주성분 2 ({explained_var[1]*100:.1f}%)",
    )
    return fig

def create_pca_3d(theme):
    pca = PCA(n_components=3, random_state=42)
    X_pca_3d = pca.fit_transform(df_scaled[feature_cols])
    explained_var = pca.explained_variance_ratio_
    
    fig = go.Figure()
    
    for diagnosis_val, label, color in [(0, '양성(B)', theme['success']), (1, '악성(M)', theme['danger'])]:
        mask = df_scaled['diagnosis'] == diagnosis_val
        fig.add_trace(go.Scatter3d(
            x=X_pca_3d[mask, 0],
            y=X_pca_3d[mask, 1],
            z=X_pca_3d[mask, 2],
            mode='markers',
            name=label,
            marker=dict(color=color, opacity=0.7, size=6, line=dict(width=0.5, color=theme['text'])),
            hovertemplate=f'<b>{label}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>',
        ))
    
    fig.update_layout(
        **get_figure_layout(theme, f"PCA 3D 투영 (설명 분산: {explained_var.sum()*100:.1f}%)"),
        scene=dict(
            xaxis_title=f"주성분 1 ({explained_var[0]*100:.1f}%)",
            yaxis_title=f"주성분 2 ({explained_var[1]*100:.1f}%)",
            zaxis_title=f"주성분 3 ({explained_var[2]*100:.1f}%)",
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor=theme['border'], linecolor=theme['border']),
            yaxis=dict(gridcolor=theme['border'], linecolor=theme['border']),
            zaxis=dict(gridcolor=theme['border'], linecolor=theme['border']),
        ),
    )
    return fig

def create_tsne_2d(theme):
    try:
        # 샘플링 (t-SNE는 계산량이 많음)
        sample_size = min(500, len(df_scaled))
        if sample_size < 4:
            raise ValueError("t-SNE 계산을 위한 충분한 데이터가 없습니다.")
        
        sample_idx = np.random.choice(len(df_scaled), sample_size, replace=False)
        X_sample = df_scaled[feature_cols].iloc[sample_idx]
        y_sample = df_scaled['diagnosis'].iloc[sample_idx]
        
        perplexity = min(30, (sample_size - 1) // 3)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne_2d = tsne.fit_transform(X_sample)
        
        fig = go.Figure()
        
        for diagnosis_val, label, color in [(0, '양성(B)', theme['success']), (1, '악성(M)', theme['danger'])]:
            mask = (y_sample == diagnosis_val).values
            if mask.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=X_tsne_2d[mask, 0],
                    y=X_tsne_2d[mask, 1],
                    mode='markers',
                    name=label,
                    marker=dict(color=color, opacity=0.7, size=10, line=dict(width=1, color=theme['text'])),
                    hovertemplate=f'<b>{label}</b><br>t-SNE 1: %{{x:.2f}}<br>t-SNE 2: %{{y:.2f}}<extra></extra>',
                ))
        
        fig.update_layout(
            **get_figure_layout(theme, "t-SNE 2D 비선형 차원 축소"),
            xaxis_title="t-SNE 차원 1",
            yaxis_title="t-SNE 차원 2",
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"t-SNE 계산 오류: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=theme['text'])
        )
        fig.update_layout(**get_figure_layout(theme, "t-SNE 2D 비선형 차원 축소"))
        return fig

def create_tsne_3d(theme):
    try:
        # 샘플링
        sample_size = min(500, len(df_scaled))
        if sample_size < 4:
            raise ValueError("t-SNE 계산을 위한 충분한 데이터가 없습니다.")
        
        sample_idx = np.random.choice(len(df_scaled), sample_size, replace=False)
        X_sample = df_scaled[feature_cols].iloc[sample_idx]
        y_sample = df_scaled['diagnosis'].iloc[sample_idx]
        
        perplexity = min(30, (sample_size - 1) // 3)
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        X_tsne_3d = tsne.fit_transform(X_sample)
        
        fig = go.Figure()
        
        for diagnosis_val, label, color in [(0, '양성(B)', theme['success']), (1, '악성(M)', theme['danger'])]:
            mask = (y_sample == diagnosis_val).values
            if mask.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_tsne_3d[mask, 0],
                    y=X_tsne_3d[mask, 1],
                    z=X_tsne_3d[mask, 2],
                    mode='markers',
                    name=label,
                    marker=dict(color=color, opacity=0.7, size=6, line=dict(width=0.5, color=theme['text'])),
                    hovertemplate=f'<b>{label}</b><br>t-SNE 1: %{{x:.2f}}<br>t-SNE 2: %{{y:.2f}}<br>t-SNE 3: %{{z:.2f}}<extra></extra>',
                ))
        
        fig.update_layout(
            **get_figure_layout(theme, "t-SNE 3D 비선형 차원 축소"),
            scene=dict(
                xaxis_title="t-SNE 차원 1",
                yaxis_title="t-SNE 차원 2",
                zaxis_title="t-SNE 차원 3",
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor=theme['border'], linecolor=theme['border']),
                yaxis=dict(gridcolor=theme['border'], linecolor=theme['border']),
                zaxis=dict(gridcolor=theme['border'], linecolor=theme['border']),
            ),
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"t-SNE 계산 오류: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=theme['text'])
        )
        fig.update_layout(**get_figure_layout(theme, "t-SNE 3D 비선형 차원 축소"))
        return fig

def create_model_comparison(theme, rf_acc, rf_prec, rf_rec, rf_f1, dt_acc, dt_prec, dt_rec, dt_f1):
    fig = go.Figure()
    
    metrics = ['정확도', '정밀도', '재현율', 'F1 점수']
    rf_values = [rf_acc, rf_prec, rf_rec, rf_f1]
    dt_values = [dt_acc, dt_prec, dt_rec, dt_f1]
    
    fig.add_trace(go.Bar(
        name='Random Forest',
        x=metrics,
        y=rf_values,
        marker_color=theme['primary'],
        marker_line=dict(color=theme['text'], width=1.5),
        hovertemplate='<b>Random Forest</b><br>%{x}: %{y:.3f}<extra></extra>',
    ))
    
    fig.add_trace(go.Bar(
        name='Decision Tree',
        x=metrics,
        y=dt_values,
        marker_color=theme['info'],
        marker_line=dict(color=theme['text'], width=1.5),
        hovertemplate='<b>Decision Tree</b><br>%{x}: %{y:.3f}<extra></extra>',
    ))
    
    fig.update_layout(
        **get_figure_layout(theme, "모델 성능 비교"),
        yaxis_title="점수",
        barmode='group',
        yaxis=dict(range=[0, 1.1]),
    )
    return fig

def create_confusion_matrix(theme, y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['양성(B)', '악성(M)'],
        y=['양성(B)', '악성(M)'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 18, "color": theme['text']},
        hovertemplate='실제: %{y}<br>예측: %{x}<br>개수: %{z}<extra></extra>',
    ))
    
    fig.update_layout(
        **get_figure_layout(theme, f"{model_name} 혼동 행렬"),
    )
    return fig

def create_feature_importance(theme):
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importances[indices],
        y=[feature_cols[i] for i in indices],
        orientation='h',
        marker_color=theme['primary'],
        marker_line=dict(color=theme['text'], width=1),
        hovertemplate='<b>%{y}</b><br>중요도: %{x:.4f}<extra></extra>',
    ))
    
    fig.update_layout(
        **get_figure_layout(theme, "Random Forest 특징 중요도 (상위 15개)"),
        xaxis_title="중요도",
        yaxis_title="특징",
    )
    return fig

def create_shap_summary(theme):
    if shap_values_class1 is None:
        # SHAP 값이 없을 경우 빈 그래프 반환
        fig = go.Figure()
        fig.add_annotation(
            text="SHAP 값이 계산되지 않았습니다.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=theme['text'])
        )
        fig.update_layout(**get_figure_layout(theme, "SHAP Summary Plot"))
        return fig
    
    try:
        mean_shap = np.abs(shap_values_class1).mean(axis=0)
        indices = np.argsort(mean_shap)[::-1][:15]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mean_shap[indices],
            y=[feature_cols[i] for i in indices if i < len(feature_cols)],
            orientation='h',
            marker_color=theme['warning'],
            marker_line=dict(color=theme['text'], width=1),
            hovertemplate='<b>%{y}</b><br>평균 |SHAP 값|: %{x:.4f}<extra></extra>',
        ))
        
        fig.update_layout(
            **get_figure_layout(theme, "SHAP Summary Plot (상위 15개 특징)"),
            xaxis_title="평균 |SHAP 값|",
            yaxis_title="특징",
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"SHAP 그래프 생성 오류: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=theme['text'])
        )
        fig.update_layout(**get_figure_layout(theme, "SHAP Summary Plot"))
        return fig

def create_shap_bar(theme):
    if shap_values_class1 is None:
        fig = go.Figure()
        fig.add_annotation(
            text="SHAP 값이 계산되지 않았습니다.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=theme['text'])
        )
        fig.update_layout(**get_figure_layout(theme, "SHAP Bar Plot"))
        return fig
    
    try:
        mean_shap = np.abs(shap_values_class1).mean(axis=0)
        indices = np.argsort(mean_shap)[::-1][:15]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[feature_cols[i] for i in indices if i < len(feature_cols)],
            y=mean_shap[indices],
            marker_color=theme['warning'],
            marker_line=dict(color=theme['text'], width=1),
            hovertemplate='<b>%{x}</b><br>평균 |SHAP 값|: %{y:.4f}<extra></extra>',
        ))
        
        fig.update_layout(
            **get_figure_layout(theme, "SHAP Bar Plot"),
            xaxis_title="특징",
            yaxis_title="평균 |SHAP 값|",
            xaxis_tickangle=-45,
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"SHAP 그래프 생성 오류: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=theme['text'])
        )
        fig.update_layout(**get_figure_layout(theme, "SHAP Bar Plot"))
        return fig

def create_permutation_importance(theme):
    try:
        result = permutation_importance(rf_clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances_mean = result.importances_mean
        importances_std = result.importances_std
        
        indices = np.argsort(importances_mean)[::-1][:15]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[feature_cols[i] for i in indices if i < len(feature_cols)],
            y=importances_mean[indices],
            error_y=dict(type='data', array=importances_std[indices], color=theme['text']),
            marker_color=theme['info'],
            marker_line=dict(color=theme['text'], width=1),
            hovertemplate='<b>%{x}</b><br>정확도 감소: %{y:.4f} ± %{error_y.array:.4f}<extra></extra>',
        ))
        
        fig.update_layout(
            **get_figure_layout(theme, "Permutation Importance (상위 15개 특징)"),
            xaxis_title="특징",
            yaxis_title="정확도 감소",
            xaxis_tickangle=-45,
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Permutation Importance 계산 오류: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=theme['text'])
        )
        fig.update_layout(**get_figure_layout(theme, "Permutation Importance"))
        return fig

# ============================================
# 인터랙티브 콜백
# ============================================
@app.callback(
    Output('boxplot-graph-container', 'children'),
    [Input('boxplot-feature', 'value'),
     Input('theme-store', 'data')]
)
def update_boxplot(feature, theme):
    try:
        if theme is None:
            theme = 'dark'
        
        if feature is None or feature not in df.columns:
            feature = feature_cols[0] if len(feature_cols) > 0 else df.columns[0]
        
        if theme == 'dark':
            theme_colors = DARK_THEME
        else:
            theme_colors = LIGHT_THEME
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df[feature],
            name=feature,
            marker_color=theme_colors['primary'],
            boxmean='sd',
            hovertemplate='<b>%{y}</b><extra></extra>',
        ))
        
        fig.update_layout(
            **get_figure_layout(theme_colors, f"Boxplot: {feature}"),
            yaxis_title="값",
            showlegend=False,
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
    except Exception as e:
        return html.Div([
            html.P(f"그래프 생성 중 오류 발생: {str(e)}", style={'color': 'red', 'padding': '1rem'})
        ])

@app.callback(
    Output('violin-graph-container', 'children'),
    [Input('violin-feature', 'value'),
     Input('theme-store', 'data')]
)
def update_violin(feature, theme):
    try:
        if theme is None:
            theme = 'dark'
        
        if feature is None or feature not in df_scaled.columns:
            feature = feature_cols[0] if len(feature_cols) > 0 else df_scaled.columns[0]
        
        if theme == 'dark':
            theme_colors = DARK_THEME
        else:
            theme_colors = LIGHT_THEME
        
        fig = go.Figure()
        
        for diagnosis_val, label, color in [(0, '양성(B)', theme_colors['success']), (1, '악성(M)', theme_colors['danger'])]:
            mask = df_scaled['diagnosis'] == diagnosis_val
            if mask.sum() > 0:
                fig.add_trace(go.Violin(
                    y=df_scaled.loc[mask, feature],
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=color,
                    line_color=color,
                    opacity=0.6,
                    hovertemplate=f'<b>{label}</b><br>값: %{{y}}<extra></extra>',
                ))
        
        fig.update_layout(
            **get_figure_layout(theme_colors, f"바이올린 차트: {feature}"),
            yaxis_title="값",
        )
        return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
    except Exception as e:
        return html.Div([
            html.P(f"그래프 생성 중 오류 발생: {str(e)}", style={'color': 'red', 'padding': '1rem'})
        ])

@app.callback(
    Output('pdp-graph-container', 'children'),
    [Input('pdp-feature', 'value'),
     Input('theme-store', 'data')]
)
def update_pdp(feature, theme):
    try:
        if theme is None:
            theme = 'dark'
        
        if feature is None or feature not in X_test.columns:
            feature = feature_cols[0] if len(feature_cols) > 0 else X_test.columns[0]
        
        if theme == 'dark':
            theme_colors = DARK_THEME
        else:
            theme_colors = LIGHT_THEME
        
        # Partial Dependence 계산
        feature_min = X_test[feature].min()
        feature_max = X_test[feature].max()
        
        if feature_min == feature_max:
            feature_min -= 1
            feature_max += 1
        
        feature_values = np.linspace(feature_min, feature_max, 50)
        
        pdp_values = []
        for fv in feature_values:
            try:
                X_temp = X_test.copy()
                X_temp[feature] = fv
                proba = rf_clf.predict_proba(X_temp)[:, 1].mean()
                pdp_values.append(proba)
            except:
                pdp_values.append(0.5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=pdp_values,
            mode='lines',
            name='PDP',
            line=dict(color=theme_colors['primary'], width=3),
            fill='tonexty',
            fillcolor=theme_colors['primary'],
            opacity=0.2,
            hovertemplate='<b>%{x:.2f}</b><br>악성 확률: %{y:.3f}<extra></extra>',
        ))
        
        fig.update_layout(
            **get_figure_layout(theme_colors, f"Partial Dependence Plot: {feature}"),
            xaxis_title=feature,
            yaxis_title="악성 확률",
            showlegend=False,
        )
        return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
    except Exception as e:
        return html.Div([
            html.P(f"그래프 생성 중 오류 발생: {str(e)}", style={'color': 'red', 'padding': '1rem'})
        ])

# ============================================
# 앱 실행
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("유방암 데이터 분석 대시보드 시작")
    print("="*60)
    print("브라우저에서 http://127.0.0.1:8050 접속하세요")
    print("="*60 + "\n")
    app.run_server(debug=True, host='127.0.0.1', port=8050)

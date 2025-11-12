import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA

# 데이터 로드
DATA_PATH = "breast_cancer_wisconsin_diagnostic.csv"
df = pd.read_csv(DATA_PATH)
df["Diagnosis"] = df["Diagnosis"].map({"M": "Malignant", "B": "Benign"})

# Plotly 스타일 설정 (최신 버전 호환)
pio.templates["my_dark"] = pio.templates["plotly_dark"]
pio.templates["my_dark"].layout.font.family = "Malgun Gothic, AppleGothic, NanumGothic, sans-serif"
pio.templates["my_dark"].layout.font.color = "#E8ECEF"
pio.templates["my_dark"].layout.colorway = ["#5DA9E9", "#EF476F", "#FFD166", "#06D6A0", "#8D99AE"]

pio.templates.default = "my_dark"
px.defaults.template = "my_dark"
# 컬러 팔레트는 각 그래프에 color_discrete_sequence로 개별 지정

# -------------------------------
# ⚙️ Dash 앱 초기화
# -------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Breast Cancer Diagnostic Dashboard"
server = app.server

# -------------------------------
# 📊 기본 변수 계산
# -------------------------------
size_features = ["radius1", "perimeter1", "area1"]
texture_features = ["texture1", "smoothness1", "compactness1"]

total_count = len(df)
malignant_count = int((df["Diagnosis"] == "Malignant").sum())
benign_count = total_count - malignant_count
malignant_ratio = malignant_count / total_count * 100

card_style = {
    "background": "linear-gradient(135deg, #1f2937 0%, #111827 100%)",
    "border": "1px solid rgba(255, 255, 255, 0.05)",
    "borderRadius": "18px",
    "boxShadow": "0 20px 40px rgba(15, 23, 42, 0.45)",
    "padding": "1.6rem",
    "height": "100%",
}

section_style = {
    "backgroundColor": "#0b0f1a",
    "borderRadius": "22px",
    "padding": "1.4rem",
    "border": "1px solid rgba(255, 255, 255, 0.04)",
    "boxShadow": "0 35px 60px rgba(2, 6, 23, 0.6)",
}

graph_config = {"displayModeBar": False}

# -------------------------------
# 🧱 Dash 레이아웃
# -------------------------------
app.layout = dbc.Container(
    [
        html.Div(
            [
                html.H1(
                    "Breast Cancer Diagnostic Dashboard",
                    className="fw-bold",
                    style={"color": "#F8FAFC", "letterSpacing": "0.08em", "marginBottom": "0.5rem"},
                ),
                html.P(
                    "End-to-end analytics cockpit for diagnostic insights and feature intelligence.",
                    style={"color": "#94A3B8", "fontSize": "1.05rem"},
                ),
            ],
            className="text-center mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            html.Small("전체 샘플", className="text-uppercase text-muted"),
                            html.H2(f"{total_count:,}", className="text-light fw-bolder"),
                            html.Div("patients", className="text-secondary"),
                        ],
                        style=card_style,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            html.Small("악성 (Malignant)", className="text-uppercase text-muted"),
                            html.H2(f"{malignant_count:,}", className="text-danger fw-bolder"),
                            html.Div(f"{malignant_ratio:.1f}% of total", className="text-secondary"),
                        ],
                        style=card_style,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            html.Small("양성 (Benign)", className="text-uppercase text-muted"),
                            html.H2(f"{benign_count:,}", className="text-info fw-bolder"),
                            html.Div(f"{100 - malignant_ratio:.1f}% of total", className="text-secondary"),
                        ],
                        style=card_style,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            html.Small("대시보드 필터", className="text-uppercase text-muted"),
                            html.Div(
                                [
                                    html.Label("특성 선택", className="text-secondary mb-1", style={"fontSize": "0.85rem"}),
                                    dcc.Dropdown(
                                        id="feature-dropdown",
                                        options=[
                                            {"label": col, "value": col}
                                            for col in sorted(df.columns)
                                            if col != "Diagnosis"
                                        ],
                                        value="radius1",
                                        clearable=False,
                                        className="text-dark",
                                        style={"backgroundColor": "rgba(15,23,42,0.8)"},
                                    ),
                                    html.Label("PCA 색상 기준", className="text-secondary mt-3 mb-1", style={"fontSize": "0.85rem"}),
                                    dcc.RadioItems(
                                        id="pca-color",
                                        options=[
                                            {"label": "Diagnosis", "value": "Diagnosis"},
                                            {"label": "Half Split (radius1 median)", "value": "radius_split"},
                                        ],
                                        value="Diagnosis",
                                        inline=False,
                                        className="text-light",
                                    ),
                                ]
                            ),
                        ],
                        style=card_style,
                    ),
                    md=3,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Diagnosis Composition", className="text-light mb-3"),
                                dcc.Graph(id="diagnosis-pie", config=graph_config, style={"height": "320px"}),
                            ],
                        ),
                        style=section_style,
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Feature Distribution", className="text-light mb-3"),
                                dcc.Graph(id="feature-distribution", config=graph_config, style={"height": "320px"}),
                            ],
                        ),
                        style=section_style,
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Feature Correlation Cluster", className="text-light mb-3"),
                                dcc.Graph(id="correlation-heatmap", config=graph_config, style={"height": "320px"}),
                            ],
                        ),
                        style=section_style,
                    ),
                    md=4,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("PCA Projection Intelligence", className="text-light mb-3"),
                                dcc.Graph(id="pca-scatter", config=graph_config, style={"height": "420px"}),
                            ],
                        ),
                        style=section_style,
                    ),
                    md=12,
                )
            ],
            className="g-4",
        ),
    ],
    fluid=True,
    style={
        "background": "radial-gradient(circle at top, #1f2937 0%, #020617 55%)",
        "minHeight": "100vh",
        "padding": "2.5rem 2rem",
    },
)

# -------------------------------
# 🔧 Helper: 기본 Figure Layout
# -------------------------------
def _base_figure(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(148, 163, 184, 0.3)",
            borderwidth=1,
        ),
    )
    return fig

# -------------------------------
# 📈 콜백 정의
# -------------------------------

@app.callback(
    Output("diagnosis-pie", "figure"),
    Input("feature-dropdown", "value"),
)
def update_pie(_):
    # Diagnosis별 개수 집계
    value_counts = df["Diagnosis"].value_counts().reset_index(name="count")
    # 컬럼: ['Diagnosis', 'count']

    fig = px.pie(
        value_counts,
        names="Diagnosis",   # 분류 이름
        values="count",      # 개수
        hole=0.45,
        title="Diagnosis Split",
        color_discrete_sequence=["#EF476F", "#118AB2"],
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0.05, 0])
    return _base_figure(fig)

@app.callback(
    Output("feature-distribution", "figure"),
    Input("feature-dropdown", "value"),
)
def update_distribution(feature):
    fig = px.violin(
        df,
        x="Diagnosis",
        y=feature,
        color="Diagnosis",
        box=True,
        points="all",
        hover_data=df.columns,
        color_discrete_map={"Malignant": "#EF476F", "Benign": "#06D6A0"},
    )
    fig.update_traces(meanline=dict(visible=True, color="#F1F5F9"))
    fig.update_layout(title=f"{feature} Distribution by Diagnosis")
    return _base_figure(fig)

@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("feature-dropdown", "value"),
)
def update_heatmap(_):
    corr = df[size_features + texture_features].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        labels=dict(color="Correlation"),
        color_continuous_scale=["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd"],
    )
    fig.update_layout(title="Structural Feature Correlations")
    return _base_figure(fig)

@app.callback(
    Output("pca-scatter", "figure"),
    Input("pca-color", "value"),
)
def update_pca(color_col):
    numeric_df = df.drop(columns=["Diagnosis"])
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(numeric_df)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Diagnosis"] = df["Diagnosis"]
    pca_df["radius_split"] = pd.cut(
        df["radius1"],
        bins=[numeric_df["radius1"].min(), numeric_df["radius1"].median(), numeric_df["radius1"].max()],
        labels=["Low", "High"],
    )

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=pca_df[color_col],
        title="PCA 2D Projection",
        color_discrete_sequence=["#EF476F", "#06D6A0"],
    )
    fig.update_traces(marker=dict(size=11, opacity=0.86, line=dict(width=1.4, color="#0f172a")))
    return _base_figure(fig)

# 🚀 서버 실행 
if __name__ == "__main__":
    app.run(debug=True)

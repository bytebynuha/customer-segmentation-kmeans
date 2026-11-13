
import os
import re
from pathlib import Path
from io import BytesIO
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CLUSTERO | Customer Segmentation",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = [
    "#FF6FAE",
    "#A78BFA",
    "#43D9B8",
    "#FFD447",
    "#FF9878",
    "#62B8FF",
    "#B6DF69",
    "#F4A6FF"
]

FEATURES = ["Age", "Annual Income", "Spending Score"]

PLOT_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "clustero-chart",
        "scale": 2
    }
}


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

:root {
    --ink: #191923;
    --cream: #FFF9F1;
    --pink: #FFB4D2;
    --purple: #BDA9FF;
    --yellow: #FFD447;
    --mint: #9BE8D3;
    --coral: #FF9878;
    --border: #191923;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

.stApp {
    background: var(--cream) !important;
    color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 85% 5%, #F9DDF0 0, transparent 28%),
        radial-gradient(circle at 10% 30%, #FFF0C7 0, transparent 25%),
        var(--cream) !important;
}

[data-testid="stHeader"] {
    background: rgba(255,249,241,.85) !important;
}

.main .block-container {
    max-width: 1600px;
    padding: 1.5rem 2.1rem 3rem;
}

h1, h2, h3, h4 {
    color: var(--ink) !important;
    font-weight: 900 !important;
    letter-spacing: -1px;
}

p, label, li, span {
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        160deg,
        #FFD966 0%,
        #FFB8D0 48%,
        #C6B4FF 100%
    ) !important;
    border-right: 3px solid var(--ink);
}

[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-weight: 900 !important;
}

.brand {
    font-size: 32px;
    font-weight: 900;
    letter-spacing: -2px;
}

.brand-star {
    color: #FF4F91;
}

.brand-tag {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 5px;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0 20px;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 28px;
}

.topbar-tag {
    background: white;
    border: 2px solid var(--ink);
    border-radius: 8px;
    padding: 12px 18px;
    box-shadow: 4px 4px 0 var(--ink);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
}

.hero {
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: 1.3fr .7fr;
    gap: 20px;
    padding: 42px;
    margin-bottom: 35px;
    border: 3px solid var(--ink);
    border-radius: 18px;
    background: linear-gradient(
        115deg,
        #FFD95B 0%,
        #FFB1C9 53%,
        #BDA9FF 100%
    );
    box-shadow: 9px 9px 0 var(--ink);
    animation: appear .65s ease-out;
}

.hero-kicker {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 24px;
}

.hero-title {
    font-size: clamp(42px, 5vw, 76px);
    line-height: .99;
    font-weight: 900;
    letter-spacing: -4px;
}

.hero-highlight {
    color: #FFFFFF;
    text-shadow: 3px 3px 0 var(--ink);
}

.hero-subtitle {
    font-size: 21px;
    font-weight: 800;
    margin-top: 22px;
}

.hero-description {
    max-width: 550px;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.8;
    margin-top: 12px;
}

.hero-art {
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    min-height: 250px;
}

.hero-coin {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 205px;
    height: 205px;
    border: 4px solid var(--ink);
    border-radius: 50%;
    background: #FF75B9;
    color: var(--ink);
    font-size: 130px;
    font-weight: 900;
    transform: rotate(-13deg);
    box-shadow: 12px 13px 0 var(--ink);
    animation: float 4s ease-in-out infinite;
}

.hero-star {
    position: absolute;
    font-size: 65px;
    font-weight: 900;
}

.star-one {
    right: 5%;
    top: 2%;
}

.star-two {
    left: 2%;
    bottom: 0;
}

.section-kicker {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #7653DD;
    margin-bottom: 9px;
}

.section-heading {
    font-size: 27px;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 17px;
}

.panel {
    padding: 22px;
    margin-bottom: 18px;
    background: rgba(255,255,255,.76);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 2px solid var(--ink);
    border-radius: 13px;
    box-shadow: 5px 5px 0 var(--ink);
    transition: transform .2s ease, box-shadow .2s ease;
}

.panel:hover {
    transform: translateY(-3px);
    box-shadow: 8px 8px 0 var(--ink);
}

.panel-title {
    font-size: 19px;
    font-weight: 900;
}

.panel-caption {
    color: #49434F;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.6;
    margin-top: 6px;
}

.metric-card {
    min-height: 135px;
    padding: 20px;
    border: 2px solid var(--ink);
    border-radius: 12px;
    box-shadow: 5px 5px 0 var(--ink);
    transition: transform .2s ease;
}

.metric-card:hover {
    transform: translateY(-5px) rotate(-1deg);
}

.metric-card.pink { background: #FFB4D2; }
.metric-card.purple { background: #BDA9FF; }
.metric-card.yellow { background: #FFD447; }
.metric-card.mint { background: #9BE8D3; }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-number {
    font-size: 37px;
    font-weight: 900;
    letter-spacing: -2px;
    margin-top: 13px;
}

.metric-note {
    font-size: 12px;
    font-weight: 600;
}

.segment-card {
    background: rgba(255,255,255,.8);
    border: 2px solid var(--ink);
    border-radius: 12px;
    padding: 19px;
    margin-bottom: 13px;
    box-shadow: 4px 4px 0 var(--ink);
    transition: transform .2s ease;
}

.segment-card:hover {
    transform: translateX(5px);
}

.segment-number {
    font-family: 'Space Mono', monospace;
    color: #7954E8;
    font-size: 11px;
    font-weight: 700;
}

.segment-title {
    font-size: 19px;
    font-weight: 900;
    margin: 9px 0;
}

.segment-detail {
    font-size: 13px;
    font-weight: 600;
    line-height: 1.8;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 9px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border: 2px solid var(--ink);
    border-radius: 8px;
    padding: 12px 18px;
    font-weight: 800;
    font-size: 14px;
    box-shadow: 3px 3px 0 var(--ink);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #FFD447 !important;
    color: var(--ink) !important;
    border-bottom: 2px solid var(--ink) !important;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 48px;
    background: #FF78B9 !important;
    color: #191923 !important;
    border: 2px solid #191923 !important;
    border-radius: 8px !important;
    box-shadow: 4px 4px 0 #191923 !important;
    font-size: 14px !important;
    font-weight: 900 !important;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #FFD447 !important;
    color: #191923 !important;
    transform: translate(-2px,-2px);
    box-shadow: 6px 6px 0 #191923 !important;
}

.stSelectbox label,
.stSlider label,
.stNumberInput label,
.stRadio label,
.stMultiSelect label {
    font-weight: 800 !important;
    font-size: 14px !important;
}

.stSelectbox [data-baseweb="select"] > div {
    min-height: 48px;
    background: white !important;
    border: 2px solid var(--ink) !important;
    border-radius: 8px !important;
}

.stNumberInput input {
    min-height: 45px;
    background: white !important;
    color: var(--ink) !important;
    border: 2px solid var(--ink) !important;
    border-radius: 8px !important;
}

.stRadio [role="radiogroup"] {
    gap: 14px;
}


/* ============================================================
   CSV UPLOADER — CLUSTERO NEO-BRUTALIST FIX
   ============================================================ */

/* Main uploader container */
[data-testid="stFileUploader"] {
    width: 100% !important;
    background: rgba(255,255,255,.65) !important;
    border: 2px dashed var(--ink) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    box-sizing: border-box !important;
}

/* Upload dropzone */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    border-radius: 9px !important;
}

/* Upload instructions */
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section p {
    color: #191923 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Browse files button */
[data-testid="stFileUploader"] section button {
    background: #FFB4D2 !important;
    color: #191923 !important;
    border: 2px solid #191923 !important;
    border-radius: 7px !important;
    font-weight: 800 !important;
    box-shadow: 2px 2px 0 #191923 !important;
    transition: all .2s ease !important;
}

[data-testid="stFileUploader"] section button:hover {
    background: #FFD447 !important;
    color: #191923 !important;
    transform: translateY(-2px) !important;
    box-shadow: 4px 4px 0 #191923 !important;
}

/* Uploaded file preview */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div {
    background: #FFFFFF !important;
    color: #191923 !important;
    border: 1.5px solid #191923 !important;
    border-radius: 9px !important;
    box-shadow: 2px 2px 0 #191923 !important;
    opacity: 1 !important;
}

/* Uploaded filename and file size */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small {
    color: #191923 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* File preview icon */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg {
    color: #191923 !important;
}

/* Remove uploaded file button */
[data-testid="stFileUploader"] button[aria-label*="Remove"],
[data-testid="stFileUploader"] button[title*="Remove"] {
    background: #FFB4D2 !important;
    color: #191923 !important;
    border: 1px solid #191923 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

/* Keep uploader responsive */
[data-testid="stFileUploader"] > div {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}


[data-testid="stDataFrame"] {
    border: 2px solid var(--ink);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stMetric"] {
    background: white;
    border: 2px solid var(--ink);
    border-radius: 10px;
    padding: 15px;
    box-shadow: 3px 3px 0 var(--ink);
}

[data-testid="stMetricLabel"] {
    color: #51465D !important;
    font-weight: 800 !important;
}

[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-weight: 900 !important;
}

div[data-testid="stCaptionContainer"] p {
    color: #51465D !important;
    font-weight: 600 !important;
}

.footer {
    text-align: center;
    padding: 28px;
    margin-top: 35px;
    background: linear-gradient(100deg,#FFD447,#FFB4D2,#BDA9FF);
    border: 2px solid var(--ink);
    border-radius: 12px;
    box-shadow: 5px 5px 0 var(--ink);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    line-height: 2;
}

@keyframes appear {
    from {opacity:0; transform:translateY(15px);}
    to {opacity:1; transform:translateY(0);}
}

@keyframes float {
    0%,100% {transform:translateY(0) rotate(-13deg);}
    50% {transform:translateY(-12px) rotate(-8deg);}
}

@media(max-width:850px) {
    .hero {
        grid-template-columns:1fr;
        padding:25px;
    }

    .hero-title {
        font-size:43px;
        letter-spacing:-2px;
    }

    .hero-art {
        min-height:210px;
    }

    .main .block-container {
        padding:1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HTML RENDERING HELPER
# ============================================================

def render_html(content):
    """
    Render HTML directly.
    Never send HTML fragments through st.code()
    or a Markdown code block.
    """
    st.html(str(content))


# ============================================================
# 4. CSV UTILITIES
# ============================================================

def normalize_column(column):
    value = str(column).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[()]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def locate(columns, possible_names):
    normalized = {
        normalize_column(column): column
        for column in columns
    }

    for name in possible_names:
        key = normalize_column(name)
        if key in normalized:
            return normalized[key]

    return None


def is_summary_csv(columns):
    names = {normalize_column(c) for c in columns}

    summary_signals = {
        "cluster",
        "customers",
        "average age",
        "average income",
        "average spending",
        "share",
        "segment"
    }

    return (
        "cluster" in names
        and (
            "customers" in names
            or "average age" in names
            or "average income" in names
        )
    )


@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:
        for separator in [None, ",", ";", "\t"]:
            try:
                options = {
                    "encoding": encoding,
                    "on_bad_lines": "error"
                }

                if separator is None:
                    options["sep"] = None
                    options["engine"] = "python"
                else:
                    options["sep"] = separator

                data = pd.read_csv(
                    BytesIO(file_bytes),
                    **options
                )

                data.columns = [
                    str(c).strip().replace("\ufeff", "")
                    for c in data.columns
                ]

                if data.empty:
                    continue

                if is_summary_csv(data.columns):
                    raise ValueError(
                        "This looks like a cluster summary CSV, "
                        "not a customer-level dataset. "
                        "Upload the original customer data CSV."
                    )

                age = locate(
                    data.columns,
                    ["Age", "Customer Age"]
                )

                income = locate(
                    data.columns,
                    [
                        "Annual Income (k$)",
                        "Annual Income",
                        "Annual_Income",
                        "Income"
                    ]
                )

                spending = locate(
                    data.columns,
                    [
                        "Spending Score (1-100)",
                        "Spending Score",
                        "Spending_Score",
                        "Spending"
                    ]
                )

                if all([age, income, spending]):
                    return data

            except ValueError:
                raise

            except Exception as error:
                last_error = error

    raise ValueError(
        "Unable to identify the customer data columns. "
        "Required: Age, Annual Income and Spending Score. "
        f"Last parsing error: {last_error}"
    )


# ============================================================
# 5. PLOTLY CHART STYLING
# ============================================================

def style_chart(fig, title, height=620):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=22, color="#191923"),
            x=0.03
        ),
        paper_bgcolor="rgba(255,255,255,0.75)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(
            family="Arial",
            color="#191923",
            size=13
        ),
        height=height,
        margin=dict(l=65, r=40, t=85, b=70),
        legend=dict(
            title="Customer Segments",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#191923",
            borderwidth=1,
            font=dict(size=13, color="#191923")
        ),
        hoverlabel=dict(
            bgcolor="#191923",
            bordercolor="#FFD447",
            font=dict(color="white", size=13)
        ),
        hovermode="closest"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E7DFE5",
        color="#191923",
        title_font=dict(size=14, color="#191923"),
        tickfont=dict(size=12, color="#191923")
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E7DFE5",
        color="#191923",
        title_font=dict(size=14, color="#191923"),
        tickfont=dict(size=12, color="#191923")
    )

    return fig


def cluster_color_map(cluster_ids):
    return {
        str(cluster): PALETTE[int(cluster) % len(PALETTE)]
        for cluster in sorted(cluster_ids)
    }


def show_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOT_CONFIG,
        theme=None
    )


# ============================================================
# 6. CHART FUNCTIONS
# ============================================================

def make_scatter(results, x, y, centers=None, title="Customer Segmentation"):
    data = results.copy()
    data["Cluster"] = data["Cluster"].astype(str)

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color="Cluster",
        color_discrete_map=cluster_color_map(
            results["Cluster"].unique()
        ),
        hover_data=[
            "CustomerID",
            "Age",
            "Annual Income",
            "Spending Score",
            "Cluster"
        ],
        opacity=0.88,
        render_mode="webgl"
    )

    fig.update_traces(
        marker=dict(
            size=13,
            line=dict(color="#191923", width=0.8)
        )
    )

    if centers is not None:
        indices = {
            "Age": 0,
            "Annual Income": 1,
            "Spending Score": 2
        }

        fig.add_trace(go.Scatter(
            x=centers[:, indices[x]],
            y=centers[:, indices[y]],
            mode="markers",
            name="Cluster Centres",
            marker=dict(
                symbol="x",
                size=22,
                color="#191923",
                line=dict(color="#FFD447", width=4)
            )
        ))

    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title="Segments",
        dragmode="pan"
    )

    return style_chart(fig, title, 680)


def make_3d(results):
    data = results.copy()
    data["Cluster"] = data["Cluster"].astype(str)

    fig = px.scatter_3d(
        data,
        x="Age",
        y="Annual Income",
        z="Spending Score",
        color="Cluster",
        color_discrete_map=cluster_color_map(
            results["Cluster"].unique()
        ),
        hover_data=["CustomerID"],
        opacity=0.9
    )

    fig.update_traces(
        marker=dict(
            size=6,
            line=dict(color="#191923", width=0.5)
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,.8)",
        font=dict(color="#191923", size=13),
        height=760,
        margin=dict(l=15, r=15, t=75, b=15),
        scene=dict(
            bgcolor="rgba(255,255,255,.4)",
            xaxis=dict(title="Age", color="#191923"),
            yaxis=dict(title="Annual Income", color="#191923"),
            zaxis=dict(title="Spending Score", color="#191923")
        )
    )

    return fig


def make_bubble(results):
    fig = px.scatter(
        results,
        x="Annual Income",
        y="Spending Score",
        size="Age",
        color=results["Cluster"].astype(str),
        color_discrete_map=cluster_color_map(
            results["Cluster"].unique()
        ),
        hover_data=["CustomerID", "Age"],
        size_max=32,
        opacity=0.85
    )

    return style_chart(
        fig,
        "Bubble Chart — Income, Spending and Age",
        680
    )


def make_density(results):
    fig = px.density_heatmap(
        results,
        x="Annual Income",
        y="Spending Score",
        nbinsx=25,
        nbinsy=25,
        color_continuous_scale=[
            "#FFF9F1",
            "#BDA9FF",
            "#FF78B9",
            "#FFD447"
        ],
        hover_data=["count"]
    )

    return style_chart(
        fig,
        "Customer Density — Income vs Spending",
        680
    )


def make_box(results):
    fig = px.box(
        results,
        x=results["Cluster"].astype(str),
        y="Spending Score",
        color=results["Cluster"].astype(str),
        color_discrete_map=cluster_color_map(
            results["Cluster"].unique()
        ),
        points="all",
        hover_data=["CustomerID", "Age", "Annual Income"]
    )

    fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="Spending Score"
    )

    return style_chart(
        fig,
        "Spending Score Distribution by Segment",
        680
    )


def make_elbow(ks, inertias, selected_k):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ks,
        y=inertias,
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#7954E8", width=4),
        marker=dict(
            size=13,
            color=PALETTE[:len(ks)],
            line=dict(color="#191923", width=1)
        ),
        hovertemplate="K: %{x}<br>Inertia: %{y:.3f}<extra></extra>"
    ))

    selected_value = inertias[ks.index(selected_k)]

    fig.add_trace(go.Scatter(
        x=[selected_k],
        y=[selected_value],
        mode="markers",
        name="Selected K",
        marker=dict(
            size=23,
            color="#FF6FAE",
            symbol="star",
            line=dict(color="#191923", width=2)
        )
    ))

    fig.add_vline(
        x=selected_k,
        line_dash="dash",
        line_color="#191923"
    )

    fig.update_layout(
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia / WCSS"
    )

    return style_chart(
        fig,
        "Elbow Method — Optimal Cluster Selection",
        570
    )


def make_distribution(results):
    counts = results["Cluster"].value_counts().sort_index()

    fig = go.Figure(go.Bar(
        x=[f"Cluster {i}" for i in counts.index],
        y=counts.values,
        marker=dict(
            color=[
                PALETTE[int(i) % len(PALETTE)]
                for i in counts.index
            ],
            line=dict(color="#191923", width=2)
        ),
        text=counts.values,
        textposition="outside",
        textfont=dict(color="#191923", size=15),
        customdata=counts.values / len(results) * 100,
        hovertemplate=(
            "%{x}<br>Customers: %{y}"
            "<br>Share: %{customdata:.1f}%<extra></extra>"
        )
    ))

    fig.update_layout(
        xaxis_title="Customer Segments",
        yaxis_title="Number of Customers"
    )

    return style_chart(
        fig,
        "Customer Distribution by Segment",
        510
    )


def make_donut(results):
    counts = results["Cluster"].value_counts().sort_index()

    fig = go.Figure(go.Pie(
        labels=[f"Cluster {i}" for i in counts.index],
        values=counts.values,
        hole=0.62,
        marker=dict(
            colors=[
                PALETTE[int(i) % len(PALETTE)]
                for i in counts.index
            ],
            line=dict(color="#191923", width=2)
        ),
        textinfo="label+percent",
        textfont=dict(color="#191923", size=13),
        hovertemplate="%{label}<br>%{value} customers<extra></extra>"
    ))

    return style_chart(
        fig,
        "Customer Segment Share",
        510
    )


def make_spending_chart(summary):
    fig = go.Figure(go.Bar(
        x=[f"Cluster {i}" for i in summary.index],
        y=summary["Average_Spending"],
        marker=dict(
            color=[
                PALETTE[int(i) % len(PALETTE)]
                for i in summary.index
            ],
            line=dict(color="#191923", width=2)
        ),
        text=summary["Average_Spending"].round(1),
        textposition="outside",
        textfont=dict(color="#191923", size=14),
        hovertemplate="%{x}<br>Average score: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="Average Spending Score"
    )

    return style_chart(
        fig,
        "Average Spending Score by Segment",
        510
    )


# ============================================================
# 7. BRAND HEADER AND HERO
# ============================================================

render_html("""
<div class="topbar">
    <div>
        <div class="brand">CLUSTER<span class="brand-star">✳</span>O</div>
        <div class="brand-tag">CUSTOMER INTELLIGENCE STUDIO</div>
    </div>
    <div class="topbar-tag">LIVE CUSTOMER ANALYTICS · K-MEANS</div>
</div>
""")

render_html("""
<div class="hero">
    <div>
        <div class="hero-kicker">✳ CUSTOMER INTELLIGENCE / DATA DISCOVERY</div>
        <div class="hero-title">
            KNOW YOUR<br>
            CUSTOMERS.<br>
            <span class="hero-highlight">FIND THE PATTERNS.</span>
        </div>
        <div class="hero-subtitle">
            Customer Segmentation, powered by K-Means.
        </div>
        <div class="hero-description">
            Turn customer data into meaningful insights.
            Discover purchasing behaviour, identify hidden
            patterns and understand what makes every
            customer segment unique.
        </div>
    </div>
    <div class="hero-art">
        <div class="hero-star star-one">✳</div>
        <div class="hero-coin">C</div>
        <div class="hero-star star-two">✳</div>
    </div>
</div>
""")


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:
    render_html("""
    <div style="font-size:28px;font-weight:900;letter-spacing:-1px;">
        CLUSTERO ✳
    </div>
    <div style="font-family:monospace;font-size:10px;
                font-weight:700;letter-spacing:2px;">
        CUSTOMER INTELLIGENCE
    </div>
    """)

    st.markdown("---")
    st.markdown("### 01 / DATASET")

    uploaded = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Required: Age, Annual Income and Spending Score."
    )

    st.markdown("---")
    st.markdown("### 02 / TECHNOLOGY")
    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - Plotly
    - Streamlit
    - Podman
    """)

    st.markdown("---")
    st.markdown("### 03 / MODEL")
    st.caption("Algorithm: K-Means")
    st.caption("Scaling: StandardScaler")
    st.caption("Validation: Elbow Method")

    st.markdown("---")
    st.caption("TURN DATA INTO DISCOVERY ✳")


# ============================================================
# 9. LANDING PAGE
# ============================================================

if uploaded is None:
    render_html("""
    <div class="section-kicker">GET STARTED / 01</div>
    <div class="section-heading">Your data has a story. Let's find it.</div>
    """)

    c1, c2, c3 = st.columns(3)

    cards = [
        (c1, "#FFB4D2", "STEP 01", "Upload Data",
         "Upload your customer-level CSV dataset.", "▤"),
        (c2, "#BDA9FF", "STEP 02", "Discover Groups",
         "Apply K-Means and uncover customer patterns.", "◉"),
        (c3, "#9BE8D3", "STEP 03", "Predict Segments",
         "Assign new customers to their predicted segments.", "✳")
    ]

    for col, color, step, title, description, symbol in cards:
        with col:
            render_html(f"""
            <div class="panel" style="background:{color};min-height:220px;">
                <div style="font-family:monospace;font-weight:900;">{step}</div>
                <div style="font-size:24px;font-weight:900;margin-top:15px;">
                    {title}
                </div>
                <div style="font-size:14px;font-weight:600;margin-top:10px;">
                    {description}
                </div>
                <div style="font-size:48px;font-weight:900;margin-top:15px;">
                    {symbol}
                </div>
            </div>
            """)

    render_html("""
    <div class="footer">
        CLUSTERO ✳<br>
        CUSTOMER SEGMENTATION · POWERED BY K-MEANS<br>
        UPLOAD YOUR DATASET TO BEGIN
    </div>
    """)
    st.stop()


# ============================================================
# 10. LOAD AND VALIDATE CUSTOMER DATA
# ============================================================

try:
    raw = load_csv(uploaded.getvalue())

except Exception as error:
    st.error(f"Unable to process your CSV: {error}")
    st.info(
        "Your customer-level CSV must contain Age, "
        "Annual Income and Spending Score columns."
    )
    st.stop()


age_col = locate(raw.columns, ["Age", "Customer Age"])

income_col = locate(
    raw.columns,
    [
        "Annual Income (k$)",
        "Annual Income",
        "Annual_Income",
        "Income"
    ]
)

score_col = locate(
    raw.columns,
    [
        "Spending Score (1-100)",
        "Spending Score",
        "Spending_Score",
        "Spending"
    ]
)

id_col = locate(
    raw.columns,
    ["CustomerID", "Customer ID", "ID"]
)

if not all([age_col, income_col, score_col]):
    st.error("Required customer columns are missing.")
    st.write("Detected columns:", list(raw.columns))
    st.stop()


# ============================================================
# 11. PREPROCESSING
# ============================================================

df = raw[[age_col, income_col, score_col]].copy()

df.columns = FEATURES

for column in FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

valid_mask = (
    df[FEATURES].notna().all(axis=1)
    & (df["Age"] > 0)
    & (df["Age"] <= 100)
    & (df["Annual Income"] >= 0)
    & (df["Spending Score"].between(1, 100))
)

df = df.loc[valid_mask].copy()

if len(df) < 3:
    st.error("At least three valid customer records are required.")
    st.stop()

if any(df[column].nunique() < 2 for column in FEATURES):
    st.error("Each feature must have at least two distinct values.")
    st.stop()

if id_col:
    customer_ids = raw.loc[df.index, id_col].to_numpy()
else:
    customer_ids = df.index.to_numpy() + 1

try:
    safe_name = Path(uploaded.name).name
    (DATA_DIR / safe_name).write_bytes(uploaded.getvalue())
except OSError:
    pass


# ============================================================
# 12. MODEL SETTINGS
# ============================================================

max_k = min(10, len(df) - 1)

render_html("""
<div class="section-kicker">MODEL CONFIGURATION / 02</div>
<div class="section-heading">Configure your clustering model.</div>
""")

left, right = st.columns([1, 2])

with left:
    k = st.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=max_k,
        value=min(5, max_k),
        key="cluster_k"
    )

with right:
    render_html("""
    <div class="panel">
        <div class="panel-title">K-Means Clustering</div>
        <div class="panel-caption">
            Customers are grouped based on similarities in age,
            annual income and spending score. StandardScaler
            standardises the features before model training.
        </div>
    </div>
    """)


# ============================================================
# 13. K-MEANS MODEL
# ============================================================

features = df[FEATURES].copy()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

ks = list(range(1, max_k + 1))
inertias = []

for number in ks:
    km = KMeans(
        n_clusters=number,
        random_state=42,
        n_init=10
    )
    km.fit(scaled_data)
    inertias.append(km.inertia_)

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(scaled_data)

results = df.copy()

results.insert(0, "CustomerID", customer_ids)
results["Cluster"] = labels

centers = scaler.inverse_transform(model.cluster_centers_)


# ============================================================
# 14. CLUSTER SUMMARY
# ============================================================

summary = results.groupby("Cluster").agg(
    Customers=("Cluster", "size"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income", "mean"),
    Average_Spending=("Spending Score", "mean")
).round(2)

income_median = results["Annual Income"].median()
spending_median = results["Spending Score"].median()


def describe_cluster(row):
    income = row["Average_Income"]
    spending = row["Average_Spending"]

    if income >= income_median and spending >= spending_median:
        return "High Income · High Spending"
    if income >= income_median and spending < spending_median:
        return "High Income · Lower Spending"
    if income < income_median and spending >= spending_median:
        return "Lower Income · High Spending"
    return "Lower Income · Lower Spending"


# ============================================================
# 15. KPI DASHBOARD
# ============================================================

render_html("""
<div class="section-kicker">LIVE ANALYTICS / 03</div>
<div class="section-heading">The numbers behind the story.</div>
""")

m1, m2, m3, m4 = st.columns(4)

metrics = [
    (m1, "CUSTOMERS", f"{len(df):,}", "pink"),
    (m2, "SEGMENTS", str(k), "yellow"),
    (m3, "FEATURES", "03", "purple"),
    (m4, "ALGORITHM", "K-MEANS", "mint")
]

for col, label, value, color in metrics:
    with col:
        render_html(f"""
        <div class="metric-card {color}">
            <div class="metric-label">{label}</div>
            <div class="metric-number">{value}</div>
        </div>
        """)


# ============================================================
# 16. NAVIGATION
# ============================================================

tab_overview, tab_data, tab_analytics, tab_prediction, tab_export = st.tabs([
    "✳ Overview",
    "▤ Data Explorer",
    "◉ Analytics",
    "✦ Prediction",
    "↓ Export Centre"
])


# ============================================================
# 17. OVERVIEW
# ============================================================

with tab_overview:
    render_html("""
    <div class="section-kicker">CUSTOMER INSIGHTS</div>
    <div class="section-heading">Meet your customer segments.</div>
    """)

    left, right = st.columns(2)

    with left:
        render_html("""
        <div class="panel">
            <div class="panel-title">Customer Distribution</div>
            <div class="panel-caption">
                Number of customers in each discovered segment.
            </div>
        </div>
        """)
        show_chart(make_distribution(results))

    with right:
        render_html("""
        <div class="panel">
            <div class="panel-title">Segment Share</div>
            <div class="panel-caption">
                Percentage contribution of every segment.
            </div>
        </div>
        """)
        show_chart(make_donut(results))

    render_html("""
    <div class="section-kicker">SEGMENT PROFILES</div>
    <div class="section-heading">Understand your customers.</div>
    """)

    for cluster_id, row in summary.iterrows():
        color = PALETTE[int(cluster_id) % len(PALETTE)]

        render_html(f"""
        <div class="segment-card">
            <div class="segment-number">CUSTOMER SEGMENT {cluster_id}</div>
            <div class="segment-title">{describe_cluster(row)}</div>
            <div class="segment-detail">
                Customers: {int(row['Customers'])}<br>
                Average age: {row['Average_Age']:.1f}<br>
                Average income: {row['Average_Income']:.1f}k<br>
                Average spending: {row['Average_Spending']:.1f}
            </div>
            <div style="height:7px;background:{color};
                        border:1px solid #191923;border-radius:5px;
                        margin-top:14px;"></div>
        </div>
        """)

    biggest = summary["Customers"].idxmax()
    top_spender = summary["Average_Spending"].idxmax()
    top_income = summary["Average_Income"].idxmax()

    a, b, c = st.columns(3)

    insights = [
        (a, "LARGEST SEGMENT",
         str(int(summary.loc[biggest, "Customers"])),
         f"Customers · Cluster {biggest}", "pink"),
        (b, "HIGHEST SPENDING",
         f"{summary.loc[top_spender, 'Average_Spending']:.1f}",
         f"Average score · Cluster {top_spender}", "mint"),
        (c, "HIGHEST INCOME",
         f"{summary.loc[top_income, 'Average_Income']:.1f}",
         f"Average income · Cluster {top_income}", "purple")
    ]

    for col, label, value, note, color in insights:
        with col:
            render_html(f"""
            <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-number">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """)


# ============================================================
# 18. DATA EXPLORER
# ============================================================

with tab_data:
    render_html("""
    <div class="section-kicker">DATA MANAGEMENT / 04</div>
    <div class="section-heading">Explore your customer dataset.</div>
    """)

    st.write("### Original uploaded dataset")

    st.dataframe(
        raw,
        use_container_width=True,
        height=420,
        hide_index=True
    )

    st.write("### Processed customer records")

    st.dataframe(
        results,
        use_container_width=True,
        height=420,
        hide_index=True
    )

    st.write("### Statistical summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    a, b, c = st.columns(3)

    a.metric("Original Records", len(raw))
    b.metric("Valid Records", len(df))
    c.metric("Removed Records", len(raw) - len(df))


# ============================================================
# 19. ANALYTICS
# ============================================================

with tab_analytics:
    render_html("""
    <div class="section-kicker">MACHINE LEARNING / 05</div>
    <div class="section-heading">Explore customer behaviour.</div>
    """)

    st.caption(
        "Interactive charts support hover, zoom, pan, selection "
        "and image export."
    )

    st.write("### 01 / Elbow Method")

    show_chart(make_elbow(ks, inertias, k))

    st.info(
        f"The current model uses K = {k}. "
        "Compare the inertia values to identify the elbow point."
    )

    st.divider()

    st.write("### 02 / Choose Your Visualisation")

    visualisation = st.radio(
        "Select your graph",
        [
            "2D Scatter",
            "3D Scatter",
            "Bubble Chart",
            "Density Plot",
            "Box Plot"
        ],
        horizontal=True,
        key="visualisation_choice"
    )

    st.write("### 03 / Feature Selection")

    x_col, y_col = st.columns(2)

    with x_col:
        x_axis = st.selectbox(
            "Horizontal axis",
            FEATURES,
            index=1,
            key="chart_x"
        )

    with y_col:
        y_axis = st.selectbox(
            "Vertical axis",
            FEATURES,
            index=2,
            key="chart_y"
        )

    if visualisation == "2D Scatter":
        if x_axis == y_axis:
            st.warning("Select two different features.")
        else:
            show_chart(
                make_scatter(
                    results,
                    x_axis,
                    y_axis,
                    centers,
                    f"{x_axis} vs {y_axis} — Customer Segmentation"
                )
            )

    elif visualisation == "3D Scatter":
        show_chart(make_3d(results))

    elif visualisation == "Bubble Chart":
        show_chart(make_bubble(results))

    elif visualisation == "Density Plot":
        show_chart(make_density(results))

    elif visualisation == "Box Plot":
        show_chart(make_box(results))

    st.divider()

    st.write("### 04 / Primary Customer Segmentation Map")

    st.caption(
        "Annual Income vs Spending Score. "
        "Each colour represents a K-Means cluster."
    )

    show_chart(
        make_scatter(
            results,
            "Annual Income",
            "Spending Score",
            centers,
            "Annual Income vs Spending Score — Customer Segments"
        )
    )

    st.divider()

    st.write("### 05 / Spending Comparison")

    show_chart(make_spending_chart(summary))

    st.write("### 06 / Segment Statistics")

    st.dataframe(summary, use_container_width=True)


# ============================================================
# 20. PREDICTION
# ============================================================

with tab_prediction:
    render_html("""
    <div class="section-kicker">PREDICTION STUDIO / 06</div>
    <div class="section-heading">Meet your next customer.</div>
    """)

    render_html("""
    <div class="panel">
        <div class="panel-title">New Customer Classification</div>
        <div class="panel-caption">
            Enter customer information. The trained K-Means model
            will assign the customer to the nearest cluster.
        </div>
    </div>
    """)

    a, b, c = st.columns(3)

    with a:
        new_age = st.number_input(
            "Customer Age",
            min_value=1,
            max_value=100,
            value=25
        )

    with b:
        new_income = st.number_input(
            "Annual Income (k$)",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

    with c:
        new_score = st.number_input(
            "Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=75
        )

    if st.button(
        "✳ PREDICT CUSTOMER SEGMENT",
        use_container_width=True
    ):
        new_customer = pd.DataFrame({
            "Age": [new_age],
            "Annual Income": [new_income],
            "Spending Score": [new_score]
        })

        new_scaled = scaler.transform(new_customer)
        predicted = int(model.predict(new_scaled)[0])
        profile = summary.loc[predicted]

        st.success(
            f"Customer assigned to Cluster {predicted}!"
        )

        color = PALETTE[predicted % len(PALETTE)]

        render_html(f"""
        <div class="panel" style="background:{color};">
            <div style="font-family:monospace;font-weight:900;">
                PREDICTED CUSTOMER SEGMENT
            </div>
            <div style="font-size:38px;font-weight:900;margin-top:12px;">
                CLUSTER {predicted}
            </div>
            <div style="font-size:20px;font-weight:800;margin-top:8px;">
                {describe_cluster(profile)}
            </div>
            <div style="font-size:14px;font-weight:700;margin-top:12px;">
                Existing customers in this segment:
                {int(profile['Customers'])}
            </div>
        </div>
        """)

        st.write("### Existing segment profile")

        p1, p2, p3 = st.columns(3)

        p1.metric("Average Age", f"{profile['Average_Age']:.1f}")
        p2.metric("Average Income", f"{profile['Average_Income']:.1f}k")
        p3.metric("Average Spending", f"{profile['Average_Spending']:.1f}")


# ============================================================
# 21. EXPORT CENTRE
# ============================================================

with tab_export:
    render_html("""
    <div class="section-kicker">REPORT CENTRE / 07</div>
    <div class="section-heading">Your analysis. Ready to go.</div>
    """)

    render_html("""
    <div class="panel">
        <div class="panel-title">Export Customer Insights</div>
        <div class="panel-caption">
            Download your customer-level clustering results
            and segment-level summary.
        </div>
    </div>
    """)

    st.write("### Clustered customer records")

    st.dataframe(
        results,
        use_container_width=True,
        height=400
    )

    csv_data = results.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "↓ DOWNLOAD CUSTOMER SEGMENTS CSV",
        data=csv_data,
        file_name="customer_segments.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.write("### Customer segment summary")

    st.dataframe(summary, use_container_width=True)

    summary_csv = summary.to_csv().encode("utf-8-sig")

    st.download_button(
        "↓ DOWNLOAD CLUSTER SUMMARY CSV",
        data=summary_csv,
        file_name="cluster_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

    try:
        (DATA_DIR / "customer_segments.csv").write_bytes(csv_data)
        (DATA_DIR / "cluster_summary.csv").write_bytes(summary_csv)

        st.success("Results saved in the mounted data directory.")

    except OSError as error:
        st.warning(f"Could not save results: {error}")


# ============================================================
# 22. FOOTER
# ============================================================

render_html("""
<div class="footer">
    <div style="font-size:27px;font-weight:900;">
        CLUSTERO ✳
    </div>
    CUSTOMER SEGMENTATION · POWERED BY K-MEANS<br>
    PYTHON · PANDAS · SCIKIT-LEARN · PLOTLY · STREAMLIT · PODMAN<br>
    LINUX AND CONTAINER TECHNOLOGY CAPSTONE PROJECT<br><br>
    TURN DATA INTO DISCOVERY.
</div>
""")

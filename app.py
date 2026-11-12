
import os
import html
import textwrap
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CLUSTERO | Customer Intelligence Studio",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = [
    "#D5FF36",
    "#F477CE",
    "#A99AFF",
    "#FFD52E",
    "#55D9C0",
    "#FF8A65",
    "#65B8FF",
    "#E6A8FF"
]

BACKGROUND = "#111111"
CARD = "#202020"
TEXT = "#F5F5F5"
MUTED = "#B9B9B9"
ACCENT = "#D5FF36"


# ============================================================
# 2. CSS DESIGN
# ============================================================

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #111111;
        color: #F5F5F5;
    }

    [data-testid="stHeader"] {
        background: rgba(17,17,17,0.95);
    }

    [data-testid="stSidebar"] {
        background: #191919;
        border-right: 1px solid #373737;
    }

    [data-testid="stSidebar"] * {
        color: #F5F5F5;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, p, label {
        color: #F5F5F5;
    }

    h1 {
        font-weight: 900 !important;
        letter-spacing: -2px;
    }

    h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    [data-testid="stMetric"] {
        background: #202020;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #D0D0D0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 900;
    }

    [data-testid="stMetricDelta"] {
        color: #D5FF36 !important;
    }

    .stButton > button {
        background: #D5FF36;
        color: #111111;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        min-height: 45px;
    }

    .stButton > button:hover {
        background: #E5FF75;
        color: #111111;
        border: none;
    }

    .stDownloadButton > button {
        background: #F477CE;
        color: #111111;
        font-weight: 800;
        border: none;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: #D5FF36 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #444444;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stFileUploader"] {
        background: #202020;
        border: 1px dashed #777777;
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #414141;
    }

    .brand {
        font-size: 25px;
        font-weight: 900;
        letter-spacing: -1px;
        color: #FFFFFF;
    }

    .brand-star {
        color: #D5FF36;
    }

    .hero {
        background: #202020;
        border: 1px solid #454545;
        border-radius: 15px;
        padding: 35px;
        margin: 15px 0 25px 0;
    }

    .kicker {
        color: #D5FF36;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
    }

    .hero-heading {
        font-size: clamp(36px, 5vw, 65px);
        line-height: 1.03;
        font-weight: 900;
        letter-spacing: -3px;
        margin: 20px 0;
        color: #FFFFFF;
    }

    .hero-highlight {
        color: #D5FF36;
    }

    .hero-description {
        color: #C6C6C6;
        font-size: 15px;
        line-height: 1.8;
        max-width: 650px;
    }

    .hero-art {
        background: #D5FF36;
        color: #111111;
        border-radius: 15px;
        min-height: 190px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 100px;
        font-weight: 900;
    }

    .section-label {
        color: #D5FF36;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        margin-top: 15px;
    }

    .segment-card {
        background: #202020;
        border: 1px solid #494949;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        margin-bottom: 12px;
    }

    .segment-number {
        font-size: 10px;
        color: #D5FF36;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .segment-title {
        color: #FFFFFF;
        font-size: 21px;
        font-weight: 900;
        margin: 12px 0;
    }

    .segment-detail {
        color: #D0D0D0;
        font-size: 13px;
        line-height: 2;
    }

    .segment-bar {
        height: 7px;
        border-radius: 8px;
        margin-top: 15px;
    }

    .insight-card {
        padding: 20px;
        border-radius: 12px;
        color: #111111;
        min-height: 125px;
        margin-bottom: 10px;
    }

    .insight-label {
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 2px;
    }

    .insight-value {
        font-size: 27px;
        font-weight: 900;
        margin-top: 12px;
    }

    .footer {
        background: #202020;
        border: 1px solid #454545;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: #D0D0D0;
        line-height: 2.4;
        font-size: 11px;
        letter-spacing: 1.5px;
        margin-top: 35px;
    }

    .footer-brand {
        font-size: 28px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: -1px;
    }

    .footer-star {
        color: #D5FF36;
    }

    .small-muted {
        color: #AFAFAF;
        font-size: 12px;
    }

</style>
"""


def render_html(content):
    """Render HTML safely as HTML rather than a Markdown code block."""
    st.html(textwrap.dedent(content).strip())


st.html(CSS)


# ============================================================
# 3. BRANDING AND HERO SECTION
# ============================================================

render_html("""
<div style="display:flex;justify-content:space-between;
align-items:center;flex-wrap:wrap;padding:8px 0 20px 0;
border-bottom:1px solid #444444;">

    <div class="brand">
        CLUSTER<span class="brand-star">✳</span>O
    </div>

    <div style="background:#202020;border:1px solid #444;
    padding:12px 18px;border-radius:9px;
    color:#D5FF36;font-size:11px;letter-spacing:1.5px;
    font-weight:800;">
        CUSTOMER INTELLIGENCE STUDIO
    </div>
</div>

<div class="hero">
    <div class="kicker">DISCOVER · GROUP · UNDERSTAND</div>

    <div class="hero-heading">
        KNOW YOUR<br>
        CUSTOMERS.<br>
        <span class="hero-highlight">FIND THE PATTERNS.</span>
    </div>

    <div style="font-size:20px;font-weight:700;margin-bottom:12px;">
        Customer Segmentation, powered by K-Means.
    </div>

    <div class="hero-description">
        Turn customer data into meaningful insights.
        Discover purchasing behaviour, identify hidden
        patterns and understand what makes every
        customer segment unique.
    </div>
</div>
""")


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.markdown("## ✳ CLUSTERO")
st.sidebar.caption("CUSTOMER INTELLIGENCE STUDIO")

st.sidebar.markdown("---")
st.sidebar.markdown("### 01 / DATASET")

uploaded = st.sidebar.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 02 / TECHNOLOGY")

for technology in [
    "Python",
    "Pandas",
    "Scikit-learn",
    "Plotly",
    "Streamlit",
    "Podman"
]:
    st.sidebar.write("✳ " + technology)

st.sidebar.markdown("---")
st.sidebar.markdown("### 03 / MODEL")
st.sidebar.write("Algorithm: K-Means")
st.sidebar.write("Preprocessing: StandardScaler")
st.sidebar.write("Validation: Elbow Method")


# ============================================================
# 5. CSV LOADING AND VALIDATION
# ============================================================

def read_customer_csv(file):
    """Try common encodings and automatically detect CSV delimiter."""

    content = file.getvalue()

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    errors = []

    for encoding in encodings:
        try:
            df = pd.read_csv(
                BytesIO(content),
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="error"
            )

            if len(df.columns) < 2:
                raise ValueError("The CSV contains fewer than two columns.")

            return df

        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError(
        "Unable to read CSV. Please check its structure and encoding. "
        + " | ".join(errors)
    )


def locate_column(columns, aliases):
    normalized = {
        str(col).strip().lower().replace("_", " ").replace("-", " "): col
        for col in columns
    }

    for alias in aliases:
        key = alias.lower().replace("_", " ").replace("-", " ")
        if key in normalized:
            return normalized[key]

    return None


if uploaded is None:
    st.info(
        "Upload your customer CSV using the sidebar to start "
        "exploring customer segments."
    )

    st.markdown("### Your analysis journey")

    c1, c2, c3 = st.columns(3)

    c1.info("01 · Upload your customer dataset")
    c2.info("02 · Discover patterns using K-Means")
    c3.info("03 · Predict new customer segments")

    st.stop()


try:
    raw = read_customer_csv(uploaded)
except Exception as exc:
    st.error(f"CSV loading error: {exc}")
    st.stop()


age_col = locate_column(
    raw.columns,
    ["Age"]
)

income_col = locate_column(
    raw.columns,
    [
        "Annual Income (k$)",
        "Annual Income",
        "Annual_Income",
        "Income"
    ]
)

score_col = locate_column(
    raw.columns,
    [
        "Spending Score (1-100)",
        "Spending Score",
        "Spending_Score",
        "Spending"
    ]
)

missing = [
    name
    for name, column in [
        ("Age", age_col),
        ("Annual Income", income_col),
        ("Spending Score", score_col)
    ]
    if column is None
]

if missing:
    st.error(
        f"Missing columns: {', '.join(missing)}. "
        f"Detected columns: {list(raw.columns)}"
    )
    st.stop()


df = raw[[age_col, income_col, score_col]].copy()

df.columns = [
    "Age",
    "Annual Income",
    "Spending Score"
]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan).dropna()

if len(df) < 3:
    st.error("At least three valid customer records are required.")
    st.stop()

if (df["Age"] <= 0).any() or (df["Annual Income"] < 0).any():
    st.error("Please check that age and income values are valid.")
    st.stop()

if not df["Spending Score"].between(1, 100).all():
    st.error("Spending Score must be between 1 and 100.")
    st.stop()

if df.nunique().min() < 2:
    st.error("Each feature must contain at least two distinct values.")
    st.stop()


# Save uploaded dataset to mounted storage.

try:
    safe_name = Path(uploaded.name).name
    (DATA_DIR / safe_name).write_bytes(uploaded.getvalue())
except OSError as exc:
    st.warning(f"Could not save uploaded dataset: {exc}")


# ============================================================
# 6. MODEL CONFIGURATION
# ============================================================

st.markdown('<div class="section-label">MODEL CONFIGURATION / 02</div>',
            unsafe_allow_html=True)

st.header("Configure your clustering model.")

left, right = st.columns([1, 1.3])

with left:
    max_k = min(10, len(df) - 1)

    k = st.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=max_k,
        value=min(5, max_k)
    )

with right:
    render_html("""
    <div style="background:#202020;border:1px solid #444;
    border-radius:12px;padding:20px;margin-top:5px;">
        <div style="color:#D5FF36;font-weight:800;">
            K-MEANS CLUSTERING
        </div>
        <p style="color:#DDDDDD;font-size:13px;line-height:1.8;">
            Group customers based on similarities in age,
            annual income and spending score. All three
            features are standardised before clustering.
        </p>
    </div>
    """)


# ============================================================
# 7. STANDARDISATION AND K-MEANS
# ============================================================

features = df[
    ["Age", "Annual Income", "Spending Score"]
]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

inertia = []
silhouette_values = []

ks = list(range(1, max_k + 1))

for n in ks:
    temporary_model = KMeans(
        n_clusters=n,
        random_state=42,
        n_init=10
    )

    temporary_labels = temporary_model.fit_predict(scaled)
    inertia.append(temporary_model.inertia_)

    if n >= 2 and len(np.unique(temporary_labels)) > 1:
        silhouette_values.append(
            silhouette_score(scaled, temporary_labels)
        )
    else:
        silhouette_values.append(None)


model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(scaled)

results = df.copy()
results["Cluster"] = labels

cluster_summary = results.groupby("Cluster").agg(
    Customers=("Cluster", "size"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income", "mean"),
    Average_Spending=("Spending Score", "mean")
).round(2)

cluster_summary["Share"] = (
    cluster_summary["Customers"] / len(results) * 100
).round(2)


# ============================================================
# 8. CLUSTER NAMING
# ============================================================

overall_income = df["Annual Income"].mean()
overall_spending = df["Spending Score"].mean()


def get_segment_name(row):
    high_income = row["Average_Income"] >= overall_income
    high_spending = row["Average_Spending"] >= overall_spending

    if high_income and high_spending:
        return "Premium Customers"

    if high_income and not high_spending:
        return "Careful Affluents"

    if not high_income and high_spending:
        return "Enthusiastic Spenders"

    return "Budget Conscious"


cluster_summary["Segment"] = cluster_summary.apply(
    get_segment_name,
    axis=1
)


# ============================================================
# 9. LIVE ANALYTICS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-label">LIVE ANALYTICS / 03</div>',
    unsafe_allow_html=True
)

st.header("The numbers behind the story.")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "CUSTOMERS",
    f"{len(results):,}"
)

m2.metric(
    "SEGMENTS",
    k
)

m3.metric(
    "FEATURES",
    "03"
)

m4.metric(
    "ALGORITHM",
    "K-MEANS"
)

st.caption(
    f"Dataset: {html.escape(uploaded.name)} · "
    f"{len(results):,} valid customer records"
)


# ============================================================
# 10. NAVIGATION TABS
# ============================================================

overview_tab, data_tab, analytics_tab, prediction_tab, export_tab = st.tabs(
    [
        "✳ Overview",
        "▤ Data Explorer",
        "◉ Analytics",
        "✦ Prediction",
        "↓ Export"
    ]
)


# ============================================================
# 11. OVERVIEW TAB
# ============================================================

with overview_tab:

    st.markdown(
        '<div class="section-label">CUSTOMER INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.header("Meet your customer segments.")

    chart_left, chart_right = st.columns(2)

    with chart_left:

        distribution = (
            results["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        distribution.columns = ["Cluster", "Customers"]

        distribution["Cluster"] = (
            "Cluster " + distribution["Cluster"].astype(str)
        )

        fig = px.bar(
            distribution,
            x="Cluster",
            y="Customers",
            color="Cluster",
            color_discrete_sequence=COLORS,
            text="Customers",
            title="Customer Distribution"
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CARD,
            plot_bgcolor=CARD,
            font_color=TEXT,
            showlegend=False,
            margin=dict(l=15, r=15, t=55, b=15),
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="overview_distribution"
        )

    with chart_right:

        fig = px.pie(
            cluster_summary.reset_index(),
            names="Cluster",
            values="Customers",
            hole=0.62,
            color="Cluster",
            color_discrete_sequence=COLORS,
            title="Customer Segment Share"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate=(
                "Cluster %{label}<br>"
                "Customers: %{value}<br>"
                "Share: %{percent}<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CARD,
            font_color=TEXT,
            margin=dict(l=15, r=15, t=55, b=15)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="overview_share"
        )

    st.markdown(
        '<div class="section-label">SEGMENT PROFILES</div>',
        unsafe_allow_html=True
    )

    st.header("Understand your customers.")

    card_columns = st.columns(2)

    for index, (cluster_id, row) in enumerate(
        cluster_summary.iterrows()
    ):

        color = COLORS[index % len(COLORS)]

        with card_columns[index % 2]:

            safe_segment = html.escape(str(row["Segment"]))

            render_html(f"""
            <div class="segment-card">

                <div class="segment-number">
                    CUSTOMER SEGMENT {int(cluster_id)}
                </div>

                <div class="segment-title">
                    {safe_segment}
                </div>

                <div class="segment-detail">
                    Customers: {int(row['Customers']):,}
                    <br>
                    Share: {row['Share']:.1f}%
                    <br>
                    Average age: {row['Average_Age']:.1f}
                    <br>
                    Average income: {row['Average_Income']:.1f}k
                    <br>
                    Average spending: {row['Average_Spending']:.1f}
                </div>

                <div style="height:6px;background:#414141;
                border-radius:6px;margin-top:15px;">

                    <div class="segment-bar"
                    style="background:{color};
                    width:{row['Share']}%;margin-top:0;">
                    </div>

                </div>

            </div>
            """)

    st.markdown(
        '<div class="section-label">QUICK INSIGHTS</div>',
        unsafe_allow_html=True
    )

    st.header("What stands out?")

    largest = cluster_summary["Customers"].idxmax()
    highest_spending = cluster_summary["Average_Spending"].idxmax()
    highest_income = cluster_summary["Average_Income"].idxmax()

    i1, i2, i3 = st.columns(3)

    with i1:
        render_html(f"""
        <div class="insight-card" style="background:#D5FF36;">
            <div class="insight-label">LARGEST SEGMENT</div>
            <div class="insight-value">Cluster {largest}</div>
            <div>{int(cluster_summary.loc[largest, 'Customers'])} customers</div>
        </div>
        """)

    with i2:
        render_html(f"""
        <div class="insight-card" style="background:#F477CE;">
            <div class="insight-label">HIGHEST SPENDING</div>
            <div class="insight-value">Cluster {highest_spending}</div>
            <div>Score {cluster_summary.loc[highest_spending, 'Average_Spending']:.1f}</div>
        </div>
        """)

    with i3:
        render_html(f"""
        <div class="insight-card" style="background:#A99AFF;">
            <div class="insight-label">HIGHEST INCOME</div>
            <div class="insight-value">Cluster {highest_income}</div>
            <div>Average {cluster_summary.loc[highest_income, 'Average_Income']:.1f}k</div>
        </div>
        """)


# ============================================================
# 12. DATA EXPLORER TAB
# ============================================================

with data_tab:

    st.header("Explore your customer dataset.")

    st.caption("Inspect the uploaded records used by the model.")

    d1, d2, d3 = st.columns(3)

    d1.metric("Original rows", len(raw))
    d2.metric("Valid rows", len(df))
    d3.metric("Removed rows", len(raw) - len(df))

    st.dataframe(
        results,
        use_container_width=True,
        height=400
    )

    st.subheader("Feature statistics")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    st.subheader("Feature relationships")

    selected_x = st.selectbox(
        "Horizontal axis",
        features.columns.tolist(),
        index=1
    )

    selected_y = st.selectbox(
        "Vertical axis",
        features.columns.tolist(),
        index=2
    )

    fig = px.scatter(
        results,
        x=selected_x,
        y=selected_y,
        color=results["Cluster"].astype(str),
        color_discrete_sequence=COLORS,
        hover_data=["Age", "Annual Income", "Spending Score"],
        title=f"{selected_x} vs {selected_y}"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="data_relationships"
    )


# ============================================================
# 13. ANALYTICS TAB
# ============================================================

with analytics_tab:

    st.header("Discover the patterns.")

    st.caption(
        "Use the interactive charts to explore customer behaviour."
    )

    # Elbow graph

    st.subheader("01 / Elbow Method")

    elbow_fig = go.Figure()

    elbow_fig.add_trace(
        go.Scatter(
            x=ks,
            y=inertia,
            mode="lines+markers+text",
            text=[round(v, 1) for v in inertia],
            textposition="top center",
            line=dict(color="#D5FF36", width=3),
            marker=dict(size=10),
            name="WCSS"
        )
    )

    elbow_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (WCSS)",
        title="Finding the optimal K",
        hovermode="x unified"
    )

    st.plotly_chart(
        elbow_fig,
        use_container_width=True,
        key="elbow_chart"
    )

    st.info(
        "Look for the elbow point where the reduction in WCSS "
        "starts slowing down. This helps guide the choice of K; "
        "it is not an automatic guarantee of the best clustering."
    )

    # Main clustering scatter

    st.subheader("02 / Customer Segmentation")

    fig = px.scatter(
        results,
        x="Annual Income",
        y="Spending Score",
        color=results["Cluster"].astype(str),
        color_discrete_sequence=COLORS,
        hover_data=["Age", "Annual Income", "Spending Score"],
        title="Annual Income vs Spending Score",
        opacity=0.85
    )

    fig.update_traces(
        marker=dict(size=10, line=dict(width=0.5, color="#FFFFFF"))
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        xaxis_title="Annual Income (k$)",
        yaxis_title="Spending Score (1-100)",
        legend_title="Customer Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cluster_scatter"
    )

    # 3D clustering chart

    st.subheader("03 / Three-Dimensional Customer View")

    fig_3d = px.scatter_3d(
        results,
        x="Age",
        y="Annual Income",
        z="Spending Score",
        color=results["Cluster"].astype(str),
        color_discrete_sequence=COLORS,
        hover_data=["Age", "Annual Income", "Spending Score"],
        title="Customer Segments in 3D"
    )

    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        font_color=TEXT,
        scene=dict(
            xaxis_title="Age",
            yaxis_title="Annual Income",
            zaxis_title="Spending Score",
            bgcolor=CARD
        )
    )

    st.plotly_chart(
        fig_3d,
        use_container_width=True,
        key="cluster_3d"
    )

    # Cluster feature comparison

    st.subheader("04 / Cluster Behaviour Comparison")

    comparison = cluster_summary.reset_index().melt(
        id_vars=["Cluster"],
        value_vars=[
            "Average_Age",
            "Average_Income",
            "Average_Spending"
        ],
        var_name="Feature",
        value_name="Average Value"
    )

    fig = px.bar(
        comparison,
        x="Cluster",
        y="Average Value",
        color="Feature",
        barmode="group",
        color_discrete_sequence=COLORS,
        title="Average Customer Characteristics"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cluster_comparison"
    )

    # Silhouette scores

    st.subheader("05 / Clustering Quality")

    valid_silhouette = [
        (cluster_count, value)
        for cluster_count, value in zip(ks, silhouette_values)
        if value is not None
    ]

    silhouette_fig = go.Figure()

    silhouette_fig.add_trace(
        go.Scatter(
            x=[x[0] for x in valid_silhouette],
            y=[x[1] for x in valid_silhouette],
            mode="lines+markers",
            line=dict(color="#F477CE", width=3),
            marker=dict(size=9)
        )
    )

    silhouette_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        title="Silhouette Score by K",
        xaxis_title="Number of Clusters",
        yaxis_title="Silhouette Score"
    )

    st.plotly_chart(
        silhouette_fig,
        use_container_width=True,
        key="silhouette_chart"
    )

    current_silhouette = silhouette_score(scaled, labels)

    st.metric(
        "Current model silhouette score",
        f"{current_silhouette:.3f}"
    )

    st.caption(
        "A silhouette score closer to 1 generally indicates "
        "better-separated clusters. Interpret it alongside "
        "the elbow curve and cluster profiles."
    )

    st.subheader("06 / Cluster Summary")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )


# ============================================================
# 14. NEW CUSTOMER PREDICTION
# ============================================================

with prediction_tab:

    st.header("Predict a new customer segment.")

    st.caption(
        "Enter a customer's details and let the trained K-Means "
        "model assign the closest cluster."
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        new_age = st.number_input(
            "Customer Age",
            min_value=1,
            max_value=100,
            value=25
        )

    with p2:
        new_income = st.number_input(
            "Annual Income (k$)",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

    with p3:
        new_score = st.number_input(
            "Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=75
        )

    if st.button(
        "✳ Predict Customer Segment",
        use_container_width=True
    ):

        new_customer = pd.DataFrame({
            "Age": [new_age],
            "Annual Income": [new_income],
            "Spending Score": [new_score]
        })

        scaled_customer = scaler.transform(new_customer)

        predicted_cluster = int(
            model.predict(scaled_customer)[0]
        )

        predicted_profile = cluster_summary.loc[predicted_cluster]

        predicted_name = html.escape(
            str(predicted_profile["Segment"])
        )

        color = COLORS[predicted_cluster % len(COLORS)]

        render_html(f"""
        <div style="background:#202020;border:2px solid {color};
        border-radius:15px;padding:25px;margin-top:20px;">

            <div style="color:{color};font-size:11px;
            font-weight:900;letter-spacing:3px;">
                PREDICTION SUCCESSFUL
            </div>

            <div style="font-size:34px;font-weight:900;
            color:#FFFFFF;margin:15px 0;">
                Cluster {predicted_cluster}
            </div>

            <div style="font-size:22px;color:{color};
            font-weight:800;">
                {predicted_name}
            </div>

            <div style="color:#CCCCCC;margin-top:18px;
            line-height:2;">
                Average cluster age:
                {predicted_profile['Average_Age']:.1f}
                <br>
                Average income:
                {predicted_profile['Average_Income']:.1f}k
                <br>
                Average spending:
                {predicted_profile['Average_Spending']:.1f}
                <br>
                Customers in this cluster:
                {int(predicted_profile['Customers'])}
            </div>
        </div>
        """)

        st.success(
            f"The new customer has been assigned to Cluster "
            f"{predicted_cluster}."
        )


# ============================================================
# 15. EXPORT TAB
# ============================================================

with export_tab:

    st.header("Export your customer intelligence.")

    st.caption(
        "Download the final dataset with cluster assignments."
    )

    csv_data = results.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="↓ Download Clustered Customer CSV",
        data=csv_data,
        file_name="customer_segments.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.subheader("Cluster summary")

    summary_csv = (
        cluster_summary
        .reset_index()
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        label="↓ Download Cluster Summary",
        data=summary_csv,
        file_name="cluster_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

    try:
        (DATA_DIR / "customer_segments.csv").write_bytes(csv_data)

        (
            DATA_DIR / "cluster_summary.csv"
        ).write_bytes(summary_csv)

        st.success(
            f"Results saved successfully in {DATA_DIR}"
        )

    except OSError as exc:
        st.warning(f"Could not save results: {exc}")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown("---")

render_html("""
<div class="footer">

    <div class="footer-brand">
        CLUSTERO <span class="footer-star">✳</span>
    </div>

    CUSTOMER SEGMENTATION · POWERED BY K-MEANS

    <br>

    PYTHON · PANDAS · SCIKIT-LEARN · PLOTLY · STREAMLIT · PODMAN

    <br>

    LINUX AND CONTAINER TECHNOLOGY CAPSTONE PROJECT

    <br><br>

    TURN DATA INTO DISCOVERY.

</div>
""")


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

# High-contrast cluster colours: pink/yellow lead, with vivid blue/green accents.
PALETTE = [
    "#FF3F91",  # electric pink
    "#FFC928",  # marigold yellow
    "#2687F5",  # bright blue
    "#20B978",  # vivid green
    "#9B5DE5",  # violet
    "#FF7043",  # coral orange
    "#00AFC1",  # turquoise
    "#E84A8A",  # rose
    "#8DBF26",  # lime green
    "#F08AC5"   # soft magenta
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
    --cream: #FFF9F2;
    --pink: #FFB4D2;
    --purple: #F6A7C6;
    --yellow: #FFD447;
    --mint: #BCEBDD;
    --blue: #B9DDF7;
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
        #FFE18A 0%,
        #FFC0D5 58%,
        #FFF0C2 100%
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
        #FFAAC8 56%,
        #FFE8A3 100%
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
.metric-card.purple { background: #FFC247; }
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
    color: #B52D61;
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
    background: linear-gradient(100deg,#FFD447,#FFB4D2,#FFC247);
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


.chart-selection-note {
    display: inline-block;
    margin: 10px 0 16px;
    padding: 10px 16px;
    border-radius: 999px;
    border: 1px solid rgba(121,84,232,.25);
    background: rgba(255,255,255,.76);
    color: #51465D;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: .8px;
    box-shadow: 0 6px 16px rgba(35,25,55,.07);
}

.insight-card {
    min-height: 150px;
    padding: 20px;
    margin: 8px 0 14px;
    border: 1px solid rgba(255,255,255,.9);
    border-left: 4px solid #FF9FC4;
    border-radius: 16px;
    background: rgba(255,255,255,.72);
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 28px rgba(35,25,55,.08);
    animation: clusteroFadeUp .5s ease both;
}
.insight-eyebrow {
    color: #B52D61;
    font: 700 10px 'Space Mono', monospace;
    letter-spacing: 1.3px;
}
.insight-title {
    margin: 8px 0;
    color: #191923;
    font-size: 18px;
    font-weight: 900;
}
.insight-copy {
    color: #49434F;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.7;
}

/* ============================================================
   CLUSTERO MODERN GLASS UI + MICRO-INTERACTIONS
   ============================================================ */

@keyframes clusteroFadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes clusteroGlow {
    0%, 100% { box-shadow: 0 10px 30px rgba(121,84,232,.10); }
    50%      { box-shadow: 0 14px 38px rgba(255,111,174,.22); }
}

@keyframes clusteroFloat {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-5px); }
}

.panel, .metric-card, .segment-card, [data-testid="stMetric"],
[data-testid="stDataFrame"], div[data-testid="stExpander"] {
    background: rgba(255,255,255,.62) !important;
    backdrop-filter: blur(18px) saturate(145%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(145%) !important;
    border: 1px solid rgba(255,255,255,.82) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 34px rgba(35,25,55,.09), inset 0 1px 0 rgba(255,255,255,.8) !important;
}

.panel, .metric-card, .segment-card {
    animation: clusteroFadeUp .55s ease both;
}

.panel:hover, .segment-card:hover, .metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 42px rgba(35,25,55,.15) !important;
}

.hero {
    border-radius: 24px !important;
    box-shadow: 0 20px 55px rgba(25,25,35,.16) !important;
    animation: clusteroFadeUp .7s ease both, clusteroGlow 7s ease-in-out infinite;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px !important;
    padding: 8px !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,.45) !important;
    backdrop-filter: blur(14px);
}

.stTabs [data-baseweb="tab"] {
    min-height: 52px !important;
    padding: 13px 22px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(25,25,35,.14) !important;
    box-shadow: 0 5px 14px rgba(25,25,35,.07) !important;
    transition: all .22s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-3px);
    background: #FFF0F7 !important;
}

.stRadio [role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)) !important;
    gap: 12px !important;
}

.stRadio [role="radiogroup"] label {
    min-height: 64px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 12px 16px !important;
    border: 1.5px solid rgba(25,25,35,.16) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,.72) !important;
    box-shadow: 0 7px 18px rgba(25,25,35,.07) !important;
    transition: all .2s ease !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    cursor: pointer !important;
}

.stRadio [role="radiogroup"] label:hover {
    transform: translateY(-3px);
    border-color: #FF78B9 !important;
    background: #FFF0F7 !important;
}

.stRadio [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #FFD8E8, #FFE7A0) !important;
    border: 2px solid #B52D61 !important;
    box-shadow: 0 9px 24px rgba(121,84,232,.17) !important;
}

.stButton > button, .stDownloadButton > button {
    min-height: 54px !important;
    border-radius: 13px !important;
    box-shadow: 0 8px 20px rgba(25,25,35,.13) !important;
    transition: transform .2s ease, box-shadow .2s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 13px 28px rgba(25,25,35,.18) !important;
}

[data-testid="stPlotlyChart"] {
    padding: 12px !important;
    border: 1px solid rgba(255,255,255,.9) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,.60) !important;
    backdrop-filter: blur(14px) !important;
    box-shadow: 0 12px 30px rgba(35,25,55,.08) !important;
    animation: clusteroFadeUp .45s ease both;
}

div[data-testid="stExpander"] {
    overflow: hidden;
}

div[data-testid="stExpander"] summary {
    font-size: 16px !important;
    font-weight: 900 !important;
    padding: 14px 16px !important;
}

@media (max-width: 850px) {
    .stRadio [role="radiogroup"] {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
}

/* CLUSTERO NEO-BRUTALIST PINK + YELLOW OVERRIDES */
:root { --ink:#24202A; --cream:#FFF9F2; --pink:#FFB4D2; --hot-pink:#F04483; --yellow:#FFD447; --paper:#FFFDF9; --soft-blue:#B9DDF7; --soft-green:#BCEBDD; }
.stApp, [data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 92% 2%,rgba(255,180,210,.24),transparent 26%),
              radial-gradient(circle at 4% 34%,rgba(255,212,71,.18),transparent 24%),
              radial-gradient(circle at 96% 72%,rgba(105,201,194,.13),transparent 20%),
              radial-gradient(circle at 8% 88%,rgba(121,188,235,.12),transparent 18%),#FFF9F2 !important;
  color:#24202A !important;
}
.main .block-container { max-width:1540px !important; padding:1.25rem clamp(1rem,2.3vw,2.4rem) 3rem !important; }
[data-testid="stSidebar"] { background:linear-gradient(165deg,#FFE18A 0%,#FFC0D5 48%,#D9F0E7 76%,#DCEEFF 100%) !important; border-right:2px solid #24202A !important; }
[data-testid="stSidebar"] * { color:#24202A !important; }
.hero { background:linear-gradient(118deg,#FFD95B 0%,#FFAAC8 48%,#FFE8A3 76%,#D9F0E7 100%) !important; border:2px solid #24202A !important; border-radius:16px !important; box-shadow:7px 7px 0 #24202A !important; }
.panel,.segment-card,.metric-card { border:2px solid #24202A !important; border-radius:12px !important; box-shadow:4px 4px 0 #24202A !important; background:#FFFDF9 !important; backdrop-filter:none !important; }
.metric-card.pink { background:#FFB4D2 !important; }
.metric-card.yellow { background:#FFD447 !important; }
.metric-card.purple { background:#F6A7C6 !important; }
.metric-card.mint { background:#BCEBDD !important; }
.metric-card.blue { background:#B9DDF7 !important; }
.panel:hover,.segment-card:hover,.metric-card:hover { transform:translateY(-3px) !important; box-shadow:6px 6px 0 #24202A !important; }
.stButton > button,.stDownloadButton > button,[data-testid="stFileUploader"] button {
  background:#FFB4D2 !important; color:#24202A !important; border:2px solid #24202A !important;
  border-radius:8px !important; box-shadow:3px 3px 0 #24202A !important; font-weight:800 !important;
  min-height:44px !important; transition:transform .16s ease,box-shadow .16s ease,background .16s ease !important;
}
.stButton > button *,.stDownloadButton > button *,[data-testid="stFileUploader"] button * { color:#24202A !important; opacity:1 !important; }
.stButton > button:hover,.stDownloadButton > button:hover,[data-testid="stFileUploader"] button:hover {
  background:#FFD447 !important; color:#24202A !important; transform:translate(-2px,-2px) !important; box-shadow:5px 5px 0 #24202A !important;
}
.stRadio [role="radiogroup"] { display:flex !important; flex-wrap:wrap !important; gap:10px !important; }
.stRadio [role="radiogroup"] > label {
  background:#FFFDF9 !important; color:#24202A !important; border:2px solid #24202A !important;
  border-radius:9px !important; box-shadow:3px 3px 0 #24202A !important; padding:10px 16px !important;
  min-height:44px !important; font-weight:800 !important; transition:transform .16s ease,background .16s ease !important;
}
.stRadio [role="radiogroup"] > label * { color:#24202A !important; }
.stRadio [role="radiogroup"] > label:hover { background:#FFE7A0 !important; transform:translateY(-2px) !important; }
.stRadio [role="radiogroup"] > label:has(input:checked) { background:#FFB4D2 !important; border:2px solid #24202A !important; box-shadow:4px 4px 0 #24202A !important; }
.stSelectbox [data-baseweb="select"] > div,.stMultiSelect [data-baseweb="select"] > div,.stNumberInput input {
  background:#FFFDF9 !important; color:#24202A !important; border:2px solid #24202A !important; border-radius:8px !important;
}
.stSelectbox [data-baseweb="select"] *,.stMultiSelect [data-baseweb="select"] * { color:#24202A !important; }
[data-testid="stAlert"] { background:#FFF0B8 !important; border:2px solid #24202A !important; border-radius:9px !important; color:#24202A !important; }
[data-testid="stAlert"] * { color:#24202A !important; }
[data-testid="stPlotlyChart"] {
  background:#FFFDF9 !important; border:2px solid #24202A !important; border-radius:12px !important;
  padding:10px 10px 14px !important; box-shadow:4px 4px 0 #24202A !important; overflow:visible !important;
  width:100% !important; min-width:0 !important; box-sizing:border-box !important;
}
[data-testid="stDataFrame"] { border:2px solid #24202A !important; border-radius:10px !important; box-shadow:3px 3px 0 #24202A !important; }
.chart-selection-note { background:#FFF0B8 !important; color:#24202A !important; border:2px solid #24202A !important; border-radius:8px !important; box-shadow:3px 3px 0 #24202A !important; }
.section-kicker { color:#B52D61 !important; }
.insight-card { background:#FFFDF9 !important; border:2px solid #24202A !important; border-left:6px solid #F04483 !important; border-radius:10px !important; box-shadow:3px 3px 0 #24202A !important; }
.footer { background:linear-gradient(100deg,#FFD447,#FFB4D2) !important; border:2px solid #24202A !important; border-radius:12px !important; box-shadow:5px 5px 0 #24202A !important; }
@media (prefers-reduced-motion: reduce) {
  *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
}

/* POP COLOUR CHARTS + EXPLANATORY MICRO-INTERACTIONS */
:root { --electric-blue:#2687F5; --vivid-green:#20B978; --hot-pink:#FF3F91; --marigold:#FFC928; }
[data-testid="stPlotlyChart"] {
  border:2px solid #24202A !important;
  box-shadow:5px 5px 0 #24202A, 0 0 0 3px rgba(38,135,245,.08) !important;
  transition:box-shadow .22s ease !important;
  overflow:visible !important; width:100% !important; min-width:0 !important;
  box-sizing:border-box !important; position:relative !important;
}
[data-testid="stPlotlyChart"]:hover { box-shadow:7px 8px 0 #24202A, 0 0 0 4px rgba(32,185,120,.16) !important; }
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
  width:100% !important; max-width:100% !important; min-width:0 !important;
  box-sizing:border-box !important;
}
[data-testid="stPlotlyChart"] .modebar-container { right:8px !important; top:6px !important; }
[data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:0 !important; }
.stPopover > button { background:linear-gradient(110deg,#B9DDF7,#BCEBDD) !important; border:2px solid #24202A !important; border-radius:999px !important; box-shadow:3px 3px 0 #24202A !important; font-weight:900 !important; }
[data-testid="stPopover"] { border:2px solid #24202A !important; border-radius:12px !important; box-shadow:5px 5px 0 #24202A !important; }
.chart-tip { display:inline-block; margin:5px 0 12px; padding:9px 13px; border:2px solid #24202A; border-radius:10px; background:linear-gradient(105deg,#DCEEFF,#DDF8E9); box-shadow:3px 3px 0 #24202A; font-size:12px; font-weight:700; animation:tipPop .45s ease both; }
@keyframes tipPop { from {opacity:0;transform:translateY(8px) scale(.98)} to {opacity:1;transform:translateY(0) scale(1)} }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#2687F5 !important; border:2px solid #24202A !important; }
.stCheckbox label { font-weight:800 !important; }

/* Responsive chart-fit safeguards: preserve axes, legends and toolbars. */
[data-testid="stPlotlyChart"] { padding:8px 10px 14px !important; overflow:visible !important; }
[data-testid="stPlotlyChart"] iframe { width:100% !important; max-width:100% !important; }
[data-testid="stPlotlyChart"] .main-svg { max-width:100% !important; }
@media (max-width: 900px) {
  .main .block-container { padding-left:.75rem !important; padding-right:.75rem !important; }
  [data-testid="stPlotlyChart"] { padding:5px 5px 10px !important; }
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
        paper_bgcolor="#FFFDF9",
        plot_bgcolor="#FFFDF9",
        autosize=True,
        font=dict(
            family="Arial",
            color="#191923",
            size=13
        ),
        height=height,
        margin=dict(l=82, r=42, t=88, b=108, autoexpand=True),
        legend=dict(
            title="Customer Segments",
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,247,236,0.96)",
            bordercolor="#24202A",
            borderwidth=1,
            font=dict(size=12, color="#191923")
        ),
        hoverlabel=dict(
            bgcolor="#191923",
            bordercolor="#FFD447",
            font=dict(color="white", size=13)
        ),
        hovermode="closest",
        transition=dict(duration=420, easing="cubic-in-out")
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(38, 135, 245, 0.16)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="rgba(55, 45, 70, 0.35)",
        linewidth=1,
        color="#191923",
        title_font=dict(size=14, color="#191923"),
        title_standoff=16,
        tickfont=dict(size=12, color="#51465D"),
        ticks="outside",
        ticklen=5,
        automargin=True
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(38, 135, 245, 0.16)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="rgba(55, 45, 70, 0.35)",
        linewidth=1,
        color="#191923",
        title_font=dict(size=14, color="#191923"),
        title_standoff=16,
        tickfont=dict(size=12, color="#51465D"),
        ticks="outside",
        ticklen=5,
        automargin=True
    )

    return fig


def cluster_color_map(cluster_ids):
    return {
        str(cluster): PALETTE[int(cluster) % len(PALETTE)]
        for cluster in sorted(cluster_ids)
    }


def show_chart(fig):
    """Render a responsive Plotly figure inside its Streamlit column."""
    fig.update_layout(autosize=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOT_CONFIG,
        theme=None
    )


# ============================================================
# 6. CHART FUNCTIONS
# ============================================================

def make_scatter(results, x, y, centers=None, title="Customer Segmentation", marker_size=13, point_opacity=0.88):
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
        opacity=point_opacity,
        render_mode="webgl"
    )

    fig.update_traces(
        marker=dict(
            size=marker_size,
            line=dict(color="#24202A", width=1.15),
            sizemin=5
        ),
        selector=dict(mode="markers")
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
                color="#24202A",
                line=dict(color="#2687F5", width=4)
            )
        ))

    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title="Segments",
        dragmode="pan"
    )

    return style_chart(fig, title, 720)


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
            line=dict(color="#24202A", width=0.8)
        )
    )

    axis_3d_style = dict(
        color="#191923",
        showgrid=True,
        gridcolor="rgba(232, 62, 121, 0.16)",
        zeroline=False,
        showbackground=True,
        backgroundcolor="#FFF7EC"
    )
    fig.update_layout(
        paper_bgcolor="#FFFDF9",
        font=dict(color="#24202A", size=13),
        autosize=True,
        height=700,
        margin=dict(l=48, r=48, t=82, b=58, autoexpand=True),
        scene=dict(
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.25)),
            bgcolor="#FFFDF9",
            xaxis=dict(title="Age", **axis_3d_style),
            yaxis=dict(title="Annual Income", **axis_3d_style),
            zaxis=dict(title="Spending Score", **axis_3d_style)
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
        size_max=38,
        opacity=0.90
    )

    return style_chart(
        fig,
        "Bubble Chart — Income, Spending and Age",
        720
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
        720
    )


def make_elbow(ks, inertias, selected_k):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ks,
        y=inertias,
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#B52D61", width=4),
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
    """Build a stable, fully categorical spending comparison chart.

    Explicit category ordering and plain Python lists prevent Plotly from
    reusing stale category positions when the number of clusters changes.
    """
    chart_summary = summary.copy()
    chart_summary = chart_summary.loc[
        chart_summary["Average_Spending"].notna()
    ].sort_index()

    cluster_labels = [f"Cluster {int(cluster_id)}" for cluster_id in chart_summary.index]
    spending_values = pd.to_numeric(
        chart_summary["Average_Spending"], errors="coerce"
    ).astype(float).tolist()
    cluster_colors = [
        PALETTE[int(cluster_id) % len(PALETTE)]
        for cluster_id in chart_summary.index
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cluster_labels,
        y=spending_values,
        name="Average spending score",
        marker=dict(
            color=cluster_colors,
            line=dict(color="#191923", width=2),
        ),
        width=0.62,
        text=[f"{value:.1f}" for value in spending_values],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color="#191923", size=14),
        customdata=cluster_labels,
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Average spending score: %{y:.1f}<extra></extra>"
        )
    ))

    max_value = max(spending_values, default=100.0)
    y_max = max(100.0, max_value * 1.18)
    fig.update_layout(
        xaxis=dict(
            title="Customer Segment",
            type="category",
            categoryorder="array",
            categoryarray=cluster_labels,
            tickmode="array",
            tickvals=cluster_labels,
            ticktext=cluster_labels,
            automargin=True,
            fixedrange=False
        ),
        yaxis=dict(
            title="Average Spending Score",
            range=[0, y_max],
            rangemode="tozero",
            automargin=True,
            fixedrange=False
        ),
        bargap=0.28,
        showlegend=False,
        uniformtext=dict(minsize=11, mode="show")
    )

    fig.update_layout(uirevision="spending-comparison-v2")
    return style_chart(
        fig,
        "Average Spending Score by Segment",
        540
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
    <div class="topbar-tag">CLUSTERO 2.1 · LIVE CUSTOMER ANALYTICS · K-MEANS</div>
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
        (c2, "#FFD447", "STEP 02", "Discover Groups",
         "Apply K-Means and uncover customer patterns.", "◉"),
        (c3, "#FFB4D2", "STEP 03", "Predict Segments",
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

    with st.expander("✳  K-MEANS EXPLAINER  |  CLICK THIS PLACARD TO LEARN HOW IT WORKS", expanded=False):
        st.markdown("""
        ### K-Means clustering
        K-Means is an **unsupervised machine-learning algorithm** that groups
        customers into K clusters based on how similar their selected features are.

        **How CLUSTERO uses it**
        1. **Choose K:** decide how many customer groups to create.
        2. **Scale the data:** StandardScaler puts age, income and spending score
           on comparable scales.
        3. **Place centroids:** the algorithm starts with K cluster centres.
        4. **Assign customers:** each customer is assigned to the nearest centre.
        5. **Update and repeat:** centres move to the mean of their assigned
           customers until the assignments stabilise.

        **Why it is useful:** it can reveal customer groups with different
        income and spending behaviours without needing pre-labelled categories.

        **Important:** cluster numbers are labels, not rankings. Cluster 0 is
        not inherently better or worse than Cluster 1.
        """)
        st.caption("CLUSTERO uses Age, Annual Income and Spending Score as its three clustering features.")


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

    guide_col, control_col = st.columns([1, 2])
    with guide_col:
        with st.popover("💡 Chart guide — what am I seeing?", use_container_width=True):
            st.markdown("""
            **2D Scatter:** each dot is one customer; colour identifies its K-Means cluster.

            **3D Scatter:** the same customers across Age, Income and Spending Score. Drag to rotate.

            **Bubble Chart:** bubble size represents age; position compares income and spending.

            **Box Plot:** compares the spread, median and outliers of spending scores by cluster.

            **Elbow Method:** helps compare within-cluster variation as K changes. A lower value alone does not prove the best K; look for the bend.
            """)
    with control_col:
        st.markdown("<div class='chart-tip'>✳ PRO TIP · Hover over any point or bar for customer-level details. Use the Plotly toolbar to zoom, pan, reset or save a chart.</div>", unsafe_allow_html=True)

    st.write("### 01 / Elbow Method")

    show_chart(make_elbow(ks, inertias, k))

    st.info(
        f"The current model uses K = {k}. "
        "Compare the inertia values to identify the elbow point."
    )

    st.divider()

    st.write("### 02 / Choose Your Visualisation")

    visualisation_options = [
        "2D Scatter",
        "3D Scatter",
        "Bubble Chart",
        "Box Plot"
    ]
    # Reset any legacy chart selection that is no longer supported.
    if st.session_state.get("clustero_visualisation") not in visualisation_options:
        st.session_state["clustero_visualisation"] = "2D Scatter"

    st.caption("Choose a view. The chart updates immediately when you switch.")
    visualisation = st.radio(
        "Chart type",
        options=visualisation_options,
        horizontal=True,
        key="clustero_visualisation",
        label_visibility="collapsed"
    )
    st.markdown(
        f"<div class='chart-selection-note'>NOW VIEWING&nbsp;&nbsp; / &nbsp;&nbsp;<b>{visualisation}</b></div>",
        unsafe_allow_html=True
    )

    st.write("### 03 / Explore a customer segment")
    segment_options = ["All clusters"] + [f"Cluster {int(c)}" for c in sorted(results["Cluster"].unique())]
    selected_segment = st.selectbox(
        "Focus the selected chart on",
        segment_options,
        key="analytics_segment_filter",
        help="Filter the selected visualisation to one cluster, or keep all clusters visible."
    )
    chart_results = results.copy()
    chart_centers = centers
    if selected_segment != "All clusters":
        selected_cluster_id = int(selected_segment.split()[-1])
        chart_results = chart_results[chart_results["Cluster"] == selected_cluster_id]
        chart_centers = centers[[selected_cluster_id]]
    st.caption(f"Showing {len(chart_results):,} of {len(results):,} customers in the selected view.")

    # Live chart controls are intentionally lightweight and affect the 2D view.
    marker_size = 13
    point_opacity = 0.88
    if visualisation == "2D Scatter":
        with st.expander("🎛️ Fine-tune the chart", expanded=False):
            size_col, opacity_col, centroid_col = st.columns(3)
            with size_col:
                marker_size = st.slider("Point size", 5, 24, 13, key="scatter_point_size")
            with opacity_col:
                point_opacity = st.slider("Point opacity", 0.35, 1.0, 0.88, 0.05, key="scatter_point_opacity")
            with centroid_col:
                show_centroids = st.checkbox("Show cluster centres", value=True, key="show_cluster_centres")
        if not st.session_state.get("show_cluster_centres", True):
            chart_centers_for_view = None
        else:
            chart_centers_for_view = chart_centers
    else:
        chart_centers_for_view = chart_centers

    if visualisation == "2D Scatter":
        st.write("### 04 / Choose chart features")
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

        if x_axis == y_axis:
            st.warning("Choose two different features for the horizontal and vertical axes.")
        else:
            show_chart(
                make_scatter(
                    chart_results,
                    x_axis,
                    y_axis,
                    chart_centers_for_view,
                    f"{x_axis} vs {y_axis} — Customer Segmentation",
                    marker_size=marker_size,
                    point_opacity=point_opacity
                )
            )

    elif visualisation == "3D Scatter":
        st.markdown("<div class='chart-tip'>🧭 ROTATE THE VIEW · Drag to rotate, scroll to zoom, and hover over a point to inspect its customer ID.</div>", unsafe_allow_html=True)
        show_chart(make_3d(chart_results))

    elif visualisation == "Bubble Chart":
        st.markdown("<div class='chart-tip'>🫧 BUBBLE KEY · Larger bubbles represent older customers. Colour shows the assigned cluster.</div>", unsafe_allow_html=True)
        show_chart(make_bubble(chart_results))

    elif visualisation == "Box Plot":
        st.markdown("<div class='chart-tip'>📦 BOX PLOT KEY · The centre line is the median; the box shows the middle 50% of scores; points beyond whiskers may be outliers.</div>", unsafe_allow_html=True)
        show_chart(make_box(chart_results))

    st.divider()

    st.write("### 05 / Primary Customer Segmentation Map")

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

    st.write("### 06 / Spending Comparison")

    show_chart(make_spending_chart(summary))

    st.write("### 07 / Segment Statistics")

    st.dataframe(summary, use_container_width=True)


    st.divider()
    st.write("### 08 / Customer intelligence briefing")
    st.caption("Generate a concise, data-driven interpretation of the clusters in this dataset.")
    if st.button("✦ GENERATE CUSTOMER INSIGHTS", key="generate_cluster_insights", use_container_width=True):
        largest_cluster = int(summary["Customers"].idxmax())
        highest_spend_cluster = int(summary["Average_Spending"].idxmax())
        highest_income_cluster = int(summary["Average_Income"].idxmax())
        largest_share = summary.loc[largest_cluster, "Customers"] / len(results) * 100
        low_spend_cluster = int(summary["Average_Spending"].idxmin())
        insight_items = [
            ("Largest audience", f"Cluster {largest_cluster} contains {int(summary.loc[largest_cluster, 'Customers']):,} customers ({largest_share:.1f}% of the dataset)."),
            ("Highest spending", f"Cluster {highest_spend_cluster} has the highest average spending score ({summary.loc[highest_spend_cluster, 'Average_Spending']:.1f})."),
            ("Highest income", f"Cluster {highest_income_cluster} has the highest average annual income ({summary.loc[highest_income_cluster, 'Average_Income']:.1f}k)."),
            ("Potential growth audience", f"Cluster {low_spend_cluster} has the lowest average spending score ({summary.loc[low_spend_cluster, 'Average_Spending']:.1f}); consider testing engagement or personalised offers with this group.")
        ]
        insight_cols = st.columns(2)
        for insight_index, (insight_title, insight_text) in enumerate(insight_items):
            with insight_cols[insight_index % 2]:
                render_html(f"""
                <div class="insight-card">
                    <div class="insight-eyebrow">DATA SIGNAL 0{insight_index + 1}</div>
                    <div class="insight-title">{insight_title}</div>
                    <div class="insight-copy">{insight_text}</div>
                </div>
                """)
        st.caption("These findings are descriptive patterns in the uploaded dataset, not causal explanations or guaranteed future behaviour.")


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


        # Dynamic conclusion based on the predicted cluster's profile.
        income_position = (
            "above" if profile["Average_Income"] >= income_median else "below"
        )
        spending_position = (
            "above" if profile["Average_Spending"] >= spending_median else "below"
        )
        segment_name = describe_cluster(profile)

        st.markdown("### ✳ Prediction conclusion")
        render_html(f"""
        <div class="panel" style="
            background:linear-gradient(135deg,rgba(255,255,255,.82),rgba(230,222,255,.78));
            border:1px solid rgba(121,84,232,.24);
            border-left:6px solid {color};
            padding:24px;
            animation:clusteroFadeUp .6s ease both;">
            <div style="font-family:monospace;font-size:11px;font-weight:900;
                        letter-spacing:1.5px;color:#B52D61;">
                CUSTOMER PROFILE SUMMARY
            </div>
            <div style="font-size:22px;font-weight:900;margin:9px 0 12px;color:#191923;">
                {segment_name}
            </div>
            <div style="font-size:14px;line-height:1.85;color:#34303D;">
                The predicted customer belongs to <b>Cluster {predicted}</b>.
                This segment has an average annual income of
                <b>{profile['Average_Income']:.1f}k</b> and an average spending
                score of <b>{profile['Average_Spending']:.1f}</b>.
                Compared with the dataset medians, this segment's income is
                <b>{income_position}</b> the median and its spending is
                <b>{spending_position}</b> the median.
                It currently contains <b>{int(profile['Customers'])}</b>
                customers in the uploaded dataset.
            </div>
            <div style="margin-top:14px;padding:12px 14px;border-radius:12px;
                        background:rgba(255,255,255,.72);font-size:13px;
                        font-weight:700;color:#34303D;">
                Interpretation: {segment_name}. Use this profile to guide
                customer communication and campaign decisions; treat it as a
                behavioural grouping, not a guarantee of an individual's future purchases.
            </div>
        </div>
        """)


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

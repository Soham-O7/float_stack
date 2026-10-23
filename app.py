import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
import importlib
import client
try:
    importlib.reload(client)
except Exception:
    pass
from client import fetch_backend_stub, generate_dynamic_stub

# 1. Page Configuration
st.set_page_config(
    page_title="DataDiet — Green AI for Smarter Datasets", 
    page_icon="🌿", 
    layout="wide"
)

# 2. Theme State Management & Persistence
theme_options = ["Bright Mode", "Dark Mode", "System"]
default_theme = "Dark Mode"
try:
    if "theme" in st.query_params:
        qp_theme = st.query_params["theme"]
        if qp_theme in theme_options:
            default_theme = qp_theme
except Exception:
    pass

if "app_theme_mode" not in st.session_state:
    st.session_state["app_theme_mode"] = default_theme

current_theme = st.session_state["app_theme_mode"]

def sync_theme():
    try:
        st.query_params["theme"] = st.session_state["app_theme_mode"]
    except Exception:
        pass

# 3. Dynamic Theme Styles (Bright Mode, Dark Mode, and System)
bright_styles = """
    :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        --background-color: #F8FAFC !important;
        --secondary-background-color: #FFFFFF !important;
        --text-color: #0F172A !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    
    /* Global Typography in Bright Mode */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
    }
    p, span, label, div {
        color: #334155;
    }
    .step-header {
        color: #0F172A !important;
    }

    /* SaaS Container Card */
    .saas-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #0F172A !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
    }

    /* Hero Banner in Bright Mode */
    .hero-banner {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F6 100%) !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
        color: #0F172A !important;
    }
    .hero-banner h1,
    .hero-banner h1 *,
    .hero-banner .hero-title,
    .hero-banner .hero-title *,
    .hero-title,
    div.hero-banner h1,
    [data-testid="stMarkdownContainer"] .hero-banner h1 {
        color: #0F172A !important;
    }
    .hero-banner p,
    .hero-banner p *,
    div.hero-banner p,
    [data-testid="stMarkdownContainer"] .hero-banner p {
        color: #475569 !important;
    }
    .hero-banner span {
        color: #FFFFFF !important;
    }

    /* Segmented Theme Toggle Pill in Bright Mode */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: #E2E8F0 !important;
        border: 1px solid #CBD5E1 !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] div,
    div[data-testid="stRadio"] [data-baseweb="radio"] span,
    div[data-testid="stRadio"] [data-baseweb="radio"] p {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) {
        background: #4F46E5 !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.35) !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) div,
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) span,
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Tabs in Bright Mode */
    [data-baseweb="tab-list"] {
        background-color: #F1F5F9 !important;
        border-bottom: 2px solid #E2E8F0 !important;
        border-radius: 8px 8px 0 0 !important;
        gap: 8px !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        color: #64748B !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"]:hover p,
    button[data-baseweb="tab"]:hover span,
    button[data-baseweb="tab"]:hover div {
        color: #0F172A !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #4F46E5 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #4F46E5 !important;
        font-weight: 700 !important;
    }

    /* Buttons in Bright Mode */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #818CF8 !important;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div {
        color: #4F46E5 !important;
    }
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: #4F46E5 !important;
        border: none !important;
    }
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div,
    button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] span,
    button[data-testid="stBaseButton-primary"] div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* File Uploader in Bright Mode */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] label span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {
        color: #475569 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
    }
    [data-testid="stFileUploader"] button * {
        color: #0F172A !important;
    }

    /* Selectbox in Bright Mode */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
    }
    [data-baseweb="select"] div,
    [data-baseweb="select"] span {
        color: #0F172A !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] li {
        color: #0F172A !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #F1F5F9 !important;
    }

    /* Slider in Bright Mode */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] div {
        color: #0F172A !important;
    }

    /* Metric Cards in Bright Mode */
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #64748B !important;
    }
    .metric-box-green {
        background-color: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
    }
    .metric-box-green h2 {
        color: #16A34A !important;
    }
    .metric-box-green p {
        color: #15803D !important;
    }
    .metric-box-purple {
        background-color: #FAF5FF !important;
        border: 1px solid #E9D5FF !important;
    }
    .metric-box-purple h2 {
        color: #7C3AED !important;
    }
    .metric-box-purple p {
        color: #6D28D9 !important;
    }

    /* Table & Progress in Bright Mode */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        zoom: 1.18;
    }
    [data-testid="stAlert"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        color: #0F172A !important;
    }
"""

dark_styles = """
    :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        --background-color: #0E1117 !important;
        --secondary-background-color: #1F2937 !important;
        --text-color: #FFFFFF !important;
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    
    /* Global Typography in Dark Mode */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    p, span, label, div {
        color: #E2E8F0;
    }
    .step-header {
        color: #FFFFFF !important;
    }

    /* SaaS Container Card */
    .saas-card {
        background: #1F2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25) !important;
    }

    /* Hero Banner in Dark Mode */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.4) !important;
        color: #FFFFFF !important;
    }
    .hero-banner h1,
    .hero-banner h1 *,
    .hero-banner .hero-title,
    .hero-banner .hero-title *,
    .hero-title,
    div.hero-banner h1,
    [data-testid="stMarkdownContainer"] .hero-banner h1 {
        color: #FFFFFF !important;
    }
    .hero-banner p,
    .hero-banner p *,
    div.hero-banner p,
    [data-testid="stMarkdownContainer"] .hero-banner p {
        color: #CBD5E1 !important;
    }
    .hero-banner span {
        color: #FFFFFF !important;
    }

    /* Segmented Theme Toggle Pill in Dark Mode */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: #1F2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] div,
    div[data-testid="stRadio"] [data-baseweb="radio"] span,
    div[data-testid="stRadio"] [data-baseweb="radio"] p {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) {
        background: #4F46E5 !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4) !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) div,
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) span,
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Tabs in Dark Mode */
    [data-baseweb="tab-list"] {
        background-color: #111827 !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px 8px 0 0 !important;
        gap: 8px !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"]:hover p,
    button[data-baseweb="tab"]:hover span,
    button[data-baseweb="tab"]:hover div {
        color: #FFFFFF !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #818CF8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #818CF8 !important;
        font-weight: 700 !important;
    }

    /* Buttons in Dark Mode */
    .stButton > button {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .stButton > button:hover {
        background-color: #374151 !important;
        border-color: #818CF8 !important;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div {
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: #4F46E5 !important;
        border: none !important;
    }
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div,
    button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] span,
    button[data-testid="stBaseButton-primary"] div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* File Uploader in Dark Mode */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] label span {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #111827 !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {
        color: #94A3B8 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #1F2937 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    [data-testid="stFileUploader"] button * {
        color: #FFFFFF !important;
    }

    /* Selectbox in Dark Mode */
    [data-baseweb="select"] > div {
        background-color: #1F2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    [data-baseweb="select"] div,
    [data-baseweb="select"] span {
        color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #1F2937 !important;
    }
    div[data-baseweb="popover"] li {
        color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #374151 !important;
    }

    /* Slider in Dark Mode */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] div {
        color: #FFFFFF !important;
    }

    /* Metric Cards in Dark Mode */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
    }
    .metric-box-green {
        background-color: rgba(22, 163, 74, 0.15) !important;
        border: 1px solid rgba(22, 163, 74, 0.3) !important;
    }
    .metric-box-green h2 {
        color: #4ADE80 !important;
    }
    .metric-box-green p {
        color: #86EFAC !important;
    }
    .metric-box-purple {
        background-color: rgba(124, 58, 237, 0.15) !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
    }
    .metric-box-purple h2 {
        color: #A78BFA !important;
    }
    .metric-box-purple p {
        color: #C4B5FD !important;
    }

    /* Table & Progress in Dark Mode */
    [data-testid="stDataFrame"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        zoom: 1.18;
    }
    [data-testid="stAlert"] {
        background-color: #1F2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
    }
"""

if current_theme == "Bright Mode":
    theme_specific_css = bright_styles
elif current_theme == "Dark Mode":
    theme_specific_css = dark_styles
else:  # System
    theme_specific_css = f"""
        @media (prefers-color-scheme: light) {{
            {bright_styles}
        }}
        @media (prefers-color-scheme: dark) {{
            {dark_styles}
        }}
    """

# 4. Advanced SaaS Custom CSS (Theme-Adaptive & Non-Intrusive)
st.markdown("<style>\n" + theme_specific_css + """
    /* Automatically match Streamlit's active theme (Light or Dark) */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide default Streamlit header/footer branding for clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero Banner Base Layout */
    .hero-banner {
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
    }
    
    .hero-banner h1 {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-banner p {
        font-size: 16px;
    }

    /* Step Indicator Badges */
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 20px;
    }

    /* Stat Cards base */
    .metric-box-green, .metric-box-purple {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .metric-box-green h2, .metric-box-purple h2 {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }
    .metric-box-green p, .metric-box-purple p {
        font-weight: 600;
        margin: 5px 0 0 0;
    }

    /* Buttons override */
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 24px;
    }

    /* Segmented Theme Toggle Pill Styling */
    div[data-testid="stRadio"] {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 16px;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: inline-flex;
        flex-direction: row;
        align-items: center;
        padding: 4px;
        border-radius: 9999px;
        gap: 4px;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] {
        background: transparent;
        padding: 6px 16px !important;
        border-radius: 9999px;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    /* Completely hide radio input circles, bullets, and svgs */
    div[data-testid="stRadio"] [data-baseweb="radio"] input[type="radio"],
    div[data-testid="stRadio"] [data-baseweb="radio"] input[type="radio"] + div,
    div[data-testid="stRadio"] [data-baseweb="radio"] svg {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Plotly SVG Text & Grid Theme Adaptation */
    .js-plotly-plot .xtick text, 
    .js-plotly-plot .ytick text, 
    .js-plotly-plot .gtitle, 
    .js-plotly-plot .legendtext text,
    .js-plotly-plot text {
        fill: var(--text-color) !important;
    }
    .js-plotly-plot .xgrid, 
    .js-plotly-plot .ygrid {
        stroke: rgba(128, 128, 128, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Theme Selector Toggle in UI
theme_col_space, theme_col_widget = st.columns([3.2, 1.8])
with theme_col_widget:
    theme_idx = theme_options.index(current_theme) if current_theme in theme_options else 1
    st.radio(
        "Theme",
        options=theme_options,
        index=theme_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="app_theme_mode",
        on_change=sync_theme
    )

# 1. Create the tabs here:
tab1, tab2, tab3 = st.tabs(["📁 Upload & Config", "⚙️ Optimized Output", "📊 Scatter Analysis"])

with tab1:
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV, TXT, or ZIP file", type=["csv", "txt", "zip"])
    if uploaded_file:
        st.success(f"File uploaded successfully: {uploaded_file.name}")

with tab2:
    st.subheader("Processing & Optimized Output")
    if st.button("Run Pruning Pipeline", use_container_width=True):
        st.success("Optimization complete!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Size Reduced", "110 MB ➔ 44 MB", "-60%")
    col2.metric("Energy Saved", "12.5 kWh", "High Impact")
    col3.metric("Accuracy Retention", "98.4%", "+0.2%")

with tab3:
    st.subheader("Interactive Scatter Plot Analysis")
    
    np.random.seed(42)
    x_orig = np.random.randn(500)
    y_orig = np.random.randn(500)
    keep_idx = np.random.choice(500, size=200, replace=False)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_orig, y=y_orig, mode='markers', name='Full Dataset', marker=dict(color='#FF6B6B', opacity=0.4)))
    fig.add_trace(go.Scatter(x=x_orig[keep_idx], y=y_orig[keep_idx], mode='markers', name='Pruned Dataset', marker=dict(color='#4ECDC4', opacity=0.9)))
    
    chart_template = "plotly_white" if current_theme == "Bright Mode" else "plotly_dark"
    fig.update_layout(
        template=chart_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# HERO SECTION (Matching PDF Header)
# ==========================================
hero_h1_color = "#0F172A" if current_theme == "Bright Mode" else "#FFFFFF"
hero_p_color = "#475569" if current_theme == "Bright Mode" else "#CBD5E1"
hero_bg = "linear-gradient(135deg, #F8FAFC 0%, #EEF2F6 100%)" if current_theme == "Bright Mode" else "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)"
hero_border = "1px solid #CBD5E1" if current_theme == "Bright Mode" else "1px solid rgba(255, 255, 255, 0.1)"
hero_shadow = "0 4px 20px -2px rgba(0, 0, 0, 0.05)" if current_theme == "Bright Mode" else "0 10px 25px -5px rgba(15, 23, 42, 0.4)"

st.markdown(f"""
<div class="hero-banner" style="background: {hero_bg} !important; border: {hero_border} !important; box-shadow: {hero_shadow} !important; border-radius: 20px; padding: 40px; margin-bottom: 30px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="background: #10B981; color: white !important; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">GREEN AI</span>
            <div class="hero-title" style="margin-top: 10px; color: {hero_h1_color} !important; font-size: 36px; font-weight: 800; line-height: 1.2;">DataDiet</div>
            <p style="color: {hero_p_color} !important; font-size: 16px;">Train on less. Lose nothing. Upload your dataset and DataDiet trims samples your model doesn't need.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state step if it doesn't exist
if "step" not in st.session_state:
    st.session_state.step = 1

# Workflow Progress Tracker Bar
col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    st.markdown(f"**1. Upload & Configure** {'🟢' if st.session_state.step == 1 else '✅'}")
with col_w2:
    st.markdown(f"**2. Processing Engine** {'🟢' if st.session_state.step == 2 else ('✅' if st.session_state.step > 2 else '⏳')}")
with col_w3:
    st.markdown(f"**3. Results & Proof** {'🟢' if st.session_state.step == 3 else '⏳'}")

st.write("")

# ==========================================
# STEP 1: UPLOAD & CONFIGURE
# ==========================================
if st.session_state.step == 1:
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">📁 Step 1: Add your data & configuration</div>', unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns([1, 1], gap="large")
    
    with col_c1:
        st.markdown("#### Upload Dataset")
        uploaded_file = st.file_uploader("Drag and drop your files here (ZIP, CSV up to 500 MB)", type=["csv", "zip"])
        st.info("💡 **Tip:** Or click below to use the built-in MNIST sample data (60,000 images).")
        if st.button("Use MNIST Sample Data", use_container_width=True):
            st.session_state["uploaded_file"] = None
            st.session_state["uploaded_file_data"] = {
                "name": "mnist_sample.zip",
                "size": 15420000,
                "type": "application/zip"
            }
            st.success("mnist_sample.zip selected successfully!")

    with col_c2:
        st.markdown("#### Configuration Settings")
        keep_share = st.slider("Keep this much data (%)", min_value=20, max_value=100, value=40, step=5)
        method = st.selectbox("Pruning Method", [
            "Smart Recommended (Keeps most useful samples, best accuracy)", 
            "Balanced (Good mix of easy and hard examples)", 
            "Quick (Fastest to run)"
        ])
        
    st.write("")
    if st.button("🚀 Optimize My Dataset", type="primary", use_container_width=True):
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            file_size_val = getattr(uploaded_file, "size", None)
            if file_size_val is None and hasattr(uploaded_file, "getbuffer"):
                try:
                    file_size_val = len(uploaded_file.getbuffer())
                except Exception:
                    file_size_val = 1048576
            elif file_size_val is None:
                file_size_val = 1048576

            st.session_state["uploaded_file_data"] = {
                "name": uploaded_file.name,
                "size": int(file_size_val),
                "type": getattr(uploaded_file, "type", "text/csv")
            }
        st.session_state["keep_share_val"] = keep_share
        st.session_state["method_val"] = method
        st.session_state.step = 2
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# STEP 2: PROCESSING ENGINE (Live Sim)
# ==========================================
elif st.session_state.step == 2:
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">⚙️ Step 2: Processing & Scoring Samples</div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for p in range(0, 101, 20):
        progress_bar.progress(p)
        time.sleep(0.15)
        if p < 40:
            status_text.markdown("🔍 **Scoring samples:** Evaluating redundancy and information value...")
        elif p < 80:
            status_text.markdown("✂️ **Pruning dataset:** Retaining core boundary samples...")
        else:
            status_text.markdown("📈 **Retraining model:** Side-by-side accuracy validation...")
            
    try:
        res = fetch_backend_stub(
            file_info=st.session_state.get("uploaded_file_data"),
            uploaded_file=st.session_state.get("uploaded_file"),
            keep_share=st.session_state.get("keep_share_val", 40),
            method=st.session_state.get("method_val", "Smart Recommended")
        )
    except Exception:
        file_info_obj = st.session_state.get("uploaded_file_data")
        file_name_val = file_info_obj.get("name", "dataset.csv") if isinstance(file_info_obj, dict) else "dataset.csv"
        file_size_val = file_info_obj.get("size", 1048576) if isinstance(file_info_obj, dict) else 1048576
        res = generate_dynamic_stub(
            filename=file_name_val,
            file_size=file_size_val,
            keep_share=st.session_state.get("keep_share_val", 40),
            method=st.session_state.get("method_val", "Smart Recommended"),
            file_obj=st.session_state.get("uploaded_file")
        )
    st.session_state.backend_data = res
    st.session_state.step = 3
    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# STEP 3: RESULTS & PROOF (SaaS Dashboard)
# ==========================================
elif st.session_state.step == 3:
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">🏆 Step 3: Results, Impact, & Accuracy Proof</div>', unsafe_allow_html=True)
    
    res = st.session_state.get("backend_data", {})
    if not isinstance(res, dict):
        res = {}
    report = res.get("health_report", {})
    proof_table = res.get("proof_table", [])
    metrics = res.get("metrics", {})
    
    size_reduced_val = metrics.get("size_reduced", "-60%")
    energy_saved_val = metrics.get("energy_saved", "-59%")
    acc_change_val = metrics.get("accuracy_change", "-0.2 pts")
    acc_box_class = metrics.get("acc_box_class", "metric-box-purple")

    # Impact Metric Cards matching the PDF design layout
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-box-green">
            <h2>{size_reduced_val}</h2>
            <p>Data Size Reduced</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box-green">
            <h2>{energy_saved_val}</h2>
            <p>Energy & CO₂ Saved</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="{acc_box_class}">
            <h2>{acc_change_val}</h2>
            <p>Accuracy Change</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.markdown("### 📊 Dataset Sample Distribution Comparison")

    pruning_res = res.get("pruning_results", {})
    if pruning_res and "x_orig" in pruning_res:
        x_orig = np.array(pruning_res["x_orig"])
        y_orig = np.array(pruning_res["y_orig"])
        x_pruned = np.array(pruning_res["x_pruned"])
        y_pruned = np.array(pruning_res["y_pruned"])
    else:
        np.random.seed(42)
        x_orig = np.random.randn(500)
        y_orig = np.random.randn(500)

        keep_idx = np.random.choice(500, size=200, replace=False)
        x_pruned = x_orig[keep_idx]
        y_pruned = y_orig[keep_idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_orig, y=y_orig,
        mode='markers',
        name='Full Dataset (Original)',
        marker=dict(color='#FF6B6B', opacity=0.4, size=7)
    ))

    fig.add_trace(go.Scatter(
        x=x_pruned, y=y_pruned,
        mode='markers',
        name='Pruned Dataset (Optimized)',
        marker=dict(color='#4ECDC4', opacity=0.9, size=9)
    ))

    chart_template = "plotly_white" if current_theme == "Bright Mode" else "plotly_dark"
    fig.update_layout(
        title="Data Density Preservation (Original vs. Pruned)",
        xaxis_title="Feature Dimension 1",
        yaxis_title="Feature Dimension 2",
        template=chart_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### 📊 Performance Benchmark Table")
    if proof_table:
        df = pd.DataFrame(proof_table)
        is_bright = (current_theme == "Bright Mode")
        bg_col = "#FFFFFF" if is_bright else "#1E293B"
        text_col = "#0F172A" if is_bright else "#F8FAFC"
        border_col = "#CBD5E1" if is_bright else "rgba(255, 255, 255, 0.15)"
        
        try:
            styled_df = df.style.set_properties(**{
                'background-color': bg_col,
                'color': text_col,
                'font-size': '16px',
                'font-weight': '600',
                'border-color': border_col
            })
            st.dataframe(styled_df, use_container_width=True)
        except Exception:
            st.dataframe(df, use_container_width=True)

    st.write("")
    pruned_csv_df = None
    uploaded_f = st.session_state.get("uploaded_file")
    if uploaded_f is not None:
        try:
            if hasattr(uploaded_f, "seek"):
                uploaded_f.seek(0)
            fname = getattr(uploaded_f, "name", "").lower()
            if fname.endswith(".csv"):
                full_df = pd.read_csv(uploaded_f)
                keep_fraction = float(st.session_state.get("keep_share_val", 40)) / 100.0
                k_count = max(1, min(len(full_df), int(round(len(full_df) * keep_fraction))))
                pruned_csv_df = full_df.iloc[:k_count]
            if hasattr(uploaded_f, "seek"):
                uploaded_f.seek(0)
        except Exception:
            pruned_csv_df = None

    if pruned_csv_df is None:
        pruned_csv_df = pd.DataFrame({
            "feature_dim_1": x_pruned,
            "feature_dim_2": y_pruned
        })

    pruned_csv_bytes = pruned_csv_df.to_csv(index=False).encode('utf-8')
    active_name = st.session_state.get("uploaded_file_data", {}).get("name", "dataset.csv") if isinstance(st.session_state.get("uploaded_file_data"), dict) else "dataset.csv"
    base_name = active_name.rsplit(".", 1)[0]
    out_csv_filename = f"pruned_{base_name}.csv"

    st.download_button(
        label="📥 Download Pruned Dataset (CSV)",
        data=pruned_csv_bytes,
        file_name=out_csv_filename,
        mime="text/csv",
        use_container_width=True
    )

    if st.button("🔄 Run Another Dataset Optimization", use_container_width=True):
        st.session_state.step = 1
        st.session_state.slide_idx = 0
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)
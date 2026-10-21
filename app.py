import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from client import fetch_backend_stub

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
if current_theme == "Bright Mode":
    theme_specific_css = """
        :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            --background-color: #F8FAFC !important;
            --secondary-background-color: #FFFFFF !important;
            --text-color: #0F172A !important;
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        .saas-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            color: #0F172A !important;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
        }
        .step-header {
            color: #0F172A !important;
        }
        [data-baseweb="tab-list"] {
            background-color: #F1F5F9 !important;
        }
        button[data-baseweb="tab"] {
            color: #64748B !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #4F46E5 !important;
        }
        [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        [data-baseweb="select"] * {
            color: #0F172A !important;
        }
        [data-testid="stFileUploader"] section {
            background-color: #F1F5F9 !important;
            border-color: #CBD5E1 !important;
        }
        [data-testid="stFileUploader"] section span, [data-testid="stFileUploader"] section small {
            color: #475569 !important;
        }
        [data-testid="stSlider"] div {
            color: #0F172A !important;
        }
        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
        }
        [data-testid="stMetricValue"] {
            color: #0F172A !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748B !important;
        }
    """
elif current_theme == "Dark Mode":
    theme_specific_css = """
        :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            --background-color: #0E1117 !important;
            --secondary-background-color: #1F2937 !important;
            --text-color: #FFFFFF !important;
            background-color: #0E1117 !important;
            color: #FFFFFF !important;
        }
        .saas-card {
            background: #1F2937 !important;
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25) !important;
        }
        .step-header {
            color: #FFFFFF !important;
        }
        [data-baseweb="tab-list"] {
            background-color: #111827 !important;
        }
        button[data-baseweb="tab"] {
            color: #94A3B8 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #818CF8 !important;
        }
        [data-baseweb="select"] > div {
            background-color: #1F2937 !important;
            color: #FFFFFF !important;
            border-color: rgba(128, 128, 128, 0.3) !important;
        }
        [data-baseweb="select"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stFileUploader"] section {
            background-color: #111827 !important;
            border-color: rgba(128, 128, 128, 0.3) !important;
        }
        [data-testid="stFileUploader"] section span, [data-testid="stFileUploader"] section small {
            color: #94A3B8 !important;
        }
        [data-testid="stSlider"] div {
            color: #FFFFFF !important;
        }
        [data-testid="stDataFrame"] {
            background-color: #1F2937 !important;
        }
        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
        }
    """
else:  # System
    theme_specific_css = """
        @media (prefers-color-scheme: light) {
            :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                --background-color: #F8FAFC !important;
                --secondary-background-color: #FFFFFF !important;
                --text-color: #0F172A !important;
                background-color: #F8FAFC !important;
                color: #0F172A !important;
            }
            .saas-card {
                background: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                color: #0F172A !important;
                box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
            }
            .step-header {
                color: #0F172A !important;
            }
            [data-baseweb="tab-list"] {
                background-color: #F1F5F9 !important;
            }
            button[data-baseweb="tab"] {
                color: #64748B !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #4F46E5 !important;
            }
            [data-baseweb="select"] > div {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border-color: #CBD5E1 !important;
            }
            [data-baseweb="select"] * {
                color: #0F172A !important;
            }
            [data-testid="stFileUploader"] section {
                background-color: #F1F5F9 !important;
                border-color: #CBD5E1 !important;
            }
            [data-testid="stFileUploader"] section span, [data-testid="stFileUploader"] section small {
                color: #475569 !important;
            }
            [data-testid="stSlider"] div {
                color: #0F172A !important;
            }
            [data-testid="stDataFrame"] {
                background-color: #FFFFFF !important;
            }
            [data-testid="stMetricValue"] {
                color: #0F172A !important;
            }
            [data-testid="stMetricLabel"] {
                color: #64748B !important;
            }
        }
        @media (prefers-color-scheme: dark) {
            :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                --background-color: #0E1117 !important;
                --secondary-background-color: #1F2937 !important;
                --text-color: #FFFFFF !important;
                background-color: #0E1117 !important;
                color: #FFFFFF !important;
            }
            .saas-card {
                background: #1F2937 !important;
                border: 1px solid rgba(128, 128, 128, 0.2) !important;
                color: #FFFFFF !important;
                box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25) !important;
            }
            .step-header {
                color: #FFFFFF !important;
            }
            [data-baseweb="tab-list"] {
                background-color: #111827 !important;
            }
            button[data-baseweb="tab"] {
                color: #94A3B8 !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #818CF8 !important;
            }
            [data-baseweb="select"] > div {
                background-color: #1F2937 !important;
                color: #FFFFFF !important;
                border-color: rgba(128, 128, 128, 0.3) !important;
            }
            [data-baseweb="select"] * {
                color: #FFFFFF !important;
            }
            [data-testid="stFileUploader"] section {
                background-color: #111827 !important;
                border-color: rgba(128, 128, 128, 0.3) !important;
            }
            [data-testid="stFileUploader"] section span, [data-testid="stFileUploader"] section small {
                color: #94A3B8 !important;
            }
            [data-testid="stSlider"] div {
                color: #FFFFFF !important;
            }
            [data-testid="stDataFrame"] {
                background-color: #1F2937 !important;
            }
            [data-testid="stMetricValue"] {
                color: #FFFFFF !important;
            }
            [data-testid="stMetricLabel"] {
                color: #94A3B8 !important;
            }
        }
    """

# 4. Advanced SaaS Custom CSS (Theme-Adaptive & Non-Intrusive)
st.markdown("<style>\n" + theme_specific_css + """
    /* Automatically match Streamlit's active theme (Light or Dark) */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Ensure all generic text elements inherit the theme's correct color */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: inherit;
    }
    
    /* Hide default Streamlit header/footer branding for clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container Card */
    .saas-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 20px;
        padding: 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    
    .hero-banner h1 {
        color: white !important;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-banner p {
        color: #94A3B8 !important;
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

    /* Stat Cards matching PDF */
    .metric-box-green {
        background-color: rgba(22, 163, 74, 0.1);
        border: 1px solid rgba(22, 163, 74, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .metric-box-green h2 {
        color: #16A34A !important;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }
    .metric-box-green p {
        color: #15803D !important;
        font-weight: 600;
        margin: 5px 0 0 0;
    }

    .metric-box-purple {
        background-color: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .metric-box-purple h2 {
        color: #7C3AED !important;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }
    .metric-box-purple p {
        color: #6D28D9 !important;
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
        margin-bottom: 12px;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: inline-flex;
        flex-direction: row;
        align-items: center;
        background: rgba(128, 128, 128, 0.12);
        padding: 3px;
        border-radius: 9999px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        gap: 3px;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] {
        background: transparent;
        padding: 5px 14px !important;
        border-radius: 9999px;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:hover {
        background: rgba(128, 128, 128, 0.1);
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) {
        background: #4F46E5 !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.35);
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) * {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
        display: none !important;
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
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="background: #10B981; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">GREEN AI</span>
            <h1 style="margin-top: 10px;">DataDiet</h1>
            <p>Train on less. Lose nothing. Upload your dataset and DataDiet trims samples your model doesn't need.</p>
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
            
    res = fetch_backend_stub()
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
    
    # Impact Metric Cards matching the PDF design layout
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="metric-box-green">
            <h2>-60%</h2>
            <p>Data Size Reduced</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-box-green">
            <h2>-59%</h2>
            <p>Energy & CO₂ Saved</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-box-purple">
            <h2>-0.2 pts</h2>
            <p>Accuracy Change</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.markdown("### 📊 Dataset Sample Distribution Comparison")

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
        st.dataframe(df, use_container_width=True)

    st.write("")
    if st.button("🔄 Run Another Dataset Optimization", use_container_width=True):
        st.session_state.step = 1
        st.session_state.slide_idx = 0
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)
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

# 2. Advanced SaaS Custom CSS (Overriding Streamlit's default look)
st.markdown("""
<style>
    /* Global App Background & Font */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide default Streamlit header/footer branding for clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container Card */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
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
        color: #0F172A;
        font-size: 20px;
        margin-bottom: 20px;
    }

    /* Stat Cards matching PDF */
    .metric-box-green {
        background-color: #F0FDF4;
        border: 1px solid #DCFCE7;
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
        background-color: #F5F3FF;
        border: 1px solid #EDE9FE;
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
</style>
""", unsafe_allow_html=True)

# Right after your closing CSS block:
st.markdown("</style>", unsafe_allow_html=True)

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
    
    import numpy as np
    import plotly.graph_objects as go
    
    np.random.seed(42)
    x_orig = np.random.randn(500)
    y_orig = np.random.randn(500)
    keep_idx = np.random.choice(500, size=200, replace=False)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_orig, y=y_orig, mode='markers', name='Full Dataset', marker=dict(color='#FF6B6B', opacity=0.4)))
    fig.add_trace(go.Scatter(x=x_orig[keep_idx], y=y_orig[keep_idx], mode='markers', name='Pruned Dataset', marker=dict(color='#4ECDC4', opacity=0.9)))
    
    fig.update_layout(template="plotly_white", height=450, margin=dict(l=20, r=20, t=30, b=20))
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
        with col_c1:
            st.markdown("#### Upload Dataset")
            
            # --- REPLACE THIS LINE: ---
            uploaded_file = st.file_uploader("Drag and drop your files here (ZIP, CSV up to 500 MB)", type=["csv", "zip"])
            
            # --- WITH THIS LINE: ---
            # uploaded_file = st.file_uploader("Drag and drop your files here", type=["csv", "zip", "txt"])
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

    # Carousel / Image Preview Section requested by teammate
    import plotly.graph_objects as go
    import numpy as np

    st.markdown("### 📊 Dataset Sample Distribution Comparison")

    # Generate clean demo data points showing high-density vs pruned points
    np.random.seed(42)
    x_orig = np.random.randn(500)
    y_orig = np.random.randn(500)

    # Select a clean subset for "after pruning"
    keep_idx = np.random.choice(500, size=200, replace=False)
    x_pruned = x_orig[keep_idx]
    y_pruned = y_orig[keep_idx]

    # Create subplots / side-by-side view
    fig = go.Figure()

    # Original Data Trace
    fig.add_trace(go.Scatter(
        x=x_orig, y=y_orig,
        mode='markers',
        name='Full Dataset (Original)',
        marker=dict(color='#FF6B6B', opacity=0.4, size=7)
    ))

    # Pruned Data Trace
    fig.add_trace(go.Scatter(
        x=x_pruned, y=y_pruned,
        mode='markers',
        name='Pruned Dataset (Optimized)',
        marker=dict(color='#4ECDC4', opacity=0.9, size=9)
    ))

    fig.update_layout(
        title="Data Density Preservation (Original vs. Pruned)",
        xaxis_title="Feature Dimension 1",
        yaxis_title="Feature Dimension 2",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Benchmark Table
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
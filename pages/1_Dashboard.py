import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import run_pipeline

st.set_page_config(page_title="Market Intelligence Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #FAFAFA; color: #111827; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right:1px solid #E5E7EB; }
    .main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
    [data-testid="stMetric"] { background: #FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:1.25rem !important; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); }
    [data-testid="stMetricValue"] { color:#111827; font-weight:800; }
    [data-testid="stMetricLabel"] { color:#6B7280; font-weight:500; }
    .stButton > button { background: linear-gradient(90deg,#3B82F6,#6366F1); color:white; border:none; border-radius:8px; padding:.6rem 1.5rem; font-weight:600; transition:all .2s ease; box-shadow:0 4px 6px -1px rgba(59,130,246,0.25); }
    .stButton > button:hover { transform:translateY(-2px); box-shadow:0 6px 10px -1px rgba(59,130,246,0.4); color: white; }
    h1 { color:#111827 !important; font-weight:800 !important; }
    h2, h3 { color:#1F2937 !important; font-weight:600 !important; }
    hr { border-color:#E5E7EB; }
    .stPlotlyChart { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Intelligence Controls")
    
    industries = [
        "Finance", "Banking", "Insurance", "Healthcare", "Pharmaceuticals",
        "Biotechnology", "Agriculture", "Food Industry", "Energy", "Oil & Gas",
        "Renewable Energy", "Technology", "Artificial Intelligence", "Cybersecurity",
        "Semiconductors", "Aviation", "Space Industry", "Automotive", "Electric Vehicles",
        "Logistics", "Retail", "E-commerce", "Real Estate", "Construction",
        "Telecommunications", "Media", "Entertainment", "Sports Industry",
        "Defense", "Geopolitics", "Cryptocurrency"
    ]
    
    regions = [
        "United States", "United Kingdom", "India", "Canada", "Australia", "Global"
    ]
    
    selected_industry = st.selectbox("Select Target Industry", industries, index=industries.index("Finance"))
    selected_region = st.selectbox("Select Target Region", regions, index=regions.index("United States"))
    article_count = st.number_input("Article Ingestion Depth", min_value=1, max_value=500, value=50, step=1)
    
    fetch_btn = st.button("🔍 Fetch Intelligence")

# ── Session State Management ──────────────────────────────────────────────────
if "dashboard_data" not in st.session_state:
    st.session_state.dashboard_data = None

if fetch_btn:
    data = run_pipeline(selected_industry, article_count, selected_region)
    if data is not None:
        st.session_state.dashboard_data = data
        st.session_state.selected_industry = selected_industry
        st.session_state.selected_region = selected_region

# ── Main Dashboard UI ─────────────────────────────────────────────────────────
ind = st.session_state.get('selected_industry', selected_industry)
reg = st.session_state.get('selected_region', selected_region)
st.title(f"📊 Market Intelligence: {ind} ({reg})")

if st.session_state.dashboard_data is not None:
    df = st.session_state.dashboard_data
    
    # ── METRIC CARDS ─────────────────────────────────────────────────────────
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    total = len(df)
    pos_count = len(df[df['sentiment'] == 'Positive'])
    neg_count = len(df[df['sentiment'] == 'Negative'])
    neu_count = len(df[df['sentiment'] == 'Neutral'])
    
    m_col1.metric("Total Articles", total)
    m_col2.metric("Positive Sentiment", f"{(pos_count/total)*100:.1f}%")
    m_col3.metric("Negative Sentiment", f"{(neg_count/total)*100:.1f}%", delta=f"-{(neg_count/total)*100:.1f}%", delta_color="inverse")
    m_col4.metric("Neutral Sentiment", f"{(neu_count/total)*100:.1f}%")
    
    st.markdown("---")
    
    # ── VISUALIZATIONS ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Sentiment Distribution")
        fig_bar = px.bar(
            df['sentiment'].value_counts().reset_index(),
            x='sentiment', y='count',
            color='sentiment',
            color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#6366F1"},
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("Sentiment Ratio")
        fig_pie = px.pie(
            df, names='sentiment',
            color='sentiment',
            color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#6366F1"},
            hole=0.4,
            template="plotly_white"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Trend Chart (Mocking time buckets since RSS can have heterogeneous dates)
    st.subheader("Sentiment Confidence Distribution")
    fig_hist = px.histogram(
        df, x="confidence", color="sentiment",
        nbins=20, barmode="overlay",
        color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#6366F1"},
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── RAW DATA ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📰 Raw Intelligence Feed")
    st.dataframe(
        df[['title', 'source', 'published', 'sentiment', 'confidence']], 
        use_container_width=True,
        hide_index=True,
        height=400
    )
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Select an industry in the sidebar and click **🔍 Fetch Intelligence** to populate the dashboard.", icon="📊")
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E5E7EB; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);
        border-radius:16px; padding:4rem 2rem; text-align:center; margin-top:2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
        <h2 style="color:#111827; margin:0 0 0.5rem; font-weight:700;">Market Pulse Ready</h2>
        <p style="color:#6B7280; font-size:1.1rem; max-width:500px; margin:0 auto;">
            Select an industry in the sidebar and fetch the latest intelligence to generate real-time analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

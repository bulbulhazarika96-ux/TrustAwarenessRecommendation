"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Page 6: System & Research Analytics
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

from config import ASSETS_DIR, METRICS_REPORT_PATH
from database.db import get_connection

st.set_page_config(
    page_title="TrustNews AI | System Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #1e293b;">
        <span style="font-size:1.8rem;">🗞️</span>
        <div>
            <div style="font-size:1.2rem;font-weight:900;color:#fff;">Trust<span style="color:#0ea5e9;">News</span> AI</div>
            <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Trust-Aware Recommendation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    user = st.session_state.get("user", {"username": "demo_user"})
    st.markdown(f'<div style="background:#1e293b;border-radius:8px;padding:8px 12px;margin-bottom:12px;font-size:0.82rem;color:#94a3b8;">👤 <b style="color:#e2e8f0;">{user["username"]}</b></div>', unsafe_allow_html=True)
    
    st.markdown("**Navigation**")
    pages = [
        ("🏠", "Home", "/"),
        ("📊", "Dashboard", "Dashboard"),
        ("🔍", "Credibility Inspector", "Fake_News_Detection"),
        ("📰", "Personalized Feed", "Personalized_Feed"),
        ("⚖️", "Ranking Analysis", "Ranking_Analysis"),
        ("📈", "Analytics", "Analytics"),
        ("👤", "Profile", "User_Profile"),
        ("📖", "About", "About"),
    ]
    for icon, label, path in pages:
        st.markdown(f'- {icon} [{label}]({path})')
    
    st.markdown("---")
    st.caption("Assam DownTown University\nB.Tech CSE | 2024–28")

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-brand-header">
    <div>
        <h1 class="tn-brand-title">
            <span>📈</span>
            <span>System & Research <span class="tn-accent">Analytics</span></span>
        </h1>
        <div class="tn-brand-subtitle">
            Direct SQLite Corpus Telemetry &bull; Temporal Engagement Dynamics &bull; Empirical ML Benchmarks
        </div>
    </div>
    <div class="tn-live-pill">
        <span class="tn-live-dot"></span>
        <span>Telemetry Stream Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 1. Fetch Real Database Data via SQL Queries ───────────────────────────────
try:
    with get_connection() as conn:
        source_df = pd.read_sql_query(
            "SELECT source, COUNT(*) as count FROM articles GROUP BY source ORDER BY count DESC LIMIT 15",
            conn,
        )
        cred_df = pd.read_sql_query(
            "SELECT credibility_score FROM articles",
            conn,
        )
        cat_df = pd.read_sql_query(
            "SELECT category, COUNT(*) as count, AVG(credibility_score) as avg_cred FROM articles GROUP BY category ORDER BY count DESC",
            conn,
        )
        activity_df = pd.read_sql_query(
            "SELECT DATE(timestamp) as date, COUNT(*) as reads, AVG(dwell_time_sec) as avg_dwell FROM reading_history GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 30",
            conn,
        )
        top_df = pd.read_sql_query(
            "SELECT title, source, views, likes, credibility_score FROM articles ORDER BY views DESC LIMIT 10",
            conn,
        )
except Exception as e:
    st.error(f"Error querying database: {e}")
    st.stop()

# ── 2. Top Summary KPI Row ────────────────────────────────────────────────────
tot_articles = len(cred_df) if not cred_df.empty else 0
avg_cred_val = cred_df["credibility_score"].mean() if not cred_df.empty else 0.0
tot_publishers = len(source_df) if not source_df.empty else 0
tot_reads = activity_df["reads"].sum() if not activity_df.empty else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num">{tot_articles:,}</div>
        <div class="tn-stat-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#10b981;">{avg_cred_val*100:.1f}%</div>
        <div class="tn-stat-label">Corpus Mean Credibility</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#0ea5e9;">{tot_publishers}</div>
        <div class="tn-stat-label">Active Publishers</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#a855f7;">{tot_reads:,}</div>
        <div class="tn-stat-label">Tracked Reads</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

# ── 3. Charts Row 1: Source Distribution & Credibility Histogram ──────────────
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("#### 🏛️ Top Publisher Distribution")
    if not source_df.empty:
        fig_source = px.bar(
            source_df,
            x="count",
            y="source",
            orientation="h",
            color="count",
            color_continuous_scale=["#0284c7", "#38bdf8"],
            labels={"count": "Article Count", "source": "News Publisher"},
        )
        fig_source.update_layout(
            yaxis=dict(autorange="reversed", gridcolor='#1e293b'),
            xaxis=dict(gridcolor='#1e293b'),
            height=340,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor='#080c14',
            plot_bgcolor='#111827',
            font=dict(color='#cbd5e1', family="Inter"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_source, use_container_width=True)
    else:
        st.info("No publisher distribution records found.")

with col_s2:
    st.markdown("#### 📊 Credibility Score Density & Thresholds")
    if not cred_df.empty:
        fig_cred = px.histogram(
            cred_df,
            x="credibility_score",
            nbins=35,
            labels={"credibility_score": "Composite Credibility Index"},
            color_discrete_sequence=["#10b981"],
        )
        fig_cred.add_vline(x=0.40, line_dash="dash", line_color="#ef4444", annotation_text="Floor (0.40)")
        fig_cred.add_vline(x=0.75, line_dash="dash", line_color="#38bdf8", annotation_text="High Trust (0.75)")
        fig_cred.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor='#080c14',
            plot_bgcolor='#111827',
            font=dict(color='#cbd5e1', family="Inter"),
            xaxis=dict(gridcolor='#1e293b', zeroline=False),
            yaxis=dict(gridcolor='#1e293b', zeroline=False, title="Story Count"),
        )
        st.plotly_chart(fig_cred, use_container_width=True)
    else:
        st.info("No credibility records found.")

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0;'>", unsafe_allow_html=True)

# ── 4. Category Breakdown & Mean Trust ────────────────────────────────────────
st.markdown("#### 🏷️ News Category Breakdown & Mean Trust Index")
if not cat_df.empty:
    fig_cat = px.bar(
        cat_df,
        x="category",
        y="count",
        color="avg_cred",
        color_continuous_scale="Viridis",
        labels={"category": "News Category", "count": "Story Count", "avg_cred": "Avg Trust (0-1)"},
        text=cat_df["avg_cred"].apply(lambda v: f"Avg: {v*100:.1f}%"),
    )
    fig_cat.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor='#080c14',
        plot_bgcolor='#111827',
        font=dict(color='#cbd5e1', family="Inter"),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', title="Total Articles"),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0;'>", unsafe_allow_html=True)

# ── 5. Reading Activity & Dwell Time ──────────────────────────────────────────
st.markdown("#### ⏱️ Reading Activity & Dwell Duration Timeline")
if not activity_df.empty:
    fig_act = go.Figure()
    fig_act.add_trace(go.Bar(
        x=activity_df["date"],
        y=activity_df["reads"],
        name="Articles Read",
        marker_color="#0ea5e9",
        yaxis="y",
    ))
    fig_act.add_trace(go.Scatter(
        x=activity_df["date"],
        y=activity_df["avg_dwell"],
        name="Avg Dwell Time (s)",
        marker_color="#10b981",
        line=dict(width=2),
        yaxis="y2",
    ))
    fig_act.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor='#080c14',
        plot_bgcolor='#111827',
        font=dict(color='#cbd5e1', family="Inter"),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(title="Articles Read", gridcolor='#1e293b'),
        yaxis2=dict(title="Avg Dwell (sec)", overlaying="y", side="right", gridcolor='#1e293b'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_act, use_container_width=True)
else:
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:1.4rem;text-align:center;color:#94a3b8;font-size:0.88rem;">
        📌 No temporal reading activity logged yet. Read articles and submit dwell times to populate reading dynamics.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0;'>", unsafe_allow_html=True)

# ── 6. Top Engagement Articles & Model Benchmarks ─────────────────────────────
col_top_art, col_model_bench = st.columns([1.3, 1.0])

with col_top_art:
    st.markdown("#### 🔥 Top Articles by Views & Engagement")
    if not top_df.empty:
        display_top = top_df.copy()
        display_top["Credibility"] = display_top["credibility_score"].apply(lambda v: f"{v*100:.1f}%")
        display_top = display_top.rename(columns={
            "title": "Article Headline",
            "source": "Publisher",
            "views": "Views",
            "likes": "Likes",
        })[["Article Headline", "Publisher", "Views", "Likes", "Credibility"]]
        st.dataframe(display_top, use_container_width=True, hide_index=True)
    else:
        st.info("No engagement records available.")

with col_model_bench:
    st.markdown("#### 🤖 Machine Learning Model Benchmarks")
    if METRICS_REPORT_PATH.exists():
        try:
            with open(METRICS_REPORT_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            
            best_mod = metrics_data.get("best_model", "Passive Aggressive")
            models_dict = metrics_data.get("models", {})

            st.markdown(f"""
            <div style="background:rgba(14,165,233,0.1);border:1px solid #0ea5e9;border-radius:8px;padding:8px 12px;margin-bottom:12px;font-size:0.82rem;color:#38bdf8;">
                🏆 <b>Top Classifier:</b> {best_mod}
            </div>
            """, unsafe_allow_html=True)

            m_rows = []
            for name, val in models_dict.items():
                m_rows.append({
                    "Model": name,
                    "Accuracy": f"{val.get('accuracy', 0)*100:.2f}%",
                    "F1-Score": f"{val.get('f1_score', 0)*100:.2f}%",
                    "ROC-AUC": f"{val.get('roc_auc', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not parse model metrics: {e}")
    else:
        st.info("ℹ️ Model comparison report (`model_comparison.json`) not found.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Page 1: Analytics & System Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

from config import METRICS_REPORT_PATH, ASSETS_DIR
from database.db import get_database_stats, get_connection

st.set_page_config(
    page_title="TrustNews AI | Analytics Dashboard",
    page_icon="📊",
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
            <span>📊</span>
            <span>Corpus & ML System <span class="tn-accent">Dashboard</span></span>
        </h1>
        <div class="tn-brand-subtitle">
            Corpus Health &bull; Credibility Distributions &bull; Calibrated NLP Model Registry
        </div>
    </div>
    <div class="tn-live-pill">
        <span class="tn-live-dot"></span>
        <span>Telemetry Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Statistics Row ────────────────────────────────────────────────────────
stats = get_database_stats()
total_art = stats.get("total_articles", 0)
real_cnt = stats.get("real_count", 0)
fake_cnt = stats.get("fake_count", 0)
avg_cred = stats.get("avg_credibility", 0.0)

real_pct = (real_cnt / max(1, total_art)) * 100
fake_pct = (fake_cnt / max(1, total_art)) * 100

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num">{total_art:,}</div>
        <div class="tn-stat-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#10b981;">{real_pct:.1f}%</div>
        <div class="tn-stat-label">Verified Authentic ({real_cnt:,})</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#ef4444;">{fake_pct:.1f}%</div>
        <div class="tn-stat-label">Flagged Deceptive ({fake_cnt:,})</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#38bdf8;">{avg_cred * 100:.1f}%</div>
        <div class="tn-stat-label">Mean Trust Score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

# ── Charts Row 1: Category Bar & Real/Fake Donut ──────────────────────────────
col_cat, col_pie = st.columns([1.5, 1.0])

with col_cat:
    st.markdown("#### 📰 Category Distribution")
    try:
        with get_connection() as conn:
            cat_df = pd.read_sql_query(
                "SELECT category, COUNT(*) as count, AVG(credibility_score) as avg_cred FROM articles GROUP BY category ORDER BY count DESC",
                conn,
            )
        if not cat_df.empty:
            fig_cat = px.bar(
                cat_df,
                x="category",
                y="count",
                color="category",
                labels={"category": "News Category", "count": "Article Count"},
                color_discrete_sequence=["#0ea5e9", "#38bdf8", "#10b981", "#6366f1", "#f59e0b", "#ec4899", "#8b5cf6"],
            )
            fig_cat.update_layout(
                showlegend=False,
                height=320,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor='#080c14',
                plot_bgcolor='#111827',
                font=dict(color='#cbd5e1', family="Inter"),
                xaxis=dict(gridcolor='#1e293b', zeroline=False),
                yaxis=dict(gridcolor='#1e293b', zeroline=False),
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No category data found.")
    except Exception as e:
        st.error(f"Error loading category distribution: {e}")

with col_pie:
    st.markdown("#### ⚖️ Authenticity Ratio")
    pie_df = pd.DataFrame({
        "Status": ["Verified Real", "Flagged Deceptive"],
        "Count": [real_cnt, fake_cnt],
    })
    fig_pie = px.pie(
        pie_df,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Verified Real": "#10b981", "Flagged Deceptive": "#ef4444"},
        hole=0.55,
    )
    fig_pie.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='#080c14',
        plot_bgcolor='#111827',
        font=dict(color='#cbd5e1', family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<hr style='border-color:#1e293b;margin:1.5rem 0;'>", unsafe_allow_html=True)

# ── Charts Row 2: Credibility Score Histogram ─────────────────────────────────
st.markdown("#### 📉 Credibility Score Distribution")
try:
    with get_connection() as conn:
        cred_df = pd.read_sql_query("SELECT credibility_score FROM articles", conn)
    
    if not cred_df.empty:
        fig_hist = px.histogram(
            cred_df,
            x="credibility_score",
            nbins=30,
            labels={"credibility_score": "Composite Credibility Score (0.0 = Low, 1.0 = High)"},
            color_discrete_sequence=["#0ea5e9"],
        )
        fig_hist.add_vline(
            x=0.40,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Credibility Floor (0.40)",
            annotation_font_color="#ef4444",
        )
        fig_hist.add_vline(
            x=0.75,
            line_dash="dash",
            line_color="#10b981",
            annotation_text="High Trust Threshold (0.75)",
            annotation_font_color="#10b981",
        )
        fig_hist.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor='#080c14',
            plot_bgcolor='#111827',
            font=dict(color='#cbd5e1', family="Inter"),
            xaxis=dict(gridcolor='#1e293b', zeroline=False),
            yaxis=dict(gridcolor='#1e293b', zeroline=False, title="Story Count"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No credibility records found.")
except Exception as e:
    st.error(f"Error computing credibility histogram: {e}")

st.markdown("<hr style='border-color:#1e293b;margin:1.5rem 0;'>", unsafe_allow_html=True)

# ── Section 3: Model Performance Table & Top Sources ──────────────────────────
col_mod, col_src = st.columns([1.4, 1.0])

with col_mod:
    st.markdown("#### 🤖 Model Performance Registry")
    if METRICS_REPORT_PATH.exists():
        try:
            with open(METRICS_REPORT_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            
            best_model_name = metrics_data.get("best_model", "Passive Aggressive")
            models_dict = metrics_data.get("models", {})
            
            st.markdown(f"""
            <div style="background:rgba(14,165,233,0.1);border:1px solid #0ea5e9;border-radius:8px;padding:8px 12px;margin-bottom:12px;font-size:0.82rem;color:#38bdf8;">
                ⭐ <b>Production Champion:</b> {best_model_name} (Highest F1-Score on Holdout Split)
            </div>
            """, unsafe_allow_html=True)
            
            rows = []
            for name, val in models_dict.items():
                rows.append({
                    "Model": name,
                    "Accuracy": f"{val.get('accuracy', 0)*100:.2f}%",
                    "Precision": f"{val.get('precision', 0)*100:.2f}%",
                    "Recall": f"{val.get('recall', 0)*100:.2f}%",
                    "F1-Score": f"{val.get('f1_score', 0)*100:.2f}%",
                    "ROC-AUC": f"{val.get('roc_auc', 0):.4f}",
                })
            
            mod_table = pd.DataFrame(rows)
            st.dataframe(mod_table, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load model comparison metrics: {e}")
    else:
        st.info("ℹ️ Model comparison report (`model_comparison.json`) not found. Models have not been trained yet.")

with col_src:
    st.markdown("#### 🏛️ Top News Publishers")
    try:
        with get_connection() as conn:
            sources_df = pd.read_sql_query(
                """
                SELECT source as 'Publisher', COUNT(*) as 'Articles',
                       ROUND(AVG(credibility_score)*100, 1) as 'Avg Trust %'
                FROM articles
                GROUP BY source
                ORDER BY COUNT(*) DESC
                LIMIT 10
                """,
                conn,
            )
        st.dataframe(sources_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load source statistics: {e}")

st.markdown("<hr style='border-color:#1e293b;margin:1.5rem 0;'>", unsafe_allow_html=True)

# ── Section 4: Recent System Activity Log ─────────────────────────────────────
st.markdown("#### ⏱️ Recent Reading & Interaction Log")
try:
    with get_connection() as conn:
        activity_df = pd.read_sql_query(
            """
            SELECT h.id as 'Log ID', h.timestamp as 'Timestamp',
                   COALESCE(u.username, 'demo_user') as 'Reader',
                   SUBSTR(a.title, 1, 55) || '...' as 'Story',
                   ROUND(h.dwell_time_sec, 1) as 'Dwell (s)',
                   ROUND(h.completion_rate * 100, 0) || '%' as 'Completion'
            FROM reading_history h
            JOIN articles a ON h.article_id = a.id
            LEFT JOIN users u ON h.user_id = u.id
            ORDER BY h.timestamp DESC
            LIMIT 8
            """,
            conn,
        )
    if not activity_df.empty:
        st.dataframe(activity_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:1.5rem;text-align:center;color:#94a3b8;font-size:0.88rem;">
            📌 No user reading sessions recorded yet. Read stories on the <b>Top Stories</b> feed or <b>Personalized Feed</b> to populate live telemetry.
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not load activity log: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

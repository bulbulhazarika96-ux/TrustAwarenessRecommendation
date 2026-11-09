"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Page 4: Multi-Objective Ranking Analysis & Algorithmic Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import DEFAULT_WEIGHTS, ASSETS_DIR, BEST_MODEL_PATH, TFIDF_VECTORIZER_PATH
from database.db import (
    get_articles,
    get_user_history,
    get_user_feedback_summary,
    get_user_preferences,
)
from src.prediction import get_model_and_vectorizer
from src.recommendation import build_user_profile_vector, compute_content_relevance
from src.ranking import rank_articles, rank_conventional_click
from src.optimization import compute_pareto_fronts
from src.article_intelligence import get_article_trust_metrics, get_trust_badge_html

st.set_page_config(
    page_title="TrustNews AI | Ranking Analysis",
    page_icon="⚖️",
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
            <span>⚖️</span>
            <span>Multi-Objective Ranking <span class="tn-accent">Analysis</span></span>
        </h1>
        <div class="tn-brand-subtitle">
            7-Objective Optimization Benchmark &bull; Trust-Aware vs Conventional Click-Only &bull; Pareto Frontier
        </div>
    </div>
    <div class="tn-live-pill">
        <span class="tn-live-dot"></span>
        <span>Empirical Benchmark</span>
    </div>
</div>
""", unsafe_allow_html=True)

user_id = st.session_state.get("user", {}).get("id", 1)
pref = get_user_preferences(user_id)
preferred_categories = pref.get("categories", ["Politics", "World News", "Technology"])

# ── Weight Configuration Expander with 7 Sliders + Credibility Floor ──────────
with st.expander("🎛️ Tune 7-Objective Weights & Credibility Guardrail", expanded=True):
    st.markdown("""
    Customize the weight vector applied in the multi-objective objective function:
    $$\\text{Score}(u, a) = w_r R + w_c C + w_e E + w_s S + w_d D + w_n N + w_f F$$
    """)
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        w_rel = st.slider("Relevance (R)", 0.0, 1.0, float(DEFAULT_WEIGHTS["relevance"]), 0.05)
        w_cred = st.slider("Credibility (C)", 0.0, 1.0, float(DEFAULT_WEIGHTS["credibility"]), 0.05)
    with w2:
        w_eng = st.slider("Engagement (E)", 0.0, 1.0, float(DEFAULT_WEIGHTS["engagement"]), 0.05)
        w_sat = st.slider("Satisfaction (S)", 0.0, 1.0, float(DEFAULT_WEIGHTS["satisfaction"]), 0.05)
    with w3:
        w_div = st.slider("Diversity (D)", 0.0, 1.0, float(DEFAULT_WEIGHTS["diversity"]), 0.05)
        w_nov = st.slider("Novelty (N)", 0.0, 1.0, float(DEFAULT_WEIGHTS["novelty"]), 0.05)
    with w4:
        w_fair = st.slider("Fairness (F)", 0.0, 1.0, float(DEFAULT_WEIGHTS["fairness"]), 0.05)
        cred_floor = st.slider("Credibility Floor", 0.10, 0.70, 0.35, 0.05)

    custom_weights = {
        "relevance": w_rel,
        "credibility": w_cred,
        "engagement": w_eng,
        "satisfaction": w_sat,
        "diversity": w_div,
        "novelty": w_nov,
        "fairness": w_fair,
    }

# ── Fetch & Compute Rankings ──────────────────────────────────────────────────
candidate_articles = get_articles(limit=60)
if not candidate_articles:
    st.info("ℹ️ No articles found in the database. Please initialize the database.")
    st.stop()

vectorizer = None
if BEST_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists():
    try:
        _, vectorizer = get_model_and_vectorizer()
    except Exception:
        pass

if vectorizer is not None:
    try:
        user_vec = build_user_profile_vector(user_id, vectorizer, preferred_categories)
        relevance_scores = compute_content_relevance(user_vec, candidate_articles, vectorizer, preferred_categories)
    except Exception:
        relevance_scores = [0.8 if a.get("category") in preferred_categories else 0.4 for a in candidate_articles]
else:
    relevance_scores = [0.8 if a.get("category") in preferred_categories else 0.4 for a in candidate_articles]

user_history = get_user_history(user_id, limit=20)
user_feedback = get_user_feedback_summary(user_id)

trust_ranked = rank_articles(
    articles=candidate_articles,
    relevance_scores=relevance_scores,
    user_history=user_history,
    user_feedback=user_feedback,
    weights=custom_weights,
    preferred_categories=preferred_categories,
    credibility_floor=cred_floor,
)

click_ranked = rank_conventional_click(trust_ranked)
pareto_analyzed = compute_pareto_fronts(trust_ranked)

# ── Side-by-Side Algorithm Comparison (Top 10 Stories) ────────────────────────
st.markdown("### ⚔️ Side-by-Side Algorithmic Comparison (Top 10 Ranked Stories)")

top_k = 10
trust_top = trust_ranked[:top_k]
click_top = click_ranked[:top_k]

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"""
    <div style="background:rgba(14,165,233,0.1);border:1px solid #0ea5e9;border-radius:10px;padding:12px 16px;margin-bottom:14px;">
        <div style="font-size:1.05rem;font-weight:800;color:#38bdf8;">🛡️ Proposed: Trust-Aware Pareto Ranking</div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;">
            Optimizes for credibility, factual veracity, sustained dwell satisfaction, and exposure fairness.
        </div>
    </div>
    """, unsafe_allow_html=True)

    for item in trust_top:
        tm = get_article_trust_metrics(item)
        badge = get_trust_badge_html(item)
        score_val = float(item.get("trust_score", 0.0))
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px 14px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.75rem;margin-bottom:6px;">
                <span style="color:#0ea5e9;font-weight:900;font-size:0.85rem;">Rank #{item.get('trust_rank')} &bull; Score: {score_val:.3f}</span>
                {badge}
            </div>
            <div style="font-size:0.92rem;font-weight:700;color:#f8fafc;line-height:1.35;margin-bottom:6px;">
                {item['title']}
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#64748b;border-top:1px solid #1e293b;padding-top:6px;">
                <span>📰 <b style="color:#cbd5e1;">{item['source']}</b></span>
                <span>Match: <b style="color:#38bdf8;">{round(float(item.get('relevance_score', 0.5))*100)}%</b></span>
                <span>Trust: <b style="color:{tm['tier_color']};">{tm['cred_pct']}%</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.1);border:1px solid #ef4444;border-radius:10px;padding:12px 16px;margin-bottom:14px;">
        <div style="font-size:1.05rem;font-weight:800;color:#f87171;">🖱️ Baseline: Conventional Click-Optimized Ranking</div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;">
            Optimizes solely for viral click velocity, sensational trigger tokens, and raw impressions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    for item in click_top:
        tm = get_article_trust_metrics(item)
        badge = get_trust_badge_html(item)
        click_val = float(item.get("conventional_click_score", 0.0))
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px 14px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.75rem;margin-bottom:6px;">
                <span style="color:#ef4444;font-weight:900;font-size:0.85rem;">Rank #{item.get('click_rank')} &bull; CTR Score: {click_val:.3f}</span>
                {badge}
            </div>
            <div style="font-size:0.92rem;font-weight:700;color:#f8fafc;line-height:1.35;margin-bottom:6px;">
                {item['title']}
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#64748b;border-top:1px solid #1e293b;padding-top:6px;">
                <span>📰 <b style="color:#cbd5e1;">{item['source']}</b></span>
                <span>Impressions: <b style="color:#cbd5e1;">{item.get('views', 10):,}</b></span>
                <span>Trust: <b style="color:{tm['tier_color']};">{tm['cred_pct']}%</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0;'>", unsafe_allow_html=True)

# ── Visualizations: Radar Chart & Pareto Scatter Plot ─────────────────────────
col_radar, col_pareto = st.columns(2)

objectives_labels = ["Relevance", "Credibility", "Engagement", "Satisfaction", "Diversity", "Novelty", "Fairness"]

trust_means = [
    float(np.mean([it.get("relevance_score", 0.5) for it in trust_top])),
    float(np.mean([it.get("credibility_score", 0.5) for it in trust_top])),
    float(np.mean([it.get("engagement_score", 0.5) for it in trust_top])),
    float(np.mean([it.get("satisfaction_score", 0.5) for it in trust_top])),
    float(np.mean([it.get("diversity_score", 0.5) for it in trust_top])),
    float(np.mean([it.get("novelty_score", 0.5) for it in trust_top])),
    float(np.mean([it.get("fairness_score", 0.5) for it in trust_top])),
]

click_means = [
    float(np.mean([it.get("relevance_score", 0.5) for it in click_top])),
    float(np.mean([it.get("credibility_score", 0.5) for it in click_top])),
    float(np.mean([it.get("engagement_score", 0.5) for it in click_top])),
    float(np.mean([it.get("satisfaction_score", 0.5) for it in click_top])),
    float(np.mean([it.get("diversity_score", 0.5) for it in click_top])),
    float(np.mean([it.get("novelty_score", 0.5) for it in click_top])),
    float(np.mean([it.get("fairness_score", 0.5) for it in click_top])),
]

with col_radar:
    st.markdown("#### 🕸️ 7-Objective Trade-Off Radar")
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=trust_means + [trust_means[0]],
        theta=objectives_labels + [objectives_labels[0]],
        fill='toself',
        name='Trust-Aware (Proposed)',
        line=dict(color='#0ea5e9', width=2),
        fillcolor='rgba(14,165,233,0.25)',
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=click_means + [click_means[0]],
        theta=objectives_labels + [objectives_labels[0]],
        fill='toself',
        name='Click-Optimized (Baseline)',
        line=dict(color='#ef4444', width=2),
        fillcolor='rgba(239,68,68,0.25)',
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1e293b', tickfont=dict(color='#64748b')),
            angularaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#cbd5e1')),
            bgcolor='#111827',
        ),
        paper_bgcolor='#080c14',
        plot_bgcolor='#111827',
        font=dict(color='#cbd5e1', family="Inter"),
        height=360,
        margin=dict(l=30, r=30, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_pareto:
    st.markdown("#### 🎯 Pareto Frontier (Credibility vs Relevance)")
    scatter_rows = []
    for a in pareto_analyzed:
        scatter_rows.append({
            "Title": a["title"][:45] + "...",
            "Credibility": a.get("credibility_score", 0.5),
            "Relevance": a.get("relevance_score", 0.5),
            "Satisfaction": max(6, int(a.get("satisfaction_score", 0.5) * 15)),
            "Status": "⭐ Pareto Optimal (Front 0)" if a.get("is_pareto_optimal") else f"Dominated (Front {a.get('pareto_front', 1)})",
        })
    scatter_df = pd.DataFrame(scatter_rows)

    fig_scatter = px.scatter(
        scatter_df,
        x="Relevance",
        y="Credibility",
        color="Status",
        size="Satisfaction",
        hover_name="Title",
        color_discrete_map={
            "⭐ Pareto Optimal (Front 0)": "#10b981",
            "Dominated (Front 1)": "#0ea5e9",
            "Dominated (Front 2)": "#f59e0b",
        },
    )
    fig_scatter.add_hline(y=cred_floor, line_dash="dash", line_color="#ef4444", annotation_text="Credibility Floor")
    fig_scatter.update_layout(
        paper_bgcolor='#080c14',
        plot_bgcolor='#111827',
        font=dict(color='#cbd5e1', family="Inter"),
        xaxis=dict(gridcolor='#1e293b', range=[0, 1.05], zeroline=False),
        yaxis=dict(gridcolor='#1e293b', range=[0, 1.05], zeroline=False),
        height=360,
        margin=dict(l=20, r=20, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0;'>", unsafe_allow_html=True)

# ── Quantitative Comparison Metrics Table ─────────────────────────────────────
st.markdown("#### 📋 Quantitative Performance Comparison (Top 10 Benchmark)")

trust_fake_count = sum(1 for a in trust_top if a.get("label") == "FAKE" or float(a.get("credibility_score", 0.5)) < 0.40)
click_fake_count = sum(1 for a in click_top if a.get("label") == "FAKE" or float(a.get("credibility_score", 0.5)) < 0.40)

comp_df = pd.DataFrame([
    {
        "Evaluation Metric": "Mean Credibility Score (C)",
        "Trust-Aware Pareto Ranking": f"{trust_means[1]:.3f}",
        "Conventional Click-Only": f"{click_means[1]:.3f}",
        "Improvement (Delta)": f"+{(trust_means[1] - click_means[1])*100:.1f}%",
    },
    {
        "Evaluation Metric": "Deceptive / Fake News Exposure Rate",
        "Trust-Aware Pareto Ranking": f"{trust_fake_count * 10.0:.1f}% ({trust_fake_count}/{top_k})",
        "Conventional Click-Only": f"{click_fake_count * 10.0:.1f}% ({click_fake_count}/{top_k})",
        "Improvement (Delta)": f"-{(click_fake_count - trust_fake_count)*10.0:.1f}% reduction",
    },
    {
        "Evaluation Metric": "Mean Content Relevance (R)",
        "Trust-Aware Pareto Ranking": f"{trust_means[0]:.3f}",
        "Conventional Click-Only": f"{click_means[0]:.3f}",
        "Improvement (Delta)": f"{(trust_means[0] - click_means[0])*100:+.1f}%",
    },
    {
        "Evaluation Metric": "Content Diversity Score (D)",
        "Trust-Aware Pareto Ranking": f"{trust_means[4]:.3f}",
        "Conventional Click-Only": f"{click_means[4]:.3f}",
        "Improvement (Delta)": f"+{(trust_means[4] - click_means[4])*100:.1f}%",
    },
    {
        "Evaluation Metric": "Long-Term Engagement Index (E)",
        "Trust-Aware Pareto Ranking": f"{trust_means[2]:.3f}",
        "Conventional Click-Only": f"{click_means[2]:.3f}",
        "Improvement (Delta)": f"+{(trust_means[2] - click_means[2])*100:.1f}%",
    },
    {
        "Evaluation Metric": "Exposure Fairness Index (F)",
        "Trust-Aware Pareto Ranking": f"{trust_means[6]:.3f}",
        "Conventional Click-Only": f"{click_means[6]:.3f}",
        "Improvement (Delta)": f"+{(trust_means[6] - click_means[6])*100:.1f}%",
    },
])

st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

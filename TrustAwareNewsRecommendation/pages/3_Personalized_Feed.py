"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Page 3: Personalized Feed with Multi-Objective Re-ranking
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from config import CATEGORIES, ASSETS_DIR, BEST_MODEL_PATH, TFIDF_VECTORIZER_PATH
from database.db import (
    get_articles,
    get_user_preferences,
    get_user_history,
    get_user_feedback_summary,
    record_feedback,
    record_reading_history,
)
from src.prediction import get_model_and_vectorizer
from src.recommendation import build_user_profile_vector, compute_content_relevance
from src.ranking import rank_articles
from src.optimization import constrained_rerank
from src.explanation import generate_recommendation_explanation
from src.article_intelligence import (
    get_article_image_url,
    get_article_trust_metrics,
    get_trust_badge_html,
    get_user_reading_stats,
)

st.set_page_config(
    page_title="TrustNews AI | Personalized Feed",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

user_id = st.session_state.get("user", {}).get("id", 1)
username = st.session_state.get("user", {}).get("username", "demo_user")

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
            <span>📰</span>
            <span>Personalized <span class="tn-accent">Trust Feed</span></span>
        </h1>
        <div class="tn-brand-subtitle">
            7-Objective Pareto Ranking &bull; Credibility Guardrails &bull; Transparent Explainability
        </div>
    </div>
    <div class="tn-live-pill">
        <span class="tn-live-dot"></span>
        <span>Recommender Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Reading Stats Card at Top ─────────────────────────────────────────────────
user_history = get_user_history(user_id, limit=30)
reading_stats = get_user_reading_stats(user_id, user_history)

st.markdown(f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1.6rem;box-shadow:0 4px 14px rgba(0,0,0,0.3);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <span style="font-size:1.05rem;font-weight:800;color:#f8fafc;display:flex;align-items:center;gap:8px;">
            👤 Reading & Engagement Profile &bull; <b style="color:#0ea5e9;">{username}</b>
        </span>
        <span style="background:rgba(14,165,233,0.15);border:1px solid #0ea5e9;color:#38bdf8;font-size:0.75rem;font-weight:700;padding:4px 10px;border-radius:999px;">
            Tier: {reading_stats['engagement_level']}
        </span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;">
        <div style="background:#080c14;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;">
            <div style="font-size:0.72rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Articles Read</div>
            <div style="font-size:1.4rem;font-weight:900;color:#0ea5e9;margin-top:2px;">{reading_stats['total_articles']}</div>
        </div>
        <div style="background:#080c14;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;">
            <div style="font-size:0.72rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Avg Dwell Time</div>
            <div style="font-size:1.4rem;font-weight:900;color:#10b981;margin-top:2px;">{reading_stats['avg_dwell_time']}s</div>
        </div>
        <div style="background:#080c14;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;">
            <div style="font-size:0.72rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Completion Rate</div>
            <div style="font-size:1.4rem;font-weight:900;color:#f59e0b;margin-top:2px;">{reading_stats['avg_completion']}%</div>
        </div>
        <div style="background:#080c14;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;">
            <div style="font-size:0.72rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Engagement Status</div>
            <div style="font-size:1.4rem;font-weight:900;color:#a855f7;margin-top:2px;">{reading_stats['engagement_level']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load User Preferences ─────────────────────────────────────────────────────
user_pref = get_user_preferences(user_id)
saved_categories = user_pref.get("categories", ["Politics", "World News", "Technology"])
user_weights = user_pref.get("weights", {})

# ── Two-Column Layout: Left Controls | Right Ranked Feed ──────────────────────
col_controls, col_feed = st.columns([1.1, 2.7])

with col_controls:
    st.markdown("#### ⚙️ Recommender Controls")
    
    pref_cats = st.multiselect(
        "Preferred Categories:",
        options=CATEGORIES,
        default=[c for c in saved_categories if c in CATEGORIES] or ["Politics", "World News"],
        help="Select topics that match your current informational interests.",
    )
    
    min_cred = st.slider(
        "Minimum Credibility Floor:",
        min_value=0.20,
        max_value=0.80,
        value=0.40,
        step=0.05,
        help="Articles scoring below this credibility floor are filtered out by guardrails.",
    )
    
    sort_strategy = st.selectbox(
        "Optimization Strategy:",
        [
            "7-Objective Trust-Aware (Balanced)",
            "High Credibility Priority (≥ 0.70)",
            "Deep Reading / Engagement",
            "Broad Topic Diversification",
        ],
    )
    
    feed_limit = st.slider("Max Feed Stories:", min_value=5, max_value=25, value=12, step=1)
    
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px;margin-top:14px;font-size:0.75rem;color:#94a3b8;line-height:1.5;">
        <b>7 Ranking Objectives:</b><br>
        &bull; <b>R:</b> Relevance (TF-IDF Cosine)<br>
        &bull; <b>C:</b> Credibility (NLP Veracity + Domain)<br>
        &bull; <b>E:</b> Long-Term Engagement<br>
        &bull; <b>S:</b> User Satisfaction<br>
        &bull; <b>D:</b> Content Diversity<br>
        &bull; <b>N:</b> Information Novelty<br>
        &bull; <b>F:</b> Publisher Fairness
    </div>
    """, unsafe_allow_html=True)

with col_feed:
    # Handle missing model files gracefully
    vectorizer = None
    if BEST_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists():
        try:
            _, vectorizer = get_model_and_vectorizer()
        except Exception as e:
            st.warning(f"Note: Vectorizer loaded with fallback. ({e})")

    # Fetch candidate articles from database
    candidates = get_articles(limit=80)
    if not candidates:
        st.info("ℹ️ No candidate articles available in the database. Please initialize or seed the database.")
        st.stop()

    user_feedback = get_user_feedback_summary(user_id)

    # Compute relevance scores (with cold-start handling)
    if vectorizer is not None:
        try:
            user_vec = build_user_profile_vector(user_id, vectorizer, preferred_categories=pref_cats)
            relevance_scores = compute_content_relevance(user_vec, candidates, vectorizer, preferred_categories=pref_cats)
        except Exception:
            relevance_scores = [0.8 if a.get("category") in pref_cats else 0.4 for a in candidates]
    else:
        # Fallback cold-start category matching
        relevance_scores = [0.8 if a.get("category") in pref_cats else 0.4 for a in candidates]

    # Execute 7-objective multi-criteria ranking
    ranked_pool = rank_articles(
        articles=candidates,
        relevance_scores=relevance_scores,
        user_history=user_history,
        user_feedback=user_feedback,
        weights=user_weights if user_weights else None,
        preferred_categories=pref_cats,
        credibility_floor=min_cred,
    )

    # Apply constrained re-ranking with credibility guardrails & MMR diversity
    reranked = constrained_rerank(
        ranked_articles=ranked_pool,
        top_k=feed_limit * 2,
        credibility_threshold=min_cred,
        max_items_per_source=3,
        diversity_lambda=0.65,
    )

    # Strategy-based post-filtering
    if sort_strategy == "High Credibility Priority (≥ 0.70)":
        final_feed = [a for a in reranked if float(a.get("credibility_score", 0.5)) >= 0.70]
    elif sort_strategy == "Deep Reading / Engagement":
        final_feed = sorted(reranked, key=lambda x: x.get("engagement_score", 0.5), reverse=True)
    elif sort_strategy == "Broad Topic Diversification":
        final_feed = sorted(reranked, key=lambda x: x.get("diversity_score", 0.5), reverse=True)
    else:
        final_feed = reranked

    final_feed = final_feed[:feed_limit]

    if not final_feed:
        st.info(f"No stories met the minimum credibility threshold ({min_cred:.2f}) or strategy filters. Try lowering the floor.")
    else:
        st.caption(f"Showing **{len(final_feed)}** personalized stories &bull; Optimization: **{sort_strategy}**")

        liked_set = set(user_feedback.get("like", []))
        disliked_set = set(user_feedback.get("dislike", []))
        saved_set = set(user_feedback.get("save", []))

        for idx, item in enumerate(final_feed, 1):
            art_id = item["id"]
            tm = get_article_trust_metrics(item)
            cred_pct = tm["cred_pct"]
            rel_score = float(item.get("relevance_score", 0.5))
            trust_score = float(item.get("trust_score", 0.5))

            # 7 score chips
            r_val = float(item.get("relevance_score", 0.5))
            c_val = float(item.get("credibility_score", 0.5))
            e_val = float(item.get("engagement_score", 0.5))
            s_val = float(item.get("satisfaction_score", 0.5))
            d_val = float(item.get("diversity_score", 0.5))
            n_val = float(item.get("novelty_score", 0.5))
            f_val = float(item.get("fairness_score", 0.5))

            fill_class = (
                "tn-cred-fill-high" if tm["tier"] == "high"
                else "tn-cred-fill-mixed" if tm["tier"] == "mixed"
                else "tn-cred-fill-low"
            )

            # Story Card with Two-Column: Image | Content
            col_img, col_txt = st.columns([1, 2.3])
            with col_img:
                st.markdown(f"""
                <div style="position:relative;width:100%;height:220px;border-radius:12px;overflow:hidden;border:1px solid #1e293b;background:#0f172a;">
                    <img src="{tm['image_url']}" style="width:100%;height:100%;object-fit:cover;" alt="{item['title']}" />
                    <div style="position:absolute;top:8px;left:8px;background:rgba(15,23,42,0.92);border:1px solid rgba(255,255,255,0.15);color:#f8fafc;font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:4px;text-transform:uppercase;">
                        {item['category']}
                    </div>
                    <div style="position:absolute;bottom:8px;right:8px;background:rgba(15,23,42,0.92);color:#94a3b8;font-size:0.65rem;font-weight:600;padding:2px 6px;border-radius:4px;">
                        ⏱️ {tm['read_time_mins']}m read
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_txt:
                st.markdown(f"""
                <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:14px 18px;min-height:220px;display:flex;flex-direction:column;justify-content:space-between;">
                    <div>
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
                            <div style="display:flex;align-items:center;gap:6px;">
                                {get_trust_badge_html(item)}
                                <span style="background:rgba(14,165,233,0.15);border:1px solid #0ea5e9;color:#38bdf8;padding:2px 8px;border-radius:6px;font-size:0.70rem;font-weight:700;">
                                    🎯 {round(rel_score * 100)}% Match
                                </span>
                            </div>
                            <span style="background:#1e293b;border:1px solid #334155;color:#cbd5e1;font-size:0.72rem;font-weight:600;padding:3px 8px;border-radius:5px;">
                                📰 {item['source']}
                            </span>
                        </div>
                        <div style="font-size:1.1rem;font-weight:800;color:#ffffff;line-height:1.35;margin-bottom:5px;">
                            {item['title']}
                        </div>
                        <div style="font-size:0.84rem;color:#94a3b8;line-height:1.45;margin-bottom:6px;">
                            {item['summary'][:160]}...
                        </div>
                    </div>
                    <div>
                        <!-- 7 Objective Score Chips -->
                        <div class="tn-chips-row">
                            <span class="tn-chip">R: <b>{r_val:.2f}</b></span>
                            <span class="tn-chip">C: <b>{c_val:.2f}</b></span>
                            <span class="tn-chip">E: <b>{e_val:.2f}</b></span>
                            <span class="tn-chip">S: <b>{s_val:.2f}</b></span>
                            <span class="tn-chip">D: <b>{d_val:.2f}</b></span>
                            <span class="tn-chip">N: <b>{n_val:.2f}</b></span>
                            <span class="tn-chip">F: <b>{f_val:.2f}</b></span>
                            <span class="tn-chip" style="border-color:#0ea5e9;color:#38bdf8;">TrustRank: <b>{trust_score:.3f}</b></span>
                        </div>
                        <!-- Credibility Bar -->
                        <div class="tn-cred-bar-wrap">
                            <div class="tn-cred-bar-meta">
                                <span>Credibility Score</span>
                                <span style="color:{tm['tier_color']};">{cred_pct}%</span>
                            </div>
                            <div class="tn-cred-track">
                                <div class="tn-cred-fill {fill_class}" style="width: {cred_pct}%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Action Buttons Row
            b_like, b_dislike, b_save, b_why = st.columns([1, 1, 1, 2.5])
            with b_like:
                is_liked = art_id in liked_set
                if st.button("❤️ Liked" if is_liked else "👍 Like", key=f"feed_like_{art_id}", disabled=is_liked):
                    record_feedback(user_id, art_id, "like")
                    st.rerun()
            with b_dislike:
                is_disliked = art_id in disliked_set
                if st.button("👎 Disliked" if is_disliked else "👎 Dislike", key=f"feed_dislike_{art_id}", disabled=is_disliked):
                    record_feedback(user_id, art_id, "dislike")
                    st.rerun()
            with b_save:
                is_saved = art_id in saved_set
                if st.button("🔖 Saved" if is_saved else "🔖 Save", key=f"feed_save_{art_id}", disabled=is_saved):
                    record_feedback(user_id, art_id, "save")
                    st.rerun()
            with b_why:
                with st.popover("💡 Why Recommended?"):
                    exp = generate_recommendation_explanation(item)
                    st.markdown(f"**{exp.get('summary', 'Curated for you.')}**")
                    st.markdown(f"- **Relevance Signal:** {exp.get('relevance_detail', '')}")
                    st.markdown(f"- **Credibility Detail:** {exp.get('credibility_detail', '')}")
                    st.markdown(f"- **Uncertainty:** {exp.get('uncertainty_level', 'Low')}")
                    if exp.get("top_drivers"):
                        st.markdown(f"- **Primary Drivers:** {', '.join(exp['top_drivers'])}")

            # Read More Expander with Dwell Time Logger
            with st.expander(f"📖 Read Full Article & Log Dwell Time (ID: #{art_id})"):
                st.markdown(f"### {item['title']}")
                st.caption(f"Source: {item['source']} &bull; Category: {item['category']} &bull; Credibility: {tm['cred_pct']}%")
                st.write(item.get("text", item["summary"]))
                st.markdown("<hr style='border-color:#1e293b;margin:10px 0;'>", unsafe_allow_html=True)
                dwell_input = st.slider("Dwell Time (seconds):", 5, 180, 45, key=f"dwell_slider_{art_id}")
                if st.button("⏱️ Log Dwell & Engagement", key=f"log_dwell_btn_{art_id}"):
                    w_cnt = len(item.get("text", "").split())
                    comp = min(1.0, dwell_input / max(10.0, w_cnt / 3.5))
                    record_reading_history(user_id, art_id, dwell_input, comp, accidental=1 if dwell_input < 6 else 0)
                    st.success(f"Recorded {dwell_input}s reading interaction ({comp*100:.0f}% completion).")
                    st.rerun()

            st.markdown("<hr style='border-color:#1e293b;margin:14px 0 18px 0;'>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Home: Top Stories Feed with Trust Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np

from config import ASSETS_DIR, CATEGORIES
from database.db import (
    init_db,
    get_database_stats,
    get_articles,
    record_feedback,
    record_reading_history,
)
from src.article_intelligence import (
    get_article_image_url,
    get_article_trust_metrics,
    get_trust_badge_html,
)
from src.recommendation import get_trending_articles

st.set_page_config(
    page_title="TrustNews AI | Top Stories",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_db()

# Session State for User Authentication
if "user" not in st.session_state:
    st.session_state["user"] = {"id": 1, "username": "demo_user"}
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True

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

# ── Brand Header with Animated Live Indicator ──────────────────────────────────
st.markdown("""
<div class="tn-brand-header">
    <div>
        <h1 class="tn-brand-title">
            <span>🗞️</span>
            <span>Trust<span class="tn-accent">News</span> AI</span>
        </h1>
        <div class="tn-brand-subtitle">
            Algorithmic Truth-Aware News Intelligence &bull; Google News / Ground News Precision
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:14px;">
        <div class="tn-live-pill">
            <span class="tn-live-dot"></span>
            <span>Live Trust Evaluation</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Statistics Row ────────────────────────────────────────────────────────
stats = get_database_stats()
total_art = stats.get("total_articles", 0)
real_cnt = stats.get("real_count", 0)
fake_cnt = stats.get("fake_count", 0)
avg_cred = stats.get("avg_credibility", 0.0)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num">{total_art:,}</div>
        <div class="tn-stat-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#10b981;">{real_cnt:,}</div>
        <div class="tn-stat-label">Verified Authentic</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#ef4444;">{fake_cnt:,}</div>
        <div class="tn-stat-label">Flagged Deceptive</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="tn-stat-card">
        <div class="tn-stat-num" style="color:#38bdf8;">{avg_cred * 100:.1f}%</div>
        <div class="tn-stat-label">System Trust Score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# Fetch Articles Gracefully
all_articles = get_articles(limit=100)
if not all_articles:
    st.info("ℹ️ No articles found in the database. Please initialize or seed the database.")
    st.markdown("""
    <div class="tn-footer">
        Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Enrich articles with trust metrics
for art in all_articles:
    art["tm"] = get_article_trust_metrics(art)

# ── HERO STORY CARD ───────────────────────────────────────────────────────────
hero_article = all_articles[0]
hero_tm = hero_article["tm"]
hero_cred_pct = hero_tm["cred_pct"]
hero_fill_class = (
    "tn-cred-fill-high" if hero_tm["tier"] == "high"
    else "tn-cred-fill-mixed" if hero_tm["tier"] == "mixed"
    else "tn-cred-fill-low"
)

st.markdown(f"""
<div class="tn-hero-card">
    <div class="tn-hero-img-wrap">
        <img src="{hero_tm['image_url']}" alt="{hero_article['title']}" />
    </div>
    <div class="tn-hero-content">
        <div>
            <div class="tn-hero-badge-row">
                <span class="tn-badge-category">⭐ LEAD STORY &bull; {hero_article['category']}</span>
                <span class="tn-badge-source">📰 {hero_article['source']}</span>
                {get_trust_badge_html(hero_article)}
            </div>
            <div class="tn-hero-title">{hero_article['title']}</div>
            <div class="tn-hero-summary">{hero_article['summary'][:240]}...</div>
            <div class="tn-cred-bar-wrap">
                <div class="tn-cred-bar-meta">
                    <span>Credibility Verification Meter</span>
                    <span style="color:{hero_tm['tier_color']};">{hero_cred_pct}% &bull; {hero_tm['tier_label']}</span>
                </div>
                <div class="tn-cred-track">
                    <div class="tn-cred-fill {hero_fill_class}" style="width: {hero_cred_pct}%;"></div>
                </div>
            </div>
        </div>
        <div class="tn-hero-meta">
            <span>⏱️ {hero_tm['read_time_mins']} min read &bull; 👁️ {hero_article.get('views', 0)} views</span>
            <span>Published: {hero_article.get('publish_date', 'Recent')}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander(f"📖 Read Full Lead Story (ID: #{hero_article['id']})"):
    st.write(hero_article.get("text", hero_article["summary"]))
    h_col1, h_col2 = st.columns([1, 1])
    with h_col1:
        if st.button("👍 Like Lead Story", key=f"hero_like_{hero_article['id']}"):
            record_feedback(st.session_state["user"]["id"], hero_article["id"], "like")
            st.success("Liked story!")
    with h_col2:
        if st.button("🔖 Save Lead Story", key=f"hero_save_{hero_article['id']}"):
            record_feedback(st.session_state["user"]["id"], hero_article["id"], "save")
            st.info("Story saved to reading list.")

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0 1.2rem 0;'>", unsafe_allow_html=True)

# ── FILTER & SEARCH BAR ───────────────────────────────────────────────────────
st.markdown("### 📰 Story Feed")

filter_col1, filter_col2, filter_col3 = st.columns([2.5, 1.5, 2])
with filter_col1:
    filter_mode = st.radio(
        "Feed Filter:",
        ["All", "🛡️ High Trust Only", "🔥 Trending Stories"],
        horizontal=True,
        label_visibility="collapsed",
    )
with filter_col2:
    selected_category = st.selectbox(
        "Category:",
        ["All Topics"] + CATEGORIES,
        label_visibility="collapsed",
    )
with filter_col3:
    search_query = st.text_input(
        "Search:",
        placeholder="🔍 Search headlines or topics...",
        label_visibility="collapsed",
    )

# Filter Logic
candidate_feed = all_articles[1:]
if filter_mode == "🔥 Trending Stories":
    candidate_feed = get_trending_articles(candidate_feed, top_k=30)
elif filter_mode == "🛡️ High Trust Only":
    candidate_feed = [a for a in candidate_feed if a["tm"]["tier"] == "high"]

if selected_category != "All Topics":
    candidate_feed = [a for a in candidate_feed if a.get("category") == selected_category]

if search_query.strip():
    q = search_query.strip().lower()
    candidate_feed = [
        a for a in candidate_feed
        if q in a.get("title", "").lower() or q in a.get("text", "").lower() or q in a.get("source", "").lower()
    ]

# ── MAIN LAYOUT: FEED (LEFT) + TRENDING SIDEBAR (RIGHT) ───────────────────────
feed_col, trending_col = st.columns([2.7, 1.1])

with feed_col:
    if not candidate_feed:
        st.info("No articles match your current filter criteria.")
    else:
        st.caption(f"Showing **{min(len(candidate_feed), 16)}** of **{len(candidate_feed)}** matching stories")
        
        # Display articles in 2-column grid
        items_to_display = candidate_feed[:16]
        for row_idx in range(0, len(items_to_display), 2):
            col_a, col_b = st.columns(2)
            for sub_idx, col in enumerate([col_a, col_b]):
                item_idx = row_idx + sub_idx
                if item_idx < len(items_to_display):
                    art = items_to_display[item_idx]
                    tm = art["tm"]
                    cred_pct = tm["cred_pct"]
                    fill_class = (
                        "tn-cred-fill-high" if tm["tier"] == "high"
                        else "tn-cred-fill-mixed" if tm["tier"] == "mixed"
                        else "tn-cred-fill-low"
                    )
                    snippet = art.get("summary", "")[:100] + ("..." if len(art.get("summary", "")) > 100 else "")
                    
                    with col:
                        st.markdown(f"""
                        <div class="tn-story-card">
                            <div class="tn-card-thumb-wrap">
                                <img src="{tm['image_url']}" alt="{art['title']}" />
                            </div>
                            <div class="tn-card-body">
                                <div>
                                    <div class="tn-card-tags">
                                        <span class="tn-badge-category">{art['category']}</span>
                                        {get_trust_badge_html(art)}
                                    </div>
                                    <div class="tn-headline">{art['title']}</div>
                                    <div class="tn-summary">{snippet}</div>
                                </div>
                                <div>
                                    <div class="tn-cred-bar-wrap">
                                        <div class="tn-cred-bar-meta">
                                            <span>Credibility</span>
                                            <span style="color:{tm['tier_color']};">{cred_pct}%</span>
                                        </div>
                                        <div class="tn-cred-track">
                                            <div class="tn-cred-fill {fill_class}" style="width: {cred_pct}%;"></div>
                                        </div>
                                    </div>
                                    <div class="tn-card-footer">
                                        <span>📰 {art['source']}</span>
                                        <span>⏱️ {tm['read_time_mins']}m read</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        btn_c1, btn_c2 = st.columns(2)
                        with btn_c1:
                            if st.button("👍 Like", key=f"like_{art['id']}"):
                                record_feedback(st.session_state["user"]["id"], art["id"], "like")
                                st.success("Liked!")
                        with btn_c2:
                            if st.button("🔖 Save", key=f"save_{art['id']}"):
                                record_feedback(st.session_state["user"]["id"], art["id"], "save")
                                st.info("Saved!")
                                
                        with st.expander(f"📖 Read Full Article (ID: #{art['id']})"):
                            st.write(art.get("text", art["summary"]))
                            dwell = st.slider("Read duration (sec):", 5, 120, 30, key=f"dwell_{art['id']}")
                            if st.button("⏱️ Log Read Session", key=f"log_read_{art['id']}"):
                                w_cnt = len(art.get("text", "").split())
                                comp_rate = min(1.0, dwell / max(10.0, w_cnt / 3.5))
                                record_reading_history(st.session_state["user"]["id"], art["id"], dwell, comp_rate)
                                st.success(f"Recorded {dwell}s reading session ({comp_rate*100:.0f}% completion).")

with trending_col:
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:1.1rem;margin-bottom:1.5rem;">
        <div style="font-size:1.1rem;font-weight:800;color:#f8fafc;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
            🔥 <span>Trending Radar</span>
        </div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:14px;">
            High-velocity stories sorted by engagement and reader momentum.
        </div>
    """, unsafe_allow_html=True)
    
    trending_list = get_trending_articles(all_articles, top_k=5)
    for rank_idx, t_art in enumerate(trending_list, 1):
        t_tm = get_article_trust_metrics(t_art)
        badge_html = get_trust_badge_html(t_art)
        st.markdown(f"""
        <div style="padding:10px 0;border-bottom:1px solid #1e293b;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-size:1rem;font-weight:900;color:#0ea5e9;">#{rank_idx}</span>
                <span style="font-size:0.72rem;color:#94a3b8;">👁️ {t_art.get('views', 10)} views</span>
                <span style="margin-left:auto;">{badge_html}</span>
            </div>
            <div style="font-size:0.86rem;font-weight:700;color:#f1f5f9;line-height:1.35;margin-bottom:4px;">
                {t_art['title']}
            </div>
            <div style="font-size:0.72rem;color:#64748b;">
                Source: <b style="color:#cbd5e1;">{t_art['source']}</b> &bull; {t_art['category']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    <b>TrustNews AI Platform</b> &bull; Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

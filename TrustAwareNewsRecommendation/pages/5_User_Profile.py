"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Page 5: User Profile & Engagement Management
"""

import streamlit as st
import pandas as pd

from config import CATEGORIES, DEFAULT_WEIGHTS, ASSETS_DIR
from database.db import (
    register_user,
    authenticate_user,
    get_user_preferences,
    save_user_preferences,
    get_user_history,
    get_saved_articles,
)
from src.article_intelligence import (
    get_article_trust_metrics,
    get_user_reading_stats,
    get_trust_badge_html,
)

st.set_page_config(
    page_title="TrustNews AI | User Profile",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session State for User Authentication
if "user" not in st.session_state:
    st.session_state["user"] = {"id": 1, "username": "demo_user"}
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True

active_user = st.session_state.get("user", {"id": 1, "username": "demo_user"})
user_id = active_user.get("id", 1)
username = active_user.get("username", "demo_user")

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
            <span>👤</span>
            <span>User Profile & <span class="tn-accent">Preferences</span></span>
        </h1>
        <div class="tn-brand-subtitle">
            Account Management &bull; 7-Objective Custom Weightings &bull; Reading History & Library
        </div>
    </div>
    <div class="tn-live-pill">
        <span class="tn-live-dot"></span>
        <span>Account Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Current Active Session Info ───────────────────────────────────────────────
col_sess_info, col_sess_btn = st.columns([3, 1])
with col_sess_info:
    st.markdown(f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px 18px;margin-bottom:1.4rem;display:flex;align-items:center;gap:12px;">
        <span style="font-size:1.6rem;">👤</span>
        <div>
            <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;">
                Signed in as <b style="color:#0ea5e9;">{username}</b> &bull; (User ID: <code style="color:#38bdf8;">#{user_id}</code>)
            </div>
            <div style="font-size:0.75rem;color:#94a3b8;">
                Account status: Authenticated session &bull; Database role: Standard Reader
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_sess_btn:
    if st.button("🚪 Switch / Logout", use_container_width=True):
        st.session_state["user"] = {"id": 1, "username": "demo_user"}
        st.session_state["authenticated"] = True
        st.info("Reset to default demo account.")
        st.rerun()

# ── Engagement Stats Card ─────────────────────────────────────────────────────
history = get_user_history(user_id, limit=50)
reading_stats = get_user_reading_stats(user_id, history)

st.markdown(f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:1.4rem;margin-bottom:1.6rem;box-shadow:0 4px 14px rgba(0,0,0,0.3);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:1.1rem;font-weight:800;color:#f8fafc;">
            📊 Reading Engagement & Activity Status
        </span>
        <span style="background:rgba(14,165,233,0.15);border:1px solid #0ea5e9;color:#38bdf8;font-size:0.75rem;font-weight:700;padding:4px 12px;border-radius:999px;">
            {reading_stats['engagement_level']}
        </span>
    </div>
    <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:12px;">
        Temporal telemetry evaluates dwell duration and paragraph completion to filter accidental bounce clicks from genuine deep reads.
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;">
        <div class="tn-stat-card" style="padding:1rem;">
            <div class="tn-stat-num" style="font-size:1.7rem;">{reading_stats['total_articles']}</div>
            <div class="tn-stat-label">Articles Read</div>
        </div>
        <div class="tn-stat-card" style="padding:1rem;">
            <div class="tn-stat-num" style="font-size:1.7rem;color:#10b981;">{reading_stats['avg_dwell_time']}s</div>
            <div class="tn-stat-label">Average Dwell Time</div>
        </div>
        <div class="tn-stat-card" style="padding:1rem;">
            <div class="tn-stat-num" style="font-size:1.7rem;color:#f59e0b;">{reading_stats['avg_completion']}%</div>
            <div class="tn-stat-label">Average Completion</div>
        </div>
        <div class="tn-stat-card" style="padding:1rem;">
            <div class="tn-stat-num" style="font-size:1.7rem;color:#a855f7;">{reading_stats['engagement_level']}</div>
            <div class="tn-stat-label">Engagement Level</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabs: Account, Preferences & Weights, Timeline, Saved ────────────────
tab_auth, tab_prefs, tab_weights, tab_hist, tab_saved = st.tabs([
    "🔐 Account Management",
    "⚙️ Topic Preferences",
    "⚖️ Custom Objective Weights",
    "📜 Reading Timeline",
    "🔖 Saved Articles Library",
])

# 1. Account Management (Login / Register side-by-side)
with tab_auth:
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 🔑 Sign In to Existing Account")
        login_u = st.text_input("Username:", key="auth_login_user")
        login_p = st.text_input("Password:", type="password", key="auth_login_pass")
        if st.button("Sign In", type="primary", key="btn_signin"):
            if not login_u or not login_p:
                st.warning("Please provide both username and password.")
            else:
                user_match = authenticate_user(login_u, login_p)
                if user_match:
                    st.session_state["user"] = dict(user_match)
                    st.session_state["authenticated"] = True
                    st.success(f"Welcome back, **{user_match['username']}**!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please verify your username and password.")

    with col_r:
        st.markdown("#### 📝 Register New Account")
        reg_u = st.text_input("New Username:", key="auth_reg_user")
        reg_p = st.text_input("New Password:", type="password", key="auth_reg_pass")
        if st.button("Create Account", key="btn_create_acc"):
            if not reg_u or not reg_p:
                st.warning("Username and password cannot be empty.")
            else:
                ok, msg = register_user(reg_u, reg_p)
                if ok:
                    st.success(f"Account **{reg_u}** successfully registered! You may now sign in.")
                else:
                    st.error(msg)

# 2. Reading Topic Preferences
with tab_prefs:
    st.markdown("#### ⚙️ Manage Preferred Categories")
    pref_data = get_user_preferences(user_id)
    current_cats = pref_data.get("categories", ["Politics", "World News"])
    current_weights = pref_data.get("weights", DEFAULT_WEIGHTS)

    selected_topics = st.multiselect(
        "Select your preferred news categories:",
        options=CATEGORIES,
        default=[c for c in current_cats if c in CATEGORIES] or ["Politics", "World News"],
        key="profile_topics_multiselect",
    )

    if st.button("💾 Save Topic Preferences", type="primary", key="btn_save_topics"):
        if save_user_preferences(user_id, selected_topics, current_weights):
            st.success("Topic preferences saved! Recommendations will adapt immediately.")
            st.rerun()
        else:
            st.error("Failed to save preferences to database.")

# 3. 7-Objective Custom Weights
with tab_weights:
    st.markdown("#### ⚖️ Customize 7 Recommender Objective Weights")
    st.caption("Adjust how the trust-aware algorithm balances each dimension when building your personalized feed.")

    w_defaults = pref_data.get("weights", DEFAULT_WEIGHTS)
    w_c1, w_c2 = st.columns(2)
    with w_c1:
        new_w_rel = st.slider("Content Relevance (TF-IDF Similarity):", 0.0, 1.0, float(w_defaults.get("relevance", 0.25)), 0.05, key="w_rel_sl")
        new_w_cred = st.slider("Credibility Score (NLP + Domain Track):", 0.0, 1.0, float(w_defaults.get("credibility", 0.30)), 0.05, key="w_cred_sl")
        new_w_eng = st.slider("Long-Term Engagement (Dwell Time):", 0.0, 1.0, float(w_defaults.get("engagement", 0.15)), 0.05, key="w_eng_sl")
        new_w_sat = st.slider("User Satisfaction (Likes / Saves):", 0.0, 1.0, float(w_defaults.get("satisfaction", 0.10)), 0.05, key="w_sat_sl")
    with w_c2:
        new_w_div = st.slider("Content Diversity (Topic Variety):", 0.0, 1.0, float(w_defaults.get("diversity", 0.10)), 0.05, key="w_div_sl")
        new_w_nov = st.slider("Novelty (Freshness / Niche Stories):", 0.0, 1.0, float(w_defaults.get("novelty", 0.05)), 0.05, key="w_nov_sl")
        new_w_fair = st.slider("Publisher Exposure Fairness:", 0.0, 1.0, float(w_defaults.get("fairness", 0.05)), 0.05, key="w_fair_sl")

    new_weights_dict = {
        "relevance": new_w_rel,
        "credibility": new_w_cred,
        "engagement": new_w_eng,
        "satisfaction": new_w_sat,
        "diversity": new_w_div,
        "novelty": new_w_nov,
        "fairness": new_w_fair,
    }

    if st.button("💾 Save Custom Weights", type="primary", key="btn_save_weights"):
        if save_user_preferences(user_id, selected_topics, new_weights_dict):
            st.success("Custom ranking weights saved successfully!")
            st.rerun()
        else:
            st.error("Error updating custom weights in database.")

# 4. Reading Timeline Table
with tab_hist:
    st.markdown("#### 📜 Reading Timeline & Dwell Log")
    if not history:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:1.5rem;text-align:center;color:#94a3b8;font-size:0.88rem;">
            📌 No reading events recorded yet. Read stories from <b>Home</b> or <b>Personalized Feed</b> and log dwell times to populate this timeline.
        </div>
        """, unsafe_allow_html=True)
    else:
        timeline_rows = []
        for h in history:
            cred_val = float(h.get("credibility_score", 0.5))
            tm = get_article_trust_metrics({"credibility_score": cred_val, "text": ""})
            timeline_rows.append({
                "Timestamp": h.get("timestamp", ""),
                "Article Headline": h.get("title", "")[:50] + "...",
                "Source": h.get("source", ""),
                "Credibility Tier": tm["tier_label"],
                "Trust %": f"{int(cred_val * 100)}%",
                "Dwell (s)": f"{h.get('dwell_time_sec', 0)}s",
                "Completion": f"{int(h.get('completion_rate', 0.5) * 100)}%",
                "Signal Type": "⚠️ Accidental Bounce" if h.get("accidental_click") else "✅ Genuine Engagement",
            })
        st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)

# 5. Saved Articles Library
with tab_saved:
    st.markdown("#### 🔖 Saved Articles Library")
    saved_list = get_saved_articles(user_id)
    if not saved_list:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:1.5rem;text-align:center;color:#94a3b8;font-size:0.88rem;">
            📌 No saved articles in your library yet. Click <b>🔖 Save</b> on any story in the feed to bookmark it for later reading.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption(f"You have **{len(saved_list)}** bookmarked articles in your private library.")
        for s in saved_list:
            tm = get_article_trust_metrics(s)
            badge = get_trust_badge_html(s)
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:14px 18px;margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
                    <div style="font-weight:800;color:#f8fafc;font-size:1.02rem;">{s['title']}</div>
                    {badge}
                </div>
                <div style="font-size:0.80rem;color:#94a3b8;margin-bottom:6px;">
                    Source: <b style="color:#cbd5e1;">{s['source']}</b> &bull; Category: <b style="color:#0ea5e9;">{s['category']}</b> &bull; Trust: <b>{tm['cred_pct']}%</b> &bull; Saved: {s.get('saved_at', 'Recently')}
                </div>
                <div style="font-size:0.85rem;color:#cbd5e1;line-height:1.45;">
                    {s.get('summary', s.get('text', ''))[:160]}...
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

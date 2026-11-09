"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Page 7: About, System Architecture & Academic Methodology
"""

import streamlit as st
from config import ASSETS_DIR

st.set_page_config(
    page_title="TrustNews AI | About & Methodology",
    page_icon="📖",
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

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-brand-header">
    <div>
        <h1 class="tn-brand-title">
            <span>📖</span>
            <span>About <span class="tn-accent">TrustNews AI</span></span>
        </h1>
        <div class="tn-brand-subtitle">
            Advanced Multi-Objective Evaluation & Trust-Aware Ranking Framework for News Recommendation
        </div>
    </div>
    <div class="tn-live-pill">
        <span class="tn-live-dot"></span>
        <span>Academic Capstone</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Executive Abstract Card ───────────────────────────────────────────────────
st.markdown("""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:1.6rem;margin-bottom:1.8rem;box-shadow:0 4px 14px rgba(0,0,0,0.3);line-height:1.7;">
    <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;margin-bottom:8px;">
        🎯 Project Executive Summary
    </div>
    <p style="color:#cbd5e1;font-size:0.92rem;margin-bottom:12px;">
        Conventional digital news recommendation engines optimize predominantly for short-term engagement heuristics such as <b>Click-Through Rate (CTR)</b> and viral amplification. While profitable for engagement monetization, this single-objective paradigm drives systemic epistemic risks: the algorithmic promotion of deceptive clickbait, ideological filter bubbles, and the unchecked spread of unverified disinformation.
    </p>
    <p style="color:#cbd5e1;font-size:0.92rem;margin:0;">
        <b>TrustNews AI</b> supersedes single-objective click-maximization with a <b>7-Objective Trust-Aware Pareto Optimization Architecture</b>. The system synthesizes calibrated NLP veracity classification, multi-factor publisher reputation tracking, epistemic uncertainty modeling, and dwell-time engagement validation into an explainable, production-ready news feed.
    </p>
</div>
""", unsafe_allow_html=True)

# ── System Architecture Pipeline (Text-Based Cards) ───────────────────────────
st.markdown("### 🏗️ End-to-End System Architecture")
st.markdown("""
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:14px;margin-bottom:1.8rem;">
    <div style="background:#0f172a;border:1px solid #1e293b;border-top:3px solid #0ea5e9;border-radius:10px;padding:14px;">
        <div style="font-size:0.75rem;color:#0ea5e9;font-weight:800;text-transform:uppercase;">Layer 1: Ingestion & Sanitization</div>
        <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;margin:4px 0;">Corpus Preprocessing</div>
        <div style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
            Sanitizes wire-agency prefixes (e.g., "Reuters / WASHINGTON") to eliminate spurious classifier artifacts. Performs n-gram vectorization and linguistic feature extraction.
        </div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-top:3px solid #10b981;border-radius:10px;padding:14px;">
        <div style="font-size:0.75rem;color:#10b981;font-weight:800;text-transform:uppercase;">Layer 2: NLP Inference</div>
        <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;margin:4px 0;">Calibrated Veracity Detection</div>
        <div style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
            Passive Aggressive classifier with calibrated decision boundaries outputs veracity posterior ($P_{real}, P_{fake}$) alongside Shannon entropy uncertainty index.
        </div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-top:3px solid #f59e0b;border-radius:10px;padding:14px;">
        <div style="font-size:0.75rem;color:#f59e0b;font-weight:800;text-transform:uppercase;">Layer 3: Multi-Factor Credibility</div>
        <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;margin:4px 0;">Credibility Index C(a)</div>
        <div style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
            Synthesizes model probability (50%), publisher track record (25%), sensationalism penalty (15%), and factual attribution markers (10%) into [0, 1].
        </div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-top:3px solid #a855f7;border-radius:10px;padding:14px;">
        <div style="font-size:0.75rem;color:#a855f7;font-weight:800;text-transform:uppercase;">Layer 4: Pareto Optimization</div>
        <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;margin:4px 0;">Constrained Re-ranking</div>
        <div style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
            Balances 7 conflicting objectives (Relevance, Credibility, Engagement, Satisfaction, Diversity, Novelty, Fairness) with credibility floors and MMR topic diversity.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Methodology Steps (Numbered Cards) ────────────────────────────────────────
st.markdown("### 📋 Research Methodology & Formulation")

methodology_steps = [
    ("1", "Data Ingestion & Wire-Prefix De-biasing", "Processed 44,898 authentic and fabricated articles from the ISOT dataset. Automated regular expressions strip publisher datelines and wire-service stamps to prevent the classifier from memorizing superficial agency tags."),
    ("2", "Multi-Model NLP Benchmarking", "Compared Logistic Regression, Passive Aggressive, Multinomial Naive Bayes, and Random Forest on 10,000 TF-IDF n-grams (1, 2). The calibrated Passive Aggressive classifier achieved superior F1-score (98.98%) and ROC-AUC (0.9993)."),
    ("3", "Epistemic Uncertainty Estimation", "Quantified model prediction confidence using Shannon entropy $H(p) = -(p \\log_2 p + (1-p) \\log_2(1-p))$ and boundary margin distances to detect borderline ambiguous claims."),
    ("4", "Dynamic User Profile Modeling", "Formulated implicit user interest vectors using TF-IDF cosine representations, updated via reading dwell times, completion rates, and explicit feedback (+2 for Likes, +3 for Bookmarks, -0.4 for Dislikes)."),
    ("5", "7-Objective Hybrid Ranking & Pareto Sorting", "Simultaneously optimizes Relevance (R), Credibility (C), Long-Term Engagement (E), Satisfaction (S), Diversity (D), Novelty (N), and Fairness (F) across non-dominated Pareto fronts."),
    ("6", "Constrained Guardrails & MMR Re-Ranking", "Enforces a strict credibility floor ($\\text{Credibility} \\ge 0.40$), source exposure caps (max 3 articles per publisher), and Maximal Marginal Relevance to prevent publisher monopolization."),
]

for num, title, desc in methodology_steps:
    st.markdown(f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:14px 18px;margin-bottom:10px;display:flex;align-items:flex-start;gap:14px;">
        <div style="background:rgba(14,165,233,0.15);border:1px solid #0ea5e9;color:#38bdf8;font-size:1.1rem;font-weight:900;width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            {num}
        </div>
        <div>
            <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;margin-bottom:3px;">{title}</div>
            <div style="font-size:0.82rem;color:#94a3b8;line-height:1.55;">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0;'>", unsafe_allow_html=True)

# ── Technologies Used & Dataset Description ───────────────────────────────────
col_tech, col_data = st.columns(2)

with col_tech:
    st.markdown("#### 💻 Technologies & Frameworks")
    tech_items = [
        ("🐍 Python 3.10+", "Core algorithmic backend, mathematical vectorization, and data structures."),
        ("⚡ Streamlit", "Full-stack reactive web interface with custom CSS components and dark-mode styling."),
        ("🤖 Scikit-Learn", "TF-IDF vectorizer, Passive Aggressive classifier, Logistic Regression, Pareto sorting."),
        ("📊 Plotly", "Interactive dark-theme visualizations (radar plots, Pareto scatter, histograms)."),
        ("🗄️ SQLite3", "ACID-compliant relational storage for indexed articles, reading sessions, and user profiles."),
        ("📐 NumPy & Pandas", "Matrix manipulation, vector cosine similarity, and dataframe telemetry."),
    ]
    for name, desc in tech_items:
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;margin-bottom:8px;">
            <div style="font-size:0.88rem;font-weight:800;color:#f8fafc;">{name}</div>
            <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

with col_data:
    st.markdown("#### 📚 Dataset & Corpus Specifications")
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;line-height:1.65;font-size:0.85rem;color:#cbd5e1;">
        <div style="font-weight:800;color:#38bdf8;margin-bottom:6px;">ISOT Fake News Corpus (University of Victoria)</div>
        <p style="margin-bottom:8px;">
            The primary training corpus comprises <b>44,898 authentic and fabricated news articles</b> collected from real-world digital outlets between 2016 and 2018:
        </p>
        <ul style="padding-left:18px;margin-bottom:12px;color:#94a3b8;">
            <li><b>21,417 Real News Articles:</b> Sourced directly from verified wire services (Reuters, Associated Press).</li>
            <li><b>23,481 Fake News Articles:</b> Flagged by independent non-partisan fact-checking organizations (PolitiFact, FactCheck.org).</li>
            <li><b>Wire Sanitization:</b> Automated elimination of <code>Reuters (WASHINGTON)</code> stamps.</li>
            <li><b>Indexed App Corpus:</b> 3,000 balanced articles stored in SQLite for responsive live demonstration.</li>
        </ul>
        <div style="border-top:1px solid #1e293b;padding-top:8px;font-size:0.75rem;color:#64748b;">
            Split: 70% Training &bull; 10% Validation &bull; 20% Held-Out Test Split.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0;'>", unsafe_allow_html=True)

# ── Academic Disclaimer & References ──────────────────────────────────────────
st.markdown("#### ⚖️ Academic Disclaimer & References")
st.markdown("""
<div style="background:rgba(30,41,59,0.5);border-left:3px solid #0ea5e9;border-radius:4px;padding:14px 18px;margin-bottom:16px;font-size:0.82rem;color:#94a3b8;line-height:1.6;">
    <b>Academic Disclaimer:</b> This system was conceptualized, implemented, and empirically evaluated as a capstone degree project for the Department of Computer Science & Engineering, Assam DownTown University. Algorithmic credibility assessments and uncertainty scores are computational aids designed to mitigate misinformation exposure; they do not supersede investigative journalistic verification or human editorial judgement.
</div>
""", unsafe_allow_html=True)

references = [
    "Ahmed, H., Traore, I., & Saad, S. (2018). 'Detecting opinion spams and fake news using text classification.' <i>Security and Privacy</i>, 1(1), e9.",
    "Carbonell, J., & Goldstein, J. (1998). 'The use of MMR, diversity-based reranking for reordering documents and producing summaries.' <i>ACM SIGIR</i>.",
    "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). ''Why should I trust you?': Explaining the predictions of any classifier.' <i>ACM KDD</i>.",
    "Steck, H. (2018). 'Calibrated recommendations.' <i>ACM RecSys</i>, pp. 154-162.",
]

for ref in references:
    st.markdown(f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:4px;'>&bull; {ref}</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

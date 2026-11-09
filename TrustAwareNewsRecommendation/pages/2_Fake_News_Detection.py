"""
TrustNews AI — Advanced Multi-Objective Trust-Aware News Recommendation System
Page 2: Credibility & Authenticity Inspector
"""

import streamlit as st
import plotly.graph_objects as go

from config import BEST_MODEL_PATH, ASSETS_DIR
from src.prediction import predict_news
from src.credibility import calculate_credibility_score

st.set_page_config(
    page_title="TrustNews AI | Credibility Inspector",
    page_icon="🔍",
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
            <span>🔍</span>
            <span>Credibility & Veracity <span class="tn-accent">Inspector</span></span>
        </h1>
        <div class="tn-brand-subtitle">
            Deep NLP Veracity Detection &bull; Multi-Factor Credibility Scoring &bull; Epistemic Uncertainty
        </div>
    </div>
    <div class="tn-live-pill">
        <span class="tn-live-dot"></span>
        <span>NLP Inference Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Check Trained Model Gracefully ────────────────────────────────────────────
if not BEST_MODEL_PATH.exists():
    st.info("ℹ️ The trained fake news detection model was not found at the expected path. Please run the model training script (`python -m src.train_model`) to generate model artifacts.")
    st.markdown("""
    <div class="tn-footer">
        Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Preset Benchmark Samples ──────────────────────────────────────────────────
PRESETS = {
    "Choose a preset sample or enter your own custom story below...": ("", "", "Reuters"),
    "Sample 1: Authentic Wire News — Diplomatic Summit (Reuters)": (
        "Diplomatic Envoys Finalize Cross-Border Clean Tech Verification Standards at Geneva Summit",
        "GENEVA - Diplomatic representatives from over forty sovereign nations agreed on Thursday to ratify the multilateral framework on accelerated industrial decarbonization. Under the finalized diplomatic protocol, signatory states will report annual emissions through international verification panels starting in 2027. Official delegates confirmed that legal drafting was concluded following five days of closed-door negotiations.",
        "Reuters"
    ),
    "Sample 2: Sensational Clickbait / Conspiracy (Social Media)": (
        "BREAKING BOMBSHELL: Big Pharma Exposed Hiding Shocking Miracle Cure That Destroys All Diseases Instantly!",
        "YOU WON'T BELIEVE WHAT THEY CONCEALED! Secret documents leaked from an anonymous whistleblower prove that global elite doctors have been suppressing this ancient organic root for thirty years! Doctors are absolutely terrified that this jaw-dropping miracle discovery is going viral. Share this before corrupt politicians take it down!",
        "Social Media Post"
    ),
    "Sample 3: Complex Financial Legislation (Wall Street Journal)": (
        "House Committee Passes Sweeping Deregulation Bill Over Strong Minority Objections",
        "The House Financial Services Committee voted 28-22 along strict party lines on Wednesday to approve comprehensive legislation easing capital reserve requirements for regional financial institutions. Lawmakers argued the measure unleashes lending, while critics warned it weakens deposit insurance guardrails.",
        "The Wall Street Journal"
    ),
}

st.markdown("#### ⚡ Benchmark Testing Presets")
selected_preset_key = st.selectbox(
    "Select a benchmark sample to test:",
    list(PRESETS.keys()),
    label_visibility="collapsed",
)

sample_title, sample_text, sample_source = PRESETS[selected_preset_key]

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ── Form Inputs ───────────────────────────────────────────────────────────────
col_form_main, col_form_side = st.columns([2.5, 1.2])

with col_form_main:
    title_val = st.text_input(
        "Headline / News Title:",
        value=sample_title,
        placeholder="Enter or paste news headline...",
    )
    text_val = st.text_area(
        "Article Body Text:",
        value=sample_text,
        height=160,
        placeholder="Paste full article text or summary paragraph here...",
    )

with col_form_side:
    source_val = st.selectbox(
        "Publisher / News Source:",
        [
            "Reuters",
            "Associated Press",
            "BBC",
            "The New York Times",
            "The Wall Street Journal",
            "The Guardian",
            "CNN",
            "Fox News",
            "Social Media Post",
            "Unverified Blog",
            "Conspiracy / Dubious",
            "Unknown",
        ],
        index=0 if sample_source == "Reuters" else (4 if "Journal" in sample_source else 8),
    )
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;font-size:0.75rem;color:#94a3b8;margin-top:10px;">
        💡 <b>How it works:</b> Our calibrated NLP model analyzes lexical patterns and style, then computes source credibility weighting and epistemic uncertainty.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 Analyze Article Credibility", type="primary", use_container_width=True)

# ── Trigger Prediction ────────────────────────────────────────────────────────
if analyze_clicked or (sample_title and selected_preset_key != list(PRESETS.keys())[0]):
    if not title_val.strip() and not text_val.strip():
        st.warning("⚠️ Please provide a headline or article text to evaluate.")
    else:
        with st.spinner("Classifying text, evaluating credibility, and extracting linguistic features..."):
            try:
                pred = predict_news(title_val, text_val)
                cred = calculate_credibility_score(pred, source=source_val)
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        is_real = pred["label"] == "REAL"
        confidence_pct = pred.get("confidence_percent", 50.0)
        cred_pct = cred.get("credibility_percent", 50.0)
        cred_score = cred.get("credibility_score", 0.5)
        uncertainty = cred.get("uncertainty", {})
        unc_score = uncertainty.get("uncertainty_score", 0.2)
        unc_level = uncertainty.get("level", "Low Uncertainty")

        st.markdown("<hr style='border-color:#1e293b;margin:1.8rem 0 1.2rem 0;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Algorithmic Assessment Results")

        # ── 1. Big REAL/FAKE Verdict Box ──────────────────────────────────────
        verdict_class = "tn-verdict-real" if is_real else "tn-verdict-fake"
        verdict_icon = "🛡️ VERIFIED AUTHENTIC (REAL)" if is_real else "⚠️ DECEPTIVE / QUESTIONABLE (FAKE)"
        verdict_color = "#10b981" if is_real else "#ef4444"

        st.markdown(f"""
        <div class="tn-verdict-box {verdict_class}">
            <div class="tn-verdict-title" style="color:{verdict_color};">
                {verdict_icon}
            </div>
            <div class="tn-verdict-sub">
                NLP Classification Confidence: <b>{confidence_pct}%</b> &bull;
                Publisher Source: <b>{source_val}</b> &bull;
                Composite Credibility: <b>{cred_pct}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 2. Three Diagnostic Cards Row ─────────────────────────────────────
        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(f"""
            <div class="tn-stat-card" style="text-align:left;height:140px;">
                <div class="tn-stat-label">Model Verdict & Posterior</div>
                <div class="tn-stat-num" style="color:{verdict_color};font-size:1.6rem;margin-top:6px;">
                    {pred['label']}
                </div>
                <div style="font-size:0.82rem;color:#cbd5e1;margin-top:4px;">
                    Model Confidence: <b>{confidence_pct}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res2:
            tier_color = "#10b981" if cred_score >= 0.75 else ("#f59e0b" if cred_score >= 0.45 else "#ef4444")
            st.markdown(f"""
            <div class="tn-stat-card" style="text-align:left;height:140px;">
                <div class="tn-stat-label">Multi-Factor Credibility</div>
                <div class="tn-stat-num" style="color:{tier_color};font-size:1.6rem;margin-top:6px;">
                    {cred_pct}%
                </div>
                <div style="font-size:0.82rem;color:#cbd5e1;margin-top:4px;">
                    Tier: <b style="color:{tier_color};">{cred['tier']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res3:
            unc_color = "#10b981" if "Low" in unc_level else ("#f59e0b" if "Moderate" in unc_level else "#ef4444")
            st.markdown(f"""
            <div class="tn-stat-card" style="text-align:left;height:140px;">
                <div class="tn-stat-label">Epistemic Uncertainty</div>
                <div class="tn-stat-num" style="color:{unc_color};font-size:1.6rem;margin-top:6px;">
                    {unc_score:.2f}
                </div>
                <div style="font-size:0.82rem;color:#cbd5e1;margin-top:4px;">
                    Boundary Distance: <b style="color:{unc_color};">{unc_level}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ── 3. Credibility Components Horizontal Bar ──────────────────────────
        st.markdown("#### 🔬 Credibility Component Deconstruction")
        comps = cred.get("components", {})
        fig_bar = go.Figure(go.Bar(
            x=[
                comps.get("classifier_signal", 0.5),
                comps.get("source_reputation", 0.5),
                comps.get("linguistic_integrity", 0.5),
                comps.get("factual_indicators", 0.5),
            ],
            y=[
                "Classifier Veracity Signal",
                "Source Reputation Track",
                "Linguistic Integrity",
                "Factual Marker Proxies",
            ],
            orientation="h",
            marker=dict(color=["#0ea5e9", "#10b981", "#f59e0b", "#a855f7"]),
            text=[
                f"{comps.get('classifier_signal', 0.5):.2f}",
                f"{comps.get('source_reputation', 0.5):.2f}",
                f"{comps.get('linguistic_integrity', 0.5):.2f}",
                f"{comps.get('factual_indicators', 0.5):.2f}",
            ],
            textposition="auto",
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0, 1.05], title="Score (0.0 to 1.0)", gridcolor="#1e293b", zeroline=False),
            yaxis=dict(gridcolor="#1e293b"),
            height=240,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="#080c14",
            plot_bgcolor="#111827",
            font=dict(color="#cbd5e1", family="Inter"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── 4. Linguistic Signals & Feature Attribution ───────────────────────
        st.markdown("#### ✍️ Linguistic Indicators & Semantic Keywords")
        t_signals, t_keywords = st.tabs(["Linguistic Signals", "Feature Attribution"])

        with t_signals:
            ling = pred.get("linguistic_signals", {})
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.metric("Sensationalism Score", f"{ling.get('sensationalism_score', 0):.2f}")
            with l2:
                st.metric("ALL-CAPS Ratio", f"{ling.get('caps_ratio', 0)*100:.1f}%")
            with l3:
                st.metric("Clickbait Triggers", f"{ling.get('clickbait_matches', 0)}")
            with l4:
                st.metric("Estimated Read Time", f"{ling.get('estimated_read_time_sec', 0)}s")

            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;margin-top:12px;font-size:0.85rem;color:#cbd5e1;">
                <b>Shannon Entropy:</b> {uncertainty.get('entropy', 0.0):.3f} bits &bull;
                <b>Interpretation:</b> {uncertainty.get('description', '')}
            </div>
            """, unsafe_allow_html=True)

        with t_keywords:
            top_feats = pred.get("top_features", {})
            fake_cues = top_feats.get("fake_indicators", [])
            real_cues = top_feats.get("real_indicators", [])

            kw1, kw2 = st.columns(2)
            with kw1:
                st.markdown("**Deceptive / Sensational Indicators (Fake Cues):**")
                if fake_cues:
                    for token, weight in fake_cues:
                        st.markdown(f"- `<span style='color:#ef4444;font-weight:700;'>{token}</span>` (Feature weight: +{weight:.3f})", unsafe_allow_html=True)
                else:
                    st.info("No strong deceptive markers detected in the text.")

            with kw2:
                st.markdown("**Objective / Factual Indicators (Real Cues):**")
                if real_cues:
                    for token, weight in real_cues:
                        st.markdown(f"- `<span style='color:#10b981;font-weight:700;'>{token}</span>` (Feature weight: +{weight:.3f})", unsafe_allow_html=True)
                else:
                    st.info("No strong objective markers detected in the text.")

        # ── 5. Classifier Disclaimer ──────────────────────────────────────────
        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.5);border-left:3px solid #0ea5e9;border-radius:4px;padding:12px 16px;margin-top:20px;font-size:0.80rem;color:#94a3b8;line-height:1.5;">
            <b>Algorithmic Transparency Disclaimer:</b> Computational credibility assessment is an empirical analysis based on lexical distributions, statistical feature weights, and domain trust heuristics. It serves as an analytical safeguard and does not replace human investigative journalism or authoritative fact-checking organizations.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tn-footer">
    Advanced Multi-Objective Evaluation and Trust-Aware Ranking Framework | Assam DownTown University | B.Tech CSE | 2024-28
</div>
""", unsafe_allow_html=True)

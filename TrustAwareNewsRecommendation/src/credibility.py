"""
Multi-factor Credibility Assessment and Uncertainty Estimation Module.
Synthesizes fake news probability, source reputation, linguistic style, and factual indicators.
"""

import math
from typing import Dict, Any, Optional
from config import SOURCE_REPUTATION


def get_source_reputation(source: Optional[str]) -> float:
    """Retrieve normalized reputation score [0, 1] for a given news publisher/source."""
    if not source:
        return SOURCE_REPUTATION["unknown"]
    src_clean = source.strip().lower()
    for known_source, score in SOURCE_REPUTATION.items():
        if known_source in src_clean:
            return score
    return SOURCE_REPUTATION["unknown"]


def estimate_factual_consistency_markers(title: str, text: str) -> float:
    """
    Estimate proxy indicators for factual consistency:
    Presence of quotes, direct attributions, date references, and official entity mentions.
    """
    text_str = str(text) if text else ""
    score = 0.5  # Neutral baseline

    # Quote marks indicate direct attribution/interviews
    quote_count = text_str.count('"') + text_str.count("“") + text_str.count("”")
    if quote_count >= 4:
        score += 0.20
    elif quote_count >= 2:
        score += 0.10

    # Attribution verbs (e.g. said, stated, reported, confirmed, announced)
    attribution_words = ["said", "stated", "reported", "confirmed", "announced", "according to", "spokesperson"]
    attr_matches = sum(1 for w in attribution_words if f" {w} " in text_str.lower())
    if attr_matches >= 3:
        score += 0.20
    elif attr_matches >= 1:
        score += 0.10

    # Negative indicator: conspiracy phrases
    conspiracy_cues = ["deep state", "crisis actor", "hoax", "wake up sheeple", "chemtrail", "cabal"]
    if any(cue in text_str.lower() for cue in conspiracy_cues):
        score -= 0.35

    return max(0.05, min(1.0, score))


def compute_uncertainty(prob_real: float) -> Dict[str, Any]:
    """
    Compute epistemic uncertainty using Shannon entropy of the binary prediction.
    Entropy is 1.0 (maximum uncertainty) when prob_real = 0.5, and 0.0 when prob_real is 1.0 or 0.0.
    """
    p = max(1e-6, min(1.0 - 1e-6, prob_real))
    entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
    # Normalized margin from decision boundary
    boundary_distance = 2.0 * abs(prob_real - 0.5)  # 0 at p=0.5, 1 at p=0 or 1
    uncertainty_score = 1.0 - boundary_distance

    if uncertainty_score >= 0.70:
        level = "High Uncertainty"
        desc = "Classifier is close to its decision boundary; independent cross-verification strongly suggested."
    elif uncertainty_score >= 0.35:
        level = "Moderate Uncertainty"
        desc = "Classifier shows reasonable confidence with minor ambiguous semantic tokens."
    else:
        level = "Low Uncertainty"
        desc = "High classifier decision margin; features align decisively with known distribution."

    return {
        "entropy": round(entropy, 4),
        "uncertainty_score": round(uncertainty_score, 4),
        "level": level,
        "description": desc
    }


def calculate_credibility_score(
    prediction_result: Dict[str, Any],
    source: Optional[str] = None,
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute unified multi-factor Credibility Score in [0, 1].
    Distinguishes classifier confidence from independent source credibility.
    """
    weights = custom_weights or {
        "classifier": 0.50,
        "source": 0.25,
        "linguistic": 0.15,
        "factual": 0.10
    }

    # 1. Classifier real probability
    real_prob = prediction_result.get("real_prob", 0.5)

    # 2. Source reputation score
    source_score = get_source_reputation(source)

    # 3. Linguistic & sensationalism score
    ling = prediction_result.get("linguistic_signals", {})
    sensationalism = ling.get("sensationalism_score", 0.2)
    linguistic_score = max(0.0, 1.0 - sensationalism)

    # 4. Factual consistency proxy
    factual_score = estimate_factual_consistency_markers(
        title="",
        text=""
    )

    # Composite Credibility Score
    composite_credibility = (
        weights["classifier"] * real_prob +
        weights["source"] * source_score +
        weights["linguistic"] * linguistic_score +
        weights["factual"] * factual_score
    )
    composite_credibility = max(0.02, min(0.98, composite_credibility))

    # Uncertainty estimation
    uncertainty = compute_uncertainty(real_prob)

    # Credibility Tier Classification
    if composite_credibility >= 0.75:
        tier = "Verified Trustworthy"
        color = "green"
        badge = "🟢 High Trust"
    elif composite_credibility >= 0.55:
        tier = "Moderate Credibility"
        color = "blue"
        badge = "🟡 Moderate Trust"
    elif composite_credibility >= 0.40:
        tier = "Low Trust / Caution"
        color = "orange"
        badge = "🟠 Low Trust"
    else:
        tier = "Flagged Unreliable"
        color = "red"
        badge = "🔴 Questionable / Unreliable"

    return {
        "credibility_score": round(composite_credibility, 4),
        "credibility_percent": round(composite_credibility * 100, 1),
        "tier": tier,
        "color": color,
        "badge": badge,
        "components": {
            "classifier_signal": round(real_prob, 4),
            "source_reputation": round(source_score, 4),
            "linguistic_integrity": round(linguistic_score, 4),
            "factual_indicators": round(factual_score, 4)
        },
        "weights": weights,
        "uncertainty": uncertainty,
        "disclaimer": (
            "Note: The credibility score is a computational assessment based on linguistic features, "
            "publisher reputation, and classifier inference. It does not replace independent journalistic fact-checking."
        )
    }


"""
Explainable Recommendation Module.
Generates human-interpretable plain-English explanations for recommended articles,
deconstructing credibility, relevance, ranking drivers, and epistemic uncertainty.
"""

from typing import Dict, Any, List


def generate_recommendation_explanation(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate transparent, accessible explanations for why an article appears in the user feed.
    """
    relevance = float(article.get("relevance_score", 0.5))
    credibility = float(article.get("credibility_score", 0.5))
    fake_prob = float(article.get("fake_prob", 0.5))
    engagement = float(article.get("engagement_score", 0.5))
    satisfaction = float(article.get("satisfaction_score", 0.5))
    diversity = float(article.get("diversity_score", 0.5))
    category = article.get("category", "General News")
    source = article.get("source", "Unknown")
    is_pareto = article.get("is_pareto_optimal", False)

    # 1. Summary headline sentence
    reasons = []
    if relevance >= 0.70:
        reasons.append(f"strong alignment with your reading interest in {category}")
    elif relevance >= 0.50:
        reasons.append(f"matches your selected topic '{category}'")

    if credibility >= 0.75:
        reasons.append(f"high credibility verification from {source}")
    elif credibility >= 0.55:
        reasons.append("acceptable factual reliability")

    if is_pareto:
        reasons.append("optimal Pareto balance of trust and engagement")

    if not reasons:
        reasons.append("curated candidate from trending feed")

    summary_sentence = "Recommended due to " + " and ".join(reasons) + "."

    # 2. Credibility explanation
    real_pct = round((1.0 - fake_prob) * 100, 1)
    if credibility >= 0.75:
        cred_text = (
            f"The NLP classifier assessed this article with a {real_pct}% probability of being genuine real news. "
            f"The source '{source}' has established editorial accountability."
        )
    elif credibility >= 0.50:
        cred_text = (
            f"The model predicted {real_pct}% real news confidence. "
            "The linguistic style is moderate with limited sensational markers."
        )
    else:
        cred_text = (
            f"Caution: Model detected potential misleading indicators (fake probability: {round(fake_prob * 100, 1)}%). "
            "Exercise critical discretion."
        )

    # 3. Uncertainty evaluation
    if fake_prob < 0.20 or fake_prob > 0.80:
        uncertainty_tag = "Low Uncertainty (High Model Confidence)"
    elif 0.35 <= fake_prob <= 0.65:
        uncertainty_tag = "Moderate-High Ambiguity (Boundary Prediction)"
    else:
        uncertainty_tag = "Moderate Uncertainty"

    # 4. Top drivers
    factors = [
        ("Relevance", relevance),
        ("Credibility", credibility),
        ("Engagement", engagement),
        ("Satisfaction", satisfaction),
        ("Diversity", diversity)
    ]
    factors.sort(key=lambda x: x[1], reverse=True)
    top_drivers = [f"{name} ({round(score * 100)}%)" for name, score in factors[:3]]

    return {
        "summary": summary_sentence,
        "relevance_detail": f"Content similarity to your active interest profile is {round(relevance * 100, 1)}%.",
        "credibility_detail": cred_text,
        "uncertainty_level": uncertainty_tag,
        "top_drivers": top_drivers,
        "category": category,
        "source": source
    }


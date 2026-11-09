"""
User Satisfaction Estimation Module.
Models explicit feedback, completion depth, preference consistency, and clickbait bounce penalties.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def compute_satisfaction_score(
    user_feedback: Dict[str, List[int]],
    user_history: List[Dict[str, Any]],
    candidate_article: Dict[str, Any],
    preferred_categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Estimate the expected user satisfaction score [0, 1] for a candidate article.
    Distinguishes genuine deep reading satisfaction from shallow clickbait exposure.
    """
    article_id = candidate_article.get("id")
    category = candidate_article.get("category", "General News")
    views = candidate_article.get("views", 10)
    likes = candidate_article.get("likes", 0)
    dislikes = candidate_article.get("dislikes", 0)
    saves = candidate_article.get("saves", 0)
    dwell_avg = candidate_article.get("dwell_time_avg", 30.0)

    # 1. Collective Social Satisfaction (Empirical feedback ratio)
    total_feedback = likes + dislikes + (saves * 1.5)
    if total_feedback > 0:
        collective_score = (likes + (saves * 1.5)) / (likes + dislikes + (saves * 1.5) + 2.0)
    else:
        collective_score = 0.50

    # 2. Preference Consistency (Does candidate align with user declared interests?)
    prefs = preferred_categories or []
    if prefs:
        preference_consistency = 0.85 if category in prefs else 0.40
    else:
        preference_consistency = 0.50

    # 3. Personal Feedback Affinity
    # Check if user previously liked/saved/disliked articles in this category
    liked_ids = set(user_feedback.get("like", []))
    saved_ids = set(user_feedback.get("save", []))
    disliked_ids = set(user_feedback.get("dislike", []))

    if article_id in liked_ids or article_id in saved_ids:
        personal_affinity = 0.95
    elif article_id in disliked_ids:
        personal_affinity = 0.05
    else:
        # Check historical category satisfaction
        cat_history = [h for h in user_history if h.get("category") == category]
        if cat_history:
            avg_comp = np.mean([h.get("completion_rate", 0.5) for h in cat_history])
            personal_affinity = float(avg_comp)
        else:
            personal_affinity = 0.50

    # 4. Clickbait Bounce Penalty
    # If dwell time is low (< 15s) relative to word length, satisfaction decreases
    bounce_penalty = 0.0
    if dwell_avg < 15.0:
        bounce_penalty = 0.15

    # Composite Satisfaction Score
    raw_score = (
        0.35 * personal_affinity +
        0.35 * collective_score +
        0.30 * preference_consistency -
        bounce_penalty
    )
    satisfaction_score = max(0.05, min(0.95, round(raw_score, 4)))

    return {
        "satisfaction_score": satisfaction_score,
        "collective_feedback_ratio": round(collective_score, 3),
        "preference_consistency": round(preference_consistency, 3),
        "personal_affinity": round(personal_affinity, 3),
        "bounce_penalty": round(bounce_penalty, 3)
    }


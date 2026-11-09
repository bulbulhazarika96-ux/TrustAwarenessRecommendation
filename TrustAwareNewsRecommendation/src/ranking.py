"""
Multi-Objective Trust-Aware Ranking Engine.
Computes and balances Relevance, Engagement, Satisfaction, Credibility, Diversity, Novelty, and Exposure Fairness.
Also implements conventional Click-Based Ranking as a baseline comparison.
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np

from config import DEFAULT_WEIGHTS
from src.engagement import compute_long_term_engagement
from src.satisfaction import compute_satisfaction_score
from src.credibility import calculate_credibility_score


def compute_novelty_score(views: int, total_views: int) -> float:
    """
    Novelty score: Information self-content based on inverse consumption probability.
    Unseen/niche articles have higher novelty; ultra-popular viral articles have lower novelty.
    """
    if total_views <= 0 or views <= 0:
        return 0.5
    prob = views / total_views
    # Scale -log2(p) to [0, 1]
    info_content = -math.log2(max(1e-5, min(1.0, prob)))
    return round(min(1.0, info_content / 15.0), 4)


def compute_exposure_fairness_penalty(
    source: str,
    source_counts: Dict[str, int],
    total_items: int
) -> float:
    """
    Fairness score: Rewards source variety, penalizes over-represented publishers to ensure diversity of voices.
    """
    count = source_counts.get(source, 0)
    ratio = count / max(1, total_items)
    # If a source already occupies > 30% of candidate items, penalize
    if ratio > 0.30:
        return 0.40
    elif ratio > 0.15:
        return 0.70
    return 1.00


def rank_articles(
    articles: List[Dict[str, Any]],
    relevance_scores: List[float],
    user_history: List[Dict[str, Any]],
    user_feedback: Dict[str, List[int]],
    weights: Optional[Dict[str, float]] = None,
    preferred_categories: Optional[List[str]] = None,
    credibility_floor: float = 0.35
) -> List[Dict[str, Any]]:
    """
    Execute 7-objective Trust-Aware Ranking.
    Returns sorted articles with comprehensive multi-objective score breakdowns.
    """
    if not articles:
        return []

    w = weights or DEFAULT_WEIGHTS
    # Normalize weights to sum to 1.0
    w_sum = sum(w.values())
    w_norm = {k: v / w_sum for k, v in w.items()}

    total_views = sum(a.get("views", 10) for a in articles)
    source_counts: Dict[str, int] = {}
    for a in articles:
        src = a.get("source", "Unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    category_counts: Dict[str, int] = {}
    for a in articles:
        cat = a.get("category", "General News")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    ranked_list = []

    for idx, article in enumerate(articles):
        # 1. Relevance Score
        R = relevance_scores[idx] if idx < len(relevance_scores) else 0.5

        # 2. Long-term Engagement Score
        eng_data = compute_long_term_engagement(user_history, article)
        E = eng_data["engagement_score"]

        # 3. User Satisfaction Score
        sat_data = compute_satisfaction_score(user_feedback, user_history, article, preferred_categories)
        S = sat_data["satisfaction_score"]

        # 4. Credibility Score
        C = float(article.get("credibility_score", 0.5))

        # 5. Content Diversity Score (Inverse category concentration)
        cat = article.get("category", "General News")
        cat_ratio = category_counts.get(cat, 1) / len(articles)
        D = round(max(0.1, 1.0 - cat_ratio), 4)

        # 6. Novelty Score (Inverse popularity / freshness)
        N = compute_novelty_score(article.get("views", 10), total_views)

        # 7. Exposure Fairness Score
        F = compute_exposure_fairness_penalty(article.get("source", "Unknown"), source_counts, len(articles))

        # Unified Trust-Aware Score
        final_trust_score = (
            w_norm.get("relevance", 0.25) * R +
            w_norm.get("credibility", 0.30) * C +
            w_norm.get("engagement", 0.15) * E +
            w_norm.get("satisfaction", 0.10) * S +
            w_norm.get("diversity", 0.10) * D +
            w_norm.get("novelty", 0.05) * N +
            w_norm.get("fairness", 0.05) * F
        )

        # Credibility Floor Penalty: Strongly depress articles below threshold
        if C < credibility_floor:
            final_trust_score *= 0.40

        # Conventional Click-Optimized Score (Baseline comparison)
        # Driven by raw views and sensationalism regardless of truth
        sensationalism = article.get("sensationalism_score", 0.3)
        click_potential = min(1.0, (article.get("views", 10) / 100.0) * (1.0 + sensationalism))
        conventional_click_score = (0.75 * click_potential) + (0.25 * R)

        ranked_list.append({
            **article,
            "relevance_score": round(R, 4),
            "engagement_score": round(E, 4),
            "satisfaction_score": round(S, 4),
            "credibility_score": round(C, 4),
            "diversity_score": round(D, 4),
            "novelty_score": round(N, 4),
            "fairness_score": round(F, 4),
            "trust_score": round(final_trust_score, 4),
            "conventional_click_score": round(conventional_click_score, 4),
            "ranking_factors": {
                "Relevance (R)": round(R, 3),
                "Credibility (C)": round(C, 3),
                "Long-Term Engagement (E)": round(E, 3),
                "User Satisfaction (S)": round(S, 3),
                "Diversity (D)": round(D, 3),
                "Novelty (N)": round(N, 3),
                "Fairness (F)": round(F, 3)
            }
        })

    # Sort descending by trust_score
    ranked_list = sorted(ranked_list, key=lambda x: x["trust_score"], reverse=True)

    # Attach rank numbers
    for rank, item in enumerate(ranked_list, 1):
        item["trust_rank"] = rank

    return ranked_list


def rank_conventional_click(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank articles purely using Conventional Click-Optimized strategy.
    Prioritizes virality, clicks, and sensationalism.
    """
    click_ranked = sorted(articles, key=lambda x: x.get("conventional_click_score", 0.0), reverse=True)
    for rank, item in enumerate(click_ranked, 1):
        item["click_rank"] = rank
    return click_ranked


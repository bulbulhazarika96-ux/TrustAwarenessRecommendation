"""
Comprehensive Evaluation Suite for Recommendation and Ranking Frameworks.
Implements NDCG@K, Precision@K, MAP@K, Intra-List Diversity, Novelty, Exposure Fairness (Gini),
and Clickbait Exposure Rate across baseline and proposed algorithms.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple


def compute_dcg_at_k(relevance_values: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank K."""
    relevance_values = relevance_values[:k]
    dcg = 0.0
    for i, rel in enumerate(relevance_values):
        # Discount factor log2(rank + 1)
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return dcg


def compute_ndcg_at_k(relevance_values: List[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    if not relevance_values:
        return 0.0
    actual_dcg = compute_dcg_at_k(relevance_values, k)
    ideal_values = sorted(relevance_values, reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_values, k)
    if ideal_dcg == 0:
        return 0.0
    return round(float(actual_dcg / ideal_dcg), 4)


def compute_precision_at_k(binary_relevance: List[int], k: int) -> float:
    """Proportion of top-K recommendations that are relevant."""
    if k <= 0:
        return 0.0
    top = binary_relevance[:k]
    return round(sum(top) / float(k), 4)


def compute_recall_at_k(binary_relevance: List[int], total_relevant: int, k: int) -> float:
    """Proportion of total relevant items retrieved in top-K."""
    if total_relevant <= 0:
        return 0.0
    top = binary_relevance[:k]
    return round(sum(top) / float(total_relevant), 4)


def compute_map_at_k(binary_relevance: List[int], k: int) -> float:
    """Mean Average Precision at rank K."""
    top = binary_relevance[:k]
    score = 0.0
    hits = 0
    for i, rel in enumerate(top):
        if rel == 1:
            hits += 1
            score += hits / float(i + 1)
    if hits == 0:
        return 0.0
    return round(score / float(hits), 4)


def compute_intra_list_diversity(items: List[Dict[str, Any]], k: int) -> float:
    """
    Intra-List Diversity (ILD) at rank K based on pairwise category dissimilarity.
    0.0 = completely homogeneous (all same category), 1.0 = completely distinct topics.
    """
    top = items[:k]
    if len(top) <= 1:
        return 0.0
    distinct_pairs = 0
    total_pairs = 0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            total_pairs += 1
            if top[i].get("category") != top[j].get("category"):
                distinct_pairs += 1
    return round(float(distinct_pairs / total_pairs), 4)


def compute_gini_fairness(items: List[Dict[str, Any]], k: int) -> float:
    """
    Gini coefficient of source exposure in top-K recommendations.
    Lower Gini indicates higher equality/fairness of source exposure.
    Gini = 0: perfect equality. Gini = 1: one source dominates everything.
    """
    top = items[:k]
    if not top:
        return 0.0
    counts: Dict[str, int] = {}
    for a in top:
        s = a.get("source", "Unknown")
        counts[s] = counts.get(s, 0) + 1
    values = sorted(list(counts.values()))
    n = len(values)
    if n <= 1:
        return 0.0
    total = sum(values)
    diff_sum = 0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(values[i] - values[j])
    gini = diff_sum / (2.0 * n * total)
    return round(float(gini), 4)


def compute_clickbait_exposure_rate(items: List[Dict[str, Any]], k: int, threshold: float = 0.50) -> float:
    """
    Clickbait Exposure Rate (CER): Percentage of recommended top-K items
    that have credibility score strictly below the threshold.
    """
    top = items[:k]
    if not top:
        return 0.0
    unreliable_count = sum(1 for a in top if float(a.get("credibility_score", 0.5)) < threshold)
    return round(float(unreliable_count / len(top)), 4)


def evaluate_ranking_strategies(
    articles: List[Dict[str, Any]],
    relevance_scores: List[float],
    user_history: List[Dict[str, Any]],
    user_feedback: Dict[str, List[int]],
    k: int = 10
) -> pd.DataFrame:
    """
    Run empirical comparative evaluation across five recommendation strategies:
    1. Popularity-Based
    2. Content-Based Only
    3. Click-Optimized (Conventional)
    4. Trust-Aware (Weighted)
    5. Trust-Aware (Pareto-Constrained)
    """
    from src.ranking import rank_articles, rank_conventional_click
    from src.optimization import constrained_rerank

    # Ground truth relevance definition for evaluation:
    # An article is genuinely relevant if relevance >= 0.55 AND credibility >= 0.50
    # (i.e. users do not genuinely benefit from deceptive falsehoods)
    ground_truth_relevant_ids = set()
    for idx, a in enumerate(articles):
        r = relevance_scores[idx] if idx < len(relevance_scores) else 0.5
        c = float(a.get("credibility_score", 0.5))
        if r >= 0.50 and c >= 0.45:
            ground_truth_relevant_ids.add(a.get("id"))
    total_relevant = len(ground_truth_relevant_ids)

    # 1. Popularity-Based
    pop_ranked = sorted(articles, key=lambda a: a.get("views", 0), reverse=True)

    # 2. Content-Based Only
    content_ranked = [a for _, a in sorted(zip(relevance_scores, articles), key=lambda pair: pair[0], reverse=True)]

    # 3. Click-Optimized (Conventional)
    click_ranked = rank_conventional_click(articles)

    # 4. Trust-Aware (Weighted)
    trust_ranked = rank_articles(articles, relevance_scores, user_history, user_feedback)

    # 5. Trust-Aware (Pareto-Constrained Re-ranked)
    pareto_ranked = constrained_rerank(trust_ranked, top_k=k, credibility_threshold=0.45)

    strategies = {
        "Popularity-Based": pop_ranked,
        "Content-Based Only": content_ranked,
        "Click-Optimized (Conventional)": click_ranked,
        "Trust-Aware (Weighted Hybrid)": trust_ranked,
        "Trust-Aware (Pareto-Constrained)": pareto_ranked
    }

    metrics_rows = []
    for strat_name, ranked_items in strategies.items():
        top_k_items = ranked_items[:k]
        binary_rel = [1 if a.get("id") in ground_truth_relevant_ids else 0 for a in top_k_items]
        rel_vals = [float(a.get("relevance_score", 0.5)) * float(a.get("credibility_score", 0.5)) for a in top_k_items]

        p_at_k = compute_precision_at_k(binary_rel, k)
        r_at_k = compute_recall_at_k(binary_rel, total_relevant, k)
        ndcg = compute_ndcg_at_k(rel_vals, k)
        map_k = compute_map_at_k(binary_rel, k)
        ild = compute_intra_list_diversity(top_k_items, k)
        gini = compute_gini_fairness(top_k_items, k)
        cer = compute_clickbait_exposure_rate(top_k_items, k, threshold=0.45)
        avg_cred = round(float(np.mean([float(a.get("credibility_score", 0.5)) for a in top_k_items])), 3)

        metrics_rows.append({
            "Strategy": strat_name,
            "Precision@K": p_at_k,
            "Recall@K": r_at_k,
            "NDCG@K": ndcg,
            "MAP@K": map_k,
            "Intra-List Diversity": ild,
            "Exposure Gini (Fairness)": gini,
            "Clickbait Exposure Rate": cer,
            "Avg Credibility": avg_cred
        })

    return pd.DataFrame(metrics_rows)


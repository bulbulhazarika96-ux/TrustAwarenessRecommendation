"""
Advanced Multi-Objective Optimization and Constrained Re-Ranking Module.
Implements Pareto Non-Dominated Sorting, Credibility Guardrails, and Diversity MMR Re-ranking.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def check_dominates(obj_a: List[float], obj_b: List[float]) -> bool:
    """
    Check if objective vector A Pareto-dominates objective vector B (higher is better).
    """
    all_geq = all(a >= b for a, b in zip(obj_a, obj_b))
    one_strict = any(a > b for a, b in zip(obj_a, obj_b))
    return all_geq and one_strict


def compute_pareto_fronts(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute Non-Dominated Pareto Fronts across [Relevance, Credibility, Satisfaction, Engagement].
    Assigns pareto_rank (0 = best front).
    """
    if not articles:
        return []

    # Objectives to optimize
    # [Relevance, Credibility, Satisfaction, Long-Term Engagement]
    obj_matrix = [
        [
            a.get("relevance_score", 0.5),
            a.get("credibility_score", 0.5),
            a.get("satisfaction_score", 0.5),
            a.get("engagement_score", 0.5)
        ]
        for a in articles
    ]

    n = len(articles)
    domination_counts = [0] * n
    dominated_sets = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                if check_dominates(obj_matrix[i], obj_matrix[j]):
                    dominated_sets[i].append(j)
                elif check_dominates(obj_matrix[j], obj_matrix[i]):
                    domination_counts[i] += 1

    # Front 0
    fronts = []
    current_front = [i for i in range(n) if domination_counts[i] == 0]
    front_rank = 0

    assigned_articles = [dict(a) for a in articles]

    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            assigned_articles[i]["pareto_front"] = front_rank
            assigned_articles[i]["is_pareto_optimal"] = (front_rank == 0)
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        front_rank += 1
        current_front = next_front

    # Handle any isolated cycles or remaining items
    for idx, a in enumerate(assigned_articles):
        if "pareto_front" not in a:
            a["pareto_front"] = front_rank
            a["is_pareto_optimal"] = False

    return assigned_articles


def constrained_rerank(
    ranked_articles: List[Dict[str, Any]],
    top_k: int = 15,
    credibility_threshold: float = 0.40,
    max_items_per_source: int = 3,
    diversity_lambda: float = 0.65
) -> List[Dict[str, Any]]:
    """
    Credibility- and Diversity-Aware Re-ranking.
    Enforces:
    1. Credibility Guardrail: Low-credibility items (< threshold) are filtered out of the top feed.
    2. Source Exposure Quota: Prevents publisher monopolization.
    3. Topic Diversity MMR: Balances relevance/trust with category novelty in the sequential feed.
    """
    if not ranked_articles:
        return []

    # First perform Pareto Front assignment
    articles = compute_pareto_fronts(ranked_articles)

    # Filter out articles failing credibility guardrail
    eligible_candidates = [
        a for a in articles if a.get("credibility_score", 0.5) >= credibility_threshold
    ]
    # Keep track of filtered articles for transparency
    depressed_candidates = [
        a for a in articles if a.get("credibility_score", 0.5) < credibility_threshold
    ]

    selected: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    selected_categories: Dict[str, int] = {}

    candidates_pool = list(eligible_candidates)

    while candidates_pool and len(selected) < top_k:
        best_candidate = None
        best_score = -9999.0
        best_idx = -1

        for idx, item in enumerate(candidates_pool):
            source = item.get("source", "Unknown")
            cat = item.get("category", "General News")

            # Check source cap
            if source_counts.get(source, 0) >= max_items_per_source:
                continue

            # Base score from Trust-Aware ranking + Pareto bonus
            base_score = item.get("trust_score", 0.5)
            if item.get("is_pareto_optimal", False):
                base_score += 0.08  # Pareto frontier bonus

            # Diversity penalty if category is already heavily represented in selected
            cat_count = selected_categories.get(cat, 0)
            category_penalty = cat_count * 0.12

            # MMR formulation: lambda * Quality - (1 - lambda) * Category Redundancy
            mmr_score = (diversity_lambda * base_score) - ((1.0 - diversity_lambda) * category_penalty)

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = item
                best_idx = idx

        if best_candidate is not None:
            source = best_candidate.get("source", "Unknown")
            cat = best_candidate.get("category", "General News")

            source_counts[source] = source_counts.get(source, 0) + 1
            selected_categories[cat] = selected_categories.get(cat, 0) + 1

            best_candidate["reranked_score"] = round(best_score, 4)
            best_candidate["rerank_reason"] = (
                f"Pareto Front {best_candidate.get('pareto_front', 0)}, "
                f"Credibility {best_candidate.get('credibility_score', 0):.2f}, "
                f"Topic Diversity Bonus"
            )
            selected.append(best_candidate)
            candidates_pool.pop(best_idx)
        else:
            # All remaining candidates in pool hit source cap; break
            break

    # If slots remain and pool has items, backfill without strict source cap
    while candidates_pool and len(selected) < top_k:
        item = candidates_pool.pop(0)
        item["reranked_score"] = item.get("trust_score", 0.5)
        item["rerank_reason"] = "Backfilled candidate"
        selected.append(item)

    # Attach final rerank positions
    for rank, item in enumerate(selected, 1):
        item["rerank_position"] = rank

    return selected


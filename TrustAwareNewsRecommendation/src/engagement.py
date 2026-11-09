"""
Long-Term User Engagement Modeling.
Analyzes reading dwell time, article completion rate, repeat visits, and temporal decay.
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np


def estimate_reading_completion(dwell_time_sec: float, word_count: int) -> float:
    """
    Calculate article reading completion rate based on dwell time and text length.
    Standard average reading speed is ~200-220 words per minute (approx. 3.5 words/sec).
    """
    if word_count <= 0:
        return 1.0
    expected_duration = max(8.0, (word_count / 3.5))
    ratio = dwell_time_sec / expected_duration
    # Diminishing returns above 1.0 (skimming or idle tab)
    if ratio > 1.2:
        return 1.0
    return max(0.05, min(1.0, round(ratio, 3)))


def is_accidental_click(dwell_time_sec: float, completion_rate: float, sensationalism_score: float) -> bool:
    """
    Detect whether an interaction was a clickbait bounce / accidental click.
    Signal: Very short dwell time (< 6 seconds) combined with high sensationalism or low completion.
    """
    if dwell_time_sec < 6.0 and (sensationalism_score > 0.50 or completion_rate < 0.15):
        return True
    return False


def compute_long_term_engagement(
    user_history: List[Dict[str, Any]],
    candidate_article: Dict[str, Any],
    session_decay_lambda: float = 0.05
) -> Dict[str, Any]:
    """
    Compute estimated sustained long-term engagement score [0, 1] for a candidate article.
    Combines:
    1. Historical dwell time average on candidate's category.
    2. Article expected completion potential.
    3. Category visit frequency (sustained habitual interest).
    4. Temporal decay across recent sessions.
    """
    candidate_category = candidate_article.get("category", "General News")
    word_count = len(candidate_article.get("text", "").split())

    if not user_history:
        # Cold start default
        baseline = 0.50
        return {
            "engagement_score": baseline,
            "completion_potential": 0.50,
            "category_affinity": 0.50,
            "visit_frequency": 0,
            "is_cold_start": True
        }

    # 1. Category visit frequency and habitual engagement
    category_reads = [h for h in user_history if h.get("category") == candidate_category]
    category_count = len(category_reads)
    category_affinity = min(1.0, category_count / max(len(user_history) * 0.35, 1.0))

    # 2. Historical completion rate in this category
    if category_reads:
        avg_completion = np.mean([h.get("completion_rate", 0.5) for h in category_reads])
        avg_dwell = np.mean([h.get("dwell_time_sec", 30.0) for h in category_reads])
    else:
        avg_completion = np.mean([h.get("completion_rate", 0.5) for h in user_history])
        avg_dwell = np.mean([h.get("dwell_time_sec", 30.0) for h in user_history])

    # 3. Expected completion potential for this article
    completion_potential = estimate_reading_completion(avg_dwell, word_count)

    # 4. Long-Term Engagement Composite
    # Penalize purely click-driven short bounces, reward sustained depth
    engagement_score = (
        0.40 * category_affinity +
        0.35 * avg_completion +
        0.25 * completion_potential
    )
    engagement_score = max(0.05, min(0.95, round(float(engagement_score), 4)))

    return {
        "engagement_score": engagement_score,
        "completion_potential": round(completion_potential, 3),
        "category_affinity": round(category_affinity, 3),
        "visit_frequency": category_count,
        "historical_avg_dwell": round(float(avg_dwell), 1),
        "is_cold_start": False
    }


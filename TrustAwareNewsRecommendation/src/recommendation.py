"""
Personalized News Recommendation Engine using Content-Based Filtering,
User Profile Vectorization, and Explicit/Implicit Feedback Modeling.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.prediction import get_model_and_vectorizer
from database.db import (
    get_articles,
    get_user_history,
    get_user_feedback_summary,
    get_user_preferences,
    get_article_by_id
)


def build_user_profile_vector(
    user_id: int,
    vectorizer: TfidfVectorizer,
    preferred_categories: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Construct a dense semantic preference vector representing user interests.
    Synthesizes explicit category choices, reading history, and weighted feedback.
    """
    history = get_user_history(user_id, limit=30)
    feedback = get_user_feedback_summary(user_id)

    profile_text_chunks = []
    chunk_weights = []

    # 1. Explicit Preferred Categories (Base interest anchor)
    if preferred_categories:
        cat_text = " ".join(preferred_categories * 3)
        profile_text_chunks.append(cat_text)
        chunk_weights.append(1.5)

    # 2. Reading History (Implicit signals)
    for h in history:
        art = get_article_by_id(h["article_id"])
        if art:
            # Dwell time & completion weighting
            comp = h.get("completion_rate", 0.5)
            weight = max(0.2, comp)
            content = f"{art.get('title', '')} {art.get('category', '')}"
            profile_text_chunks.append(content)
            chunk_weights.append(weight)

    # 3. Positive Feedback (Like = +2.0, Save = +3.0)
    for art_id in feedback.get("like", []):
        art = get_article_by_id(art_id)
        if art:
            profile_text_chunks.append(f"{art.get('title', '')} {art.get('category', '')}")
            chunk_weights.append(2.0)

    for art_id in feedback.get("save", []):
        art = get_article_by_id(art_id)
        if art:
            profile_text_chunks.append(f"{art.get('title', '')} {art.get('category', '')}")
            chunk_weights.append(3.0)

    # If user has no interactions and no categories, return None (cold start)
    if not profile_text_chunks:
        return None

    # Vectorize chunks
    tfidf_chunks = vectorizer.transform(profile_text_chunks)
    weights = np.array(chunk_weights).reshape(-1, 1)

    # Weighted average vector
    weighted_vec = np.sum(tfidf_chunks.toarray() * weights, axis=0, keepdims=True)
    norm = np.linalg.norm(weighted_vec)
    if norm > 0:
        weighted_vec = weighted_vec / norm

    # Negative feedback suppression (Dislikes)
    disliked_ids = feedback.get("dislike", [])
    if disliked_ids:
        dislike_texts = []
        for d_id in disliked_ids:
            art = get_article_by_id(d_id)
            if art:
                dislike_texts.append(f"{art.get('title', '')} {art.get('category', '')}")
        if dislike_texts:
            dislike_vecs = vectorizer.transform(dislike_texts).toarray()
            dislike_mean = np.mean(dislike_vecs, axis=0, keepdims=True)
            weighted_vec = weighted_vec - (0.4 * dislike_mean)
            # Clip negative elements to zero
            weighted_vec = np.clip(weighted_vec, 0, None)
            norm = np.linalg.norm(weighted_vec)
            if norm > 0:
                weighted_vec = weighted_vec / norm

    return weighted_vec


def compute_content_relevance(
    user_vector: Optional[np.ndarray],
    articles: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    preferred_categories: Optional[List[str]] = None
) -> List[float]:
    """
    Compute cosine similarity relevance scores between user profile and candidate articles.
    """
    if not articles:
        return []

    texts = [f"{a.get('title', '')} {a.get('category', '')} {a.get('text', '')[:300]}" for a in articles]
    art_vecs = vectorizer.transform(texts)

    if user_vector is None:
        # Cold start: category matching and baseline relevance
        prefs = set(preferred_categories or [])
        relevance_scores = []
        for a in articles:
            base = 0.80 if a.get("category") in prefs else 0.40
            relevance_scores.append(base)
        return relevance_scores

    sims = cosine_similarity(user_vector, art_vecs)[0]
    # Scale similarities to [0.1, 0.95] for consistent normalized scoring
    scaled_scores = [max(0.05, min(0.98, float(s))) for s in sims]
    return scaled_scores


def get_trending_articles(articles: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Rank articles purely by viral momentum / views and likes."""
    def popularity_key(a):
        return a.get("views", 0) + (a.get("likes", 0) * 3) - (a.get("dislikes", 0) * 2)

    return sorted(articles, key=popularity_key, reverse=True)[:top_k]


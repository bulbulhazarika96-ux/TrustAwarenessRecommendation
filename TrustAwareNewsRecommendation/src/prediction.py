"""
Real-time fake news inference and feature explanation module.
"""

import os
import warnings
import joblib
import numpy as np
from typing import Dict, Any, List

# Suppress sklearn deprecation warnings for legacy saved models
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from config import BEST_MODEL_PATH, TFIDF_VECTORIZER_PATH
from src.preprocessing import clean_text, extract_linguistic_signals

# Module-level cache
_CACHED_MODEL = None
_CACHED_VECTORIZER = None


def get_model_and_vectorizer():
    """Load and cache the trained fake news model and TF-IDF vectorizer."""
    global _CACHED_MODEL, _CACHED_VECTORIZER
    if _CACHED_MODEL is None or _CACHED_VECTORIZER is None:
        if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(TFIDF_VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Model or vectorizer not found at {BEST_MODEL_PATH}. "
                "Please run 'python -m src.train_model' first to train models."
            )
        _CACHED_MODEL = joblib.load(BEST_MODEL_PATH)
        _CACHED_VECTORIZER = joblib.load(TFIDF_VECTORIZER_PATH)
    return _CACHED_MODEL, _CACHED_VECTORIZER


def explain_prediction(model, vectorizer, text_vector, feature_names, top_n: int = 5) -> Dict[str, List[tuple]]:
    """
    Extract top words in the input text that contributed to FAKE or REAL prediction.
    """
    indices = text_vector.nonzero()[1]
    words_in_doc = [(feature_names[idx], idx) for idx in indices]

    # For linear models like LogisticRegression
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        # Positive coef indicates class 1 (FAKE), negative indicates class 0 (REAL)
        scored_words = [(word, float(coefs[idx])) for word, idx in words_in_doc]
        fake_cues = sorted([sw for sw in scored_words if sw[1] > 0], key=lambda x: x[1], reverse=True)[:top_n]
        real_cues = sorted([sw for sw in scored_words if sw[1] < 0], key=lambda x: x[1])[:top_n]
        return {
            "fake_indicators": fake_cues,
            "real_indicators": [(w, abs(s)) for w, s in real_cues]
        }
    return {"fake_indicators": [], "real_indicators": []}


def predict_news(title: str, text: str) -> Dict[str, Any]:
    """
    Predict whether a news article is REAL or FAKE.
    Returns prediction probabilities, confidence, linguistic markers, and explanations.
    """
    model, vectorizer = get_model_and_vectorizer()

    # Clean input
    clean_t = clean_text(title, remove_wire_prefix=False)
    clean_b = clean_text(text, remove_wire_prefix=True)
    full_content = (clean_t + " " + clean_b).strip()

    if not full_content:
        return {
            "label": "UNKNOWN",
            "fake_prob": 0.5,
            "real_prob": 0.5,
            "confidence": 0.5,
            "linguistic_signals": {},
            "top_features": {"fake_indicators": [], "real_indicators": []},
            "error": "Empty text provided."
        }

    # Extract linguistic signals
    ling_signals = extract_linguistic_signals(title, text)

    # Transform
    vec = vectorizer.transform([full_content])

    # Probabilities: class 0 = REAL, class 1 = FAKE
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        fake_prob = float(1 / (1 + np.exp(-score)))
        real_prob = 1.0 - fake_prob
    else:
        pred = model.predict(vec)[0]
        fake_prob = 1.0 if pred == 1 else 0.0
        real_prob = 1.0 - fake_prob

    # Predicted class
    label = "FAKE" if fake_prob >= 0.50 else "REAL"
    confidence = max(fake_prob, real_prob)

    # Explainability cues
    feature_names = vectorizer.get_feature_names_out()
    explanation = explain_prediction(model, vectorizer, vec, feature_names)

    return {
        "label": label,
        "fake_prob": round(fake_prob, 4),
        "real_prob": round(real_prob, 4),
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "linguistic_signals": ling_signals,
        "top_features": explanation,
        "content_length": len(full_content.split())
    }


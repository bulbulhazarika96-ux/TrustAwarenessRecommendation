"""
Configuration settings for the Trust-Aware News Recommendation and Fake News Detection Framework.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
ASSETS_DIR = BASE_DIR / "assets"
PAGES_DIR = BASE_DIR / "pages"

# Dataset Files
RAW_FAKE_PATH = DATASET_DIR / "Fake.csv"
RAW_TRUE_PATH = DATASET_DIR / "True.csv"
CLEANED_DATASET_PATH = DATASET_DIR / "news_dataset.csv"
SAMPLE_DATASET_PATH = DATASET_DIR / "sample_news.json"

# Database
DB_PATH = DATABASE_DIR / "news_system.db"

# Model Paths
BEST_MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_REPORT_PATH = MODELS_DIR / "model_comparison.json"
ARTICLE_EMBEDDINGS_PATH = MODELS_DIR / "article_features.pkl"

# Fake News Classification Hyperparameters
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# Multi-Objective Ranking Weights (Default Adaptive Weights)
DEFAULT_WEIGHTS = {
    "relevance": 0.25,
    "credibility": 0.30,
    "engagement": 0.15,
    "satisfaction": 0.10,
    "diversity": 0.10,
    "novelty": 0.05,
    "fairness": 0.05
}

# Credibility Thresholds
MIN_CREDIBILITY_THRESHOLD = 0.40  # Minimum credibility score to qualify for top recommendation
HIGH_CREDIBILITY_THRESHOLD = 0.75  # Highly trusted threshold

# Verified Source Reputation Lookup Table
SOURCE_REPUTATION = {
    "reuters": 0.95,
    "associated press": 0.95,
    "bbc": 0.92,
    "national public radio": 0.90,
    "bloomberg": 0.90,
    "the wall street journal": 0.88,
    "the new york times": 0.88,
    "the washington post": 0.86,
    "the guardian": 0.85,
    "politico": 0.82,
    "cnn": 0.78,
    "fox news": 0.70,
    "unknown": 0.50,
    "unverified blog": 0.30,
    "social media post": 0.25,
    "conspiracy / dubious": 0.10
}

# Standard News Categories
CATEGORIES = [
    "Politics",
    "World News",
    "Government",
    "Business & Finance",
    "Technology",
    "Science & Health",
    "General News"
]


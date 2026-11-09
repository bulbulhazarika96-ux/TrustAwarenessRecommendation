"""
Database manager for SQLite with thread-safe operations, user authentication, and data access.
"""

import sqlite3
import hashlib
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from config import DB_PATH, DEFAULT_WEIGHTS
from database.schema import (
    CREATE_USERS_TABLE,
    CREATE_ARTICLES_TABLE,
    CREATE_USER_PREFERENCES_TABLE,
    CREATE_READING_HISTORY_TABLE,
    CREATE_USER_FEEDBACK_TABLE,
    CREATE_RECOMMENDATION_LOGS_TABLE,
    INDEXES
)


def get_connection():
    """Create and return a database connection with Row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and indexes."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(CREATE_USERS_TABLE)
        cursor.execute(CREATE_ARTICLES_TABLE)
        cursor.execute(CREATE_USER_PREFERENCES_TABLE)
        cursor.execute(CREATE_READING_HISTORY_TABLE)
        cursor.execute(CREATE_USER_FEEDBACK_TABLE)
        cursor.execute(CREATE_RECOMMENDATION_LOGS_TABLE)
        for idx in INDEXES:
            cursor.execute(idx)
        conn.commit()


# ==================== User Authentication ====================

def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """Securely hash password using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt_bytes = os.urandom(16)
    else:
        salt_bytes = bytes.fromhex(salt)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 100000)
    return pwd_hash.hex(), salt_bytes.hex()


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user with hashed password."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    pwd_hash, salt = hash_password(password)
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, pwd_hash, salt)
            )
            user_id = cursor.lastrowid
            # Default preferences
            cursor.execute(
                "INSERT INTO user_preferences (user_id, preferred_categories, custom_weights) VALUES (?, ?, ?)",
                (user_id, json.dumps(["Politics", "World News", "Technology"]), json.dumps(DEFAULT_WEIGHTS))
            )
            conn.commit()
            return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    except Exception as e:
        return False, f"Registration failed: {str(e)}"


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials and return user dict if valid."""
    username = username.strip()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, salt, created_at FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return None
        stored_hash = row["password_hash"]
        salt = row["salt"]
        computed_hash, _ = hash_password(password, salt)
        if computed_hash == stored_hash:
            return dict(row)
        return None


def get_user_preferences(user_id: int) -> Dict[str, Any]:
    """Get preferred categories and custom weights for a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT preferred_categories, custom_weights FROM user_preferences WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            categories = json.loads(row["preferred_categories"]) if row["preferred_categories"] else []
            weights = json.loads(row["custom_weights"]) if row["custom_weights"] else DEFAULT_WEIGHTS
            return {"categories": categories, "weights": weights}
        return {"categories": ["Politics", "World News"], "weights": DEFAULT_WEIGHTS}


def save_user_preferences(user_id: int, categories: List[str], weights: Dict[str, float]) -> bool:
    """Save or update user preferred categories and weights."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_preferences (user_id, preferred_categories, custom_weights)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferred_categories = excluded.preferred_categories,
                    custom_weights = excluded.custom_weights
                """,
                (user_id, json.dumps(categories), json.dumps(weights))
            )
            conn.commit()
            return True
    except Exception:
        return False


# ==================== Article Queries & Seeding ====================

def seed_articles(df: pd.DataFrame, max_count: int = 5000):
    """Seed articles table from dataframe if empty."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM articles")
        count = cursor.fetchone()["count"]
        if count > 0:
            return  # Already populated

        subset = df.head(max_count)
        records = []
        for _, row in subset.iterrows():
            records.append((
                str(row.get("news_id", f"N_{_}")),
                str(row.get("title", "")),
                str(row.get("text", "")),
                str(row.get("summary", row.get("text", "")[:200] + "...")),
                str(row.get("category", "General News")),
                str(row.get("source", "Unknown")),
                str(row.get("publish_date", "2020-01-01")),
                str(row.get("label", "REAL")),
                float(row.get("fake_prob", 0.0)),
                float(row.get("credibility_score", 0.5)),
                int(row.get("views", 10)),
                float(row.get("dwell_time_avg", 35.0)),
                int(row.get("likes", 0)),
                int(row.get("dislikes", 0)),
                int(row.get("saves", 0))
            ))

        cursor.executemany(
            """
            INSERT OR IGNORE INTO articles (
                news_id, title, text, summary, category, source, publish_date,
                label, fake_prob, credibility_score, views, dwell_time_avg,
                likes, dislikes, saves
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records
        )
        conn.commit()


def get_articles(limit: int = 50, category: Optional[str] = None, search: Optional[str] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieve articles with optional filtering."""
    query = "SELECT * FROM articles WHERE 1=1"
    params = []

    if category and category != "All":
        query += " AND category = ?"
        params.append(category)

    if search and search.strip():
        query += " AND (title LIKE ? OR text LIKE ?)"
        term = f"%{search.strip()}%"
        params.extend([term, term])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_article_by_id(article_id: int) -> Optional[Dict[str, Any]]:
    """Fetch article details by internal ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


# ==================== User Interactions & Feedback ====================

def record_reading_history(user_id: int, article_id: int, dwell_time_sec: float, completion_rate: float, accidental: int = 0):
    """Log reading duration and completion rate."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reading_history (user_id, article_id, dwell_time_sec, completion_rate, accidental_click)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, article_id, dwell_time_sec, completion_rate, accidental)
        )
        # Update article views and running dwell time average
        cursor.execute(
            """
            UPDATE articles 
            SET views = views + 1,
                dwell_time_avg = (dwell_time_avg * views + ?) / (views + 1)
            WHERE id = ?
            """,
            (dwell_time_sec, article_id)
        )
        conn.commit()


def record_feedback(user_id: int, article_id: int, action: str) -> bool:
    """Record user feedback (like, dislike, save) and update article stats."""
    action = action.lower()
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO user_feedback (user_id, article_id, action) VALUES (?, ?, ?)",
                (user_id, article_id, action)
            )
            if action == "like":
                cursor.execute("UPDATE articles SET likes = likes + 1 WHERE id = ?", (article_id,))
            elif action == "dislike":
                cursor.execute("UPDATE articles SET dislikes = dislikes + 1 WHERE id = ?", (article_id,))
            elif action == "save":
                cursor.execute("UPDATE articles SET saves = saves + 1 WHERE id = ?", (article_id,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Action already recorded
            return False


def get_user_feedback_summary(user_id: int) -> Dict[str, List[int]]:
    """Get IDs of articles liked, disliked, or saved by user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT article_id, action FROM user_feedback WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        res = {"like": [], "dislike": [], "save": []}
        for r in rows:
            if r["action"] in res:
                res[r["action"]].append(r["article_id"])
        return res


def get_user_history(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent reading history with article metadata."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT h.id as history_id, h.timestamp, h.dwell_time_sec, h.completion_rate, h.accidental_click,
                   a.id as article_id, a.title, a.category, a.source, a.credibility_score, a.label
            FROM reading_history h
            JOIN articles a ON h.article_id = a.id
            WHERE h.user_id = ?
            ORDER BY h.timestamp DESC
            LIMIT ?
            """,
            (user_id, limit)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_saved_articles(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve articles saved by user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.*, f.timestamp as saved_at
            FROM user_feedback f
            JOIN articles a ON f.article_id = a.id
            WHERE f.user_id = ? AND f.action = 'save'
            ORDER BY f.timestamp DESC
            """,
            (user_id,)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_database_stats() -> Dict[str, Any]:
    """Aggregate statistics for the dashboard."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM articles")
        total_articles = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM articles WHERE label = 'REAL'")
        real_count = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM articles WHERE label = 'FAKE'")
        fake_count = cursor.fetchone()["total"]

        cursor.execute("SELECT AVG(credibility_score) as avg_cred FROM articles")
        avg_cred = cursor.fetchone()["avg_cred"] or 0.0

        cursor.execute("SELECT COUNT(*) as total FROM reading_history")
        total_reads = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM users")
        total_users = cursor.fetchone()["total"]

        return {
            "total_articles": total_articles,
            "real_count": real_count,
            "fake_count": fake_count,
            "avg_credibility": round(avg_cred, 3),
            "total_reads": total_reads,
            "total_users": total_users
        }


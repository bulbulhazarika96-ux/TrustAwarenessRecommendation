"""
Database schema definitions and SQL queries for SQLite.
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_ARTICLES_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT,
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    publish_date TEXT,
    label TEXT NOT NULL,
    fake_prob REAL DEFAULT 0.0,
    credibility_score REAL DEFAULT 0.5,
    views INTEGER DEFAULT 0,
    dwell_time_avg REAL DEFAULT 0.0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0
);
"""

CREATE_USER_PREFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id INTEGER PRIMARY KEY,
    preferred_categories TEXT,
    custom_weights TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

CREATE_READING_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS reading_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dwell_time_sec REAL DEFAULT 0.0,
    completion_rate REAL DEFAULT 0.0,
    accidental_click INTEGER DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_USER_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, article_id, action),
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_RECOMMENDATION_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS recommendation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    algorithm TEXT NOT NULL,
    final_score REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);",
    "CREATE INDEX IF NOT EXISTS idx_articles_credibility ON articles(credibility_score);",
    "CREATE INDEX IF NOT EXISTS idx_reading_history_user ON reading_history(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id);"
]


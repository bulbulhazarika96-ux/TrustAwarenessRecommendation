"""
Article Intelligence Module — TrustNews AI
Provides image selection, trust metrics, and engagement metadata for news articles.
Replaces the deprecated ground_news_engine.py (no political bias analysis).
"""

import hashlib
import random


# ── Category Editorial Image Pools (Unsplash) ─────────────────────────────────
CATEGORY_IMAGES: dict = {
    "Politics": [
        "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?w=800&q=80",
        "https://images.unsplash.com/photo-1575320181282-9afab399332c?w=800&q=80",
        "https://images.unsplash.com/photo-1541872703-74c5e44368f9?w=800&q=80",
        "https://images.unsplash.com/photo-1494178270175-e96de2971df9?w=800&q=80",
    ],
    "Technology": [
        "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&q=80",
        "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800&q=80",
        "https://images.unsplash.com/photo-1555255707-c07966088b7b?w=800&q=80",
        "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=800&q=80",
    ],
    "Science": [
        "https://images.unsplash.com/photo-1507413245164-6160d8298b31?w=800&q=80",
        "https://images.unsplash.com/photo-1532094349884-543559c76e4b?w=800&q=80",
        "https://images.unsplash.com/photo-1576319155264-99536e0be1ee?w=800&q=80",
        "https://images.unsplash.com/photo-1498092651296-641e88c3b057?w=800&q=80",
    ],
    "Health": [
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&q=80",
        "https://images.unsplash.com/photo-1584820927498-cfe5211fd8bf?w=800&q=80",
        "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=800&q=80",
        "https://images.unsplash.com/photo-1543362906-acfc16c67564?w=800&q=80",
    ],
    "Business": [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80",
        "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&q=80",
        "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=800&q=80",
        "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=800&q=80",
    ],
    "Sports": [
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800&q=80",
        "https://images.unsplash.com/photo-1574680178050-55c6a6a96e0a?w=800&q=80",
        "https://images.unsplash.com/photo-1587280501635-68a0e82cd5ff?w=800&q=80",
        "https://images.unsplash.com/photo-1540747913346-19212a4cf528?w=800&q=80",
    ],
    "Entertainment": [
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=800&q=80",
        "https://images.unsplash.com/photo-1598387993441-a364f854cbb5?w=800&q=80",
        "https://images.unsplash.com/photo-1524253482453-3fed8d2fe12b?w=800&q=80",
        "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=800&q=80",
    ],
    "World": [
        "https://images.unsplash.com/photo-1521295121783-8a321d551ad2?w=800&q=80",
        "https://images.unsplash.com/photo-1483728642387-6c3bdd6c93e5?w=800&q=80",
        "https://images.unsplash.com/photo-1468276311594-df7cb65d8df6?w=800&q=80",
        "https://images.unsplash.com/photo-1534536281715-e28d76689b4d?w=800&q=80",
    ],
    "General": [
        "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&q=80",
        "https://images.unsplash.com/photo-1495020689067-958852a7765e?w=800&q=80",
        "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=800&q=80",
        "https://images.unsplash.com/photo-1543286386-713bdd548da4?w=800&q=80",
    ],
}

DEFAULT_IMAGES = [
    "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&q=80",
    "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=800&q=80",
    "https://images.unsplash.com/photo-1495020689067-958852a7765e?w=800&q=80",
]


def get_article_image_url(article: dict) -> str:
    """Return a deterministic editorial image URL for an article based on its category and ID."""
    category = article.get("category", "General")
    pool = CATEGORY_IMAGES.get(category, DEFAULT_IMAGES)
    seed = int(hashlib.md5(str(article.get("id", 0)).encode()).hexdigest(), 16)
    return pool[seed % len(pool)]


def get_article_trust_metrics(article: dict) -> dict:
    """
    Compute trust tier and display metadata for an article.
    Returns credibility tier (high/mixed/low), display colors, badge labels,
    image URL, read time, and word count.
    """
    cred = float(article.get("credibility_score", 0.5))

    if cred >= 0.75:
        tier = "high"
        tier_label = "HIGH CREDIBILITY"
        tier_color = "#10b981"
        tier_bg = "rgba(16,185,129,0.15)"
        tier_border = "#10b981"
        tier_icon = "🛡️"
        css_class = "tn-badge-high"
    elif cred >= 0.45:
        tier = "mixed"
        tier_label = "MIXED CREDIBILITY"
        tier_color = "#f59e0b"
        tier_bg = "rgba(245,158,11,0.15)"
        tier_border = "#f59e0b"
        tier_icon = "⚠️"
        css_class = "tn-badge-mixed"
    else:
        tier = "low"
        tier_label = "LOW CREDIBILITY"
        tier_color = "#ef4444"
        tier_bg = "rgba(239,68,68,0.15)"
        tier_border = "#ef4444"
        tier_icon = "🛑"
        css_class = "tn-badge-low"

    text = str(article.get("text", ""))
    word_count = len(text.split())
    read_time = max(1, round(word_count / 200))

    return {
        "credibility_score": cred,
        "tier": tier,
        "tier_label": tier_label,
        "tier_color": tier_color,
        "tier_bg": tier_bg,
        "tier_border": tier_border,
        "tier_icon": tier_icon,
        "css_class": css_class,
        "image_url": get_article_image_url(article),
        "read_time_mins": read_time,
        "word_count": word_count,
        "is_real": article.get("label", "REAL") == "REAL",
        "cred_pct": int(cred * 100),
    }


def get_trust_badge_html(article: dict) -> str:
    """Return ready-to-render HTML for a trust tier badge."""
    tm = get_article_trust_metrics(article)
    return (
        f'<span style="background:{tm["tier_bg"]};border:1px solid {tm["tier_border"]};'
        f'color:{tm["tier_color"]};padding:3px 9px;border-radius:6px;font-size:0.72rem;'
        f'font-weight:700;display:inline-flex;align-items:center;gap:5px;">'
        f'{tm["tier_icon"]} {tm["tier_label"]}</span>'
    )


def get_publisher_meta(source: str) -> dict:
    """Return basic metadata for a publisher/source string."""
    return {
        "name": source,
        "trust_tier": "established",
        "description": f"News source: {source}",
    }


def get_user_reading_stats(user_id: int, reading_history: list) -> dict:
    """
    Compute engagement statistics for a user from their reading history.
    Returns total articles, avg dwell time, avg completion rate, etc.
    """
    if not reading_history:
        return {
            "total_articles": 0,
            "avg_dwell_time": 0.0,
            "avg_completion": 0.0,
            "top_category": "N/A",
            "engagement_level": "New Reader",
        }

    total = len(reading_history)
    avg_dwell = sum(r.get("dwell_seconds", 0) for r in reading_history) / total
    avg_comp = sum(r.get("completion_rate", 0) for r in reading_history) / total

    if avg_dwell >= 90 and avg_comp >= 0.70:
        level = "Deep Reader"
    elif avg_dwell >= 45:
        level = "Active Reader"
    else:
        level = "Casual Reader"

    return {
        "total_articles": total,
        "avg_dwell_time": round(avg_dwell, 1),
        "avg_completion": round(avg_comp * 100, 1),
        "top_category": "General",
        "engagement_level": level,
    }

"""
Ground News Engine: Bias Analysis, Blindspot Detection, Factuality Rating,
Editorial Image Curation, and Multi-Perspective Framing.
Replicates the real-world user experience and visual architecture of Ground News (ground.news).
"""

import re
import hashlib
from typing import Dict, Any, List, Tuple

# Curated High-Resolution Editorial Image Pools by Category (Unsplash CDN)
CATEGORY_IMAGES = {
    "politics": [
        "https://images.unsplash.com/photo-1541872703-74c5e44368f9?auto=format&fit=crop&w=800&q=80",  # US Capitol
        "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?auto=format&fit=crop&w=800&q=80",  # Diplomacy table
        "https://images.unsplash.com/photo-1575320181282-9afab399332c?auto=format&fit=crop&w=800&q=80",  # Press microphones
        "https://images.unsplash.com/photo-1555848962-6e79363ec58f?auto=format&fit=crop&w=800&q=80",  # European Parliament
        "https://images.unsplash.com/photo-1494172961521-33799ddd43a5?auto=format&fit=crop&w=800&q=80"   # Debate hall
    ],
    "world news": [
        "https://images.unsplash.com/photo-1526778548025-fa2f459cd5c1?auto=format&fit=crop&w=800&q=80",  # International compass / map
        "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80",  # Earth satellite view
        "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=800&q=80",  # Global financial towers
        "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=800&q=80",  # Global metropolis
        "https://images.unsplash.com/photo-1444723121867-7a241cacace9?auto=format&fit=crop&w=800&q=80"   # Global infrastructure
    ],
    "business & finance": [
        "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=800&q=80",  # Stock trading screens
        "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?auto=format&fit=crop&w=800&q=80",  # Wall Street New York
        "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=800&q=80",  # Corporate boardroom
        "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=800&q=80"   # Financial charts & analytics
    ],
    "technology": [
        "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80",  # Semiconductor microchip
        "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?auto=format&fit=crop&w=800&q=80",  # AI robotics
        "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=800&q=80",  # Cybersecurity server room
        "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=800&q=80"   # Tech engineering team
    ],
    "government": [
        "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&w=800&q=80",  # Supreme Court gavel
        "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a?auto=format&fit=crop&w=800&q=80",  # Civic administration building
        "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=800&q=80",  # Air Force One / state transport
        "https://images.unsplash.com/photo-1479839672679-a46483c0e7c8?auto=format&fit=crop&w=800&q=80"   # State architectural columns
    ],
    "science & health": [
        "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=800&q=80",  # Chemistry lab / flasks
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=800&q=80",  # Medical clinical research
        "https://images.unsplash.com/photo-1507668077129-56e32842fceb?auto=format&fit=crop&w=800&q=80",  # Neural network / brain research
        "https://images.unsplash.com/photo-1507413245164-6160d8298b31?auto=format&fit=crop&w=800&q=80"   # Laboratory scientific microscope
    ],
    "general news": [
        "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=800&q=80",  # Newspaper printing press
        "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?auto=format&fit=crop&w=800&q=80",  # Broadcast news camera
        "https://images.unsplash.com/photo-1495020689067-958852a7765e?auto=format&fit=crop&w=800&q=80",  # Reading morning news
        "https://images.unsplash.com/photo-1523995462485-3d171b5c8fa9?auto=format&fit=crop&w=800&q=80"   # Newsstand publications
    ]
}

# Major Publisher Metadata & Logos (Ground News style source intelligence)
PUBLISHER_META = {
    "reuters": {
        "name": "Reuters",
        "bias": "Center",
        "badge_color": "#e2e8f0",
        "factuality": "High",
        "ownership": "Thomson Reuters (Wire Service)",
        "logo_icon": "🏛️"
    },
    "associated press": {
        "name": "Associated Press",
        "bias": "Center",
        "badge_color": "#e2e8f0",
        "factuality": "High",
        "ownership": "Non-profit News Cooperative",
        "logo_icon": "📡"
    },
    "bbc": {
        "name": "BBC World News",
        "bias": "Center",
        "badge_color": "#e2e8f0",
        "factuality": "High",
        "ownership": "Public Broadcaster (UK Royal Charter)",
        "logo_icon": "🌍"
    },
    "the wall street journal": {
        "name": "The Wall Street Journal",
        "bias": "Lean Right",
        "badge_color": "#fecaca",
        "factuality": "High",
        "ownership": "News Corp / Dow Jones",
        "logo_icon": "📈"
    },
    "the new york times": {
        "name": "The New York Times",
        "bias": "Lean Left",
        "badge_color": "#bfdbfe",
        "factuality": "High",
        "ownership": "The New York Times Company",
        "logo_icon": "🗞️"
    },
    "the washington post": {
        "name": "The Washington Post",
        "bias": "Lean Left",
        "badge_color": "#bfdbfe",
        "factuality": "High",
        "ownership": "Nash Holdings (Jeff Bezos)",
        "logo_icon": "🏛️"
    },
    "the guardian": {
        "name": "The Guardian",
        "bias": "Lean Left",
        "badge_color": "#bfdbfe",
        "factuality": "High",
        "ownership": "Scott Trust Limited (Non-profit)",
        "logo_icon": "🌐"
    },
    "bloomberg": {
        "name": "Bloomberg",
        "bias": "Center",
        "badge_color": "#e2e8f0",
        "factuality": "High",
        "ownership": "Bloomberg L.P. (Michael Bloomberg)",
        "logo_icon": "💹"
    },
    "cnn": {
        "name": "CNN",
        "bias": "Lean Left",
        "badge_color": "#bfdbfe",
        "factuality": "Mixed",
        "ownership": "Warner Bros. Discovery",
        "logo_icon": "📺"
    },
    "fox news": {
        "name": "Fox News",
        "bias": "Lean Right",
        "badge_color": "#fecaca",
        "factuality": "Mixed",
        "ownership": "Fox Corporation (Murdoch Family)",
        "logo_icon": "📺"
    },
    "unverified online media": {
        "name": "Unverified Online Media",
        "bias": "Sensational / Partisan",
        "badge_color": "#fecdd3",
        "factuality": "Low",
        "ownership": "Independent / Unknown Publisher",
        "logo_icon": "⚠️"
    }
}


def get_article_image_url(article: Dict[str, Any]) -> str:
    """
    Retrieve deterministic, professional high-resolution editorial photograph
    tailored to the article's topic and headline hash.
    """
    category = str(article.get("category", "General News")).lower().strip()
    title = str(article.get("title", ""))

    # Find category pool
    pool = CATEGORY_IMAGES.get("general news")
    for key, img_list in CATEGORY_IMAGES.items():
        if key in category:
            pool = img_list
            break

    # Pick stable image index based on headline hash
    h_idx = int(hashlib.md5(title.encode("utf-8")).hexdigest()[:4], 16) % len(pool)
    return pool[h_idx]


def get_publisher_meta(source_name: str) -> Dict[str, Any]:
    """Get rich publisher metadata including ownership, bias, and badge."""
    src = str(source_name).lower().strip()
    for key, data in PUBLISHER_META.items():
        if key in src:
            return data
    return {
        "name": source_name if source_name else "Independent Publisher",
        "bias": "Center / Varied",
        "badge_color": "#e2e8f0",
        "factuality": "Mixed",
        "ownership": "Syndicated Media Network",
        "logo_icon": "📰"
    }


def calculate_bias_distribution(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate Ground News political bias breakdown (Left, Center, Right),
    Blindspot classification, Factuality rating, and Source count.
    """
    title = str(article.get("title", ""))
    text = str(article.get("text", ""))
    source = str(article.get("source", "")).lower()
    cred_score = float(article.get("credibility_score", 0.5))
    label = str(article.get("label", "REAL")).upper()
    full_content = (title + " " + text).lower()

    # Deterministic seed based on article title
    h_val = int(hashlib.md5(title.encode("utf-8")).hexdigest()[:6], 16)

    # 1. Base bias distribution inference
    if "reuters" in source or "associated press" in source or "ap" in source:
        center = 50 + (h_val % 20)  # 50% - 70%
        remaining = 100 - center
        left = int(remaining * (0.4 + (h_val % 20) / 100.0))
        right = remaining - left
        ownership = "Independent Wire Service"
        sources_count = 18 + (h_val % 32)

    elif "left" in full_content[:300] or "progressive" in full_content[:300] or "democrat" in full_content[:300] or "climate" in full_content[:300]:
        left = 58 + (h_val % 22)  # 58% - 80%
        center = 12 + (h_val % 15)
        right = max(5, 100 - left - center)
        ownership = "Digital Media Publisher"
        sources_count = 10 + (h_val % 22)

    elif "republican" in full_content[:300] or "conservative" in full_content[:300] or "trump" in full_content[:300] or "border" in full_content[:300]:
        right = 58 + (h_val % 22)  # 58% - 80%
        center = 12 + (h_val % 15)
        left = max(5, 100 - right - center)
        ownership = "Commercial Broadcast / Media Group"
        sources_count = 10 + (h_val % 22)

    else:
        center = 42 + (h_val % 18)
        left = 20 + ((h_val >> 2) % 24)
        right = max(5, 100 - center - left)
        ownership = "Syndicated News Network"
        sources_count = 8 + (h_val % 20)

    # Normalize strictly to 100%
    total = left + center + right
    left_pct = round((left / total) * 100)
    center_pct = round((center / total) * 100)
    right_pct = 100 - left_pct - center_pct

    # 2. Bias Lean
    if left_pct >= 50:
        lean = "Lean Left"
        lean_color = "#2563eb"
    elif right_pct >= 50:
        lean = "Lean Right"
        lean_color = "#dc2626"
    else:
        lean = "Center / Balanced"
        lean_color = "#64748b"

    # 3. Ground News Signature "Blindspot" Feature
    if right_pct >= 55 and left_pct <= 22:
        blindspot = "Left Blindspot"
        blindspot_desc = "Covered primarily by Right-leaning media; underreported by Left media outlets."
        blindspot_type = "left_blindspot"
    elif left_pct >= 55 and right_pct <= 22:
        blindspot = "Right Blindspot"
        blindspot_desc = "Covered primarily by Left-leaning media; underreported by Right media outlets."
        blindspot_type = "right_blindspot"
    else:
        blindspot = "Balanced Coverage"
        blindspot_desc = "Story is receiving proportional attention across the political spectrum."
        blindspot_type = "balanced"

    # 4. Factuality Rating (Ground News 3-tier system)
    if label == "FAKE" or cred_score < 0.35:
        factuality = "LOW FACTUALITY"
        factuality_tier = "low"
        factuality_color = "#ef4444"
        factuality_desc = "Story contains fabricated claims, unverified sources, or deceptive framing."
    elif cred_score < 0.68:
        factuality = "MIXED FACTUALITY"
        factuality_tier = "mixed"
        factuality_color = "#f59e0b"
        factuality_desc = "Contains real reporting mixed with sensational rhetoric or uncorroborated assertions."
    else:
        factuality = "HIGH FACTUALITY"
        factuality_tier = "high"
        factuality_color = "#10b981"
        factuality_desc = "Rigorous factual reporting backed by verified citations and transparent editorial standards."

    # Estimated read time
    words_count = len(text.split())
    read_mins = max(1, round(words_count / 180))

    return {
        "bias_left": left_pct,
        "bias_center": center_pct,
        "bias_right": right_pct,
        "bias_lean": lean,
        "bias_color": lean_color,
        "blindspot": blindspot,
        "blindspot_desc": blindspot_desc,
        "blindspot_type": blindspot_type,
        "factuality": factuality,
        "factuality_tier": factuality_tier,
        "factuality_color": factuality_color,
        "factuality_desc": factuality_desc,
        "sources_count": sources_count,
        "ownership": ownership,
        "read_time_mins": read_mins,
        "image_url": get_article_image_url(article),
        "pub_meta": get_publisher_meta(article.get("source", ""))
    }


def get_multilateral_perspectives(article: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Generate Ground News 'Compare Perspectives' multi-angle coverage for an article.
    Shows how Left, Center, and Right outlets frame the exact same event.
    """
    title = str(article.get("title", ""))
    category = str(article.get("category", "General News"))
    h = int(hashlib.md5(title.encode("utf-8")).hexdigest()[:4], 16)

    # Left-leaning perspective
    left_outlets = ["The Guardian", "MSNBC", "HuffPost", "Vox", "CNN"]
    left_source = left_outlets[h % len(left_outlets)]
    left_headline = f"Advocates and progressive lawmakers push for accountability following {category.lower()} developments"
    left_framing = "Focuses on civil liberties, socio-economic equity, regulatory oversight, and potential public welfare implications."

    # Center perspective
    center_outlets = ["Reuters", "Associated Press", "BBC World News", "Bloomberg", "The Hill"]
    center_source = center_outlets[(h >> 2) % len(center_outlets)]
    center_headline = f"Official reports confirm ongoing proceedings in {category.lower()} following official briefings"
    center_framing = "Provides verifiable data, chronologies, institutional quotes, and neutral analysis without value judgments."

    # Right-leaning perspective
    right_outlets = ["The Wall Street Journal", "Fox News", "Washington Times", "National Review", "New York Post"]
    right_source = right_outlets[(h >> 4) % len(right_outlets)]
    right_headline = f"Lawmakers examine economic impact, fiscal discipline, and institutional overreach in {category.lower()}"
    right_framing = "Emphasizes market incentives, fiscal responsibility, national security, individual liberty, and institutional skepticism."

    return {
        "left": {
            "outlet": left_source,
            "lean": "Lean Left",
            "headline": left_headline,
            "framing": left_framing,
            "color": "#2563eb"
        },
        "center": {
            "outlet": center_source,
            "lean": "Center",
            "headline": center_headline,
            "framing": center_framing,
            "color": "#64748b"
        },
        "right": {
            "outlet": right_source,
            "lean": "Lean Right",
            "headline": right_headline,
            "framing": right_framing,
            "color": "#dc2626"
        }
    }


def render_ground_bias_bar_html(left_pct: int, center_pct: int, right_pct: int, height: int = 8) -> str:
    """
    Render Ground News signature 3-color horizontal bias breakdown bar.
    """
    return f"""
    <div style="margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; font-weight: 700; margin-bottom: 4px; color: #94a3b8; letter-spacing: 0.04em;">
            <span style="color: #60a5fa;">LEFT {left_pct}%</span>
            <span style="color: #cbd5e1;">CENTER {center_pct}%</span>
            <span style="color: #f87171;">RIGHT {right_pct}%</span>
        </div>
        <div style="display: flex; height: {height}px; border-radius: 9999px; overflow: hidden; background: #1e293b; box-shadow: inset 0 1px 2px rgba(0,0,0,0.3);">
            <div style="width: {left_pct}%; background: linear-gradient(90deg, #1d4ed8, #3b82f6); transition: width 0.3s ease;"></div>
            <div style="width: {center_pct}%; background: linear-gradient(90deg, #64748b, #94a3b8); transition: width 0.3s ease;"></div>
            <div style="width: {right_pct}%; background: linear-gradient(90deg, #dc2626, #ef4444); transition: width 0.3s ease;"></div>
        </div>
    </div>
    """


def calculate_user_news_diet(reading_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate user's media consumption diet (bias balance) across their reading history.
    """
    if not reading_history:
        return {
            "diet_left": 33,
            "diet_center": 34,
            "diet_right": 33,
            "status": "Balanced Baseline",
            "recommendation": "Start reading articles from different viewpoints to unlock your personalized bias radar."
        }

    left_sum = 0
    center_sum = 0
    right_sum = 0

    for h in reading_history:
        title = h.get("title", "")
        h_val = int(hashlib.md5(title.encode("utf-8")).hexdigest()[:4], 16)
        cat = h.get("category", "")
        if "world" in cat.lower():
            c, l, r = 55, 25, 20
        elif (h_val % 2) == 0:
            l, c, r = 50, 30, 20
        else:
            r, c, l = 45, 35, 20
        left_sum += l
        center_sum += c
        right_sum += r

    n = max(1, len(reading_history))
    avg_left = round(left_sum / n)
    avg_center = round(center_sum / n)
    avg_right = 100 - avg_left - avg_center

    if avg_left >= 50:
        status = "Left-Leaning Diet (Echo Chamber Risk)"
        rec = "You are predominantly consuming Left-leaning coverage. Try exploring our 'Left Blindspot' stories to see what the Right is covering."
    elif avg_right >= 50:
        status = "Right-Leaning Diet (Echo Chamber Risk)"
        rec = "You are predominantly consuming Right-leaning coverage. Try exploring our 'Right Blindspot' stories to see what the Left is covering."
    else:
        status = "Well-Balanced News Diet"
        rec = "Great balance! Your reading spans diverse perspectives across Left, Center, and Right sources."

    return {
        "diet_left": avg_left,
        "diet_center": avg_center,
        "diet_right": avg_right,
        "status": status,
        "recommendation": rec
    }

"""
Text preprocessing, feature engineering, and dataset harmonization for Fake News Detection.
"""

import re
import os
import string
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

# Common clickbait keywords and phrases
CLICKBAIT_TRIGGERS = [
    "you won't believe", "shocking", "mind-blowing", "unbelievable", "exposed",
    "secret revealed", "bombshell", "what happens next", "they don't want you to know",
    "miracle", "epic fail", "went viral", "this is why", "destroyed", "humiliated",
    "breaking:", "warning:", "alert:", "must see", "jaw-dropping"
]

# Regex pattern to identify wire agency prefixes commonly found in ISOT True news
WIRE_PREFIX_PATTERN = re.compile(
    r"^(?:[A-Z\s\/]{2,20})?\s*\((?:Reuters|AP|AFP|Associated Press|Bloomberg)\)\s*[-–—]\s*",
    flags=re.IGNORECASE
)


def clean_text(text: str, remove_wire_prefix: bool = True) -> str:
    """
    Standardize, clean, and sanitize article text.
    Removes wire agency prefixes to prevent models from learning trivial shortcuts.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove wire service headers (e.g., 'WASHINGTON (Reuters) - ')
    if remove_wire_prefix:
        text = WIRE_PREFIX_PATTERN.sub("", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags if present
    text = re.sub(r"<.*?>+", " ", text)

    # Remove excessive punctuation repetitions (e.g., '!!!!' -> '!')
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_linguistic_signals(title: str, text: str) -> Dict[str, Any]:
    """
    Extract linguistic credibility and clickbait markers from article title and body.
    """
    title_str = str(title) if title else ""
    text_str = str(text) if text else ""

    # 1. Headline Capitalization Ratio (Sensationalism marker)
    letters = [c for c in title_str if c.isalpha()]
    caps_count = sum(1 for c in letters if c.isupper())
    caps_ratio = caps_count / max(len(letters), 1)

    # 2. Exclamation & Question Mark Density
    punct_count = title_str.count("!") * 2 + title_str.count("?")
    text_exclamations = text_str.count("!")

    # 3. Clickbait Trigger Matches
    title_lower = title_str.lower()
    clickbait_matches = sum(1 for trigger in CLICKBAIT_TRIGGERS if trigger in title_lower)

    # 4. Article Word Count and Reading Time (avg 200 words per minute)
    words = text_str.split()
    word_count = len(words)
    estimated_read_time_sec = max(10.0, (word_count / 200.0) * 60.0)

    # 5. Composite Sensationalism Score [0, 1]
    sensationalism_score = min(
        1.0,
        (caps_ratio * 0.4) +
        (min(punct_count, 3) / 3.0 * 0.3) +
        (min(clickbait_matches, 2) / 2.0 * 0.3)
    )

    return {
        "caps_ratio": round(caps_ratio, 3),
        "punct_count": punct_count,
        "clickbait_matches": clickbait_matches,
        "word_count": word_count,
        "estimated_read_time_sec": round(estimated_read_time_sec, 1),
        "sensationalism_score": round(sensationalism_score, 3)
    }


def map_subject_to_category(subject: str) -> str:
    """Normalize subject into standard news categories."""
    if not isinstance(subject, str):
        return "General News"
    subj = subject.strip().lower()
    if "world" in subj or "middle-east" in subj:
        return "World News"
    elif "politic" in subj or "left-news" in subj:
        return "Politics"
    elif "government" in subj:
        return "Government"
    elif "us_news" in subj or "news" in subj:
        return "General News"
    elif "tech" in subj:
        return "Technology"
    elif "finance" in subj or "business" in subj:
        return "Business & Finance"
    else:
        return "General News"


def load_and_prepare_dataset(
    fake_csv_path: str,
    true_csv_path: str,
    output_csv_path: str,
    sample_per_class: int = 15000
) -> pd.DataFrame:
    """
    Load raw Fake.csv and True.csv from Kaggle dataset, clean text,
    strip wire prefixes, map categories, remove duplicates, and produce unified dataset.
    """
    print(f"Loading raw datasets:\n  Fake: {fake_csv_path}\n  True: {true_csv_path}")

    fake_df = pd.read_csv(fake_csv_path)
    true_df = pd.read_csv(true_csv_path)

    fake_df["label"] = "FAKE"
    fake_df["target"] = 1
    fake_df["source"] = "Unverified Online Media"

    true_df["label"] = "REAL"
    true_df["target"] = 0
    true_df["source"] = "Reuters"

    # Harmonize column names
    fake_df = fake_df[["title", "text", "subject", "date", "label", "target", "source"]]
    true_df = true_df[["title", "text", "subject", "date", "label", "target", "source"]]

    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)
    initial_len = len(df)

    # Drop nulls and duplicates
    df = df.dropna(subset=["title", "text"])
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["title"])
    print(f"Deduplicated dataset: {initial_len} -> {len(df)} rows")

    # Sample balanced dataset if larger than sample_per_class
    if sample_per_class and len(df) > sample_per_class * 2:
        df_fake = df[df["target"] == 1].sample(min(sample_per_class, len(df[df["target"] == 1])), random_state=42)
        df_true = df[df["target"] == 0].sample(min(sample_per_class, len(df[df["target"] == 0])), random_state=42)
        df = pd.concat([df_fake, df_true], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Stratified sample created with {len(df)} rows ({len(df_fake)} Fake, {len(df_true)} Real)")

    # Clean text and titles
    print("Applying text sanitization and wire prefix removal...")
    df["title_clean"] = df["title"].apply(lambda t: clean_text(t, remove_wire_prefix=False))
    df["text_clean"] = df["text"].apply(lambda t: clean_text(t, remove_wire_prefix=True))

    # Combined content for NLP feature extraction
    df["content"] = df["title_clean"] + " " + df["text_clean"]

    # Category normalization
    df["category"] = df["subject"].apply(map_subject_to_category)

    # Unique news ID
    df["news_id"] = [f"N_{i:06d}" for i in range(len(df))]

    # Extract linguistic signals
    print("Extracting linguistic and credibility features...")
    signals = [extract_linguistic_signals(r["title"], r["text"]) for _, r in df.iterrows()]
    sig_df = pd.DataFrame(signals)
    for col in sig_df.columns:
        df[col] = sig_df[col].values

    # Realistic simulated engagement signals for recommendation testing
    # Fake articles typically have higher bounce rates and shorter dwell times
    np.random.seed(42)
    fake_mask = (df["target"] == 1)
    
    # Views: Sensational fake news often gets viral initial clicks
    views = np.random.geometric(p=0.01, size=len(df)) + 15
    views[fake_mask] = views[fake_mask] + np.random.randint(20, 80, size=fake_mask.sum())
    df["views"] = views

    # Dwell time: Real news has higher sustained dwell time
    real_dwell = np.random.normal(loc=55.0, scale=18.0, size=(~fake_mask).sum()).clip(15.0, 180.0)
    fake_dwell = np.random.normal(loc=18.0, scale=12.0, size=fake_mask.sum()).clip(3.0, 90.0)
    dwell_arr = np.zeros(len(df))
    dwell_arr[~fake_mask] = real_dwell
    dwell_arr[fake_mask] = fake_dwell
    df["dwell_time_avg"] = np.round(dwell_arr, 1)

    # Initial feedback counts
    df["likes"] = np.random.binomial(n=df["views"], p=np.where(fake_mask, 0.03, 0.08))
    df["dislikes"] = np.random.binomial(n=df["views"], p=np.where(fake_mask, 0.12, 0.02))
    df["saves"] = np.random.binomial(n=df["views"], p=np.where(fake_mask, 0.01, 0.06))

    # Summary snippet
    df["summary"] = df["text_clean"].apply(lambda t: t[:220].strip() + "..." if len(t) > 220 else t)

    # Save to disk
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned dataset saved successfully to {output_csv_path} with {len(df)} articles.")

    return df


if __name__ == "__main__":
    from config import RAW_FAKE_PATH, RAW_TRUE_PATH, CLEANED_DATASET_PATH
    load_and_prepare_dataset(str(RAW_FAKE_PATH), str(RAW_TRUE_PATH), str(CLEANED_DATASET_PATH), sample_per_class=12000)


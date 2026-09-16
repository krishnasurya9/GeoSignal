"""
src/preprocessing.py
Phase I → II: Data Harmonization + Goldstein Mapping + News Cleaning

Steps:
  1. Standardize GDELT columns → canonical schema
  2. Apply ISO 8601 dates, drop nulls, deduplicate
  3. Map CAMEO EventRootCode → Goldstein Scale (continuous intensity)
  4. Clean raw news text (lowercase, strip HTML, dedup by Jaccard)

Usage:
    python src/preprocessing.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_RAW, DATA_PROC, GOLDSTEIN_MAP_PATH

# ── Canonical output schema ────────────────────────────────────────────────────
CANONICAL_COLS = [
    "event_id",       # unique event identifier
    "date",           # ISO 8601 date
    "event_code",     # CAMEO 4-digit EventCode
    "event_root",     # CAMEO 2-digit root
    "goldstein",      # Goldstein intensity score (float, -10 to +10)
    "actor1",         # primary actor country/group
    "actor2",         # target actor country/group
    "action_country", # country where action occurred
    "action_lat",     # latitude
    "action_lon",     # longitude
    "avg_tone",       # average media tone
    "num_articles",   # media coverage count
    "source_url",     # originating article URL
]


# ── Goldstein mapping ─────────────────────────────────────────────────────────

def load_goldstein_map() -> dict:
    """
    Load CAMEO EventCode → Goldstein score mapping.
    Falls back to a minimal built-in dictionary if the CSV does not yet exist.
    """
    if GOLDSTEIN_MAP_PATH.exists():
        df = pd.read_csv(GOLDSTEIN_MAP_PATH, dtype={"cameo_code": str})
        return dict(zip(df["cameo_code"], df["goldstein_score"].astype(float)))

    # Minimal fallback: root-code → score (from published literature)
    logger.warning("Goldstein map CSV not found – using built-in root-code fallback.")
    return {
        "01": 3.0,  "02": 4.0,  "03": 5.0,  "04": 2.0,  "05": 7.0,
        "06": 6.0,  "07": 5.0,  "08": -2.0, "09": -5.0, "10": -3.0,
        "11": -4.0, "12": -5.0, "13": -7.0, "14": -8.0, "15": -9.0,
        "16": -5.0, "17": -6.0, "18": -7.0, "19": -8.0, "20": -10.0,
    }


# ── GDELT harmonization ───────────────────────────────────────────────────────

def harmonize_gdelt(raw_path: Path) -> pd.DataFrame:
    """
    Standardize raw GDELT parquet into the canonical schema.

    Args:
        raw_path: Path to raw GDELT parquet file.

    Returns:
        Cleaned, deduplicated DataFrame in canonical schema.
    """
    logger.info(f"Harmonizing GDELT from {raw_path}")
    df = pd.read_parquet(raw_path)

    goldstein_map = load_goldstein_map()

    out = pd.DataFrame()
    out["event_id"]       = df["GlobalEventID"].astype(str)
    out["date"]           = pd.to_datetime(df["Day"], format="%Y%m%d", errors="coerce")
    out["event_code"]     = df["EventCode"].fillna("").astype(str).str.zfill(4)
    out["event_root"]     = df["EventRootCode"].fillna("").astype(str).str.zfill(2)
    out["goldstein"]      = (
        df["GoldsteinScale"]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(out["event_root"].map(goldstein_map))
    )
    out["actor1"]         = df["Actor1Name"].fillna(df["Actor1CountryCode"]).fillna("").astype(str)
    out["actor2"]         = df["Actor2Name"].fillna(df["Actor2CountryCode"]).fillna("").astype(str)
    out["action_country"] = df["ActionGeo_CountryCode"].fillna("").astype(str)
    out["action_lat"]     = pd.to_numeric(df["ActionGeo_Lat"],  errors="coerce")
    out["action_lon"]     = pd.to_numeric(df["ActionGeo_Long"], errors="coerce")
    out["avg_tone"]       = pd.to_numeric(df["AvgTone"],        errors="coerce")
    out["num_articles"]   = pd.to_numeric(df["NumArticles"],    errors="coerce").fillna(0).astype(int)
    out["source_url"]     = df["SOURCEURL"].fillna("").astype(str)

    before = len(out)
    out.dropna(subset=["date", "event_code"], inplace=True)
    out.drop_duplicates(subset=["event_id"], inplace=True)
    logger.info(f"Dropped {before - len(out):,} rows (nulls + duplicates). Remaining: {len(out):,}")

    return out[CANONICAL_COLS].reset_index(drop=True)


# ── Text cleaning ─────────────────────────────────────────────────────────────

_HTML_RE   = re.compile(r"<[^>]+>")
_CLEAN_RE  = re.compile(r"[^a-z0-9\s]")
_SPACE_RE  = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalize a raw news article string:
      - Lowercase
      - Strip HTML tags
      - Remove non-alphanumeric characters
      - Collapse whitespace
    """
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _CLEAN_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def jaccard_similarity(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity between two strings."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate_news(news_df: pd.DataFrame,
                     text_col: str = "text",
                     threshold: float = 0.85) -> pd.DataFrame:
    """
    Remove near-duplicate news articles using Jaccard similarity.

    Args:
        news_df:   DataFrame with at least a `text_col` column.
        text_col:  Column name containing cleaned article text.
        threshold: Jaccard threshold above which articles are considered duplicate.

    Returns:
        Deduplicated DataFrame.
    """
    texts  = news_df[text_col].tolist()
    keep   = [True] * len(texts)

    for i in range(len(texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(texts)):
            if keep[j] and jaccard_similarity(texts[i], texts[j]) >= threshold:
                keep[j] = False

    result = news_df[keep].reset_index(drop=True)
    removed = len(news_df) - len(result)
    logger.info(f"News dedup: removed {removed} near-duplicates ({threshold:.0%} Jaccard threshold)")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw_gdelt = DATA_RAW / "gdelt_raw.parquet"
    if not raw_gdelt.exists():
        logger.error(f"Raw GDELT file not found: {raw_gdelt}. Run data_loader.py first.")
        return

    processed = harmonize_gdelt(raw_gdelt)
    out_path  = DATA_PROC / "processed_events.parquet"
    processed.to_parquet(out_path, index=False)
    logger.success(f"Processed events saved → {out_path} ({len(processed):,} rows)")

    # Preview
    print(processed.head(5).to_string())


if __name__ == "__main__":
    main()

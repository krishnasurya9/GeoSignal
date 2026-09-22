"""
geosignal.data.preprocessing
Phase I→II — GDELT harmonization, Goldstein mapping, text cleaning.

Canonical output schema (processed_events.parquet):
    event_id, date, event_code, event_root, goldstein,
    actor1, actor2, action_country, action_lat, action_lon,
    avg_tone, num_articles, source_url
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import DATA_RAW, DATA_PROC, GOLDSTEIN_MAP_PATH

CANONICAL_COLS = [
    "event_id", "date", "event_code", "event_root", "goldstein",
    "actor1", "actor2", "action_country", "action_lat", "action_lon",
    "avg_tone", "num_articles", "source_url",
]

# ── Goldstein mapping ─────────────────────────────────────────────────────────

def load_goldstein_map() -> dict:
    """
    Load CAMEO EventCode → Goldstein score from CSV.
    Falls back to root-code approximation (L-005) if file missing.
    """
    if GOLDSTEIN_MAP_PATH.exists():
        df = pd.read_csv(GOLDSTEIN_MAP_PATH, dtype={"cameo_code": str})
        return dict(zip(df["cameo_code"], df["goldstein_score"].astype(float)))

    logger.warning("Goldstein map CSV not found — using built-in root-code fallback (L-005).")
    return {
        "01":  3.0, "02":  4.0, "03":  5.0, "04":  2.0, "05":  7.0,
        "06":  6.0, "07":  5.0, "08": -2.0, "09": -5.0, "10": -3.0,
        "11": -4.0, "12": -5.0, "13": -7.0, "14": -8.0, "15": -9.0,
        "16": -5.0, "17": -6.0, "18": -7.0, "19": -8.0, "20":-10.0,
    }


# ── GDELT harmonization ───────────────────────────────────────────────────────

def harmonize_gdelt(raw_path: Path) -> pd.DataFrame:
    """
    Map raw GDELT parquet to the canonical 13-column event schema.

    - Uses GDELT-supplied GoldsteinScale; falls back to root-code map.
    - Drops rows with null date or event_code.
    - Deduplicates on GlobalEventID.

    Args:
        raw_path: Path to gdelt_raw.parquet

    Returns:
        Clean DataFrame in CANONICAL_COLS schema.
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

_HTML_RE  = re.compile(r"<[^>]+>")
_CLEAN_RE = re.compile(r"[^a-z0-9\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Lowercase, strip HTML, remove non-alphanumeric, collapse whitespace.
    Must NOT be applied before spaCy NER — use raw_text for NER (DEC-005).
    """
    text = str(text).lower()
    text = _HTML_RE.sub(" ", text)
    text = _CLEAN_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate_news(news_df: pd.DataFrame,
                     text_col: str = "text",
                     threshold: float = 0.85) -> pd.DataFrame:
    """Remove near-duplicate articles by token-level Jaccard similarity."""
    texts = news_df[text_col].tolist()
    keep  = [True] * len(texts)
    for i in range(len(texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(texts)):
            if keep[j] and jaccard_similarity(texts[i], texts[j]) >= threshold:
                keep[j] = False
    result = news_df[keep].reset_index(drop=True)
    logger.info(f"News dedup: removed {len(news_df) - len(result)} near-duplicates.")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw = DATA_RAW / "gdelt_raw.parquet"
    if not raw.exists():
        logger.error(f"Missing {raw} — run loader.py first.")
        return
    processed = harmonize_gdelt(raw)
    out = DATA_PROC / "processed_events.parquet"
    processed.to_parquet(out, index=False)
    logger.success(f"Saved {len(processed):,} events → {out}")


if __name__ == "__main__":
    main()

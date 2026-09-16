"""
src/news_loader.py
News Corpus Acquisition for GeoSignal

Loads unstructured historical news from GDELT Global Knowledge Graph (GKG) v1.
GKG flat files are available at: http://data.gdeltproject.org/gkg/
No BigQuery or API key required.

The GKG provides per-article metadata including:
  - Publication date
  - Source URL
  - Themes (GDELT theme taxonomy)
  - Persons, Organisations, Locations (extracted)
  - Tone and sentiment

Schema produced (news_corpus.parquet):
  article_id  | str  | Stable hash of source URL + date
  date        | date | Publication date (parsed from GKG filename / field)
  source_url  | str  | Original article URL
  themes      | str  | Raw GDELT theme string (semicolon-separated)
  persons     | str  | Raw extracted persons
  orgs        | str  | Raw extracted organisations
  locations   | str  | Raw extracted locations
  tone        | float| Goldstein-style average tone
  raw_text    | str  | Title/URL-derived text (GKG does not contain full body text)
  text        | str  | Cleaned/normalised version for BM25/TF-IDF

NOTE: GDELT GKG does NOT contain full article body text. The 'raw_text'
field is constructed from available metadata (themes, persons, orgs,
locations) to serve as a proxy for BM25 retrieval. This is explicitly
documented as a limitation. For genuine full-text retrieval, the
source_url would need to be fetched separately.

Usage:
    python src/news_loader.py --start 2024-01-01 --end 2024-03-31
"""

import argparse
import hashlib
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_RAW, DATA_PROC

# GDELT GKG v1 base URL and column definitions
GKG_BASE_URL = "http://data.gdeltproject.org/gkg/"

# GKG v1 has 15 columns (tab-separated)
# Ref: https://www.gdeltproject.org/data/lookups/GKG_1.0_Column_List.pdf
GKG_COLUMNS = [
    "Date",          # YYYYMMDDHHMMSS
    "NumArticles",   # number of source articles
    "V1Counts",      # counts string
    "V1Themes",      # semicolon-separated GDELT themes
    "V1Locations",   # location metadata
    "V1Persons",     # persons mentioned
    "V1Organisations",  # organisations mentioned
    "V1Tone",        # tone metadata (comma-separated: tone,pos,neg,polarity,arf,wrd,wordcount)
    "V1Cameoeventids",  # linked event IDs
    "V1Sources",     # source URLs (semicolon-sep)
    "V1SourceURLs",  # source URLs full
    "V1GKGRecordId", # GKG record identifier
    "V1Extras",      # extra fields
    "V1Quotations",  # quotations
    "V1AllNames",    # all person/org names
]


def _gkg_filename(date: datetime) -> str:
    """Returns the GKG v1 daily filename for a given date."""
    return f"{date.strftime('%Y%m%d')}.gkg.csv.zip"


def _stable_article_id(source_url: str, date_str: str) -> str:
    """Generate a stable article_id from URL + date."""
    raw = f"{source_url}|{date_str}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _parse_tone(tone_str: str) -> float:
    """Extract overall tone value (first field) from GKG tone string."""
    try:
        return float(tone_str.split(",")[0])
    except (ValueError, IndexError, AttributeError):
        return 0.0


def _build_proxy_text(row: pd.Series) -> str:
    """
    Build a proxy 'text' field from GKG metadata fields.

    GDELT GKG does not contain full article body text.
    This proxy is constructed from themes, persons, organisations, and locations
    to enable BM25/TF-IDF retrieval. This limitation is documented explicitly.

    The proxy text intentionally uses raw (uncleaned) strings so that NER can
    run on it with full capitalization. A cleaned version is produced separately.
    """
    parts = []
    for field in ["V1Themes", "V1Persons", "V1Organisations", "V1Locations"]:
        val = row.get(field, "")
        if pd.notna(val) and str(val).strip():
            # Replace semicolons/commas with spaces for readability
            cleaned = str(val).replace(";", " ").replace(",", " ").strip()
            parts.append(cleaned)
    return " ".join(parts)


def download_gkg_range(start: str, end: str) -> pd.DataFrame:
    """
    Download GDELT GKG v1 daily files and build the news corpus.

    Args:
        start: ISO date string 'YYYY-MM-DD'
        end:   ISO date string 'YYYY-MM-DD'

    Returns:
        DataFrame in news_corpus schema.
    """
    out_dir  = DATA_RAW / "gkg"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)
    date     = start_dt
    frames   = []

    pbar = tqdm(total=(end_dt - start_dt).days + 1, desc="GKG days")
    while date <= end_dt:
        fname     = _gkg_filename(date)
        local_zip = out_dir / fname

        if not local_zip.exists():
            url  = GKG_BASE_URL + fname
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200:
                local_zip.write_bytes(resp.content)
                logger.debug(f"Downloaded {fname}")
            else:
                logger.warning(f"GKG: {fname} not available (HTTP {resp.status_code}). Skipping.")
                date += timedelta(days=1)
                pbar.update(1)
                continue
        else:
            logger.debug(f"Using cached {fname}")

        try:
            with zipfile.ZipFile(local_zip) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f, sep="\t", header=None,
                        names=GKG_COLUMNS[:min(len(GKG_COLUMNS), 15)],
                        dtype=str, low_memory=False,
                        on_bad_lines="skip",
                    )
                    frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to parse {fname}: {e}")

        date += timedelta(days=1)
        pbar.update(1)

    pbar.close()

    if not frames:
        logger.error("No GKG data loaded.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    logger.info(f"GKG raw: {len(raw):,} records loaded.")
    return _harmonize_gkg(raw)


def _harmonize_gkg(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw GKG DataFrame into the news_corpus schema.

    Returns:
        DataFrame with columns:
          article_id, date, source_url, themes, persons, orgs,
          locations, tone, raw_text, text
    """
    from preprocessing import clean_text  # local import
    try:
        from preprocessing import clean_text
    except ImportError:
        from src.preprocessing import clean_text

    out = pd.DataFrame()

    # Parse date (GKG Date field is YYYYMMDDHHMMSS)
    out["date"] = pd.to_datetime(
        raw["Date"].str[:8], format="%Y%m%d", errors="coerce"
    )

    # Source URL (prefer V1SourceURLs, fallback to V1Sources)
    def _first_url(s: str) -> str:
        if pd.isna(s) or not str(s).strip():
            return ""
        return str(s).split(";")[0].strip()

    out["source_url"]  = raw.get("V1SourceURLs", raw.get("V1Sources", "")).apply(_first_url)
    out["themes"]      = raw.get("V1Themes", "").fillna("")
    out["persons"]     = raw.get("V1Persons", "").fillna("")
    out["orgs"]        = raw.get("V1Organisations", "").fillna("")
    out["locations"]   = raw.get("V1Locations", "").fillna("")
    out["tone"]        = raw.get("V1Tone", "").apply(_parse_tone)

    # Build proxy text from metadata (raw_text for NER; text for retrieval)
    out["raw_text"]    = raw.apply(_build_proxy_text, axis=1)
    out["text"]        = out["raw_text"].apply(
        lambda t: clean_text(t) if pd.notna(t) and t else ""
    )

    # Stable article ID
    out["article_id"]  = out.apply(
        lambda r: _stable_article_id(r["source_url"], str(r["date"])), axis=1
    )

    # Drop rows with no usable date or empty proxy text
    before = len(out)
    out.dropna(subset=["date"], inplace=True)
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    logger.info(f"GKG harmonized: {before:,} → {len(out):,} usable records.")

    cols = ["article_id", "date", "source_url", "themes", "persons",
            "orgs", "locations", "tone", "raw_text", "text"]
    return out[cols]


def main():
    parser = argparse.ArgumentParser(description="GeoSignal News Corpus Loader (GDELT GKG)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-03-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    logger.info(f"Loading GDELT GKG news corpus: {args.start} to {args.end}")
    news_df = download_gkg_range(args.start, args.end)

    if news_df.empty:
        logger.error("No news corpus built. Check network or date range.")
        return

    out_path = DATA_PROC / "news_corpus.parquet"
    news_df.to_parquet(out_path, index=False)
    logger.success(f"News corpus saved → {out_path}  ({len(news_df):,} articles)")
    logger.info("\nSample:")
    print(news_df[["date", "source_url", "tone", "text"]].head(5).to_string())

    # Document the limitation clearly
    logger.warning(
        "LIMITATION: GKG proxy text is constructed from metadata fields "
        "(themes, persons, orgs, locations) NOT full article body text. "
        "BM25/TF-IDF retrieval quality depends on metadata coverage. "
        "See docs/DATA_SOURCES.md for details."
    )


if __name__ == "__main__":
    main()

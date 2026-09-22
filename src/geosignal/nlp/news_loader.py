"""
geosignal.nlp.news_loader
GDELT GKG v1 news corpus acquisition.

Produces news_corpus.parquet with schema:
    article_id, date, source_url, themes, persons, orgs,
    locations, tone, raw_text, text

LIMITATION (L-001): GKG v1 contains no full article body text.
raw_text is a proxy constructed from V1Themes + V1Persons +
V1Organisations + V1Locations metadata fields.
"""

import argparse
import hashlib
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from config.settings import DATA_RAW, DATA_PROC, GDELT_GKG_URL
from geosignal.data.preprocessing import clean_text

GKG_COLUMNS = [
    "Date", "NumArticles", "V1Counts", "V1Themes", "V1Locations",
    "V1Persons", "V1Organisations", "V1Tone", "V1Cameoeventids",
    "V1Sources", "V1SourceURLs", "V1GKGRecordId", "V1Extras",
    "V1Quotations", "V1AllNames",
]


def download_gkg_range(start: str, end: str) -> pd.DataFrame:
    """
    Download GDELT GKG v1 daily ZIPs for [start, end].
    Files cached under data/raw/gkg/. Returns harmonized news_corpus DataFrame.
    """
    out_dir  = DATA_RAW / "gkg"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)
    date     = start_dt
    frames   = []

    pbar = tqdm(total=(end_dt - start_dt).days + 1, desc="GKG days")
    while date <= end_dt:
        fname     = f"{date.strftime('%Y%m%d')}.gkg.csv.zip"
        local_zip = out_dir / fname

        if not local_zip.exists():
            url  = GDELT_GKG_URL + fname
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200:
                local_zip.write_bytes(resp.content)
            else:
                logger.warning(f"GKG {fname}: HTTP {resp.status_code} — skipping.")
                date += timedelta(days=1)
                pbar.update(1)
                continue

        try:
            with zipfile.ZipFile(local_zip) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f, sep="\t", header=None,
                        names=GKG_COLUMNS[:15], dtype=str,
                        low_memory=False, on_bad_lines="skip",
                    )
                    frames.append(df)
        except Exception as exc:
            logger.warning(f"Failed to parse {fname}: {exc}")

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
    """Transform raw GKG rows into the news_corpus schema (vectorized)."""

    def get_col(name: str) -> pd.Series:
        return raw[name].astype("string").fillna("") if name in raw.columns \
               else pd.Series("", index=raw.index, dtype="string")

    out = pd.DataFrame(index=raw.index)
    out["date"] = pd.to_datetime(
        raw["Date"].astype("string").str[:8], format="%Y%m%d", errors="coerce"
    )

    themes    = get_col("V1Themes")
    persons   = get_col("V1Persons")
    orgs      = get_col("V1Organisations")
    locations = get_col("V1Locations")

    src_urls = get_col("V1SourceURLs").mask(
        get_col("V1SourceURLs").str.strip().eq(""), get_col("V1Sources")
    )
    out["source_url"] = src_urls.str.split(";", n=1).str[0].str.strip().fillna("")
    out["themes"]    = themes
    out["persons"]   = persons
    out["orgs"]      = orgs
    out["locations"] = locations
    out["tone"]      = pd.to_numeric(
        get_col("V1Tone").str.split(",", n=1).str[0], errors="coerce"
    ).fillna(0.0)

    # Proxy text: metadata concatenation (no full article body — L-001)
    def _clean_meta(s: pd.Series) -> pd.Series:
        return s.str.replace(";", " ", regex=False).str.replace(",", " ", regex=False).str.strip()

    out["raw_text"] = (
        _clean_meta(themes) + " " + _clean_meta(persons) + " " +
        _clean_meta(orgs)   + " " + _clean_meta(locations)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Drop unusable rows before expensive text processing
    usable = out["date"].notna() & out["raw_text"].str.strip().ne("")
    out    = out.loc[usable].copy()
    logger.info(f"GKG: {len(raw):,} → {len(out):,} usable records after filtering.")

    out["text"] = out["raw_text"].map(lambda t: clean_text(t) if t else "")
    out = out[out["text"].str.strip().ne("")].copy()

    id_input     = out["source_url"].astype(str) + "|" + out["date"].dt.strftime("%Y-%m-%d").astype(str)
    out["article_id"] = id_input.map(lambda x: hashlib.md5(x.encode()).hexdigest())

    cols = ["article_id", "date", "source_url", "themes", "persons",
            "orgs", "locations", "tone", "raw_text", "text"]
    return out[cols].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="GeoSignal — GKG news loader")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2024-03-31")
    args = parser.parse_args()

    news_df = download_gkg_range(args.start, args.end)
    if news_df.empty:
        logger.error("Empty news corpus — check network or date range.")
        return
    out = DATA_PROC / "news_corpus.parquet"
    news_df.to_parquet(out, index=False)
    logger.success(f"News corpus → {out}  ({len(news_df):,} articles)")


if __name__ == "__main__":
    main()

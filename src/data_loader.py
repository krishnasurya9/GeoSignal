"""
src/data_loader.py
Phase I: Structured Geopolitical Data Acquisition and Harmonization

Supports three sources:
  - GDELT 1.0 (CSV flat files) — primary baseline
  - ICEWS    (Harvard Dataverse API, .tsv)
  - UCDP GED (REST API, JSON)

Usage:
    python src/data_loader.py --source gdelt --start 2023-01-01 --end 2024-12-31
    python src/data_loader.py --source ucdp  --start 2023-01-01 --end 2024-12-31
"""

import argparse
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

# Import project settings
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_RAW, GDELT_BASE_URL, UCDP_API_BASE

# ── GDELT column names (GDELT 1.0 event table) ────────────────────────────────
GDELT_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

# ── GDELT ─────────────────────────────────────────────────────────────────────

def _gdelt_filename(date: datetime) -> str:
    """Returns the GDELT daily export filename for a given date."""
    return f"{date.strftime('%Y%m%d')}.export.CSV.zip"


def download_gdelt_range(start: str, end: str) -> pd.DataFrame:
    """
    Download GDELT 1.0 daily CSV files for the given date range.
    Files are stored in data/raw/gdelt/.

    Args:
        start: ISO date string, e.g. '2023-10-01'
        end:   ISO date string, e.g. '2023-10-31'

    Returns:
        Combined DataFrame of all events in the date range.
    """
    out_dir = DATA_RAW / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)
    date     = start_dt
    frames   = []

    while date <= end_dt:
        fname    = _gdelt_filename(date)
        local_zip = out_dir / fname

        if not local_zip.exists():
            url = GDELT_BASE_URL + fname
            logger.info(f"Downloading {url}")
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                local_zip.write_bytes(resp.content)
            else:
                logger.warning(f"GDELT: {fname} not found (HTTP {resp.status_code}). Skipping.")
                date += timedelta(days=1)
                continue
        else:
            logger.debug(f"Using cached {local_zip.name}")

        with zipfile.ZipFile(local_zip) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, sep="\t", header=None,
                                 names=GDELT_COLUMNS, dtype=str, low_memory=False)
                frames.append(df)

        date += timedelta(days=1)

    if not frames:
        logger.error("No GDELT data downloaded. Check date range or network.")
        return pd.DataFrame(columns=GDELT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    logger.success(f"GDELT: {len(combined):,} raw events loaded.")
    return combined


# ── UCDP GED ──────────────────────────────────────────────────────────────────

def download_ucdp(start: str, end: str) -> pd.DataFrame:
    """
    Pull UCDP Georeferenced Event Dataset (GED) via REST API.

    Args:
        start: ISO year or date string (year only is sufficient for UCDP API)
        end:   ISO year or date string

    Returns:
        DataFrame of UCDP events.
    """
    out_dir  = DATA_RAW / "ucdp"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_year = datetime.fromisoformat(start).year
    end_year   = datetime.fromisoformat(end).year

    frames = []
    for year in tqdm(range(start_year, end_year + 1), desc="UCDP years"):
        cache_path = out_dir / f"ucdp_ged_{year}.parquet"
        if cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
            continue

        url  = f"{UCDP_API_BASE}/gedevents"
        page = 1
        year_frames = []

        while True:
            params = {"year": year, "pagesize": 1000, "page": page}
            resp   = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data   = resp.json()

            if not data.get("Result"):
                break
            year_frames.append(pd.DataFrame(data["Result"]))

            # Check if there are more pages
            total   = data.get("TotalCount", 0)
            fetched = page * 1000
            if fetched >= total:
                break
            page += 1

        if year_frames:
            df = pd.concat(year_frames, ignore_index=True)
            df.to_parquet(cache_path, index=False)
            frames.append(df)
            logger.info(f"UCDP {year}: {len(df):,} events")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.success(f"UCDP GED: {len(combined):,} events total.")
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoSignal Phase I: Data Loader")
    parser.add_argument("--source", choices=["gdelt", "ucdp", "both"], default="gdelt")
    parser.add_argument("--start",  default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    if args.source in ("gdelt", "both"):
        df_gdelt = download_gdelt_range(args.start, args.end)
        out = DATA_RAW / "gdelt_raw.parquet"
        df_gdelt.to_parquet(out, index=False)
        logger.success(f"Saved raw GDELT → {out}")

    if args.source in ("ucdp", "both"):
        df_ucdp = download_ucdp(args.start, args.end)
        out = DATA_RAW / "ucdp_raw.parquet"
        df_ucdp.to_parquet(out, index=False)
        logger.success(f"Saved raw UCDP → {out}")


if __name__ == "__main__":
    main()

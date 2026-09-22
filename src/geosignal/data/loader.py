"""
geosignal.data.loader
Phase I — Structured geopolitical data acquisition.

Supports GDELT 1.0 (primary) and UCDP GED (supplementary).
Downloaded files are cached under data/raw/ and never re-downloaded.
"""

import argparse
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from config.settings import DATA_RAW, GDELT_BASE_URL, UCDP_API_BASE

# ── GDELT 1.0 column list (58 fields, tab-separated) ─────────────────────────
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

def download_gdelt_range(start: str, end: str) -> pd.DataFrame:
    """
    Download GDELT 1.0 daily export ZIPs for [start, end] (inclusive).
    Files are cached in data/raw/gdelt/. Combined DataFrame is returned
    and saved to data/raw/gdelt_raw.parquet.

    Args:
        start: 'YYYY-MM-DD'
        end:   'YYYY-MM-DD'

    Returns:
        Combined raw GDELT DataFrame (all 58 columns, string dtype).
    """
    out_dir = DATA_RAW / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)
    date     = start_dt
    frames   = []

    total_days = (end_dt - start_dt).days + 1
    pbar = tqdm(total=total_days, desc="GDELT days")

    while date <= end_dt:
        fname     = f"{date.strftime('%Y%m%d')}.export.CSV.zip"
        local_zip = out_dir / fname

        if not local_zip.exists():
            url  = GDELT_BASE_URL + fname
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                local_zip.write_bytes(resp.content)
            else:
                logger.warning(f"GDELT {fname}: HTTP {resp.status_code} — skipping.")
                date += timedelta(days=1)
                pbar.update(1)
                continue
        else:
            logger.debug(f"Cached: {fname}")

        try:
            with zipfile.ZipFile(local_zip) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".CSV"))
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f, sep="\t", header=None,
                        names=GDELT_COLUMNS, dtype=str, low_memory=False,
                    )
                    frames.append(df)
        except Exception as exc:
            logger.warning(f"Failed to parse {fname}: {exc}")

        date += timedelta(days=1)
        pbar.update(1)

    pbar.close()

    if not frames:
        logger.error("No GDELT data loaded — check date range or network.")
        return pd.DataFrame(columns=GDELT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    out_path = DATA_RAW / "gdelt_raw.parquet"
    combined.to_parquet(out_path, index=False)
    logger.success(f"GDELT: {len(combined):,} events → {out_path}")
    return combined


# ── UCDP GED ──────────────────────────────────────────────────────────────────

def download_ucdp(start: str, end: str) -> pd.DataFrame:
    """
    Pull UCDP Georeferenced Event Dataset via REST API, year by year.
    Supplementary source — not used in the primary dev pipeline.

    Args:
        start: 'YYYY-MM-DD'  (year extracted)
        end:   'YYYY-MM-DD'  (year extracted)

    Returns:
        Combined UCDP GED DataFrame.
    """
    out_dir = DATA_RAW / "ucdp"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_year = datetime.fromisoformat(start).year
    end_year   = datetime.fromisoformat(end).year
    frames     = []

    for year in tqdm(range(start_year, end_year + 1), desc="UCDP years"):
        cache = out_dir / f"ucdp_ged_{year}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue

        url = f"{UCDP_API_BASE}/gedevents"
        page, year_frames = 1, []
        while True:
            resp = requests.get(url, params={"year": year, "pagesize": 1000, "page": page}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("Result"):
                break
            year_frames.append(pd.DataFrame(data["Result"]))
            if page * 1000 >= data.get("TotalCount", 0):
                break
            page += 1

        if year_frames:
            df = pd.concat(year_frames, ignore_index=True)
            df.to_parquet(cache, index=False)
            frames.append(df)
            logger.info(f"UCDP {year}: {len(df):,} events")

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    logger.success(f"UCDP GED: {len(combined):,} events total.")
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoSignal — data loader")
    parser.add_argument("--source", choices=["gdelt", "ucdp", "both"], default="gdelt")
    parser.add_argument("--start",  default="2024-01-01")
    parser.add_argument("--end",    default="2024-03-31")
    args = parser.parse_args()

    if args.source in ("gdelt", "both"):
        download_gdelt_range(args.start, args.end)
    if args.source in ("ucdp", "both"):
        df = download_ucdp(args.start, args.end)
        (DATA_RAW / "ucdp_raw.parquet").write_bytes(b"")  # placeholder
        df.to_parquet(DATA_RAW / "ucdp_raw.parquet", index=False)


if __name__ == "__main__":
    main()

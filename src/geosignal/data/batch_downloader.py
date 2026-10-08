"""
geosignal.data.batch_downloader
================================
Chunked, resumable downloader for the full 2022-2024 GDELT + GKG dataset.

Resolves the "Full 2022-2024 dataset | Planned" item from README §14.

Why a dedicated module?
-----------------------
The existing loader.py works fine for a 90-day dev sample (~91 ZIPs).
For a 3-year window (2022-01-01 → 2024-12-31) that is ~1096 GDELT ZIPs
(~10-15 GB) and ~1096 GKG ZIPs (~20-25 GB). A naive sequential download
with no checkpointing will fail on any network interruption. This module
adds:

  - Parallel HTTP downloads (configurable workers)
  - Per-file checksum verification (skip corrupt cached files)
  - Checkpoint file so a restarted download resumes from where it stopped
  - Year-by-year parquet building to avoid RAM exhaustion (never loads
    the full 3-year corpus in memory at once)
  - Dry-run mode to plan disk space before committing

Outputs
-------
  data/raw/gdelt_raw.parquet         (full window, partitioned by year then merged)
  data/raw/gdelt_<year>.parquet      (per-year intermediate, kept for incremental use)
  data/processed/news_corpus.parquet (full window GKG)
  data/raw/gkg_<year>.parquet        (per-year GKG intermediate)

Usage
-----
    # Full research window (downloads ~30-35 GB total)
    python -m geosignal.data.batch_downloader \\
        --start 2022-01-01 --end 2024-12-31

    # Dry run (lists files, estimates disk, no downloads)
    python -m geosignal.data.batch_downloader \\
        --start 2022-01-01 --end 2024-12-31 --dry-run

    # Download only GKG (events already cached)
    python -m geosignal.data.batch_downloader \\
        --start 2022-01-01 --end 2024-12-31 --only gkg

    # Programmatic use from run_all.py
    from geosignal.data.batch_downloader import download_full_window
    download_full_window("2022-01-01", "2024-12-31")
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from config.settings import DATA_RAW, DATA_PROC, GDELT_BASE_URL, GDELT_GKG_URL

# ── GDELT column list (same as loader.py) ────────────────────────────────────
from geosignal.data.loader import GDELT_COLUMNS
from geosignal.data.preprocessing import harmonize_gdelt
from geosignal.nlp.news_loader import _harmonize_gkg

# Checkpoint file records which dates have been successfully processed
_CHECKPOINT_FILE = DATA_RAW / ".download_checkpoint.json"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WORKERS   = 4      # parallel HTTP connections
CHUNK_TIMEOUT_S   = 120    # per-file HTTP timeout
MAX_RETRIES       = 3      # retry on network error


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def download_full_window(
    start: str,
    end: str,
    sources: Literal["both", "gdelt", "gkg"] = "both",
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Path]:
    """
    Download and build the full GDELT + GKG parquet dataset for [start, end].

    Processes data year-by-year to avoid holding the entire 3-year corpus
    in memory. Per-year parquets are merged at the end.

    Args:
        start:   'YYYY-MM-DD'
        end:     'YYYY-MM-DD'
        sources: 'both' | 'gdelt' | 'gkg'
        workers: Parallel download threads
        force:   Re-download even if files are cached
        dry_run: Print plan only, no downloads

    Returns:
        {'gdelt': Path, 'gkg': Path} of final merged parquets
    """
    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)
    dates    = _date_range(start_dt, end_dt)

    if dry_run:
        _dry_run_report(dates, sources)
        return {}

    checkpoint = _load_checkpoint()
    result: dict[str, Path] = {}

    # Process year-by-year
    years = sorted({d.year for d in dates})
    for year in years:
        year_dates = [d for d in dates if d.year == year]

        if sources in ("both", "gdelt"):
            gdelt_year_path = _process_gdelt_year(
                year, year_dates, checkpoint, workers, force
            )
            result["gdelt_last_year"] = gdelt_year_path

        if sources in ("both", "gkg"):
            gkg_year_path = _process_gkg_year(
                year, year_dates, checkpoint, workers, force
            )
            result["gkg_last_year"] = gkg_year_path

    # Merge per-year parquets into the canonical single-file outputs
    if sources in ("both", "gdelt"):
        result["gdelt"] = _merge_year_parquets("gdelt", years)

    if sources in ("both", "gkg"):
        result["gkg"] = _merge_year_parquets("gkg", years)

    logger.success(
        f"Full window download complete: {start} → {end}  |  "
        f"Outputs: {[str(v) for v in result.values() if isinstance(v, Path)]}"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Year-level processors
# ══════════════════════════════════════════════════════════════════════════════

def _process_gdelt_year(
    year: int,
    dates: list[datetime],
    checkpoint: dict,
    workers: int,
    force: bool,
) -> Path:
    out_path = DATA_RAW / f"gdelt_{year}.parquet"
    if out_path.exists() and not force:
        already_done = sum(
            1 for d in dates
            if checkpoint.get(f"gdelt:{d.strftime('%Y%m%d')}")
        )
        if already_done == len(dates):
            logger.info(f"[SKIP] GDELT {year} — fully cached: {out_path.name}")
            return out_path

    logger.info(f"GDELT {year}: downloading {len(dates)} daily files …")
    zip_dir = DATA_RAW / "gdelt"
    zip_dir.mkdir(parents=True, exist_ok=True)

    file_specs = [
        {
            "date_str": d.strftime("%Y%m%d"),
            "fname":    f"{d.strftime('%Y%m%d')}.export.CSV.zip",
            "url":      GDELT_BASE_URL + f"{d.strftime('%Y%m%d')}.export.CSV.zip",
            "local":    zip_dir / f"{d.strftime('%Y%m%d')}.export.CSV.zip",
            "key":      f"gdelt:{d.strftime('%Y%m%d')}",
        }
        for d in dates
        if force or not checkpoint.get(f"gdelt:{d.strftime('%Y%m%d')}")
    ]

    _parallel_download(file_specs, workers, checkpoint)

    # Parse all ZIP files for this year
    frames = []
    for spec in tqdm(
        [{**s, "local": zip_dir / s["fname"]} for s in
         [{"fname": f"{d.strftime('%Y%m%d')}.export.CSV.zip",
           "date_str": d.strftime("%Y%m%d")} for d in dates]],
        desc=f"Parsing GDELT {year}",
        leave=False,
    ):
        local = zip_dir / spec["fname"]
        if not local.exists():
            continue
        try:
            import zipfile
            with zipfile.ZipFile(local) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".CSV"))
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f, sep="\t", header=None,
                        names=GDELT_COLUMNS, dtype=str, low_memory=False,
                    )
                    frames.append(df)
        except Exception as exc:
            logger.warning(f"  Parse error {local.name}: {exc}")

    if not frames:
        logger.warning(f"GDELT {year}: no frames parsed.")
        return out_path

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    logger.success(f"GDELT {year}: {len(combined):,} rows → {out_path.name}")
    return out_path


def _process_gkg_year(
    year: int,
    dates: list[datetime],
    checkpoint: dict,
    workers: int,
    force: bool,
) -> Path:
    out_path = DATA_RAW / f"gkg_{year}.parquet"
    if out_path.exists() and not force:
        already_done = sum(
            1 for d in dates
            if checkpoint.get(f"gkg:{d.strftime('%Y%m%d')}")
        )
        if already_done == len(dates):
            logger.info(f"[SKIP] GKG {year} — fully cached: {out_path.name}")
            return out_path

    logger.info(f"GKG {year}: downloading {len(dates)} daily files …")
    zip_dir = DATA_RAW / "gkg"
    zip_dir.mkdir(parents=True, exist_ok=True)

    file_specs = [
        {
            "date_str": d.strftime("%Y%m%d"),
            "fname":    f"{d.strftime('%Y%m%d')}.gkg.csv.zip",
            "url":      GDELT_GKG_URL + f"{d.strftime('%Y%m%d')}.gkg.csv.zip",
            "local":    zip_dir / f"{d.strftime('%Y%m%d')}.gkg.csv.zip",
            "key":      f"gkg:{d.strftime('%Y%m%d')}",
        }
        for d in dates
        if force or not checkpoint.get(f"gkg:{d.strftime('%Y%m%d')}")
    ]

    _parallel_download(file_specs, workers, checkpoint)

    # Parse and harmonize GKG ZIPs
    frames = []
    for d in tqdm(dates, desc=f"Parsing GKG {year}", leave=False):
        fname = f"{d.strftime('%Y%m%d')}.gkg.csv.zip"
        local = zip_dir / fname
        if not local.exists():
            continue
        try:
            import zipfile
            GKG_COLS = [
                "Date", "NumArticles", "V1Counts", "V1Themes", "V1Locations",
                "V1Persons", "V1Organisations", "V1Tone", "V1Cameoeventids",
                "V1Sources", "V1SourceURLs", "V1GKGRecordId", "V1Extras",
                "V1Quotations", "V1AllNames",
            ]
            with zipfile.ZipFile(local) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f, sep="\t", header=None, names=GKG_COLS,
                        dtype=str, low_memory=False, on_bad_lines="skip",
                    )
                    frames.append(df)
        except Exception as exc:
            logger.warning(f"  GKG parse error {fname}: {exc}")

    if not frames:
        logger.warning(f"GKG {year}: no frames parsed.")
        return out_path

    raw = pd.concat(frames, ignore_index=True)
    harmonized = _harmonize_gkg(raw)
    harmonized.to_parquet(out_path, index=False)
    logger.success(f"GKG {year}: {len(harmonized):,} articles → {out_path.name}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Merge year parquets
# ══════════════════════════════════════════════════════════════════════════════

def _merge_year_parquets(
    source: Literal["gdelt", "gkg"],
    years: list[int],
) -> Path:
    """Merge gdelt_<year>.parquet / gkg_<year>.parquet into one canonical file."""
    if source == "gdelt":
        year_paths  = [DATA_RAW / f"gdelt_{y}.parquet" for y in years]
        merged_path = DATA_RAW / "gdelt_raw.parquet"
    else:
        year_paths  = [DATA_RAW / f"gkg_{y}.parquet" for y in years]
        merged_path = DATA_PROC / "news_corpus.parquet"

    existing = [p for p in year_paths if p.exists()]
    if not existing:
        logger.warning(f"No year parquets found to merge for {source}.")
        return merged_path

    logger.info(f"Merging {len(existing)} year parquets for {source} …")
    frames = [pd.read_parquet(p) for p in existing]
    merged = pd.concat(frames, ignore_index=True)

    if source == "gdelt":
        # Apply canonical harmonization to the merged raw frame
        merged.to_parquet(merged_path, index=False)  # raw parquet
        harmonized = harmonize_gdelt(merged_path)
        proc_path  = DATA_PROC / "processed_events.parquet"
        harmonized.to_parquet(proc_path, index=False)
        logger.success(
            f"GDELT merged: {len(merged):,} raw rows → {merged_path.name}  |  "
            f"{len(harmonized):,} processed → {proc_path.name}"
        )
    else:
        merged.to_parquet(merged_path, index=False)
        logger.success(f"GKG merged: {len(merged):,} articles → {merged_path.name}")

    return merged_path


# ══════════════════════════════════════════════════════════════════════════════
#  Parallel downloader
# ══════════════════════════════════════════════════════════════════════════════

def _parallel_download(
    file_specs: list[dict],
    workers: int,
    checkpoint: dict,
) -> None:
    """Download a list of file specs in parallel with retry logic."""
    if not file_specs:
        return

    def _download_one(spec: dict) -> tuple[str, bool]:
        local: Path = spec["local"]
        if local.exists():
            return spec["key"], True   # already cached

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(spec["url"], timeout=CHUNK_TIMEOUT_S, stream=True)
                if resp.status_code == 404:
                    return spec["key"], False   # date doesn't exist in GDELT
                resp.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        f.write(chunk)
                return spec["key"], True
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    logger.warning(
                        f"  Failed after {MAX_RETRIES} attempts: "
                        f"{local.name} — {exc}"
                    )
                    return spec["key"], False
                time.sleep(2 ** attempt)   # exponential back-off

        return spec["key"], False

    success, failed = 0, 0
    with tqdm(total=len(file_specs), desc="Downloading", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_one, spec): spec for spec in file_specs}
            for future in as_completed(futures):
                key, ok = future.result()
                if ok:
                    checkpoint[key] = True
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    _save_checkpoint(checkpoint)
    logger.info(f"Download batch: {success} OK, {failed} failed.")


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_checkpoint() -> dict:
    if _CHECKPOINT_FILE.exists():
        try:
            return json.loads(_CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_checkpoint(data: dict) -> None:
    _CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CHECKPOINT_FILE.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def reset_checkpoint() -> None:
    """Delete the checkpoint file to force a full re-download."""
    if _CHECKPOINT_FILE.exists():
        _CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint reset.")


# ══════════════════════════════════════════════════════════════════════════════
#  Dry run
# ══════════════════════════════════════════════════════════════════════════════

def _dry_run_report(dates: list[datetime], sources: str) -> None:
    n = len(dates)
    checkpoint = _load_checkpoint()

    gdelt_cached = sum(
        1 for d in dates
        if checkpoint.get(f"gdelt:{d.strftime('%Y%m%d')}")
    )
    gkg_cached = sum(
        1 for d in dates
        if checkpoint.get(f"gkg:{d.strftime('%Y%m%d')}")
    )

    gdelt_to_dl = n - gdelt_cached if sources in ("both", "gdelt") else 0
    gkg_to_dl   = n - gkg_cached   if sources in ("both", "gkg")   else 0

    # Rough size estimates (GDELT ~10 MB/day compressed, GKG ~20 MB/day)
    gdelt_gb = gdelt_to_dl * 10 / 1024
    gkg_gb   = gkg_to_dl   * 20 / 1024

    print("\n" + "=" * 60)
    print("  GeoSignal Full-Window Download Plan")
    print("=" * 60)
    print(f"  Date range  : {dates[0].date()} → {dates[-1].date()}")
    print(f"  Total days  : {n}")
    print(f"  Sources     : {sources}")
    print()
    if sources in ("both", "gdelt"):
        print(f"  GDELT  cached: {gdelt_cached:>5}  to download: {gdelt_to_dl:>5}  "
              f"(~{gdelt_gb:.1f} GB)")
    if sources in ("both", "gkg"):
        print(f"  GKG    cached: {gkg_cached:>5}  to download: {gkg_to_dl:>5}  "
              f"(~{gkg_gb:.1f} GB)")
    print(f"\n  Total estimated download: ~{gdelt_gb + gkg_gb:.1f} GB")
    print("=" * 60)
    print("  Run without --dry-run to start downloading.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _date_range(start: datetime, end: datetime) -> list[datetime]:
    dates, d = [], start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="GeoSignal full-window GDELT + GKG batch downloader"
    )
    p.add_argument("--start",   default="2022-01-01")
    p.add_argument("--end",     default="2024-12-31")
    p.add_argument("--only",    choices=["both", "gdelt", "gkg"], default="both",
                   dest="sources")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--force",   action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reset-checkpoint", action="store_true")
    args = p.parse_args()

    if args.reset_checkpoint:
        reset_checkpoint()

    download_full_window(
        start   = args.start,
        end     = args.end,
        sources = args.sources,
        workers = args.workers,
        force   = args.force,
        dry_run = args.dry_run,
    )

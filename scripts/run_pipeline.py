"""
scripts/run_pipeline.py
Full GeoSignal pipeline — runs all stages end-to-end.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --start 2024-01-01 --end 2024-03-31
    python scripts/run_pipeline.py --start 2022-01-01 --end 2024-12-31  # full research window
    python scripts/run_pipeline.py --skip-download   # reuse cached raw data
    python scripts/run_pipeline.py --skip-nlp        # reuse existing event_news_pairs.parquet

Each stage reads clearly defined inputs and writes a named output to
data/interim/ or data/processed/. Completed stages are skipped if their
output already exists (use --force to rerun).
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from config.settings import DATA_RAW, DATA_PROC, RESULTS_DIR, NSE_SECTOR_TICKERS, NIFTY50_TICKER

INTERIM  = PROJECT_ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

# ── Stage helpers ─────────────────────────────────────────────────────────────

def _skip(path: Path, force: bool, label: str) -> bool:
    if path.exists() and not force:
        logger.info(f"[SKIP] {label} — output exists: {path.name}")
        return True
    return False


# ── Stage 1: Download GDELT ───────────────────────────────────────────────────

def stage_download_gdelt(start: str, end: str, force: bool):
    out = DATA_RAW / "gdelt_raw.parquet"
    if _skip(out, force, "Stage 1: GDELT download"):
        return
    logger.info(f"Stage 1: Downloading GDELT {start} → {end}")
    from geosignal.data.loader import download_gdelt_range
    download_gdelt_range(start, end)


# ── Stage 2: Download GKG news corpus ────────────────────────────────────────

def stage_download_gkg(start: str, end: str, force: bool):
    out = DATA_PROC / "news_corpus.parquet"
    if _skip(out, force, "Stage 2: GKG news corpus"):
        return
    logger.info(f"Stage 2: Downloading GKG {start} → {end}")
    from geosignal.nlp.news_loader import download_gkg_range
    news_df = download_gkg_range(start, end)
    news_df.to_parquet(out, index=False)
    logger.success(f"Stage 2 complete → {out}  ({len(news_df):,} articles)")


# ── Stage 3: Preprocess events ────────────────────────────────────────────────

def stage_preprocess(force: bool):
    out = DATA_PROC / "processed_events.parquet"
    if _skip(out, force, "Stage 3: Event preprocessing"):
        return
    logger.info("Stage 3: Harmonizing GDELT events")
    from geosignal.data.preprocessing import harmonize_gdelt
    events = harmonize_gdelt(DATA_RAW / "gdelt_raw.parquet")
    events.to_parquet(out, index=False)
    logger.success(f"Stage 3 complete → {out}  ({len(events):,} events)")


# ── Stage 4: NLP alignment ────────────────────────────────────────────────────

def stage_nlp_alignment(force: bool):
    pairs_out  = INTERIM / "event_news_pairs.parquet"
    if _skip(pairs_out, force, "Stage 4: NLP alignment"):
        return
    logger.info("Stage 4: Building NewsAligner and aligning events to articles")
    import pandas as pd
    from geosignal.nlp.alignment import NewsAligner

    events   = pd.read_parquet(DATA_PROC / "processed_events.parquet")
    news_df  = pd.read_parquet(DATA_PROC / "news_corpus.parquet")

    aligner  = NewsAligner(news_df)
    pairs    = aligner.align_all(events)
    pairs.to_parquet(pairs_out, index=False)
    logger.success(f"Stage 4 complete → {pairs_out}  ({len(pairs):,} pairs)")


# ── Stage 5: Sector mapping ───────────────────────────────────────────────────

def stage_sector_mapping(force: bool):
    out = INTERIM / "event_news_sector.parquet"
    if _skip(out, force, "Stage 5: Sector mapping"):
        return
    logger.info("Stage 5: Enriching pairs with GICS sectors")
    import pandas as pd
    from geosignal.sectors.sector_mapping import enrich_pairs_with_sectors

    pairs    = pd.read_parquet(INTERIM / "event_news_pairs.parquet")
    enriched = enrich_pairs_with_sectors(pairs)
    enriched.to_parquet(out, index=False)
    logger.success(f"Stage 5 complete → {out}")


# ── Stage 6: Market data ──────────────────────────────────────────────────────

def stage_market_data(start: str, end: str, force: bool):
    out = DATA_PROC / "nse_returns.parquet"
    if _skip(out, force, "Stage 6: NSE market data"):
        return
    logger.info(f"Stage 6: Downloading NSE indices {start} → {end}")
    from geosignal.markets.market_data import download_nse_indices, compute_daily_returns
    prices  = download_nse_indices(start, end)
    returns = compute_daily_returns(prices)
    returns.to_parquet(out)
    logger.success(f"Stage 6 complete → {out}")


# ── Stage 7: Event study + temporal analysis ──────────────────────────────────

def stage_temporal_analysis(force: bool):
    out = DATA_PROC / "temporal_results.parquet"
    if _skip(out, force, "Stage 7: Temporal analysis"):
        return
    logger.info("Stage 7: Running event study (MacKinlay 1997)")
    import pandas as pd
    from geosignal.analysis.event_study import EventStudy, compute_raw_impact

    returns  = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
    returns.index = pd.to_datetime(returns.index)
    pairs    = pd.read_parquet(INTERIM / "event_news_sector.parquet")
    events   = pd.read_parquet(DATA_PROC / "processed_events.parquet")[["event_id", "date"]].rename(
                   columns={"date": "event_date"})
    merged   = pairs.merge(events, on="event_id", how="left").dropna(subset=["event_date"])

    es = EventStudy(returns)
    records = []
    for _, row in merged.iterrows():
        event_date = pd.Timestamp(row["event_date"])
        sector     = row.get("sector", "Unknown")
        ticker     = next(
            (v for k, v in NSE_SECTOR_TICKERS.items() if k.lower() in sector.lower()), None
        )
        if not ticker:
            continue
        raw     = compute_raw_impact(returns, ticker, event_date)
        ev      = es.run(ticker, event_date)
        records.append({
            "event_id"    : row["event_id"],
            "event_date"  : event_date,
            "article_date": row.get("article_date"),
            "sector"      : sector,
            "ticker"      : ticker,
            "CAR_pct"     : ev.get("CAR"),
            "t_stat"      : ev.get("t_stat"),
            "p_value"     : ev.get("p_value"),
            "significant" : ev.get("significant"),
            **raw,
        })

    results = pd.DataFrame(records)
    results.to_parquet(out, index=False)
    logger.success(f"Stage 7 complete → {out}  ({len(results):,} records)")


# ── Stage 8: Impact dataset assembly ─────────────────────────────────────────

def stage_impact_analysis(force: bool):
    datasets_dir = RESULTS_DIR / "datasets"
    out = datasets_dir / "integrated_validation_dataset.parquet"
    if _skip(out, force, "Stage 8: Impact analysis"):
        return
    logger.info("Stage 8: Assembling integrated validation dataset")
    import pandas as pd
    from geosignal.analysis.impact_analysis import build_integrated_dataset

    events  = pd.read_parquet(DATA_PROC / "processed_events.parquet")
    temp    = pd.read_parquet(DATA_PROC / "temporal_results.parquet")
    pairs   = pd.read_parquet(INTERIM  / "event_news_sector.parquet")

    integrated = build_integrated_dataset(events, temp, pairs)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    integrated.to_parquet(out, index=False)
    integrated.to_csv(datasets_dir / "integrated_validation_dataset.csv", index=False)
    logger.success(f"Stage 8 complete → {out}  ({len(integrated)} rows)")

    summary = integrated.groupby("validation").size().reset_index(name="count")
    logger.info(f"\nValidation summary:\n{summary.to_string(index=False)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoSignal full pipeline")
    parser.add_argument("--start",          default="2024-01-01")
    parser.add_argument("--end",            default="2024-03-31")
    parser.add_argument("--force",          action="store_true",
                        help="Rerun all stages even if outputs exist")
    parser.add_argument("--skip-download",  action="store_true",
                        help="Skip raw data download (reuse cached)")
    parser.add_argument("--skip-nlp",       action="store_true",
                        help="Skip NLP alignment (reuse existing pairs)")
    args = parser.parse_args()

    t0 = time.time()
    logger.info(f"GeoSignal pipeline: {args.start} → {args.end}")

    if not args.skip_download:
        stage_download_gdelt(args.start, args.end, args.force)
        stage_download_gkg(args.start, args.end, args.force)

    stage_preprocess(args.force)

    if not args.skip_nlp:
        stage_nlp_alignment(args.force)

    stage_sector_mapping(args.force)
    stage_market_data(args.start, args.end, args.force)
    stage_temporal_analysis(args.force)
    stage_impact_analysis(args.force)

    elapsed = time.time() - t0
    logger.success(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

"""
run_all.py
==========
GeoSignal -- single-file master runner for the end-to-end analysis pipeline.

Pipeline stages:

  Stage 1   Download GDELT events
  Stage 2   Download GKG news corpus
  Stage 3   Preprocess / harmonise events
  Stage 4   NLP alignment (BM25 + TF-IDF)
  Stage 5   Sector mapping (GICS)
  Stage 6   NSE market data (yfinance)
  Stage 7   MacKinlay (1997) event study
  Stage 8   Integrated validation dataset
  Stage 9   Granger analysis (unfinished algorithm area)
  Stage 10  India I-O matrix build  (WIOD 2016 -> synthetic fallback)
  Stage 11  Leontief I-O propagation (unfinished empirical algorithm area)
  Stage 12  NetworkX geopolitical influence graph
  Stage 13  Run summary

Usage
-----
  # Dev sample (Q1 2024), reuse cached raw data, auto I-O matrix:
      python run_all.py --skip-download

  # Full 2022-2024 research window (parallel download, ~30 GB):
      python run_all.py --full --start 2022-01-01 --end 2024-12-31

  # Supply a real India MoSPI / WIOD CSV instead of the auto builder:
      python run_all.py --skip-download --io-matrix config/india_io_wiod_2014.csv

  # Force rerun of everything:
      python run_all.py --force

  # Dry-run the full-window download to check disk requirements:
      python run_all.py --full --dry-run

Outputs (all under results/)
-----------------------------
  datasets/integrated_validation_dataset.parquet / .csv
  datasets/granger_results.csv
  datasets/leontief_shocks.csv
  datasets/network_edges.csv
  datasets/network_nodes.csv
  figures/CAR_by_sector.png
  figures/leontief_heatmap.png
  figures/network_graph.png
  figures/network_graph_sectors.png
  run_summary.txt
  run_summary.json
  logs/run_<timestamp>.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    DATA_RAW, DATA_PROC, RESULTS_DIR,
    NSE_SECTOR_TICKERS, NIFTY50_TICKER,
    GRANGER_MAX_LAGS,
)

INTERIM      = PROJECT_ROOT / "data" / "interim"
DATASETS_DIR = RESULTS_DIR / "datasets"
FIGURES_DIR  = RESULTS_DIR / "figures"
LOGS_DIR     = RESULTS_DIR / "logs"

for _d in [INTERIM, DATASETS_DIR, FIGURES_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _skip(path: Path, force: bool, label: str) -> bool:
    if path.exists() and not force:
        logger.info(f"[SKIP] {label} — output exists: {path.name}")
        return True
    return False


def _read(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.error(f"Missing required file for {label}: {path}")
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1 -- Download GDELT events
# ══════════════════════════════════════════════════════════════════════════════

def stage_download_gdelt(start: str, end: str, force: bool, full: bool, workers: int) -> None:
    out = DATA_RAW / "gdelt_raw.parquet"
    if _skip(out, force, "Stage 1 · GDELT download"):
        return
    if full:
        logger.info(f"Stage 1 · Full-window GDELT download  {start} -> {end}  ({workers} workers)")
        from geosignal.data.batch_downloader import download_full_window
        download_full_window(start, end, sources="gdelt", workers=workers, force=force)
    else:
        logger.info(f"Stage 1 · GDELT download  {start} -> {end}")
        from geosignal.data.loader import download_gdelt_range
        download_gdelt_range(start, end)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 -- Download GKG news corpus
# ══════════════════════════════════════════════════════════════════════════════

def stage_download_gkg(start: str, end: str, force: bool, full: bool, workers: int) -> None:
    out = DATA_PROC / "news_corpus.parquet"
    if _skip(out, force, "Stage 2 · GKG corpus download"):
        return
    if full:
        logger.info(f"Stage 2 · Full-window GKG download  {start} -> {end}  ({workers} workers)")
        from geosignal.data.batch_downloader import download_full_window
        download_full_window(start, end, sources="gkg", workers=workers, force=force)
    else:
        logger.info(f"Stage 2 · GKG download  {start} -> {end}")
        from geosignal.nlp.news_loader import download_gkg_range
        news_df = download_gkg_range(start, end)
        news_df.to_parquet(out, index=False)
        logger.success(f"Stage 2 complete -> {out}  ({len(news_df):,} articles)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3 -- Preprocess events
# ══════════════════════════════════════════════════════════════════════════════

def stage_preprocess(force: bool) -> None:
    out = DATA_PROC / "processed_events.parquet"
    if _skip(out, force, "Stage 3 · Event preprocessing"):
        return
    logger.info("Stage 3 · Harmonising GDELT events")
    from geosignal.data.preprocessing import harmonize_gdelt
    events = harmonize_gdelt(DATA_RAW / "gdelt_raw.parquet")
    events.to_parquet(out, index=False)
    logger.success(f"Stage 3 complete -> {out}  ({len(events):,} events)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4 -- NLP alignment
# ══════════════════════════════════════════════════════════════════════════════

def stage_nlp_alignment(force: bool, full: bool = False) -> None:
    out = INTERIM / "event_news_pairs.parquet"
    if _skip(out, force, "Stage 4 · NLP alignment"):
        return
    logger.info("Stage 4 · BM25 + TF-IDF event-article alignment")
    from geosignal.nlp.alignment import NewsAligner

    events  = pd.read_parquet(DATA_PROC / "processed_events.parquet")
    news_df = pd.read_parquet(DATA_PROC / "news_corpus.parquet")

    if not full:
        # Deterministic development sample.
        # Keep the sample small enough to validate the complete pipeline quickly.
        sample_size = min(30, len(events))
        events = (
            events
            .sort_values("date")
            .head(sample_size)
            .reset_index(drop=True)
        )
        logger.info(
            f"Dev mode: limiting Stage 4 NLP alignment to "
            f"{len(events):,} events."
        )
    else:
        logger.info(
            f"Full mode: processing all {len(events):,} events."
        )

    aligner = NewsAligner(news_df)
    pairs   = aligner.align_all(events)
    pairs.to_parquet(out, index=False)
    logger.success(f"Stage 4 complete -> {out}  ({len(pairs):,} pairs)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 5 -- Sector mapping
# ══════════════════════════════════════════════════════════════════════════════

def stage_sector_mapping(force: bool) -> None:
    out = INTERIM / "event_news_sector.parquet"
    if _skip(out, force, "Stage 5 · Sector mapping"):
        return
    logger.info("Stage 5 · Enriching pairs with GICS sectors")
    from geosignal.sectors.sector_mapping import enrich_pairs_with_sectors

    pairs    = pd.read_parquet(INTERIM / "event_news_pairs.parquet")
    enriched = enrich_pairs_with_sectors(pairs)
    enriched.to_parquet(out, index=False)
    logger.success(f"Stage 5 complete -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 6 -- NSE market data
# ══════════════════════════════════════════════════════════════════════════════

def stage_market_data(start: str, end: str, force: bool) -> None:
    out = DATA_PROC / "nse_returns.parquet"
    if _skip(out, force, "Stage 6 · NSE market data"):
        return
    logger.info(f"Stage 6 · Downloading NSE indices  {start} -> {end}")
    from geosignal.markets.market_data import download_nse_indices, compute_daily_returns
    prices  = download_nse_indices(start, end)
    returns = compute_daily_returns(prices)
    returns.to_parquet(out)
    logger.success(f"Stage 6 complete -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 7 -- MacKinlay (1997) event study
# ══════════════════════════════════════════════════════════════════════════════

def stage_event_study(force: bool) -> None:
    out = DATA_PROC / "temporal_results.parquet"
    if _skip(out, force, "Stage 7 · Event study"):
        return
    logger.info("Stage 7 · MacKinlay (1997) event study -- AR / CAR / t-test")
    from geosignal.analysis.event_study import EventStudy, compute_raw_impact

    returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
    returns.index = pd.to_datetime(returns.index)

    pairs  = pd.read_parquet(INTERIM / "event_news_sector.parquet")
    events = pd.read_parquet(DATA_PROC / "processed_events.parquet")[
                 ["event_id", "date"]].rename(columns={"date": "event_date"})
    merged = pairs.merge(events, on="event_id", how="left").dropna(subset=["event_date"])

    es      = EventStudy(returns)
    records = []

    for _, row in merged.iterrows():
        event_date = pd.Timestamp(row["event_date"])
        sector     = row.get("sector", "Unknown")
        ticker     = next(
            (v for k, v in NSE_SECTOR_TICKERS.items()
             if k.lower() in sector.lower()), None,
        )
        if not ticker:
            continue
        raw = compute_raw_impact(returns, ticker, event_date)
        ev  = es.run(ticker, event_date)
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

    pd.DataFrame(records).to_parquet(out, index=False)
    logger.success(f"Stage 7 complete -> {out}  ({len(records):,} records)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 8 -- Integrated validation dataset
# ══════════════════════════════════════════════════════════════════════════════

def stage_impact_analysis(force: bool) -> None:
    out = DATASETS_DIR / "integrated_validation_dataset.parquet"
    if _skip(out, force, "Stage 8 · Impact assembly"):
        return
    logger.info("Stage 8 · Assembling integrated validation dataset")
    from geosignal.analysis.impact_analysis import build_integrated_dataset

    events   = pd.read_parquet(DATA_PROC / "processed_events.parquet")
    temporal = pd.read_parquet(DATA_PROC / "temporal_results.parquet")
    pairs    = pd.read_parquet(INTERIM  / "event_news_sector.parquet")

    integrated = build_integrated_dataset(events, temporal, pairs)
    integrated.to_parquet(out, index=False)
    integrated.to_csv(DATASETS_DIR / "integrated_validation_dataset.csv", index=False)
    logger.success(f"Stage 8 complete -> {out}  ({len(integrated)} rows)")
    summary = integrated.groupby("validation").size().reset_index(name="count")
    logger.info(f"\nValidation distribution:\n{summary.to_string(index=False)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 9 -- Toda-Yamamoto Granger causality
# ══════════════════════════════════════════════════════════════════════════════

def stage_granger(force: bool) -> None:
    """
    Test whether daily aggregate geopolitical risk (mean |Goldstein|)
    Granger-predicts each NSE sectoral return.

    CONSTRAINT (L-007 / DEC-007): all results use 'Granger-predicts'
    language -- never 'causes'.
    """
    out = DATASETS_DIR / "granger_results.csv"
    if _skip(out, force, "Stage 9 · Granger causality"):
        return
    logger.info("Stage 9 · Toda-Yamamoto Granger causality (per sector)")
    from geosignal.analysis.granger_causality import granger_causality_test

    events  = _read(DATA_PROC / "processed_events.parquet", "Granger-events")
    returns = _read(DATA_PROC / "nse_returns.parquet",       "Granger-returns")
    if events is None or returns is None:
        logger.warning("Stage 9 skipped -- prerequisite files missing.")
        return

    returns.index = pd.to_datetime(returns.index)
    events["date"] = pd.to_datetime(events["date"])

    # Daily geo-risk proxy: mean absolute Goldstein score
    geo_risk = (
        events.groupby("date")["goldstein"]
              .apply(lambda x: x.abs().mean())
              .rename("geo_risk")
    )

    records: list[dict] = []
    for sector_name, ticker in NSE_SECTOR_TICKERS.items():
        if ticker not in returns.columns:
            logger.warning(f"  {ticker} not in returns -- skipping.")
            continue

        aligned = pd.DataFrame({
            "geo_risk"     : geo_risk,
            "sector_return": returns[ticker].dropna(),
        }).dropna()

        if len(aligned) < GRANGER_MAX_LAGS * 4:
            logger.warning(
                f"  {sector_name}: {len(aligned)} obs -- insufficient "
                f"for Granger at lag={GRANGER_MAX_LAGS} (L-009). "
                "Expand to 2022-2024 with --full."
            )
            records.append({
                "sector": sector_name, "ticker": ticker,
                "lag": None, "F_stat": None, "p_value": None,
                "significant": False,
                "note": f"insufficient_obs ({len(aligned)})",
            })
            continue

        gc_df = granger_causality_test(
            geo_risk      = aligned["geo_risk"],
            sector_return = aligned["sector_return"],
            max_lag       = GRANGER_MAX_LAGS,
            toda_yamamoto = True,
        )

        if gc_df.empty:
            records.append({"sector": sector_name, "ticker": ticker,
                            "lag": None, "F_stat": None, "p_value": None,
                            "significant": False, "note": "test_failed"})
        else:
            best = gc_df.sort_values("p_value").iloc[0]
            records.append({
                "sector"     : sector_name,
                "ticker"     : ticker,
                "lag"        : int(best["lag"]),
                "F_stat"     : round(float(best["F_stat"]), 4) if best["F_stat"] else None,
                "p_value"    : round(float(best["p_value"]), 4) if best["p_value"] else None,
                "significant": bool(best["significant"]),
                "note"       : "geo_risk Granger-predicts sector_return",
            })
            logger.info(f"  {sector_name}: {gc_df['significant'].sum()} significant lag(s)")

    granger_df = pd.DataFrame(records)
    granger_df.to_csv(out, index=False)
    n_sig = int(granger_df["significant"].sum()) if "significant" in granger_df.columns else 0
    logger.success(
        f"Stage 9 complete -> {out}  |  "
        f"{n_sig}/{len(NSE_SECTOR_TICKERS)} sectors Granger-predictive"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 10 -- India I-O matrix (WIOD / synthetic)
# ══════════════════════════════════════════════════════════════════════════════

def stage_build_io_matrix(
    io_matrix_path: Path | None,
    force: bool,
) -> Path:
    """
    Build (or locate) the India I-O coefficient matrix CSV.

    Priority:
      1. User supplied --io-matrix <path>         -> use as-is
      2. Cached config/india_io_wiod_2014.csv     -> skip rebuild
      3. io_loader.build_india_io_matrix(auto)    -> WIOD download or synthetic

    Returns the Path to the final CSV (used by Stage 11).
    """
    # If user gave an explicit path, validate and use it directly
    if io_matrix_path and io_matrix_path.exists():
        logger.info(f"Stage 10 · Using user-supplied I-O matrix: {io_matrix_path}")
        return io_matrix_path

    if io_matrix_path and not io_matrix_path.exists():
        logger.warning(
            f"--io-matrix path not found: {io_matrix_path}. "
            "Will auto-build."
        )

    # Check for already-built files in config/
    config_dir  = PROJECT_ROOT / "config"
    wiod_cache  = config_dir / "india_io_wiod_2014.csv"
    synth_cache = config_dir / "india_io_synthetic_2023.csv"

    if wiod_cache.exists() and not force:
        logger.info(f"Stage 10 · Using cached WIOD matrix: {wiod_cache}")
        return wiod_cache
    if synth_cache.exists() and not force:
        logger.info(f"Stage 10 · Using cached synthetic matrix: {synth_cache}")
        return synth_cache

    logger.info("Stage 10 · Building India I-O matrix (WIOD -> synthetic fallback)")
    from geosignal.propagation.io_loader import build_india_io_matrix
    return build_india_io_matrix(source="auto", force=force)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 11 -- Leontief shock propagation
# ══════════════════════════════════════════════════════════════════════════════

def stage_leontief(io_csv_path: Path, force: bool) -> None:
    """
    Propagate sector shocks via Leontief inverse.
    Shocks = mean significant CAR_pct per sector (from Stage 7).
    Skips the stage when the matrix or empirical sector shocks are unavailable.
    """
    out = DATASETS_DIR / "leontief_shocks.csv"
    if _skip(out, force, "Stage 11 · Leontief propagation"):
        return

    from geosignal.propagation.leontief import LeontiefPropagator

    propagator   = LeontiefPropagator.from_csv(io_csv_path)
    matrix_label = io_csv_path.name
    is_toy       = "toy" in matrix_label or "synthetic" in matrix_label

    if is_toy:
        logger.warning(
            f"Stage 11 skipped -- {matrix_label} is synthetic/illustrative, "
            "so it cannot produce an empirical propagation result."
        )
        return
    logger.info(f"Stage 11 · Leontief -- empirical I-O matrix: {matrix_label}")

    # Derive shocks from significant event-study CARs
    temporal_path = DATA_PROC / "temporal_results.parquet"
    sector_shocks: dict[str, float] = {}
    if temporal_path.exists():
        temp = pd.read_parquet(temporal_path)
        sig  = temp[temp["significant"] == True] if "significant" in temp.columns else temp
        if "sector" in sig.columns and "CAR_pct" in sig.columns:
            sector_shocks = (
                sig.groupby("sector")["CAR_pct"].mean()
                   .dropna().div(100).to_dict()
            )

    if not sector_shocks:
        logger.warning(
            "Stage 11 skipped -- no significant, empirical sector CAR shocks are available."
        )
        return

    records: list[dict] = []
    for shocked_sector, shock_mag in sector_shocks.items():
        matched = next(
            (s for s in propagator.sector_labels
             if shocked_sector.lower() in s.lower()
             or s.lower() in shocked_sector.lower()),
            None,
        )
        if not matched:
            continue
        delta_x = propagator.propagate_shock(matched, shock_mag)
        for recv, impact in delta_x.items():
            records.append({
                "io_matrix"        : matrix_label,
                "shocked_sector"   : shocked_sector,
                "io_shocked_label" : matched,
                "shock_magnitude"  : round(shock_mag, 6),
                "receiving_sector" : recv,
                "delta_output"     : round(float(impact), 6),
                "note"             : "empirical I-O propagation from significant sector CARs",
            })

    if not records:
        logger.warning("Stage 11: no Leontief records -- check sector name matching.")
        return

    pd.DataFrame(records).to_csv(out, index=False)
    logger.success(f"Stage 11 complete -> {out}  ({len(records)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 12 -- NetworkX geopolitical influence graph
# ══════════════════════════════════════════════════════════════════════════════

def stage_network_graph(io_csv_path: Path | None, force: bool) -> None:
    """
    Build the full geopolitical influence graph using GeoSignalGraph:
      - Actor -> Actor edges (GDELT co-occurrence, Goldstein-weighted)
      - Actor -> Sector edges (event study CAR-weighted)
      - Sector -> Sector edges from I-O matrix (optional)
    Compute PageRank, betweenness, eigenvector centrality.
    Export edge list, node list, and PNG figures.
    """
    edges_out = DATASETS_DIR / "network_edges.csv"
    if _skip(edges_out, force, "Stage 12 · NetworkX graph"):
        return
    logger.info("Stage 12 · Building GeoSignal NetworkX graph")

    from geosignal.graph.network import GeoSignalGraph

    events   = _read(DATA_PROC / "processed_events.parquet",  "graph-events")
    temporal = _read(DATA_PROC / "temporal_results.parquet",   "graph-temporal")
    pairs    = _read(INTERIM   / "event_news_sector.parquet",  "graph-pairs")

    if events is None:
        logger.warning("Stage 12 skipped -- events file missing.")
        return

    g = GeoSignalGraph()
    g.build(events_df=events, temporal_df=temporal, pairs_df=pairs)

    # Add I-O layer if matrix is available
    if io_csv_path and io_csv_path.exists():
        g.add_io_edges(io_csv_path, threshold=0.03)

    g.compute_centrality()
    g.export(DATASETS_DIR)
    g.visualise(FIGURES_DIR)

    stats = g.summary()
    logger.success(
        f"Stage 12 complete | "
        f"{stats.get('n_nodes', '?')} nodes, "
        f"{stats.get('n_edges', '?')} edges | "
        f"density={stats.get('density', '?')}"
    )

    # Persist pipeline summary
    summary_path = DATASETS_DIR / "network_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure helpers (non-blocking, failures are warnings not errors)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_car_by_sector() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = DATA_PROC / "temporal_results.parquet"
        if not path.exists():
            return
        temp = pd.read_parquet(path)
        if "sector" not in temp.columns or "CAR_pct" not in temp.columns:
            return

        base = temp[temp["significant"] == True] if "significant" in temp.columns else temp
        if base.empty:
            return

        by_sector = base.groupby("sector")["CAR_pct"].mean().sort_values()
        fig, ax   = plt.subplots(figsize=(10, 6))
        colors    = ["#e74c3c" if v < 0 else "#2ecc71" for v in by_sector.values]
        by_sector.plot(kind="barh", color=colors, edgecolor="white", ax=ax)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean CAR (%) -- significant events only (p < 0.05)")
        ax.set_title(
            "Cumulative Abnormal Returns by Sector\n"
            "(MacKinlay 1997 event study)"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out = FIGURES_DIR / "CAR_by_sector.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Figure -> {out}")
    except Exception as exc:
        logger.warning(f"CAR figure failed: {exc}")


def _plot_leontief_heatmap() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        path = DATASETS_DIR / "leontief_shocks.csv"
        if not path.exists():
            return
        leo = pd.read_csv(path)
        pivot = leo.pivot_table(
            index="receiving_sector",
            columns="io_shocked_label",
            values="delta_output",
            aggfunc="sum",
        )
        is_illus = "synthetic" in str(leo["io_matrix"].iloc[0]) \
                   if "io_matrix" in leo.columns else True

        fig, ax = plt.subplots(
            figsize=(max(8, pivot.shape[1] * 1.5),
                     max(6, pivot.shape[0] * 0.9))
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, linewidths=0.5, ax=ax)
        title = "Leontief I-O Shock Propagation -- Delta Output"
        if is_illus:
            title += "\n[synthetic/illustrative matrix -- replace with MoSPI WIOD for empirical results]"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Shocked sector")
        ax.set_ylabel("Receiving sector")
        plt.tight_layout()
        out = FIGURES_DIR / "leontief_heatmap.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Figure -> {out}")
    except Exception as exc:
        logger.warning(f"Leontief heatmap failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 13 -- Run summary
# ══════════════════════════════════════════════════════════════════════════════

def stage_run_summary(args: argparse.Namespace, io_csv_path: Path | None, elapsed: float) -> None:
    logger.info("Stage 13 · Run summary")

    summary: dict = {
        "run_timestamp"  : datetime.utcnow().isoformat(),
        "pipeline_args"  : vars(args),
        "elapsed_seconds": round(elapsed, 1),
        "sections"       : {},
    }
    lines: list[str] = [
        "=" * 72,
        "  GeoSignal -- Pipeline Run Summary",
        f"  Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Window    : {args.start} -> {args.end}",
        f"  Mode      : {'full 2022-2024' if args.full else 'dev sample'}",
        "=" * 72,
        "",
    ]

    # ── Event study ───────────────────────────────────────────────────────────
    tp = DATA_PROC / "temporal_results.parquet"
    if tp.exists():
        temp    = pd.read_parquet(tp)
        n_total = len(temp)
        n_sig   = int(temp["significant"].sum()) if "significant" in temp.columns else 0
        mean_car = float(temp["CAR_pct"].mean()) if "CAR_pct" in temp.columns else float("nan")
        lines  += [
            "-- Event Study (MacKinlay 1997) ------------------------------------------",
            f"  Total event-sector records  : {n_total:,}",
            f"  Significant (p < 0.05)      : {n_sig:,}"
            + (f"  ({100*n_sig/n_total:.1f}%)" if n_total else ""),
            f"  Mean CAR across all events  : {mean_car:.4f}%",
            "",
        ]
        if "sector" in temp.columns:
            by_sec = (
                temp.groupby("sector")
                    .agg(n=("event_id","count"),
                         mean_CAR=("CAR_pct","mean"),
                         n_sig=("significant","sum"))
                    .sort_values("n_sig", ascending=False)
            )
            for _, r in by_sec.iterrows():
                lines.append(
                    f"    {r.name:<32}  n={int(r['n']):>5}  "
                    f"mean_CAR={r['mean_CAR']:+.3f}%  sig={int(r['n_sig'])}"
                )
            lines.append("")
        summary["sections"]["event_study"] = {
            "n_records": n_total, "n_significant": n_sig,
            "mean_CAR_pct": round(mean_car, 4) if not np.isnan(mean_car) else None,
        }

    # ── Granger ───────────────────────────────────────────────────────────────
    gp = DATASETS_DIR / "granger_results.csv"
    if gp.exists():
        gc    = pd.read_csv(gp)
        n_sig = int(gc["significant"].sum()) if "significant" in gc.columns else 0
        lines += [
            "-- Granger Causality (Toda-Yamamoto) ------------------------------------",
            "  [L-007] Granger-predictive precedence only -- NOT causal mechanism.",
            f"  Sectors where geo_risk Granger-predicts returns (p<0.05): "
            f"{n_sig}/{len(gc)}",
        ]
        for _, r in gc[gc.get("significant", pd.Series(dtype=bool)) == True].iterrows():
            lines.append(
                f"    {r.get('sector','?'):<32}  lag={r.get('lag','?')}  "
                f"F={r.get('F_stat','?')}  p={r.get('p_value','?')}"
            )
        lines.append("")
        summary["sections"]["granger"] = {
            "n_significant": n_sig, "n_tested": len(gc),
        }

    # ── I-O matrix ────────────────────────────────────────────────────────────
    if io_csv_path and io_csv_path.exists():
        is_real = "wiod" in io_csv_path.name.lower() or "mospi" in io_csv_path.name.lower()
        lines += [
            "-- I-O Matrix -----------------------------------------------------------",
            f"  Source  : {io_csv_path.name}",
            f"  Type    : {'REAL (WIOD / MoSPI)' if is_real else 'synthetic (GDP-share estimate)'}",
            "",
        ]
        summary["sections"]["io_matrix"] = {
            "file": io_csv_path.name, "real": is_real,
        }

    # ── Leontief ─────────────────────────────────────────────────────────────
    lp = DATASETS_DIR / "leontief_shocks.csv"
    if lp.exists():
        leo    = pd.read_csv(lp)
        is_syn = "synthetic" in str(leo["io_matrix"].iloc[0]) if "io_matrix" in leo.columns else True
        top5   = (
            leo.groupby("receiving_sector")["delta_output"]
               .sum().abs().sort_values(ascending=False).head(5)
        )
        lines += [
            "-- Leontief I-O Propagation ----------------------------------------------",
            f"  Matrix  : {leo['io_matrix'].iloc[0] if 'io_matrix' in leo.columns else '?'}",
        ]
        if is_syn:
            lines.append(
                "  NOTE    : synthetic matrix -- replace with real MoSPI/WIOD for "
                "empirical results (L-006)"
            )
        lines += ["  Top 5 receiving sectors by |delta output|:"]
        for sec, val in top5.items():
            lines.append(f"    {sec:<32}  |d_output| = {val:.4f}")
        lines.append("")
        summary["sections"]["leontief"] = {
            "n_rows": len(leo), "synthetic": is_syn,
        }

    # ── Network graph ─────────────────────────────────────────────────────────
    ns = DATASETS_DIR / "network_summary.json"
    if ns.exists():
        net = json.loads(ns.read_text(encoding="utf-8"))
        lines += [
            "-- NetworkX Graph -------------------------------------------------------",
            f"  Nodes   : {net.get('n_nodes','?')}  "
            f"(actors={net.get('n_actor_nodes','?')}, "
            f"sectors={net.get('n_sector_nodes','?')})",
            f"  Edges   : {net.get('n_edges','?')}  |  density={net.get('density','?')}",
            f"  Top actors (PageRank): "
            f"{', '.join(str(a) for a in net.get('top_actors_by_pagerank', [])[:5])}",
            f"  Figures : {FIGURES_DIR / 'network_graph.png'}",
            "  (DEC-008: plain NetworkX -- GNN layer planned as October extension)",
            "",
        ]
        summary["sections"]["network"] = net

    # ── Validation dataset ────────────────────────────────────────────────────
    vp = DATASETS_DIR / "integrated_validation_dataset.parquet"
    if vp.exists():
        val     = pd.read_parquet(vp)
        v_cnts  = val["validation"].value_counts().to_dict() if "validation" in val.columns else {}
        lines  += [
            "-- Integrated Validation Dataset ----------------------------------------",
            f"  Rows    : {len(val):,}",
        ]
        for lbl, cnt in v_cnts.items():
            lines.append(f"  {lbl:<22}: {cnt:>6}")
        lines.append("")
        summary["sections"]["validation_dataset"] = {
            "n_rows": len(val), "labels": v_cnts,
        }

    # ── Limitations ───────────────────────────────────────────────────────────
    lines += [
        "-- Known Limitations ------------------------------------------------------",
        "  L-001  GKG proxy text != full article body.",
        "  L-002  GDELT dates = media detection, not event occurrence.",
        "  L-003  GDELT false-positive rate ~15-20%.",
        "  L-004  Yahoo Finance NSE data may have gaps.",
        "  L-005  Goldstein fallback = root-code approximation.",
        "  L-006  Leontief uses synthetic matrix unless WIOD CSV is supplied.",
        "  L-007  Granger = predictive precedence, NOT causation.",
        "  L-008  Sector mapping is keyword-based, not ontological.",
        "  L-009  90-day dev sample underpowered for Granger at lag=12.",
        f"  Full details: {PROJECT_ROOT / 'docs' / 'LIMITATIONS.md'}",
        "",
        "=" * 72,
        f"  Pipeline elapsed: {elapsed:.1f}s",
        "=" * 72,
    ]

    (RESULTS_DIR / "run_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (RESULTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.success(f"Stage 13 complete -> {RESULTS_DIR / 'run_summary.txt'}")
    print("\n" + "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
#  Logging setup
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logging() -> None:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"run_{ts}.log"
    logger.add(
        log_file,
        rotation="50 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_file}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GeoSignal -- master pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python run_all.py --skip-download
  python run_all.py --full --start 2022-01-01 --end 2024-12-31
  python run_all.py --full --dry-run
  python run_all.py --skip-download --io-matrix config/india_io_wiod_2014.csv
  python run_all.py --force
        """,
    )
    p.add_argument("--start",         default="2024-01-01",
                   help="Pipeline start date (YYYY-MM-DD)")
    p.add_argument("--end",           default="2024-03-31",
                   help="Pipeline end date (YYYY-MM-DD)")
    p.add_argument("--force",         action="store_true",
                   help="Rerun all stages even if outputs exist")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip GDELT + GKG download (reuse cached raw data)")
    p.add_argument("--skip-nlp",      action="store_true",
                   help="Skip NLP alignment (reuse existing event_news_pairs.parquet)")
    p.add_argument("--full",          action="store_true",
                   help="Use batch_downloader for parallel chunked download "
                        "(recommended for --start 2022-01-01 --end 2024-12-31)")
    p.add_argument("--workers",       type=int, default=4,
                   help="Parallel download workers (--full mode only, default 4)")
    p.add_argument("--dry-run",       action="store_true",
                   help="Show download plan without downloading (--full only)")
    p.add_argument("--io-matrix",     default=None, type=Path,
                   help="Path to real India MoSPI/WIOD I-O CSV. "
                        "If omitted, auto-builds via io_loader (WIOD -> synthetic).")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _build_parser().parse_args()
    _setup_logging()

    # Dry-run exits early after printing the download plan
    if args.dry_run:
        if not args.full:
            logger.warning("--dry-run only has effect with --full.")
            return
        from geosignal.data.batch_downloader import download_full_window
        download_full_window(
            args.start, args.end,
            sources="both",
            workers=args.workers,
            dry_run=True,
        )
        return

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("GeoSignal -- master pipeline")
    logger.info(f"  Window  : {args.start} -> {args.end}")
    logger.info(f"  Mode    : {'full (batch)' if args.full else 'dev sample (sequential)'}")
    logger.info(f"  Force   : {args.force}")
    logger.info(f"  Workers : {args.workers}")
    logger.info("=" * 60)

    # ── Core pipeline ─────────────────────────────────────────────────────────
    if not args.skip_download:
        stage_download_gdelt(args.start, args.end, args.force, args.full, args.workers)
        stage_download_gkg(args.start, args.end, args.force, args.full, args.workers)

    stage_preprocess(args.force)

    if not args.skip_nlp:
        stage_nlp_alignment(args.force, full=args.full)

    stage_sector_mapping(args.force)
    stage_market_data(args.start, args.end, args.force)
    stage_event_study(args.force)
    stage_impact_analysis(args.force)

    # ── Extended analysis ─────────────────────────────────────────────────────
    stage_granger(args.force)

    io_csv = stage_build_io_matrix(args.io_matrix, args.force)   # Stage 10

    stage_leontief(io_csv, args.force)                            # Stage 11
    stage_network_graph(io_csv, args.force)                       # Stage 12

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_car_by_sector()
    _plot_leontief_heatmap()

    # ── Run summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    stage_run_summary(args, io_csv, elapsed)

    logger.success(f"All done in {elapsed:.1f}s")
    logger.info(f"  Results   : {RESULTS_DIR}")
    logger.info(f"  Dashboard : streamlit run dashboard.py")


if __name__ == "__main__":
    main()

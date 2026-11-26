"""
experiments/case_studies/run_pipeline.py
=========================================
Standalone driver: computes MacKinlay (1997) event-study CARs for all 6
GeoSignal cases, draws propagation maps, and writes cross_case_summary.csv.

Does NOT import the legacy case modules (redsea.py, semiconductor.py)
which call plt.show() and hang when run non-interactively.

Run from project root:
    python experiments/case_studies/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import DATA_PROC, RESULTS_DIR
from experiments.case_studies.case_registry import ALL_CASES
from experiments.case_studies.run_all_cases import (
    build_cross_case_summary,
    print_console_summary,
    _isnan,
)
from geosignal.analysis.event_study import EventStudy, compute_raw_impact
from geosignal.visualization.propagation_map import draw_propagation_map

DATASETS_DIR = RESULTS_DIR / "datasets"
FIGURES_DIR  = RESULTS_DIR / "figures"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load shared data ──────────────────────────────────────────────────────────
logger.info("Loading returns data ...")

# Prefer the full-range CSV (Jun 2023–Apr 2024) over the parquet which only
# goes to 2024-01-12 and therefore lacks post-event data for CS-02 through CS-06.
_csv_path = RESULTS_DIR / "datasets" / "historical_nse_sector_indices_20230601_20240430.csv"
_parquet_path = DATA_PROC / "nse_returns.parquet"

if _csv_path.exists():
    # CSV contains price levels; convert to log returns
    _prices = pd.read_csv(_csv_path, index_col=0, parse_dates=True)
    _prices.index = pd.to_datetime(_prices.index)
    _prices.sort_index(inplace=True)
    returns = np.log(_prices / _prices.shift(1)).dropna(how="all")
    logger.info(f"  Built returns from CSV: {returns.shape}, range {returns.index.min().date()} -> {returns.index.max().date()}")
else:
    returns = pd.read_parquet(_parquet_path)
    returns.index = pd.to_datetime(returns.index)
    logger.info(f"  nse_returns parquet: {returns.shape}, range {returns.index.min().date()} -> {returns.index.max().date()}")

logger.info(f"  Cols: {list(returns.columns[:8])}")

events_path = DATA_PROC / "processed_events.parquet"
events_df = pd.read_parquet(events_path) if events_path.exists() else None
if events_df is not None:
    logger.info(f"  processed_events: {events_df.shape}")
else:
    logger.warning("  processed_events.parquet not found — propagation maps will use defaults")

# ── Run event studies ─────────────────────────────────────────────────────────
results: list[dict] = []
es = EventStudy(returns)

for case in ALL_CASES:
    logger.info(f"Running event study: {case.case_id} — {case.label}")

    ev  = es.run(case.nse_ticker, case.anchor)
    raw = compute_raw_impact(returns, case.nse_ticker, case.anchor)

    car_pct   = ev.get("CAR")       if ev else None  # key is "CAR" in EventStudy
    t_stat    = ev.get("t_stat")    if ev else None
    p_value   = ev.get("p_value")   if ev else None
    sig       = ev.get("significant") if ev else None

    if p_value is not None and not _isnan(p_value):
        validation = "Confirmed" if float(p_value) < 0.05 else "Null"
    else:
        validation = "Insufficient"

    row = {
        "case_id"        : case.case_id,
        "label"          : case.label,
        "stratum"        : case.stratum,
        "anchor_date"    : case.anchor_date,
        "nse_ticker"     : case.nse_ticker,
        "shocked_sector" : case.shocked_sector,
        "shock_magnitude": case.shock_magnitude,
        "CAR_pct"        : car_pct,
        "t_stat"         : t_stat,
        "p_value"        : p_value,
        "significant"    : sig,
        "validation"     : validation,
        **raw,
        "status"  : "OK" if ev else "INSUFFICIENT",
        "elapsed_s": 0.0,
    }
    results.append(row)

    if ev:
        logger.success(
            f"  {case.case_id}: CAR={car_pct:+.4f}%  "
            f"p={p_value:.4f}  [{validation}]"
        )
    else:
        logger.warning(f"  {case.case_id}: insufficient data for event study")

    # Write per-case CSV (filename matches _load_case_result expectation)
    csv_name = f"{case.case_id.lower().replace('-', '')}_event_study.csv"
    pd.DataFrame([row]).to_csv(DATASETS_DIR / csv_name, index=False)
    logger.info(f"  Wrote {csv_name}")

# ── Propagation maps ──────────────────────────────────────────────────────────
logger.info("Drawing propagation maps ...")
for case in ALL_CASES:
    try:
        paths = draw_propagation_map(case, temporal_df=None, events_df=events_df)
        logger.success(f"  {case.case_id} map -> {paths['low']}")
    except Exception as exc:
        logger.error(f"  {case.case_id} map failed: {exc}")

# ── Cross-case summary ────────────────────────────────────────────────────────
summary_df = build_cross_case_summary(results)

csv_out = DATASETS_DIR / "cross_case_summary.csv"

summary_df.to_csv(csv_out, index=False)
logger.success(f"cross_case_summary.csv -> {csv_out}")

print_console_summary(summary_df)
logger.success("run_pipeline.py complete.")

"""
experiments/case_studies/run_pipeline.py
=========================================
Standalone driver: computes event-study CARs for all 6 GeoSignal cases,
draws propagation maps, and writes cross_case_summary.csv.

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
from geosignal.analysis.event_study import EventStudy, compute_raw_impact
from geosignal.visualization.propagation_map import draw_propagation_map

DATASETS_DIR = RESULTS_DIR / "datasets"
FIGURES_DIR  = RESULTS_DIR / "figures"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load shared data ──────────────────────────────────────────────────────────
logger.info("Loading parquet files ...")
returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
returns.index = pd.to_datetime(returns.index)
logger.info(f"  nse_returns: {returns.shape}, tickers={list(returns.columns[:8])}...")

events_path = DATA_PROC / "processed_events.parquet"
events_df = pd.read_parquet(events_path) if events_path.exists() else None
logger.info(f"  processed_events: {events_df.shape if events_df is not None else 'N/A'}")

# ── Run event studies ─────────────────────────────────────────────────────────
results: list[dict] = []
es = EventStudy(returns)

for case in ALL_CASES:
    logger.info(f"Running event study: {case.case_id} -- {case.label}")

    ev  = es.run(case.nse_ticker, case.anchor)
    raw = compute_raw_impact(returns, case.nse_ticker, case.anchor)

    car_pct   = ev.get("CAR")          if ev else None   # EventStudy returns key "CAR"
    t_stat    = ev.get("t_stat")       if ev else None
    p_value   = ev.get("p_value")      if ev else None
    sig       = ev.get("significant")  if ev else None

    # Verdict for cross-case table
    if p_value is not None and not (isinstance(p_value, float) and np.isnan(p_value)):
        if float(p_value) < 0.05:
            validation = "Confirmed +" if (car_pct is not None and float(car_pct) > 0) else "Confirmed -"
        else:
            validation = "Null"
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
        "status"         : "OK" if ev else "INSUFFICIENT",
        "elapsed_s"      : 0.0,
    }
    # Merge impact windows
    row.update(raw)
    results.append(row)

    if ev:
        logger.success(f"  {case.case_id}: CAR={car_pct:+.3f}%  p={p_value:.4f}  [{validation}]")
    else:
        logger.warning(f"  {case.case_id}: insufficient data")

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
        import traceback
        logger.debug(traceback.format_exc())

# ── Cross-case summary ────────────────────────────────────────────────────────
from experiments.case_studies.run_all_cases import (
    build_cross_case_summary, write_latex_table, print_console_summary
)

summary_df = build_cross_case_summary(results)
csv_out = DATASETS_DIR / "cross_case_summary.csv"
tex_out = DATASETS_DIR / "cross_case_summary.tex"
summary_df.to_csv(csv_out, index=False)
write_latex_table(summary_df, tex_out)
print_console_summary(summary_df)
logger.success(f"cross_case_summary.csv -> {csv_out}")

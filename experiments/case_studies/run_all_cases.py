"""
experiments/case_studies/run_all_cases.py
==========================================
Runner for all 6 GeoSignal case studies.

What this script does
---------------------
1. Runs each case study module in sequence (timeline + returns +
   event study + Leontief figures, per-case CSV).
2. Draws the propagation map for every case via propagation_map.py.
3. Assembles a cross-case summary table:
     results/datasets/cross_case_summary.csv      (machine-readable)
     results/datasets/cross_case_summary.tex      (LaTeX table, paper-ready)
4. Prints a formatted console summary.

Usage
-----
    # Run all 6 cases:
    python experiments/case_studies/run_all_cases.py

    # Run one case only (still writes cross-case table for completed cases):
    python experiments/case_studies/run_all_cases.py --case CS-03

    # Skip cases whose CSV already exists (default: reruns everything):
    python experiments/case_studies/run_all_cases.py --skip-existing

    # Draw propagation maps only (no case script rerun):
    python experiments/case_studies/run_all_cases.py --maps-only

Settings are frozen before any run (per case-selection protocol §9).
Do not change filter parameters after seeing results.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from config.settings import DATA_PROC, RESULTS_DIR
from experiments.case_studies.case_registry import ALL_CASES, get_case, CaseStudy
from geosignal.visualization.propagation_map import draw_propagation_map

DATASETS_DIR = RESULTS_DIR / "datasets"
FIGURES_DIR  = RESULTS_DIR / "figures"
LOGS_DIR     = RESULTS_DIR / "logs"

for _d in [DATASETS_DIR, FIGURES_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Map case_id → importable module path (relative to project root)
CASE_MODULES: dict[str, str] = {
    "CS-01": "experiments.case_studies.redsea",
    "CS-02": "experiments.case_studies.semiconductor",
    "CS-03": "experiments.case_studies.iran_sanctions",
    "CS-04": "experiments.case_studies.china_rare_earth",
    "CS-05": "experiments.case_studies.russia_energy",
    "CS-06": "experiments.case_studies.india_pharma_api",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Per-case runner
# ══════════════════════════════════════════════════════════════════════════════

def run_case(case: CaseStudy, skip_existing: bool = False) -> dict:
    """
    Run one case study script and return a result record.

    Returns a dict with keys used by the cross-case summary table.
    On failure, returns a dict with status='ERROR' and the exception message.
    """
    csv_path = (
        case.datasets_dir
        / f"{case.case_id.lower().replace('-', '')}_event_study.csv"
    )

    if skip_existing and csv_path.exists():
        logger.info(f"[SKIP] {case.case_id} — CSV exists: {csv_path.name}")
        return _load_case_result(case, csv_path, status="SKIPPED")

    module_path = CASE_MODULES.get(case.case_id)
    if not module_path:
        logger.error(f"{case.case_id}: no module registered in CASE_MODULES.")
        return {"case_id": case.case_id, "status": "NO_MODULE"}

    logger.info(f"Running {case.case_id}: {case.label} ...")
    t0 = time.time()

    try:
        # Import and exec the case module as a script (same as running it
        # directly with python).  importlib.import_module is cleaner than
        # exec(open(...)) and respects the sys.path we already set.
        mod = importlib.import_module(module_path)
        # Force reload so reruns pick up any code changes
        importlib.reload(mod)
        elapsed = time.time() - t0
        logger.success(f"  {case.case_id} done in {elapsed:.1f}s")
        return _load_case_result(case, csv_path, status="OK", elapsed=elapsed)

    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(f"  {case.case_id} FAILED in {elapsed:.1f}s: {exc}")
        logger.debug(traceback.format_exc())
        return {
            "case_id"    : case.case_id,
            "label"      : case.label,
            "stratum"    : case.stratum,
            "anchor_date": case.anchor_date,
            "nse_ticker" : case.nse_ticker,
            "status"     : "ERROR",
            "error_msg"  : str(exc),
            "elapsed_s"  : round(elapsed, 1),
        }


def _load_case_result(
    case: CaseStudy,
    csv_path: Path,
    status: str = "OK",
    elapsed: float = 0.0,
) -> dict:
    """Read the per-case event-study CSV and return a summary row."""
    base = {
        "case_id"    : case.case_id,
        "label"      : case.label,
        "stratum"    : case.stratum,
        "anchor_date": case.anchor_date,
        "nse_ticker" : case.nse_ticker,
        "shocked_sector": case.shocked_sector,
        "shock_magnitude": case.shock_magnitude,
        "status"     : status,
        "elapsed_s"  : round(elapsed, 1),
        "CAR_pct"    : None,
        "t_stat"     : None,
        "p_value"    : None,
        "significant": None,
        "impact_1d"  : None,
        "impact_7d"  : None,
        "impact_14d" : None,
        "impact_30d" : None,
        "validation" : "Insufficient",
    }

    if not csv_path.exists():
        return base

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return base
        row = df.iloc[0]
        for col in ["CAR_pct", "t_stat", "p_value", "significant",
                    "impact_1d", "impact_7d", "impact_14d", "impact_30d"]:
            if col in row:
                base[col] = row[col]

        p = base.get("p_value")
        if p is not None and not _isnan(p):
            base["validation"] = "Confirmed" if float(p) < 0.05 else "Null"
    except Exception as exc:
        logger.warning(f"  Could not read CSV for {case.case_id}: {exc}")

    return base


def _isnan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Propagation maps
# ══════════════════════════════════════════════════════════════════════════════

def run_maps(cases: list[CaseStudy]) -> None:
    """Draw the propagation map for each case."""
    # Load shared pipeline outputs once
    temporal_df = None
    events_df   = None
    t_path = DATA_PROC / "temporal_results.parquet"
    e_path = DATA_PROC / "processed_events.parquet"
    if t_path.exists():
        try:
            temporal_df = pd.read_parquet(t_path)
        except Exception as exc:
            logger.warning(f"Could not load temporal_results: {exc}")
    if e_path.exists():
        try:
            events_df = pd.read_parquet(e_path)
        except Exception as exc:
            logger.warning(f"Could not load processed_events: {exc}")

    for case in cases:
        logger.info(f"Drawing propagation map: {case.case_id} ...")
        try:
            paths = draw_propagation_map(case, temporal_df, events_df)
            logger.success(f"  Map saved: {paths['high']}")
        except Exception as exc:
            logger.error(f"  Map failed for {case.case_id}: {exc}")
            logger.debug(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-case summary
# ══════════════════════════════════════════════════════════════════════════════

def build_cross_case_summary(results: list[dict]) -> pd.DataFrame:
    """Assemble the cross-case summary DataFrame."""
    cols = [
        "case_id", "label", "stratum", "anchor_date", "nse_ticker",
        "shocked_sector", "shock_magnitude",
        "CAR_pct", "t_stat", "p_value", "significant", "validation",
        "impact_1d", "impact_7d", "impact_14d", "impact_30d",
        "status", "elapsed_s",
    ]
    df = pd.DataFrame(results)
    # Keep only columns that exist
    present = [c for c in cols if c in df.columns]
    return df[present].reset_index(drop=True)


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write a LaTeX longtable of the cross-case event-study results.
    Rows sorted by anchor_date.  Confirmed results bold.
    """
    rows: list[str] = []
    rows.append(r"\begin{longtable}{llccrrrl}")
    rows.append(r"\caption{GeoSignal Cross-Case Event Study Results} \\")
    rows.append(r"\toprule")
    rows.append(
        r"\textbf{ID} & \textbf{Case} & \textbf{Str.} & \textbf{Anchor} & "
        r"\textbf{CAR (\%)} & \textbf{t} & \textbf{p} & \textbf{Verdict} \\"
    )
    rows.append(r"\midrule")
    rows.append(r"\endfirsthead")
    rows.append(r"\toprule")
    rows.append(
        r"\textbf{ID} & \textbf{Case} & \textbf{Str.} & \textbf{Anchor} & "
        r"\textbf{CAR (\%)} & \textbf{t} & \textbf{p} & \textbf{Verdict} \\"
    )
    rows.append(r"\midrule")
    rows.append(r"\endhead")
    rows.append(r"\midrule \multicolumn{8}{r}{\textit{Continued...}} \\")
    rows.append(r"\endfoot")
    rows.append(r"\bottomrule")
    rows.append(r"\endlastfoot")

    sorted_df = df.sort_values("anchor_date") if "anchor_date" in df.columns else df
    for _, row in sorted_df.iterrows():
        car = row.get("CAR_pct")
        t   = row.get("t_stat")
        p   = row.get("p_value")
        v   = row.get("validation", "—")

        car_str = f"{float(car):+.3f}" if car is not None and not _isnan(car) else "—"
        t_str   = f"{float(t):+.3f}"  if t   is not None and not _isnan(t)   else "—"
        p_str   = f"{float(p):.4f}"   if p   is not None and not _isnan(p)   else "—"

        # Bold if confirmed significant
        confirmed = (v == "Confirmed")
        label_safe = str(row.get("label", "")).replace("&", r"\&").replace("%", r"\%")
        case_id    = str(row.get("case_id", ""))
        stratum    = str(row.get("stratum", ""))
        anchor     = str(row.get("anchor_date", ""))

        if confirmed:
            line = (
                rf"\textbf{{{case_id}}} & \textbf{{{label_safe}}} & "
                rf"\textbf{{{stratum}}} & \textbf{{{anchor}}} & "
                rf"\textbf{{{car_str}}} & \textbf{{{t_str}}} & "
                rf"\textbf{{{p_str}}} & \textbf{{Confirmed}} \\"
            )
        else:
            line = (
                rf"{case_id} & {label_safe} & {stratum} & {anchor} & "
                rf"{car_str} & {t_str} & {p_str} & {v} \\"
            )
        rows.append(line)

    rows.append(r"\end{longtable}")

    # Add note about methodology
    rows += [
        "",
        r"% MacKinlay (1997) market model; estimation window = 120 trading days;",
        r"% event window = (-5, +30); t-test on CAR; p < 0.05 = Confirmed.",
        r"% Leontief I-O results are separate (see figures).",
    ]

    out_path.write_text("\n".join(rows), encoding="utf-8")
    logger.success(f"LaTeX table -> {out_path}")


def print_console_summary(df: pd.DataFrame) -> None:
    """Print a clean cross-case table to stdout."""
    print("\n" + "=" * 90)
    print("  GeoSignal — Cross-Case Event Study Summary")
    print("=" * 90)
    print(
        f"  {'ID':<8} {'Stratum':<8} {'Anchor':<12} {'Ticker':<14} "
        f"{'CAR%':>8} {'p':>8} {'Verdict':<14} {'Status'}"
    )
    print("-" * 90)

    for _, row in df.sort_values("anchor_date").iterrows():
        car = row.get("CAR_pct")
        p   = row.get("p_value")
        car_s = f"{float(car):+.3f}" if car is not None and not _isnan(car) else "  —"
        p_s   = f"{float(p):.4f}"   if p   is not None and not _isnan(p)   else "  —"
        v     = row.get("validation", "—")
        star  = " ★" if v == "Confirmed" else ""

        print(
            f"  {row.get('case_id',''):<8} "
            f"{row.get('stratum',''):<8} "
            f"{str(row.get('anchor_date','')):<12} "
            f"{str(row.get('nse_ticker','')):<14} "
            f"{car_s:>8} "
            f"{p_s:>8} "
            f"{v + star:<14} "
            f"{row.get('status','')}"
        )

    print("-" * 90)
    confirmed = int((df["validation"] == "Confirmed").sum()) if "validation" in df.columns else 0
    null_cnt  = int((df["validation"] == "Null").sum())      if "validation" in df.columns else 0
    insuf     = int((df["validation"] == "Insufficient").sum()) if "validation" in df.columns else 0
    errors    = int((df["status"] == "ERROR").sum())          if "status" in df.columns else 0

    print(f"\n  Confirmed (p<0.05) : {confirmed}/{len(df)}")
    print(f"  Null               : {null_cnt}/{len(df)}")
    print(f"  Insufficient data  : {insuf}/{len(df)}")
    if errors:
        print(f"  Errors             : {errors}/{len(df)}")
    print(
        "\n  NOTE: Results are descriptive only for selected cases. "
        "Do not generalise beyond the 2024 Q1 archive."
    )
    print("=" * 90 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Figures: combined cross-case CAR comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_case_car(df: pd.DataFrame) -> None:
    """
    Horizontal bar chart comparing CAR(%) across all cases.
    Saves to results/figures/cross_case_CAR.png
    """
    plot_df = df[df["CAR_pct"].notna()].copy() if "CAR_pct" in df.columns else df
    if plot_df.empty:
        logger.info("No CAR data available for cross-case plot.")
        return

    import matplotlib.pyplot as plt

    plot_df = plot_df.sort_values("CAR_pct")
    labels  = plot_df["case_id"].astype(str) + "  " + plot_df["label"].astype(str).str[:30]
    values  = plot_df["CAR_pct"].astype(float)
    colors  = []
    for _, row in plot_df.iterrows():
        if row.get("validation") == "Confirmed" and float(row.get("CAR_pct", 0)) < 0:
            colors.append("#e74c3c")
        elif row.get("validation") == "Confirmed":
            colors.append("#27ae60")
        else:
            colors.append("#95a5a6")

    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.9)))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor="white", height=0.6)

    # p-value annotations
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        p = row.get("p_value")
        if p is not None and not _isnan(p):
            p_str = f"p={float(p):.4f}"
            ax.text(
                bar.get_width() + (0.05 if bar.get_width() >= 0 else -0.05),
                bar.get_y() + bar.get_height() / 2,
                p_str, va="center",
                ha="left" if bar.get_width() >= 0 else "right",
                fontsize=8, color="#2c3e50",
            )

    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("Cumulative Abnormal Return (%)", fontsize=10)
    ax.set_title(
        "GeoSignal — Cross-Case CAR Comparison\n"
        "(MacKinlay 1997 event study · red=confirmed neg · "
        "green=confirmed pos · grey=null)",
        fontsize=10,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Confirmed negative (p<0.05)"),
        mpatches.Patch(color="#27ae60", label="Confirmed positive (p<0.05)"),
        mpatches.Patch(color="#95a5a6", label="Null / insufficient"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "cross_case_CAR.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Cross-case CAR figure -> {out}")


import matplotlib.patches as mpatches   # already imported above but kept explicit


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run all GeoSignal case studies and build cross-case summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python experiments/case_studies/run_all_cases.py
  python experiments/case_studies/run_all_cases.py --case CS-03
  python experiments/case_studies/run_all_cases.py --skip-existing
  python experiments/case_studies/run_all_cases.py --maps-only
        """,
    )
    p.add_argument("--case",          default="all",
                   help="Run one case by ID (e.g. CS-03) or 'all'")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cases whose per-case CSV already exists")
    p.add_argument("--maps-only",     action="store_true",
                   help="Only draw propagation maps, skip case script reruns")
    args = p.parse_args()

    # ── Setup log file ────────────────────────────────────────────────────────
    from datetime import datetime
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"cases_{ts}.log"
    logger.add(log_file, rotation="20 MB",
               format="{time:HH:mm:ss} | {level:<8} | {message}",
               encoding="utf-8")
    logger.info(f"Case runner started — log: {log_file}")

    # ── Resolve case list ─────────────────────────────────────────────────────
    if args.case == "all":
        cases = ALL_CASES
    else:
        try:
            cases = [get_case(args.case)]
        except KeyError as exc:
            logger.error(str(exc))
            sys.exit(1)

    t_total = time.time()

    # ── Run case scripts ──────────────────────────────────────────────────────
    results: list[dict] = []
    if not args.maps_only:
        for case in cases:
            rec = run_case(case, skip_existing=args.skip_existing)
            results.append(rec)
    else:
        # Build results from existing CSVs only
        for case in cases:
            csv_path = (
                case.datasets_dir
                / f"{case.case_id.lower().replace('-', '')}_event_study.csv"
            )
            results.append(_load_case_result(case, csv_path, status="CSV_ONLY"))

    # ── Propagation maps ──────────────────────────────────────────────────────
    run_maps(cases)

    # ── Cross-case summary ────────────────────────────────────────────────────
    summary_df = build_cross_case_summary(results)

    csv_out = DATASETS_DIR / "cross_case_summary.csv"
    tex_out = DATASETS_DIR / "cross_case_summary.tex"

    summary_df.to_csv(csv_out, index=False)
    logger.success(f"Cross-case CSV -> {csv_out}")

    write_latex_table(summary_df, tex_out)

    plot_cross_case_car(summary_df)

    print_console_summary(summary_df)

    elapsed = time.time() - t_total
    logger.success(f"All cases done in {elapsed:.1f}s")
    logger.info(f"Results: {DATASETS_DIR}")
    logger.info(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

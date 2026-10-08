"""Descriptive event-window returns for the 2024-03-12 chip-controls case.

Candidate baskets are explicit researcher hypotheses. The analysis is not a
causal estimate and does not assert that the news pipeline selected them.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "datasets"
EVENT = pd.Timestamp("2024-03-12")
WINDOWS = (1, 3, 7, 14, 30)
ESTIMATION_N = 120
BASKETS = {
    "Global semiconductor firms (candidate exposure)": {
        "benchmark": "^GSPC", "members": ["NVDA", "TSM", "ASML", "INTC"]},
    "US semiconductor ETFs (context only)": {
        "benchmark": "^GSPC", "members": ["SOXX", "SMH"]},
    "Indian large-cap IT (indirect context only)": {
        "benchmark": "^NSEI", "members": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"]},
}

def fit_model(asset: pd.Series, market: pd.Series) -> tuple[float, float, int, str, str]:
    fit = pd.concat([asset.rename("a"), market.rename("m")], axis=1).dropna()
    fit = fit.loc[fit.index < EVENT].tail(ESTIMATION_N)
    if len(fit) < ESTIMATION_N:
        raise ValueError(f"Expected {ESTIMATION_N} pre-event returns, found {len(fit)}")
    design = np.column_stack([np.ones(len(fit)), fit.m.to_numpy()])
    alpha, beta = np.linalg.lstsq(design, fit.a.to_numpy(), rcond=None)[0]
    return float(alpha), float(beta), len(fit), fit.index.min().date().isoformat(), fit.index.max().date().isoformat()

def main() -> None:
    prices = pd.read_parquet(DATA / "chip_controls_candidate_prices_20230601_20240430.parquet")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    returns = np.log(prices / prices.shift(1))
    rows, used, exclusions = [], {}, {}
    for basket, spec in BASKETS.items():
        benchmark = spec["benchmark"]
        members = [t for t in spec["members"] if t in returns and returns[t].loc[returns.index < EVENT].notna().sum() >= ESTIMATION_N]
        exclusions[basket] = [t for t in spec["members"] if t not in members]
        if not members:
            continue
        used[basket] = {"benchmark": benchmark, "members": members}
        returns[basket] = returns[members].mean(axis=1, skipna=False)
        entities = [(basket, members)] + [(t, [t]) for t in members]
        for label, constituents in entities:
            alpha, beta, n, est_start, est_end = fit_model(returns[label], returns[benchmark])
            event = pd.concat([returns[label].rename("asset"), returns[benchmark].rename("market")], axis=1)
            event = event.loc[event.index >= EVENT].dropna()
            event["abnormal"] = event.asset - (alpha + beta * event.market)
            for window in WINDOWS:
                w = event.head(window)
                if len(w) < window: continue
                rows.append({
                    "event_date": EVENT.date().isoformat(), "series": label,
                    "constituents": ";".join(constituents), "benchmark": benchmark,
                    "window_trading_days": window, "first_return_date": w.index.min().date().isoformat(),
                    "last_return_date": w.index.max().date().isoformat(),
                    "estimation_observations": n, "estimation_start": est_start, "estimation_end": est_end,
                    "market_model_alpha_daily": alpha, "market_model_beta": beta,
                    "raw_cumulative_return_pct": (np.exp(w.asset.sum()) - 1) * 100,
                    "benchmark_cumulative_return_pct": (np.exp(w.market.sum()) - 1) * 100,
                    "market_model_CAR_pct_log_approx": w.abnormal.sum() * 100,
                    "interpretation": "descriptive single-event market-model estimate; no causal or predictive claim",
                })
    detail = pd.DataFrame(rows)
    detail.to_csv(DATA / "chip_controls_market_model_event_study_20240312.csv", index=False)
    summary = {
        "case": "US semiconductor export-controls news, 2024-03-12",
        "source": "Yahoo Finance adjusted-close history, downloaded 2026-10-05, requested 2023-06-01 through 2024-04-30",
        "benchmark_models": used, "excluded_candidates": exclusions,
        "event_windows_trading_days": list(WINDOWS), "estimation_window_trading_days": ESTIMATION_N,
        "limitations": ["Researcher-defined baskets; not system-generated security matches.",
                        "The event date is the reporting date and may not equal first public information time.",
                        "One event; descriptive market-model CAR only, without significance testing or causal attribution.",
                        "The Indian IT basket is indirect context, not a semiconductor exposure measure."],
        "outputs": ["chip_controls_candidate_prices_20230601_20240430.csv",
                    "chip_controls_candidate_prices_20230601_20240430.parquet",
                    "chip_controls_market_model_event_study_20240312.csv"],
    }
    (DATA / "chip_controls_market_event_study_20240312.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nBASKET WINDOWS")
    print(detail[detail.series.isin(used)].to_string(index=False))

if __name__ == "__main__": main()

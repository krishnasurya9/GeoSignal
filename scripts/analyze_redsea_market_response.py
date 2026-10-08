"""Descriptive market-model event study for the Red Sea case.

Uses archived Yahoo Finance adjusted closes. Portfolios are transparent,
equal-weight candidate-exposure baskets, not verified affected securities.
The one-case estimates are descriptive and do not establish causality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "datasets"
EVENT_DATE = pd.Timestamp("2024-01-12")
BENCHMARK = "^NSEI"
WINDOWS = (1, 3, 7, 14, 30)
ESTIMATION_OBSERVATIONS = 120

PORTFOLIOS = {
    "Ports and logistics (candidate exposure)": [
        "ADANIPORTS.NS", "CONCOR.NS", "GESHIP.NS", "SCI.NS", "AEGISLOG.NS",
    ],
    "Energy (candidate exposure)": [
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",
    ],
    "FMCG (candidate exposure)": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    ],
}


def market_model(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    fit = pd.concat([y.rename("asset"), x.rename("market")], axis=1).dropna()
    if len(fit) < ESTIMATION_OBSERVATIONS:
        raise ValueError(f"Only {len(fit)} pre-event observations; need {ESTIMATION_OBSERVATIONS}.")
    fit = fit.tail(ESTIMATION_OBSERVATIONS)
    design = np.column_stack([np.ones(len(fit)), fit["market"].to_numpy()])
    alpha, beta = np.linalg.lstsq(design, fit["asset"].to_numpy(), rcond=None)[0]
    return float(alpha), float(beta), int(len(fit))


def summarize_series(name: str, observed: pd.Series, market: pd.Series,
                     estimation: pd.DataFrame, constituents: list[str]) -> list[dict]:
    alpha, beta, n_est = market_model(estimation[BENCHMARK], estimation[name])
    event = pd.concat([observed.rename("asset_return"), market.rename("market_return")], axis=1)
    event = event.loc[event.index >= EVENT_DATE].dropna()
    event["expected_return"] = alpha + beta * event["market_return"]
    event["abnormal_return"] = event["asset_return"] - event["expected_return"]
    rows = []
    for window in WINDOWS:
        w = event.head(window)
        if len(w) < window:
            continue
        rows.append({
            "event_date": EVENT_DATE.date().isoformat(),
            "portfolio": name,
            "constituent_count": len(constituents),
            "constituents": ";".join(constituents),
            "window_trading_days": window,
            "first_return_date": w.index.min().date().isoformat(),
            "last_return_date": w.index.max().date().isoformat(),
            "estimation_observations": n_est,
            "estimation_window_start": estimation.dropna().tail(n_est).index.min().date().isoformat(),
            "estimation_window_end": estimation.dropna().tail(n_est).index.max().date().isoformat(),
            "market_model_alpha_daily": alpha,
            "market_model_beta": beta,
            "raw_cumulative_return_pct": (np.exp(w["asset_return"].sum()) - 1) * 100,
            "benchmark_cumulative_return_pct": (np.exp(w["market_return"].sum()) - 1) * 100,
            "market_model_CAR_pct_log_approx": w["abnormal_return"].sum() * 100,
            "interpretation": "descriptive single-event market-model estimate; not causal or statistically validated",
        })
    return rows


def main() -> None:
    prices_path = DATA / "redsea_candidate_security_prices_20230601_20240430.parquet"
    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    if BENCHMARK not in prices:
        raise ValueError(f"Benchmark {BENCHMARK} is missing from {prices_path.name}.")

    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    available = set(prices.columns)
    result_rows: list[dict] = []
    used: dict[str, list[str]] = {}
    for portfolio_name, planned in PORTFOLIOS.items():
        constituents = [ticker for ticker in planned if ticker in available and prices[ticker].notna().sum() > 130]
        if not constituents:
            continue
        used[portfolio_name] = constituents
        # Daily equal-weight log-return basket with fixed constituents.
        returns[portfolio_name] = returns[constituents].mean(axis=1, skipna=False)
        pre = returns.loc[returns.index < EVENT_DATE, [BENCHMARK, portfolio_name]].dropna()
        observed = returns[portfolio_name]
        result_rows.extend(summarize_series(
            portfolio_name, observed, returns[BENCHMARK], pre, constituents
        ))

    # Include individual securities as a transparent supplement to basket results.
    for portfolio_name, constituents in used.items():
        for ticker in constituents:
            pre = returns.loc[returns.index < EVENT_DATE, [BENCHMARK, ticker]].dropna()
            result_rows.extend(summarize_series(
                ticker, returns[ticker], returns[BENCHMARK], pre, [ticker]
            ))

    # Raw broad-market returns are useful context, but the market cannot be
    # market-model-adjusted against itself.
    market_rows = []
    market = returns[BENCHMARK].loc[lambda s: s.index >= EVENT_DATE].dropna()
    for window in WINDOWS:
        w = market.head(window)
        if len(w) == window:
            market_rows.append({
                "event_date": EVENT_DATE.date().isoformat(),
                "ticker": BENCHMARK,
                "window_trading_days": window,
                "first_return_date": w.index.min().date().isoformat(),
                "last_return_date": w.index.max().date().isoformat(),
                "raw_cumulative_return_pct": (np.exp(w.sum()) - 1) * 100,
            })

    detail = pd.DataFrame(result_rows)
    detail.to_csv(DATA / "redsea_market_model_event_study_20240112.csv", index=False)
    pd.DataFrame(market_rows).to_csv(DATA / "redsea_market_benchmark_windows_20240112.csv", index=False)
    summary = {
        "case": "Red Sea / Houthi event sample",
        "event_date": EVENT_DATE.date().isoformat(),
        "source": "Yahoo Finance adjusted close history downloaded for 2023-06-01 through 2024-04-30",
        "benchmark": BENCHMARK,
        "estimation_window_trading_days": ESTIMATION_OBSERVATIONS,
        "event_windows_trading_days": list(WINDOWS),
        "candidate_exposure_baskets": used,
        "candidate_exposure_warning": "Basket membership is a researcher-defined exposure hypothesis, not a validated system prediction.",
        "market_model_warning": "One event only; CARs are descriptive. No p-values or causal claims are made.",
        "missing_tickers": {
            "^CNXFMCG": "Yahoo Finance returned no historical observations for the requested interval; equal-weight FMCG constituents used as an explicit proxy.",
            "HPCL.NS": "No data returned for requested interval; excluded from Energy basket.",
            "TATAMOTORS.NS": "No data returned for requested interval; excluded from candidate security set.",
        },
        "outputs": [
            "redsea_candidate_security_prices_20230601_20240430.parquet",
            "redsea_candidate_security_prices_20230601_20240430.csv",
            "redsea_market_model_event_study_20240112.csv",
            "redsea_market_benchmark_windows_20240112.csv",
        ],
    }
    (DATA / "redsea_market_event_study_20240112.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nBASKET EVENT-STUDY RESULTS")
    print(detail[detail["constituent_count"].gt(1)][[
        "portfolio", "window_trading_days", "raw_cumulative_return_pct",
        "benchmark_cumulative_return_pct", "market_model_CAR_pct_log_approx",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

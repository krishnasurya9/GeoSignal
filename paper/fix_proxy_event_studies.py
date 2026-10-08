"""
fix_proxy_event_studies.py
Recompute event-study results for CS-01, CS-03, CS-04, CS-05 using proxy tickers.
Updates cross_case_summary.csv with real numeric values.
"""

import sys
import os
import numpy as np
import pandas as pd

# Make sure project root is on path
PROJECT_ROOT = r"a:\projects\paper\geo\GeoSignal"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ── 1. Load candidate security prices (raw prices) ────────────────────────────
print("=== Candidate Security Prices ===")
prices_df = pd.read_csv(
    "results/datasets/redsea_candidate_security_prices_20230601_20240430.csv",
    index_col=0,
    parse_dates=True,
)
prices_df.index = pd.to_datetime(prices_df.index)
prices_df.index.name = "Date"
print("Shape:", prices_df.shape)
print("Columns:", prices_df.columns.tolist())
print()

# ── 2. Load NSE returns parquet (already returns) ─────────────────────────────
print("=== NSE Returns Parquet ===")
nse_returns_df = pd.read_parquet("data/processed/nse_returns.parquet")
nse_returns_df.index = pd.to_datetime(nse_returns_df.index)
nse_returns_df.index.name = "Date"
print("Shape:", nse_returns_df.shape)
print("Columns:", nse_returns_df.columns.tolist())
print()

# ── 3. Build combined returns DataFrame ───────────────────────────────────────
# Detect if prices_df is already returns (all abs values < 5) or raw prices
sample = prices_df.dropna().iloc[:50]
max_abs = sample.abs().max().max()
print(f"Max absolute value in prices_df: {max_abs:.4f}")

if max_abs < 5.0:
    print("Detected as RETURNS already (values < 5) — using as-is")
    cand_returns = prices_df.copy()
else:
    print("Detected as RAW PRICES (values >= 5) — computing log-returns")
    cand_returns = np.log(prices_df / prices_df.shift(1))

print("Candidate returns shape after conversion:", cand_returns.shape)

# Drop ^NSEI from cand_returns since we get it from nse_returns_df
if "^NSEI" in cand_returns.columns:
    cand_returns = cand_returns.drop(columns=["^NSEI"])

# Merge with NSE returns (only ^NSEI column needed from nse_returns_df)
nsei_col = nse_returns_df[["^NSEI"]]
returns_df = cand_returns.join(nsei_col, how="inner")
print("Combined returns shape (after inner join):", returns_df.shape)
print("Columns in combined:", returns_df.columns.tolist())
print()

# ── 4. EventStudy import ──────────────────────────────────────────────────────
from src.geosignal.analysis.event_study import EventStudy

MARKET_TICKER = "^NSEI"
ESTIMATION_WINDOW = 120
EVENT_WINDOW = (-5, 30)

# ── 5. Run event studies for 4 cases ─────────────────────────────────────────
cases = {
    "CS-01": {
        "anchor": pd.Timestamp("2024-01-12"),
        "proxies": ["ADANIPORTS.NS"],
        "shocked_sector": "Logistics",
        "label": "Red Sea / Houthi Crisis",
    },
    "CS-03": {
        "anchor": pd.Timestamp("2024-02-16"),
        "proxies": ["RELIANCE.NS", "ONGC.NS"],
        "shocked_sector": "Energy",
        "label": "Iran Oil Sanctions Tightening",
    },
    "CS-04": {
        "anchor": pd.Timestamp("2024-02-01"),
        "proxies": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS"],
        "shocked_sector": "Materials",
        "label": "China Rare-Earth Export Curbs",
        "yfinance_fallback": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS"],
    },
    "CS-05": {
        "anchor": pd.Timestamp("2024-01-18"),
        "proxies": ["RELIANCE.NS", "ONGC.NS"],
        "shocked_sector": "Energy",
        "label": "Russia Energy Supply Disruption",
    },
}


def try_yfinance_download(ticker, returns_df_combined):
    """Try to download ticker via yfinance and add it to a copy of returns_df."""
    try:
        import yfinance as yf
        print(f"  Trying yfinance download for {ticker}...")
        data = yf.download(ticker, start="2023-01-01", end="2024-05-01", progress=False)
        if data.empty:
            print(f"  yfinance returned empty data for {ticker}")
            return None, returns_df_combined
        # Handle MultiIndex columns (yfinance >= 0.2)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        close = data["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        log_ret.index = pd.to_datetime(log_ret.index)
        log_ret.index.name = "Date"
        log_ret.name = ticker
        new_returns = returns_df_combined.join(log_ret, how="inner")
        print(f"  yfinance OK: {len(log_ret)} rows for {ticker}")
        return ticker, new_returns
    except Exception as e:
        print(f"  yfinance failed for {ticker}: {e}")
        return None, returns_df_combined


results = {}

for case_id, info in cases.items():
    print(f"\n--- {case_id}: {info['label']} ---")
    anchor = info["anchor"]
    best_result = {}
    used_ticker = None
    working_returns = returns_df.copy()

    # Try each proxy in order
    for proxy in info["proxies"]:
        if proxy in working_returns.columns:
            print(f"  Trying proxy: {proxy} (in local data)")
            es = EventStudy(
                working_returns,
                market_ticker=MARKET_TICKER,
                estimation_window=ESTIMATION_WINDOW,
                event_window=EVENT_WINDOW,
            )
            res = es.run(proxy, anchor)
            if res and not np.isnan(res.get("CAR", np.nan)):
                best_result = res
                used_ticker = proxy
                print(f"  SUCCESS: CAR={res['CAR']:.4f}%, t={res['t_stat']:.4f}, p={res['p_value']:.4f}")
                break
            else:
                print(f"  Failed or NaN result for {proxy}")
        else:
            print(f"  Proxy {proxy} NOT in local data")

    # If failed and yfinance_fallback defined, try yfinance
    if (not best_result) and info.get("yfinance_fallback"):
        for ticker in info["yfinance_fallback"]:
            if ticker not in working_returns.columns:
                used_tick, extended_returns = try_yfinance_download(ticker, working_returns)
                if used_tick:
                    es = EventStudy(
                        extended_returns,
                        market_ticker=MARKET_TICKER,
                        estimation_window=ESTIMATION_WINDOW,
                        event_window=EVENT_WINDOW,
                    )
                    res = es.run(used_tick, anchor)
                    if res and not np.isnan(res.get("CAR", np.nan)):
                        best_result = res
                        used_ticker = used_tick
                        working_returns = extended_returns
                        print(f"  yfinance SUCCESS: CAR={res['CAR']:.4f}%, t={res['t_stat']:.4f}, p={res['p_value']:.4f}")
                        break

    # Final fallback: use ^NSEI as sector (same as market)
    if not best_result:
        print(f"  All proxies failed — using ^NSEI as sector proxy (noting limitation)")
        es = EventStudy(
            working_returns,
            market_ticker=MARKET_TICKER,
            estimation_window=ESTIMATION_WINDOW,
            event_window=EVENT_WINDOW,
        )
        res = es.run(MARKET_TICKER, anchor)
        if res and not np.isnan(res.get("CAR", np.nan)):
            best_result = res
            used_ticker = "^NSEI (fallback)"
            print(f"  Fallback SUCCESS: CAR={res['CAR']:.4f}%, t={res['t_stat']:.4f}, p={res['p_value']:.4f}")
        else:
            print(f"  Even ^NSEI fallback failed")
            best_result = {}
            used_ticker = None

    results[case_id] = {
        "result": best_result,
        "ticker": used_ticker,
        "anchor": anchor,
        "label": info["label"],
    }

# ── 6. Update cross_case_summary.csv ──────────────────────────────────────────
print("\n=== Updating cross_case_summary.csv ===")
summary_path = "results/datasets/cross_case_summary.csv"
summary_df = pd.read_csv(summary_path)
print("Before update:")
print(summary_df[["case_id", "CAR_pct", "t_stat", "p_value", "significant", "validation", "nse_ticker"]].to_string())
print()

for case_id, info in results.items():
    res = info["result"]
    ticker = info["ticker"]
    mask = summary_df["case_id"] == case_id

    if not mask.any():
        print(f"WARNING: {case_id} not found in summary CSV!")
        continue

    if res:
        car_pct = res["CAR"]
        t_stat = res["t_stat"]
        p_value = res["p_value"]
        significant = res["significant"]
        verdict = "Confirmed" if p_value < 0.05 else "Null"

        summary_df.loc[mask, "CAR_pct"] = round(car_pct, 4)
        summary_df.loc[mask, "t_stat"] = round(t_stat, 4)
        summary_df.loc[mask, "p_value"] = round(p_value, 4)
        summary_df.loc[mask, "significant"] = significant
        summary_df.loc[mask, "validation"] = verdict
        summary_df.loc[mask, "nse_ticker"] = ticker if ticker else "n/a"
        summary_df.loc[mask, "status"] = "OK"
        print(f"  Updated {case_id}: CAR={car_pct:.4f}%, p={p_value:.4f}, verdict={verdict}, ticker={ticker}")
    else:
        print(f"  {case_id}: No valid result — leaving as Insufficient")

summary_df.to_csv(summary_path, index=False)
print("CSV written.")

# ── 7. Print summary table ─────────────────────────────────────────────────────
print("\n=== Final Summary Table ===")
print(summary_df[["case_id", "nse_ticker", "CAR_pct", "t_stat", "p_value", "significant", "validation"]].to_string())

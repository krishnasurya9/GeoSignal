"""
notebooks/06_granger_causality.py
Phase IV — ADF stationarity, Toda-Yamamoto Granger tests, cross-correlation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import ccf

from config.settings import DATA_PROC, RESULTS_DIR, NSE_SECTOR_TICKERS, GRANGER_MAX_LAGS
from src.temporal_analysis import granger_causality_test, test_stationarity

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Build geo-risk series ─────────────────────────────────────────────────────

events  = pd.read_parquet(DATA_PROC / "processed_events.parquet")
returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
events["date"] = pd.to_datetime(events["date"])
returns.index  = pd.to_datetime(returns.index)

geo_risk = -events.groupby("date")["goldstein"].mean().sort_index()
geo_risk.name = "geo_risk"

geo_aligned = geo_risk.reindex(returns.index, method="ffill").fillna(geo_risk.mean())
print(f"geo_risk obs: {len(geo_risk)}  |  aligned: {len(geo_aligned)}")

# ── Geo-risk time series ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 4))
roll = geo_risk.rolling(7, min_periods=1).mean()
ax.fill_between(geo_risk.index, geo_risk.values, alpha=0.2, color="coral")
ax.plot(roll.index, roll.values, color="coral", lw=2, label="7-day MA")
ax.axhline(geo_risk.mean(), color="black", ls="--", lw=0.8)
ax.set_title("Daily Geopolitical Risk Index  (–Goldstein mean)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.xticks(rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "geo_risk.png", dpi=150)
plt.show()

# ── ADF stationarity ──────────────────────────────────────────────────────────

adf_rows = []
adf_geo  = test_stationarity(geo_risk)
adf_rows.append({"series": "geo_risk", **adf_geo})

for name, ticker in NSE_SECTOR_TICKERS.items():
    if ticker in returns.columns:
        r = test_stationarity(returns[ticker].dropna())
        adf_rows.append({"series": f"{name} ({ticker})", **r})

adf_df = pd.DataFrame(adf_rows).set_index("series")
print(adf_df.to_string())

# ── Granger causality tests ───────────────────────────────────────────────────

summary = []
detail  = {}

for name, ticker in NSE_SECTOR_TICKERS.items():
    if ticker not in returns.columns:
        continue
    sec    = returns[ticker].dropna()
    geo    = geo_aligned.reindex(sec.index).dropna()
    common = sec.index.intersection(geo.index)

    if len(common) < GRANGER_MAX_LAGS * 4:
        summary.append({
            "sector": name, "ticker": ticker, "n_obs": len(common),
            "min_p": None, "best_lag": None, "significant": None, "note": "underpowered",
        })
        continue

    gc = granger_causality_test(geo.loc[common], sec.loc[common])
    if gc.empty:
        continue
    detail[name] = gc
    best = gc.loc[gc["p_value"].idxmin()]
    summary.append({
        "sector": name, "ticker": ticker, "n_obs": len(common),
        "min_p": round(best["p_value"], 4), "best_lag": int(best["lag"]),
        "significant": bool(gc["significant"].any()), "note": "",
    })

gc_df = pd.DataFrame(summary)
gc_df.to_csv(RESULTS_DIR / "granger_summary.csv", index=False)
print(gc_df.to_string(index=False))

# ── Significance bar chart ────────────────────────────────────────────────────

gc_plot = gc_df.dropna(subset=["min_p"])
if not gc_plot.empty:
    neg_log_p = -np.log10(gc_plot["min_p"].clip(1e-10))
    colors    = ["steelblue" if s else "lightgrey" for s in gc_plot["significant"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(gc_plot["sector"], neg_log_p, color=colors, alpha=0.85)
    ax.axvline(-np.log10(0.05), color="red",     ls="--", label="α=0.05")
    ax.axvline(-np.log10(0.01), color="darkred",  ls=":",  label="α=0.01")
    ax.set_title("Granger: geo_risk → sector_return  (–log₁₀ min p-value)")
    ax.set_xlabel("–log₁₀(min p-value)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "granger_significance.png", dpi=150)
    plt.show()

# ── Lag-profile for significant sectors ──────────────────────────────────────

sig_sectors = gc_df[gc_df["significant"] == True]["sector"].tolist()
if sig_sectors:
    fig, axes = plt.subplots(1, len(sig_sectors),
                             figsize=(6 * len(sig_sectors), 4), sharey=False)
    if len(sig_sectors) == 1:
        axes = [axes]
    for ax, name in zip(axes, sig_sectors):
        gc = detail[name]
        colors = ["steelblue" if s else "lightgrey" for s in gc["significant"]]
        ax.bar(gc["lag"], -np.log10(gc["p_value"].clip(1e-10)),
               color=colors, alpha=0.85)
        ax.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
        ax.set_title(f"{name}  –log₁₀(p) by lag")
        ax.set_xlabel("Lag (days)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "granger_lag_profile.png", dpi=150)
    plt.show()

# ── Cross-correlation ─────────────────────────────────────────────────────────

max_lag = min(20, len(returns) // 4)
fig, ax = plt.subplots(figsize=(14, 5))
common_len = 0

for name, ticker in list(NSE_SECTOR_TICKERS.items())[:6]:
    if ticker not in returns.columns:
        continue
    sec    = returns[ticker].dropna()
    geo    = geo_aligned.reindex(sec.index).dropna()
    common = sec.index.intersection(geo.index)
    if len(common) < max_lag * 3:
        continue
    common_len = len(common)
    cors = ccf(geo.loc[common], sec.loc[common], adjusted=False)[:max_lag + 1]
    ax.plot(range(max_lag + 1), cors, marker="o", ms=3, lw=1.5, label=name)

ci = 1.96 / np.sqrt(common_len) if common_len > 0 else 0.1
ax.axhline(0,   color="black", lw=0.8)
ax.axhline( ci, color="grey",  ls="--", lw=0.8)
ax.axhline(-ci, color="grey",  ls="--", lw=0.8)
ax.set_title(f"Cross-correlation: geo_risk → sector_return  (lag 0–{max_lag})")
ax.set_xlabel("Lag (trading days)")
ax.set_ylabel("CCF")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "crosscorr.png", dpi=150)
plt.show()

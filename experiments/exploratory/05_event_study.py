"""
notebooks/05_event_study.py
Phase IV — MacKinlay (1997) event study: CAR distribution, sector breakdown, walkthrough
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import DATA_PROC, RESULTS_DIR, NSE_SECTOR_TICKERS, NIFTY50_TICKER
from src.temporal_analysis import EventStudy

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Load ──────────────────────────────────────────────────────────────────────

results = pd.read_parquet(DATA_PROC / "temporal_results.parquet")
returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
returns.index = pd.to_datetime(returns.index)

print(f"Temporal results: {results.shape}")
print(results.dtypes)

# ── CAR distribution + significance ──────────────────────────────────────────

cars = results["CAR_pct"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(cars.clip(-20, 20), bins=60, color="steelblue", alpha=0.85, edgecolor="white")
axes[0].axvline(cars.mean(), color="red", ls="--", label=f"μ={cars.mean():.3f}%")
axes[0].axvline(0, color="black", ls=":", lw=1)
axes[0].set_title("CAR Distribution (±20% clip)")
axes[0].set_xlabel("CAR (%)")
axes[0].set_ylabel("Events")
axes[0].legend()

if "significant" in results.columns:
    vc = results["significant"].value_counts()
    axes[1].pie(vc.values,
                labels=[f"{'Sig' if k else 'Not sig'}\n{v:,}" for k, v in vc.items()],
                colors=["steelblue", "lightgrey"], autopct="%1.1f%%")
    axes[1].set_title("CAR Statistical Significance (p < 0.05)")

plt.tight_layout()
plt.savefig(FIG_DIR / "car_dist.png", dpi=150)
plt.show()

# ── CAR by sector ─────────────────────────────────────────────────────────────

if "sector" in results.columns:
    sec_stats = (
        results.groupby("sector")["CAR_pct"]
        .agg(["mean", "median", "count", "std"])
        .round(4).sort_values("mean")
    )
    print(sec_stats.to_string())

    colors = ["tomato" if v < 0 else "steelblue" for v in sec_stats["mean"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sec_stats.index, sec_stats["mean"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Mean CAR (%) by GICS Sector")
    ax.set_xlabel("Mean CAR (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "car_by_sector.png", dpi=150)
    plt.show()

# ── Raw impact windows ────────────────────────────────────────────────────────

impact_cols = [c for c in results.columns if c.startswith("impact_")]
if impact_cols:
    fig, axes = plt.subplots(1, len(impact_cols), figsize=(5 * len(impact_cols), 4))
    if len(impact_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, impact_cols):
        v = results[col].dropna()
        ax.hist(v.clip(-15, 15), bins=50, color="coral", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", ls=":", lw=1)
        ax.axvline(v.mean(), color="red", ls="--", label=f"μ={v.mean():.3f}%")
        ax.set_title(col)
        ax.legend(fontsize=8)
    plt.suptitle("Raw Cumulative Returns by Window", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "impact_windows.png", dpi=150)
    plt.show()

# ── Walkthrough: most significant event ──────────────────────────────────────

if "significant" in results.columns and results["significant"].any():
    row    = results[results["significant"]].nsmallest(1, "p_value").iloc[0]
    ticker = row.get("ticker")
    edate  = pd.Timestamp(row["event_date"])

    es = EventStudy(returns)
    ev = es.run(ticker, edate)

    if ev and "ar_series" in ev:
        ar  = ev["ar_series"] * 100
        car = ar.cumsum()
        t   = list(range(len(ar)))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(t, ar.values,
                    color=["tomato" if v < 0 else "steelblue" for v in ar.values],
                    alpha=0.85)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_title(f"AR(%)  {ticker}  {edate.date()}")
        axes[0].set_xlabel("Days after event")
        axes[0].set_ylabel("AR (%)")

        axes[1].plot(t, car.values, color="steelblue", lw=2, marker="o", ms=3)
        axes[1].fill_between(t, 0, car.values, alpha=0.18, color="steelblue")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title(f"CAR={ev['CAR']:.3f}%  t={ev['t_stat']}  p={ev['p_value']}")
        axes[1].set_xlabel("Days after event")
        axes[1].set_ylabel("CAR (%)")

        plt.suptitle("MacKinlay (1997) Event Study Walkthrough", y=1.01)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "event_study_walkthrough.png", dpi=150)
        plt.show()
        print(f"α={ev['alpha']:.6f}  β={ev['beta']:.6f}")
        print(f"CAR={ev['CAR']}%  t={ev['t_stat']}  p={ev['p_value']}")

# ── p-value distribution ──────────────────────────────────────────────────────

if "p_value" in results.columns:
    pv = results["p_value"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pv, bins=50, color="mediumpurple", alpha=0.85, edgecolor="white")
    ax.axvline(0.05, color="red",    ls="--", label="α=0.05")
    ax.axvline(0.01, color="darkred", ls=":",  label="α=0.01")
    ax.set_title("p-value Distribution")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pvalue_dist.png", dpi=150)
    plt.show()
    print(f"p<0.05: {(pv<0.05).sum():,}  ({100*(pv<0.05).mean():.1f}%)")
    print(f"p<0.01: {(pv<0.01).sum():,}  ({100*(pv<0.01).mean():.1f}%)")

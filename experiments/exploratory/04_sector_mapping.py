"""
notebooks/04_sector_mapping.py
Phase III — GICS sector distribution, NSE returns coverage, risk/return profile
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config.settings import DATA_PROC, RESULTS_DIR, NSE_SECTOR_TICKERS, NIFTY50_TICKER

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Load ──────────────────────────────────────────────────────────────────────

pairs   = pd.read_parquet(DATA_PROC / "event_news_sector.parquet")
returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
returns.index = pd.to_datetime(returns.index)

print(f"Sector pairs: {pairs.shape}")
print(f"NSE returns:  {returns.shape}  {returns.index.min().date()} → {returns.index.max().date()}")

# ── GICS sector distribution ──────────────────────────────────────────────────

sc = pairs["sector"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].barh(sc.index, sc.values,
             color=sns.color_palette("tab10", len(sc)), alpha=0.9)
axes[0].invert_yaxis()
axes[0].set_title("GICS Sector Assignment")
axes[0].set_xlabel("Pair count")

sc_no_unk = sc.drop("Unknown", errors="ignore")
axes[1].pie(sc_no_unk.values, labels=sc_no_unk.index,
            colors=sns.color_palette("tab10", len(sc_no_unk)),
            autopct="%1.1f%%", startangle=140)
axes[1].set_title("GICS Distribution (excl. Unknown)")

plt.tight_layout()
plt.savefig(FIG_DIR / "sector_dist.png", dpi=150)
plt.show()

print(f"\nUnknown rate: {100*(pairs['sector']=='Unknown').mean():.1f}%")

# ── Data coverage audit ───────────────────────────────────────────────────────

coverage = (returns.notna().sum() / len(returns) * 100).round(1).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
coverage.plot(kind="barh", ax=ax, color="steelblue", alpha=0.85)
ax.axvline(95, color="red", linestyle="--", label="95% threshold")
ax.set_title("NSE Index Data Coverage (%)")
ax.set_xlabel("%")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "nse_coverage.png", dpi=150)
plt.show()

# ── Cumulative returns ────────────────────────────────────────────────────────

cum_ret = (1 + returns.fillna(0)).cumprod() - 1

fig, ax = plt.subplots(figsize=(14, 6))
for col in cum_ret.columns:
    lw = 2.5 if col == NIFTY50_TICKER else 1.2
    ls = "--" if col == NIFTY50_TICKER else "-"
    ax.plot(cum_ret.index, cum_ret[col] * 100, label=col, lw=lw, ls=ls)
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_title("NSE Sectoral Indices — Cumulative Return (%)")
ax.set_ylabel("%")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30, ha="right")
ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "cumulative_returns.png", dpi=150)
plt.show()

# ── Return correlation heatmap ────────────────────────────────────────────────

corr = returns.dropna(how="all").corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.4,
            cbar_kws={"label": "Pearson r"})
ax.set_title("NSE Sectoral Return Correlations")
plt.tight_layout()
plt.savefig(FIG_DIR / "return_corr.png", dpi=150)
plt.show()

# ── Risk / return scatter ─────────────────────────────────────────────────────

ann_vol = returns.std() * np.sqrt(252) * 100
ann_ret = returns.mean() * 252 * 100

fig, ax = plt.subplots(figsize=(9, 6))
for t in ann_vol.index:
    ax.scatter(ann_vol[t], ann_ret[t], s=80, zorder=3)
    ax.annotate(t.replace("^", ""), (ann_vol[t], ann_ret[t]),
                xytext=(5, 0), textcoords="offset points", fontsize=8)
ax.axhline(0, color="black", ls=":", lw=0.8)
ax.set_xlabel("Annualised Vol (%)")
ax.set_ylabel("Annualised Return (%)")
ax.set_title("NSE Sectors — Risk / Return")
plt.tight_layout()
plt.savefig(FIG_DIR / "risk_return.png", dpi=150)
plt.show()

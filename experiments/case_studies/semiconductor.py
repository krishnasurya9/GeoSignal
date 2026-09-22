"""
experiments/case_studies/semiconductor.py
Semiconductor Export Controls Case Study (US BIS rules, 2022–2024)

Requires completed pipeline outputs in data/processed/ and data/interim/.
Run: python experiments/case_studies/semiconductor.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_PROC, RESULTS_DIR, NIFTY50_TICKER
from geosignal.analysis.event_study import EventStudy
from geosignal.analysis.granger_causality import granger_causality_test
from geosignal.propagation.leontief import LeontiefPropagator

INTERIM  = PROJECT_ROOT / "data" / "interim"
FIG_DIR  = RESULTS_DIR / "figures" / "semicon"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Parameters ────────────────────────────────────────────────────────────────

DATES = {
    "CHIPS Act signed":    pd.Timestamp("2022-08-09"),
    "BIS EAR rule":        pd.Timestamp("2022-10-07"),
    "BIS update Oct 2023": pd.Timestamp("2023-10-17"),
    "Jan 2024 tightening": pd.Timestamp("2024-01-29"),
}
WINDOW    = ("2022-01-01", "2024-03-31")
IT_TICKER = "^CNXIT"
SC_KW     = ["semiconductor", "chip", "euv", "export control", "tsmc",
             "nvidia", "asml", "entity list"]

# ── Load ──────────────────────────────────────────────────────────────────────

events  = pd.read_parquet(DATA_PROC / "processed_events.parquet")
returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
events["date"] = pd.to_datetime(events["date"])
returns.index  = pd.to_datetime(returns.index)

mask = (
    events["actor1"].fillna("").str.lower().str.contains("|".join(SC_KW)) |
    events["actor2"].fillna("").str.lower().str.contains("|".join(SC_KW))
)
sc_events  = events[mask & (events["date"] >= WINDOW[0]) & (events["date"] <= WINDOW[1])]
all_window = events[(events["date"] >= WINDOW[0]) & (events["date"] <= WINDOW[1])]
print(f"Semiconductor events in window: {len(sc_events):,}")

# ── Event timeline ────────────────────────────────────────────────────────────

daily_sc = sc_events.groupby("date").size().sort_index()
daily_gs = all_window.groupby("date")["goldstein"].mean()

fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
axes[0].bar(daily_sc.index, daily_sc.values, color="mediumpurple", alpha=0.8, width=1)
for label, dt in DATES.items():
    if daily_sc.index.min() <= dt <= daily_sc.index.max():
        axes[0].axvline(dt, ls="--", lw=1.5, label=label[:28])
axes[0].set_title("Semiconductor Export Control — GDELT Event Count")
axes[0].set_ylabel("Events")
axes[0].legend(fontsize=7)

roll7 = daily_gs.rolling(7, min_periods=1).mean()
axes[1].fill_between(daily_gs.index, daily_gs.values, alpha=0.2, color="steelblue")
axes[1].plot(roll7.index, roll7.values, color="steelblue", lw=2)
axes[1].axhline(0, color="black", ls=":", lw=0.8)
for dt in DATES.values():
    if daily_gs.index.min() <= dt <= daily_gs.index.max():
        axes[1].axvline(dt, ls="--", lw=1, color="grey")
axes[1].set_title("Mean Goldstein — 7-day Rolling")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "semicon_timeline.png", dpi=150)
plt.show()

# ── NSE IT vs Nifty50 ─────────────────────────────────────────────────────────

if IT_TICKER in returns.columns:
    wr  = returns.loc[WINDOW[0]:WINDOW[1], [NIFTY50_TICKER, IT_TICKER]]
    cum = (1 + wr.fillna(0)).cumprod() - 1
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(cum.index, cum[NIFTY50_TICKER] * 100, label="Nifty50", lw=2.5, ls="--", color="black")
    ax.plot(cum.index, cum[IT_TICKER] * 100,       label="CNXIT",   lw=2,   color="mediumpurple")
    for label, dt in DATES.items():
        if cum.index.min() <= dt <= cum.index.max():
            ax.axvline(dt, ls="--", lw=1.2)
            ax.annotate(label[:22], (dt, ax.get_ylim()[1] * 0.9),
                        rotation=90, fontsize=7, va="top", ha="right")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_title("NSE IT Index vs Nifty50 — Semiconductor Export Control Window")
    ax.set_ylabel("Cumulative return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "semicon_returns.png", dpi=150)
    plt.show()

# ── Per-date event study ──────────────────────────────────────────────────────

if IT_TICKER in returns.columns:
    es = EventStudy(returns)
    rows = []
    for label, dt in DATES.items():
        ev = es.run(IT_TICKER, dt)
        if ev:
            rows.append({"event": label, "date": dt.date(),
                         "CAR_%": ev.get("CAR"), "t": ev.get("t_stat"),
                         "p": ev.get("p_value"), "sig": ev.get("significant")})
    ev_table = pd.DataFrame(rows)
    print(ev_table.to_string(index=False))
    ev_table.to_csv(RESULTS_DIR / "datasets" / "semicon_event_study.csv", index=False)

# ── Granger: semiconductor risk → CNXIT ──────────────────────────────────────

if len(sc_events) > 0 and IT_TICKER in returns.columns:
    sc_geo = (
        -sc_events.groupby("date")["goldstein"].mean()
        .reindex(returns.index, method="ffill").fillna(0)
    )
    it_ret = returns[IT_TICKER].dropna()
    common = it_ret.index.intersection(sc_geo.index)
    if len(common) >= 48:
        gc = granger_causality_test(sc_geo.loc[common], it_ret.loc[common])
        if not gc.empty:
            print(gc.to_string(index=False))
    else:
        print(f"Insufficient obs ({len(common)}) for Granger — expand dataset.")

# ── Leontief: semiconductor −20% ─────────────────────────────────────────────

lp    = LeontiefPropagator.toy_5sector()
shock = lp.propagate_shock("Semiconductor", -0.20)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(shock.index, shock.values,
        color=["tomato" if v < 0 else "steelblue" for v in shock.values], alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Leontief: Semiconductor −20%  [export control scenario]  [ILLUSTRATIVE]")
ax.set_xlabel("Δ total output")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "semicon_leontief.png", dpi=150)
plt.show()
print(shock.round(4).to_string())

"""
experiments/case_studies/redsea.py
Red Sea / Houthi Crisis Case Study (Dec 2023 – Q1 2024)

Requires completed pipeline outputs in data/processed/ and data/interim/.
Run: python experiments/case_studies/redsea.py
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
from geosignal.propagation.leontief import LeontiefPropagator

INTERIM  = PROJECT_ROOT / "data" / "interim"
FIG_DIR  = RESULTS_DIR / "figures" / "redsea"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Parameters ────────────────────────────────────────────────────────────────

CRISIS_START = pd.Timestamp("2023-12-18")   # Maersk/MSC Cape rerouting
CRISIS_PEAK  = pd.Timestamp("2024-01-12")   # US/UK strikes on Houthi targets
WINDOW       = ("2023-11-01", "2024-03-31")

RS_KW = ["suez", "red sea", "houthi", "yemen", "shipping", "bab al-mandab"]

# ── Load ──────────────────────────────────────────────────────────────────────

events  = pd.read_parquet(DATA_PROC / "processed_events.parquet")
returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
events["date"] = pd.to_datetime(events["date"])
returns.index  = pd.to_datetime(returns.index)

mask = (
    events["actor1"].fillna("").str.lower().str.contains("|".join(RS_KW)) |
    events["actor2"].fillna("").str.lower().str.contains("|".join(RS_KW)) |
    events["action_country"].fillna("").str.lower().isin(["ye", "yem"])
)
rs_events  = events[mask & (events["date"] >= WINDOW[0]) & (events["date"] <= WINDOW[1])]
all_window = events[(events["date"] >= WINDOW[0]) & (events["date"] <= WINDOW[1])]
print(f"Red Sea events in window: {len(rs_events):,}")

# ── Event timeline ────────────────────────────────────────────────────────────

daily_cnt = rs_events.groupby("date").size().sort_index()
daily_gs  = all_window.groupby("date")["goldstein"].mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].bar(daily_cnt.index, daily_cnt.values, color="coral", alpha=0.8, width=1)
axes[0].axvline(CRISIS_START, color="red",     ls="--", lw=1.5, label="Rerouting starts")
axes[0].axvline(CRISIS_PEAK,  color="darkred", ls=":",  lw=1.5, label="US/UK strikes")
axes[0].set_title("Red Sea Crisis — GDELT Daily Event Count")
axes[0].set_ylabel("Events")
axes[0].legend()

roll7 = daily_gs.rolling(7, min_periods=1).mean()
axes[1].fill_between(daily_gs.index, daily_gs.values, alpha=0.2, color="steelblue")
axes[1].plot(roll7.index, roll7.values, color="steelblue", lw=2)
axes[1].axhline(0, color="black", ls=":", lw=0.8)
axes[1].axvline(CRISIS_START, color="red",     ls="--", lw=1.5)
axes[1].axvline(CRISIS_PEAK,  color="darkred", ls=":",  lw=1.5)
axes[1].set_title("Mean Goldstein — 7-day Rolling")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "redsea_timeline.png", dpi=150)
plt.show()

# ── NSE cumulative returns in crisis window ───────────────────────────────────

FOCUS = [NIFTY50_TICKER, "^CNXINFRA", "^CNXENERGY", "^CNXMETAL"]
avail = [t for t in FOCUS if t in returns.columns]
cum   = (1 + returns.loc[WINDOW[0]:WINDOW[1], avail].fillna(0)).cumprod() - 1

fig, ax = plt.subplots(figsize=(14, 6))
for col in cum.columns:
    ax.plot(cum.index, cum[col] * 100, label=col.replace("^", ""),
            lw=2.5 if col == NIFTY50_TICKER else 1.5)
ax.axvline(CRISIS_START, color="red",     ls="--", lw=1.5, label="Rerouting starts")
ax.axvline(CRISIS_PEAK,  color="darkred", ls=":",  lw=1.5, label="US/UK strikes")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_title("NSE Cumulative Returns — Red Sea Crisis Window")
ax.set_ylabel("Cumulative return (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.xticks(rotation=30, ha="right")
ax.legend(ncol=3)
plt.tight_layout()
plt.savefig(FIG_DIR / "redsea_returns.png", dpi=150)
plt.show()

# ── Event study: CNXINFRA @ CRISIS_PEAK ──────────────────────────────────────

IT = "^CNXINFRA"
if IT in returns.columns:
    es = EventStudy(returns)
    ev = es.run(IT, CRISIS_PEAK)
    if ev and "ar_series" in ev:
        ar  = ev["ar_series"] * 100
        car = ar.cumsum()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(range(len(ar)), ar.values,
                    color=["tomato" if v < 0 else "steelblue" for v in ar.values], alpha=0.85)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_title(f"AR(%)  CNXINFRA  @ {CRISIS_PEAK.date()}")
        axes[0].set_xlabel("Days after event")
        axes[0].set_ylabel("AR (%)")
        axes[1].plot(range(len(car)), car.values, color="steelblue", lw=2, marker="o", ms=3)
        axes[1].fill_between(range(len(car)), 0, car.values, alpha=0.2, color="steelblue")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title(f"CAR={ev['CAR']:.3f}%  p={ev['p_value']:.4f}")
        axes[1].set_xlabel("Days after event")
        plt.suptitle("Red Sea Crisis — CNXINFRA Event Study", y=1.01)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "redsea_event_study.png", dpi=150)
        plt.show()
        print(f"CAR={ev['CAR']}%  t={ev['t_stat']}  p={ev['p_value']}")

# ── Leontief: logistics −10% ──────────────────────────────────────────────────

lp    = LeontiefPropagator.toy_5sector()
shock = lp.propagate_shock("Logistics", -0.10)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(shock.index, shock.values,
        color=["tomato" if v < 0 else "steelblue" for v in shock.values], alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Leontief: Logistics −10%  [Red Sea rerouting scenario]  [ILLUSTRATIVE]")
ax.set_xlabel("Δ total output")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "redsea_leontief.png", dpi=150)
plt.show()
print(shock.round(4).to_string())

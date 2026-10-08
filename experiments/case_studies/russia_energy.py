"""
experiments/case_studies/russia_energy.py
CS-05 — Russia Energy Supply Disruption (Jan 2024)
==================================================
Stratum B: Conflict / shipping disruption

Real-world anchor
-----------------
Escalation of Black Sea maritime attacks on Russian crude tankers,
18 Jan 2024.  Ukrainian drone strikes on Russian oil infrastructure and
the resulting Black Sea freight premium forced Indian refiners — who source
~40 pct of crude from Russia — to absorb significantly higher transport
costs or seek spot-market alternatives.

India channel
-------------
Indian refiners face a direct freight cost increase on Urals crude.
Nifty Energy (^CNXENERGY) captures the refining-sector response.
Nifty FMCG is a secondary proxy for downstream fuel-cost pass-through.

Run
---
    python experiments/case_studies/russia_energy.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_PROC, RESULTS_DIR, NIFTY50_TICKER
from geosignal.analysis.event_study import EventStudy, compute_raw_impact
from geosignal.propagation.leontief import LeontiefPropagator
from experiments.case_studies.case_registry import get_case

sns.set_theme(style="whitegrid", palette="muted")

CASE     = get_case("CS-05")
FIG_DIR  = CASE.fig_dir
ANCHOR   = CASE.anchor
WINDOW   = ("2023-10-01", "2024-03-31")

SECONDARY_TICKERS = ["^CNXFMCG", NIFTY50_TICKER]

# ── Load ──────────────────────────────────────────────────────────────────────

events  = pd.read_parquet(DATA_PROC / "processed_events.parquet")
returns = pd.read_parquet(DATA_PROC / "nse_returns.parquet")
events["date"]  = pd.to_datetime(events["date"])
returns.index   = pd.to_datetime(returns.index)

# ── Filter events ─────────────────────────────────────────────────────────────

kw_mask = pd.Series(False, index=events.index)
for kw in CASE.keywords:
    kw_mask |= (
        events["actor1"].fillna("").str.lower().str.contains(kw, regex=False) |
        events["actor2"].fillna("").str.lower().str.contains(kw, regex=False) |
        events["source_url"].fillna("").str.lower().str.contains(kw, regex=False)
    )

cameo_mask   = events["event_root"].isin(CASE.cameo_roots)
country_mask = (
    events["action_country"].isin(CASE.action_countries)
    if CASE.action_countries else pd.Series(True, index=events.index)
)

case_events = events[
    (kw_mask | (cameo_mask & country_mask)) &
    (events["date"] >= WINDOW[0]) &
    (events["date"] <= WINDOW[1])
].copy()

all_window = events[
    (events["date"] >= WINDOW[0]) & (events["date"] <= WINDOW[1])
].copy()

print(f"\n{CASE.label}")
print(f"  GDELT events matched : {len(case_events):,}")
print(f"  Unique dates         : {case_events['date'].nunique()}")
print(f"  Anchor               : {ANCHOR.date()}")

# ── Figure 1: Event timeline ──────────────────────────────────────────────────

daily_cnt = case_events.groupby("date").size().sort_index()
daily_gs  = all_window.groupby("date")["goldstein"].mean()
roll7     = daily_gs.rolling(7, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].bar(daily_cnt.index, daily_cnt.values,
            color="#e67e22", alpha=0.75, width=1)
axes[0].axvline(ANCHOR, color="darkorange", ls="--", lw=2,
                label=f"Black Sea escalation ({ANCHOR.date()})")
axes[0].set_title(f"{CASE.label} — GDELT Daily Event Count")
axes[0].set_ylabel("Events")
axes[0].legend(fontsize=9)

axes[1].fill_between(daily_gs.index, daily_gs.values, alpha=0.2, color="steelblue")
axes[1].plot(roll7.index, roll7.values, color="steelblue", lw=2)
axes[1].axhline(0, color="black", ls=":", lw=0.8)
axes[1].axvline(ANCHOR, color="darkorange", ls="--", lw=1.5)
axes[1].set_title("Mean Goldstein Score — 7-day Rolling")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "cs05_timeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'cs05_timeline.png'}")

# ── Figure 2: NSE Energy vs Nifty50 ──────────────────────────────────────────

focus_tickers = [CASE.nse_ticker] + SECONDARY_TICKERS
avail = [t for t in focus_tickers if t in returns.columns]
wr    = returns.loc[WINDOW[0]:WINDOW[1], avail].fillna(0)
cum   = (1 + wr).cumprod() - 1

fig, ax = plt.subplots(figsize=(14, 6))
styles  = ["-", "--", "-.", ":"]
for i, col in enumerate(cum.columns):
    lw = 2.5 if col == NIFTY50_TICKER else 1.8
    ax.plot(cum.index, cum[col] * 100, label=col.replace("^", ""),
            lw=lw, ls=styles[i % len(styles)])

ax.axvline(ANCHOR, color="darkorange", ls="--", lw=2,
           label=f"Anchor ({ANCHOR.date()})")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_title(f"{CASE.label} — NSE Cumulative Returns")
ax.set_ylabel("Cumulative return (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.xticks(rotation=30, ha="right")
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "cs05_returns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'cs05_returns.png'}")

# ── Event study ───────────────────────────────────────────────────────────────

rows = []
if CASE.nse_ticker in returns.columns:
    es  = EventStudy(returns)
    ev  = es.run(CASE.nse_ticker, ANCHOR)
    raw = compute_raw_impact(returns, CASE.nse_ticker, ANCHOR)

    if ev and "ar_series" in ev:
        ar  = ev["ar_series"] * 100
        car = ar.cumsum()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(range(len(ar)), ar.values,
                    color=["#e74c3c" if v < 0 else "#2ecc71" for v in ar.values],
                    alpha=0.85)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_title(f"AR (%)  {CASE.nse_ticker}  anchor={ANCHOR.date()}")
        axes[0].set_xlabel("Days after event")
        axes[0].set_ylabel("AR (%)")

        axes[1].plot(range(len(car)), car.values,
                     color="steelblue", lw=2, marker="o", ms=3)
        axes[1].fill_between(range(len(car)), 0, car.values,
                             alpha=0.2, color="steelblue")
        axes[1].axhline(0, color="black", lw=0.8)
        sig_str = "SIGNIFICANT" if ev.get("significant") else "not significant"
        axes[1].set_title(
            f"CAR={ev['CAR']:.3f}%  t={ev['t_stat']:.3f}  "
            f"p={ev['p_value']:.4f}  [{sig_str}]"
        )
        axes[1].set_xlabel("Days after event")
        plt.suptitle(f"{CASE.label} — Event Study", y=1.01)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cs05_event_study.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  CAR={ev['CAR']}%  t={ev['t_stat']}  p={ev['p_value']}")

    rows.append({
        "case_id"    : CASE.case_id,
        "label"      : CASE.label,
        "stratum"    : CASE.stratum,
        "anchor_date": CASE.anchor_date,
        "nse_ticker" : CASE.nse_ticker,
        "CAR_pct"    : ev.get("CAR") if ev else None,
        "t_stat"     : ev.get("t_stat") if ev else None,
        "p_value"    : ev.get("p_value") if ev else None,
        "significant": ev.get("significant") if ev else None,
        **({k: v for k, v in raw.items()} if raw else {}),
    })

pd.DataFrame(rows).to_csv(
    CASE.datasets_dir / f"{CASE.case_id.lower().replace('-','')}_event_study.csv",
    index=False,
)

# ── Leontief ──────────────────────────────────────────────────────────────────

_config        = PROJECT_ROOT / "config"
_io_candidates = [
    _config / "india_io_wiod_2014.csv",
    _config / "india_io_synthetic_2023.csv",
]
_io_path = next((p for p in _io_candidates if p.exists()), None)

if _io_path:
    lp       = LeontiefPropagator.from_csv(_io_path)
    io_label = _io_path.name
else:
    lp       = LeontiefPropagator.toy_5sector()
    io_label = "toy_5sector_illustrative"

matched = next(
    (s for s in lp.sector_labels
     if CASE.shocked_sector.lower() in s.lower()
     or s.lower() in CASE.shocked_sector.lower()),
    lp.sector_labels[0],
)
shock = lp.propagate_shock(matched, CASE.shock_magnitude)

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(shock.index, shock.values,
        color=["#e74c3c" if v < 0 else "#2ecc71" for v in shock.values],
        alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_title(
    f"{CASE.label} — Leontief: {matched} "
    f"{CASE.shock_magnitude*100:.0f}%\n[{io_label}]"
)
ax.set_xlabel("Δ total output")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "cs05_leontief.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(shock.round(4).to_string())
print(f"\n  All figures saved to {FIG_DIR}")

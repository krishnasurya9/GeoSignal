"""
notebooks/02_event_processing.py
Phase I→II — Harmonized events: Goldstein distribution, country × CAMEO heatmap, tone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config.settings import DATA_PROC, RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Load ──────────────────────────────────────────────────────────────────────

events = pd.read_parquet(DATA_PROC / "processed_events.parquet")
events["date"] = pd.to_datetime(events["date"])

print(f"Events: {events.shape}")
print(events.dtypes)
print()
print(events.describe(include="all").T.to_string())

# ── Goldstein distribution + rolling time series ──────────────────────────────

gs    = events["goldstein"].dropna()
ts_gs = events.groupby("date")["goldstein"].mean().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(gs, bins=60, color="steelblue", alpha=0.85, edgecolor="white")
axes[0].axvline(gs.mean(), color="red", linestyle="--", label=f"μ = {gs.mean():.2f}")
axes[0].axvline(0, color="black", linestyle=":", lw=1)
axes[0].set_title("Goldstein Score Distribution")
axes[0].set_xlabel("Score  (−10 conflictual → +10 cooperative)")
axes[0].set_ylabel("Events")
axes[0].legend()

roll7 = ts_gs.rolling(7, min_periods=1).mean()
axes[1].fill_between(ts_gs.index, ts_gs.values, alpha=0.25, color="steelblue")
axes[1].plot(roll7.index, roll7.values, color="steelblue", lw=2, label="7-day MA")
axes[1].axhline(0, color="black", lw=0.8, linestyle=":")
axes[1].set_title("Daily Mean Goldstein — 7-day Rolling")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "goldstein.png", dpi=150)
plt.show()

# ── Country × CAMEO root heatmap ──────────────────────────────────────────────

if "event_root" in events.columns and "action_country" in events.columns:
    top_r = events["event_root"].value_counts().head(10).index
    top_c = events["action_country"].value_counts().head(12).index
    pivot = (
        events[events["event_root"].isin(top_r) & events["action_country"].isin(top_c)]
        .groupby(["action_country", "event_root"]).size().unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.4, cbar_kws={"label": "Count"})
    ax.set_title("Events: Top Countries × CAMEO Root Codes")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "country_root_heatmap.png", dpi=150)
    plt.show()

# ── Avg tone ──────────────────────────────────────────────────────────────────

if "avg_tone" in events.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tone = events["avg_tone"].dropna()
    axes[0].hist(tone.clip(-20, 20), bins=60, color="coral", alpha=0.85, edgecolor="white")
    axes[0].axvline(tone.mean(), color="darkred", linestyle="--", label=f"μ = {tone.mean():.2f}")
    axes[0].axvline(0, color="black", lw=1, linestyle=":")
    axes[0].set_title("Avg Tone Distribution (±20 clip)")
    axes[0].legend()

    sample = events[["goldstein", "avg_tone"]].dropna().sample(min(5000, len(events)), random_state=42)
    r = sample["goldstein"].corr(sample["avg_tone"])
    axes[1].scatter(sample["goldstein"], sample["avg_tone"], alpha=0.12, s=5, color="steelblue")
    axes[1].set_title(f"Goldstein vs Avg Tone  (r = {r:.3f})")
    axes[1].set_xlabel("Goldstein")
    axes[1].set_ylabel("Avg Tone")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "tone_goldstein.png", dpi=150)
    plt.show()

# ── India-focused subset ──────────────────────────────────────────────────────

india = events[
    events["action_country"].fillna("").str.contains("IN", case=True) |
    events["actor1"].fillna("").str.contains("IND", case=True) |
    events["actor2"].fillna("").str.contains("IND", case=True)
]
print(f"\nIndia-linked: {len(india):,} / {len(events):,} ({100*len(india)/len(events):.1f}%)")
print(india["event_root"].value_counts().head(15).to_string())

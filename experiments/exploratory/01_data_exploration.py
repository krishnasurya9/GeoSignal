"""
notebooks/01_data_exploration.py
Phase I — Raw data inspection: GDELT events + GKG news corpus
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config.settings import DATA_RAW, DATA_PROC, RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_columns", 60)

# ── Load ──────────────────────────────────────────────────────────────────────

df_raw  = pd.read_parquet(DATA_RAW  / "gdelt_raw.parquet")
df_news = pd.read_parquet(DATA_PROC / "news_corpus.parquet")

print(f"GDELT raw:   {df_raw.shape[0]:>8,} rows  {df_raw.shape[1]} cols")
print(f"News corpus: {df_news.shape[0]:>8,} rows  {df_news.shape[1]} cols")
print()
print(df_raw.dtypes)
print()

# Null audit
null_pct = (df_raw.isnull().sum() / len(df_raw) * 100).round(1).sort_values(ascending=False)
print("Null % per column (top 20):")
print(null_pct.head(20).to_string())

# ── Daily volume ──────────────────────────────────────────────────────────────

date_col = "SQLDATE" if "SQLDATE" in df_raw.columns else df_raw.columns[1]
df_raw["_date"] = pd.to_datetime(df_raw[date_col].astype(str), format="%Y%m%d", errors="coerce")
daily = df_raw.groupby("_date").size()

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

axes[0].bar(daily.index, daily.values, width=0.9, color="steelblue", alpha=0.8)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[0].set_title("GDELT Daily Event Volume")
axes[0].set_ylabel("Events / day")

if "date" in df_news.columns:
    news_daily = df_news.groupby("date").size()
    axes[1].bar(news_daily.index, news_daily.values, width=0.9, color="teal", alpha=0.8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].set_title("GKG Daily Article Volume")
    axes[1].set_ylabel("Articles / day")

plt.tight_layout()
plt.savefig(FIG_DIR / "daily_volume.png", dpi=150)
plt.show()

# ── CAMEO root code distribution ──────────────────────────────────────────────

CAMEO_LABELS = {
    "01": "Public statement", "02": "Appeal", "03": "Intent cooperate",
    "04": "Consult", "05": "Diplomatic coop", "06": "Material coop",
    "07": "Provide aid", "08": "Yield", "09": "Investigate", "10": "Demand",
    "11": "Disapprove", "12": "Reject", "13": "Threaten", "14": "Protest",
    "15": "Force posture", "16": "Reduce relations", "17": "Coerce",
    "18": "Assault", "19": "Fight", "20": "Mass violence",
}

root_col   = "EventRootCode" if "EventRootCode" in df_raw.columns else df_raw.columns[27]
top_roots  = df_raw[root_col].value_counts().head(15)
labels     = [f"{c} – {CAMEO_LABELS.get(str(c), str(c))}" for c in top_roots.index]

fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(labels, top_roots.values, color="steelblue", alpha=0.85)
ax.invert_yaxis()
ax.set_title("Top 15 CAMEO Root Codes — 2024 Q1")
ax.set_xlabel("Event count")
plt.tight_layout()
plt.savefig(FIG_DIR / "cameo_roots.png", dpi=150)
plt.show()

# ── Action country distribution ───────────────────────────────────────────────

country_col = next((c for c in df_raw.columns if "CountryCode" in c), None)
if country_col:
    top_countries = df_raw[country_col].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 4))
    top_countries.plot(kind="bar", ax=ax, color="coral", alpha=0.85)
    ax.set_title("Top 20 Action Countries")
    ax.set_xlabel("Country code")
    ax.set_ylabel("Event count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "action_countries.png", dpi=150)
    plt.show()

# ── News text length ──────────────────────────────────────────────────────────

if "text" in df_news.columns:
    lens = df_news["text"].fillna("").str.len()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lens.clip(0, 2000), bins=60, color="mediumpurple", alpha=0.85, edgecolor="white")
    ax.axvline(lens.median(), color="red", linestyle="--", label=f"Median {lens.median():.0f} chars")
    ax.set_title("GKG Proxy Text Length")
    ax.set_xlabel("Characters (clipped at 2000)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "text_length.png", dpi=150)
    plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────

summary = pd.DataFrame({
    "dataset":      ["GDELT raw", "News corpus"],
    "rows":         [len(df_raw), len(df_news)],
    "date_min":     [df_raw["_date"].min().date(),
                     df_news["date"].min() if "date" in df_news.columns else None],
    "date_max":     [df_raw["_date"].max().date(),
                     df_news["date"].max() if "date" in df_news.columns else None],
    "null_pct_avg": [round(df_raw.isnull().mean().mean(), 4),
                     round(df_news.isnull().mean().mean(), 4)],
})
print(summary.to_string(index=False))

"""
notebooks/03_nlp_alignment.py
Phase II — BM25 / TF-IDF cosine scores, NER label distribution, event-article match rate
"""

import sys, json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import DATA_PROC, RESULTS_DIR, MIN_COSINE_SIM

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Load ──────────────────────────────────────────────────────────────────────

pairs  = pd.read_parquet(DATA_PROC / "event_news_pairs.parquet")
events = pd.read_parquet(DATA_PROC / "processed_events.parquet")

print(f"Pairs:  {pairs.shape}")
print(f"Events: {len(events):,}")
print(pairs.dtypes)

# ── Score distributions ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bm25 = pairs["bm25_score"].dropna()
axes[0].hist(bm25, bins=60, color="steelblue", alpha=0.85, edgecolor="white")
axes[0].set_title("BM25 Score Distribution")
axes[0].set_xlabel("BM25 score")
axes[0].set_ylabel("Pair count")
axes[0].text(0.65, 0.85, f"n={len(bm25):,}\nμ={bm25.mean():.3f}",
             transform=axes[0].transAxes)

cos = pairs["cosine_sim"].dropna()
axes[1].hist(cos, bins=60, color="teal", alpha=0.85, edgecolor="white")
axes[1].axvline(MIN_COSINE_SIM, color="red", linestyle="--",
                label=f"threshold={MIN_COSINE_SIM}")
n_above = (cos >= MIN_COSINE_SIM).sum()
axes[1].text(0.45, 0.85,
             f"≥ threshold: {n_above:,} ({100*n_above/len(cos):.1f}%)",
             transform=axes[1].transAxes)
axes[1].set_title("TF-IDF Cosine Similarity")
axes[1].set_xlabel("Cosine sim")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "nlp_scores.png", dpi=150)
plt.show()

# ── BM25 vs cosine scatter ────────────────────────────────────────────────────

sample = pairs[["bm25_score", "cosine_sim"]].dropna().sample(min(5000, len(pairs)), random_state=42)
r = sample["bm25_score"].corr(sample["cosine_sim"])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(sample["bm25_score"], sample["cosine_sim"], alpha=0.15, s=6, c="steelblue")
ax.axhline(MIN_COSINE_SIM, color="red", linestyle="--", lw=1,
           label=f"cos threshold={MIN_COSINE_SIM}")
ax.set_xlabel("BM25 score")
ax.set_ylabel("Cosine similarity")
ax.set_title(f"BM25 vs Cosine  (r={r:.3f}, n=5 000)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "bm25_vs_cosine.png", dpi=150)
plt.show()

# ── Articles per event + match rate ──────────────────────────────────────────

ape = pairs.groupby("event_id").size()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(ape.clip(0, 15), bins=15, color="coral", alpha=0.85, edgecolor="white")
axes[0].set_title("Articles per Event (clipped @15)")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("Events")

total   = len(events)
matched = ape.index.nunique()
axes[1].pie([matched, total - matched],
            labels=[f"Matched\n{matched:,}", f"No articles\n{total-matched:,}"],
            colors=["steelblue", "lightgrey"], autopct="%1.1f%%", startangle=90)
axes[1].set_title("Event–Article Match Rate")

plt.tight_layout()
plt.savefig(FIG_DIR / "event_match.png", dpi=150)
plt.show()

# ── NER label + top entity strings ───────────────────────────────────────────

labels_all, ents_all = [], []
for raw in pairs["entities"].dropna():
    try:
        for e in json.loads(raw):
            labels_all.append(e.get("label", "UNK"))
            ents_all.append(e.get("text", ""))
    except Exception:
        pass

lc = Counter(labels_all)
sorted_lc = sorted(lc.items(), key=lambda x: x[1], reverse=True)
top_ents  = Counter(ents_all).most_common(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].barh([x[0] for x in sorted_lc], [x[1] for x in sorted_lc],
             color="mediumpurple", alpha=0.85)
axes[0].invert_yaxis()
axes[0].set_title("NER Label Distribution")
axes[0].set_xlabel("Count")

axes[1].barh([x[0] for x in top_ents[::-1]], [x[1] for x in top_ents[::-1]],
             color="coral", alpha=0.85)
axes[1].set_title("Top 20 Extracted Entity Strings")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig(FIG_DIR / "ner_entities.png", dpi=150)
plt.show()

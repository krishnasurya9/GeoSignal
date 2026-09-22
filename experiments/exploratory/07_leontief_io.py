"""
notebooks/07_leontief_io.py
Phase V — Leontief I-O shock propagation, inverse heatmap, network graph, sensitivity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from config.settings import RESULTS_DIR
from src.impact_analysis import LeontiefPropagator

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Toy A matrix (illustrative — replace with MoSPI 2017-18 / WIOD) ──────────

LABELS = ["Energy", "Semiconductor", "Automotive", "Logistics", "ConsumerGoods"]
A = np.array([
    [0.10, 0.05, 0.08, 0.15, 0.04],
    [0.02, 0.12, 0.20, 0.01, 0.05],
    [0.03, 0.08, 0.10, 0.10, 0.12],
    [0.15, 0.04, 0.07, 0.05, 0.08],
    [0.01, 0.02, 0.03, 0.04, 0.10],
])

lp = LeontiefPropagator(A, LABELS)
L_df = pd.DataFrame(lp.L.round(4), index=LABELS, columns=LABELS)
print("Leontief inverse L = (I-A)^{-1}:")
print(L_df.to_string())

# ── Multi-shock comparison ────────────────────────────────────────────────────

SHOCKS = {
    "Energy −15%":        ("Energy",        -0.15),
    "Semiconductor −20%": ("Semiconductor", -0.20),
    "Logistics −10%":     ("Logistics",     -0.10),
}

df_shocks = pd.DataFrame(
    {label: lp.propagate_shock(sector, mag) for label, (sector, mag) in SHOCKS.items()}
)
print(df_shocks.round(4).to_string())

fig, axes = plt.subplots(1, len(SHOCKS), figsize=(5 * len(SHOCKS), 5), sharey=True)
for ax, (label, series) in zip(axes, df_shocks.items()):
    colors = ["tomato" if v < 0 else "steelblue" for v in series.values]
    ax.barh(series.index, series.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(label)
    ax.set_xlabel("Δ output")
    ax.invert_yaxis()
plt.suptitle("Leontief I-O Shock Propagation  [ILLUSTRATIVE — toy matrix]", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "leontief_shocks.png", dpi=150)
plt.show()

# ── Leontief inverse heatmap ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(L_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
            linewidths=0.4, cbar_kws={"label": "Total output multiplier"})
ax.set_title("Leontief Inverse  L = (I−A)⁻¹")
ax.set_xlabel("Sector receiving shock")
ax.set_ylabel("Sector absorbing impact")
plt.tight_layout()
plt.savefig(FIG_DIR / "leontief_inverse.png", dpi=150)
plt.show()

# ── Network graph ─────────────────────────────────────────────────────────────

THRESHOLD = 0.05
G = nx.DiGraph()
G.add_nodes_from(LABELS)
for i, src in enumerate(LABELS):
    for j, tgt in enumerate(LABELS):
        if i != j and A[i, j] > THRESHOLD:
            G.add_edge(src, tgt, weight=A[i, j])

pos    = nx.circular_layout(G)
widths = [G[u][v]["weight"] * 15 for u, v in G.edges()]

fig, ax = plt.subplots(figsize=(9, 9))
nx.draw_networkx_nodes(G, pos, node_size=1800, node_color="steelblue", alpha=0.85, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", ax=ax)
nx.draw_networkx_edges(G, pos, width=widths, edge_color="grey", alpha=0.7,
                       arrows=True, arrowstyle="->", arrowsize=20, ax=ax)
ax.set_title(f"Inter-sector I-O Linkages  (A > {THRESHOLD})")
ax.axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "io_network.png", dpi=150)
plt.show()

# ── Sensitivity analysis ──────────────────────────────────────────────────────

magnitudes  = np.linspace(-0.30, -0.05, 20)
sensitivity = pd.DataFrame(
    {f"{m*100:.0f}%": lp.propagate_shock("Semiconductor", m) for m in magnitudes}
).T

fig, ax = plt.subplots(figsize=(12, 5))
for col in sensitivity.columns:
    ax.plot(sensitivity.index, sensitivity[col], marker="o", ms=3, label=col)
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_title("Sensitivity: Semiconductor Shock Magnitude → Sector Output")
ax.set_xlabel("Shock magnitude")
ax.set_ylabel("Δ total output")
ax.legend(ncol=3, fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "leontief_sensitivity.png", dpi=150)
plt.show()

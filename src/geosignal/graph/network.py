"""
geosignal.graph.network
========================
NetworkX geopolitical influence graph layer.

Resolves the "NetworkX graph layer | Planned" item from README §14
and DEC-008 (no GNN at this stage — graph representation only).

Graph schema
------------
Nodes
~~~~~
  type="actor"   — country/entity codes from GDELT actor fields
  type="sector"  — GICS sector labels (from sector mapping)

  Attributes on every node:
    node_type, event_count (actors) / n_records (sectors),
    pagerank, betweenness, in_degree, out_degree

Edges
~~~~~
  actor  → actor   (type="actor_cooccurrence")
    weight   = number of shared GDELT event records
    goldstein_mean = mean Goldstein intensity of co-occurring events

  actor  → sector  (type="actor_sector_impact")
    weight      = mean |CAR_pct| from event study (0 if no event study)
    n_events    = number of event records linking actor to sector
    sig_fraction = fraction of event-study records with p < 0.05

  sector → sector  (type="sector_io_linkage")  [optional, from I-O matrix]
    weight = A[i,j] technical coefficient from Leontief matrix

Outputs
-------
  results/datasets/network_edges.csv       — edge list (source, target, attrs)
  results/datasets/network_nodes.csv       — node list (node, attrs)
  results/figures/network_graph.png        — spring-layout visualisation
  results/figures/network_graph_sectors.png — sector-only sub-graph

Usage
-----
    from geosignal.graph.network import GeoSignalGraph

    g = GeoSignalGraph()
    g.build(events_df, temporal_df, pairs_df)
    g.add_io_edges(io_matrix_path)      # optional Leontief layer
    g.compute_centrality()
    g.export(results_dir)
    g.visualise(figures_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    logger.error("networkx not installed. Run: pip install networkx")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# GICS sector labels used in GeoSignal (must match sector_mapping.py)
SECTOR_LABELS = {
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Utilities", "Real Estate",
    # NSE ticker-mapped names
    "Auto", "Bank", "FMCG", "IT", "Metal",
}

# Visual style
NODE_COLOURS = {
    "actor" : "#3498db",   # blue
    "sector": "#e74c3c",   # red
    "io"    : "#2ecc71",   # green (sector–sector I-O edges)
}
EDGE_ALPHA   = 0.45
TOP_N_NODES  = 60          # cap for visualisation readability


class GeoSignalGraph:
    """
    Builds and analyses the GeoSignal geopolitical influence graph.

    Attributes:
        G:  networkx.DiGraph — the live graph object (public for notebook use)
    """

    def __init__(self) -> None:
        if not _HAS_NX:
            raise ImportError("networkx is required. Run: pip install networkx")
        self.G: nx.DiGraph = nx.DiGraph()
        self._built = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        events_df: pd.DataFrame,
        temporal_df: Optional[pd.DataFrame] = None,
        pairs_df: Optional[pd.DataFrame] = None,
    ) -> "GeoSignalGraph":
        """
        Populate the graph from pipeline outputs.

        Args:
            events_df:   processed_events.parquet (canonical schema)
            temporal_df: temporal_results.parquet  (event study results)
            pairs_df:    event_news_sector.parquet (sector-enriched pairs)

        Returns:
            self  (for method chaining)
        """
        logger.info("Building GeoSignal network graph …")
        self.G.clear()

        self._add_actor_actor_edges(events_df)

        if temporal_df is not None and pairs_df is not None:
            self._add_actor_sector_edges(events_df, temporal_df, pairs_df)
        elif pairs_df is not None:
            self._add_actor_sector_from_pairs(events_df, pairs_df)

        self._tag_node_types()
        self._built = True
        logger.success(
            f"Graph built: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )
        return self

    def _add_actor_actor_edges(self, events_df: pd.DataFrame) -> None:
        """Actor → Actor edges weighted by co-occurrence count."""
        needed = {"actor1", "actor2", "goldstein"}
        if not needed.issubset(events_df.columns):
            logger.warning("events_df missing actor1/actor2/goldstein — skipping actor edges.")
            return

        actor_pairs = (
            events_df[["actor1", "actor2", "goldstein"]]
            .dropna(subset=["actor1", "actor2"])
            .query("actor1 != '' and actor2 != ''")
        )

        for (a1, a2), grp in actor_pairs.groupby(["actor1", "actor2"]):
            a1, a2 = str(a1).strip(), str(a2).strip()
            if not a1 or not a2 or a1 == a2:
                continue
            weight          = len(grp)
            goldstein_mean  = float(grp["goldstein"].mean())
            goldstein_abs   = float(grp["goldstein"].abs().mean())

            if self.G.has_edge(a1, a2):
                self.G[a1][a2]["weight"]         += weight
                self.G[a1][a2]["goldstein_mean"]  = (
                    (self.G[a1][a2]["goldstein_mean"] + goldstein_mean) / 2
                )
            else:
                self.G.add_edge(
                    a1, a2,
                    weight          = weight,
                    goldstein_mean  = round(goldstein_mean, 4),
                    goldstein_abs   = round(goldstein_abs, 4),
                    edge_type       = "actor_cooccurrence",
                )

    def _add_actor_sector_edges(
        self,
        events_df: pd.DataFrame,
        temporal_df: pd.DataFrame,
        pairs_df: pd.DataFrame,
    ) -> None:
        """Actor → Sector edges weighted by mean |CAR_pct| from event study."""
        ev_sub = events_df[["event_id", "actor1", "actor2"]].copy()

        merged = temporal_df.merge(ev_sub, on="event_id", how="left")
        if "sector" not in merged.columns and "sector" in pairs_df.columns:
            sector_map = (
                pairs_df.groupby("event_id")["sector"]
                .agg(lambda x: x.mode().iloc[0] if len(x) else "Unknown")
            )
            merged["sector"] = merged["event_id"].map(sector_map)

        if "CAR_pct" not in merged.columns:
            logger.warning("temporal_df has no CAR_pct — using event counts only.")
            merged["CAR_pct"] = np.nan

        for (actor, sector), grp in merged.dropna(
                subset=["actor1", "sector"]).groupby(["actor1", "sector"]):
            actor, sector = str(actor).strip(), str(sector).strip()
            if not actor or not sector or sector == "Unknown":
                continue

            mean_car   = float(grp["CAR_pct"].abs().mean()) if grp["CAR_pct"].notna().any() else 0.0
            sig_frac   = float(grp["significant"].mean()) if "significant" in grp.columns else 0.0
            n_events   = len(grp)

            self.G.add_edge(
                actor, sector,
                weight       = round(mean_car, 4),
                n_events     = n_events,
                sig_fraction = round(sig_frac, 3),
                edge_type    = "actor_sector_impact",
            )

    def _add_actor_sector_from_pairs(
        self,
        events_df: pd.DataFrame,
        pairs_df: pd.DataFrame,
    ) -> None:
        """
        Simpler actor → sector edges (no event study), weighted by
        co-occurrence count. Used when temporal results are unavailable.
        """
        ev_sub = events_df[["event_id", "actor1"]].copy()
        merged = pairs_df.merge(ev_sub, on="event_id", how="left")

        for (actor, sector), grp in merged.dropna(
                subset=["actor1", "sector"]).groupby(["actor1", "sector"]):
            actor, sector = str(actor).strip(), str(sector).strip()
            if not actor or not sector or sector == "Unknown":
                continue
            self.G.add_edge(
                actor, sector,
                weight    = len(grp),
                n_events  = len(grp),
                edge_type = "actor_sector_impact",
            )

    def _tag_node_types(self) -> None:
        for node in self.G.nodes():
            is_sector = node in SECTOR_LABELS
            self.G.nodes[node]["node_type"]  = "sector" if is_sector else "actor"
            self.G.nodes[node]["in_degree"]  = self.G.in_degree(node)
            self.G.nodes[node]["out_degree"] = self.G.out_degree(node)

    # ── I-O layer ─────────────────────────────────────────────────────────────

    def add_io_edges(self, io_matrix_path: Path, threshold: float = 0.03) -> "GeoSignalGraph":
        """
        Add sector → sector edges from the Leontief I-O coefficient matrix.
        Only adds edges where A[i,j] >= threshold (avoids cluttering graph).

        Args:
            io_matrix_path: CSV produced by io_loader.build_india_io_matrix().
            threshold:       Minimum coefficient to add an edge.
        """
        if not io_matrix_path.exists():
            logger.warning(f"I-O matrix not found: {io_matrix_path} — skipping I-O edges.")
            return self

        a = pd.read_csv(io_matrix_path, index_col=0)
        n_added = 0
        for src in a.index:
            for tgt in a.columns:
                coeff = float(a.loc[src, tgt])
                if coeff >= threshold:
                    self.G.add_edge(
                        str(src), str(tgt),
                        weight    = round(coeff, 4),
                        edge_type = "sector_io_linkage",
                    )
                    n_added += 1
        logger.info(f"I-O layer: added {n_added} sector→sector edges (threshold={threshold})")
        self._tag_node_types()
        return self

    # ── Centrality ────────────────────────────────────────────────────────────

    def compute_centrality(self) -> "GeoSignalGraph":
        """
        Compute PageRank, betweenness, and eigenvector centrality.
        Results stored as node attributes.
        """
        if self.G.number_of_nodes() == 0:
            return self

        try:
            pagerank = nx.pagerank(self.G, weight="weight", max_iter=200)
            for node, val in pagerank.items():
                self.G.nodes[node]["pagerank"] = round(val, 8)
            logger.info("PageRank computed.")
        except Exception as exc:
            logger.warning(f"PageRank failed: {exc}")

        try:
            betweenness = nx.betweenness_centrality(
                self.G, weight="weight", normalized=True
            )
            for node, val in betweenness.items():
                self.G.nodes[node]["betweenness"] = round(val, 8)
            logger.info("Betweenness centrality computed.")
        except Exception as exc:
            logger.warning(f"Betweenness failed: {exc}")

        try:
            # Eigenvector centrality on undirected view for convergence
            ev_cent = nx.eigenvector_centrality_numpy(
                self.G.to_undirected(), weight="weight"
            )
            for node, val in ev_cent.items():
                self.G.nodes[node]["eigenvector"] = round(float(val), 8)
            logger.info("Eigenvector centrality computed.")
        except Exception as exc:
            logger.warning(f"Eigenvector centrality failed: {exc}")

        return self

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, output_dir: Path) -> dict[str, Path]:
        """
        Write edge list and node list CSVs.

        Returns:
            {'edges': Path, 'nodes': Path}
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Edge list
        edge_rows = []
        for u, v, data in self.G.edges(data=True):
            edge_rows.append({"source": u, "target": v, **data})
        edges_df = pd.DataFrame(edge_rows)
        edges_path = output_dir / "network_edges.csv"
        edges_df.to_csv(edges_path, index=False)

        # Node list
        node_rows = []
        for node, data in self.G.nodes(data=True):
            node_rows.append({"node": node, **data})
        nodes_df = pd.DataFrame(node_rows)
        nodes_path = output_dir / "network_nodes.csv"
        nodes_df.to_csv(nodes_path, index=False)

        logger.success(
            f"Graph exported: {len(edges_df)} edges → {edges_path.name}  |  "
            f"{len(nodes_df)} nodes → {nodes_path.name}"
        )
        return {"edges": edges_path, "nodes": nodes_path}

    # ── Visualise ─────────────────────────────────────────────────────────────

    def visualise(
        self,
        figures_dir: Path,
        top_n: int = TOP_N_NODES,
        seed: int = 42,
    ) -> list[Path]:
        """
        Render two figures:
          1. Full graph (top N nodes by degree)
          2. Sector-only sub-graph

        Returns list of figure paths.
        """
        if not _HAS_MPL:
            logger.warning("matplotlib not available — skipping visualisation.")
            return []

        figures_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        # ── Figure 1: full graph (top N by degree) ────────────────────────────
        by_degree = sorted(
            self.G.nodes(), key=lambda n: self.G.degree(n), reverse=True
        )
        sub_nodes = by_degree[:top_n]
        sub = self.G.subgraph(sub_nodes).copy()

        path = self._draw_graph(
            sub,
            title=(
                f"GeoSignal — Geopolitical Influence Network\n"
                f"(Top {top_n} nodes by degree  ·  node size ∝ PageRank)"
            ),
            out_path=figures_dir / "network_graph.png",
            seed=seed,
        )
        output_paths.append(path)

        # ── Figure 2: sector-only sub-graph ──────────────────────────────────
        sector_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "sector"
        ]
        if len(sector_nodes) >= 2:
            sector_sub = self.G.subgraph(sector_nodes).copy()
            if sector_sub.number_of_edges() > 0:
                path2 = self._draw_graph(
                    sector_sub,
                    title="GeoSignal — Sector I-O Linkage Sub-graph",
                    out_path=figures_dir / "network_graph_sectors.png",
                    seed=seed,
                    edge_colour="#2ecc71",
                )
                output_paths.append(path2)

        return output_paths

    def _draw_graph(
        self,
        G: "nx.DiGraph",
        title: str,
        out_path: Path,
        seed: int = 42,
        edge_colour: str = "#555555",
    ) -> Path:
        """Render and save one graph figure."""
        fig, ax = plt.subplots(figsize=(16, 11))

        pos = nx.spring_layout(G, seed=seed, k=2.8)

        node_colours = [
            NODE_COLOURS.get(G.nodes[n].get("node_type", "actor"), "#95a5a6")
            for n in G.nodes()
        ]
        pageranks = [G.nodes[n].get("pagerank", 0) for n in G.nodes()]
        max_pr    = max(pageranks) if max(pageranks) > 0 else 1e-9
        node_sizes = [
            400 + 2000 * (pr / max_pr) for pr in pageranks
        ]

        raw_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_w = max(raw_weights) if raw_weights else 1
        edge_widths = [0.4 + 3.0 * (w / max_w) for w in raw_weights]

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colours,
            node_size=node_sizes,
            alpha=0.88,
            ax=ax,
        )
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=EDGE_ALPHA,
            edge_color=edge_colour,
            arrows=True,
            arrowsize=14,
            connectionstyle="arc3,rad=0.08",
            ax=ax,
        )
        # Labels only for top-degree nodes to avoid clutter
        top_label_nodes = sorted(
            G.nodes(), key=lambda n: G.degree(n), reverse=True
        )[:min(25, G.number_of_nodes())]
        labels = {n: n for n in top_label_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

        legend_handles = [
            mpatches.Patch(facecolor=NODE_COLOURS["actor"],  label="Actor (country/entity)"),
            mpatches.Patch(facecolor=NODE_COLOURS["sector"], label="Sector (GICS)"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
        ax.set_title(title, fontsize=11, pad=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Graph figure → {out_path}")
        return out_path

    # ── Summary stats ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return key graph statistics as a dict for analysis summaries."""
        if not _HAS_NX or self.G.number_of_nodes() == 0:
            return {}

        actor_nodes  = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "actor"]
        sector_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "sector"]
        edge_types   = {}
        for _, _, d in self.G.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        top_actors = sorted(
            actor_nodes,
            key=lambda n: self.G.nodes[n].get("pagerank", 0),
            reverse=True,
        )[:10]

        return {
            "n_nodes"       : self.G.number_of_nodes(),
            "n_edges"       : self.G.number_of_edges(),
            "n_actor_nodes" : len(actor_nodes),
            "n_sector_nodes": len(sector_nodes),
            "edge_type_counts": edge_types,
            "density"       : round(nx.density(self.G), 6),
            "top_actors_by_pagerank": top_actors,
            "is_weakly_connected": nx.is_weakly_connected(self.G)
                                    if self.G.number_of_nodes() > 0 else False,
        }

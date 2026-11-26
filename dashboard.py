"""
dashboard.py
============
GeoSignal — Streamlit output browser.

Lets you browse every pipeline output without opening parquet files manually.

Run:
    pip install streamlit
    streamlit run dashboard.py

Then open  http://localhost:8501  in your browser.

Pages
-----
  Overview         Run metadata, validation label counts, quick stats
  Event Study      CAR table + interactive bar chart per sector
  Granger          Granger results table + significance summary
  Leontief I-O     Shock propagation table + heatmap
  Network Graph    Pre-rendered network figure + edge list
  Raw Data         Browse any parquet/CSV file in the results tree
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths (same as run_all.py) ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
INTERIM      = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR  = PROJECT_ROOT / "results"
DATASETS_DIR = RESULTS_DIR  / "datasets"
FIGURES_DIR  = RESULTS_DIR  / "figures"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoSignal Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🌍 GeoSignal")
st.sidebar.caption("Geopolitical Event → Economic Impact")

PAGES = [
    "Overview",
    "Event Study",
    "Granger Causality",
    "Leontief I-O",
    "Network Graph",
    "Raw Data Browser",
]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Run pipeline:**\n```\npython run_all.py --skip-download\n```"
)

# ── Helper: cached loaders ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _missing(label: str) -> None:
    st.warning(
        f"**{label}** not found. "
        "Run `python run_all.py --skip-download` to generate it.",
        icon="⚠️",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("🌍 GeoSignal — Pipeline Overview")
    st.caption(
        "Geopolitical Event → Economic Impact Analysis | India NSE Sectoral Markets"
    )

    summary = _load_json(RESULTS_DIR / "run_summary.json")

    if summary:
        col1, col2, col3, col4 = st.columns(4)
        es  = summary.get("sections", {}).get("event_study", {})
        gc  = summary.get("sections", {}).get("granger", {})
        net = summary.get("sections", {}).get("network", {})
        val = summary.get("sections", {}).get("validation_dataset", {})

        col1.metric("Event–sector records",
                    f"{es.get('n_records', '—'):,}" if es.get("n_records") else "—")
        col2.metric("Significant CARs (p<0.05)",
                    f"{es.get('n_significant', '—'):,}" if es.get("n_significant") is not None else "—")
        col3.metric("Granger-predictive sectors",
                    f"{gc.get('n_significant', '—')} / {gc.get('n_sectors_tested', '—')}")
        col4.metric("Network nodes / edges",
                    f"{net.get('n_nodes', '—')} / {net.get('n_edges', '—')}")

        st.markdown("---")
        args_info = summary.get("pipeline_args", {})
        st.markdown(
            f"**Last run:** {summary.get('run_timestamp', '—')}  |  "
            f"**Window:** {args_info.get('start', '?')} → {args_info.get('end', '?')}  |  "
            f"**Elapsed:** {summary.get('elapsed_seconds', '—')}s"
        )

        # Validation label counts
        v_labels = val.get("labels", {})
        if v_labels:
            st.subheader("Validation Label Distribution")
            vcol1, vcol2 = st.columns([1, 2])
            label_df = pd.DataFrame(
                list(v_labels.items()), columns=["Label", "Count"]
            ).sort_values("Count", ascending=False)
            vcol1.dataframe(label_df, use_container_width=True, hide_index=True)
            vcol2.bar_chart(label_df.set_index("Label"))

    else:
        st.info(
            "No summary found. Run the pipeline first:\n"
            "```\npython run_all.py --skip-download\n```"
        )

    # Pipeline run summary
    txt_path = RESULTS_DIR / "run_summary.txt"
    if txt_path.exists():
        with st.expander("Pipeline run summary"):
            st.text(txt_path.read_text(encoding="utf-8"))

    # File inventory
    st.subheader("Output File Inventory")
    file_map = {
        "processed_events.parquet"              : DATA_PROC / "processed_events.parquet",
        "news_corpus.parquet"                   : DATA_PROC / "news_corpus.parquet",
        "nse_returns.parquet"                   : DATA_PROC / "nse_returns.parquet",
        "temporal_results.parquet"              : DATA_PROC / "temporal_results.parquet",
        "event_news_pairs.parquet"              : INTERIM   / "event_news_pairs.parquet",
        "event_news_sector.parquet"             : INTERIM   / "event_news_sector.parquet",
        "integrated_validation_dataset.parquet" : DATASETS_DIR / "integrated_validation_dataset.parquet",
        "granger_results.csv"                   : DATASETS_DIR / "granger_results.csv",
        "leontief_shocks.csv"                   : DATASETS_DIR / "leontief_shocks.csv",
        "network_edges.csv"                     : DATASETS_DIR / "network_edges.csv",
    }
    inv_rows = []
    for name, path in file_map.items():
        exists = path.exists()
        size   = f"{path.stat().st_size / 1024:.1f} KB" if exists else "—"
        inv_rows.append({"File": name, "Exists": "✅" if exists else "❌", "Size": size})
    st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Event Study
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Event Study":
    st.title("📈 Event Study — MacKinlay (1997)")
    st.caption(
        "Abnormal Returns (AR) and Cumulative Abnormal Returns (CAR) per sector. "
        "Estimation window = 120 trading days; event window = (−5, +30)."
    )

    temp = _load_parquet(DATA_PROC / "temporal_results.parquet")
    if temp is None:
        _missing("temporal_results.parquet")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    sectors  = ["All"] + sorted(temp["sector"].dropna().unique().tolist()) if "sector" in temp.columns else ["All"]
    sel_sec  = col1.selectbox("Sector", sectors)
    sig_only = col2.checkbox("Significant only (p < 0.05)", value=False)
    top_n    = col3.slider("Show top N rows", 50, 2000, 200)

    filtered = temp.copy()
    if sel_sec != "All" and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"] == sel_sec]
    if sig_only and "significant" in filtered.columns:
        filtered = filtered[filtered["significant"] == True]
    filtered = filtered.head(top_n)

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Summary bar chart
    st.subheader("Mean CAR (%) by Sector")
    if "sector" in temp.columns and "CAR_pct" in temp.columns:
        base = temp[temp["significant"] == True] if (sig_only and "significant" in temp.columns) else temp
        by_sector = (
            base.groupby("sector")["CAR_pct"]
                .mean()
                .dropna()
                .sort_values()
                .reset_index()
                .rename(columns={"CAR_pct": "Mean CAR (%)"})
        )
        if not by_sector.empty:
            st.bar_chart(by_sector.set_index("sector"))

    # CAR figure if it was saved
    car_fig = FIGURES_DIR / "CAR_by_sector.png"
    if car_fig.exists():
        with st.expander("📊 Saved CAR figure"):
            st.image(str(car_fig))

    st.markdown("---")
    st.subheader("Impact window cumulative returns")
    impact_cols = [c for c in temp.columns if c.startswith("impact_")]
    if impact_cols:
        st.dataframe(
            temp[["event_id", "sector", "event_date"] + impact_cols].head(top_n),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Granger Causality
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Granger Causality":
    st.title("📊 Granger Causality (Toda-Yamamoto)")
    st.warning(
        "**L-007 / DEC-007 — Methodological constraint:**  "
        "These results describe Granger-predictive precedence only, not "
        "causality. Use 'geo_risk Granger-predicts sector_return', not 'causes'.",
        icon="⚠️",
    )

    gc = _load_csv(DATASETS_DIR / "granger_results.csv")
    if gc is None:
        _missing("granger_results.csv")
        st.stop()

    n_sig = int(gc["significant"].sum()) if "significant" in gc.columns else 0
    total = len(gc)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sectors tested", total)
    col2.metric("Granger-significant (p<0.05)", n_sig)
    col3.metric("Not significant / insufficient", total - n_sig)

    st.dataframe(gc, use_container_width=True, hide_index=True)

    if "p_value" in gc.columns and gc["p_value"].notna().any():
        st.subheader("p-value by sector")
        pv = gc[["sector", "p_value"]].dropna().set_index("sector").sort_values("p_value")
        st.bar_chart(pv)

    st.markdown("---")
    st.caption(
        "L-009: 90-day dev sample is underpowered for Granger at lag=12. "
        "Expand to 2022-2024 for valid inference."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Leontief I-O
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Leontief I-O":
    st.title("🔗 Leontief I-O Shock Propagation")

    leo = _load_csv(DATASETS_DIR / "leontief_shocks.csv")
    if leo is None:
        _missing("leontief_shocks.csv")
        st.stop()

    is_toy = "toy" in str(leo["io_matrix"].iloc[0]) if "io_matrix" in leo.columns else True
    if is_toy:
        st.warning(
            "**L-006 / DEC-009 — ILLUSTRATIVE ONLY.**  "
            "The current I-O matrix is a 5-sector toy matrix. "
            "Pass `--io-matrix <path.csv>` (India MoSPI 2017-18 or WIOD) "
            "to run_all.py for empirical results.",
            icon="⚠️",
        )
    else:
        st.success("Using real I-O matrix: " + str(leo["io_matrix"].iloc[0]))

    # Filters
    shocked_sectors = ["All"] + sorted(leo["io_shocked_label"].dropna().unique().tolist())
    sel_shock = st.selectbox("Filter by shocked sector", shocked_sectors)
    filtered  = leo if sel_shock == "All" else leo[leo["io_shocked_label"] == sel_shock]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Heatmap pivot
    st.subheader("Δ Output heatmap (shocked → receiving)")
    try:
        pivot = leo.pivot_table(
            index="receiving_sector",
            columns="io_shocked_label",
            values="delta_output",
            aggfunc="sum",
        )
        st.dataframe(pivot.style.background_gradient(cmap="RdBu_r", axis=None),
                     use_container_width=True)
    except Exception as exc:
        st.error(f"Could not build pivot: {exc}")

    # Saved heatmap figure
    hm_fig = FIGURES_DIR / "leontief_heatmap.png"
    if hm_fig.exists():
        with st.expander("📊 Saved heatmap figure"):
            st.image(str(hm_fig))


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Network Graph
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Network Graph":
    st.title("🕸️ Geopolitical Influence Network")
    st.caption(
        "Directed graph: Actor → Sector (weighted by mean |CAR_pct|) "
        "and Actor → Actor (weighted by event co-occurrence). "
        "DEC-008: plain NetworkX — GNN layer planned."
    )

    # Show saved figure
    net_fig = FIGURES_DIR / "network_graph.png"
    if net_fig.exists():
        st.image(str(net_fig), use_column_width=True)
    else:
        st.info("Network figure not yet generated. Run `python run_all.py`.")

    # Edge list
    edges = _load_csv(DATASETS_DIR / "network_edges.csv")
    if edges is not None:
        st.subheader("Edge list")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total edges", len(edges))

        edge_types = ["All"] + sorted(edges["edge_type"].dropna().unique().tolist()) \
            if "edge_type" in edges.columns else ["All"]
        sel_type = col2.selectbox("Edge type", edge_types)
        top_n    = col3.slider("Show top N rows", 50, 5000, 500)

        filtered_edges = edges if sel_type == "All" else edges[edges["edge_type"] == sel_type]
        filtered_edges = filtered_edges.sort_values(
            "weight", ascending=False
        ).head(top_n) if "weight" in filtered_edges.columns else filtered_edges.head(top_n)

        st.dataframe(filtered_edges, use_container_width=True, hide_index=True)
    else:
        _missing("network_edges.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Page: Raw Data Browser
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Raw Data Browser":
    st.title("🗂️ Raw Data Browser")
    st.caption("Browse any pipeline output file directly.")

    # Build a flat list of all parquet + csv files under the project
    all_files: list[Path] = []
    for pattern in ["**/*.parquet", "**/*.csv"]:
        all_files.extend(
            p for p in PROJECT_ROOT.glob(pattern)
            if ".git" not in p.parts and "__pycache__" not in p.parts
        )
    all_files.sort()

    if not all_files:
        st.error("No parquet or CSV files found in the project.")
        st.stop()

    rel_names   = [str(p.relative_to(PROJECT_ROOT)) for p in all_files]
    selected    = st.selectbox("Select file", rel_names)
    chosen_path = PROJECT_ROOT / selected

    col1, col2 = st.columns(2)
    max_rows    = col1.slider("Max rows to show", 50, 5000, 500)
    show_dtypes = col2.checkbox("Show column dtypes", value=False)

    if chosen_path.suffix == ".parquet":
        df = _load_parquet(chosen_path)
    else:
        df = _load_csv(chosen_path)

    if df is None:
        st.error(f"Could not load {chosen_path}")
    else:
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

        if show_dtypes:
            st.dataframe(
                df.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"}),
                use_container_width=True,
                hide_index=True,
            )

        # Search / filter
        search = st.text_input("Filter rows (substring in any column)", "")
        display = df.head(max_rows)
        if search:
            mask    = display.apply(
                lambda col: col.astype(str).str.contains(search, case=False, na=False)
            ).any(axis=1)
            display = display[mask]

        st.dataframe(display, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = display.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download filtered view as CSV",
            data=csv_bytes,
            file_name=f"{chosen_path.stem}_filtered.csv",
            mime="text/csv",
        )

"""
geosignal.visualization.propagation_map
========================================
Generic static PNG propagation map for any GeoSignal case study.

Design intent
-------------
The map is the paper's central visual: it makes the pipeline's reasoning
chain geographic and falsifiable.  For each case it shows:

  Layer 1  Event origin
           Red star at (origin_lat, origin_lon) with Goldstein score label.

  Layer 2  Directly affected countries
           Orange filled circles at country centroids, sized by mean
           |Goldstein| of GDELT events in the case filter window.
           Country ISO-2 codes taken from CaseStudy.affected_iso2.

  Layer 3  India outcome arrow
           A curved arrow from the origin to India (20°N, 77°E) labelled
           with the pipeline's detected sector and CAR result.
           Colour: green if CAR > 0 or p >= 0.05 (null),
                   red    if CAR < 0 and p < 0.05 (confirmed negative),
                   grey   if no event-study result.

  Inset panel (right)
           Mini bar chart of CAR(%) over [1d, 7d, 14d, 30d] windows
           from temporal_results.parquet (or the per-case CSV if that
           is not available).  Empty if no results found.

Generic API
-----------
Any future case that is registered in case_registry.CaseStudy plugs in
automatically — no code changes needed in this file.

    from geosignal.visualization.propagation_map import draw_propagation_map
    from experiments.case_studies.case_registry import get_case

    path = draw_propagation_map(get_case("CS-01"))
    path = draw_propagation_map(get_case("CS-06"))   # same call, different case

Dependencies
------------
    pip install matplotlib cartopy          # preferred (high-quality basemap)
    OR
    pip install matplotlib geopandas        # fallback (Natural Earth shapefile)

The module detects which backend is available and uses the best one.
If neither cartopy nor geopandas is installed it falls back to a simplified
lon/lat scatter plot with no coastlines — still usable, just less pretty.

Outputs
-------
    results/figures/<case_id>/propagation_map.png   (300 dpi, paper-ready)
    results/figures/<case_id>/propagation_map_sm.png (150 dpi, README/slides)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

# ── Backend detection ─────────────────────────────────────────────────────────

_HAS_CARTOPY  = importlib.util.find_spec("cartopy")  is not None
_HAS_GEOPANDAS = importlib.util.find_spec("geopandas") is not None

# ── Country centroid lookup (ISO-2 → (lat, lon)) ─────────────────────────────
# Hand-curated for the countries that appear across all 6 cases.
# Extend freely — the map will pick up new entries automatically.
COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "AF": (33.9, 67.7),   "AU": (-25.3, 133.8), "BR": (-14.2, -51.9),
    "CN": (35.9, 104.2),  "DE": (51.2, 10.5),   "EG": (26.8, 30.8),
    "EU": (50.1, 14.4),   "FR": (46.2, 2.2),    "GB": (55.4, -3.4),
    "ID": (-0.8, 113.9),  "IN": (20.6, 78.9),   "IR": (32.4, 53.7),
    "JP": (36.2, 138.3),  "KR": (35.9, 127.8),  "KW": (29.3, 47.5),
    "MY": (4.2, 109.5),   "NL": (52.1, 5.3),    "PK": (30.4, 69.3),
    "QA": (25.4, 51.2),   "RU": (61.5, 105.3),  "SA": (23.9, 45.1),
    "SG": (1.4, 103.8),   "TH": (15.9, 100.9),  "TR": (38.9, 35.2),
    "TW": (23.7, 121.0),  "UA": (48.4, 31.2),   "US": (37.1, -95.7),
    "VN": (14.1, 108.3),  "YE": (15.6, 48.5),   "ZA": (-30.6, 22.9),
}

INDIA_CENTROID = COUNTRY_CENTROIDS["IN"]   # (20.6, 78.9)


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def draw_propagation_map(
    case,                           # CaseStudy instance from case_registry
    temporal_df: Optional[pd.DataFrame] = None,
    events_df:   Optional[pd.DataFrame] = None,
    output_dir:  Optional[Path] = None,
    dpi_high: int = 300,
    dpi_low:  int = 150,
) -> dict[str, Path]:
    """
    Render the propagation map for one case study.

    Args:
        case:        CaseStudy instance (from case_registry.get_case()).
        temporal_df: temporal_results.parquet loaded as DataFrame (optional).
                     Used to pull CAR and p-value for the outcome arrow.
        events_df:   processed_events.parquet (optional).
                     Used to size Layer-2 country circles by event intensity.
        output_dir:  Override default output directory (case.fig_dir).
        dpi_high:    DPI for the paper-quality PNG.
        dpi_low:     DPI for the slide/README PNG.

    Returns:
        {'high': Path, 'low': Path}
    """
    out_dir = output_dir or case.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect event-study result for this case ──────────────────────────────
    car_result = _get_car_result(case, temporal_df)

    # ── Collect per-country event intensity ───────────────────────────────────
    country_intensity = _get_country_intensity(case, events_df)

    # ── Draw ──────────────────────────────────────────────────────────────────
    if _HAS_CARTOPY:
        fig = _draw_cartopy(case, car_result, country_intensity)
    elif _HAS_GEOPANDAS:
        fig = _draw_geopandas(case, car_result, country_intensity)
    else:
        fig = _draw_fallback(case, car_result, country_intensity)

    # ── Save ──────────────────────────────────────────────────────────────────
    high_path = out_dir / "propagation_map.png"
    low_path  = out_dir / "propagation_map_sm.png"
    fig.savefig(high_path, dpi=dpi_high, bbox_inches="tight")
    fig.savefig(low_path,  dpi=dpi_low,  bbox_inches="tight")
    plt.close(fig)
    return {"high": high_path, "low": low_path}


# ══════════════════════════════════════════════════════════════════════════════
#  Cartopy backend (best quality)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_cartopy(case, car_result: dict, country_intensity: dict) -> plt.Figure:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj  = ccrs.Robinson()
    fig   = plt.figure(figsize=(18, 10))
    gs    = fig.add_gridspec(1, 5, wspace=0.05)
    ax    = fig.add_subplot(gs[0, :4], projection=proj)
    ax_r  = fig.add_subplot(gs[0, 4])

    # ── Basemap ───────────────────────────────────────────────────────────────
    ax.set_global()
    ax.add_feature(cfeature.LAND,   facecolor="#f5f5f0", edgecolor="white")
    ax.add_feature(cfeature.OCEAN,  facecolor="#d6eaf8")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#aaaaaa")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#888888")
    ax.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")

    _add_layers(ax, case, car_result, country_intensity, transform=ccrs.PlateCarree())
    _add_map_title(ax, case, car_result)
    _add_legend(ax)
    _add_inset_car(ax_r, case, car_result)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  GeoPandas backend
# ══════════════════════════════════════════════════════════════════════════════

def _draw_geopandas(case, car_result: dict, country_intensity: dict) -> plt.Figure:
    import geopandas as gpd

    fig  = plt.figure(figsize=(18, 10))
    gs   = fig.add_gridspec(1, 5, wspace=0.05)
    ax   = fig.add_subplot(gs[0, :4])
    ax_r = fig.add_subplot(gs[0, 4])

    # Natural Earth low-res world shapefile (bundled with geopandas)
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        world.plot(ax=ax, color="#f5f5f0", edgecolor="#aaaaaa", linewidth=0.4)
    except Exception:
        ax.set_facecolor("#d6eaf8")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.axis("off")

    _add_layers(ax, case, car_result, country_intensity, transform=None)
    _add_map_title(ax, case, car_result)
    _add_legend(ax)
    _add_inset_car(ax_r, case, car_result)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Fallback backend (no coastlines, just markers)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_fallback(case, car_result: dict, country_intensity: dict) -> plt.Figure:
    fig  = plt.figure(figsize=(18, 10))
    gs   = fig.add_gridspec(1, 5, wspace=0.05)
    ax   = fig.add_subplot(gs[0, :4])
    ax_r = fig.add_subplot(gs[0, 4])

    ax.set_facecolor("#d6eaf8")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-75, 85)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(0.5, 0.5, "Install cartopy or geopandas\nfor coastlines",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="grey", style="italic")

    _add_layers(ax, case, car_result, country_intensity, transform=None)
    _add_map_title(ax, case, car_result)
    _add_legend(ax)
    _add_inset_car(ax_r, case, car_result)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Shared layer renderer  (works for all three backends)
# ══════════════════════════════════════════════════════════════════════════════

def _add_layers(ax, case, car_result: dict, country_intensity: dict,
                transform) -> None:
    """
    Draw all three geographic layers onto ax.
    transform is cartopy.crs.PlateCarree() for cartopy, None otherwise.
    """
    kwargs = {"transform": transform} if transform is not None else {}

    # ── Layer 2: affected countries (orange circles) ──────────────────────────
    for iso2 in case.affected_iso2:
        if iso2 not in COUNTRY_CENTROIDS or iso2 == "IN":
            continue
        lat, lon = COUNTRY_CENTROIDS[iso2]
        intensity = country_intensity.get(iso2, 1.0)
        size      = max(40, min(400, intensity * 60))
        ax.scatter(lon, lat, s=size, color="#e67e22", alpha=0.65,
                   zorder=3, edgecolors="white", linewidths=0.8, **kwargs)
        ax.text(lon + 1.5, lat + 1.5, iso2, fontsize=6.5,
                color="#884400", zorder=4,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                **kwargs)

    # ── Layer 3: India outcome marker ─────────────────────────────────────────
    i_lat, i_lon = INDIA_CENTROID
    ax.scatter(i_lon, i_lat, s=300, color="#2980b9", alpha=0.9,
               zorder=5, edgecolors="white", linewidths=1.2,
               marker="D", **kwargs)
    ax.text(i_lon + 2, i_lat - 3, "India", fontsize=8, color="#1a5276",
            fontweight="bold", zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            **kwargs)

    # ── Layer 1: event origin (red star) ─────────────────────────────────────
    ax.scatter(case.origin_lon, case.origin_lat,
               s=350, color="#e74c3c", alpha=0.95,
               zorder=7, edgecolors="white", linewidths=1.2,
               marker="*", **kwargs)
    ax.text(case.origin_lon + 2, case.origin_lat + 2,
            f"Origin\nGoldstein≈{_origin_goldstein(case):.1f}",
            fontsize=7, color="#922b21", zorder=8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            **kwargs)

    # ── India propagation arrow ───────────────────────────────────────────────
    _draw_india_arrow(ax, case, car_result, transform)


def _draw_india_arrow(ax, case, car_result: dict, transform) -> None:
    """Curved arrow from origin to India, coloured by event-study result."""
    car = car_result.get("CAR_pct")
    p   = car_result.get("p_value")

    if car is None:
        arrow_color = "#7f8c8d"   # grey — no result
        result_label = f"{case.shocked_sector}\n(no result)"
    elif p is not None and p < 0.05:
        arrow_color  = "#e74c3c" if car < 0 else "#27ae60"
        sig_str      = "p<0.05"
        result_label = f"{case.nse_ticker.replace('^','')}\nCAR={car:+.2f}%  {sig_str}"
    else:
        arrow_color  = "#7f8c8d"
        p_str        = f"p={p:.3f}" if p is not None else "p=n/a"
        result_label = f"{case.nse_ticker.replace('^','')}\nCAR={car:+.2f}%  {p_str} (null)"

    o_lon, o_lat = case.origin_lon, case.origin_lat
    i_lon, i_lat = INDIA_CENTROID[1], INDIA_CENTROID[0]

    # Mid-point for bezier curvature
    mid_lon = (o_lon + i_lon) / 2
    mid_lat = (o_lat + i_lat) / 2 + 15   # arc upward

    # Draw dashed curved line via intermediate point
    lons = np.linspace(o_lon, i_lon, 60)
    # Quadratic bezier approximation
    t    = np.linspace(0, 1, 60)
    lats = (1-t)**2 * o_lat + 2*(1-t)*t * mid_lat + t**2 * i_lat

    plot_kwargs: dict = {}
    if transform is not None:
        plot_kwargs["transform"] = transform

    ax.plot(lons, lats, color=arrow_color, lw=1.8, ls="--",
            alpha=0.85, zorder=6, **plot_kwargs)

    # Arrowhead near India
    arrow_dx = i_lon - lons[-5]
    arrow_dy = i_lat - lats[-5]
    ax.annotate(
        "", xy=(i_lon, i_lat), xytext=(lons[-5], lats[-5]),
        arrowprops=dict(
            arrowstyle="-|>", color=arrow_color,
            lw=1.8, mutation_scale=14,
        ),
        zorder=7,
        **({"xycoords": transform, "textcoords": transform}
           if transform is not None else {}),
    )

    # Label near midpoint
    label_lon = mid_lon + 2
    label_lat = mid_lat + 2
    ax.text(label_lon, label_lat, result_label,
            fontsize=7, color=arrow_color, zorder=8,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=arrow_color, alpha=0.85, lw=0.8),
            path_effects=[pe.withStroke(linewidth=1, foreground="white")],
            **plot_kwargs)


def _add_map_title(ax, case, car_result: dict) -> None:
    sig = ""
    if car_result.get("p_value") is not None:
        sig = ("  [CONFIRMED p<0.05]" if car_result["p_value"] < 0.05
               else f"  [null p={car_result['p_value']:.3f}]")
    ax.set_title(
        f"GeoSignal Propagation Map  —  {case.case_id}: {case.label}\n"
        f"Anchor: {case.anchor_date}  |  Stratum {case.stratum}  |  "
        f"Shocked sector: {case.shocked_sector}{sig}",
        fontsize=10, pad=10,
    )


def _add_legend(ax) -> None:
    handles = [
        mpatches.Patch(color="#e74c3c",  label="Event origin (★)"),
        mpatches.Patch(color="#e67e22",  label="Directly affected countries"),
        mpatches.Patch(color="#2980b9",  label="India (outcome market)"),
        mpatches.Patch(color="#e74c3c",  label="Confirmed negative CAR (p<0.05)"),
        mpatches.Patch(color="#27ae60",  label="Confirmed positive CAR (p<0.05)"),
        mpatches.Patch(color="#7f8c8d",  label="Null result / no data"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7,
              framealpha=0.9, ncol=2)


# ══════════════════════════════════════════════════════════════════════════════
#  Inset: CAR bar chart
# ══════════════════════════════════════════════════════════════════════════════

def _add_inset_car(ax_r, case, car_result: dict) -> None:
    """Mini bar chart of raw cumulative returns at 1/7/14/30-day horizons."""
    windows     = [1, 7, 14, 30]
    labels_str  = ["1d", "7d", "14d", "30d"]
    values      = [car_result.get(f"impact_{w}d") for w in windows]
    has_data    = any(v is not None for v in values)

    if not has_data:
        ax_r.text(0.5, 0.5, "No impact data\navailable",
                  ha="center", va="center", transform=ax_r.transAxes,
                  fontsize=9, color="grey", style="italic")
        ax_r.set_title("Impact windows", fontsize=9)
        ax_r.axis("off")
        return

    vals   = [v if v is not None else 0.0 for v in values]
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in vals]

    ax_r.bar(labels_str, vals, color=colors, alpha=0.85, edgecolor="white")
    ax_r.axhline(0, color="black", lw=0.8, ls="--")
    ax_r.set_title(
        f"{case.nse_ticker.replace('^','')}\nRaw cumul. return (%)",
        fontsize=8,
    )
    ax_r.set_ylabel("Return (%)", fontsize=8)
    ax_r.tick_params(axis="both", labelsize=7)
    ax_r.grid(axis="y", alpha=0.3)

    # Annotate CAR from event study if available
    car = car_result.get("CAR_pct")
    p   = car_result.get("p_value")
    if car is not None:
        sig = "★" if (p is not None and p < 0.05) else ""
        ax_r.text(0.5, 0.02,
                  f"EventStudy CAR={car:+.3f}%{sig}",
                  ha="center", va="bottom",
                  transform=ax_r.transAxes,
                  fontsize=7, color="#2c3e50",
                  bbox=dict(boxstyle="round,pad=0.2",
                            fc="lightyellow", ec="grey", alpha=0.9))


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_car_result(case, temporal_df: Optional[pd.DataFrame]) -> dict:
    """
    Extract CAR, p-value, and impact windows for this case from temporal_df.
    Falls back to the per-case CSV written by the case study script.
    Returns empty dict if nothing found.
    """
    result: dict = {}

    # Try temporal_df (full pipeline output)
    if temporal_df is not None and not temporal_df.empty:
        anchor = case.anchor
        subset = temporal_df[
            (temporal_df.get("ticker", pd.Series(dtype=str)) == case.nse_ticker) |
            (temporal_df.get("sector", pd.Series(dtype=str)).str.contains(
                case.shocked_sector, case=False, na=False))
        ].copy()
        if "event_date" in subset.columns:
            subset["event_date"] = pd.to_datetime(subset["event_date"])
            window = subset[
                (subset["event_date"] >= anchor - pd.Timedelta(days=3)) &
                (subset["event_date"] <= anchor + pd.Timedelta(days=3))
            ]
            if not window.empty:
                row = window.sort_values("p_value").iloc[0]
                result = {
                    "CAR_pct"  : row.get("CAR_pct"),
                    "t_stat"   : row.get("t_stat"),
                    "p_value"  : row.get("p_value"),
                    "significant": row.get("significant"),
                }
                for w in [1, 7, 14, 30]:
                    k = f"impact_{w}d"
                    if k in row:
                        result[k] = row[k]
                return result

    # Fall back to per-case CSV
    csv_path = case.datasets_dir / f"{case.case_id.lower().replace('-','')}_event_study.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                row = df.iloc[0]
                result = {
                    "CAR_pct"   : row.get("CAR_pct"),
                    "t_stat"    : row.get("t_stat"),
                    "p_value"   : row.get("p_value"),
                    "significant": row.get("significant"),
                }
                for w in [1, 7, 14, 30]:
                    k = f"impact_{w}d"
                    if k in row:
                        result[k] = row[k]
        except Exception:
            pass

    return result


def _get_country_intensity(case, events_df: Optional[pd.DataFrame]) -> dict[str, float]:
    """
    For each country in case.affected_iso2, compute mean |Goldstein| of
    GDELT events in the case window.  Used to size Layer-2 circles.
    Returns a dict {iso2: intensity_scalar}.
    """
    intensity: dict[str, float] = {iso2: 1.0 for iso2 in case.affected_iso2}
    if events_df is None or events_df.empty:
        return intensity
    if "action_country" not in events_df.columns or "goldstein" not in events_df.columns:
        return intensity

    events_df = events_df.copy()
    events_df["date"] = pd.to_datetime(events_df["date"])
    window = events_df[
        (events_df["date"] >= case.anchor - pd.Timedelta(days=7)) &
        (events_df["date"] <= case.anchor + pd.Timedelta(days=14))
    ]

    for iso2 in case.affected_iso2:
        subset = window[window["action_country"] == iso2]
        if not subset.empty:
            intensity[iso2] = float(subset["goldstein"].abs().mean())

    return intensity


def _origin_goldstein(case) -> float:
    """Return the Goldstein scale implied by the shock magnitude as a proxy."""
    # shock_magnitude is negative for adverse events: -0.10 → ~5 on Goldstein
    return min(10.0, abs(case.shock_magnitude) * 100)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — draw map for one or all cases
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_root))

    from experiments.case_studies.case_registry import ALL_CASES, get_case
    from config.settings import DATA_PROC, RESULTS_DIR

    p = argparse.ArgumentParser(
        description="Draw propagation map for one or all cases"
    )
    p.add_argument("--case", default="all",
                   help="Case ID (e.g. CS-01) or 'all'")
    args = p.parse_args()

    # Load pipeline outputs if available
    temporal_df = None
    events_df   = None
    t_path = DATA_PROC / "temporal_results.parquet"
    e_path = DATA_PROC / "processed_events.parquet"
    if t_path.exists():
        temporal_df = pd.read_parquet(t_path)
    if e_path.exists():
        events_df = pd.read_parquet(e_path)

    cases_to_run = ALL_CASES if args.case == "all" else [get_case(args.case)]
    for case in cases_to_run:
        print(f"Drawing map for {case.case_id}: {case.label} ...", end=" ")
        paths = draw_propagation_map(case, temporal_df, events_df)
        print(f"saved -> {paths['high']}")

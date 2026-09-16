"""
src/impact_analysis.py
Phase V: Integration, Validation, and Explanation

Assembles the final validated event-impact dataset:
  event_id | date | event_type | actor | target | sector | 
  news_count | impact_7d | impact_14d | impact_30d | CAR | p_value | validation

Also implements Leontief Input-Output propagation for supply-chain ripple estimation.

Usage:
    python src/impact_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROC, RESULTS_DIR


# ── Validation Logic ──────────────────────────────────────────────────────────

def assign_validation_label(row: pd.Series) -> str:
    """
    Determine whether the event-impact relationship is validated:
      - 'Confirmed' : CAR is statistically significant (p < 0.05)
      - 'Null'      : No significant impact detected
      - 'Insufficient' : Missing data / not enough observations
    """
    p = row.get("p_value")
    if pd.isna(p):
        return "Insufficient"
    return "Confirmed" if p < 0.05 else "Null"


def build_integrated_dataset(events_df: pd.DataFrame,
                              temporal_df: pd.DataFrame,
                              pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all pipeline outputs into the final integrated validation dataset.

    Args:
        events_df:   Canonical events (from preprocessing.py)
        temporal_df: Event study results (from temporal_analysis.py)
        pairs_df:    Event-news pairs with sector (from nlp_pipeline.py + sector_mapping.py)

    Returns:
        Integrated DataFrame matching the paper's final table schema.
    """
    # Count news articles per event
    news_counts = (
        pairs_df.groupby("event_id")
                .size()
                .reset_index(name="news_count")
    )

    # Base: join events with temporal results
    merged = events_df.merge(temporal_df, on="event_id", how="left")
    merged = merged.merge(news_counts, on="event_id", how="left")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    # Sector from pairs (most common per event)
    if "sector" in pairs_df.columns:
        sector_mode = (
            pairs_df[pairs_df["sector"] != "Unknown"]
            .groupby("event_id")["sector"]
            .agg(lambda x: x.mode()[0] if not x.empty else "Unknown")
            .reset_index()
        )
        merged = merged.merge(sector_mode, on="event_id", how="left", suffixes=("_old", ""))
        if "sector_old" in merged.columns:
            merged["sector"] = merged["sector"].fillna(merged["sector_old"])
            merged.drop(columns=["sector_old"], inplace=True)

    # Assign validation label
    merged["validation"] = merged.apply(assign_validation_label, axis=1)

    # Final column selection
    final_cols = [
        "event_id", "date", "event_root", "goldstein",
        "actor1", "actor2", "action_country", "sector",
        "news_count",
        "impact_7d", "impact_14d", "impact_30d",
        "CAR_pct", "t_stat", "p_value", "significant",
        "validation",
    ]
    available = [c for c in final_cols if c in merged.columns]
    return merged[available].reset_index(drop=True)


# ── Leontief I-O Propagation ──────────────────────────────────────────────────

class LeontiefPropagator:
    """
    Simplified Leontief Input-Output propagation.

    Models how a supply shock in one sector propagates through inter-industry
    linkages using the Leontief inverse: (I - A)^{-1}

    The technical coefficient matrix A can be derived from:
      - World Input-Output Database (WIOD)
      - OECD ICIO tables
      - RBI / MoSPI India I-O tables (2017-18)
    """

    def __init__(self, io_matrix: np.ndarray, sector_labels: list[str]):
        """
        Args:
            io_matrix:     Square technical coefficient matrix A (n x n).
                           A[i,j] = value of input i required to produce 1 unit of j.
            sector_labels: List of sector names corresponding to matrix rows/columns.
        """
        n = io_matrix.shape[0]
        assert io_matrix.shape == (n, n), "IO matrix must be square."
        assert len(sector_labels) == n, "sector_labels length must match matrix size."

        self.A             = io_matrix
        self.sector_labels = sector_labels
        self.n             = n

        # Leontief inverse L = (I - A)^{-1}
        I = np.eye(n)
        self.L = np.linalg.inv(I - io_matrix)
        logger.info(f"Leontief inverse computed for {n} sectors.")

    def propagate_shock(self, shock_sector: str, shock_magnitude: float) -> pd.Series:
        """
        Compute total output impact of a demand/supply shock in one sector.

        Args:
            shock_sector:    Name of the shocked sector.
            shock_magnitude: Magnitude of the final demand shock (e.g. -0.15 = 15% reduction).

        Returns:
            pd.Series of total output changes per sector.
        """
        if shock_sector not in self.sector_labels:
            raise ValueError(f"Sector '{shock_sector}' not in label list.")

        idx               = self.sector_labels.index(shock_sector)
        delta_d           = np.zeros(self.n)
        delta_d[idx]      = shock_magnitude

        # Total output change = Leontief inverse × final demand shock
        delta_x = self.L @ delta_d

        return pd.Series(delta_x, index=self.sector_labels).sort_values()

    @classmethod
    def from_csv(cls, path: Path) -> "LeontiefPropagator":
        """Load A matrix from a CSV file (first row/col = sector labels)."""
        df = pd.read_csv(path, index_col=0)
        return cls(io_matrix=df.values, sector_labels=list(df.index))


def _demo_leontief() -> None:
    """Quick demonstration with a 5-sector toy economy."""
    labels = ["Energy", "Semiconductor", "Automotive", "Logistics", "ConsumerGoods"]
    # Toy A matrix: fraction of input from row-sector needed per unit of col-sector
    A = np.array([
        [0.10, 0.05, 0.08, 0.15, 0.04],
        [0.02, 0.12, 0.20, 0.01, 0.05],
        [0.03, 0.08, 0.10, 0.10, 0.12],
        [0.15, 0.04, 0.07, 0.05, 0.08],
        [0.01, 0.02, 0.03, 0.04, 0.10],
    ])

    lp    = LeontiefPropagator(A, labels)
    shock = lp.propagate_shock("Semiconductor", shock_magnitude=-0.15)
    logger.info("\nLeontief propagation (Semiconductor −15% shock):\n" + str(shock.round(4)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    events_path   = DATA_PROC / "processed_events.parquet"
    temporal_path = DATA_PROC / "temporal_results.parquet"
    pairs_path    = DATA_PROC / "event_news_sector.parquet"

    missing = [p for p in [events_path, temporal_path, pairs_path] if not p.exists()]
    if missing:
        logger.error(f"Missing files: {missing}. Run preceding pipeline steps first.")
        return

    events_df   = pd.read_parquet(events_path)
    temporal_df = pd.read_parquet(temporal_path)
    pairs_df    = pd.read_parquet(pairs_path)

    integrated = build_integrated_dataset(events_df, temporal_df, pairs_df)

    # Save
    out_csv     = RESULTS_DIR / "integrated_validation_dataset.csv"
    out_parquet = RESULTS_DIR / "integrated_validation_dataset.parquet"
    integrated.to_csv(out_csv,     index=False)
    integrated.to_parquet(out_parquet, index=False)
    logger.success(f"Integrated dataset saved → {out_csv}  ({len(integrated)} rows)")

    # Summary stats
    summary = integrated.groupby("validation").size().reset_index(name="count")
    logger.info("\nValidation summary:\n" + summary.to_string(index=False))

    # Leontief demo
    _demo_leontief()


if __name__ == "__main__":
    main()

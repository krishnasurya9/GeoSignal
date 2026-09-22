"""
geosignal.analysis.impact_analysis
Phase V — Assemble the integrated event-impact validation dataset.

Merges:
  - processed_events.parquet      (canonical event schema)
  - temporal_results.parquet      (event study CAR / raw impact)
  - event_news_sector.parquet     (sector-enriched event-news pairs)

Assigns validation labels:
  Confirmed    p < 0.05
  Null         p >= 0.05
  Insufficient missing data
"""

import pandas as pd
from loguru import logger

from config.settings import DATA_PROC, RESULTS_DIR


def assign_validation_label(row: pd.Series) -> str:
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
        events_df:   canonical events (from preprocessing)
        temporal_df: event study results (from event_study)
        pairs_df:    sector-enriched pairs (from sector_mapping)

    Returns:
        Integrated DataFrame with validation labels.
    """
    news_counts = (
        pairs_df.groupby("event_id").size().reset_index(name="news_count")
    )
    merged = events_df.merge(temporal_df, on="event_id", how="left")
    merged = merged.merge(news_counts,    on="event_id", how="left")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    if "sector" in pairs_df.columns:
        sector_mode = (
            pairs_df[pairs_df["sector"] != "Unknown"]
            .groupby("event_id")["sector"]
            .agg(lambda x: x.mode()[0] if len(x) else "Unknown")
            .reset_index()
        )
        merged = merged.merge(sector_mode, on="event_id", how="left", suffixes=("_old", ""))
        if "sector_old" in merged.columns:
            merged["sector"] = merged["sector"].fillna(merged["sector_old"])
            merged.drop(columns=["sector_old"], inplace=True)

    merged["validation"] = merged.apply(assign_validation_label, axis=1)

    final_cols = [
        "event_id", "date", "event_root", "goldstein",
        "actor1", "actor2", "action_country", "sector", "news_count",
        "impact_7d", "impact_14d", "impact_30d",
        "CAR_pct", "t_stat", "p_value", "significant", "validation",
    ]
    available = [c for c in final_cols if c in merged.columns]
    return merged[available].reset_index(drop=True)


def main():
    events_path  = DATA_PROC / "processed_events.parquet"
    temporal_path = DATA_PROC / "temporal_results.parquet"
    pairs_path   = DATA_PROC / "event_news_sector.parquet"

    missing = [p for p in [events_path, temporal_path, pairs_path] if not p.exists()]
    if missing:
        logger.error(f"Missing inputs: {missing}")
        return

    events_df   = pd.read_parquet(events_path)
    temporal_df = pd.read_parquet(temporal_path)
    pairs_df    = pd.read_parquet(pairs_path)

    integrated = build_integrated_dataset(events_df, temporal_df, pairs_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets_dir = RESULTS_DIR / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    integrated.to_csv(datasets_dir    / "integrated_validation_dataset.csv",     index=False)
    integrated.to_parquet(datasets_dir / "integrated_validation_dataset.parquet", index=False)
    logger.success(f"Integrated dataset → {datasets_dir}  ({len(integrated)} rows)")

    summary = integrated.groupby("validation").size().reset_index(name="count")
    logger.info(f"\nValidation summary:\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()

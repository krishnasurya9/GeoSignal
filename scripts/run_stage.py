"""
scripts/run_stage.py
Run a single named pipeline stage independently.

Usage:
    python scripts/run_stage.py --stage preprocess
    python scripts/run_stage.py --stage nlp
    python scripts/run_stage.py --stage sectors
    python scripts/run_stage.py --stage market --start 2022-01-01 --end 2024-12-31
    python scripts/run_stage.py --stage temporal
    python scripts/run_stage.py --stage impact
    python scripts/run_stage.py --stage leontief

Available stages:
    download_gdelt   Download GDELT 1.0 raw events
    download_gkg     Download GDELT GKG news corpus
    preprocess       Harmonize raw GDELT → processed_events.parquet
    nlp              BM25 + TF-IDF alignment → event_news_pairs.parquet
    sectors          GICS sector enrichment → event_news_sector.parquet
    market           NSE index download → nse_returns.parquet
    temporal         Event study + CAR → temporal_results.parquet
    impact           Assemble integrated validation dataset
    leontief         Run illustrative Leontief shock (no file output)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from run_pipeline import (
    stage_download_gdelt,
    stage_download_gkg,
    stage_preprocess,
    stage_nlp_alignment,
    stage_sector_mapping,
    stage_market_data,
    stage_temporal_analysis,
    stage_impact_analysis,
)


def stage_leontief():
    from geosignal.propagation.leontief import LeontiefPropagator
    lp    = LeontiefPropagator.toy_5sector()
    shock = lp.propagate_shock("Semiconductor", -0.15)
    logger.info(f"Leontief shock (Semiconductor −15%):\n{shock.round(4).to_string()}")


STAGE_MAP = {
    "download_gdelt": lambda a: stage_download_gdelt(a.start, a.end, a.force),
    "download_gkg"  : lambda a: stage_download_gkg(a.start, a.end, a.force),
    "preprocess"    : lambda a: stage_preprocess(a.force),
    "nlp"           : lambda a: stage_nlp_alignment(a.force),
    "sectors"       : lambda a: stage_sector_mapping(a.force),
    "market"        : lambda a: stage_market_data(a.start, a.end, a.force),
    "temporal"      : lambda a: stage_temporal_analysis(a.force),
    "impact"        : lambda a: stage_impact_analysis(a.force),
    "leontief"      : lambda _: stage_leontief(),
}


def main():
    parser = argparse.ArgumentParser(description="GeoSignal — run single pipeline stage")
    parser.add_argument("--stage",  required=True, choices=list(STAGE_MAP))
    parser.add_argument("--start",  default="2024-01-01")
    parser.add_argument("--end",    default="2024-03-31")
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    logger.info(f"Running stage: {args.stage}")
    STAGE_MAP[args.stage](args)
    logger.success(f"Stage '{args.stage}' complete.")


if __name__ == "__main__":
    main()

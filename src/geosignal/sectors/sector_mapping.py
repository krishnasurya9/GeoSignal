"""
geosignal.sectors.sector_mapping
Entity → GICS sector mapping + sector-enrichment of event–news pairs.

Mapping is deterministic keyword lookup (L-008).
EntityRuler labels (SEMICONDUCTOR, ENERGY, LOGISTICS, …) take priority
over plain-text keyword matching.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import GICS_MAP_PATH

# ── Keyword → GICS sector mapping ────────────────────────────────────────────

KEYWORD_TO_GICS: dict[str, str] = {
    # Energy
    "oil": "Energy", "crude": "Energy", "petroleum": "Energy",
    "natural gas": "Energy", "lng": "Energy", "petroline": "Energy",
    "brent": "Energy", "opec": "Energy",
    # Semiconductors / IT
    "semiconductor": "Information Technology", "chip": "Information Technology",
    "euv": "Information Technology", "nand": "Information Technology",
    "microcontroller": "Information Technology", "semiconductor fab": "Information Technology",
    "eda software": "Information Technology",
    # Logistics / Transportation
    "suez": "Industrials", "shipping": "Industrials", "freight": "Industrials",
    "container": "Industrials", "logistics": "Industrials",
    "cape of good hope": "Industrials",
    # Financials
    "bank": "Financials", "insurance": "Financials", "credit": "Financials",
    # Consumer Staples / Agriculture
    "rice": "Consumer Staples", "food": "Consumer Staples",
    "agriculture": "Consumer Staples", "wheat": "Consumer Staples",
    # Materials
    "fertiliser": "Materials", "lithium": "Materials", "cobalt": "Materials",
    "rare earth": "Materials", "copper": "Materials", "aluminum": "Materials",
    # Health Care
    "pharmaceutical": "Health Care", "pharma": "Health Care",
    # Consumer Discretionary
    "automotive": "Consumer Discretionary", "vehicle": "Consumer Discretionary",
    "automobile": "Consumer Discretionary",
}

# EntityRuler label → GICS override (takes priority over text keywords)
LABEL_TO_GICS: dict[str, str] = {
    "SEMICONDUCTOR":    "Information Technology",
    "ENERGY":           "Energy",
    "LOGISTICS":        "Industrials",
    "AGRICULTURE":      "Consumer Staples",
    "CRITICAL_MINERAL": "Materials",
}


def map_entity_to_gics(entity_text: str, entity_label: str = "") -> str:
    """
    Map an extracted entity to a GICS sector.

    Priority:
      1. EntityRuler label (SEMICONDUCTOR, ENERGY, …)
      2. Keyword substring match on entity_text
      3. 'Unknown'

    Args:
        entity_text:  Raw entity string from NER.
        entity_label: spaCy entity label (optional).

    Returns:
        GICS sector string or 'Unknown'.
    """
    if entity_label in LABEL_TO_GICS:
        return LABEL_TO_GICS[entity_label]
    text_lower = entity_text.lower()
    for keyword, sector in KEYWORD_TO_GICS.items():
        if keyword in text_lower:
            return sector
    return "Unknown"


def enrich_pairs_with_sectors(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'sector' and 'all_sectors' columns to event–news pairs by mapping
    NER entities. Most common non-Unknown sector per row is chosen.

    Args:
        pairs_df: DataFrame with 'entities' column (JSON list).

    Returns:
        pairs_df with added 'sector' and 'all_sectors' columns.
    """
    sectors_list = []
    all_sectors_list = []

    for _, row in pairs_df.iterrows():
        try:
            entities = json.loads(row.get("entities", "[]"))
        except (json.JSONDecodeError, TypeError):
            entities = []

        row_sectors = [
            map_entity_to_gics(e["text"], e.get("label", ""))
            for e in entities
            if e.get("label") in (
                "ORG", "GPE", "PRODUCT",
                "SEMICONDUCTOR", "ENERGY", "LOGISTICS",
                "AGRICULTURE", "CRITICAL_MINERAL",
            )
        ]
        row_sectors = [s for s in row_sectors if s != "Unknown"]
        sector = max(set(row_sectors), key=row_sectors.count) if row_sectors else "Unknown"
        sectors_list.append(sector)
        all_sectors_list.append(row_sectors)

    out = pairs_df.copy()
    out["sector"]      = sectors_list
    out["all_sectors"] = [str(s) for s in all_sectors_list]
    return out

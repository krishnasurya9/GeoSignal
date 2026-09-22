"""tests/test_sector_mapping.py — GICS mapping and pair enrichment"""
import json
import pandas as pd
import pytest
from geosignal.sectors.sector_mapping import map_entity_to_gics, enrich_pairs_with_sectors


# ── map_entity_to_gics ────────────────────────────────────────────────────────

def test_energy_keyword():
    assert map_entity_to_gics("crude oil") == "Energy"

def test_energy_label_override():
    assert map_entity_to_gics("anything", "ENERGY") == "Energy"

def test_semiconductor_keyword():
    assert map_entity_to_gics("semiconductor fab") == "Information Technology"

def test_semiconductor_label_override():
    assert map_entity_to_gics("irrelevant text", "SEMICONDUCTOR") == "Information Technology"

def test_logistics_keyword():
    assert map_entity_to_gics("Suez Canal blocked") == "Industrials"

def test_logistics_label_override():
    assert map_entity_to_gics("anything", "LOGISTICS") == "Industrials"

def test_agriculture_keyword():
    assert map_entity_to_gics("rice shipment") == "Consumer Staples"

def test_agriculture_label_override():
    assert map_entity_to_gics("x", "AGRICULTURE") == "Consumer Staples"

def test_materials_keyword():
    assert map_entity_to_gics("lithium battery") == "Materials"

def test_critical_mineral_label():
    assert map_entity_to_gics("cobalt mine", "CRITICAL_MINERAL") == "Materials"

def test_pharma_keyword():
    assert map_entity_to_gics("pharma company") == "Health Care"

def test_unknown():
    assert map_entity_to_gics("random gibberish xyz") == "Unknown"

def test_case_insensitive():
    assert map_entity_to_gics("CRUDE OIL PIPELINE") == "Energy"


# ── enrich_pairs_with_sectors ─────────────────────────────────────────────────

def _make_pairs(entity_list: list[list[dict]]) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id"    : [f"E{i}" for i in range(len(entity_list))],
        "article_date": pd.date_range("2024-01-01", periods=len(entity_list)),
        "entities"    : [json.dumps(e) for e in entity_list],
    })


def test_enrich_assigns_sector():
    pairs = _make_pairs([[{"text": "crude oil", "label": "ORG"}]])
    result = enrich_pairs_with_sectors(pairs)
    assert result["sector"].iloc[0] == "Energy"


def test_enrich_unknown_fallback():
    pairs = _make_pairs([[{"text": "xyzzy gibberish", "label": "ORG"}]])
    result = enrich_pairs_with_sectors(pairs)
    assert result["sector"].iloc[0] == "Unknown"


def test_enrich_empty_entities():
    pairs = _make_pairs([[]])
    result = enrich_pairs_with_sectors(pairs)
    assert result["sector"].iloc[0] == "Unknown"


def test_enrich_label_priority():
    """EntityRuler label beats plain text keyword."""
    pairs = _make_pairs([[{"text": "shipping route", "label": "SEMICONDUCTOR"}]])
    result = enrich_pairs_with_sectors(pairs)
    # SEMICONDUCTOR label → Information Technology (overrides "shipping" → Industrials)
    assert result["sector"].iloc[0] == "Information Technology"


def test_enrich_most_common_sector():
    """Most common non-Unknown sector is selected."""
    entities = [
        {"text": "crude oil", "label": "ORG"},
        {"text": "oil pipeline", "label": "ORG"},
        {"text": "semiconductor chip", "label": "ORG"},
    ]
    pairs = _make_pairs([entities])
    result = enrich_pairs_with_sectors(pairs)
    assert result["sector"].iloc[0] == "Energy"   # 2 energy vs 1 IT

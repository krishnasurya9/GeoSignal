"""
geosignal.nlp.entity_extraction
spaCy NER + EntityRuler pipeline for geopolitical entity extraction.

Custom labels added via EntityRuler (runs BEFORE statistical NER):
    SEMICONDUCTOR, ENERGY, LOGISTICS, AGRICULTURE, CRITICAL_MINERAL

NER must run on raw_text (preserves capitalisation for proper noun
detection). Never run on lowercased cleaned text (DEC-005).
"""

import json
from pathlib import Path

import spacy
from loguru import logger
from spacy.language import Language

from config.settings import ENTITY_RULES_PATH


def build_nlp_pipeline() -> Language:
    """
    Load en_core_web_sm and prepend a custom EntityRuler with
    industry-specific patterns from config/entity_rules.jsonl.

    Returns:
        Configured spaCy Language pipeline.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )

    patterns = _load_entity_patterns()
    ruler    = nlp.add_pipe("entity_ruler", before="ner",
                             config={"overwrite_ents": False})
    ruler.add_patterns(patterns)
    logger.info(f"EntityRuler loaded: {len(patterns)} custom patterns.")
    return nlp


def _load_entity_patterns() -> list[dict]:
    """Load patterns from JSONL; fall back to built-in set if file missing."""
    if ENTITY_RULES_PATH.exists():
        patterns = []
        with open(ENTITY_RULES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    patterns.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping invalid EntityRuler line: {line!r} ({exc})")
        return patterns

    logger.warning(f"EntityRuler JSONL not found at {ENTITY_RULES_PATH} — using built-in fallback.")
    return [
        {"label": "SEMICONDUCTOR",    "pattern": "EUV lithography"},
        {"label": "SEMICONDUCTOR",    "pattern": "NAND flash"},
        {"label": "SEMICONDUCTOR",    "pattern": "advanced silicon"},
        {"label": "SEMICONDUCTOR",    "pattern": "fabrication plant"},
        {"label": "SEMICONDUCTOR",    "pattern": "microcontroller"},
        {"label": "ENERGY",           "pattern": "Brent crude"},
        {"label": "ENERGY",           "pattern": "Petroline"},
        {"label": "LOGISTICS",        "pattern": "Suez Canal"},
        {"label": "LOGISTICS",        "pattern": "Cape of Good Hope"},
        {"label": "LOGISTICS",        "pattern": "freight surcharge"},
        {"label": "LOGISTICS",        "pattern": "container throughput"},
        {"label": "AGRICULTURE",      "pattern": "rice shipment"},
        {"label": "AGRICULTURE",      "pattern": "food inflation"},
        {"label": "CRITICAL_MINERAL", "pattern": "lithium"},
        {"label": "CRITICAL_MINERAL", "pattern": "cobalt"},
        {"label": "CRITICAL_MINERAL", "pattern": "rare earth"},
    ]


def extract_entities(text: str, nlp: Language) -> list[dict]:
    """
    Run the NLP pipeline on text and return extracted entities.

    Args:
        text: Raw (uncleaned) text — must preserve capitalisation for NER.
        nlp:  Pre-built spaCy pipeline from build_nlp_pipeline().

    Returns:
        List of {"text": str, "label": str, "start": int, "end": int}
    """
    doc = nlp(text[:100_000])  # guard against very long strings
    return [
        {"text": ent.text, "label": ent.label_,
         "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]

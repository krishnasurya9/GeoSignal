"""tests/test_nlp.py — NLP alignment (tiny fixture, no full corpus)"""
import pandas as pd
import pytest
from geosignal.nlp.alignment import NewsAligner
from geosignal.nlp.entity_extraction import build_nlp_pipeline, extract_entities


# ── NewsAligner ───────────────────────────────────────────────────────────────

def test_news_aligner_builds(tiny_news):
    aligner = NewsAligner(tiny_news, max_tfidf_features=200)
    assert aligner.bm25 is not None
    assert aligner.corpus_matrix is not None


def test_retrieve_returns_dataframe(tiny_news, tiny_events):
    aligner = NewsAligner(tiny_news, max_tfidf_features=200)
    event   = tiny_events.iloc[0]
    result  = aligner.retrieve("semiconductor chip", pd.Timestamp(event["date"]), top_n=3)
    assert isinstance(result, pd.DataFrame)


def test_retrieve_respects_temporal_band(tiny_news, tiny_events):
    aligner = NewsAligner(tiny_news, max_tfidf_features=200)
    event   = tiny_events.iloc[0]
    result  = aligner.retrieve("semiconductor", pd.Timestamp(event["date"]),
                                top_n=10, temporal_band=1)
    # All returned articles should be within ±1 day of event date
    if not result.empty:
        lo = pd.Timestamp(event["date"]) - pd.Timedelta(days=1)
        hi = pd.Timestamp(event["date"]) + pd.Timedelta(days=1)
        assert (result["date"] >= lo).all()
        assert (result["date"] <= hi).all()


def test_cosine_scores_no_refit(tiny_news):
    """cosine_scores uses transform() only — vectorizer must not be refitted."""
    aligner = NewsAligner(tiny_news, max_tfidf_features=200)
    scores  = aligner.cosine_scores(
        "semiconductor export control",
        ["semiconductor chip supply chain", "oil tanker shipping route"]
    )
    assert len(scores) == 2
    # More relevant candidate should score higher
    assert scores[0] >= scores[1]


def test_cosine_scores_empty_candidates(tiny_news):
    aligner = NewsAligner(tiny_news, max_tfidf_features=200)
    scores  = aligner.cosine_scores("anything", [])
    assert scores == []


# ── Entity extraction ─────────────────────────────────────────────────────────

def test_entity_extraction_returns_list():
    nlp  = build_nlp_pipeline()
    ents = extract_entities("TSMC manufactures EUV lithography chips.", nlp)
    assert isinstance(ents, list)


def test_entity_extraction_has_required_keys():
    nlp  = build_nlp_pipeline()
    ents = extract_entities("India and China signed a trade agreement.", nlp)
    for e in ents:
        assert "text" in e
        assert "label" in e
        assert "start" in e
        assert "end" in e


def test_entity_extraction_custom_label():
    """EntityRuler should fire SEMICONDUCTOR on known patterns."""
    nlp  = build_nlp_pipeline()
    ents = extract_entities("The factory uses EUV lithography equipment.", nlp)
    labels = [e["label"] for e in ents]
    assert "SEMICONDUCTOR" in labels


def test_entity_extraction_handles_empty():
    nlp  = build_nlp_pipeline()
    ents = extract_entities("", nlp)
    assert ents == []

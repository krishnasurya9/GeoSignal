"""tests/test_data.py — preprocessing and text cleaning"""
import pandas as pd
import pytest
from geosignal.data.preprocessing import (
    clean_text, jaccard_similarity, load_goldstein_map,
    deduplicate_news,
)


def test_clean_text_lowercases():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_strips_html():
    assert clean_text("<b>bold</b> text") == "bold text"


def test_clean_text_removes_punctuation():
    assert clean_text("hello, world!") == "hello world"


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_empty():
    assert clean_text("") == ""


def test_jaccard_identical():
    assert jaccard_similarity("the cat sat", "the cat sat") == 1.0


def test_jaccard_disjoint():
    assert jaccard_similarity("abc def", "xyz uvw") == 0.0


def test_jaccard_partial():
    j = jaccard_similarity("cat sat mat", "cat bat hat")
    assert 0 < j < 1


def test_load_goldstein_map_returns_dict():
    gmap = load_goldstein_map()
    assert isinstance(gmap, dict)
    assert len(gmap) > 0


def test_load_goldstein_map_values_are_floats():
    gmap = load_goldstein_map()
    for v in gmap.values():
        assert isinstance(v, float)


def test_deduplicate_news_removes_duplicate():
    df = pd.DataFrame({
        "text": ["hello world foo bar baz", "hello world foo bar baz", "completely different text"]
    })
    result = deduplicate_news(df, threshold=0.9)
    assert len(result) == 2   # one duplicate removed


def test_deduplicate_news_keeps_distinct():
    df = pd.DataFrame({
        "text": ["alpha beta gamma", "delta epsilon zeta", "theta iota kappa"]
    })
    result = deduplicate_news(df, threshold=0.85)
    assert len(result) == 3

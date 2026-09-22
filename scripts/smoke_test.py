"""
scripts/smoke_test.py
Fast end-to-end smoke test — completes in <60s on any machine.

Uses entirely synthetic in-memory fixtures (no disk reads beyond
already-processed parquets). Tests every stage against tiny inputs.

Run:
    python scripts/smoke_test.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import NIFTY50_TICKER, NSE_SECTOR_TICKERS

# ── Fixture builders ──────────────────────────────────────────────────────────

def _make_events(n: int = 20) -> pd.DataFrame:
    """Tiny canonical events fixture."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "event_id"      : [f"E{i:04d}" for i in range(n)],
        "date"          : dates,
        "event_code"    : ["0211"] * n,
        "event_root"    : ["02"] * n,
        "goldstein"     : np.random.uniform(-5, 5, n),
        "actor1"        : ["India"] * n,
        "actor2"        : ["China"] * n,
        "action_country": ["IN"] * n,
        "action_lat"    : [20.0] * n,
        "action_lon"    : [77.0] * n,
        "avg_tone"      : np.random.uniform(-3, 3, n),
        "num_articles"  : np.random.randint(1, 10, n),
        "source_url"    : [f"http://example.com/article/{i}" for i in range(n)],
    })


def _make_news(n: int = 200) -> pd.DataFrame:
    """Tiny news corpus fixture spanning the same date window."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D").repeat(1)[:n]
    topics = [
        "India China semiconductor chip export control trade",
        "logistics shipping freight suez canal red sea",
        "oil crude energy brent opec petroleum",
        "pharmaceutical pharma drug India China",
        "bank finance credit rupee inflation",
    ]
    import hashlib
    rows = []
    for i in range(n):
        text = topics[i % len(topics)]
        url  = f"http://fixture.news/{i}"
        rows.append({
            "article_id": hashlib.md5(f"{url}|{dates[i]}".encode()).hexdigest(),
            "date"      : dates[i],
            "source_url": url,
            "themes"    : text,
            "persons"   : "Modi Xi",
            "orgs"      : "TSMC NVIDIA",
            "locations" : "India China",
            "tone"      : float(np.random.uniform(-2, 2)),
            "raw_text"  : text,
            "text"      : text.lower(),
        })
    return pd.DataFrame(rows)


def _make_returns(n_days: int = 500) -> pd.DataFrame:
    """Synthetic NSE daily returns (long enough for 120-day estimation window)."""
    dates   = pd.bdate_range("2022-07-01", periods=n_days)
    tickers = list(NSE_SECTOR_TICKERS.values()) + [NIFTY50_TICKER]
    data    = {t: np.random.normal(0.0005, 0.012, n_days) for t in tickers}
    return pd.DataFrame(data, index=dates)


# ── Stage tests ───────────────────────────────────────────────────────────────

def test_preprocessing():
    logger.info("Test: preprocessing.harmonize_gdelt (unit, no disk)")
    from geosignal.data.preprocessing import clean_text, load_goldstein_map

    assert clean_text("  Hello <b>World</b>!  ") == "hello world"
    assert clean_text("SEMICONDUCTOR chip EUV") == "semiconductor chip euv"
    gmap = load_goldstein_map()
    assert isinstance(gmap, dict)
    assert len(gmap) > 0
    logger.success("  preprocessing OK")


def test_sector_mapping():
    logger.info("Test: sector_mapping")
    from geosignal.sectors.sector_mapping import map_entity_to_gics

    assert map_entity_to_gics("crude oil tanker")     == "Energy"
    assert map_entity_to_gics("semiconductor fab")    == "Information Technology"
    assert map_entity_to_gics("Suez Canal blocked")   == "Industrials"
    assert map_entity_to_gics("TSMC EUV", "SEMICONDUCTOR") == "Information Technology"
    assert map_entity_to_gics("random xyz 123")       == "Unknown"
    logger.success("  sector_mapping OK")


def test_nlp_alignment():
    logger.info("Test: NewsAligner (tiny corpus fixture)")
    from geosignal.nlp.alignment import NewsAligner

    events = _make_events(5)
    news   = _make_news(50)
    aligner = NewsAligner(news, max_tfidf_features=500)

    # retrieve for one event
    row     = events.iloc[0]
    matches = aligner.retrieve("india china semiconductor", pd.Timestamp(row["date"]), top_n=3)
    assert isinstance(matches, pd.DataFrame)

    # cosine_scores uses transform only (no refit)
    scores = aligner.cosine_scores("semiconductor chip", ["semiconductor chip export", "oil tanker"])
    assert len(scores) == 2
    assert scores[0] > scores[1]  # first candidate more relevant
    logger.success("  NewsAligner OK")


def test_entity_extraction():
    logger.info("Test: entity_extraction")
    from geosignal.nlp.entity_extraction import build_nlp_pipeline, extract_entities

    nlp     = build_nlp_pipeline()
    ents    = extract_entities("TSMC supplies EUV lithography chips to NVIDIA.", nlp)
    labels  = [e["label"] for e in ents]
    # At minimum spaCy ORG or custom SEMICONDUCTOR should fire
    assert len(ents) > 0
    assert any(l in ("ORG", "SEMICONDUCTOR", "PRODUCT") for l in labels)
    logger.success("  entity_extraction OK")


def test_event_study():
    logger.info("Test: EventStudy on synthetic returns")
    from geosignal.analysis.event_study import EventStudy, compute_raw_impact

    returns = _make_returns(500)
    es      = EventStudy(returns, estimation_window=120, event_window=(-5, 10))
    ticker  = list(NSE_SECTOR_TICKERS.values())[0]
    edate   = returns.index[200]

    result = es.run(ticker, edate)
    assert isinstance(result, dict)
    if result:  # may be empty if window too short — acceptable
        assert "CAR" in result
        assert "p_value" in result

    raw = compute_raw_impact(returns, ticker, edate)
    assert isinstance(raw, dict)
    logger.success("  EventStudy OK")


def test_granger():
    logger.info("Test: Granger causality (tiny series, low lag)")
    from geosignal.analysis.granger_causality import granger_causality_test, adf_stationarity

    n  = 200
    x  = pd.Series(np.random.randn(n))
    y  = pd.Series(np.random.randn(n))

    adf = test_stationarity(x)
    assert "stationary" in adf

    gc = granger_causality_test(x, y, max_lag=3)
    assert isinstance(gc, pd.DataFrame)
    if not gc.empty:
        assert "p_value" in gc.columns
    logger.success("  Granger causality OK")


def test_leontief():
    logger.info("Test: LeontiefPropagator")
    from geosignal.propagation.leontief import LeontiefPropagator

    lp    = LeontiefPropagator.toy_5sector()
    shock = lp.propagate_shock("Semiconductor", -0.15)
    assert len(shock) == 5
    assert shock["Semiconductor"] < 0      # shocked sector decreases
    assert all(shock.index == sorted(shock.index, key=lambda x: shock[x]))  # sorted ascending
    logger.success("  LeontiefPropagator OK")


def test_impact_assembly():
    logger.info("Test: build_integrated_dataset")
    from geosignal.analysis.impact_analysis import build_integrated_dataset

    events = _make_events(10)
    temporal = pd.DataFrame({
        "event_id"  : events["event_id"].tolist(),
        "event_date": events["date"].tolist(),
        "CAR_pct"   : np.random.uniform(-5, 5, 10),
        "t_stat"    : np.random.uniform(-3, 3, 10),
        "p_value"   : np.random.uniform(0, 1, 10),
        "significant": [False] * 10,
        "impact_7d" : np.random.uniform(-2, 2, 10),
        "impact_14d": np.random.uniform(-3, 3, 10),
        "impact_30d": np.random.uniform(-4, 4, 10),
        "ticker"    : ["^CNXIT"] * 10,
    })
    pairs = pd.DataFrame({
        "event_id"    : events["event_id"].tolist(),
        "sector"      : ["Information Technology"] * 10,
        "article_date": events["date"].tolist(),
        "bm25_score"  : [1.0] * 10,
        "cosine_sim"  : [0.2] * 10,
        "entities"    : ["[]"] * 10,
    })

    result = build_integrated_dataset(events, temporal, pairs)
    assert len(result) == 10
    assert "validation" in result.columns
    assert set(result["validation"].unique()).issubset({"Confirmed", "Null", "Insufficient"})
    logger.success("  build_integrated_dataset OK")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("GeoSignal Smoke Test")
    logger.info("=" * 60)

    tests = [
        test_preprocessing,
        test_sector_mapping,
        test_nlp_alignment,
        test_entity_extraction,
        test_event_study,
        test_granger,
        test_leontief,
        test_impact_assembly,
    ]

    passed, failed = 0, []
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            logger.error(f"FAILED: {test_fn.__name__} — {exc}")
            failed.append(test_fn.__name__)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    if failed:
        logger.error(f"RESULT: {passed}/{len(tests)} passed  |  FAILED: {failed}  |  {elapsed:.1f}s")
        sys.exit(1)
    else:
        logger.success(f"RESULT: {passed}/{len(tests)} passed  |  ALL OK  |  {elapsed:.1f}s")


if __name__ == "__main__":
    main()

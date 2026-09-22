"""tests/test_analysis.py — event study, Granger, Leontief, impact assembly"""
import numpy as np
import pandas as pd
import pytest

from config.settings import NSE_SECTOR_TICKERS, NIFTY50_TICKER
from geosignal.analysis.event_study import EventStudy, compute_raw_impact
from geosignal.analysis.granger_causality import granger_causality_test, adf_stationarity
from geosignal.analysis.impact_analysis import build_integrated_dataset, assign_validation_label
from geosignal.propagation.leontief import LeontiefPropagator


# ── EventStudy ────────────────────────────────────────────────────────────────

def test_event_study_returns_dict(synthetic_returns):
    ticker = list(NSE_SECTOR_TICKERS.values())[0]
    es     = EventStudy(synthetic_returns, estimation_window=120, event_window=(-5, 10))
    result = es.run(ticker, synthetic_returns.index[200])
    assert isinstance(result, dict)


def test_event_study_has_expected_keys(synthetic_returns):
    ticker = list(NSE_SECTOR_TICKERS.values())[0]
    es     = EventStudy(synthetic_returns, estimation_window=120, event_window=(-5, 10))
    result = es.run(ticker, synthetic_returns.index[200])
    if result:
        for key in ("CAR", "t_stat", "p_value", "significant", "alpha", "beta"):
            assert key in result


def test_event_study_missing_ticker(synthetic_returns):
    es = EventStudy(synthetic_returns)
    result = es.run("^NONEXISTENT", synthetic_returns.index[200])
    assert result == {}


def test_event_study_too_short_window(synthetic_returns):
    es     = EventStudy(synthetic_returns, estimation_window=120, event_window=(-5, 10))
    # Event at position 10 — estimation window will be < 30 days
    result = es.run(list(NSE_SECTOR_TICKERS.values())[0], synthetic_returns.index[10])
    assert result == {}


def test_compute_raw_impact(synthetic_returns):
    ticker = list(NSE_SECTOR_TICKERS.values())[0]
    result = compute_raw_impact(synthetic_returns, ticker, synthetic_returns.index[200])
    assert isinstance(result, dict)
    assert "impact_7d" in result


def test_compute_raw_impact_missing_ticker(synthetic_returns):
    result = compute_raw_impact(synthetic_returns, "^MISSING", synthetic_returns.index[200])
    assert result == {}


# ── Stationarity ──────────────────────────────────────────────────────────────

def test_stationarity_returns_dict():
    s   = pd.Series(np.random.randn(200))
    res = adf_stationarity(s)
    assert "stationary" in res
    assert "adf_stat"   in res
    assert "p_value"    in res
    assert "n_diffs"    in res


def test_stationarity_white_noise_is_stationary():
    rng = np.random.default_rng(0)
    s   = pd.Series(rng.standard_normal(300))
    res = adf_stationarity(s)
    assert res["stationary"] is True


# ── Granger causality ─────────────────────────────────────────────────────────

def test_granger_returns_dataframe():
    rng = np.random.default_rng(1)
    x   = pd.Series(rng.standard_normal(200))
    y   = pd.Series(rng.standard_normal(200))
    gc  = granger_causality_test(x, y, max_lag=3)
    assert isinstance(gc, pd.DataFrame)


def test_granger_insufficient_data():
    x  = pd.Series(np.random.randn(10))
    y  = pd.Series(np.random.randn(10))
    gc = granger_causality_test(x, y, max_lag=5)
    assert gc.empty


def test_granger_column_schema():
    rng = np.random.default_rng(2)
    x   = pd.Series(rng.standard_normal(200))
    y   = pd.Series(rng.standard_normal(200))
    gc  = granger_causality_test(x, y, max_lag=3)
    if not gc.empty:
        for col in ("lag", "F_stat", "p_value", "significant"):
            assert col in gc.columns


# ── LeontiefPropagator ────────────────────────────────────────────────────────

def test_leontief_toy_5sector():
    lp = LeontiefPropagator.toy_5sector()
    assert lp.n == 5


def test_leontief_shock_length():
    lp    = LeontiefPropagator.toy_5sector()
    shock = lp.propagate_shock("Semiconductor", -0.15)
    assert len(shock) == 5


def test_leontief_shock_sorted_ascending():
    lp    = LeontiefPropagator.toy_5sector()
    shock = lp.propagate_shock("Energy", -0.20)
    assert list(shock.values) == sorted(shock.values)


def test_leontief_shocked_sector_negative():
    lp    = LeontiefPropagator.toy_5sector()
    shock = lp.propagate_shock("Logistics", -0.10)
    assert shock["Logistics"] < 0


def test_leontief_unknown_sector_raises():
    lp = LeontiefPropagator.toy_5sector()
    with pytest.raises(ValueError, match="not in sector_labels"):
        lp.propagate_shock("Finance", -0.10)


def test_leontief_non_square_raises():
    with pytest.raises(ValueError, match="square"):
        LeontiefPropagator(np.eye(3)[:2], ["A", "B"])


# ── Impact analysis ───────────────────────────────────────────────────────────

def _make_minimal_events(n=8):
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "event_id"      : [f"E{i}" for i in range(n)],
        "date"          : dates,
        "event_code"    : ["0211"] * n,
        "event_root"    : ["02"] * n,
        "goldstein"     : np.zeros(n),
        "actor1"        : ["India"] * n,
        "actor2"        : ["China"] * n,
        "action_country": ["IN"] * n,
        "action_lat"    : [20.0] * n,
        "action_lon"    : [77.0] * n,
        "avg_tone"      : np.zeros(n),
        "num_articles"  : [1] * n,
        "source_url"    : [""] * n,
    })


def test_assign_validation_confirmed():
    row = pd.Series({"p_value": 0.01})
    assert assign_validation_label(row) == "Confirmed"


def test_assign_validation_null():
    row = pd.Series({"p_value": 0.5})
    assert assign_validation_label(row) == "Null"


def test_assign_validation_insufficient():
    row = pd.Series({"p_value": np.nan})
    assert assign_validation_label(row) == "Insufficient"


def test_build_integrated_dataset_schema():
    import json
    events   = _make_minimal_events(5)
    temporal = pd.DataFrame({
        "event_id"  : events["event_id"],
        "event_date": events["date"],
        "CAR_pct"   : np.zeros(5),
        "t_stat"    : np.zeros(5),
        "p_value"   : [0.01, 0.5, np.nan, 0.04, 0.8],
        "significant": [True, False, False, True, False],
        "impact_7d" : np.zeros(5),
        "impact_14d": np.zeros(5),
        "impact_30d": np.zeros(5),
        "ticker"    : ["^CNXIT"] * 5,
    })
    pairs = pd.DataFrame({
        "event_id"    : events["event_id"],
        "sector"      : ["Information Technology"] * 5,
        "article_date": events["date"],
        "bm25_score"  : [1.0] * 5,
        "cosine_sim"  : [0.1] * 5,
        "entities"    : [json.dumps([])] * 5,
    })
    result = build_integrated_dataset(events, temporal, pairs)
    assert len(result) == 5
    assert "validation" in result.columns
    assert set(result["validation"]).issubset({"Confirmed", "Null", "Insufficient"})
    # Confirmed where p < 0.05
    assert result[result["p_value"] == 0.01]["validation"].iloc[0] == "Confirmed"

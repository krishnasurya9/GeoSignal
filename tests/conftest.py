"""
tests/conftest.py  — shared pytest fixtures (tiny, in-memory only)
"""
import hashlib
import numpy as np
import pandas as pd
import pytest

from config.settings import NSE_SECTOR_TICKERS, NIFTY50_TICKER


@pytest.fixture
def tiny_events():
    n = 15
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "event_id"      : [f"E{i:03d}" for i in range(n)],
        "date"          : dates,
        "event_code"    : ["0211"] * n,
        "event_root"    : ["02"] * n,
        "goldstein"     : np.linspace(-5, 5, n),
        "actor1"        : ["India"] * n,
        "actor2"        : ["China"] * n,
        "action_country": ["IN"] * n,
        "action_lat"    : [20.0] * n,
        "action_lon"    : [77.0] * n,
        "avg_tone"      : np.zeros(n),
        "num_articles"  : [3] * n,
        "source_url"    : [f"http://x.com/{i}" for i in range(n)],
    })


@pytest.fixture
def tiny_news():
    topics = [
        "India China semiconductor chip export control trade policy",
        "logistics shipping freight suez canal red sea disruption",
        "oil crude energy brent opec petroleum supply",
        "pharmaceutical pharma drug India generic export",
        "bank finance credit rupee inflation monetary policy",
    ] * 20   # 100 articles
    dates = pd.date_range("2024-01-01", periods=len(topics), freq="D")
    rows = []
    for i, (text, date) in enumerate(zip(topics, dates)):
        url = f"http://fixture.test/{i}"
        rows.append({
            "article_id": hashlib.md5(f"{url}|{date}".encode()).hexdigest(),
            "date"      : date,
            "source_url": url,
            "themes"    : text,
            "persons"   : "Modi Xi",
            "orgs"      : "TSMC NVIDIA",
            "locations" : "India China",
            "tone"      : 0.0,
            "raw_text"  : text,
            "text"      : text.lower(),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_returns():
    """500 trading days of synthetic returns — enough for 120-day estimation window."""
    n       = 500
    dates   = pd.bdate_range("2022-07-01", periods=n)
    tickers = list(NSE_SECTOR_TICKERS.values()) + [NIFTY50_TICKER]
    rng     = np.random.default_rng(42)
    data    = {t: rng.normal(0.0005, 0.012, n) for t in tickers}
    return pd.DataFrame(data, index=dates)

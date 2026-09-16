"""
src/sector_mapping.py
Phase III: Sectoral Taxonomy + Outcome Data Architecture

Implements:
  1. Entity → GICS / NIC-2008 mapping
  2. NSE Nifty sectoral index download (yfinance)
  3. FRED macroeconomic indicator download

Usage:
    python src/sector_mapping.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_RAW, DATA_PROC, RESULTS_DIR,
    NSE_SECTOR_TICKERS, NIFTY50_TICKER,
    FRED_API_KEY, GICS_MAP_PATH, NIC_MAP_PATH,
)


# ── Entity → Sector Mapping ────────────────────────────────────────────────────

# Deterministic keyword → GICS sector mapping
# In production, replace with the full GICS taxonomy CSV
KEYWORD_TO_GICS = {
    # Energy
    "oil": "Energy", "crude": "Energy", "petroleum": "Energy",
    "natural gas": "Energy", "lng": "Energy", "petroline": "Energy",
    "brent": "Energy", "opec": "Energy",
    # Semiconductors / Technology
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
    "fertiliser": "Materials",
    # Materials / Critical Minerals
    "lithium": "Materials", "cobalt": "Materials", "rare earth": "Materials",
    "copper": "Materials", "aluminum": "Materials",
    # Healthcare
    "pharmaceutical": "Health Care", "pharma": "Health Care",
    # Automotive
    "automotive": "Consumer Discretionary", "vehicle": "Consumer Discretionary",
    "automobile": "Consumer Discretionary",
}


def map_entity_to_gics(entity_text: str) -> str:
    """
    Map an extracted entity string to its GICS sector via keyword lookup.

    Args:
        entity_text: Raw entity string from NER pipeline.

    Returns:
        GICS sector string, or 'Unknown' if no match found.
    """
    text_lower = entity_text.lower()
    for keyword, sector in KEYWORD_TO_GICS.items():
        if keyword in text_lower:
            return sector
    return "Unknown"


def enrich_pairs_with_sectors(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'sector' column to each event-news pair by mapping NER entities.

    Args:
        pairs_df: DataFrame with 'entities' column (JSON list of entity dicts).

    Returns:
        DataFrame with added 'sector' and 'mapped_entities' columns.
    """
    import json

    sectors = []
    mapped  = []

    for _, row in pairs_df.iterrows():
        try:
            entities = json.loads(row.get("entities", "[]"))
        except (json.JSONDecodeError, TypeError):
            entities = []

        row_sectors = []
        for ent in entities:
            if ent.get("label") in ("ORG", "GPE", "PRODUCT",
                                     "SEMICONDUCTOR", "ENERGY",
                                     "LOGISTICS", "AGRICULTURE", "CRITICAL_MINERAL"):
                s = map_entity_to_gics(ent["text"])
                if s != "Unknown":
                    row_sectors.append(s)

        # Take most common sector; fall back to 'Unknown'
        if row_sectors:
            sector = max(set(row_sectors), key=row_sectors.count)
        else:
            sector = "Unknown"

        sectors.append(sector)
        mapped.append(row_sectors)

    pairs_df = pairs_df.copy()
    pairs_df["sector"]          = sectors
    pairs_df["all_sectors"]     = [str(s) for s in mapped]
    return pairs_df


# ── NSE Sectoral Index Download ───────────────────────────────────────────────

def download_nse_indices(start: str, end: str) -> pd.DataFrame:
    """
    Download NSE sectoral indices + Nifty50 (broad market) via yfinance.

    Args:
        start: Start date string 'YYYY-MM-DD'
        end:   End date string   'YYYY-MM-DD'

    Returns:
        Multi-column DataFrame: columns = ticker symbols, rows = trading days.
        Values are daily closing prices.
    """
    tickers_to_dl = list(NSE_SECTOR_TICKERS.values()) + [NIFTY50_TICKER]

    logger.info(f"Downloading {len(tickers_to_dl)} NSE indices from {start} to {end}...")
    raw = yf.download(
        tickers_to_dl,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if "Close" in raw.columns.get_level_values(0):
        prices = raw["Close"]
    else:
        prices = raw  # fallback for single ticker

    out_path = DATA_RAW / "nse_sector_prices.parquet"
    prices.to_parquet(out_path)
    logger.success(f"NSE prices saved → {out_path}  shape={prices.shape}")
    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price DataFrame to percentage daily returns.
    """
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    logger.info(f"Daily returns computed: {returns.shape}")
    return returns


# ── FRED Macroeconomic Data ───────────────────────────────────────────────────

def download_fred_series(series_ids: dict, start: str, end: str) -> pd.DataFrame:
    """
    Download FRED macroeconomic time-series data.

    Args:
        series_ids: dict of {label: FRED_series_id}, e.g. {"WTI_Oil": "DCOILWTICO"}
        start:      Start date 'YYYY-MM-DD'
        end:        End date   'YYYY-MM-DD'

    Returns:
        DataFrame with FRED series as columns.
    """
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
    except ImportError:
        logger.error("fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()

    frames = {}
    for label, sid in series_ids.items():
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            frames[label] = s
            logger.info(f"FRED: {label} ({sid}) → {len(s)} observations")
        except Exception as e:
            logger.warning(f"FRED: Failed to fetch {label} ({sid}): {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    out_path = DATA_RAW / "fred_macro.parquet"
    df.to_parquet(out_path)
    logger.success(f"FRED data saved → {out_path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

# Default FRED series relevant to Indian macro + commodity markets
DEFAULT_FRED_SERIES = {
    "WTI_Oil"   : "DCOILWTICO",      # WTI crude price ($/bbl)
    "Brent_Oil" : "DCOILBRENTEU",   # Brent crude price
    "US_CPI"    : "CPIAUCSL",        # US CPI (inflation proxy)
    "USD_INR"   : "DEXINUS",         # USD/INR exchange rate
    # Note: BDIY (Baltic Dry Index) was discontinued on FRED — removed
}


def main():
    start = "2022-01-01"
    end   = "2024-12-31"

    # 1. Download NSE indices
    prices  = download_nse_indices(start, end)
    returns = compute_daily_returns(prices)
    returns.to_parquet(DATA_PROC / "nse_returns.parquet")

    # 2. Download FRED macro indicators
    if FRED_API_KEY:
        download_fred_series(DEFAULT_FRED_SERIES, start, end)
    else:
        logger.warning("FRED_API_KEY not set. Skipping FRED download.")

    # 3. Enrich event-news pairs with sector mapping
    pairs_path = DATA_PROC / "event_news_pairs.parquet"
    if pairs_path.exists():
        pairs_df = pd.read_parquet(pairs_path)
        enriched = enrich_pairs_with_sectors(pairs_df)
        out_path = DATA_PROC / "event_news_sector.parquet"
        enriched.to_parquet(out_path, index=False)
        logger.success(f"Sector-enriched pairs saved → {out_path}")
    else:
        logger.warning("Run nlp_pipeline.py first to generate event_news_pairs.parquet")


if __name__ == "__main__":
    main()

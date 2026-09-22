"""
geosignal.markets.market_data
NSE sectoral index download (yfinance) + FRED macroeconomic data.

Outputs:
    data/raw/nse_sector_prices.parquet
    data/processed/nse_returns.parquet
    data/raw/fred_macro.parquet   (requires FRED_API_KEY)
"""

import pandas as pd
import yfinance as yf
from loguru import logger

from config.settings import (
    DATA_RAW, DATA_PROC,
    NSE_SECTOR_TICKERS, NIFTY50_TICKER,
    FRED_API_KEY,
)

# ── FRED series of interest ───────────────────────────────────────────────────
DEFAULT_FRED_SERIES = {
    "WTI_Oil"  : "DCOILWTICO",    # WTI crude ($/bbl)
    "Brent_Oil": "DCOILBRENTEU",  # Brent crude
    "US_CPI"   : "CPIAUCSL",      # US CPI
    "USD_INR"  : "DEXINUS",       # USD/INR exchange rate
}


def download_nse_indices(start: str, end: str) -> pd.DataFrame:
    """
    Download NSE sectoral indices + Nifty50 via yfinance.
    Saves prices to data/raw/nse_sector_prices.parquet.

    Args:
        start: 'YYYY-MM-DD'
        end:   'YYYY-MM-DD'

    Returns:
        Daily closing price DataFrame (columns = tickers).
    """
    tickers = list(NSE_SECTOR_TICKERS.values()) + [NIFTY50_TICKER]
    logger.info(f"Downloading {len(tickers)} NSE indices: {start} → {end}")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw

    out = DATA_RAW / "nse_sector_prices.parquet"
    prices.to_parquet(out)
    logger.success(f"NSE prices → {out}  {prices.shape}")
    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage daily returns from a price DataFrame."""
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    logger.info(f"Daily returns: {returns.shape}")
    return returns


def download_fred_series(series_ids: dict | None = None,
                         start: str = "2022-01-01",
                         end:   str = "2024-12-31") -> pd.DataFrame:
    """
    Download FRED macroeconomic series. Requires FRED_API_KEY in .env.

    Args:
        series_ids: {label: FRED_id} dict. Defaults to DEFAULT_FRED_SERIES.
        start / end: date strings.

    Returns:
        DataFrame of FRED series (columns = labels).
    """
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping FRED download.")
        return pd.DataFrame()

    try:
        from fredapi import Fred
    except ImportError:
        logger.error("fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()

    fred     = Fred(api_key=FRED_API_KEY)
    ids      = series_ids or DEFAULT_FRED_SERIES
    frames   = {}
    for label, sid in ids.items():
        try:
            frames[label] = fred.get_series(sid, observation_start=start, observation_end=end)
            logger.info(f"FRED {label} ({sid}): {len(frames[label])} obs")
        except Exception as exc:
            logger.warning(f"FRED {label} failed: {exc}")

    if not frames:
        return pd.DataFrame()

    df  = pd.DataFrame(frames)
    out = DATA_RAW / "fred_macro.parquet"
    df.to_parquet(out)
    logger.success(f"FRED data → {out}")
    return df


def main():
    start, end = "2022-01-01", "2024-12-31"
    prices  = download_nse_indices(start, end)
    returns = compute_daily_returns(prices)
    returns.to_parquet(DATA_PROC / "nse_returns.parquet")
    logger.success(f"NSE returns → data/processed/nse_returns.parquet")
    download_fred_series(start=start, end=end)


if __name__ == "__main__":
    main()

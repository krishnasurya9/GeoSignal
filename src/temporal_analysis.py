"""
src/temporal_analysis.py
Phase IV: Temporal Alignment + Impact Measurement

Implements:
  1. Raw percentage-change impact at 7 / 14 / 30-day horizons
  2. Event Study Methodology (MacKinlay 1997):
       - Market model (OLS) over estimation window
       - Abnormal Returns (AR) during event window
       - Cumulative Abnormal Returns (CAR) + t-test significance
  3. Granger Causality Test (with ADF pre-check + Toda-Yamamoto option)
  4. Basic VAR model construction

Usage:
    python src/temporal_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_PROC, RESULTS_DIR,
    ESTIMATION_WINDOW, EVENT_WINDOW, IMPACT_WINDOWS,
    GRANGER_MAX_LAGS, ADF_SIGNIFICANCE, NSE_SECTOR_TICKERS, NIFTY50_TICKER,
)


# ── Raw Impact Calculation ────────────────────────────────────────────────────

def compute_raw_impact(returns: pd.DataFrame,
                       sector_ticker: str,
                       event_date: pd.Timestamp,
                       windows: list[int] = IMPACT_WINDOWS) -> dict:
    """
    Calculate raw cumulative percentage change over each temporal window.

    Args:
        returns:       DataFrame of daily percentage returns (columns = tickers).
        sector_ticker: Ticker of the sector to evaluate.
        event_date:    Event date.
        windows:       List of days post-event to evaluate (e.g. [7, 14, 30]).

    Returns:
        Dict: {window: cumulative_return_pct}
    """
    if sector_ticker not in returns.columns:
        logger.warning(f"Ticker {sector_ticker} not in returns DataFrame.")
        return {}

    series = returns[sector_ticker].dropna()
    try:
        start_idx = series.index.searchsorted(event_date)
    except Exception:
        return {}

    impact = {}
    for w in windows:
        end_idx = start_idx + w
        if end_idx >= len(series):
            impact[f"impact_{w}d"] = np.nan
        else:
            window_returns  = series.iloc[start_idx:end_idx]
            cumulative      = (1 + window_returns).prod() - 1
            impact[f"impact_{w}d"] = round(cumulative * 100, 4)

    return impact


# ── Event Study: MacKinlay (1997) ─────────────────────────────────────────────

class EventStudy:
    """
    Classic market-model event study.

    Steps:
      1. Estimate OLS(sector_return ~ market_return) over estimation window.
      2. Predict expected return during event window.
      3. Compute AR = actual - expected.
      4. Cumulate to CAR; test significance via t-test.
    """

    def __init__(self, returns: pd.DataFrame,
                 market_ticker: str = NIFTY50_TICKER,
                 estimation_window: int = ESTIMATION_WINDOW,
                 event_window: tuple = EVENT_WINDOW):
        self.returns           = returns
        self.market_ticker     = market_ticker
        self.estimation_window = estimation_window
        self.pre_event_days    = event_window[0]   # negative = before event
        self.post_event_days   = event_window[1]

    def run(self, sector_ticker: str, event_date: pd.Timestamp) -> dict:
        """
        Run the event study for one sector + event.

        Returns:
            Dict with keys: alpha, beta, CAR, t_stat, p_value, significant,
                            ar_series (pd.Series)
        """
        if self.market_ticker not in self.returns.columns:
            logger.error(f"Market ticker {self.market_ticker} missing from returns.")
            return {}
        if sector_ticker not in self.returns.columns:
            logger.warning(f"Sector ticker {sector_ticker} missing.")
            return {}

        sector_ret = self.returns[sector_ticker].dropna()
        market_ret = self.returns[self.market_ticker].reindex(sector_ret.index).dropna()
        common_idx = sector_ret.index.intersection(market_ret.index)

        sector_ret = sector_ret.loc[common_idx]
        market_ret = market_ret.loc[common_idx]

        try:
            event_pos = common_idx.searchsorted(event_date)
        except Exception as e:
            logger.error(f"Event date lookup failed: {e}")
            return {}

        # ── Estimation window ─────────────────────────────────────────────────
        est_start = max(0, event_pos - self.estimation_window + self.pre_event_days)
        est_end   = event_pos + self.pre_event_days

        if est_end - est_start < 30:
            logger.warning("Estimation window too short (<30 days). Skipping.")
            return {}

        y    = sector_ret.iloc[est_start:est_end].values
        X    = sm.add_constant(market_ret.iloc[est_start:est_end].values)
        ols  = sm.OLS(y, X).fit()
        alpha, beta = ols.params[0], ols.params[1]
        resid_var   = np.var(ols.resid, ddof=2)

        # ── Event window ──────────────────────────────────────────────────────
        ev_start = event_pos + self.pre_event_days
        ev_end   = min(len(common_idx), event_pos + self.post_event_days)

        actual_ret   = sector_ret.iloc[ev_start:ev_end]
        expected_ret = alpha + beta * market_ret.iloc[ev_start:ev_end]
        ar_series    = actual_ret - expected_ret

        car = ar_series.sum()
        n   = len(ar_series)

        if n == 0 or resid_var == 0:
            return {}

        # t-statistic for CAR significance
        se     = np.sqrt(n * resid_var)
        t_stat = car / se
        p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

        return {
            "alpha"      : round(alpha, 6),
            "beta"       : round(beta, 6),
            "CAR"        : round(car * 100, 4),   # as percentage
            "t_stat"     : round(t_stat, 4),
            "p_value"    : round(p_val, 4),
            "significant": p_val < 0.05,
            "n_days"     : n,
            "ar_series"  : ar_series,
        }


# ── ADF Stationarity Test ─────────────────────────────────────────────────────

def test_stationarity(series: pd.Series, significance: float = ADF_SIGNIFICANCE) -> dict:
    """
    Augmented Dickey-Fuller test for unit roots.

    Returns:
        Dict: {stationary: bool, adf_stat: float, p_value: float, n_diffs: int}
    """
    result = adfuller(series.dropna(), autolag="AIC")
    is_stationary = result[1] < significance

    n_diffs = 0
    temp = series.dropna()
    while not is_stationary and n_diffs < 3:
        temp = temp.diff().dropna()
        result = adfuller(temp, autolag="AIC")
        is_stationary = result[1] < significance
        n_diffs += 1

    return {
        "stationary": result[1] < significance,
        "adf_stat"  : round(result[0], 4),
        "p_value"   : round(result[1], 4),
        "n_diffs"   : n_diffs,
    }


# ── Granger Causality ─────────────────────────────────────────────────────────

def granger_causality_test(geo_risk: pd.Series,
                            sector_return: pd.Series,
                            max_lag: int = GRANGER_MAX_LAGS,
                            toda_yamamoto: bool = True) -> pd.DataFrame:
    """
    Test whether geo_risk Granger-causes sector_return.

    Args:
        geo_risk:       Time series of geopolitical risk intensity (Goldstein-based).
        sector_return:  Time series of sectoral returns.
        max_lag:        Maximum lag order to test.
        toda_yamamoto:  If True, apply Toda-Yamamoto correction for non-stationarity.

    Returns:
        DataFrame with lag, F-statistic, p-value, and significance flag.
    """
    aligned = pd.DataFrame({
        "geo_risk"     : geo_risk,
        "sector_return": sector_return,
    }).dropna()

    if len(aligned) < max_lag * 4:
        logger.warning("Insufficient observations for Granger test.")
        return pd.DataFrame()

    # ADF pre-check
    adf_geo    = test_stationarity(aligned["geo_risk"])
    adf_sector = test_stationarity(aligned["sector_return"])

    d_max = max(adf_geo["n_diffs"], adf_sector["n_diffs"])

    if toda_yamamoto and d_max > 0:
        # Toda-Yamamoto: augment max_lag by integration order
        effective_max_lag = max_lag + d_max
        logger.info(f"Toda-Yamamoto: effective max lag = {effective_max_lag} (d_max={d_max})")
    else:
        effective_max_lag = max_lag
        # Use differenced series
        for _ in range(d_max):
            aligned = aligned.diff().dropna()

    test_data = aligned[["sector_return", "geo_risk"]].values

    try:
        results = grangercausalitytests(test_data, maxlag=effective_max_lag)
    except Exception as e:
        logger.error(f"Granger test failed: {e}")
        return pd.DataFrame()

    records = []
    for lag, test_result in results.items():
        f_test = test_result[0].get("ssr_ftest", (None, None, None, None))
        records.append({
            "lag"      : lag,
            "F_stat"   : round(f_test[0], 4) if f_test[0] else None,
            "p_value"  : round(f_test[1], 4) if f_test[1] else None,
            "df_denom" : f_test[2] if f_test[2] else None,
            "significant": f_test[1] < ADF_SIGNIFICANCE if f_test[1] else False,
        })

    return pd.DataFrame(records)


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    returns_path       = DATA_PROC / "nse_returns.parquet"
    pairs_path         = DATA_PROC / "event_news_sector.parquet"
    canonical_ev_path  = DATA_PROC / "processed_events.parquet"

    missing = [p for p in [returns_path, pairs_path, canonical_ev_path] if not p.exists()]
    if missing:
        logger.error(f"Missing files: {[str(p) for p in missing]}. "
                     "Run sector_mapping.py and preprocessing.py first.")
        return

    returns     = pd.read_parquet(returns_path)
    pairs       = pd.read_parquet(pairs_path)
    canon_evs   = pd.read_parquet(canonical_ev_path)[["event_id", "date"]].rename(
                      columns={"date": "event_date"})

    # ── Join pairs → canonical events to get the TRUE geopolitical event date ──
    # BUG FIX: pairs only carry article_date (news publication date).
    # The event study must be anchored to event_date (when the geopolitical
    # event actually occurred), NOT when the news article was published.
    events = pairs.merge(canon_evs, on="event_id", how="left")

    n_missing_dates = events["event_date"].isna().sum()
    if n_missing_dates > 0:
        logger.warning(f"{n_missing_dates} pairs have no matching canonical event date — skipping.")
    events = events.dropna(subset=["event_date"])

    es      = EventStudy(returns)
    records = []

    for _, event in events.iterrows():
        # Use canonical geopolitical event date — not article_date
        event_date = pd.Timestamp(event["event_date"])
        sector     = event.get("sector", "Unknown")

        # Find matching NSE ticker
        ticker = next(
            (v for k, v in NSE_SECTOR_TICKERS.items() if k.lower() in sector.lower()),
            None
        )
        if not ticker:
            continue

        raw_impact = compute_raw_impact(returns, ticker, event_date)
        ev_study   = es.run(ticker, event_date)

        rec = {
            "event_id"    : event["event_id"],
            "event_date"  : event_date,          # canonical geopolitical event date
            "article_date": event.get("article_date"),  # kept for traceability
            "sector"      : sector,
            "ticker"      : ticker,
            "CAR_pct"     : ev_study.get("CAR"),
            "t_stat"      : ev_study.get("t_stat"),
            "p_value"     : ev_study.get("p_value"),
            "significant" : ev_study.get("significant"),
            **raw_impact,
        }
        records.append(rec)

    results_df = pd.DataFrame(records)
    out_path   = DATA_PROC / "temporal_results.parquet"
    results_df.to_parquet(out_path, index=False)
    logger.success(f"Temporal analysis complete → {out_path} ({len(results_df)} records)")
    print(results_df.head(10).to_string())


if __name__ == "__main__":
    main()

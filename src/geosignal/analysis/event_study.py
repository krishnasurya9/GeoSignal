"""
geosignal.analysis.event_study
MacKinlay (1997) market-model event study.

Computes:
  - OLS market model (alpha, beta) over estimation window
  - Abnormal Returns (AR) during event window
  - Cumulative Abnormal Returns (CAR)
  - t-statistic and p-value for CAR significance

Also provides compute_raw_impact() for simple cumulative-return windows.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy import stats

from config.settings import (
    ESTIMATION_WINDOW, EVENT_WINDOW, IMPACT_WINDOWS,
    NIFTY50_TICKER,
)


def compute_raw_impact(returns: pd.DataFrame,
                       sector_ticker: str,
                       event_date: pd.Timestamp,
                       windows: list[int] = IMPACT_WINDOWS) -> dict:
    """
    Cumulative percentage change over each post-event horizon.

    Returns:
        {impact_Xd: float} for each window X in windows.
    """
    if sector_ticker not in returns.columns:
        logger.warning(f"Ticker {sector_ticker} not in returns.")
        return {}

    series = returns[sector_ticker].dropna()
    try:
        start_idx = series.index.searchsorted(event_date)
    except Exception:
        return {}

    result = {}
    for w in windows:
        end_idx = start_idx + w
        if end_idx >= len(series):
            result[f"impact_{w}d"] = np.nan
        else:
            cum = (1 + series.iloc[start_idx:end_idx]).prod() - 1
            result[f"impact_{w}d"] = round(float(cum) * 100, 4)
    return result


class EventStudy:
    """
    Classic MacKinlay (1997) market-model event study.

    Steps:
      1. Estimate OLS(sector ~ market) over the estimation window.
      2. Compute expected returns in the event window.
      3. AR = actual − expected; CAR = cumsum(AR).
      4. t-test on CAR.
    """

    def __init__(self, returns: pd.DataFrame,
                 market_ticker: str = NIFTY50_TICKER,
                 estimation_window: int = ESTIMATION_WINDOW,
                 event_window: tuple = EVENT_WINDOW):
        self.returns           = returns
        self.market_ticker     = market_ticker
        self.estimation_window = estimation_window
        self.pre_event_days    = event_window[0]
        self.post_event_days   = event_window[1]

    def run(self, sector_ticker: str, event_date: pd.Timestamp) -> dict:
        """
        Run one event study.

        Returns dict with keys:
            alpha, beta, CAR, t_stat, p_value, significant, n_days, ar_series
        Returns empty dict if data is insufficient.
        """
        if self.market_ticker not in self.returns.columns:
            logger.error(f"Market ticker {self.market_ticker} missing.")
            return {}
        if sector_ticker not in self.returns.columns:
            logger.warning(f"Sector ticker {sector_ticker} missing.")
            return {}

        sec = self.returns[sector_ticker].dropna()
        mkt = self.returns[self.market_ticker].reindex(sec.index).dropna()
        idx = sec.index.intersection(mkt.index)
        sec, mkt = sec.loc[idx], mkt.loc[idx]

        try:
            ev_pos = idx.searchsorted(event_date)
        except Exception as exc:
            logger.error(f"Event date lookup failed: {exc}")
            return {}

        # Estimation window
        est_start = max(0, ev_pos - self.estimation_window + self.pre_event_days)
        est_end   = ev_pos + self.pre_event_days
        if est_end - est_start < 30:
            logger.warning("Estimation window < 30 trading days — skipping.")
            return {}

        y    = sec.iloc[est_start:est_end].values
        X    = sm.add_constant(mkt.iloc[est_start:est_end].values)
        ols  = sm.OLS(y, X).fit()
        alpha, beta = ols.params[0], ols.params[1]
        resid_var   = float(np.var(ols.resid, ddof=2))

        # Event window
        ev_start = ev_pos + self.pre_event_days
        ev_end   = min(len(idx), ev_pos + self.post_event_days)
        actual   = sec.iloc[ev_start:ev_end]
        expected = alpha + beta * mkt.iloc[ev_start:ev_end]
        ar       = actual - expected

        car = float(ar.sum())
        n   = len(ar)
        if n == 0 or resid_var == 0:
            return {}

        se     = float(np.sqrt(n * resid_var))
        t_stat = car / se
        p_val  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2)))

        return {
            "alpha"      : round(alpha, 6),
            "beta"       : round(beta, 6),
            "CAR"        : round(car * 100, 4),
            "t_stat"     : round(t_stat, 4),
            "p_value"    : round(p_val, 4),
            "significant": p_val < 0.05,
            "n_days"     : n,
            "ar_series"  : ar,
        }

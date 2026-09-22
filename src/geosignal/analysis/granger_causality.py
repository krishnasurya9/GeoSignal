"""
geosignal.analysis.granger_causality
ADF stationarity testing + Toda-Yamamoto Granger causality.

IMPORTANT (L-007, DEC-007):
  Granger causality tests predictive precedence only.
  Always describe results as "geo_risk Granger-predicts sector_return"
  — never as "geo_risk causes sector_return".
"""

import pandas as pd
import numpy as np
from loguru import logger
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from config.settings import GRANGER_MAX_LAGS, ADF_SIGNIFICANCE


def adf_stationarity(series: pd.Series,
                     significance: float = ADF_SIGNIFICANCE) -> dict:
    """
    Augmented Dickey-Fuller test for unit roots. Iteratively differences up to d=3.

    Returns:
        {stationary, adf_stat, p_value, n_diffs}
    """
    s = series.dropna()
    result        = adfuller(s, autolag="AIC")
    is_stationary = result[1] < significance
    n_diffs       = 0

    while not is_stationary and n_diffs < 3:
        s      = s.diff().dropna()
        result = adfuller(s, autolag="AIC")
        is_stationary = result[1] < significance
        n_diffs += 1

    return {
        "stationary": bool(result[1] < significance),
        "adf_stat"  : round(float(result[0]), 4),
        "p_value"   : round(float(result[1]), 4),
        "n_diffs"   : n_diffs,
    }


def granger_causality_test(geo_risk: pd.Series,
                            sector_return: pd.Series,
                            max_lag: int = GRANGER_MAX_LAGS,
                            toda_yamamoto: bool = True) -> pd.DataFrame:
    """
    Test whether geo_risk Granger-predicts sector_return.

    Applies ADF pre-check. If non-stationary and toda_yamamoto=True,
    augments max_lag by integration order (Toda-Yamamoto correction).
    Guards: requires > 4 × max_lag observations (DEC-007).

    Args:
        geo_risk:       Daily geopolitical risk series.
        sector_return:  Daily sector return series.
        max_lag:        Maximum lag order.
        toda_yamamoto:  Apply Toda-Yamamoto correction.

    Returns:
        DataFrame: {lag, F_stat, p_value, df_denom, significant}
        Empty DataFrame if insufficient data or test fails.
    """
    aligned = pd.DataFrame({
        "geo_risk"     : geo_risk,
        "sector_return": sector_return,
    }).dropna()

    if len(aligned) < max_lag * 4:
        logger.warning(
            f"Insufficient observations ({len(aligned)}) for Granger at lag={max_lag}. "
            "Expand dataset to 2022-2024 (L-009)."
        )
        return pd.DataFrame()

    adf_geo    = adf_stationarity(aligned["geo_risk"])
    adf_sector = adf_stationarity(aligned["sector_return"])
    d_max      = max(adf_geo["n_diffs"], adf_sector["n_diffs"])

    if toda_yamamoto and d_max > 0:
        effective_lag = max_lag + d_max
        logger.info(f"Toda-Yamamoto: effective max lag = {effective_lag} (d_max={d_max})")
    else:
        effective_lag = max_lag
        for _ in range(d_max):
            aligned = aligned.diff().dropna()

    try:
        gc_results = grangercausalitytests(
            aligned[["sector_return", "geo_risk"]].values,
            maxlag=effective_lag,
            verbose=False,
        )
    except Exception as exc:
        logger.error(f"Granger test failed: {exc}")
        return pd.DataFrame()

    records = []
    for lag, res in gc_results.items():
        f = res[0].get("ssr_ftest", (None, None, None, None))
        records.append({
            "lag"       : lag,
            "F_stat"    : round(f[0], 4) if f[0] is not None else None,
            "p_value"   : round(f[1], 4) if f[1] is not None else None,
            "df_denom"  : f[2],
            "significant": (f[1] < ADF_SIGNIFICANCE) if f[1] is not None else False,
        })

    return pd.DataFrame(records)

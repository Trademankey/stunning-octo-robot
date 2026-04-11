"""
Statistical & distributional features.

Rolling z-score, skewness, kurtosis, ADF stationarity, entropy,
autocorrelation, variance ratio, log returns moments.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import structlog
from numba import njit
from scipy import stats as sp_stats

log = structlog.get_logger(__name__)

WINDOWS = [14, 30, 60, 120]


def compute_statistical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add ~20 statistical features to the DataFrame."""
    close = df["close"].to_numpy().astype(np.float64)
    log_ret = np.log(close[1:] / (close[:-1] + 1e-15))
    log_ret = np.concatenate([[0.0], log_ret])

    features = {}

    for w in WINDOWS:
        suffix = f"_{w}"

        # Rolling z-score
        features[f"zscore{suffix}"] = _rolling_zscore(close, w)

        # Rolling skewness
        features[f"skew{suffix}"] = _rolling_stat(log_ret, w, "skew")

        # Rolling kurtosis
        features[f"kurt{suffix}"] = _rolling_stat(log_ret, w, "kurt")

        # Rolling standard deviation of returns
        features[f"ret_std{suffix}"] = _rolling_stat(log_ret, w, "std")

    # Log returns
    features["log_return"] = log_ret
    features["log_return_abs"] = np.abs(log_ret)

    # ADF test statistic (rolling, expensive — only for longer window)
    features["adf_stat_60"] = _rolling_adf(close, 60)

    # Entropy of returns
    features["entropy_30"] = _rolling_entropy(log_ret, 30)
    features["entropy_60"] = _rolling_entropy(log_ret, 60)

    # Autocorrelation lag-1
    features["autocorr_1"] = _rolling_autocorr(log_ret, 30, lag=1)
    features["autocorr_5"] = _rolling_autocorr(log_ret, 30, lag=5)

    # Variance ratio (mean-reversion / momentum indicator)
    features["var_ratio_5"] = _variance_ratio(log_ret, period=5)

    # Realised volatility (5-min proxied by rolling)
    features["realised_vol_14"] = _rolling_stat(log_ret, 14, "std") * np.sqrt(252 * 24 * 4)

    # Parkinson volatility estimator
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    features["parkinson_vol"] = _parkinson_vol(high, low, 14)

    # Add to DataFrame
    for name, arr in features.items():
        if len(arr) == len(df):
            df = df.with_columns(pl.Series(name=name, values=arr))

    return df


# ── Numba-accelerated helpers ────────────────────────

@njit(cache=True)
def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        mu = np.mean(w)
        sigma = np.std(w)
        if sigma > 1e-15:
            out[i] = (arr[i] - mu) / sigma
    return out


@njit(cache=True)
def _parkinson_vol(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    factor = 1.0 / (4.0 * np.log(2.0))
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            hl = np.log(high[j] / (low[j] + 1e-15))
            s += hl * hl
        out[i] = np.sqrt(factor * s / window)
    return out


def _rolling_stat(arr: np.ndarray, window: int, stat_type: str) -> np.ndarray:
    """Compute rolling statistic using numpy."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        if stat_type == "skew":
            out[i] = float(sp_stats.skew(w, nan_policy="omit"))
        elif stat_type == "kurt":
            out[i] = float(sp_stats.kurtosis(w, nan_policy="omit"))
        elif stat_type == "std":
            out[i] = np.std(w, ddof=1) if len(w) > 1 else 0.0
    return out


def _rolling_adf(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling ADF test statistic (computationally heavy)."""
    from statsmodels.tsa.stattools import adfuller
    n = len(arr)
    out = np.full(n, np.nan)
    step = max(1, window // 4)  # Compute every N-th bar for speed
    for i in range(window - 1, n, step):
        w = arr[i - window + 1: i + 1]
        try:
            result = adfuller(w, maxlag=5, autolag=None)
            val = result[0]
            # Fill forward
            for j in range(i, min(i + step, n)):
                out[j] = val
        except Exception:
            pass
    # Forward-fill
    last = np.nan
    for i in range(n):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _rolling_entropy(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Shannon entropy of discretised returns."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        # Discretise into bins
        counts, _ = np.histogram(w, bins=10)
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 0]
        out[i] = -np.sum(probs * np.log2(probs))
    return out


def _rolling_autocorr(arr: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    """Rolling autocorrelation at given lag."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window + lag - 1, n):
        w = arr[i - window + 1: i + 1]
        if len(w) > lag + 1:
            c = np.corrcoef(w[lag:], w[:-lag])
            out[i] = c[0, 1] if c.shape == (2, 2) else 0.0
    return out


def _variance_ratio(arr: np.ndarray, period: int = 5) -> np.ndarray:
    """Variance ratio test: VR(q) = Var(q-period returns) / (q * Var(1-period returns))."""
    n = len(arr)
    out = np.full(n, np.nan)
    window = period * 10
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        var1 = np.var(w, ddof=1) if len(w) > 1 else 1e-10
        q_ret = np.array([np.sum(w[j: j + period]) for j in range(0, len(w) - period, period)])
        varq = np.var(q_ret, ddof=1) if len(q_ret) > 1 else 1e-10
        out[i] = varq / (period * var1 + 1e-10)
    return out

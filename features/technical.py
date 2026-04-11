"""
Technical analysis features — TA-Lib + pandas-ta.

Generates: SuperTrend, Ichimoku, Bollinger, MACD, RSI, ADX, ATR,
Stochastic, OBV, MFI, CMF, Hurst exponent, Fractal Dimension, VWAP.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import polars as pl
import structlog

log = structlog.get_logger(__name__)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    log.warning("TA-Lib not installed — using pandas-ta fallback")

try:
    import pandas_ta as pta
    HAS_PTA = True
except ImportError:
    HAS_PTA = False


def compute_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add 40+ technical indicator columns to an OHLCV DataFrame.
    Input must have columns: open, high, low, close, volume.
    """
    pdf = df.to_pandas()
    o, h, l, c, v = pdf["open"], pdf["high"], pdf["low"], pdf["close"], pdf["volume"]

    features: Dict[str, np.ndarray] = {}

    # ── Trend ────────────────────────────────────────
    if HAS_TALIB:
        features["sma_10"] = talib.SMA(c, timeperiod=10)
        features["sma_20"] = talib.SMA(c, timeperiod=20)
        features["sma_50"] = talib.SMA(c, timeperiod=50)
        features["ema_12"] = talib.EMA(c, timeperiod=12)
        features["ema_26"] = talib.EMA(c, timeperiod=26)
        features["ema_50"] = talib.EMA(c, timeperiod=50)

        macd, macd_signal, macd_hist = talib.MACD(c, 12, 26, 9)
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["macd_hist"] = macd_hist

        features["adx"] = talib.ADX(h, l, c, timeperiod=14)
        features["plus_di"] = talib.PLUS_DI(h, l, c, timeperiod=14)
        features["minus_di"] = talib.MINUS_DI(h, l, c, timeperiod=14)
    else:
        features["sma_10"] = c.rolling(10).mean().values
        features["sma_20"] = c.rolling(20).mean().values
        features["sma_50"] = c.rolling(50).mean().values
        features["ema_12"] = c.ewm(span=12).mean().values
        features["ema_26"] = c.ewm(span=26).mean().values
        features["ema_50"] = c.ewm(span=50).mean().values

    # ── Momentum ─────────────────────────────────────
    if HAS_TALIB:
        features["rsi_14"] = talib.RSI(c, timeperiod=14)
        features["rsi_7"] = talib.RSI(c, timeperiod=7)
        slowk, slowd = talib.STOCH(h, l, c, 14, 3, 0, 3, 0)
        features["stoch_k"] = slowk
        features["stoch_d"] = slowd
        features["cci"] = talib.CCI(h, l, c, timeperiod=14)
        features["willr"] = talib.WILLR(h, l, c, timeperiod=14)
        features["roc_10"] = talib.ROC(c, timeperiod=10)
        features["mom_10"] = talib.MOM(c, timeperiod=10)
    else:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features["rsi_14"] = (100 - 100 / (1 + rs)).values

    # ── Volatility ───────────────────────────────────
    if HAS_TALIB:
        features["atr_14"] = talib.ATR(h, l, c, timeperiod=14)
        features["atr_7"] = talib.ATR(h, l, c, timeperiod=7)
        features["natr"] = talib.NATR(h, l, c, timeperiod=14)
        upper, middle, lower = talib.BBANDS(c, 20, 2, 2)
        features["bb_upper"] = upper
        features["bb_middle"] = middle
        features["bb_lower"] = lower
        features["bb_width"] = (upper - lower) / (middle + 1e-10)
        features["bb_pctb"] = (c - lower) / (upper - lower + 1e-10)
    else:
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        features["atr_14"] = tr.rolling(14).mean().values

    # ── Volume ───────────────────────────────────────
    if HAS_TALIB:
        features["obv"] = talib.OBV(c, v)
        features["ad"] = talib.AD(h, l, c, v)
        features["mfi"] = talib.MFI(h, l, c, v, timeperiod=14)
    features["volume_sma_20"] = v.rolling(20).mean().values
    features["volume_ratio"] = (v / (v.rolling(20).mean() + 1e-10)).values

    # ── VWAP ─────────────────────────────────────────
    typical_price = (h + l + c) / 3
    cum_vol = v.cumsum()
    cum_tp_vol = (typical_price * v).cumsum()
    features["vwap"] = (cum_tp_vol / (cum_vol + 1e-10)).values
    features["vwap_deviation"] = ((c - features["vwap"]) / (features["vwap"] + 1e-10))
    if isinstance(features["vwap_deviation"], pd.Series):
        features["vwap_deviation"] = features["vwap_deviation"].values

    # ── SuperTrend (pandas-ta) ───────────────────────
    if HAS_PTA:
        try:
            st = pta.supertrend(h, l, c, length=10, multiplier=3.0)
            if st is not None and len(st.columns) >= 4:
                features["supertrend"] = st.iloc[:, 0].values
                features["supertrend_dir"] = st.iloc[:, 1].values
        except Exception:
            pass

    # ── Ichimoku ─────────────────────────────────────
    if HAS_PTA:
        try:
            ich = pta.ichimoku(h, l, c)
            if ich is not None and len(ich) == 2:
                ich_df = ich[0]
                for col in ich_df.columns:
                    short_name = col.replace("ISA_", "ichi_a_").replace("ISB_", "ichi_b_").replace("ITS_", "ichi_ts_").replace("IKS_", "ichi_ks_")
                    features[short_name[:20]] = ich_df[col].values
        except Exception:
            pass

    # ── Derived ratios ───────────────────────────────
    features["close_to_sma20"] = (c / (features.get("sma_20", c.rolling(20).mean().values) + 1e-10)).values if "sma_20" in features else np.full(len(c), np.nan)
    features["close_to_sma50"] = (c / (features.get("sma_50", c.rolling(50).mean().values) + 1e-10)).values if "sma_50" in features else np.full(len(c), np.nan)
    features["high_low_range"] = ((h - l) / (c + 1e-10)).values
    features["body_ratio"] = ((c - o).abs() / (h - l + 1e-10)).values

    # Convert back to Polars
    for col_name, arr in features.items():
        if isinstance(arr, pd.Series):
            arr = arr.values
        if isinstance(arr, np.ndarray) and len(arr) == len(df):
            df = df.with_columns(
                pl.Series(name=col_name, values=arr.astype(np.float64))
            )

    return df


def compute_hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """Compute Hurst exponent via rescaled range (R/S) analysis.

    The R/S statistic is applied to the **returns** (first differences)
    of the input series — this is the financially meaningful convention:

    * H ≈ 0.5  →  random walk (returns are i.i.d.)
    * H > 0.5  →  trending / persistent
    * H < 0.5  →  mean-reverting / anti-persistent

    Uses non-overlapping chunks at each lag.  Only lags with ≥ 3 valid
    R/S values are kept so the log-log regression is well-conditioned.
    """
    # Work on returns, not levels
    returns = np.diff(series)
    n = len(returns)
    if n < max_lag * 2:
        max_lag = n // 4
    if max_lag < 10:
        return 0.5

    lags = np.unique(np.geomspace(10, max_lag, num=40, dtype=int))
    valid_lags: list = []
    valid_rs: list = []

    for lag in lags:
        chunks = [returns[i: i + lag] for i in range(0, n - lag, lag)]
        rs_values: list = []
        for chunk in chunks:
            if len(chunk) < lag:
                continue
            mean_c = np.mean(chunk)
            dev = np.cumsum(chunk - mean_c)
            r = np.max(dev) - np.min(dev)
            s = np.std(chunk, ddof=1)
            if s > 1e-15:
                rs_values.append(r / s)
        if len(rs_values) >= 3:
            valid_lags.append(lag)
            valid_rs.append(np.mean(rs_values))

    if len(valid_lags) < 4:
        return 0.5

    log_lags = np.log(np.array(valid_lags, dtype=np.float64))
    log_rs = np.log(np.array(valid_rs, dtype=np.float64))
    try:
        poly = np.polyfit(log_lags, log_rs, 1)
        return float(np.clip(poly[0], 0, 1))
    except Exception:
        return 0.5


def compute_fractal_dimension(series: np.ndarray, k_max: int = 10) -> float:
    """Higuchi fractal dimension."""
    n = len(series)
    if n < k_max * 4:
        return 1.5

    ln_lengths = []
    ln_ks = []
    for k in range(1, k_max + 1):
        lengths = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            diff_sum = np.sum(np.abs(np.diff(series[indices])))
            norm = (n - 1) / (k * ((n - m) // k) * k + 1e-10)
            lengths.append(diff_sum * norm)
        if lengths:
            ln_lengths.append(np.log(np.mean(lengths) + 1e-10))
            ln_ks.append(np.log(1.0 / k))

    if len(ln_lengths) < 3:
        return 1.5
    try:
        poly = np.polyfit(ln_ks, ln_lengths, 1)
        return float(np.clip(poly[0], 1.0, 2.0))
    except Exception:
        return 1.5

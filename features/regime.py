"""
Market regime detection.

Methods:
  - Hidden Markov Model (Gaussian HMM, 3 regimes)
  - Isolation Forest anomaly scoring
  - Volatility-regime clustering
  - Red Queen adaptive regime (rolling regime shift detection)
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
import structlog
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = structlog.get_logger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    log.warning("hmmlearn not installed — HMM regime disabled")


class MarketRegime(IntEnum):
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    MEAN_REVERTING = 2
    HIGH_VOLATILITY = 3
    LOW_VOLATILITY = 4
    UNKNOWN = -1


class RegimeDetector:
    """Multi-method regime classification engine."""

    def __init__(self, n_regimes: int = 3, contamination: float = 0.05):
        self.n_regimes = n_regimes
        self.contamination = contamination
        self._hmm: Optional[GaussianHMM] = None
        self._iforest: Optional[IsolationForest] = None
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pl.DataFrame) -> None:
        """Fit HMM + Isolation Forest on historical features."""
        features = self._extract_regime_features(df)
        if features is None or len(features) < 50:
            log.warning("regime.insufficient_data")
            return

        X = self._scaler.fit_transform(features)

        # HMM
        if HAS_HMM:
            self._hmm = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            try:
                self._hmm.fit(X)
                log.info("regime.hmm_fitted", n_components=self.n_regimes)
            except Exception as exc:
                log.error("regime.hmm_fit_failed", error=str(exc))
                self._hmm = None

        # Isolation Forest
        self._iforest = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        self._iforest.fit(X)

        self._fitted = True
        log.info("regime.fitted")

    def predict(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Predict regimes for a DataFrame. Returns dict of arrays."""
        features = self._extract_regime_features(df)
        n = len(df)
        result: Dict[str, np.ndarray] = {
            "hmm_regime": np.full(n, MarketRegime.UNKNOWN, dtype=int),
            "anomaly_score": np.zeros(n),
            "is_anomaly": np.zeros(n, dtype=bool),
            "vol_regime": np.full(n, MarketRegime.UNKNOWN, dtype=int),
        }

        if features is None or not self._fitted:
            return result

        X = self._scaler.transform(features)
        offset = n - len(X)

        # HMM regimes
        if self._hmm is not None:
            try:
                hmm_states = self._hmm.predict(X)
                result["hmm_regime"][offset:] = hmm_states
                # Decode to meaningful labels using means
                result["hmm_regime_proba"] = self._hmm.predict_proba(X)
            except Exception:
                pass

        # Isolation Forest
        if self._iforest is not None:
            scores = self._iforest.decision_function(X)
            labels = self._iforest.predict(X)
            result["anomaly_score"][offset:] = scores
            result["is_anomaly"][offset:] = labels == -1

        # Volatility-based regime
        result["vol_regime"] = self._classify_vol_regime(df)

        return result

    def detect_regime_shift(self, df: pl.DataFrame, lookback: int = 50) -> bool:
        """Red Queen: detect if regime has shifted in recent bars."""
        if len(df) < lookback * 2:
            return False
        regimes = self.predict(df)
        recent = regimes["hmm_regime"][-lookback:]
        prior = regimes["hmm_regime"][-lookback * 2: -lookback]
        if len(set(recent)) == 0 or len(set(prior)) == 0:
            return False
        # Check if modal regime changed
        recent_mode = int(np.bincount(recent[recent >= 0]).argmax()) if np.any(recent >= 0) else -1
        prior_mode = int(np.bincount(prior[prior >= 0]).argmax()) if np.any(prior >= 0) else -1
        return recent_mode != prior_mode

    def _classify_vol_regime(self, df: pl.DataFrame) -> np.ndarray:
        """Simple volatility percentile classification."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)
        regimes = np.full(n, MarketRegime.UNKNOWN, dtype=int)

        if n < 30:
            return regimes

        returns = np.diff(np.log(close + 1e-15))
        for i in range(29, n - 1):
            window_ret = returns[max(0, i - 29): i + 1]
            vol = np.std(window_ret)
            mean_ret = np.mean(window_ret)

            # Long-term vol percentile
            long_ret = returns[max(0, i - 99): i + 1]
            vol_pctile = np.mean(vol > np.std(long_ret) * np.array([0.5, 1.0, 1.5]))

            if vol_pctile > 0.7:
                regimes[i + 1] = MarketRegime.HIGH_VOLATILITY
            elif vol_pctile < 0.3:
                regimes[i + 1] = MarketRegime.LOW_VOLATILITY
            elif mean_ret > 0.001:
                regimes[i + 1] = MarketRegime.TRENDING_UP
            elif mean_ret < -0.001:
                regimes[i + 1] = MarketRegime.TRENDING_DOWN
            else:
                regimes[i + 1] = MarketRegime.MEAN_REVERTING

        return regimes

    def _extract_regime_features(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Extract features suitable for regime detection."""
        close = df["close"].to_numpy().astype(np.float64)
        if len(close) < 30:
            return None

        returns = np.diff(np.log(close + 1e-15))
        n = len(returns)

        # Rolling stats
        features = []
        for i in range(20, n):
            w = returns[i - 20: i]
            features.append([
                np.mean(w),          # mean return
                np.std(w),           # volatility
                float(np.corrcoef(w[:-1], w[1:])[0, 1]) if len(w) > 2 else 0,  # autocorr
                np.mean(w > 0),      # win rate
                np.max(np.abs(w)),   # max move
            ])

        return np.array(features)

    def save(self, path: str) -> None:
        dump({"hmm": self._hmm, "iforest": self._iforest, "scaler": self._scaler}, path)

    def load(self, path: str) -> None:
        data = load(path)
        self._hmm = data["hmm"]
        self._iforest = data["iforest"]
        self._scaler = data["scaler"]
        self._fitted = True

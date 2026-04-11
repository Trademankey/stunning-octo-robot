"""
Master Feature Pipeline — orchestrates all feature modules.

Produces 80+ features per bar for each symbol:
  - 40+ TA (technical.py)
  - 20+ statistical (statistical.py)
  - 15+ orderbook microstructure (orderbook.py)
  - 5+ regime labels (regime.py)
  - 10+ latent embeddings (dimensionality.py)
  - On-chain & sentiment (injected externally)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import structlog
from joblib import Parallel, delayed

from config.settings import get_settings
from features.dimensionality import DimensionalityReducer
from features.orderbook import compute_orderbook_features
from features.regime import RegimeDetector
from features.statistical import compute_statistical_features
from features.technical import (
    compute_fractal_dimension,
    compute_hurst_exponent,
    compute_technical_features,
)

log = structlog.get_logger(__name__)


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.
    Call .fit() on training data, then .transform() for inference.
    """

    def __init__(self):
        self._settings = get_settings()
        self.regime_detector = RegimeDetector(n_regimes=3)
        self.dim_reducer = DimensionalityReducer(
            n_pca=10, n_umap=3, n_autoencoder=8
        )
        self._fitted = False
        self._feature_names: List[str] = []

    def fit(self, data: Dict[str, pl.DataFrame]) -> None:
        """
        Fit regime detector + dimensionality reducer on training data.
        data: {symbol: DataFrame} — expects OHLCV columns.
        """
        all_features = []
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            enriched = self._compute_core_features(df)
            feature_cols = [c for c in enriched.columns if c not in ("time", "open", "high", "low", "close", "volume")]
            X = enriched.select(feature_cols).to_numpy()
            X = np.nan_to_num(X, nan=0.0)
            all_features.append(X)

        if not all_features:
            log.warning("pipeline.no_data_to_fit")
            return

        X_all = np.vstack(all_features)

        # Fit regime on first symbol (BTC typically)
        first_sym = list(data.keys())[0]
        self.regime_detector.fit(data[first_sym])

        # Fit dimensionality reduction
        self.dim_reducer.fit(X_all)

        self._fitted = True
        log.info("pipeline.fitted", total_samples=X_all.shape[0], n_features=X_all.shape[1])

    def transform(
        self,
        df: pl.DataFrame,
        symbol: str = "",
        orderbook: Optional[Dict] = None,
        onchain: Optional[Dict[str, float]] = None,
        sentiment: Optional[Dict[str, float]] = None,
    ) -> pl.DataFrame:
        """
        Transform a single symbol's OHLCV into the full feature matrix.
        Returns DataFrame with all feature columns appended.
        """
        # Core TA + Statistical
        enriched = self._compute_core_features(df)

        # Hurst + Fractal
        close = df["close"].to_numpy().astype(np.float64)
        hurst = compute_hurst_exponent(close)
        fractal = compute_fractal_dimension(close)
        enriched = enriched.with_columns([
            pl.lit(hurst).alias("hurst_exponent"),
            pl.lit(fractal).alias("fractal_dimension"),
        ])

        # Regime
        if self._fitted:
            regimes = self.regime_detector.predict(df)
            for key, arr in regimes.items():
                if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == len(enriched):
                    enriched = enriched.with_columns(
                        pl.Series(name=f"regime_{key}", values=arr.astype(np.float64))
                    )

        # Orderbook features
        if orderbook:
            ob_feats = compute_orderbook_features(orderbook)
            for k, v in ob_feats.items():
                enriched = enriched.with_columns(pl.lit(float(v)).alias(f"ob_{k}"))

        # On-chain features
        if onchain:
            for k, v in onchain.items():
                val = float(v) if v is not None else 0.0
                enriched = enriched.with_columns(pl.lit(val).alias(f"onchain_{k}"))

        # Sentiment features
        if sentiment:
            for k, v in sentiment.items():
                enriched = enriched.with_columns(pl.lit(float(v)).alias(f"sent_{k}"))

        # Dimensionality reduction (latent features)
        if self._fitted:
            feature_cols = [
                c for c in enriched.columns
                if c not in ("time", "open", "high", "low", "close", "volume")
            ]
            X = enriched.select(feature_cols).to_numpy()
            X = np.nan_to_num(X, nan=0.0)
            try:
                latent = self.dim_reducer.transform(X)
                for i in range(latent.shape[1]):
                    enriched = enriched.with_columns(
                        pl.Series(name=f"latent_{i}", values=latent[:, i])
                    )
            except Exception as exc:
                log.debug("pipeline.latent_failed", error=str(exc))

        # Store feature names
        self._feature_names = [
            c for c in enriched.columns
            if c not in ("time", "open", "high", "low", "close", "volume")
        ]

        return enriched

    def get_feature_matrix(
        self, df: pl.DataFrame, dropna: bool = True
    ) -> np.ndarray:
        """Extract pure numpy feature matrix for model input."""
        feature_cols = [
            c for c in df.columns
            if c not in ("time", "open", "high", "low", "close", "volume")
        ]
        X = df.select(feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if dropna:
            # Remove rows that are all zero (warmup period)
            mask = np.any(X != 0, axis=1)
            X = X[mask]
        return X

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def _compute_core_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute TA + statistical features."""
        enriched = compute_technical_features(df)
        enriched = compute_statistical_features(enriched)
        return enriched

    def transform_batch(
        self,
        data: Dict[str, pl.DataFrame],
        orderbooks: Optional[Dict[str, Dict]] = None,
        onchain_data: Optional[Dict[str, Dict]] = None,
        sentiment_data: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, pl.DataFrame]:
        """Transform multiple symbols in parallel."""
        results = {}
        for symbol, df in data.items():
            ob = (orderbooks or {}).get(symbol)
            oc = (onchain_data or {}).get(symbol.split("/")[0])
            sent = (sentiment_data or {}).get(symbol.split("/")[0])
            try:
                results[symbol] = self.transform(df, symbol, ob, oc, sent)
            except Exception as exc:
                log.error("pipeline.transform_error", symbol=symbol, error=str(exc))
        return results

    def save(self, base_path: str) -> None:
        self.regime_detector.save(f"{base_path}/regime_detector.joblib")
        self.dim_reducer.save(f"{base_path}/dim_reducer.joblib")

    def load(self, base_path: str) -> None:
        self.regime_detector.load(f"{base_path}/regime_detector.joblib")
        self.dim_reducer.load(f"{base_path}/dim_reducer.joblib")
        self._fitted = True

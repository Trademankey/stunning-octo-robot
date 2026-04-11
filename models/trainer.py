"""
Model Trainer — orchestrates training, validation, and online retraining.

Handles:
  - Walk-forward train/val splits
  - Feature pipeline fitting
  - Ensemble training
  - Checkpoint management
  - Scheduled online retraining (every N hours)
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl
import structlog

from config.settings import get_settings
from data.db import Database
from features.pipeline import FeaturePipeline
from models.ensemble import EnsembleModel

log = structlog.get_logger(__name__)

CHECKPOINT_DIR = Path("./model_checkpoints")


class ModelTrainer:
    """Full training orchestrator."""

    def __init__(self, db: Database, feature_pipeline: FeaturePipeline):
        self._settings = get_settings()
        self._db = db
        self._pipeline = feature_pipeline
        self.ensemble = EnsembleModel()
        self._last_train_time: Optional[float] = None

    async def initial_train(
        self,
        data: Dict[str, pl.DataFrame],
        val_ratio: float = 0.2,
    ) -> Dict:
        """
        Run full training on historical data.
        data: {symbol: OHLCV DataFrame}
        """
        log.info("trainer.initial_train_start")
        start = time.time()

        # Fit feature pipeline
        self._pipeline.fit(data)

        # Transform all data
        all_X = []
        all_y = []
        all_prices = []

        for symbol, df in data.items():
            if len(df) < 200:
                continue

            enriched = self._pipeline.transform(df, symbol=symbol)
            feature_cols = [
                c for c in enriched.columns
                if c not in ("time", "open", "high", "low", "close", "volume")
            ]
            X = enriched.select(feature_cols).to_numpy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Target: forward returns
            close = enriched["close"].to_numpy()
            y = np.zeros(len(close))
            horizon = self._settings.model.forecast_horizon
            for i in range(len(close) - horizon):
                y[i] = (close[i + horizon] - close[i]) / (close[i] + 1e-15)

            # Trim warmup (first 100 bars)
            X = X[100:]
            y = y[100:]
            prices = close[100:]

            all_X.append(X)
            all_y.append(y)
            all_prices.append(prices)

        if not all_X:
            log.error("trainer.no_data")
            return {"error": "no data"}

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        prices = np.concatenate(all_prices)

        # Train / Val split (time-based)
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        prices_train = prices[:split_idx]
        prices_val = prices[split_idx:]

        # Train ensemble
        metrics = self.ensemble.train(
            X_train, y_train, X_val, y_val,
            prices_train=prices_train,
            prices_val=prices_val,
            seq_len=self._settings.model.sequence_length,
        )

        # Save checkpoint
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.ensemble.save(str(CHECKPOINT_DIR / "latest"))
        self._pipeline.save(str(CHECKPOINT_DIR / "latest"))

        elapsed = time.time() - start
        self._last_train_time = time.time()

        log.info("trainer.initial_train_done", elapsed_s=elapsed, metrics=metrics)

        # Record metrics to DB
        for model_name, model_metrics in metrics.items():
            if isinstance(model_metrics, dict):
                for metric_name, value in model_metrics.items():
                    if isinstance(value, (int, float)):
                        await self._db.record_model_metric(
                            model_name, metric_name, float(value)
                        )

        return metrics

    async def online_retrain(
        self,
        data: Dict[str, pl.DataFrame],
    ) -> Optional[Dict]:
        """
        Incremental online retraining if enough time has passed.
        """
        if self._last_train_time is None:
            return None

        hours_since = (time.time() - self._last_train_time) / 3600
        if hours_since < self._settings.model.retrain_interval_hours:
            return None

        log.info("trainer.online_retrain_start", hours_since=hours_since)
        metrics = await self.initial_train(data, val_ratio=0.15)
        return metrics

    def load_checkpoint(self) -> bool:
        """Load latest checkpoint if available."""
        path = CHECKPOINT_DIR / "latest"
        if path.exists():
            try:
                self.ensemble.load(str(path))
                self._pipeline.load(str(path))
                self._last_train_time = time.time()
                log.info("trainer.checkpoint_loaded")
                return True
            except Exception as exc:
                log.error("trainer.checkpoint_load_failed", error=str(exc))
        return False

    def predict(
        self, X: np.ndarray, prices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate ensemble signal from feature matrix."""
        return self.ensemble.predict(
            X, prices, seq_len=self._settings.model.sequence_length
        )

    def predict_with_uncertainty(
        self, X: np.ndarray, prices: Optional[np.ndarray] = None
    ):
        """Predict with MC-Dropout uncertainty."""
        return self.ensemble.predict_with_uncertainty(
            X, prices, seq_len=self._settings.model.sequence_length
        )

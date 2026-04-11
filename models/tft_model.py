"""
Temporal Fusion Transformer (TFT) wrapper.

Uses PyTorch-Forecasting's TFT for multi-horizon probabilistic forecasts
with variable selection, interpretable attention, and quantile outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import torch

from config.settings import get_settings

log = structlog.get_logger(__name__)

try:
    import pytorch_lightning as ptl
    from pytorch_forecasting import (
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )
    from pytorch_forecasting.metrics import QuantileLoss
    HAS_TFT = True
except ImportError:
    HAS_TFT = False
    log.warning("pytorch-forecasting not installed — TFT disabled")


class TFTModel:
    """Wrapper around pytorch-forecasting TFT for crypto price prediction."""

    def __init__(self):
        self._settings = get_settings().model
        self._model: Optional[Any] = None
        self._trainer: Optional[Any] = None
        self._dataset_params: Optional[Dict] = None

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        target: str = "log_return",
        group: str = "symbol",
        time_idx: str = "time_idx",
        known_reals: Optional[List[str]] = None,
        unknown_reals: Optional[List[str]] = None,
    ) -> Tuple[Any, Any]:
        """Create TimeSeriesDataSet for training and validation."""
        if not HAS_TFT:
            raise RuntimeError("pytorch-forecasting required for TFT")

        max_encoder_length = self._settings.sequence_length
        max_prediction_length = self._settings.forecast_horizon

        if known_reals is None:
            known_reals = []
        if unknown_reals is None:
            # Auto-detect numeric columns
            unknown_reals = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in (target, time_idx, group, "time") and c not in known_reals
            ][:30]  # Cap for memory

        training_cutoff = int(df[time_idx].max() - max_prediction_length)

        training = TimeSeriesDataSet(
            df[df[time_idx] <= training_cutoff],
            time_idx=time_idx,
            target=target,
            group_ids=[group],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=known_reals or [time_idx],
            time_varying_unknown_reals=[target] + unknown_reals[:25],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, df, predict=True, stop_randomization=True
        )

        self._dataset_params = {
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
        }

        return training, validation

    def train(
        self,
        training_dataset: Any,
        validation_dataset: Any,
        max_epochs: int = 30,
    ) -> Dict[str, float]:
        """Train the TFT model."""
        if not HAS_TFT:
            return {"error": "pytorch-forecasting not installed"}

        train_loader = training_dataset.to_dataloader(
            train=True, batch_size=self._settings.batch_size, num_workers=0
        )
        val_loader = validation_dataset.to_dataloader(
            train=False, batch_size=self._settings.batch_size, num_workers=0
        )

        self._model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self._settings.learning_rate,
            hidden_size=self._settings.tft_hidden_size,
            attention_head_size=self._settings.tft_attention_heads,
            dropout=0.1,
            hidden_continuous_size=self._settings.tft_hidden_size // 2,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4,
        )

        self._trainer = ptl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            gradient_clip_val=0.1,
            enable_model_summary=True,
            callbacks=[
                ptl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, mode="min"
                ),
            ],
        )

        self._trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Validate
        val_metrics = self._trainer.validate(self._model, dataloaders=val_loader)
        return {"val_loss": val_metrics[0]["val_loss"] if val_metrics else float("inf")}

    def predict(self, dataset: Any) -> np.ndarray:
        """Generate predictions. Returns median quantile forecast."""
        if self._model is None or not HAS_TFT:
            return np.array([])

        loader = dataset.to_dataloader(
            train=False, batch_size=self._settings.batch_size, num_workers=0
        )
        predictions = self._model.predict(loader, mode="prediction")
        return predictions.numpy() if isinstance(predictions, torch.Tensor) else np.array(predictions)

    def predict_with_uncertainty(self, dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Return (median_pred, std_pred) using quantile spread as uncertainty."""
        if self._model is None or not HAS_TFT:
            return np.array([]), np.array([])

        loader = dataset.to_dataloader(
            train=False, batch_size=self._settings.batch_size, num_workers=0
        )
        raw = self._model.predict(loader, mode="quantiles")
        if isinstance(raw, torch.Tensor):
            raw = raw.numpy()

        median = raw[:, :, 3] if raw.ndim == 3 else raw  # Q50
        # Uncertainty from IQR (Q75 - Q25)
        if raw.ndim == 3 and raw.shape[2] >= 7:
            uncertainty = raw[:, :, 5] - raw[:, :, 1]  # Q75 - Q25
        else:
            uncertainty = np.zeros_like(median)

        return median, uncertainty

    def get_attention_weights(self, dataset: Any) -> Optional[np.ndarray]:
        """Extract attention weights for interpretability."""
        if self._model is None or not HAS_TFT:
            return None
        try:
            loader = dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
            interpretation = self._model.interpret_output(
                self._model.predict(loader, mode="raw"), reduction="sum"
            )
            return interpretation["attention_weights"].numpy()
        except Exception:
            return None

    def save(self, path: str) -> None:
        if self._model is not None and self._trainer is not None:
            self._trainer.save_checkpoint(path)

    def load(self, path: str) -> None:
        if HAS_TFT:
            self._model = TemporalFusionTransformer.load_from_checkpoint(path)

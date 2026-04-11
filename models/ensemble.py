"""
Stacking Meta-Ensemble — combines all model predictions.

Sub-models:
  1. Gradient Boosting (XGB + LGB + CB blend)
  2. LSTM + Attention (sequence model)
  3. TFT (temporal fusion transformer)
  4. RL Agent (position sizing signal)

Meta-learner: Ridge regression on out-of-fold predictions.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from joblib import dump, load
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from config.settings import get_settings
from models.gradient_boost import GradientBoostEnsemble
from models.lstm_attention import LSTMAttentionModel
from models.rl_agent import RLAgent

log = structlog.get_logger(__name__)


class EnsembleModel:
    """
    Stacking ensemble with online reweighting.

    Produces final signal ∈ [-1, +1]:
      -1 = strong short, 0 = neutral, +1 = strong long
    """

    def __init__(self):
        self._settings = get_settings().model
        self.gbm = GradientBoostEnsemble()
        self.lstm = LSTMAttentionModel()
        self.rl = RLAgent(algo="PPO")
        self._meta_learner: Optional[RidgeCV] = None
        self._model_weights = {"gbm": 0.35, "lstm": 0.35, "rl": 0.30}
        self._fitted = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        prices_train: Optional[np.ndarray] = None,
        prices_val: Optional[np.ndarray] = None,
        seq_len: int = 128,
    ) -> Dict[str, Any]:
        """
        Full training pipeline: train sub-models → generate OOF preds → fit meta-learner.
        """
        metrics: Dict[str, Any] = {}

        # ── 1. GBM (flat features) ──────────────────
        y_binary = (y_train > 0).astype(int)
        y_val_binary = (y_val > 0).astype(int)
        gbm_metrics = self.gbm.train(X_train, y_binary, X_val, y_val_binary, tune=True)
        metrics["gbm"] = gbm_metrics

        # ── 2. LSTM (sequential features) ────────────
        X_seq_train, y_seq_train = self._make_sequences(X_train, y_train, seq_len)
        X_seq_val, y_seq_val = self._make_sequences(X_val, y_val, seq_len)

        if X_seq_train.shape[0] > 10:
            self.lstm.input_dim = X_seq_train.shape[2]
            lstm_metrics = self.lstm.train(X_seq_train, y_seq_train, X_seq_val, y_seq_val, epochs=50)
            metrics["lstm"] = lstm_metrics
        else:
            log.warning("ensemble.lstm_skipped — insufficient sequences")

        # ── 3. RL Agent ──────────────────────────────
        if prices_train is not None:
            rl_features_train = X_train[: len(prices_train)]
            rl_metrics = self.rl.train(
                rl_features_train, prices_train,
                X_val[: len(prices_val)] if prices_val is not None else None,
                prices_val,
            )
            metrics["rl"] = rl_metrics

        # ── 4. Meta-learner (stacking) ──────────────
        if self._settings.ensemble_weights_method == "stacking":
            self._fit_meta_learner(X_train, X_val, y_val, prices_val, seq_len)

        self._fitted = True
        log.info("ensemble.trained", metrics=metrics)
        return metrics

    def predict(
        self,
        X: np.ndarray,
        prices: Optional[np.ndarray] = None,
        seq_len: int = 128,
    ) -> np.ndarray:
        """
        Generate ensemble signal ∈ [-1, +1] for each sample.
        """
        n = X.shape[0]

        # GBM prediction (probability → centered signal)
        gbm_pred = self.gbm.predict(X)  # [0, 1]
        gbm_signal = (gbm_pred - 0.5) * 2  # [-1, 1]

        # LSTM prediction
        lstm_signal = np.zeros(n)
        X_seq, _ = self._make_sequences(X, np.zeros(n), seq_len)
        if X_seq.shape[0] > 0:
            lstm_raw = self.lstm.predict(X_seq)
            # Pad beginning (no sequence available yet)
            lstm_signal[seq_len:seq_len + len(lstm_raw)] = np.clip(lstm_raw, -1, 1)

        # RL prediction
        rl_signal = np.zeros(n)
        if prices is not None:
            rl_raw = self.rl.predict(X, prices)
            rl_signal[: len(rl_raw)] = rl_raw

        # Combine
        if self._meta_learner is not None:
            # Stack sub-model predictions
            stack = np.column_stack([gbm_signal, lstm_signal, rl_signal])
            signal = self._meta_learner.predict(stack)
        else:
            # Weighted average
            w = self._model_weights
            signal = (
                w["gbm"] * gbm_signal +
                w["lstm"] * lstm_signal +
                w["rl"] * rl_signal
            )

        return np.clip(signal, -1, 1)

    def predict_with_uncertainty(
        self, X: np.ndarray, prices: Optional[np.ndarray] = None, seq_len: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (signal, uncertainty). Uses MC-Dropout from LSTM."""
        n = X.shape[0]
        signal = self.predict(X, prices, seq_len)

        # LSTM MC-Dropout uncertainty
        X_seq, _ = self._make_sequences(X, np.zeros(n), seq_len)
        uncertainty = np.ones(n) * 0.5  # default uncertainty

        if X_seq.shape[0] > 0:
            _, mc_std = self.lstm.predict_mc_dropout(X_seq, n_samples=30)
            uncertainty[seq_len:seq_len + len(mc_std)] = mc_std

        return signal, uncertainty

    def _fit_meta_learner(
        self, X_train, X_val, y_val, prices_val, seq_len
    ) -> None:
        """Fit Ridge meta-learner on validation predictions."""
        n = X_val.shape[0]

        gbm_pred = (self.gbm.predict(X_val) - 0.5) * 2

        lstm_pred = np.zeros(n)
        X_seq, _ = self._make_sequences(X_val, np.zeros(n), seq_len)
        if X_seq.shape[0] > 0:
            raw = self.lstm.predict(X_seq)
            lstm_pred[seq_len:seq_len + len(raw)] = np.clip(raw, -1, 1)

        rl_pred = np.zeros(n)
        if prices_val is not None:
            raw = self.rl.predict(X_val, prices_val)
            rl_pred[: len(raw)] = raw

        stack = np.column_stack([gbm_pred, lstm_pred, rl_pred])
        y_target = np.sign(y_val)  # direction

        # Trim warmup
        mask = np.any(stack != 0, axis=1)
        if mask.sum() > 50:
            self._meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            self._meta_learner.fit(stack[mask], y_target[mask])
            log.info("meta_learner.fitted", alpha=self._meta_learner.alpha_)

    @staticmethod
    def _make_sequences(
        X: np.ndarray, y: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create (N, seq_len, features) sequences for LSTM."""
        n = len(X)
        if n <= seq_len:
            return np.empty((0, seq_len, X.shape[1])), np.empty(0)
        X_seq = np.array([X[i - seq_len: i] for i in range(seq_len, n)])
        y_seq = y[seq_len:]
        return X_seq, y_seq

    def save(self, base_path: str) -> None:
        os.makedirs(base_path, exist_ok=True)
        self.gbm.save(base_path)
        self.lstm.save(f"{base_path}/lstm_model.pt")
        self.rl.save(f"{base_path}/rl_model")
        if self._meta_learner:
            dump(self._meta_learner, f"{base_path}/meta_learner.joblib")
        dump(self._model_weights, f"{base_path}/weights.joblib")

    def load(self, base_path: str) -> None:
        self.gbm.load(base_path)
        if os.path.exists(f"{base_path}/lstm_model.pt"):
            self.lstm.load(f"{base_path}/lstm_model.pt")
        if os.path.exists(f"{base_path}/rl_model.zip"):
            self.rl.load(f"{base_path}/rl_model")
        meta_path = f"{base_path}/meta_learner.joblib"
        if os.path.exists(meta_path):
            self._meta_learner = load(meta_path)
        weights_path = f"{base_path}/weights.joblib"
        if os.path.exists(weights_path):
            self._model_weights = load(weights_path)
        self._fitted = True

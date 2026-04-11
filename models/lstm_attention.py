"""
LSTM + Multi-Head Self-Attention model for sequence-to-one prediction.

Architecture:
  Input → LayerNorm → BiLSTM → Attention → FC → Output (direction + magnitude)
Supports Monte Carlo Dropout for uncertainty estimation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import structlog

from config.settings import get_settings

log = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.warning("PyTorch not installed — LSTM model disabled")


# Network classes are only defined when torch is available.
if HAS_TORCH:
    class _MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn_out, _ = self.attn(x, x, x)
            return self.norm(x + self.dropout(attn_out))

    class LSTMAttentionNet(nn.Module):
        """BiLSTM + Self-Attention with MC-Dropout."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            n_heads: int = 4,
            dropout: float = 0.2,
            output_dim: int = 1,
        ):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            )
            self.attention = _MultiHeadAttention(hidden_dim * 2, n_heads, dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),   # MC Dropout — stays active at inference
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(x)
            lstm_out, _ = self.lstm(x)           # (B, T, 2*H)
            attn_out = self.attention(lstm_out)   # (B, T, 2*H)
            last = attn_out[:, -1, :]            # (B, 2*H) — last timestep
            return self.fc(last)                  # (B, output_dim)


class LSTMAttentionModel:
    """High-level trainer/predictor wrapper."""

    def __init__(self, input_dim: int = 80):
        self._settings = get_settings().model
        self.input_dim = input_dim
        self._model = None
        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if HAS_TORCH else None
        )

    def build(self) -> None:
        if not HAS_TORCH:
            log.warning("lstm.build skipped — torch not installed")
            return
        self._model = LSTMAttentionNet(
            input_dim=self.input_dim,
            hidden_dim=self._settings.lstm_hidden_size,
            num_layers=self._settings.lstm_num_layers,
            n_heads=4,
            dropout=0.2,
            output_dim=1,
        ).to(self._device)
        log.info("lstm_attention.built", params=sum(p.numel() for p in self._model.parameters()))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
    ) -> dict:
        """
        Train on sequences. X shape: (N, seq_len, features), y shape: (N,).
        """
        if not HAS_TORCH:
            return {"best_val_loss": float("inf")}
        if self._model is None:
            self.build()

        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(-1)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(-1)
        )
        train_loader = DataLoader(train_ds, batch_size=self._settings.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self._settings.batch_size)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._settings.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.HuberLoss()

        best_val = float("inf")
        best_state = None

        for epoch in range(epochs):
            # Train
            self._model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # Validate
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    pred = self._model(xb)
                    val_loss += criterion(pred, yb).item()

            avg_val = val_loss / max(len(val_loader), 1)
            if avg_val < best_val:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                log.info("lstm.epoch", epoch=epoch + 1, train_loss=train_loss / len(train_loader), val_loss=avg_val)

        if best_state:
            self._model.load_state_dict(best_state)

        return {"best_val_loss": best_val}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Standard prediction (dropout disabled)."""
        if self._model is None:
            return np.zeros(X.shape[0])
        self._model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self._device)
            pred = self._model(tensor)
        return pred.cpu().numpy().flatten()

    def predict_mc_dropout(
        self, X: np.ndarray, n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout uncertainty estimation.
        Returns (mean_pred, std_pred).
        """
        if self._model is None:
            n = X.shape[0]
            return np.zeros(n), np.ones(n)

        self._model.train()  # Keep dropout active
        preds = []
        tensor = torch.FloatTensor(X).to(self._device)
        with torch.no_grad():
            for _ in range(n_samples):
                p = self._model(tensor).cpu().numpy().flatten()
                preds.append(p)

        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: str) -> None:
        if self._model and HAS_TORCH:
            torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        if not HAS_TORCH:
            return
        self.build()
        if self._model is not None:
            self._model.load_state_dict(
                torch.load(path, map_location=self._device)
            )

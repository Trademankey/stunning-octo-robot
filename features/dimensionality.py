"""
Dimensionality reduction & latent feature extraction.

Methods: PCA, t-SNE (via UMAP for speed), Autoencoder latents.
Produces compressed feature vectors for downstream models.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import structlog
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = structlog.get_logger(__name__)

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DimensionalityReducer:
    """PCA + UMAP + Autoencoder latent space extractor."""

    def __init__(
        self,
        n_pca: int = 10,
        n_umap: int = 3,
        n_autoencoder: int = 8,
        ae_hidden: int = 32,
    ):
        self.n_pca = n_pca
        self.n_umap = n_umap
        self.n_autoencoder = n_autoencoder
        self.ae_hidden = ae_hidden

        self._scaler = StandardScaler()
        self._pca = PCA(n_components=n_pca)
        self._umap: Optional[UMAP] = None
        self._autoencoder = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit all reduction methods on feature matrix X (n_samples, n_features)."""
        if X.shape[0] < 100 or X.shape[1] < 5:
            log.warning("dimreduce.insufficient_data", shape=X.shape)
            return

        # Clean
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.fit_transform(X)

        # PCA
        n_comp = min(self.n_pca, X.shape[1], X.shape[0])
        self._pca = PCA(n_components=n_comp)
        self._pca.fit(X_scaled)
        log.info("pca.fitted", explained_var=float(self._pca.explained_variance_ratio_.sum()))

        # UMAP
        if HAS_UMAP:
            n_umap = min(self.n_umap, n_comp)
            self._umap = UMAP(
                n_components=n_umap, n_neighbors=30,
                min_dist=0.1, metric="euclidean", random_state=42,
            )
            try:
                self._umap.fit(X_scaled)
                log.info("umap.fitted", n_components=n_umap)
            except Exception as exc:
                log.warning("umap.fit_failed", error=str(exc))
                self._umap = None

        # Autoencoder
        if HAS_TORCH:
            self._fit_autoencoder(X_scaled)

        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features → concatenated latent vector."""
        if not self._fitted:
            return X

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X)

        parts = []

        # PCA
        pca_out = self._pca.transform(X_scaled)
        parts.append(pca_out)

        # UMAP
        if self._umap is not None:
            try:
                umap_out = self._umap.transform(X_scaled)
                parts.append(umap_out)
            except Exception:
                pass

        # Autoencoder
        if self._autoencoder is not None and HAS_TORCH:
            try:
                with torch.no_grad():
                    tensor = torch.FloatTensor(X_scaled)
                    latent = self._autoencoder.encode(tensor).numpy()
                    parts.append(latent)
            except Exception:
                pass

        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    # ── Autoencoder ──────────────────────────────────
    def _fit_autoencoder(self, X: np.ndarray) -> None:
        input_dim = X.shape[1]
        latent_dim = min(self.n_autoencoder, input_dim)

        model = _FeatureAutoencoder(input_dim, self.ae_hidden, latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        dataset = torch.FloatTensor(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        model.train()
        for epoch in range(50):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        self._autoencoder = model
        log.info("autoencoder.fitted", latent_dim=latent_dim, final_loss=total_loss / len(loader))

    def save(self, path: str) -> None:
        state = {
            "scaler": self._scaler,
            "pca": self._pca,
            "umap": self._umap,
            "fitted": self._fitted,
        }
        dump(state, path)
        if self._autoencoder is not None and HAS_TORCH:
            torch.save(self._autoencoder.state_dict(), path + ".ae.pt")

    def load(self, path: str) -> None:
        state = load(path)
        self._scaler = state["scaler"]
        self._pca = state["pca"]
        self._umap = state["umap"]
        self._fitted = state["fitted"]


if HAS_TORCH:
    class _FeatureAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            return self.decoder(z)
else:
    class _FeatureAutoencoder:
        pass

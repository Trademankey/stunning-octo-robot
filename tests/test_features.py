"""
Tests for feature engineering pipeline.
"""
import numpy as np
import polars as pl
import pytest

from features.orderbook import compute_orderbook_features
from features.statistical import _rolling_zscore, compute_statistical_features
from features.technical import compute_hurst_exponent, compute_fractal_dimension


def _make_ohlcv(n: int = 500) -> pl.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)  # floor
    return pl.DataFrame({
        "time": pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1) + pl.duration(minutes=15 * n),
            interval="15m", eager=True
        )[:n],
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.5),
        "low": close - np.abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.exponential(1000, n),
    })


class TestTechnicalFeatures:
    def test_hurst_random_walk(self):
        """iid returns → H ≈ 0.5 (tolerance for finite-sample noise)."""
        np.random.seed(0)
        rw = np.cumsum(np.random.randn(5000))
        h = compute_hurst_exponent(rw)
        assert 0.35 < h < 0.75, f"Expected ~0.5, got {h}"

    def test_hurst_trending(self):
        """Positively-autocorrelated returns (AR1 φ=+0.6) → H > 0.5."""
        np.random.seed(1)
        returns = np.zeros(5000)
        for i in range(1, 5000):
            returns[i] = 0.6 * returns[i - 1] + np.random.randn()
        prices = 100 + np.cumsum(returns)
        h = compute_hurst_exponent(prices)
        assert h > 0.55, f"Persistent series Hurst should be > 0.55, got {h}"

    def test_fractal_dimension_range(self):
        """Fractal dimension should be between 1 and 2."""
        np.random.seed(42)
        series = np.cumsum(np.random.randn(500))
        fd = compute_fractal_dimension(series)
        assert 1.0 <= fd <= 2.0, f"FD out of range: {fd}"


class TestStatisticalFeatures:
    def test_rolling_zscore_shape(self):
        arr = np.random.randn(100)
        result = _rolling_zscore(arr, 20)
        assert len(result) == 100
        assert np.isnan(result[0])
        assert not np.isnan(result[50])

    def test_statistical_features_added(self):
        df = _make_ohlcv(200)
        result = compute_statistical_features(df)
        assert "zscore_14" in result.columns
        assert "skew_30" in result.columns
        assert "entropy_30" in result.columns
        assert "log_return" in result.columns
        assert len(result) == len(df)


class TestOrderbookFeatures:
    def test_basic_orderbook(self):
        ob = {
            "bids": [[100.0, 10], [99.5, 20], [99.0, 30]],
            "asks": [[100.5, 15], [101.0, 25], [101.5, 35]],
        }
        feats = compute_orderbook_features(ob)
        assert "spread" in feats
        assert feats["spread"] == pytest.approx(0.5, abs=0.01)
        assert "imbalance_1" in feats
        assert "whale_imbalance" in feats

    def test_empty_orderbook(self):
        feats = compute_orderbook_features({"bids": [], "asks": []})
        assert feats["spread"] == 0.0
        assert feats["imbalance_1"] == 0.0

    def test_imbalance_direction(self):
        """More bid volume → positive imbalance."""
        ob = {
            "bids": [[100.0, 100], [99.5, 100]],
            "asks": [[100.5, 10], [101.0, 10]],
        }
        feats = compute_orderbook_features(ob)
        assert feats["imbalance_1"] > 0

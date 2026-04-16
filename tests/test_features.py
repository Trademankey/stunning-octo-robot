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


class TestFeaturePipelineConsistency:
    """Regression: feature count must be identical with and without optional data."""

    def test_same_column_count_with_and_without_optional_data(self):
        """The XGBoost 78-vs-99 bug: transform must always produce the same columns."""
        from features.pipeline import FeaturePipeline

        df = _make_ohlcv(300)
        pipe = FeaturePipeline()

        # Without any optional data
        out_bare = pipe.transform(df, symbol="TEST/USDT")

        # With orderbook + on-chain + sentiment
        ob = {
            "bids": [[100.0, 10], [99.5, 20], [99.0, 30]],
            "asks": [[100.5, 15], [101.0, 25], [101.5, 35]],
        }
        onchain = {"funding_rate": 0.01, "open_interest": 5000, "exchange_netflow": -100,
                    "nvt_ratio": 50, "mvrv": 1.5, "sopr": 1.01, "whale_txns": 42}
        sentiment = {"news": 0.3, "reddit": 0.1, "twitter": -0.2, "weighted_avg": 0.1}

        out_full = pipe.transform(df, symbol="TEST/USDT",
                                  orderbook=ob, onchain=onchain, sentiment=sentiment)

        bare_feat = [c for c in out_bare.columns if c not in ("time","open","high","low","close","volume")]
        full_feat = [c for c in out_full.columns if c not in ("time","open","high","low","close","volume")]

        assert len(bare_feat) == len(full_feat), (
            f"Feature count mismatch: bare={len(bare_feat)} vs full={len(full_feat)}. "
            f"Missing in bare: {set(full_feat)-set(bare_feat)}. "
            f"Extra in bare: {set(bare_feat)-set(full_feat)}"
        )
        assert bare_feat == full_feat, "Feature column ORDER differs"

    def test_ob_onchain_sent_columns_always_present(self):
        """Verify canonical columns exist even when no data is provided."""
        from features.pipeline import FeaturePipeline, _ONCHAIN_KEYS, _SENT_KEYS

        df = _make_ohlcv(200)
        pipe = FeaturePipeline()
        out = pipe.transform(df, symbol="TEST/USDT")

        cols = set(out.columns)
        for k in _ONCHAIN_KEYS:
            assert f"onchain_{k}" in cols, f"Missing onchain_{k}"
        for k in _SENT_KEYS:
            assert f"sent_{k}" in cols, f"Missing sent_{k}"
        assert "ob_spread" in cols, "Missing ob_spread"
        assert "ob_imbalance_5" in cols, "Missing ob_imbalance_5"

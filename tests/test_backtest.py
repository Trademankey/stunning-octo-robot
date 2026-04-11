"""
Tests for backtest engine and metrics.
"""
import numpy as np
import polars as pl
import pytest

from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.walk_forward import MonteCarloSimulator, WalkForwardOptimiser


def _make_equity_curve(n: int = 1000, trend: float = 0.0001) -> np.ndarray:
    """Synthetic equity curve with drift + noise."""
    np.random.seed(42)
    returns = trend + np.random.randn(n) * 0.005
    equity = 10000 * np.cumprod(1 + returns)
    return np.concatenate([[10000], equity])


def _make_trade_pnls(n: int = 200, win_rate: float = 0.55) -> np.ndarray:
    """Synthetic trade P&L series."""
    np.random.seed(42)
    pnls = []
    for _ in range(n):
        if np.random.random() < win_rate:
            pnls.append(np.random.exponential(50))
        else:
            pnls.append(-np.random.exponential(35))
    return np.array(pnls)


class TestMetrics:
    def test_profitable_strategy(self):
        equity = _make_equity_curve(1000, trend=0.0003)
        trades = _make_trade_pnls(200, win_rate=0.6)
        m = compute_metrics(equity, trades)
        assert m.total_return > 0
        assert m.sharpe_ratio > 0
        assert m.win_rate > 0.5
        assert m.profit_factor > 1.0

    def test_losing_strategy(self):
        equity = _make_equity_curve(1000, trend=-0.0003)
        trades = _make_trade_pnls(200, win_rate=0.3)
        m = compute_metrics(equity, trades)
        assert m.total_return < 0
        assert m.sharpe_ratio < 0

    def test_drawdown_positive(self):
        equity = _make_equity_curve(500)
        trades = _make_trade_pnls(100)
        m = compute_metrics(equity, trades)
        assert m.max_drawdown > 0
        assert m.max_drawdown < 1.0

    def test_sqn_calculation(self):
        trades = _make_trade_pnls(100, win_rate=0.6)
        m = compute_metrics(_make_equity_curve(500), trades)
        # SQN > 2 is considered good
        assert abs(m.sqn) > 0  # Just check it computes

    def test_empty_trades(self):
        equity = _make_equity_curve(100)
        m = compute_metrics(equity, np.array([]))
        assert m.total_trades == 0
        assert m.win_rate == 0


class TestWalkForward:
    def test_splits_generated(self):
        wf = WalkForwardOptimiser(n_splits=5)
        splits = wf.generate_splits(1000)
        assert len(splits) >= 3
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            assert train[-1] < test[0]  # No leakage


class TestMonteCarlo:
    def test_trade_resampling(self):
        mc = MonteCarloSimulator(n_simulations=100)
        trades = _make_trade_pnls(100, win_rate=0.6)
        results = mc.run_trade_resampling(trades)
        assert "profit_probability" in results
        assert "ruin_probability" in results
        assert results["profit_probability"] > 0
        assert results["final_equity_mean"] > 0

    def test_path_simulation(self):
        mc = MonteCarloSimulator(n_simulations=50)
        returns = np.random.randn(500) * 0.01 + 0.0001
        results = mc.run_path_simulation(returns, n_forward_bars=100)
        assert "final_mean" in results
        assert results["final_mean"] > 0

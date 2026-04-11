"""
Walk-Forward Optimisation & Monte Carlo stress testing.

Walk-forward: expanding/rolling window train → validate → advance
Monte Carlo:  bootstrap trade resampling + MC Dropout uncertainty
Stress tests: replay on 2022 crash, 2021 bull, 2024–25 sideways
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from backtest.metrics import BacktestMetrics, compute_metrics
from config.settings import get_settings

log = structlog.get_logger(__name__)


class WalkForwardOptimiser:
    """Walk-forward validation with expanding windows."""

    def __init__(self, n_splits: int = 5, train_ratio: float = 0.7):
        self._settings = get_settings()
        self.n_splits = n_splits
        self.train_ratio = train_ratio

    def generate_splits(
        self, n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding-window train/test index splits."""
        splits = []
        min_train = int(n_samples * 0.3)
        test_size = int(n_samples * (1 - self.train_ratio) / self.n_splits)

        for i in range(self.n_splits):
            train_end = min_train + (i + 1) * test_size
            test_end = min(train_end + test_size, n_samples)

            if train_end >= n_samples or test_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        train_fn,
        predict_fn,
        initial_equity: float = 10000.0,
    ) -> List[BacktestMetrics]:
        """
        Run walk-forward backtest.

        train_fn(X_train, y_train) → model
        predict_fn(model, X_test) → signals
        """
        splits = self.generate_splits(len(X))
        all_metrics: List[BacktestMetrics] = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            log.info(f"walk_forward.fold_{fold}", train=len(train_idx), test=len(test_idx))

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            prices_test = prices[test_idx]

            # Train
            model = train_fn(X_train, y_train)

            # Predict
            signals = predict_fn(model, X_test)

            # Simulate
            equity_curve, trade_pnls = self._simulate_trades(
                signals, prices_test, initial_equity
            )
            metrics = compute_metrics(equity_curve, np.array(trade_pnls), initial_equity)
            all_metrics.append(metrics)

            log.info(
                f"walk_forward.fold_{fold}_done",
                sharpe=metrics.sharpe_ratio,
                total_return=metrics.total_return,
                max_dd=metrics.max_drawdown,
            )

        return all_metrics

    def _simulate_trades(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        initial_equity: float,
    ) -> Tuple[np.ndarray, List[float]]:
        """Simple signal-based simulation."""
        equity = initial_equity
        position = 0.0
        entry_price = 0.0
        equity_curve = [equity]
        trade_pnls: List[float] = []
        fee_rate = 0.0004

        for i in range(1, len(signals)):
            price = prices[i]
            signal = signals[i] if i < len(signals) else 0

            # PnL from existing position
            if position != 0:
                price_change = (price - prices[i - 1]) / (prices[i - 1] + 1e-15)
                equity += position * price_change * equity * 0.5  # conservative

            # Signal action
            if signal > 0.2 and position <= 0:
                # Go long
                if position < 0:
                    pnl = (entry_price - price) * abs(position) * equity * 0.5 / entry_price
                    trade_pnls.append(pnl - fee_rate * equity * 0.01)
                position = signal
                entry_price = price
                equity -= fee_rate * equity * 0.01

            elif signal < -0.2 and position >= 0:
                # Go short
                if position > 0:
                    pnl = (price - entry_price) * position * equity * 0.5 / entry_price
                    trade_pnls.append(pnl - fee_rate * equity * 0.01)
                position = signal
                entry_price = price
                equity -= fee_rate * equity * 0.01

            equity_curve.append(equity)

        return np.array(equity_curve), trade_pnls


class MonteCarloSimulator:
    """Monte Carlo simulation for robustness testing."""

    def __init__(self, n_simulations: int = 1000):
        self._settings = get_settings()
        self.n_simulations = n_simulations

    def run_trade_resampling(
        self,
        trade_pnls: np.ndarray,
        initial_equity: float = 10000.0,
    ) -> Dict[str, np.ndarray]:
        """
        Bootstrap resample trade sequence to estimate:
        - Distribution of final equity
        - Distribution of max drawdown
        - Confidence intervals
        """
        n_trades = len(trade_pnls)
        if n_trades < 10:
            return {"error": "insufficient trades"}

        final_equities = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)

        for sim in range(self.n_simulations):
            # Resample with replacement
            sampled = np.random.choice(trade_pnls, size=n_trades, replace=True)
            equity_curve = initial_equity + np.cumsum(sampled)
            equity_curve = np.concatenate([[initial_equity], equity_curve])

            final_equities[sim] = equity_curve[-1]

            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / (peak + 1e-15)
            max_drawdowns[sim] = np.max(dd)

        return {
            "final_equity_mean": float(np.mean(final_equities)),
            "final_equity_median": float(np.median(final_equities)),
            "final_equity_5pct": float(np.percentile(final_equities, 5)),
            "final_equity_95pct": float(np.percentile(final_equities, 95)),
            "max_dd_mean": float(np.mean(max_drawdowns)),
            "max_dd_95pct": float(np.percentile(max_drawdowns, 95)),
            "ruin_probability": float(np.mean(final_equities < initial_equity * 0.5)),
            "profit_probability": float(np.mean(final_equities > initial_equity)),
            "final_equities": final_equities,
            "max_drawdowns": max_drawdowns,
        }

    def run_path_simulation(
        self,
        returns: np.ndarray,
        n_forward_bars: int = 1000,
        initial_equity: float = 10000.0,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate forward paths using historical return distribution.
        """
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        paths = np.zeros((self.n_simulations, n_forward_bars + 1))
        paths[:, 0] = initial_equity

        for sim in range(self.n_simulations):
            for t in range(1, n_forward_bars + 1):
                ret = np.random.normal(mean_ret, std_ret)
                paths[sim, t] = paths[sim, t - 1] * (1 + ret)

        return {
            "paths": paths,
            "final_mean": float(np.mean(paths[:, -1])),
            "final_5pct": float(np.percentile(paths[:, -1], 5)),
            "final_95pct": float(np.percentile(paths[:, -1], 95)),
        }


class StressTest:
    """Replay strategy on specific historical stress periods."""

    STRESS_PERIODS = {
        "2022_crash": {
            "description": "Luna/FTX crash period",
            "start": "2022-05-01",
            "end": "2022-12-31",
        },
        "2021_bull": {
            "description": "2021 bull run",
            "start": "2021-01-01",
            "end": "2021-11-30",
        },
        "2024_sideways": {
            "description": "2024 consolidation / chop",
            "start": "2024-03-01",
            "end": "2024-09-30",
        },
        "covid_crash": {
            "description": "March 2020 COVID crash",
            "start": "2020-02-15",
            "end": "2020-04-15",
        },
    }

    def filter_period(
        self, data: np.ndarray, timestamps: np.ndarray, period_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Filter data to a stress period."""
        period = self.STRESS_PERIODS.get(period_name)
        if period is None:
            return None

        start = np.datetime64(period["start"])
        end = np.datetime64(period["end"])
        mask = (timestamps >= start) & (timestamps <= end)

        if mask.sum() < 50:
            return None

        return data[mask], timestamps[mask]

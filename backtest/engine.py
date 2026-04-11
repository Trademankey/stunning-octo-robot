"""
Backtest Engine — top-level orchestrator.

Runs the full pipeline:
  1. Load historical data
  2. Compute features
  3. Train model (walk-forward)
  4. Generate signals
  5. Simulate execution
  6. Compute metrics
  7. Monte Carlo robustness check
  8. Stress tests
  9. Print report
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import structlog

from backtest.metrics import BacktestMetrics, compute_metrics, format_metrics
from backtest.walk_forward import MonteCarloSimulator, StressTest, WalkForwardOptimiser
from config.settings import get_settings
from features.pipeline import FeaturePipeline
from models.ensemble import EnsembleModel
from strategies.position_sizer import PositionSizer
from strategies.signal_generator import SignalGenerator

log = structlog.get_logger(__name__)


class BacktestEngine:
    """
    Full-pipeline backtester with walk-forward validation.
    Usage:
        engine = BacktestEngine()
        results = engine.run({"BTC/USDT": btc_df, "ETH/USDT": eth_df})
    """

    def __init__(
        self,
        initial_equity: float = 10000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        slippage_bps: float = 1.0,
    ):
        self._settings = get_settings()
        self.initial_equity = initial_equity
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps

        self.pipeline = FeaturePipeline()
        self.signal_gen = SignalGenerator()
        self.sizer = PositionSizer()
        self.wf = WalkForwardOptimiser(
            n_splits=self._settings.model.walk_forward_splits
        )
        self.mc = MonteCarloSimulator(
            n_simulations=self._settings.model.monte_carlo_simulations
        )

    def run(
        self,
        data: Dict[str, pl.DataFrame],
        walk_forward: bool = True,
    ) -> Dict:
        """
        Run complete backtest. Returns dict of metrics + MC results.
        """
        start = time.time()
        log.info("backtest.start", pairs=list(data.keys()))

        # ── 1. Feature Engineering ───────────────────
        self.pipeline.fit(data)
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_prices: List[np.ndarray] = []

        for symbol, df in data.items():
            if len(df) < 300:
                continue
            enriched = self.pipeline.transform(df, symbol=symbol)
            feature_cols = [
                c for c in enriched.columns
                if c not in ("time", "open", "high", "low", "close", "volume")
            ]
            X = enriched.select(feature_cols).to_numpy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            close = enriched["close"].to_numpy()
            horizon = self._settings.model.forecast_horizon
            y = np.zeros(len(close))
            for i in range(len(close) - horizon):
                y[i] = (close[i + horizon] - close[i]) / (close[i] + 1e-15)

            X, y, close = X[100:], y[100:], close[100:]
            all_X.append(X)
            all_y.append(y)
            all_prices.append(close)

        if not all_X:
            return {"error": "insufficient data"}

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        prices = np.concatenate(all_prices)

        # ── 2. Walk-Forward Backtest ─────────────────
        if walk_forward:
            wf_metrics = self.wf.run(
                X, y, prices,
                train_fn=self._train_model,
                predict_fn=self._predict_model,
                initial_equity=self.initial_equity,
            )
        else:
            wf_metrics = []

        # ── 3. Full-sample backtest ──────────────────
        split = int(len(X) * 0.7)
        model = self._train_model(X[:split], y[:split])
        signals = self._predict_model(model, X[split:])
        prices_test = prices[split:]

        equity_curve, trade_pnls = self._simulate_full(
            signals, prices_test, self.initial_equity
        )
        main_metrics = compute_metrics(
            equity_curve, np.array(trade_pnls),
            self.initial_equity, total_fees=self.taker_fee * len(trade_pnls) * 100,
        )

        # ── 4. Monte Carlo ───────────────────────────
        mc_results = {}
        if len(trade_pnls) > 20:
            mc_results = self.mc.run_trade_resampling(
                np.array(trade_pnls), self.initial_equity
            )
            # Remove large arrays for clean output
            mc_results.pop("final_equities", None)
            mc_results.pop("max_drawdowns", None)

        elapsed = time.time() - start

        # ── 5. Print Report ──────────────────────────
        print("\n" + format_metrics(main_metrics))
        print(f"\n⏱  Backtest completed in {elapsed:.1f}s")
        print(f"📊 Walk-forward folds: {len(wf_metrics)}")

        if wf_metrics:
            avg_sharpe = np.mean([m.sharpe_ratio for m in wf_metrics])
            avg_return = np.mean([m.total_return for m in wf_metrics])
            print(f"   Avg fold Sharpe:  {avg_sharpe:.3f}")
            print(f"   Avg fold Return:  {avg_return:.2%}")

        if mc_results:
            print(f"\n🎲 Monte Carlo ({self.mc.n_simulations} sims):")
            print(f"   Profit probability:    {mc_results.get('profit_probability', 0):.1%}")
            print(f"   Ruin probability:      {mc_results.get('ruin_probability', 0):.1%}")
            print(f"   95th pctl Max DD:      {mc_results.get('max_dd_95pct', 0):.2%}")
            print(f"   5th pctl final equity: ${mc_results.get('final_equity_5pct', 0):,.2f}")

        return {
            "metrics": main_metrics,
            "walk_forward": [m.to_dict() for m in wf_metrics],
            "monte_carlo": mc_results,
            "elapsed_seconds": elapsed,
        }

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train a simple gradient boost for backtesting speed."""
        from models.gradient_boost import GradientBoostEnsemble
        gbm = GradientBoostEnsemble()
        y_binary = (y_train > 0).astype(int)
        split = int(len(X_train) * 0.8)
        gbm.train(
            X_train[:split], y_binary[:split],
            X_train[split:], y_binary[split:],
            tune=False,
        )
        return gbm

    def _predict_model(self, model, X_test: np.ndarray) -> np.ndarray:
        """Generate signals from model predictions."""
        probs = model.predict(X_test)
        signals = (probs - 0.5) * 2  # Map [0,1] → [-1,1]
        return signals

    def _simulate_full(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        initial_equity: float,
    ):
        """Full simulation with fees and slippage."""
        equity = initial_equity
        position = 0.0
        entry_price = 0.0
        equity_curve = [equity]
        trade_pnls = []

        for i in range(1, len(signals)):
            price = prices[i]
            signal = signals[i] if i < len(signals) else 0
            prev_price = prices[i - 1]

            # PnL from existing position
            if position != 0:
                ret = (price - prev_price) / (prev_price + 1e-15)
                pnl = position * ret * equity * 0.3
                equity += pnl

            # Signal-based rebalancing
            if abs(signal) > 0.15 and np.sign(signal) != np.sign(position):
                # Close existing
                if position != 0:
                    close_pnl = (price - entry_price) / (entry_price + 1e-15) * np.sign(position)
                    realised = close_pnl * abs(position) * equity * 0.3
                    fee = abs(position) * equity * 0.3 * self.taker_fee
                    slip = abs(position) * equity * 0.3 * self.slippage_bps / 10000
                    trade_pnls.append(realised - fee - slip)

                # Open new
                position = signal * 0.5
                entry_price = price
                equity -= equity * abs(position) * 0.3 * self.taker_fee

            equity_curve.append(max(equity, 0))

        return np.array(equity_curve), trade_pnls

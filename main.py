#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  Institutional-Grade Crypto Trading Bot — Main Entry Point
═══════════════════════════════════════════════════════════════════════

Modes:
  python main.py --mode paper      Run paper trading (default)
  python main.py --mode live       Run live trading
  python main.py --mode backtest   Run backtest + report
  python main.py --mode train      Train models only

Architecture:
  1. Boot infrastructure (DB, Redis, Prometheus)
  2. Load / train models
  3. Start WebSocket feeds
  4. Main loop: receive candle → featurize → predict → decide → execute
  5. Monitor + alert via Telegram & Grafana

"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl
import structlog

# ── Structured logging setup ────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger("main")

from config.pairs import PairUniverse
from config.settings import get_settings
from data.db import Database
from data.historical import HistoricalDataLoader
from data.onchain import OnChainDataFetcher
from data.sentiment import SentimentAggregator
from data.websocket_feed import WebSocketFeed
from execution.order_manager import OrderManager
from features.pipeline import FeaturePipeline
from models.trainer import ModelTrainer
from monitoring.health_check import HealthChecker
from monitoring.prometheus_metrics import MetricsCollector
from monitoring.telegram_bot import TelegramNotifier
from strategies.position_sizer import PositionSizer
from strategies.risk_manager import RiskManager
from strategies.signal_generator import SignalDirection, SignalGenerator


class TradingBot:
    """Top-level bot orchestrator."""

    def __init__(self, mode: str = "paper"):
        self.settings = get_settings()
        self.settings.trading_mode = mode  # type: ignore
        self.mode = mode

        # Infrastructure
        self.db = Database()
        self.ws = WebSocketFeed()
        self.historical = HistoricalDataLoader(self.db)

        # Data sources
        self.onchain = OnChainDataFetcher()
        self.sentiment = SentimentAggregator()

        # Feature + Model pipeline
        self.pipeline = FeaturePipeline()
        self.trainer = ModelTrainer(self.db, self.pipeline)

        # Strategy
        self.signal_gen = SignalGenerator()
        self.sizer = PositionSizer()
        self.pair_universe = PairUniverse(
            max_correlated=self.settings.risk.max_correlated_pairs
        )
        self.risk = RiskManager(self.pair_universe)

        # Execution
        self.order_mgr = OrderManager(self.db, self.risk)

        # Monitoring
        self.metrics = MetricsCollector()
        self.telegram = TelegramNotifier()
        self.health = HealthChecker()

        # State
        self._running = False
        self._candle_buffer: Dict[str, Dict] = {}
        self._last_feature_time: Dict[str, float] = {}
        self._iteration_count = 0

    async def start(self) -> None:
        """Boot all subsystems and enter main loop."""
        log.info("bot.starting", mode=self.mode, pairs=self.settings.trading_pairs)

        # ── 1. Start infrastructure ──────────────────
        await self.db.connect()
        self.metrics.start_server(port=8000)

        # ── 2. Start data sources ────────────────────
        await self.historical.start()
        await self.onchain.start()
        await self.sentiment.start()

        # ── 3. Load or train models ──────────────────
        if not self.trainer.load_checkpoint():
            log.info("bot.no_checkpoint — training from scratch")
            await self._train_models()

        # ── 4. Start execution ───────────────────────
        await self.order_mgr.start()

        # ── 5. Start monitoring ──────────────────────
        await self.telegram.start()
        self.telegram.set_risk_manager(self.risk)

        # ── 6. Initialise risk state ─────────────────
        self.risk.reset_daily(
            equity=self.sizer.__class__.__init__  # placeholder; would be from DB
            if False else 10000.0
        )
        self.risk.current_equity = 10000.0
        self.risk.peak_equity = 10000.0

        # ── 7. Register WS callbacks ─────────────────
        self.ws.on_candle(self._on_candle)
        self.ws.on_orderbook(self._on_orderbook)

        # ── 8. Start WebSocket feeds ─────────────────
        self._running = True

        await asyncio.gather(
            self.ws.start(),
            self._main_loop(),
            self.health.run_periodic(self.settings.monitoring.health_check_interval),
            self._retrain_loop(),
            self._daily_reset_loop(),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        log.info("bot.stopping")

        await self.ws.stop()
        await self.order_mgr.stop()
        await self.historical.stop()
        await self.onchain.stop()
        await self.sentiment.stop()
        await self.telegram.stop()
        await self.db.disconnect()

        log.info("bot.stopped")

    # ── Main trading loop ────────────────────────────
    async def _main_loop(self) -> None:
        """Periodic strategy evaluation loop."""
        while self._running:
            try:
                await asyncio.sleep(15)  # Evaluate every 15 seconds
                await self._evaluate_all_pairs()
                self._iteration_count += 1

                # Update Prometheus
                self.metrics.update_uptime()
                exposure = self.risk.get_portfolio_exposure()
                self.metrics.update_positions(
                    int(exposure["n_positions"]),
                    exposure["net_exposure"],
                    exposure["gross_exposure"],
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("main_loop.error", error=str(exc))
                self.metrics.record_error("main_loop")

    async def _evaluate_all_pairs(self) -> None:
        """Evaluate signals for every trading pair."""
        for symbol in self.settings.trading_pairs:
            try:
                await self._evaluate_pair(symbol)
            except Exception as exc:
                log.error("evaluate.error", symbol=symbol, error=str(exc))

    async def _evaluate_pair(self, symbol: str) -> None:
        """Full signal generation + execution for one pair."""
        t0 = time.time()

        # ── 1. Get latest data ───────────────────────
        tf = self.settings.primary_timeframe
        df = await self.db.get_ohlcv(symbol, tf, limit=500)
        if len(df) < 200:
            return

        # ── 2. Get orderbook from cache ──────────────
        ob = await self.ws.get_cached_orderbook(symbol)

        # ── 3. Feature engineering ───────────────────
        asset = symbol.split("/")[0]

        # On-chain + sentiment (fetch less frequently)
        onchain_data = None
        sentiment_data = None
        if self._iteration_count % 20 == 0:  # Every ~5 minutes
            try:
                onchain_data = await self.onchain.fetch_all_metrics(asset)
                sentiment_data = await self.sentiment.get_sentiment(asset)
            except Exception:
                pass

        enriched = self.pipeline.transform(
            df, symbol=symbol,
            orderbook=ob,
            onchain=onchain_data,
            sentiment=sentiment_data,
        )

        # ── 4. Model prediction ──────────────────────
        feature_cols = [
            c for c in enriched.columns
            if c not in ("time", "open", "high", "low", "close", "volume")
        ]
        X = enriched.select(feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        prices = enriched["close"].to_numpy()

        signal_arr, uncertainty_arr = self.trainer.predict_with_uncertainty(X, prices)

        raw_signal = float(signal_arr[-1]) if len(signal_arr) > 0 else 0.0
        uncertainty = float(uncertainty_arr[-1]) if len(uncertainty_arr) > 0 else 1.0

        # Record to Prometheus
        self.metrics.model_signal_value.labels(symbol=symbol).set(raw_signal)
        self.metrics.model_uncertainty.labels(symbol=symbol).set(uncertainty)
        self.metrics.model_predict_latency.observe(time.time() - t0)

        # ── 5. Get ATR for sizing ────────────────────
        atr = 0.0
        if "atr_14" in enriched.columns:
            atr_vals = enriched["atr_14"].to_numpy()
            atr = float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else 0.0

        # ── 6. Get regime ────────────────────────────
        regime = 0
        if "regime_vol_regime" in enriched.columns:
            regime = int(enriched["regime_vol_regime"].to_numpy()[-1])
        self.metrics.current_regime.set(regime)

        # ── 7. Generate trade signal ─────────────────
        ob_features = None
        if ob:
            from features.orderbook import compute_orderbook_features
            ob_features = compute_orderbook_features(ob)

        current_price = float(prices[-1])
        trade_signal = self.signal_gen.generate(
            symbol=symbol,
            raw_signal=raw_signal,
            uncertainty=uncertainty,
            current_price=current_price,
            atr=atr,
            regime=regime,
            orderbook_features=ob_features,
            sentiment=sentiment_data,
            onchain=onchain_data,
        )

        self.metrics.record_signal(trade_signal.direction.value)

        # ── 8. Check existing stops ──────────────────
        stopped = await self.order_mgr.check_and_close_stops({symbol: current_price})

        # ── 9. Execute if actionable ─────────────────
        if trade_signal.direction != SignalDirection.FLAT:
            quantity = self.sizer.compute(
                account_equity=self.risk.current_equity,
                signal_strength=trade_signal.strength,
                entry_price=current_price,
                stop_loss=trade_signal.stop_loss,
                atr=atr,
            )

            if quantity > 0:
                result = await self.order_mgr.execute_signal(trade_signal, quantity)
                if result:
                    self.metrics.record_trade(trade_signal.direction.value, symbol)
                    await self.telegram.send_trade(
                        f"{symbol} {trade_signal.direction.value.upper()} "
                        f"qty={quantity:.6f} @ {current_price:.2f}\n"
                        f"Signal: {raw_signal:.3f} | Regime: {trade_signal.regime}"
                    )

    # ── WebSocket callbacks ──────────────────────────
    async def _on_candle(self, data: Dict) -> None:
        """Handle incoming candle data."""
        symbol = data["symbol"]
        self._candle_buffer[symbol] = data
        # Insert to DB (throttled)
        last = self._last_feature_time.get(symbol, 0)
        if time.time() - last > 10:
            await self.db.insert_ohlcv(symbol, data["timeframe"], [data])
            self._last_feature_time[symbol] = time.time()

    async def _on_orderbook(self, data: Dict) -> None:
        """Handle incoming orderbook snapshot."""
        pass  # Cached in Redis by WebSocketFeed

    # ── Periodic tasks ───────────────────────────────
    async def _retrain_loop(self) -> None:
        """Periodically retrain models on fresh data."""
        while self._running:
            await asyncio.sleep(3600)  # Check hourly
            try:
                data = await self.historical.fetch_all_pairs(
                    timeframe=self.settings.primary_timeframe, days=90
                )
                result = await self.trainer.online_retrain(data)
                if result:
                    await self.telegram.send_alert(f"Models retrained: {result}")
            except Exception as exc:
                log.error("retrain.error", error=str(exc))

    async def _daily_reset_loop(self) -> None:
        """Reset daily counters at UTC midnight."""
        while self._running:
            now = datetime.now(timezone.utc)
            next_midnight = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if next_midnight <= now:
                from datetime import timedelta
                next_midnight += timedelta(days=1)

            wait = (next_midnight - now).total_seconds()
            await asyncio.sleep(wait)

            self.risk.reset_daily(self.risk.current_equity)
            await self.db.record_equity(
                self.risk.current_equity,
                (self.risk.peak_equity - self.risk.current_equity) / self.risk.peak_equity,
                self.risk.daily_pnl,
            )
            await self.telegram.send_daily_report(
                f"Equity: ${self.risk.current_equity:,.2f}\n"
                f"Daily P&L: ${self.risk.daily_pnl:,.2f}\n"
                f"Positions: {len(self.risk.positions)}"
            )

    async def _train_models(self) -> None:
        """Full model training from historical data."""
        log.info("bot.training_models")
        data = await self.historical.fetch_all_pairs(
            timeframe=self.settings.primary_timeframe, days=365
        )
        await self.trainer.initial_train(data)

    # ── Backtest mode ────────────────────────────────
    async def run_backtest(self) -> None:
        """Run offline backtest."""
        from backtest.engine import BacktestEngine

        log.info("backtest.starting")

        await self.db.connect()
        await self.historical.start()

        data = await self.historical.fetch_all_pairs(
            timeframe=self.settings.primary_timeframe, days=365
        )

        engine = BacktestEngine(initial_equity=10000)
        results = engine.run(data, walk_forward=True)

        await self.historical.stop()
        await self.db.disconnect()

        return results


# ── Entry Point ──────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Institutional-Grade Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["paper", "live", "backtest", "train"],
        default="paper", help="Trading mode (default: paper)",
    )
    return parser.parse_args()


async def async_main(mode: str) -> None:
    bot = TradingBot(mode=mode)

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown_handler)
        except NotImplementedError:
            pass  # Windows

    if mode == "backtest":
        await bot.run_backtest()
    elif mode == "train":
        await bot.db.connect()
        await bot.historical.start()
        await bot._train_models()
        await bot.historical.stop()
        await bot.db.disconnect()
        log.info("Training complete.")
    else:
        await bot.start()


def main() -> None:
    args = parse_args()
    log.info("bot.boot", mode=args.mode, python=sys.version)

    try:
        import uvloop
        uvloop.install()
        log.info("uvloop installed")
    except ImportError:
        pass

    asyncio.run(async_main(args.mode))


if __name__ == "__main__":
    main()

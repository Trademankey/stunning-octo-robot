"""
Prometheus metrics exporter.

Exposes key trading metrics for scraping:
  - Equity, P&L, drawdown
  - Position count, exposure
  - Signal counts, model latency
  - System health (uptime, memory, errors)
"""
from __future__ import annotations

import time
from typing import Optional

import structlog
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

from config.settings import get_settings

log = structlog.get_logger(__name__)


class MetricsCollector:
    """Prometheus metrics for the trading bot."""

    def __init__(self):
        # ── Portfolio ────────────────────────────────
        self.equity = Gauge("bot_equity_usd", "Current account equity in USD")
        self.daily_pnl = Gauge("bot_daily_pnl_usd", "Daily realised P&L")
        self.unrealised_pnl = Gauge("bot_unrealised_pnl_usd", "Unrealised P&L")
        self.drawdown = Gauge("bot_drawdown_pct", "Current drawdown percentage")
        self.max_drawdown = Gauge("bot_max_drawdown_pct", "Maximum drawdown percentage")

        # ── Positions ────────────────────────────────
        self.open_positions = Gauge("bot_open_positions", "Number of open positions")
        self.net_exposure = Gauge("bot_net_exposure_usd", "Net exposure in USD")
        self.gross_exposure = Gauge("bot_gross_exposure_usd", "Gross exposure in USD")

        # ── Trading ──────────────────────────────────
        self.trades_total = Counter("bot_trades_total", "Total trades executed", ["side", "symbol"])
        self.signals_generated = Counter("bot_signals_total", "Total signals generated", ["direction"])
        self.orders_rejected = Counter("bot_orders_rejected_total", "Orders rejected by risk", ["reason"])

        # ── Model ────────────────────────────────────
        self.model_predict_latency = Histogram(
            "bot_model_predict_seconds", "Model prediction latency",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        )
        self.model_signal_value = Gauge("bot_model_signal", "Latest model signal", ["symbol"])
        self.model_uncertainty = Gauge("bot_model_uncertainty", "Model uncertainty", ["symbol"])

        # ── System ───────────────────────────────────
        self.uptime = Gauge("bot_uptime_seconds", "Bot uptime in seconds")
        self.errors_total = Counter("bot_errors_total", "Total errors", ["component"])
        self.ws_reconnects = Counter("bot_ws_reconnects_total", "WebSocket reconnections")

        # ── Regime ───────────────────────────────────
        self.current_regime = Gauge("bot_market_regime", "Current market regime code")

        self._start_time = time.time()
        self._started = False

    def start_server(self, port: Optional[int] = None) -> None:
        """Start Prometheus HTTP metrics server."""
        if self._started:
            return
        if port is None:
            port = 8000
        try:
            start_http_server(port)
            self._started = True
            log.info("prometheus.started", port=port)
        except Exception as exc:
            log.error("prometheus.start_failed", error=str(exc))

    def update_portfolio(
        self,
        equity: float,
        daily_pnl: float,
        unrealised: float,
        drawdown: float,
        max_dd: float,
    ) -> None:
        self.equity.set(equity)
        self.daily_pnl.set(daily_pnl)
        self.unrealised_pnl.set(unrealised)
        self.drawdown.set(drawdown)
        self.max_drawdown.set(max_dd)

    def update_positions(self, n_positions: int, net_exp: float, gross_exp: float) -> None:
        self.open_positions.set(n_positions)
        self.net_exposure.set(net_exp)
        self.gross_exposure.set(gross_exp)

    def record_trade(self, side: str, symbol: str) -> None:
        self.trades_total.labels(side=side, symbol=symbol).inc()

    def record_signal(self, direction: str) -> None:
        self.signals_generated.labels(direction=direction).inc()

    def record_rejection(self, reason: str) -> None:
        self.orders_rejected.labels(reason=reason).inc()

    def record_error(self, component: str) -> None:
        self.errors_total.labels(component=component).inc()

    def update_uptime(self) -> None:
        self.uptime.set(time.time() - self._start_time)

"""
Order Manager — routes signals to paper or live execution.

Features:
  - Paper / live mode switching
  - TWAP / VWAP smart order routing
  - WebSocket order status tracking
  - Graceful error handling + circuit breaker
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import structlog

from config.settings import get_settings
from data.db import Database
from execution.live_trader import LiveTrader
from execution.paper_trader import PaperTrader
from execution.smart_router import SmartOrderRouter
from strategies.risk_manager import RiskManager
from strategies.signal_generator import SignalDirection, TradeSignal

log = structlog.get_logger(__name__)


class OrderManager:
    """
    Central order management — dispatches to paper or live execution.
    Tracks open orders, fills, and manages smart order routing.
    """

    def __init__(self, db: Database, risk_manager: RiskManager):
        self._settings = get_settings()
        self._db = db
        self._risk = risk_manager
        self._paper = PaperTrader(db)
        self._live = LiveTrader(db) if self._settings.trading_mode == "live" else None
        self._router = SmartOrderRouter()
        self._is_live = self._settings.trading_mode == "live"
        self._pending_orders: Dict[str, Dict] = {}
        self._circuit_breaker_fails = 0
        self._circuit_breaker_max = 5

    async def start(self) -> None:
        if self._is_live and self._live:
            await self._live.start()
        log.info("order_manager.started", mode=self._settings.trading_mode)

    async def stop(self) -> None:
        if self._is_live and self._live:
            await self._live.stop()
        log.info("order_manager.stopped")

    async def execute_signal(
        self,
        signal: TradeSignal,
        quantity: float,
        use_smart_routing: bool = True,
    ) -> Optional[Dict]:
        """
        Execute a trade signal.
        Returns order result dict or None if blocked.
        """
        # Risk check
        allowed, reason = self._risk.can_trade(signal)
        if not allowed:
            log.info("order.blocked", symbol=signal.symbol, reason=reason)
            return None

        # Circuit breaker
        if self._circuit_breaker_fails >= self._circuit_breaker_max:
            log.error("order.circuit_breaker_open", fails=self._circuit_breaker_fails)
            return None

        try:
            if self._is_live and self._live:
                if use_smart_routing and quantity * signal.entry_price > 10000:
                    result = await self._router.execute_twap(
                        self._live, signal, quantity,
                        n_slices=5, interval_seconds=10,
                    )
                else:
                    result = await self._live.execute(signal, quantity)
            else:
                result = await self._paper.execute(signal, quantity)

            if result:
                # Register position
                self._risk.open_position(signal, quantity)

                # Store trade in DB
                await self._db.insert_trade({
                    "symbol": signal.symbol,
                    "side": signal.direction.value,
                    "order_type": "market",
                    "quantity": quantity,
                    "price": signal.entry_price,
                    "strategy": signal.regime,
                    "model_signal": signal.raw_signal,
                    "regime": signal.regime,
                })

                self._circuit_breaker_fails = 0
                log.info(
                    "order.executed",
                    symbol=signal.symbol,
                    side=signal.direction.value,
                    qty=quantity,
                    price=signal.entry_price,
                )
                return result

        except Exception as exc:
            self._circuit_breaker_fails += 1
            log.error(
                "order.execution_failed",
                symbol=signal.symbol, error=str(exc),
                circuit_breaker=self._circuit_breaker_fails,
            )

        return None

    async def close_position(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Close an existing position."""
        pos = self._risk.positions.get(symbol)
        if pos is None:
            return None

        signal = TradeSignal(
            symbol=symbol,
            direction=SignalDirection.SHORT if pos.side == "long" else SignalDirection.LONG,
            strength=1.0,
            raw_signal=0.0,
            regime="close",
            entry_price=current_price,
        )

        try:
            if self._is_live and self._live:
                result = await self._live.execute(signal, pos.quantity)
            else:
                result = await self._paper.execute(signal, pos.quantity)

            pnl = self._risk.close_position(symbol, current_price)

            await self._db.insert_trade({
                "symbol": symbol,
                "side": signal.direction.value,
                "order_type": "market",
                "quantity": pos.quantity,
                "price": current_price,
                "pnl": pnl,
                "strategy": "close",
            })

            return result

        except Exception as exc:
            log.error("order.close_failed", symbol=symbol, error=str(exc))
            return None

    async def check_and_close_stops(self, prices: Dict[str, float]) -> List[str]:
        """Check all stops and close triggered positions."""
        # Hard stops
        hard_stops = self._risk.check_stops(prices)
        # Trailing stops
        trail_stops = self._risk.update_trailing_stops(prices)

        all_stops = list(set(hard_stops + trail_stops))
        for symbol in all_stops:
            price = prices.get(symbol)
            if price:
                await self.close_position(symbol, price)

        return all_stops

    def reset_circuit_breaker(self) -> None:
        self._circuit_breaker_fails = 0
        log.info("order.circuit_breaker_reset")

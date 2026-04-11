"""
Live Trader — real exchange execution via CCXT.

Features:
  - Spot and USDT-margined futures support
  - Hedge mode position management
  - WebSocket order update streaming
  - Exponential backoff on failures
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import ccxt.async_support as ccxt_async
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from data.db import Database
from strategies.signal_generator import SignalDirection, TradeSignal

log = structlog.get_logger(__name__)


class LiveTrader:
    """Real exchange order execution."""

    def __init__(self, db: Database):
        self._settings = get_settings()
        self._db = db
        self._exchange: Optional[ccxt_async.Exchange] = None

    async def start(self) -> None:
        exchange_cls = getattr(ccxt_async, self._settings.exchange.id)
        self._exchange = exchange_cls({
            "apiKey": self._settings.exchange.api_key.get_secret_value(),
            "secret": self._settings.exchange.api_secret.get_secret_value(),
            "sandbox": self._settings.exchange.sandbox,
            "enableRateLimit": True,
            "options": {
                "defaultType": self._settings.market_type,
                "hedgeMode": self._settings.hedge_mode,
            },
        })
        await self._exchange.load_markets()

        # Set hedge mode if futures
        if self._settings.market_type == "futures" and self._settings.hedge_mode:
            try:
                await self._exchange.set_position_mode(hedged=True)
                log.info("live_trader.hedge_mode_set")
            except Exception as exc:
                log.warning("live_trader.hedge_mode_failed", error=str(exc))

        log.info("live_trader.started", exchange=self._settings.exchange.id)

    async def stop(self) -> None:
        if self._exchange:
            await self._exchange.close()
        log.info("live_trader.stopped")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def execute(self, signal: TradeSignal, quantity: float) -> Dict[str, Any]:
        """Place a market order on the exchange."""
        if self._exchange is None:
            raise RuntimeError("Exchange not connected")

        side = "buy" if signal.direction == SignalDirection.LONG else "sell"

        params = {}
        if self._settings.market_type == "futures" and self._settings.hedge_mode:
            params["positionSide"] = "LONG" if side == "buy" else "SHORT"

        try:
            order = await self._exchange.create_market_order(
                signal.symbol, side, quantity, params=params
            )

            result = {
                "order_id": order["id"],
                "symbol": signal.symbol,
                "side": side,
                "quantity": order.get("filled", quantity),
                "price": order.get("average", signal.entry_price),
                "fee": order.get("fee", {}).get("cost", 0),
                "status": order["status"],
                "timestamp": order.get("timestamp"),
            }

            log.info("live.order_placed", **result)
            return result

        except ccxt_async.InsufficientFunds as exc:
            log.error("live.insufficient_funds", error=str(exc))
            raise
        except ccxt_async.ExchangeError as exc:
            log.error("live.exchange_error", error=str(exc))
            raise

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a futures pair."""
        if self._exchange and self._settings.market_type == "futures":
            try:
                await self._exchange.set_leverage(leverage, symbol)
                log.info("live.leverage_set", symbol=symbol, leverage=leverage)
            except Exception as exc:
                log.warning("live.leverage_failed", error=str(exc))

    async def set_stop_loss(
        self, symbol: str, side: str, stop_price: float, quantity: float
    ) -> Optional[Dict]:
        """Place a stop-loss order."""
        if self._exchange is None:
            return None

        order_side = "sell" if side == "long" else "buy"
        params = {"stopPrice": stop_price}

        if self._settings.hedge_mode:
            params["positionSide"] = "LONG" if side == "long" else "SHORT"

        try:
            order = await self._exchange.create_order(
                symbol, "stop_market", order_side, quantity, params=params
            )
            log.info("live.stop_placed", symbol=symbol, stop=stop_price)
            return order
        except Exception as exc:
            log.error("live.stop_failed", error=str(exc))
            return None

    async def get_balance(self) -> Dict[str, float]:
        """Fetch account balance."""
        if self._exchange is None:
            return {}
        try:
            balance = await self._exchange.fetch_balance()
            return {
                "total": balance.get("total", {}).get("USDT", 0),
                "free": balance.get("free", {}).get("USDT", 0),
                "used": balance.get("used", {}).get("USDT", 0),
            }
        except Exception as exc:
            log.error("live.balance_error", error=str(exc))
            return {}

    async def get_positions(self) -> list:
        """Fetch open positions (futures)."""
        if self._exchange is None:
            return []
        try:
            positions = await self._exchange.fetch_positions()
            return [p for p in positions if float(p.get("contracts", 0)) > 0]
        except Exception as exc:
            log.error("live.positions_error", error=str(exc))
            return []

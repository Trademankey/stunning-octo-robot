"""
Real-time WebSocket data feed via CCXT Pro.

Streams OHLCV, order-book, and trades for every configured pair.
Includes exponential backoff reconnection and Redis caching.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

import ccxt.pro as ccxtpro
import orjson
import redis.asyncio as aioredis
import structlog

from config.settings import get_settings

log = structlog.get_logger(__name__)

CallbackType = Callable[..., Coroutine[Any, Any, None]]


class WebSocketFeed:
    """CCXT-Pro WebSocket manager with graceful reconnection."""

    MAX_RETRIES = 50
    BASE_DELAY = 1.0  # seconds
    MAX_DELAY = 60.0

    def __init__(self):
        self._settings = get_settings()
        self._exchange: Optional[ccxtpro.Exchange] = None
        self._redis: Optional[aioredis.Redis] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Callback registries
        self._on_candle: List[CallbackType] = []
        self._on_orderbook: List[CallbackType] = []
        self._on_trade: List[CallbackType] = []

    # ── public API ───────────────────────────────────
    def on_candle(self, cb: CallbackType) -> None:
        self._on_candle.append(cb)

    def on_orderbook(self, cb: CallbackType) -> None:
        self._on_orderbook.append(cb)

    def on_trade(self, cb: CallbackType) -> None:
        self._on_trade.append(cb)

    async def start(self) -> None:
        self._exchange = self._create_exchange()
        self._redis = aioredis.from_url(
            self._settings.redis.url, decode_responses=False
        )
        self._running = True

        for symbol in self._settings.trading_pairs:
            self._tasks.append(asyncio.create_task(
                self._watch_ohlcv(symbol), name=f"ohlcv-{symbol}"
            ))
            self._tasks.append(asyncio.create_task(
                self._watch_orderbook(symbol), name=f"ob-{symbol}"
            ))
            self._tasks.append(asyncio.create_task(
                self._watch_trades(symbol), name=f"trades-{symbol}"
            ))
        log.info("websocket_feed.started", pairs=self._settings.trading_pairs)

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._exchange:
            await self._exchange.close()
        if self._redis:
            await self._redis.close()
        log.info("websocket_feed.stopped")

    # ── internal loops ───────────────────────────────
    async def _watch_ohlcv(self, symbol: str) -> None:
        retries = 0
        while self._running:
            try:
                for tf in self._settings.timeframes:
                    candles = await self._exchange.watch_ohlcv(symbol, tf, limit=5)
                    if candles:
                        latest = candles[-1]
                        data = {
                            "symbol": symbol, "timeframe": tf,
                            "timestamp": latest[0], "open": latest[1],
                            "high": latest[2], "low": latest[3],
                            "close": latest[4], "volume": latest[5],
                        }
                        # Cache in Redis (60s TTL per tf)
                        cache_key = f"ohlcv:{symbol}:{tf}"
                        await self._redis.set(
                            cache_key, orjson.dumps(data), ex=60
                        )
                        for cb in self._on_candle:
                            await cb(data)
                retries = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                retries += 1
                delay = min(self.BASE_DELAY * (2 ** retries), self.MAX_DELAY)
                log.warning(
                    "ws.ohlcv.reconnect",
                    symbol=symbol, error=str(exc), retry=retries, delay=delay,
                )
                if retries >= self.MAX_RETRIES:
                    log.error("ws.ohlcv.max_retries", symbol=symbol)
                    break
                await asyncio.sleep(delay)

    async def _watch_orderbook(self, symbol: str) -> None:
        retries = 0
        while self._running:
            try:
                ob = await self._exchange.watch_order_book(symbol, limit=25)
                compact = {
                    "symbol": symbol,
                    "bids": ob["bids"][:10],
                    "asks": ob["asks"][:10],
                    "timestamp": ob.get("timestamp", int(time.time() * 1000)),
                }
                cache_key = f"orderbook:{symbol}"
                await self._redis.set(
                    cache_key, orjson.dumps(compact), ex=5
                )
                for cb in self._on_orderbook:
                    await cb(compact)
                retries = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                retries += 1
                delay = min(self.BASE_DELAY * (2 ** retries), self.MAX_DELAY)
                log.warning("ws.ob.reconnect", symbol=symbol, error=str(exc), delay=delay)
                if retries >= self.MAX_RETRIES:
                    break
                await asyncio.sleep(delay)

    async def _watch_trades(self, symbol: str) -> None:
        retries = 0
        while self._running:
            try:
                trades = await self._exchange.watch_trades(symbol)
                for t in trades:
                    data = {
                        "symbol": symbol,
                        "price": t["price"],
                        "amount": t["amount"],
                        "side": t["side"],
                        "timestamp": t["timestamp"],
                    }
                    for cb in self._on_trade:
                        await cb(data)
                retries = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                retries += 1
                delay = min(self.BASE_DELAY * (2 ** retries), self.MAX_DELAY)
                log.warning("ws.trades.reconnect", symbol=symbol, error=str(exc), delay=delay)
                if retries >= self.MAX_RETRIES:
                    break
                await asyncio.sleep(delay)

    # ── helpers ──────────────────────────────────────
    def _create_exchange(self) -> ccxtpro.Exchange:
        exchange_class = getattr(ccxtpro, self._settings.exchange.id)
        return exchange_class({
            "apiKey": self._settings.exchange.api_key.get_secret_value(),
            "secret": self._settings.exchange.api_secret.get_secret_value(),
            "sandbox": self._settings.exchange.sandbox,
            "enableRateLimit": True,
            "rateLimit": self._settings.exchange.rate_limit,
            "options": {
                "defaultType": self._settings.market_type,
                "hedgeMode": self._settings.hedge_mode,
            },
        })

    async def get_cached_orderbook(self, symbol: str) -> Optional[Dict]:
        raw = await self._redis.get(f"orderbook:{symbol}")
        return orjson.loads(raw) if raw else None

    async def get_cached_candle(self, symbol: str, tf: str) -> Optional[Dict]:
        raw = await self._redis.get(f"ohlcv:{symbol}:{tf}")
        return orjson.loads(raw) if raw else None

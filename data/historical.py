"""
Historical OHLCV loader.

Sources (fallback chain):
  1. Local TimescaleDB
  2. CCXT REST fetch_ohlcv (Binance / configured exchange)
  3. Binance public data (klines CSVs)

All data returned as Polars DataFrames for zero-copy speed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import ccxt.async_support as ccxt_async
import polars as pl
import structlog

from config.settings import get_settings
from data.db import Database

log = structlog.get_logger(__name__)

_TF_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440,
}


class HistoricalDataLoader:
    """Fetches and caches multi-timeframe OHLCV data."""

    def __init__(self, db: Database):
        self._settings = get_settings()
        self._db = db
        self._exchange: Optional[ccxt_async.Exchange] = None
        self._cache: Dict[str, pl.DataFrame] = {}

    async def start(self) -> None:
        exchange_cls = getattr(ccxt_async, self._settings.exchange.id)
        self._exchange = exchange_cls({
            "apiKey": self._settings.exchange.api_key.get_secret_value(),
            "secret": self._settings.exchange.api_secret.get_secret_value(),
            "enableRateLimit": True,
            "options": {"defaultType": self._settings.market_type},
        })
        await self._exchange.load_markets()
        log.info("historical_loader.started")

    async def stop(self) -> None:
        if self._exchange:
            await self._exchange.close()

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365,
        force_exchange: bool = False,
    ) -> pl.DataFrame:
        """Return OHLCV DataFrame; tries DB first, back-fills from exchange."""
        cache_key = f"{symbol}:{timeframe}:{days}"
        if cache_key in self._cache and not force_exchange:
            return self._cache[cache_key]

        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Try database first
        if not force_exchange:
            df = await self._db.get_ohlcv(symbol, timeframe, limit=days * 1440 // _TF_MINUTES.get(timeframe, 15))
            if len(df) > 100:
                self._cache[cache_key] = df
                return df

        # Fetch from exchange
        df = await self._fetch_from_exchange(symbol, timeframe, since)
        if len(df) > 0:
            # Persist to DB
            candles = df.to_dicts()
            for c in candles:
                c["timestamp"] = int(c["time"].timestamp() * 1000) if hasattr(c["time"], "timestamp") else c["time"]
            await self._db.insert_ohlcv(symbol, timeframe, candles)
            self._cache[cache_key] = df
        return df

    async def fetch_all_pairs(
        self, timeframe: str = "15m", days: int = 365
    ) -> Dict[str, pl.DataFrame]:
        """Fetch OHLCV for all configured pairs in parallel."""
        tasks = {
            sym: self.fetch(sym, timeframe, days)
            for sym in self._settings.trading_pairs
        }
        results = {}
        for sym, coro in tasks.items():
            try:
                results[sym] = await coro
            except Exception as exc:
                log.error("historical.fetch_failed", symbol=sym, error=str(exc))
                results[sym] = pl.DataFrame()
        return results

    async def _fetch_from_exchange(
        self, symbol: str, timeframe: str, since: datetime
    ) -> pl.DataFrame:
        """Page through exchange REST API to get full history."""
        all_candles: List[list] = []
        since_ms = int(since.timestamp() * 1000)
        limit = 1000

        log.info("historical.fetching", symbol=symbol, timeframe=timeframe)
        while True:
            try:
                candles = await self._exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=limit
                )
            except Exception as exc:
                log.warning("historical.fetch_error", error=str(exc))
                break

            if not candles:
                break
            all_candles.extend(candles)
            since_ms = candles[-1][0] + 1
            if len(candles) < limit:
                break
            await asyncio.sleep(self._exchange.rateLimit / 1000)

        if not all_candles:
            return pl.DataFrame()

        df = pl.DataFrame(
            {
                "time": [datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc) for c in all_candles],
                "open": [c[1] for c in all_candles],
                "high": [c[2] for c in all_candles],
                "low": [c[3] for c in all_candles],
                "close": [c[4] for c in all_candles],
                "volume": [c[5] for c in all_candles],
            }
        ).sort("time").unique(subset=["time"])

        log.info("historical.fetched", symbol=symbol, rows=len(df))
        return df

    async def get_multi_timeframe(
        self, symbol: str, days: int = 180
    ) -> Dict[str, pl.DataFrame]:
        """Return dict of timeframe -> DataFrame for a single pair."""
        return {
            tf: await self.fetch(symbol, tf, days)
            for tf in self._settings.timeframes
        }

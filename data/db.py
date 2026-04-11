"""
Async PostgreSQL + TimescaleDB connection pool & CRUD helpers.

Uses asyncpg for raw speed, SQLAlchemy async for complex queries.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
import polars as pl
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings

log = structlog.get_logger(__name__)


class Database:
    """Manages a high-performance asyncpg connection pool."""

    def __init__(self):
        self._settings = get_settings().database
        self._pool: Optional[asyncpg.Pool] = None
        self._engine = None

    # ── lifecycle ────────────────────────────────────
    async def connect(self) -> None:
        dsn = (
            f"postgresql://{self._settings.user}:"
            f"{self._settings.password.get_secret_value()}"
            f"@{self._settings.host}:{self._settings.port}"
            f"/{self._settings.db}"
        )
        self._pool = await asyncpg.create_pool(
            dsn, min_size=5, max_size=20, command_timeout=60,
        )
        self._engine = create_async_engine(
            self._settings.dsn, pool_size=10, max_overflow=5,
        )
        log.info("database.connected", host=self._settings.host)

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
        if self._engine:
            await self._engine.dispose()
        log.info("database.disconnected")

    # ── OHLCV ────────────────────────────────────────
    async def insert_ohlcv(
        self, symbol: str, timeframe: str, candles: List[Dict[str, Any]]
    ) -> int:
        if not candles:
            return 0
        rows = [
            (
                datetime.utcfromtimestamp(c["timestamp"] / 1000),
                symbol, timeframe,
                c["open"], c["high"], c["low"], c["close"], c["volume"],
            )
            for c in candles
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
        return len(rows)

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
        since: Optional[datetime] = None,
    ) -> pl.DataFrame:
        query = """
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = $1 AND timeframe = $2
        """
        params: list = [symbol, timeframe]
        if since:
            query += " AND time >= $3"
            params.append(since)
        query += " ORDER BY time DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame()

        return pl.DataFrame(
            {
                "time": [r["time"] for r in rows],
                "open": [float(r["open"]) for r in rows],
                "high": [float(r["high"]) for r in rows],
                "low": [float(r["low"]) for r in rows],
                "close": [float(r["close"]) for r in rows],
                "volume": [float(r["volume"]) for r in rows],
            }
        ).sort("time")

    # ── Trades ───────────────────────────────────────
    async def insert_trade(self, trade: Dict[str, Any]) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO trades
                    (symbol, side, order_type, quantity, price, fee, pnl, strategy, model_signal, regime, notes)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                RETURNING id
                """,
                trade["symbol"], trade["side"], trade["order_type"],
                trade["quantity"], trade["price"],
                trade.get("fee", 0), trade.get("pnl", 0),
                trade.get("strategy", ""), trade.get("model_signal", 0),
                trade.get("regime", ""), trade.get("notes", "{}"),
            )
        return row["id"]

    # ── Positions ────────────────────────────────────
    async def upsert_position(self, pos: Dict[str, Any]) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO positions
                    (symbol, side, entry_price, quantity, stop_loss, take_profit, strategy, status)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (id) DO UPDATE SET
                    exit_price  = EXCLUDED.exit_price,
                    closed_at   = EXCLUDED.closed_at,
                    pnl         = EXCLUDED.pnl,
                    pnl_pct     = EXCLUDED.pnl_pct,
                    status      = EXCLUDED.status
                RETURNING id
                """,
                pos["symbol"], pos["side"], pos["entry_price"],
                pos["quantity"], pos.get("stop_loss", 0),
                pos.get("take_profit", 0), pos.get("strategy", ""),
                pos.get("status", "open"),
            )
        return row["id"]

    async def close_position(self, pos_id: int, exit_price: float, pnl: float) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE positions SET
                    exit_price = $2, pnl = $3, pnl_pct = $4,
                    closed_at = NOW(), status = 'closed'
                WHERE id = $1
                """,
                pos_id, exit_price, pnl, 0.0,
            )

    # ── Equity Curve ─────────────────────────────────
    async def record_equity(self, equity: float, drawdown: float, daily_pnl: float) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO equity_curve (time, equity, drawdown, daily_pnl) VALUES (NOW(), $1, $2, $3)",
                equity, drawdown, daily_pnl,
            )

    # ── Model Metrics ────────────────────────────────
    async def record_model_metric(
        self, model_name: str, metric_name: str, value: float, meta: str = "{}"
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO model_metrics (model_name, metric_name, metric_value, metadata)
                   VALUES ($1,$2,$3,$4::jsonb)""",
                model_name, metric_name, value, meta,
            )

    # ── Feature Snapshots ────────────────────────────
    async def save_features(self, symbol: str, features_json: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO feature_snapshots (time, symbol, features) VALUES (NOW(), $1, $2::jsonb)",
                symbol, features_json,
            )

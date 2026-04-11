"""
On-chain data fetcher — Dune Analytics & Glassnode.

Captures: funding rates, open interest, liquidations, whale flows,
exchange in/outflows, MVRV, SOPR, NVT.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import polars as pl
import structlog

from config.settings import get_settings

log = structlog.get_logger(__name__)


class OnChainDataFetcher:
    """Async on-chain metrics aggregator."""

    GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"
    DUNE_BASE = "https://api.dune.com/api/v1"

    def __init__(self):
        s = get_settings()
        self._glassnode_key = s.onchain.glassnode_api_key.get_secret_value()
        self._dune_key = s.onchain.dune_api_key.get_secret_value()
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        log.info("onchain.started")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ── Glassnode endpoints ──────────────────────────
    async def get_funding_rate(self, asset: str = "BTC") -> Optional[float]:
        return await self._glassnode_metric(
            "derivatives/futures_funding_rate_perpetual", asset
        )

    async def get_open_interest(self, asset: str = "BTC") -> Optional[float]:
        return await self._glassnode_metric(
            "derivatives/futures_open_interest_sum", asset
        )

    async def get_exchange_netflow(self, asset: str = "BTC") -> Optional[float]:
        return await self._glassnode_metric(
            "transactions/transfers_to_exchanges_count", asset
        )

    async def get_nvt_ratio(self, asset: str = "BTC") -> Optional[float]:
        return await self._glassnode_metric("indicators/nvt", asset)

    async def get_mvrv(self, asset: str = "BTC") -> Optional[float]:
        return await self._glassnode_metric("market/mvrv", asset)

    async def get_sopr(self, asset: str = "BTC") -> Optional[float]:
        return await self._glassnode_metric("indicators/sopr", asset)

    async def get_whale_transactions(self, asset: str = "BTC") -> Optional[float]:
        """Transactions > $1M."""
        return await self._glassnode_metric(
            "transactions/transfers_volume_large_count", asset
        )

    async def fetch_all_metrics(self, asset: str = "BTC") -> Dict[str, Optional[float]]:
        """Fetch all on-chain metrics concurrently."""
        tasks = {
            "funding_rate": self.get_funding_rate(asset),
            "open_interest": self.get_open_interest(asset),
            "exchange_netflow": self.get_exchange_netflow(asset),
            "nvt_ratio": self.get_nvt_ratio(asset),
            "mvrv": self.get_mvrv(asset),
            "sopr": self.get_sopr(asset),
            "whale_txns": self.get_whale_transactions(asset),
        }
        results = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as exc:
                log.warning("onchain.metric_failed", metric=name, error=str(exc))
                results[name] = None
        return results

    # ── Dune Analytics ───────────────────────────────
    async def run_dune_query(self, query_id: int) -> Optional[List[Dict]]:
        """Execute a Dune query and return results."""
        if not self._dune_key:
            return None
        headers = {"X-DUNE-API-KEY": self._dune_key}
        try:
            # Trigger execution
            async with self._session.post(
                f"{self.DUNE_BASE}/query/{query_id}/execute",
                headers=headers,
            ) as resp:
                data = await resp.json()
                execution_id = data.get("execution_id")

            if not execution_id:
                return None

            # Poll for results
            for _ in range(30):
                async with self._session.get(
                    f"{self.DUNE_BASE}/execution/{execution_id}/results",
                    headers=headers,
                ) as resp:
                    data = await resp.json()
                    state = data.get("state")
                    if state == "QUERY_STATE_COMPLETED":
                        return data.get("result", {}).get("rows", [])
                    elif state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                        return None
                await asyncio.sleep(2)
        except Exception as exc:
            log.error("dune.query_error", query_id=query_id, error=str(exc))
        return None

    async def get_liquidations(self) -> Optional[List[Dict]]:
        """Fetch recent liquidation data from Dune (user-configured query)."""
        # Placeholder Dune query ID — replace with your saved query
        return await self.run_dune_query(query_id=3_600_000)

    # ── internal ─────────────────────────────────────
    async def _glassnode_metric(
        self, endpoint: str, asset: str
    ) -> Optional[float]:
        if not self._glassnode_key:
            return None
        url = f"{self.GLASSNODE_BASE}/{endpoint}"
        params = {"a": asset, "api_key": self._glassnode_key, "f": "JSON", "s": "24h"}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data and isinstance(data, list):
                    return data[-1].get("v")
        except Exception as exc:
            log.debug("glassnode.error", endpoint=endpoint, error=str(exc))
        return None

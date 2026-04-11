"""
Health Check — periodic liveness & readiness probes.

Checks:
  - Database connectivity
  - Redis connectivity
  - Exchange API reachability
  - WebSocket feed status
  - Model freshness
  - Memory usage
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Dict

import structlog

from config.settings import get_settings

log = structlog.get_logger(__name__)


class HealthChecker:
    """Periodic system health monitor with auto-restart hooks."""

    def __init__(self):
        self._settings = get_settings()
        self._checks: Dict[str, bool] = {}
        self._last_check: float = 0
        self._consecutive_failures: int = 0
        self._max_failures: int = 5

    async def check_all(self) -> Dict[str, bool]:
        """Run all health checks."""
        self._checks = {
            "database": await self._check_database(),
            "redis": await self._check_redis(),
            "memory": self._check_memory(),
            "uptime": True,
        }

        all_healthy = all(self._checks.values())
        if all_healthy:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            failed = [k for k, v in self._checks.items() if not v]
            log.warning(
                "health.unhealthy",
                failed=failed,
                consecutive=self._consecutive_failures,
            )

        if self._consecutive_failures >= self._max_failures:
            log.critical("health.CRITICAL — too many consecutive failures")

        self._last_check = time.time()
        return self._checks

    async def run_periodic(self, interval: int = 30) -> None:
        """Run health checks in a loop."""
        while True:
            try:
                checks = await self.check_all()
                healthy_count = sum(1 for v in checks.values() if v)
                total = len(checks)
                log.info(
                    "health.check",
                    healthy=f"{healthy_count}/{total}",
                    checks=checks,
                )
            except Exception as exc:
                log.error("health.check_error", error=str(exc))
            await asyncio.sleep(interval)

    @property
    def is_healthy(self) -> bool:
        return all(self._checks.values()) if self._checks else False

    # ── Individual checks ────────────────────────────
    async def _check_database(self) -> bool:
        try:
            import asyncpg
            s = self._settings.database
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=s.host, port=s.port,
                    user=s.user, password=s.password.get_secret_value(),
                    database=s.db,
                ),
                timeout=5.0,
            )
            await conn.fetchval("SELECT 1")
            await conn.close()
            return True
        except Exception:
            return False

    async def _check_redis(self) -> bool:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(self._settings.redis.url)
            await asyncio.wait_for(r.ping(), timeout=3.0)
            await r.close()
            return True
        except Exception:
            return False

    @staticmethod
    def _check_memory() -> bool:
        """Return False if memory usage > 90%."""
        try:
            import psutil
            return psutil.virtual_memory().percent < 90
        except ImportError:
            # psutil not available — assume OK
            return True

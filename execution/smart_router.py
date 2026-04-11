"""
Smart Order Router — TWAP / VWAP execution algorithms.

Splits large orders into smaller slices to minimise market impact.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from strategies.signal_generator import TradeSignal

log = structlog.get_logger(__name__)


class SmartOrderRouter:
    """TWAP / VWAP smart order routing for large orders."""

    async def execute_twap(
        self,
        trader: Any,  # LiveTrader or PaperTrader
        signal: TradeSignal,
        total_quantity: float,
        n_slices: int = 5,
        interval_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Time-Weighted Average Price execution.
        Splits order into N equal slices at regular intervals.
        """
        slice_qty = total_quantity / n_slices
        fills: List[Dict] = []
        total_filled = 0.0
        total_cost = 0.0

        log.info(
            "twap.start",
            symbol=signal.symbol,
            total_qty=total_quantity,
            slices=n_slices,
            interval=interval_seconds,
        )

        for i in range(n_slices):
            try:
                fill = await trader.execute(signal, slice_qty)
                if fill:
                    filled_qty = fill.get("quantity", slice_qty)
                    filled_price = fill.get("price", signal.entry_price)
                    total_filled += filled_qty
                    total_cost += filled_qty * filled_price
                    fills.append(fill)
            except Exception as exc:
                log.warning(f"twap.slice_failed", slice=i, error=str(exc))

            if i < n_slices - 1:
                await asyncio.sleep(interval_seconds)

        avg_price = total_cost / (total_filled + 1e-15)

        result = {
            "algo": "TWAP",
            "symbol": signal.symbol,
            "total_filled": total_filled,
            "avg_price": avg_price,
            "n_fills": len(fills),
            "fills": fills,
        }
        log.info("twap.complete", **{k: v for k, v in result.items() if k != "fills"})
        return result

    async def execute_vwap(
        self,
        trader: Any,
        signal: TradeSignal,
        total_quantity: float,
        volume_profile: Optional[np.ndarray] = None,
        n_slices: int = 10,
        interval_seconds: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Volume-Weighted Average Price execution.
        Uses historical volume profile to size each slice.
        """
        if volume_profile is None or len(volume_profile) != n_slices:
            # Default: equal weight
            weights = np.ones(n_slices) / n_slices
        else:
            weights = volume_profile / (volume_profile.sum() + 1e-15)

        fills: List[Dict] = []
        total_filled = 0.0
        total_cost = 0.0

        log.info(
            "vwap.start",
            symbol=signal.symbol,
            total_qty=total_quantity,
            slices=n_slices,
        )

        for i in range(n_slices):
            slice_qty = total_quantity * weights[i]
            if slice_qty < 1e-8:
                continue

            try:
                fill = await trader.execute(signal, slice_qty)
                if fill:
                    filled_qty = fill.get("quantity", slice_qty)
                    filled_price = fill.get("price", signal.entry_price)
                    total_filled += filled_qty
                    total_cost += filled_qty * filled_price
                    fills.append(fill)
            except Exception as exc:
                log.warning("vwap.slice_failed", slice=i, error=str(exc))

            if i < n_slices - 1:
                await asyncio.sleep(interval_seconds)

        avg_price = total_cost / (total_filled + 1e-15)

        result = {
            "algo": "VWAP",
            "symbol": signal.symbol,
            "total_filled": total_filled,
            "avg_price": avg_price,
            "n_fills": len(fills),
        }
        log.info("vwap.complete", **result)
        return result

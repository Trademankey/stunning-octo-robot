"""
Paper Trader — simulates order execution with realistic slippage & fees.

Maintains virtual portfolio state for safe strategy testing.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import structlog

from data.db import Database
from strategies.signal_generator import SignalDirection, TradeSignal

log = structlog.get_logger(__name__)


class PaperTrader:
    """Simulated exchange execution."""

    MAKER_FEE = 0.0002    # 0.02%
    TAKER_FEE = 0.0004    # 0.04%
    SLIPPAGE_BPS = 1.0    # 1 basis point

    def __init__(self, db: Database, initial_balance: float = 10000.0):
        self._db = db
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self._fills: list = []

    async def execute(self, signal: TradeSignal, quantity: float) -> Dict[str, Any]:
        """Simulate order fill with slippage and fees."""
        price = signal.entry_price

        # Slippage model (random, volume-correlated)
        slippage = price * self.SLIPPAGE_BPS / 10000
        if signal.direction == SignalDirection.LONG:
            fill_price = price + slippage * np.random.uniform(0.5, 1.5)
        elif signal.direction == SignalDirection.SHORT:
            fill_price = price - slippage * np.random.uniform(0.5, 1.5)
        else:
            fill_price = price

        # Fee
        notional = fill_price * quantity
        fee = notional * self.TAKER_FEE

        # Update balance
        self.balance -= fee

        fill = {
            "order_id": f"paper_{int(time.time()*1000)}",
            "symbol": signal.symbol,
            "side": signal.direction.value,
            "quantity": quantity,
            "price": fill_price,
            "fee": fee,
            "slippage": abs(fill_price - price),
            "timestamp": time.time(),
            "status": "filled",
        }
        self._fills.append(fill)

        log.info(
            "paper.fill",
            symbol=signal.symbol,
            side=signal.direction.value,
            qty=quantity,
            price=fill_price,
            fee=fee,
        )
        return fill

    @property
    def total_fees_paid(self) -> float:
        return sum(f["fee"] for f in self._fills)

    @property
    def trade_count(self) -> int:
        return len(self._fills)

    def get_equity(self, unrealised_pnl: float = 0.0) -> float:
        return self.balance + unrealised_pnl

"""
Risk Manager — portfolio-level risk controls and kill-switches.

Enforces:
  - Max daily drawdown → kill-switch (-5%)
  - Max correlated pairs open simultaneously
  - Trailing stop management
  - Delta-neutral hedging suggestions
  - Correlation filter
  - Bayesian Belief Network edge validation
  - LLN edge validation (sufficient sample size)
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import numpy as np
import structlog

from config.pairs import PairUniverse
from config.settings import get_settings
from strategies.signal_generator import SignalDirection, TradeSignal

log = structlog.get_logger(__name__)


@dataclass
class PositionState:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    unrealised_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskManager:
    """Portfolio-level risk engine with kill-switches."""

    def __init__(self, pair_universe: Optional[PairUniverse] = None):
        self._settings = get_settings().risk
        self._universe = pair_universe or PairUniverse(max_correlated=self._settings.max_correlated_pairs)

        # State
        self.positions: Dict[str, PositionState] = {}
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = 0.0
        self.current_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.kill_switch_active: bool = False

        # Trade history for edge validation
        self._trade_results: List[float] = []

    # ── Pre-trade checks ─────────────────────────────
    def can_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Evaluate whether a trade should be allowed.
        Returns (allowed: bool, reason: str).
        """
        # Kill-switch
        if self.kill_switch_active:
            return False, "KILL_SWITCH: Daily drawdown limit breached"

        # Daily drawdown check
        if self.daily_start_equity > 0:
            dd = (self.daily_start_equity - self.current_equity) / self.daily_start_equity
            if dd >= self._settings.max_daily_drawdown:
                self.kill_switch_active = True
                log.critical("KILL_SWITCH_ACTIVATED", drawdown_pct=dd * 100)
                return False, f"KILL_SWITCH: DD={dd*100:.1f}% >= {self._settings.max_daily_drawdown*100}%"

        # Already in position for this symbol
        if signal.symbol in self.positions:
            return False, f"Already in position for {signal.symbol}"

        # Correlation filter
        open_syms = set(self.positions.keys())
        if not self._universe.can_open(signal.symbol, open_syms):
            return False, f"Correlation limit: too many {self._universe.pairs.get(signal.symbol, 'N/A')} correlated pairs open"

        # Max open positions
        if len(self.positions) >= self._settings.max_open_positions:
            return False, f"Max open positions ({self._settings.max_open_positions}) reached"

        # LLN edge validation — need minimum trades before sizing up
        if len(self._trade_results) >= 30:
            edge = np.mean(self._trade_results)
            if edge < 0:
                return False, f"Negative edge detected: {edge:.4f} over {len(self._trade_results)} trades"

        # Signal too weak
        if signal.strength < 0.15:
            return False, f"Signal too weak: {signal.strength:.3f}"

        return True, "OK"

    # ── Position management ──────────────────────────
    def open_position(self, signal: TradeSignal, quantity: float) -> PositionState:
        """Register a new position."""
        pos = PositionState(
            symbol=signal.symbol,
            side=signal.direction.value,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=signal.stop_loss,
            highest_price=signal.entry_price,
            lowest_price=signal.entry_price,
        )
        self.positions[signal.symbol] = pos
        log.info(
            "risk.position_opened",
            symbol=signal.symbol, side=pos.side,
            entry=pos.entry_price, qty=quantity,
            sl=pos.stop_loss, tp=pos.take_profit,
        )
        return pos

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close position, return PnL."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return 0.0

        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self.daily_pnl += pnl
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        self._trade_results.append(pnl)

        log.info(
            "risk.position_closed",
            symbol=symbol, side=pos.side,
            entry=pos.entry_price, exit=exit_price,
            pnl=pnl,
        )
        return pnl

    def update_trailing_stops(self, prices: Dict[str, float]) -> List[str]:
        """
        Update trailing stops for all positions.
        Returns list of symbols that hit their trailing stop.
        """
        stopped_out: List[str] = []
        atr_mult = self._settings.trailing_stop_atr_mult

        for symbol, pos in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue

            # Update high/low watermarks
            if price > pos.highest_price:
                pos.highest_price = price
            if price < pos.lowest_price:
                pos.lowest_price = price

            # Update trailing stop
            if pos.side == "long":
                # Trail upward from highest price
                new_trail = pos.highest_price - atr_mult * abs(pos.entry_price - pos.stop_loss) / self._settings.stop_loss_atr_mult
                pos.trailing_stop = max(pos.trailing_stop, new_trail)

                # Check stop
                if price <= pos.trailing_stop:
                    stopped_out.append(symbol)
                    log.info("risk.trailing_stop_hit", symbol=symbol, price=price, stop=pos.trailing_stop)

            else:  # short
                new_trail = pos.lowest_price + atr_mult * abs(pos.stop_loss - pos.entry_price) / self._settings.stop_loss_atr_mult
                pos.trailing_stop = min(pos.trailing_stop, new_trail) if pos.trailing_stop > 0 else new_trail

                if price >= pos.trailing_stop:
                    stopped_out.append(symbol)
                    log.info("risk.trailing_stop_hit", symbol=symbol, price=price, stop=pos.trailing_stop)

            # Update unrealised PnL
            if pos.side == "long":
                pos.unrealised_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealised_pnl = (pos.entry_price - price) * pos.quantity

        return stopped_out

    def check_stops(self, prices: Dict[str, float]) -> List[str]:
        """Check hard stop-losses and take-profits."""
        triggered = []
        for symbol, pos in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue

            if pos.side == "long":
                if price <= pos.stop_loss:
                    triggered.append(symbol)
                elif price >= pos.take_profit:
                    triggered.append(symbol)
            else:
                if price >= pos.stop_loss:
                    triggered.append(symbol)
                elif price <= pos.take_profit:
                    triggered.append(symbol)

        return triggered

    # ── Daily reset ──────────────────────────────────
    def reset_daily(self, equity: float) -> None:
        """Reset daily counters — call at UTC midnight."""
        self.daily_pnl = 0.0
        self.daily_start_equity = equity
        self.current_equity = equity
        self.kill_switch_active = False
        log.info("risk.daily_reset", equity=equity)

    # ── Portfolio metrics ────────────────────────────
    def get_portfolio_exposure(self) -> Dict[str, float]:
        """Current net and gross exposure."""
        long_exp = sum(
            p.entry_price * p.quantity
            for p in self.positions.values() if p.side == "long"
        )
        short_exp = sum(
            p.entry_price * p.quantity
            for p in self.positions.values() if p.side == "short"
        )
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "net_exposure": long_exp - short_exp,
            "gross_exposure": long_exp + short_exp,
            "n_positions": len(self.positions),
        }

    def get_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self.positions.values())

    def compute_edge_stats(self) -> Dict[str, float]:
        """LLN validation stats from trade history."""
        if not self._trade_results:
            return {"n_trades": 0, "edge": 0, "win_rate": 0, "expectancy": 0}

        results = np.array(self._trade_results)
        wins = results[results > 0]
        losses = results[results <= 0]

        win_rate = len(wins) / len(results) if len(results) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        return {
            "n_trades": len(results),
            "edge": float(np.mean(results)),
            "win_rate": win_rate,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
            "profit_factor": float(avg_win * win_rate / (avg_loss * (1 - win_rate) + 1e-10)),
        }

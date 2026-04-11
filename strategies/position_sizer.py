"""
Position Sizing Engine.

Methods:
  - Fractional Kelly Criterion
  - Volatility Targeting (target vol = 15% annualised)
  - ATR-based sizing
  - Combined: min(Kelly, VolTarget, MaxRisk)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

from config.settings import get_settings

log = structlog.get_logger(__name__)


class PositionSizer:
    """Compute optimal position size given signal, volatility, and account state."""

    def __init__(self):
        s = get_settings().risk
        self.max_risk = s.max_risk_per_trade
        self.kelly_fraction = s.kelly_fraction
        self.max_leverage = s.max_leverage
        self.target_annual_vol = 0.15  # 15% target portfolio volatility

    def compute(
        self,
        account_equity: float,
        signal_strength: float,  # [0, 1]
        entry_price: float,
        stop_loss: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
        atr: float = 0.0,
        current_vol: float = 0.0,  # annualised realised vol
    ) -> float:
        """
        Returns position size in base currency (e.g., BTC quantity).
        """
        if entry_price <= 0 or account_equity <= 0 or signal_strength <= 0:
            return 0.0

        # ── 1. Risk-based sizing (% risk per trade) ──
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance < 1e-10:
            risk_distance = atr if atr > 0 else entry_price * 0.01

        risk_amount = account_equity * self.max_risk
        risk_size = risk_amount / risk_distance

        # ── 2. Kelly sizing ──────────────────────────
        kelly_full = self._kelly_criterion(win_rate, avg_win_loss_ratio)
        kelly_size = (
            account_equity * kelly_full * self.kelly_fraction * signal_strength
        ) / entry_price

        # ── 3. Volatility-target sizing ──────────────
        if current_vol > 0:
            vol_scalar = self.target_annual_vol / (current_vol + 1e-10)
            vol_size = (account_equity * vol_scalar * signal_strength) / entry_price
        else:
            vol_size = risk_size  # fallback

        # ── 4. Take minimum of all methods ───────────
        size = min(risk_size, kelly_size, vol_size)

        # ── 5. Apply leverage cap ────────────────────
        max_notional = account_equity * self.max_leverage
        max_size = max_notional / entry_price
        size = min(size, max_size)

        # ── 6. Scale by signal strength ──────────────
        size *= signal_strength

        # Floor at zero
        size = max(size, 0.0)

        log.debug(
            "position_sizer.computed",
            risk_size=risk_size, kelly_size=kelly_size,
            vol_size=vol_size, final_size=size,
            signal=signal_strength,
        )

        return size

    @staticmethod
    def _kelly_criterion(win_rate: float, avg_win_loss_ratio: float) -> float:
        """
        Full Kelly: f* = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        b = avg_win_loss_ratio
        p = win_rate
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b

        # Clamp: never bet more than 25% of bankroll (full Kelly is aggressive)
        return max(0.0, min(kelly, 0.25))

    @staticmethod
    def compute_var_position(
        account_equity: float,
        entry_price: float,
        daily_vol: float,
        confidence: float = 0.99,
        max_var_pct: float = 0.02,
    ) -> float:
        """
        Value-at-Risk position sizing.
        Max position s.t. 99% daily VaR < max_var_pct of equity.
        """
        from scipy.stats import norm
        z = norm.ppf(confidence)
        max_loss = account_equity * max_var_pct
        if daily_vol <= 0 or entry_price <= 0:
            return 0.0
        size = max_loss / (z * daily_vol * entry_price)
        return max(size, 0.0)

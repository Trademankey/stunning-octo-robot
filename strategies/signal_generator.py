"""
Signal Generator — fuses ensemble model output with regime / liquidity /
sentiment / on-chain filters into a final trade decision.

Decision flow:
  1. Ensemble model → raw signal [-1, +1]
  2. Regime filter → allow / block / attenuate
  3. Sentiment filter → boost / dampen
  4. Liquidity filter → block illiquid
  5. On-chain filter → whale divergence check
  6. Uncertainty gate → block low-confidence
  7. Turtle Soup reversal overlay
  8. Output: TradeSignal dataclass
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import structlog

from config.settings import get_settings
from features.regime import MarketRegime

log = structlog.get_logger(__name__)


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TradeSignal:
    symbol: str
    direction: SignalDirection
    strength: float          # [0, 1] — confidence
    raw_signal: float        # [-1, +1] from ensemble
    regime: str
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    uncertainty: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)


class SignalGenerator:
    """Produces filtered, risk-adjusted trade signals."""

    # Thresholds
    SIGNAL_THRESHOLD = 0.15      # Minimum |signal| to act
    UNCERTAINTY_MAX = 0.6        # Block if uncertainty > this
    SENTIMENT_BOOST = 0.1        # Max sentiment boost
    LIQUIDITY_MIN_DEPTH = 50000  # Min $50k depth within 1% of mid

    def __init__(self):
        self._settings = get_settings()

    def generate(
        self,
        symbol: str,
        raw_signal: float,
        uncertainty: float,
        current_price: float,
        atr: float,
        regime: int,
        orderbook_features: Optional[Dict[str, float]] = None,
        sentiment: Optional[Dict[str, float]] = None,
        onchain: Optional[Dict[str, Optional[float]]] = None,
    ) -> TradeSignal:
        """
        Process raw model signal through all filters → TradeSignal.
        """
        signal = float(raw_signal)
        regime_name = MarketRegime(regime).name if regime in [e.value for e in MarketRegime] else "UNKNOWN"

        meta: Dict = {"filters_applied": []}

        # ── 1. Uncertainty gate ──────────────────────
        if uncertainty > self.UNCERTAINTY_MAX:
            signal *= 0.3
            meta["filters_applied"].append(f"uncertainty_dampened ({uncertainty:.3f})")

        # ── 2. Regime filter ─────────────────────────
        signal = self._apply_regime_filter(signal, regime, meta)

        # ── 3. Sentiment overlay ─────────────────────
        if sentiment:
            signal = self._apply_sentiment_filter(signal, sentiment, meta)

        # ── 4. Liquidity check ───────────────────────
        if orderbook_features:
            signal = self._apply_liquidity_filter(signal, orderbook_features, meta)

        # ── 5. On-chain divergence ───────────────────
        if onchain:
            signal = self._apply_onchain_filter(signal, onchain, meta)

        # ── 6. Turtle Soup reversal check ────────────
        # (detect false breakouts at key levels)
        if orderbook_features and atr > 0:
            signal = self._turtle_soup_overlay(
                signal, current_price, atr, orderbook_features, meta
            )

        # ── Final decision ───────────────────────────
        abs_signal = abs(signal)
        if abs_signal < self.SIGNAL_THRESHOLD:
            direction = SignalDirection.FLAT
            strength = 0.0
        elif signal > 0:
            direction = SignalDirection.LONG
            strength = min(abs_signal, 1.0)
        else:
            direction = SignalDirection.SHORT
            strength = min(abs_signal, 1.0)

        # ── SL / TP ─────────────────────────────────
        sl_mult = self._settings.risk.stop_loss_atr_mult
        tp_mult = self._settings.risk.take_profit_atr_mult

        if direction == SignalDirection.LONG:
            stop_loss = current_price - sl_mult * atr
            take_profit = current_price + tp_mult * atr
        elif direction == SignalDirection.SHORT:
            stop_loss = current_price + sl_mult * atr
            take_profit = current_price - tp_mult * atr
        else:
            stop_loss = 0.0
            take_profit = 0.0

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            raw_signal=raw_signal,
            regime=regime_name,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            uncertainty=uncertainty,
            metadata=meta,
        )

    # ── Filters ──────────────────────────────────────

    def _apply_regime_filter(
        self, signal: float, regime: int, meta: Dict
    ) -> float:
        """Attenuate or block signal based on market regime."""
        if regime == MarketRegime.HIGH_VOLATILITY:
            signal *= 0.5  # Reduce size in high vol
            meta["filters_applied"].append("high_vol_dampened")
        elif regime == MarketRegime.MEAN_REVERTING:
            # Flip momentum signals in mean-reverting regime
            if abs(signal) > 0.3:
                signal *= 0.7
                meta["filters_applied"].append("mean_revert_dampened")
        elif regime == MarketRegime.TRENDING_UP:
            if signal < 0:
                signal *= 0.5  # Reduce shorts in uptrend
                meta["filters_applied"].append("uptrend_short_dampened")
        elif regime == MarketRegime.TRENDING_DOWN:
            if signal > 0:
                signal *= 0.5  # Reduce longs in downtrend
                meta["filters_applied"].append("downtrend_long_dampened")
        return signal

    def _apply_sentiment_filter(
        self, signal: float, sentiment: Dict[str, float], meta: Dict
    ) -> float:
        """Boost or dampen signal based on sentiment alignment."""
        sent_score = sentiment.get("weighted_avg", 0.0)

        if np.sign(signal) == np.sign(sent_score) and abs(sent_score) > 0.3:
            boost = min(abs(sent_score) * self.SENTIMENT_BOOST, self.SENTIMENT_BOOST)
            signal += np.sign(signal) * boost
            meta["filters_applied"].append(f"sentiment_boost ({sent_score:.3f})")
        elif np.sign(signal) != np.sign(sent_score) and abs(sent_score) > 0.5:
            signal *= 0.7
            meta["filters_applied"].append(f"sentiment_divergence ({sent_score:.3f})")

        return signal

    def _apply_liquidity_filter(
        self, signal: float, ob: Dict[str, float], meta: Dict
    ) -> float:
        """Block signal if insufficient liquidity."""
        bid_depth = ob.get("bid_depth_1pct", 0)
        ask_depth = ob.get("ask_depth_1pct", 0)
        min_depth = min(bid_depth, ask_depth)

        if min_depth < self.LIQUIDITY_MIN_DEPTH:
            signal *= 0.3
            meta["filters_applied"].append(f"low_liquidity (${min_depth:,.0f})")

        # Spread check
        spread_bps = ob.get("spread_bps", 0)
        if spread_bps > 10:
            signal *= 0.5
            meta["filters_applied"].append(f"wide_spread ({spread_bps:.1f}bps)")

        return signal

    def _apply_onchain_filter(
        self, signal: float, onchain: Dict[str, Optional[float]], meta: Dict
    ) -> float:
        """Check on-chain divergence (whale flows vs signal)."""
        funding = onchain.get("funding_rate")
        whale_txns = onchain.get("whale_txns")

        # High positive funding + long signal → crowded trade
        if funding is not None and funding > 0.001 and signal > 0.3:
            signal *= 0.7
            meta["filters_applied"].append(f"crowded_long (funding={funding:.5f})")

        # High negative funding + short signal → crowded
        if funding is not None and funding < -0.001 and signal < -0.3:
            signal *= 0.7
            meta["filters_applied"].append(f"crowded_short (funding={funding:.5f})")

        return signal

    def _turtle_soup_overlay(
        self,
        signal: float,
        price: float,
        atr: float,
        ob: Dict[str, float],
        meta: Dict,
    ) -> float:
        """
        Turtle Soup: detect false breakout at key orderbook levels.
        If price just broke past a heavy wall but imbalance is reversing,
        fade the breakout.
        """
        imbalance = ob.get("imbalance_5", 0)

        # If we have a strong long signal but imbalance is heavily negative
        # (more asks being stacked), it's a potential bull trap
        if signal > 0.4 and imbalance < -0.5:
            signal *= 0.5
            meta["filters_applied"].append("turtle_soup_bull_trap")

        if signal < -0.4 and imbalance > 0.5:
            signal *= 0.5
            meta["filters_applied"].append("turtle_soup_bear_trap")

        return signal

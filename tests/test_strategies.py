"""
Tests for strategy components: signal generation, position sizing, risk management.
"""
import numpy as np
import pytest

from strategies.position_sizer import PositionSizer
from strategies.risk_manager import RiskManager
from strategies.signal_generator import SignalDirection, SignalGenerator, TradeSignal


class TestSignalGenerator:
    def setup_method(self):
        self.gen = SignalGenerator()

    def test_strong_long_signal(self):
        signal = self.gen.generate(
            symbol="BTC/USDT", raw_signal=0.6, uncertainty=0.1,
            current_price=50000, atr=500, regime=0,
        )
        assert signal.direction == SignalDirection.LONG
        assert signal.strength > 0.3
        assert signal.stop_loss < 50000
        assert signal.take_profit > 50000

    def test_weak_signal_flat(self):
        signal = self.gen.generate(
            symbol="BTC/USDT", raw_signal=0.05, uncertainty=0.1,
            current_price=50000, atr=500, regime=0,
        )
        assert signal.direction == SignalDirection.FLAT

    def test_high_uncertainty_dampens(self):
        strong = self.gen.generate(
            symbol="BTC/USDT", raw_signal=0.5, uncertainty=0.1,
            current_price=50000, atr=500, regime=0,
        )
        weak = self.gen.generate(
            symbol="BTC/USDT", raw_signal=0.5, uncertainty=0.8,
            current_price=50000, atr=500, regime=0,
        )
        assert weak.strength <= strong.strength

    def test_short_signal(self):
        signal = self.gen.generate(
            symbol="ETH/USDT", raw_signal=-0.7, uncertainty=0.1,
            current_price=3000, atr=50, regime=1,
        )
        assert signal.direction == SignalDirection.SHORT
        assert signal.stop_loss > 3000
        assert signal.take_profit < 3000


class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer()

    def test_basic_sizing(self):
        size = self.sizer.compute(
            account_equity=10000, signal_strength=0.8,
            entry_price=50000, stop_loss=49000,
            win_rate=0.55, avg_win_loss_ratio=1.5,
        )
        assert size > 0
        # Should not exceed max leverage
        notional = size * 50000
        assert notional <= 10000 * self.sizer.max_leverage

    def test_zero_signal_zero_size(self):
        size = self.sizer.compute(
            account_equity=10000, signal_strength=0.0,
            entry_price=50000, stop_loss=49000,
        )
        assert size == 0.0

    def test_kelly_negative_edge(self):
        """Negative edge should produce zero Kelly sizing."""
        k = PositionSizer._kelly_criterion(win_rate=0.3, avg_win_loss_ratio=0.5)
        assert k == 0.0


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager()
        self.rm.daily_start_equity = 10000
        self.rm.current_equity = 10000
        self.rm.peak_equity = 10000

    def test_can_trade_normal(self):
        signal = TradeSignal(
            symbol="BTC/USDT", direction=SignalDirection.LONG,
            strength=0.5, raw_signal=0.5, regime="TRENDING_UP",
            entry_price=50000, stop_loss=49000, take_profit=52000,
        )
        allowed, reason = self.rm.can_trade(signal)
        assert allowed, reason

    def test_kill_switch_blocks(self):
        self.rm.kill_switch_active = True
        signal = TradeSignal(
            symbol="BTC/USDT", direction=SignalDirection.LONG,
            strength=0.5, raw_signal=0.5, regime="test",
            entry_price=50000,
        )
        allowed, reason = self.rm.can_trade(signal)
        assert not allowed
        assert "KILL_SWITCH" in reason

    def test_daily_drawdown_triggers_kill(self):
        self.rm.current_equity = 9400  # > 5% drawdown from 10000
        signal = TradeSignal(
            symbol="BTC/USDT", direction=SignalDirection.LONG,
            strength=0.5, raw_signal=0.5, regime="test",
            entry_price=50000,
        )
        allowed, reason = self.rm.can_trade(signal)
        assert not allowed
        assert self.rm.kill_switch_active

    def test_duplicate_symbol_blocked(self):
        signal = TradeSignal(
            symbol="BTC/USDT", direction=SignalDirection.LONG,
            strength=0.5, raw_signal=0.5, regime="test",
            entry_price=50000, stop_loss=49000, take_profit=52000,
        )
        self.rm.open_position(signal, 0.1)

        allowed, reason = self.rm.can_trade(signal)
        assert not allowed
        assert "Already in position" in reason

    def test_position_pnl(self):
        signal = TradeSignal(
            symbol="ETH/USDT", direction=SignalDirection.LONG,
            strength=0.5, raw_signal=0.5, regime="test",
            entry_price=3000, stop_loss=2900, take_profit=3300,
        )
        self.rm.open_position(signal, 1.0)
        pnl = self.rm.close_position("ETH/USDT", 3100)
        assert pnl == pytest.approx(100.0, abs=0.01)

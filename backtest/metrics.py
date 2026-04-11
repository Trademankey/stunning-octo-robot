"""
Backtest metrics — comprehensive performance analytics.

Computes: Sharpe, Sortino, Calmar, Omega, Profit Factor, Max DD,
Win Rate, Expectancy, SQN, and more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class BacktestMetrics:
    """Container for all backtest performance metrics."""

    # Returns series
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # bars

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Trade-based
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0  # bars
    sqn: float = 0.0  # System Quality Number

    # Portfolio
    final_equity: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    leverage_utilisation: float = 0.0

    # Additional
    monthly_returns: List[float] = field(default_factory=list)
    worst_month: float = 0.0
    best_month: float = 0.0
    pct_positive_months: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "sqn": self.sqn,
            "final_equity": self.final_equity,
        }


def compute_metrics(
    equity_curve: np.ndarray,
    trade_pnls: np.ndarray,
    initial_equity: float = 10000.0,
    bars_per_year: int = 365 * 24 * 4,  # 15m bars
    risk_free_rate: float = 0.04,
    total_fees: float = 0.0,
) -> BacktestMetrics:
    """Compute full metrics suite from equity curve and trade PnLs."""
    m = BacktestMetrics()

    if len(equity_curve) < 2:
        return m

    # ── Returns series ───────────────────────────────
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-15)
    m.total_return = float((equity_curve[-1] / equity_curve[0]) - 1)
    m.final_equity = float(equity_curve[-1])
    m.total_fees = total_fees

    # Annualised return
    n_bars = len(equity_curve)
    years = n_bars / bars_per_year
    if years > 0:
        m.annual_return = float((1 + m.total_return) ** (1 / years) - 1)

    # ── Drawdown ─────────────────────────────────────
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / (peak + 1e-15)
    m.max_drawdown = float(np.max(drawdowns))

    # Drawdown duration
    in_dd = drawdowns > 0
    if np.any(in_dd):
        dd_runs = np.diff(np.where(np.concatenate(([in_dd[0]], in_dd[:-1] != in_dd[1:], [True])))[0])
        m.max_drawdown_duration = int(np.max(dd_runs)) if len(dd_runs) > 0 else 0

    # ── Risk-adjusted ratios ─────────────────────────
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) + 1e-15
    rf_per_bar = risk_free_rate / bars_per_year

    # Sharpe
    m.sharpe_ratio = float(
        (mean_ret - rf_per_bar) / std_ret * np.sqrt(bars_per_year)
    )

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-15
    m.sortino_ratio = float(
        (mean_ret - rf_per_bar) / downside_std * np.sqrt(bars_per_year)
    )

    # Calmar
    if m.max_drawdown > 0:
        m.calmar_ratio = float(m.annual_return / m.max_drawdown)

    # Omega (threshold = 0)
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0]).sum()
    m.omega_ratio = float(gains / (losses + 1e-15))

    # ── Trade-based metrics ──────────────────────────
    if len(trade_pnls) > 0:
        m.total_trades = len(trade_pnls)
        wins = trade_pnls[trade_pnls > 0]
        losses_arr = trade_pnls[trade_pnls <= 0]

        m.win_rate = float(len(wins) / len(trade_pnls))
        m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        m.avg_loss = float(np.mean(np.abs(losses_arr))) if len(losses_arr) > 0 else 0.0
        m.largest_win = float(np.max(wins)) if len(wins) > 0 else 0.0
        m.largest_loss = float(np.min(losses_arr)) if len(losses_arr) > 0 else 0.0

        # Profit Factor
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(losses_arr).sum()) if len(losses_arr) > 0 else 1e-15
        m.profit_factor = gross_profit / gross_loss

        # Expectancy
        m.expectancy = float(np.mean(trade_pnls))

        # SQN (System Quality Number)
        if len(trade_pnls) > 1:
            m.sqn = float(
                np.mean(trade_pnls) / (np.std(trade_pnls, ddof=1) + 1e-15) * np.sqrt(len(trade_pnls))
            )

    return m


def format_metrics(m: BacktestMetrics) -> str:
    """Pretty-print metrics table."""
    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║          BACKTEST PERFORMANCE REPORT              ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Total Return:       {m.total_return:>10.2%}                  ║",
        f"║  Annual Return:      {m.annual_return:>10.2%}                  ║",
        f"║  Max Drawdown:       {m.max_drawdown:>10.2%}                  ║",
        f"║  Final Equity:       ${m.final_equity:>12,.2f}              ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Sharpe Ratio:       {m.sharpe_ratio:>10.3f}                  ║",
        f"║  Sortino Ratio:      {m.sortino_ratio:>10.3f}                  ║",
        f"║  Calmar Ratio:       {m.calmar_ratio:>10.3f}                  ║",
        f"║  Omega Ratio:        {m.omega_ratio:>10.3f}                  ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Total Trades:       {m.total_trades:>10d}                  ║",
        f"║  Win Rate:           {m.win_rate:>10.2%}                  ║",
        f"║  Profit Factor:      {m.profit_factor:>10.3f}                  ║",
        f"║  Expectancy:         ${m.expectancy:>10.2f}                  ║",
        f"║  SQN:                {m.sqn:>10.3f}                  ║",
        f"║  Avg Win:            ${m.avg_win:>10.2f}                  ║",
        f"║  Avg Loss:           ${m.avg_loss:>10.2f}                  ║",
        f"║  Largest Win:        ${m.largest_win:>10.2f}                  ║",
        f"║  Largest Loss:       ${m.largest_loss:>10.2f}                  ║",
        f"║  Total Fees:         ${m.total_fees:>10.2f}                  ║",
        "╚══════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)

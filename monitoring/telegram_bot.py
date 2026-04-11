"""
Telegram Bot — real-time notifications and emergency controls.

Commands:
  /status     — current positions, equity, P&L
  /signals    — recent signals
  /stop       — emergency kill-switch
  /resume     — resume trading
  /metrics    — performance metrics
"""
from __future__ import annotations

import asyncio
from typing import Optional

import structlog

from config.settings import get_settings

log = structlog.get_logger(__name__)

try:
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


class TelegramNotifier:
    """Async Telegram bot for alerts and commands."""

    def __init__(self):
        self._settings = get_settings().telegram
        self._bot: Optional[Bot] = None
        self._app: Optional[Application] = None
        self._risk_manager = None  # Injected later

    def set_risk_manager(self, rm) -> None:
        self._risk_manager = rm

    async def start(self) -> None:
        if not HAS_TELEGRAM or not self._settings.enabled:
            log.info("telegram.disabled")
            return

        token = self._settings.bot_token.get_secret_value()
        if not token:
            log.warning("telegram.no_token")
            return

        self._bot = Bot(token=token)
        self._app = Application.builder().token(token).build()

        # Register command handlers
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("metrics", self._cmd_metrics))
        self._app.add_handler(CommandHandler("help", self._cmd_help))

        log.info("telegram.started")

    async def stop(self) -> None:
        if self._app:
            await self._app.shutdown()

    # ── Notifications ────────────────────────────────
    async def send_signal(self, message: str) -> None:
        await self._send(f"🔔 *Signal*\n{message}")

    async def send_trade(self, message: str) -> None:
        await self._send(f"💰 *Trade*\n{message}")

    async def send_alert(self, message: str) -> None:
        await self._send(f"⚠️ *Alert*\n{message}")

    async def send_error(self, message: str) -> None:
        await self._send(f"🚨 *Error*\n{message}")

    async def send_daily_report(self, report: str) -> None:
        await self._send(f"📊 *Daily Report*\n{report}")

    async def send_kill_switch(self, reason: str) -> None:
        await self._send(f"🛑 *KILL SWITCH ACTIVATED*\n{reason}")

    # ── Command Handlers ─────────────────────────────
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self._risk_manager is None:
            await update.message.reply_text("⏳ Bot not fully initialized")
            return

        exposure = self._risk_manager.get_portfolio_exposure()
        positions = self._risk_manager.positions
        upnl = self._risk_manager.get_unrealised_pnl()

        lines = [
            "📈 *Bot Status*",
            f"Equity: ${self._risk_manager.current_equity:,.2f}",
            f"Daily P&L: ${self._risk_manager.daily_pnl:,.2f}",
            f"Unrealised: ${upnl:,.2f}",
            f"Positions: {len(positions)}",
            f"Net exposure: ${exposure['net_exposure']:,.2f}",
            f"Kill switch: {'🔴 ACTIVE' if self._risk_manager.kill_switch_active else '🟢 OFF'}",
        ]

        for sym, pos in positions.items():
            lines.append(f"  {sym}: {pos.side} @ {pos.entry_price:.2f} (P&L: ${pos.unrealised_pnl:.2f})")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self._risk_manager:
            self._risk_manager.kill_switch_active = True
            await update.message.reply_text("🛑 *Kill switch ACTIVATED*. All trading halted.", parse_mode="Markdown")
        else:
            await update.message.reply_text("⏳ Bot not initialized")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self._risk_manager:
            self._risk_manager.kill_switch_active = False
            await update.message.reply_text("🟢 Trading *resumed*.", parse_mode="Markdown")
        else:
            await update.message.reply_text("⏳ Bot not initialized")

    async def _cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self._risk_manager:
            stats = self._risk_manager.compute_edge_stats()
            lines = [
                "📊 *Edge Statistics*",
                f"Trades: {stats['n_trades']}",
                f"Win Rate: {stats['win_rate']:.1%}",
                f"Expectancy: ${stats['expectancy']:.2f}",
                f"Profit Factor: {stats['profit_factor']:.2f}",
                f"Avg Win: ${stats['avg_win']:.2f}",
                f"Avg Loss: ${stats['avg_loss']:.2f}",
            ]
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        else:
            await update.message.reply_text("⏳ No data yet")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = (
            "🤖 *Trading Bot Commands*\n"
            "/status — Positions & equity\n"
            "/metrics — Performance stats\n"
            "/stop — Emergency halt\n"
            "/resume — Resume trading\n"
            "/help — This message"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    # ── Internal ─────────────────────────────────────
    async def _send(self, text: str) -> None:
        if not self._bot or not self._settings.chat_id:
            return
        try:
            await self._bot.send_message(
                chat_id=self._settings.chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as exc:
            log.warning("telegram.send_failed", error=str(exc))

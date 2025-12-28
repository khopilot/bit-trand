"""
Telegram Control for BTC Elite Trader

Bidirectional Telegram bot for monitoring and control.
Supports commands for status, balance, pause/resume, and kill switch.

Author: khopilot
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("btc_trader.telegram")

# Try to import telegram library
try:
    from telegram import Update, BotCommand
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    logger.warning("python-telegram-bot not installed, Telegram features disabled")

    # Create stub types when telegram not installed
    class Update:
        pass

    class ContextTypes:
        DEFAULT_TYPE = None

    class BotCommand:
        def __init__(self, *args, **kwargs):
            pass

    Application = None
    CommandHandler = None


class TelegramControl:
    """
    Telegram bot for monitoring and control.

    Commands:
    - /status - System status, position, P&L
    - /balance - Account balance
    - /pause - Pause trading
    - /resume - Resume trading
    - /kill confirm - Emergency kill switch
    - /arb - Show funding arb paper trading status
    - /arb_start - Start paper trading simulation
    - /arb_stop - Stop paper trading simulation
    - /help - Show available commands
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        authorized_users: Optional[list[int]] = None,
    ):
        """
        Initialize TelegramControl.

        Args:
            bot_token: Telegram bot API token
            chat_id: Default chat ID for notifications
            authorized_users: List of authorized user IDs
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.authorized_users = authorized_users or []

        self._app: Optional[Application] = None
        self._running = False

        # Callbacks for bot commands
        self._status_callback: Optional[Callable[[], dict]] = None
        self._balance_callback: Optional[Callable[[], dict]] = None
        self._pause_callback: Optional[Callable[[], None]] = None
        self._resume_callback: Optional[Callable[[], None]] = None
        self._kill_callback: Optional[Callable[[], None]] = None

        # Callbacks for arb paper trading commands
        self._arb_status_callback: Optional[Callable[[], dict]] = None
        self._arb_start_callback: Optional[Callable[[], None]] = None
        self._arb_stop_callback: Optional[Callable[[], None]] = None

        # Callbacks for additional arb commands
        self._arb_position_callback: Optional[Callable[[], dict]] = None
        self._arb_history_callback: Optional[Callable[[int], dict]] = None
        self._arb_stats_callback: Optional[Callable[[], dict]] = None
        self._arb_rate_callback: Optional[Callable[[], dict]] = None
        self._arb_config_callback: Optional[Callable[[], dict]] = None

        # Callbacks for directional bot commands
        self._dir_status_callback: Optional[Callable[[], dict]] = None
        self._dir_signals_callback: Optional[Callable[[], dict]] = None
        self._dir_trades_callback: Optional[Callable[[int], dict]] = None
        self._dir_stop_callback: Optional[Callable[[], None]] = None

        # Callbacks for beast bot commands
        self._beast_status_callback: Optional[Callable[[], dict]] = None
        self._beast_mode_callback: Optional[Callable[[], dict]] = None
        self._beast_stop_callback: Optional[Callable[[], None]] = None

        if not HAS_TELEGRAM:
            logger.warning("Telegram library not available")
            return

        if not bot_token:
            logger.info("No bot token provided, Telegram notifications disabled")
            return

        self._app = Application.builder().token(bot_token).build()
        self._register_handlers()

        logger.info("TelegramControl initialized")

    def _register_handlers(self) -> None:
        """Register command handlers."""
        if not self._app:
            return

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("balance", self._cmd_balance))
        self._app.add_handler(CommandHandler("pause", self._cmd_pause))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("kill", self._cmd_kill))

        # Arb paper trading commands
        self._app.add_handler(CommandHandler("arb", self._cmd_arb_status))
        self._app.add_handler(CommandHandler("arb_start", self._cmd_arb_start))
        self._app.add_handler(CommandHandler("arb_stop", self._cmd_arb_stop))

        # Additional arb commands
        self._app.add_handler(CommandHandler("pos", self._cmd_position))
        self._app.add_handler(CommandHandler("history", self._cmd_history))
        self._app.add_handler(CommandHandler("stats", self._cmd_stats))
        self._app.add_handler(CommandHandler("rate", self._cmd_rate))
        self._app.add_handler(CommandHandler("config", self._cmd_config))

        # Directional bot commands
        self._app.add_handler(CommandHandler("dir", self._cmd_dir_status))
        self._app.add_handler(CommandHandler("signals", self._cmd_signals))
        self._app.add_handler(CommandHandler("trades", self._cmd_trades))
        self._app.add_handler(CommandHandler("compare", self._cmd_compare))

        # Beast bot commands
        self._app.add_handler(CommandHandler("beast", self._cmd_beast_status))
        self._app.add_handler(CommandHandler("mode", self._cmd_mode))
        self._app.add_handler(CommandHandler("trinity", self._cmd_trinity))

    def set_callbacks(
        self,
        status: Optional[Callable[[], dict]] = None,
        balance: Optional[Callable[[], dict]] = None,
        pause: Optional[Callable[[], None]] = None,
        resume: Optional[Callable[[], None]] = None,
        kill: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callbacks for bot commands."""
        self._status_callback = status
        self._balance_callback = balance
        self._pause_callback = pause
        self._resume_callback = resume
        self._kill_callback = kill

    def set_arb_callbacks(
        self,
        status: Optional[Callable[[], dict]] = None,
        start: Optional[Callable[[], None]] = None,
        stop: Optional[Callable[[], None]] = None,
        position: Optional[Callable[[], dict]] = None,
        history: Optional[Callable[[int], dict]] = None,
        stats: Optional[Callable[[], dict]] = None,
        rate: Optional[Callable[[], dict]] = None,
        config: Optional[Callable[[], dict]] = None,
    ) -> None:
        """Set callbacks for arb paper trading commands."""
        self._arb_status_callback = status
        self._arb_start_callback = start
        self._arb_stop_callback = stop
        self._arb_position_callback = position
        self._arb_history_callback = history
        self._arb_stats_callback = stats
        self._arb_rate_callback = rate
        self._arb_config_callback = config

    def set_dir_callbacks(
        self,
        status: Optional[Callable[[], dict]] = None,
        signals: Optional[Callable[[], dict]] = None,
        trades: Optional[Callable[[int], dict]] = None,
        stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callbacks for directional bot commands."""
        self._dir_status_callback = status
        self._dir_signals_callback = signals
        self._dir_trades_callback = trades
        self._dir_stop_callback = stop

    def set_beast_callbacks(
        self,
        status: Optional[Callable[[], dict]] = None,
        mode: Optional[Callable[[], dict]] = None,
        stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callbacks for beast bot commands."""
        self._beast_status_callback = status
        self._beast_mode_callback = mode
        self._beast_stop_callback = stop

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        if not self.authorized_users:
            return True  # No restrictions if no users specified
        return user_id in self.authorized_users

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized")
            return

        await update.message.reply_text(
            "BTC Elite Trader Bot\n\n"
            "Use /help to see available commands."
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not self._is_authorized(update.effective_user.id):
            return

        help_text = """
*BTC Elite Trader Commands*

*Funding Arbitrage:*
/arb - Show paper trading status
/pos - Detailed position info
/history - Funding payment history
/stats - Performance statistics
/rate - Next funding rate info
/config - Show configuration

*Control:*
/arb\\_start - Start paper trading
/arb\\_stop - Stop paper trading

*General:*
/status - Show system status
/help - Show this help message

_Trading bot is running in {mode} mode._
        """.format(mode="LIVE" if self._running else "STANDBY")

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._is_authorized(update.effective_user.id):
            return

        # Use arb formatter if arb callbacks are set (paper trading mode)
        if self._arb_status_callback:
            try:
                status = self._arb_status_callback()
                msg = self._format_arb_status(status)
            except Exception as e:
                msg = f"Error getting status: {e}"
        elif self._status_callback:
            try:
                status = self._status_callback()
                msg = self._format_status(status)
            except Exception as e:
                msg = f"Error getting status: {e}"
        else:
            msg = "Status not available"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /balance command."""
        if not self._is_authorized(update.effective_user.id):
            return

        if self._balance_callback:
            try:
                balance = self._balance_callback()
                msg = self._format_balance(balance)
            except Exception as e:
                msg = f"Error getting balance: {e}"
        else:
            msg = "Balance not available"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pause command."""
        if not self._is_authorized(update.effective_user.id):
            return

        if self._pause_callback:
            try:
                self._pause_callback()
                await update.message.reply_text("Trading PAUSED")
            except Exception as e:
                await update.message.reply_text(f"Error pausing: {e}")
        else:
            await update.message.reply_text("Pause not available")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /resume command."""
        if not self._is_authorized(update.effective_user.id):
            return

        if self._resume_callback:
            try:
                self._resume_callback()
                await update.message.reply_text("Trading RESUMED")
            except Exception as e:
                await update.message.reply_text(f"Error resuming: {e}")
        else:
            await update.message.reply_text("Resume not available")

    async def _cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /kill command - requires confirmation."""
        if not self._is_authorized(update.effective_user.id):
            return

        args = context.args
        if not args or args[0].lower() != "confirm":
            await update.message.reply_text(
                "⚠️ KILL SWITCH\n\n"
                "This will halt ALL trading immediately.\n"
                "To confirm, type: /kill confirm"
            )
            return

        if self._kill_callback:
            try:
                self._kill_callback()
                await update.message.reply_text(
                    "🛑 KILL SWITCH ACTIVATED\n\n"
                    "All trading has been halted."
                )
            except Exception as e:
                await update.message.reply_text(f"Error activating kill switch: {e}")
        else:
            await update.message.reply_text("Kill switch not available")

    async def _cmd_arb_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /arb command - show paper trading status."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_status_callback:
            await update.message.reply_text("Paper trading not available")
            return

        try:
            status = self._arb_status_callback()
            msg = self._format_arb_status(status)
        except Exception as e:
            msg = f"Error getting arb status: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_arb_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /arb_start command - start paper trading."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_start_callback:
            await update.message.reply_text("Paper trading start not available")
            return

        # Check if already running
        if self._arb_status_callback:
            try:
                status = self._arb_status_callback()
                if status.get("active"):
                    await update.message.reply_text("✅ Paper trading already running")
                    return
            except Exception:
                pass  # Continue with start attempt

        try:
            self._arb_start_callback()
            await update.message.reply_text("🚀 Paper trading STARTED")
        except Exception as e:
            await update.message.reply_text(f"Error starting paper trading: {e}")

    async def _cmd_arb_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /arb_stop command - stop paper trading."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_stop_callback:
            await update.message.reply_text("Paper trading stop not available")
            return

        try:
            self._arb_stop_callback()
            await update.message.reply_text("⏹️ Paper trading STOPPED")
        except Exception as e:
            await update.message.reply_text(f"Error stopping paper trading: {e}")

    async def _cmd_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pos command - show detailed position."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_position_callback:
            await update.message.reply_text("Position info not available")
            return

        try:
            pos = self._arb_position_callback()
            msg = self._format_position(pos)
        except Exception as e:
            msg = f"Error getting position: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /history command - show funding payment history."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_history_callback:
            await update.message.reply_text("History not available")
            return

        try:
            history = self._arb_history_callback(10)
            msg = self._format_history(history)
        except Exception as e:
            msg = f"Error getting history: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stats command - show performance statistics."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_stats_callback:
            await update.message.reply_text("Stats not available")
            return

        try:
            stats = self._arb_stats_callback()
            msg = self._format_stats(stats)
        except Exception as e:
            msg = f"Error getting stats: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_rate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /rate command - show next funding rate info."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_rate_callback:
            await update.message.reply_text("Rate info not available")
            return

        try:
            rate = self._arb_rate_callback()
            msg = self._format_rate(rate)
        except Exception as e:
            msg = f"Error getting rate: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /config command - show configuration."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._arb_config_callback:
            await update.message.reply_text("Config not available")
            return

        try:
            config = self._arb_config_callback()
            msg = self._format_config(config)
        except Exception as e:
            msg = f"Error getting config: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    # ==========================================
    # Directional Bot Commands
    # ==========================================

    async def _cmd_dir_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /dir command - show directional bot status."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._dir_status_callback:
            await update.message.reply_text("Directional bot not running")
            return

        try:
            status = self._dir_status_callback()
            msg = self._format_dir_status(status)
        except Exception as e:
            msg = f"Error getting dir status: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /signals command - show current indicators."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._dir_signals_callback:
            await update.message.reply_text("Signals not available")
            return

        try:
            signals = self._dir_signals_callback()
            msg = self._format_signals(signals)
        except Exception as e:
            msg = f"Error getting signals: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /trades command - show recent trades."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._dir_trades_callback:
            await update.message.reply_text("Trades not available")
            return

        try:
            data = self._dir_trades_callback(10)
            msg = self._format_trades(data)
        except Exception as e:
            msg = f"Error getting trades: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_compare(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /compare command - compare arb vs directional."""
        if not self._is_authorized(update.effective_user.id):
            return

        arb_data = None
        dir_data = None

        if self._arb_status_callback:
            try:
                arb_data = self._arb_status_callback()
            except Exception:
                pass

        if self._dir_status_callback:
            try:
                dir_data = self._dir_status_callback()
            except Exception:
                pass

        msg = self._format_compare(arb_data, dir_data)
        await update.message.reply_text(msg, parse_mode="Markdown")

    def _format_dir_status(self, status: dict) -> str:
        """Format directional bot status."""
        if status.get("error"):
            return f"❌ Error: {status['error']}"

        pos = status.get("position", {})
        unrealized = status.get("unrealized", {})
        stats = status.get("stats", {})
        indicators = status.get("indicators", {})
        btc_price = status.get("btc_price", 0)

        side = pos.get("side", "FLAT")
        side_emoji = "🟢" if side == "LONG" else "🔴" if side == "SHORT" else "⚪"

        # Trend arrow
        ema_12 = indicators.get("ema_12", 0)
        ema_26 = indicators.get("ema_26", 0)
        trend = "📈" if ema_12 > ema_26 else "📉"

        return f"""
📊 *DIRECTIONAL BOT*

*Position:* {side_emoji} `{side}`
{f"  Entry: `${pos.get('entry_price', 0):,.2f}`" if side != "FLAT" else ""}
{f"  Unrealized: `${unrealized.get('pnl_usd', 0):+,.2f}` (`{unrealized.get('pnl_pct', 0):+.2f}%`)" if side != "FLAT" else ""}

*Stats:*
  Trades: `{stats.get('total_trades', 0)}`
  Win Rate: `{stats.get('win_rate', 0):.1f}%`
  Total P&L: `${stats.get('total_pnl', 0):+,.2f}`

*Indicators:* {trend}
  RSI: `{indicators.get('rsi', 50):.0f}`
  FNG: `{indicators.get('fng', 50)}`

₿ BTC: `${btc_price:,.2f}`
        """.strip()

    def _format_signals(self, signals: dict) -> str:
        """Format current signals."""
        if signals.get("error"):
            return f"❌ Error: {signals['error']}"

        trend = signals.get("trend", "NEUTRAL")
        trend_emoji = "📈" if trend == "BULLISH" else "📉" if trend == "BEARISH" else "➡️"

        return f"""
📡 *CURRENT SIGNALS*

*Price:* `${signals.get('price', 0):,.2f}`

*Trend:* {trend_emoji} `{trend}`
  EMA 12: `${signals.get('ema_12', 0):,.2f}`
  EMA 26: `${signals.get('ema_26', 0):,.2f}`
  Spread: `{signals.get('ema_spread_pct', 0):.2f}%`

*RSI:* `{signals.get('rsi', 50):.1f}` - {signals.get('rsi_status', 'NEUTRAL')}

*Fear & Greed:* `{signals.get('fng', 50)}` - {signals.get('fng_status', 'NEUTRAL')}

*Bollinger:* {signals.get('bb_status', 'INSIDE BANDS')}
  Upper: `${signals.get('bb_upper', 0):,.2f}`
  Lower: `${signals.get('bb_lower', 0):,.2f}`
        """.strip()

    def _format_trades(self, data: dict) -> str:
        """Format recent trades."""
        trades = data.get("trades", [])
        stats = data.get("stats", {})

        if not trades:
            return "📜 *Recent Trades*\n\nNo trades yet."

        lines = ["📜 *RECENT TRADES*\n"]

        for t in reversed(trades[-5:]):  # Last 5
            emoji = "✅" if t.get("pnl_usd", 0) > 0 else "❌"
            side = t.get("side", "LONG")
            pnl = t.get("pnl_usd", 0)
            pnl_pct = t.get("pnl_pct", 0)
            lines.append(f"{emoji} {side}: `${pnl:+,.2f}` (`{pnl_pct:+.1f}%`)")

        lines.append(f"\n*Total:* {stats.get('total_trades', 0)} trades")
        lines.append(f"*Win Rate:* `{stats.get('win_rate', 0):.1f}%`")
        lines.append(f"*P&L:* `${stats.get('total_pnl', 0):+,.2f}`")

        return "\n".join(lines)

    def _format_compare(self, arb_data: dict, dir_data: dict) -> str:
        """Format comparison between arb and directional."""
        lines = ["📊 *STRATEGY COMPARISON*\n"]

        # Arb data
        if arb_data and arb_data.get("active"):
            arb_pnl = arb_data.get("pnl", {}).get("all_time", 0)
            arb_days = arb_data.get("pnl", {}).get("days_running", 0)
            arb_win = arb_data.get("pnl", {}).get("win_rate", 0)
            lines.append("*ARB BOT:*")
            lines.append(f"  P&L: `${arb_pnl:+,.2f}`")
            lines.append(f"  Days: `{arb_days:.1f}`")
            lines.append(f"  Win Rate: `{arb_win:.1f}%`")
        else:
            lines.append("*ARB BOT:* Not running")

        lines.append("")

        # Dir data
        if dir_data and dir_data.get("active"):
            stats = dir_data.get("stats", {})
            dir_pnl = stats.get("total_pnl", 0)
            dir_trades = stats.get("total_trades", 0)
            dir_win = stats.get("win_rate", 0)
            lines.append("*DIRECTIONAL BOT:*")
            lines.append(f"  P&L: `${dir_pnl:+,.2f}`")
            lines.append(f"  Trades: `{dir_trades}`")
            lines.append(f"  Win Rate: `{dir_win:.1f}%`")
        else:
            lines.append("*DIRECTIONAL BOT:* Not running")

        # Combined
        lines.append("")
        arb_pnl = arb_data.get("pnl", {}).get("all_time", 0) if arb_data else 0
        dir_pnl = dir_data.get("stats", {}).get("total_pnl", 0) if dir_data else 0
        total = arb_pnl + dir_pnl
        lines.append(f"*COMBINED P&L:* `${total:+,.2f}`")

        return "\n".join(lines)

    # ==========================================
    # Beast Bot Commands
    # ==========================================

    async def _cmd_beast_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /beast command - show beast hybrid bot status."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._beast_status_callback:
            await update.message.reply_text("Beast bot not running")
            return

        try:
            status = self._beast_status_callback()
            msg = self._format_beast_status(status)
        except Exception as e:
            msg = f"Error getting beast status: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mode command - show beast mode details."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not self._beast_mode_callback:
            await update.message.reply_text("Beast mode info not available")
            return

        try:
            mode_data = self._beast_mode_callback()
            msg = self._format_beast_mode(mode_data)
        except Exception as e:
            msg = f"Error getting mode: {e}"

        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_trinity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /trinity command - compare all 3 strategies."""
        if not self._is_authorized(update.effective_user.id):
            return

        arb_data = None
        dir_data = None
        beast_data = None

        if self._arb_status_callback:
            try:
                arb_data = self._arb_status_callback()
            except Exception:
                pass

        if self._dir_status_callback:
            try:
                dir_data = self._dir_status_callback()
            except Exception:
                pass

        if self._beast_status_callback:
            try:
                beast_data = self._beast_status_callback()
            except Exception:
                pass

        msg = self._format_trinity(arb_data, dir_data, beast_data)
        await update.message.reply_text(msg, parse_mode="Markdown")

    def _format_beast_status(self, status: dict) -> str:
        """Format beast bot status."""
        if status.get("error"):
            return f"Error: {status['error']}"

        mode = status.get("mode", "NEUTRAL")
        mode_reason = status.get("mode_reason", "")
        pos = status.get("position", {})
        pnl = status.get("pnl", {})
        stats = status.get("stats", {})
        indicators = status.get("indicators", {})
        btc_price = status.get("btc_price", 0)

        # Mode emoji
        mode_emoji = {
            "FULL_LONG": "+++",
            "HALF_LONG": "++",
            "NEUTRAL": "=",
            "HALF_SHORT": "--",
            "FULL_SHORT": "---",
        }.get(mode, "?")

        # Net exposure
        net_btc = pos.get("net_btc", 0)
        net_value = pos.get("net_value", 0)

        return f"""
*THE BEAST - HYBRID TRADER*

*Mode:* `{mode_emoji} {mode}`
*Reason:* {mode_reason}

*Position:*
  Spot:  `+{pos.get('spot_btc', 0):.6f} BTC` (`${pos.get('spot_value', 0):,.0f}`)
  Perp:  `{pos.get('perp_btc', 0):.6f} BTC` (`${pos.get('perp_value', 0):,.0f}`)
  Net:   `{net_btc:+.6f} BTC` (`${net_value:+,.0f}`)

*P&L:*
  Funding:   `${pnl.get('funding', 0):+,.2f}` ({pnl.get('funding_payments', 0)} payments)
  Direction: `${pnl.get('directional', 0):+,.2f}`
  *TOTAL:*   `${pnl.get('total', 0):+,.2f}`

*Stats:*
  Mode Changes: `{stats.get('mode_changes', 0)}`
  Max Drawdown: `${stats.get('max_drawdown', 0):.2f}`

*Indicators:*
  RSI: `{indicators.get('rsi', 50):.0f}` | FNG: `{indicators.get('fng', 50)}`

BTC: `${btc_price:,.2f}`
        """.strip()

    def _format_beast_mode(self, mode_data: dict) -> str:
        """Format beast mode details."""
        if mode_data.get("error"):
            return f"Error: {mode_data['error']}"

        current_mode = mode_data.get("current_mode", "NEUTRAL")
        reason = mode_data.get("reason", "")
        hedge_ratio = mode_data.get("hedge_ratio", 1.0)
        net_exposure_pct = mode_data.get("net_exposure_pct", 0)
        mode_dist = mode_data.get("mode_distribution", {})
        recent = mode_data.get("recent_changes", [])
        indicators = mode_data.get("indicators", {})

        lines = [
            "*BEAST MODE DETAILS*",
            "",
            f"*Current:* `{current_mode}`",
            f"*Reason:* {reason}",
            f"*Hedge Ratio:* `{hedge_ratio * 100:.0f}%`",
            f"*Net Exposure:* `{net_exposure_pct:+.0f}%`",
            "",
            "*Indicators:*",
            f"  Trend: `{indicators.get('ema_trend', 'NEUTRAL')}`",
            f"  RSI: `{indicators.get('rsi', 50):.0f}`",
            f"  FNG: `{indicators.get('fng', 50)}`",
        ]

        if mode_dist:
            lines.append("")
            lines.append("*Time in Mode:*")
            for m, pct in sorted(mode_dist.items(), key=lambda x: -x[1]):
                lines.append(f"  {m}: `{pct:.1f}%`")

        if recent:
            lines.append("")
            lines.append("*Recent Changes:*")
            for mc in recent[-3:]:
                lines.append(f"  {mc['from']} -> {mc['to']}")

        return "\n".join(lines)

    def _format_trinity(self, arb_data: dict, dir_data: dict, beast_data: dict) -> str:
        """Format trinity comparison - all 3 bots."""
        lines = ["*THE TRINITY - ALL STRATEGIES*", ""]

        # Collect data
        arb_pnl = 0
        dir_pnl = 0
        beast_pnl = 0

        # ARB
        if arb_data and arb_data.get("active"):
            arb_pnl = arb_data.get("pnl", {}).get("all_time", 0)
            arb_win = arb_data.get("pnl", {}).get("win_rate", 0)
            lines.append(f"*ARB BOT* (Safe)")
            lines.append(f"  P&L: `${arb_pnl:+,.2f}`")
            lines.append(f"  Win: `{arb_win:.0f}%` | Risk: LOW")
        else:
            lines.append("*ARB BOT:* Not running")

        lines.append("")

        # DIR
        if dir_data and dir_data.get("active"):
            stats = dir_data.get("stats", {})
            dir_pnl = stats.get("total_pnl", 0)
            dir_trades = stats.get("total_trades", 0)
            dir_win = stats.get("win_rate", 0)
            lines.append(f"*DIR BOT* (Hunter)")
            lines.append(f"  P&L: `${dir_pnl:+,.2f}`")
            lines.append(f"  Trades: `{dir_trades}` | Win: `{dir_win:.0f}%`")
        else:
            lines.append("*DIR BOT:* Not running")

        lines.append("")

        # BEAST
        if beast_data and beast_data.get("active"):
            beast_pnl = beast_data.get("pnl", {}).get("total", 0)
            beast_mode = beast_data.get("mode", "NEUTRAL")
            beast_changes = beast_data.get("stats", {}).get("mode_changes", 0)
            lines.append(f"*BEAST* (Monster)")
            lines.append(f"  P&L: `${beast_pnl:+,.2f}`")
            lines.append(f"  Mode: `{beast_mode}` | Changes: `{beast_changes}`")
        else:
            lines.append("*BEAST:* Not running")

        # Summary
        lines.append("")
        lines.append("─" * 20)
        total = arb_pnl + dir_pnl + beast_pnl
        lines.append(f"*TOTAL P&L:* `${total:+,.2f}`")

        # Best performer
        if arb_pnl or dir_pnl or beast_pnl:
            best = max(
                [("ARB", arb_pnl), ("DIR", dir_pnl), ("BEAST", beast_pnl)],
                key=lambda x: x[1]
            )
            lines.append(f"*Best:* {best[0]} (`${best[1]:+,.2f}`)")

        return "\n".join(lines)

    def _format_arb_status(self, status: dict) -> str:
        """Format arb paper trading status for display."""
        if not status.get("active"):
            return (
                "📊 *Funding Arb - Paper Trading*\n\n"
                f"{status.get('message', 'Not active')}\n\n"
                "Use /arb\\_start to begin paper trading."
            )

        if status.get("error"):
            return f"❌ Error: {status['error']}"

        pos = status.get("position", {})
        funding = status.get("funding", {})
        pnl = status.get("pnl", {})
        btc_price = status.get("btc_price", 0)
        is_paused = status.get("paused", False)

        hours = int(funding.get("next_funding_hours", 0))
        minutes = int((funding.get("next_funding_hours", 0) - hours) * 60)

        status_emoji = "⏸️ PAUSED" if is_paused else "✅ Running"

        return f"""
📊 *FUNDING ARB - PAPER TRADING*

*Position:*
  Spot:  `{pos.get('spot_qty', 0):.4f}` BTC (`${pos.get('notional_usd', 0):,.0f}`)
  Perp: `-{pos.get('perp_qty', 0):.4f}` BTC (`${pos.get('notional_usd', 0):,.0f}`)
  Delta: `$0.00` (neutral)

*Funding:*
  Current Rate: `{funding.get('current_rate_pct', '0%')}`
  Next Payment: `{hours}h {minutes}m`

*P&L Today:* `${pnl.get('today', 0):+,.2f}`
*P&L All-Time:* `${pnl.get('all_time', 0):+,.2f}`

{status_emoji} for `{pnl.get('days_running', 0):.1f}` days
📈 Win Rate: `{pnl.get('win_rate', 0):.1f}%`
₿ BTC: `${btc_price:,.2f}`
        """.strip()

    def _format_position(self, pos: dict) -> str:
        """Format position details for display."""
        if not pos.get("active"):
            return f"📦 *Position*\n\n{pos.get('message', 'No active position')}"

        if pos.get("error"):
            return f"❌ Error: {pos['error']}"

        entry_time = pos.get("entry_time")
        if entry_time:
            entry_str = entry_time.strftime("%b %d, %H:%M UTC")
        else:
            entry_str = "Unknown"

        return f"""
📦 *POSITION DETAILS*

*Entry:* `${pos.get('entry_price', 0):,.2f}` ({entry_str})
*Current:* `${pos.get('current_price', 0):,.2f}`
*Duration:* `{pos.get('duration_hours', 0)}h {pos.get('duration_minutes', 0)}m`

*Spot:* `+{pos.get('spot_qty', 0):.4f}` BTC (`${pos.get('spot_value_usd', 0):,.0f}`)
*Perp:* `-{pos.get('perp_qty', 0):.4f}` BTC (`${pos.get('perp_value_usd', 0):,.0f}`)
*Net Delta:* `$0.00` (neutral)

*Unrealized P&L:* `${pos.get('unrealized_pnl', 0):+,.2f}`
*Funding P&L:* `${pos.get('funding_pnl', 0):+,.2f}`
        """.strip()

    def _format_history(self, history: dict) -> str:
        """Format funding payment history for display."""
        payments = history.get("payments", [])

        if not payments:
            return "📜 *Funding History*\n\nNo payments recorded yet."

        lines = ["📜 *FUNDING HISTORY* (Last 10)\n"]

        for p in payments:
            ts = p.get("timestamp")
            if ts:
                time_str = ts.strftime("%b %d %H:%M")
            else:
                time_str = "Unknown"

            emoji = "✅" if p.get("positive") else "❌"
            lines.append(
                f"`{time_str}` | `{p.get('rate_pct', '0%')}` | `${p.get('payment_usd', 0):+.2f}` {emoji}"
            )

        lines.append(f"\n*Total:* {history.get('count', 0)} payments | `${history.get('total', 0):+,.2f}`")

        return "\n".join(lines)

    def _format_stats(self, stats: dict) -> str:
        """Format performance statistics for display."""
        if not stats.get("active"):
            return f"📈 *Stats*\n\n{stats.get('message', 'No active position')}"

        # Base stats that are always shown
        lines = [
            "📈 *PERFORMANCE STATS*",
            "",
            f"*Win Rate:* `{stats.get('win_rate', 0):.1f}%`",
            f"*Total Payments:* `{stats.get('total_payments', 0)}`",
            f"*Total Received:* `${stats.get('total_received', 0):+,.2f}`",
            f"*Avg Rate:* `{stats.get('avg_rate', 0) * 100:+.4f}%`",
            "",
            f"*Running:* `{stats.get('days_running', 0):.1f}` days",
        ]

        # Projections only if we have sufficient data
        if stats.get("has_sufficient_data") and stats.get("daily_avg") is not None:
            lines.extend([
                f"*Daily Avg:* `${stats.get('daily_avg'):+,.2f}`",
                f"*Projected Monthly:* `${stats.get('monthly_projected'):+,.2f}`",
                f"*Projected APY:* `{stats.get('apy'):.2f}%`",
            ])
        else:
            lines.extend([
                "",
                "_Insufficient data for projections_",
                "_Need 1+ day or 3+ payments_",
            ])

        return "\n".join(lines)

    def _format_rate(self, rate: dict) -> str:
        """Format next funding rate info for display."""
        if rate.get("error"):
            return f"❌ Error: {rate['error']}"

        status = rate.get("status", "ok")
        if status == "warning":
            status_emoji = "⚠️"
        elif status == "paused":
            status_emoji = "⏸️"
        else:
            status_emoji = "✅"

        next_time = rate.get("next_funding_time")
        if next_time:
            time_str = next_time.strftime("%H:%M UTC")
        else:
            time_str = "Unknown"

        return f"""
⏱️ *NEXT FUNDING*

*Predicted Rate:* `{rate.get('predicted_rate', 0) * 100:+.4f}%`
*Expected Payment:* `${rate.get('expected_payment', 0):+,.2f}`

*Time Until:* `{rate.get('hours', 0)}h {rate.get('minutes', 0)}m`
*Settlement:* `{time_str}`

*Status:* {status_emoji} {rate.get('status_text', '')}
        """.strip()

    def _format_config(self, config: dict) -> str:
        """Format configuration for display."""
        status = "✅ Active" if config.get("is_running") else "⏹️ Stopped"
        if config.get("is_paused"):
            status = "⏸️ Paused"

        return f"""
⚙️ *CONFIGURATION*

*Notional:* `${config.get('notional_usd', 0):,.0f}`
*Symbol:* `{config.get('symbol', 'BTCUSDT')}`
*Auto-pause:* `< {config.get('auto_pause_threshold', 0) * 100:.2f}%`
*Rate Check:* Every `{config.get('rate_check_interval_min', 30)}` min

*Status:* {status}
*Telegram:* `{'Enabled' if config.get('telegram_enabled') else 'Disabled'}`
        """.strip()

    def _format_status(self, status: dict) -> str:
        """Format status for display."""
        lines = ["*BTC Elite Trader Status*", ""]

        if "position" in status:
            pos = status["position"]
            if pos:
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                lines.append(f"Position: LONG {pos.get('quantity', 0):.6f} BTC")
                lines.append(f"Entry: ${pos.get('entry_price', 0):,.2f}")
                lines.append(f"Current: ${pos.get('current_price', 0):,.2f}")
                lines.append(f"P&L: {pnl_pct:+.2f}%")
            else:
                lines.append("Position: None")

        lines.append("")

        if "risk" in status:
            risk = status["risk"]
            lines.append(f"Daily P&L: ${risk.get('daily_pnl', 0):+,.2f}")
            lines.append(f"Trades Today: {risk.get('trades_today', 0)}")
            lines.append(f"Drawdown: {risk.get('current_drawdown', 0) * 100:.1f}%")

            if risk.get("is_killed"):
                lines.insert(2, "⚠️ *KILL SWITCH ACTIVE*")
            elif risk.get("is_paused"):
                lines.insert(2, "⏸️ *PAUSED*")
            else:
                lines.insert(2, "✅ *RUNNING*")

        lines.append("")
        lines.append(f"_Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_")

        return "\n".join(lines)

    def _format_balance(self, balance: dict) -> str:
        """Format balance for display."""
        lines = ["*Account Balance*", ""]

        total = balance.get("total", 0)
        free = balance.get("free", 0)
        used = balance.get("used", 0)

        lines.append(f"Total: ${total:,.2f}")
        lines.append(f"Available: ${free:,.2f}")
        lines.append(f"In Use: ${used:,.2f}")

        if "btc" in balance:
            lines.append(f"BTC: {balance['btc']:.8f}")

        return "\n".join(lines)

    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to the default chat.

        Args:
            message: Message text
            parse_mode: Telegram parse mode

        Returns:
            True if sent successfully
        """
        if not self._app or not self.chat_id:
            logger.info("Telegram message (no bot): %s", message[:100])
            return False

        try:
            await self._app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            return True
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def send_message_sync(self, message: str) -> bool:
        """Synchronous wrapper for send_message."""
        if not self._app or not self.chat_id:
            print(f"[Telegram] {message}")
            return False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_message(message))
                return True
            else:
                return loop.run_until_complete(self.send_message(message))
        except Exception as e:
            logger.error("Failed to send sync message: %s", e)
            return False

    async def send_trade_notification(
        self,
        action: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
        reason: str = "",
    ) -> None:
        """Send trade execution notification."""
        emoji = "🟢" if action.upper() == "BUY" else "🔴"
        total = price * quantity

        msg = f"{emoji} *{action.upper()}*\n\n"
        msg += f"Price: ${price:,.2f}\n"
        msg += f"Quantity: {quantity:.6f} BTC\n"
        msg += f"Total: ${total:,.2f}\n"

        if pnl is not None:
            msg += f"P&L: ${pnl:+,.2f}\n"

        if reason:
            msg += f"Reason: {reason}\n"

        msg += f"\n_Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_"

        await self.send_message(msg)

    async def send_alert(self, title: str, message: str, level: str = "info") -> None:
        """Send alert notification."""
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }
        emoji = emoji_map.get(level, "ℹ️")

        msg = f"{emoji} *{title}*\n\n{message}"
        await self.send_message(msg)

    async def start(self) -> None:
        """Start the Telegram bot."""
        if not self._app:
            logger.info("Telegram bot not configured, skipping start")
            return

        self._running = True
        logger.info("Starting Telegram bot...")

        # Start polling in background
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        # Register commands in Telegram menu
        try:
            commands = [
                BotCommand("trinity", "Compare all 3 strategies"),
                BotCommand("arb", "Arb bot status"),
                BotCommand("dir", "Directional bot status"),
                BotCommand("beast", "Beast hybrid bot status"),
                BotCommand("mode", "Beast mode details"),
                BotCommand("compare", "Compare arb vs dir"),
                BotCommand("signals", "Current indicators"),
                BotCommand("trades", "Recent dir trades"),
                BotCommand("pos", "Arb position info"),
                BotCommand("stats", "Arb statistics"),
                BotCommand("rate", "Next funding rate"),
                BotCommand("help", "Show all commands"),
            ]
            await self._app.bot.set_my_commands(commands)
            logger.info("Telegram commands registered")
        except Exception as e:
            logger.warning("Failed to register commands: %s", e)

        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self._app:
            return

        self._running = False

        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

        logger.info("Telegram bot stopped")


# Fallback for when telegram library not installed
class TelegramFallback:
    """Fallback when python-telegram-bot not installed."""

    def __init__(self, *args, **kwargs):
        logger.info("Using Telegram fallback (console output)")

    def set_callbacks(self, *args, **kwargs):
        pass

    async def send_message(self, message: str, **kwargs) -> bool:
        print(f"[Telegram] {message}")
        return True

    def send_message_sync(self, message: str) -> bool:
        print(f"[Telegram] {message}")
        return True

    async def send_trade_notification(self, action: str, price: float, quantity: float, **kwargs):
        total = price * quantity
        print(f"[Trade] {action}: {quantity:.6f} BTC @ ${price:,.2f} (${total:,.2f})")

    async def send_alert(self, title: str, message: str, **kwargs):
        print(f"[Alert] {title}: {message}")

    async def start(self):
        pass

    async def stop(self):
        pass


# Export appropriate class based on availability
if not HAS_TELEGRAM:
    TelegramControl = TelegramFallback

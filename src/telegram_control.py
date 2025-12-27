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
    from telegram import Update
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


class TelegramControl:
    """
    Telegram bot for monitoring and control.

    Commands:
    - /status - System status, position, P&L
    - /balance - Account balance
    - /pause - Pause trading
    - /resume - Resume trading
    - /kill confirm - Emergency kill switch
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

/status - Show system status, position, P&L
/balance - Show account balance
/pause - Pause trading
/resume - Resume trading
/kill confirm - Emergency kill switch (stops all trading)
/help - Show this help message

Trading bot is running in {mode} mode.
        """.format(mode="LIVE" if self._running else "STANDBY")

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._is_authorized(update.effective_user.id):
            return

        if self._status_callback:
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

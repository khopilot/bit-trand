"""
Telegram Notifier for Paper Trading

Sends paper trading notifications to Telegram for funding events,
status updates, and alerts.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

from .position_tracker import FundingPayment, PaperPosition

logger = logging.getLogger("btc_trader.paper_trading.telegram")


class TelegramNotifier:
    """
    Sends paper trading notifications to Telegram.

    Features:
    - Funding payment notifications (every 8 hours)
    - Status updates
    - Startup/shutdown notifications
    - Alert notifications
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    TIMEOUT = 10

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self._enabled = bool(self.bot_token and self.chat_id)

        if self._enabled:
            logger.info("TelegramNotifier initialized")
        else:
            logger.warning(
                "TelegramNotifier disabled - missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
            )

    def is_enabled(self) -> bool:
        """Check if Telegram notifications are enabled."""
        return self._enabled

    def _send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to Telegram.

        Args:
            text: Message text
            parse_mode: Parse mode (Markdown or HTML)

        Returns:
            True if sent successfully
        """
        if not self._enabled:
            # Print to console as fallback
            print(f"\n[TELEGRAM] {text}\n")
            return False

        try:
            url = self.API_URL.format(token=self.bot_token)
            response = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()

            logger.debug("Telegram message sent successfully")
            return True

        except requests.RequestException as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def send_funding_notification(
        self,
        payment: FundingPayment,
        position: PaperPosition,
        all_time_summary: Dict,
    ) -> bool:
        """
        Send funding payment notification.

        Args:
            payment: The funding payment record
            position: Current position
            all_time_summary: All-time performance summary
        """
        # Determine emoji based on payment direction
        if payment.payment_usd >= 0:
            emoji = "💰"
            direction = "RECEIVED"
        else:
            emoji = "💸"
            direction = "PAID"

        days_running = all_time_summary.get("days_running", 0)
        win_rate = all_time_summary.get("win_rate_pct", 0)

        message = f"""
{emoji} *FUNDING PAYMENT*

Rate: `{payment.rate * 100:+.4f}%`
Payment: `{payment.payment_usd:+.2f}` USD [{direction}]
BTC Price: `${payment.btc_price:,.2f}`

📊 Cumulative: `${payment.cumulative_usd:,.2f}`
📅 Running: `{days_running:.1f}` days
📈 Win Rate: `{win_rate:.1f}%`
"""
        return self._send_message(message.strip())

    def send_status_update(
        self,
        position: PaperPosition,
        funding_rate: float,
        btc_price: float,
        next_funding_hours: float,
        daily_summary: Dict,
        all_time_summary: Dict,
    ) -> bool:
        """
        Send periodic status update.

        Args:
            position: Current position
            funding_rate: Current funding rate
            btc_price: Current BTC price
            next_funding_hours: Hours until next funding
            daily_summary: Daily P&L summary
            all_time_summary: All-time performance summary
        """
        hours = int(next_funding_hours)
        minutes = int((next_funding_hours - hours) * 60)

        daily_pnl = daily_summary.get("total_today", 0)
        all_time_pnl = all_time_summary.get("total_funding", 0)
        days_running = all_time_summary.get("days_running", 0)
        win_rate = all_time_summary.get("win_rate_pct", 0)

        message = f"""
📊 *FUNDING ARB - STATUS*

*Position:*
  Spot:  `{position.spot_qty:.4f}` BTC (`${position.notional_usd:,.0f}`)
  Perp: `-{position.perp_qty:.4f}` BTC (`${position.notional_usd:,.0f}`)
  Delta: `$0.00` (neutral)

*Funding:*
  Current Rate: `{funding_rate * 100:+.4f}%`
  Next Payment: `{hours}h {minutes}m`

*P&L Today:* `${daily_pnl:+,.2f}`
*P&L All-Time:* `${all_time_pnl:+,.2f}`

✅ Running for `{days_running:.1f}` days
📈 Win Rate: `{win_rate:.1f}%`
"""
        return self._send_message(message.strip())

    def send_startup_notification(
        self,
        position: PaperPosition,
        btc_price: float,
        is_new: bool = False,
    ) -> bool:
        """
        Send startup notification.

        Args:
            position: The position
            btc_price: Current BTC price
            is_new: Whether this is a new position
        """
        mode = "NEW POSITION" if is_new else "RESUMED"
        emoji = "🚀" if is_new else "▶️"

        message = f"""
{emoji} *PAPER TRADING {mode}*

💵 Notional: `${position.notional_usd:,.2f}`
₿ BTC Price: `${btc_price:,.2f}`
📦 BTC Qty: `{position.spot_qty:.6f}`

Strategy: Long Spot + Short Perp
Goal: Capture funding every 8h

_Notifications enabled for funding events_
"""
        return self._send_message(message.strip())

    def send_shutdown_notification(
        self,
        position: PaperPosition,
        all_time_summary: Dict,
    ) -> bool:
        """
        Send shutdown notification.

        Args:
            position: The position
            all_time_summary: All-time performance summary
        """
        total_funding = all_time_summary.get("total_funding", 0)
        days_running = all_time_summary.get("days_running", 0)
        win_rate = all_time_summary.get("win_rate_pct", 0)
        total_payments = all_time_summary.get("total_payments", 0)

        message = f"""
⏹️ *PAPER TRADING STOPPED*

📅 Duration: `{days_running:.2f}` days
🔢 Payments: `{total_payments}`
📈 Win Rate: `{win_rate:.1f}%`

💰 *Final P&L:* `${total_funding:+,.2f}`

_Paper trading simulation ended_
"""
        return self._send_message(message.strip())

    def send_daily_summary(
        self,
        position: PaperPosition,
        daily_summary: Dict,
        all_time_summary: Dict,
    ) -> bool:
        """
        Send daily summary at 00:00 UTC.

        Args:
            position: Current position
            daily_summary: Yesterday's summary
            all_time_summary: All-time summary
        """
        daily_funding = daily_summary.get("funding_received", 0)
        daily_payments = daily_summary.get("funding_payments", 0)
        total_funding = all_time_summary.get("total_funding", 0)
        days_running = all_time_summary.get("days_running", 0)

        # Calculate daily rate
        daily_rate_pct = (daily_funding / position.notional_usd * 100) if position.notional_usd > 0 else 0

        message = f"""
📅 *DAILY SUMMARY*

Yesterday:
  Payments: `{daily_payments}`
  Funding: `${daily_funding:+,.2f}`
  Rate: `{daily_rate_pct:+.3f}%`

All-Time:
  Total: `${total_funding:+,.2f}`
  Days: `{days_running:.1f}`

_New day starting..._
"""
        return self._send_message(message.strip())

    def send_alert(
        self,
        title: str,
        message: str,
        level: str = "info",
    ) -> bool:
        """
        Send alert notification.

        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error, critical)
        """
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }
        emoji = emoji_map.get(level, "ℹ️")

        text = f"""
{emoji} *{title.upper()}*

{message}
"""
        return self._send_message(text.strip())

    def send_rate_alert(
        self,
        current_rate: float,
        expected_rate: float,
        btc_price: float,
    ) -> bool:
        """
        Send alert when funding rate is anomalous.

        Args:
            current_rate: Current funding rate
            expected_rate: Expected average rate
            btc_price: Current BTC price
        """
        variance_pct = ((current_rate - expected_rate) / expected_rate * 100) if expected_rate != 0 else 0

        if current_rate < 0:
            level = "warning"
            title = "NEGATIVE FUNDING RATE"
            detail = "You will PAY this funding period"
        elif abs(variance_pct) > 100:
            level = "info"
            title = "HIGH FUNDING RATE"
            detail = "Rate significantly above average"
        else:
            return False  # No alert needed

        message = f"""
Rate: `{current_rate * 100:+.4f}%`
Expected: `{expected_rate * 100:.4f}%`
Variance: `{variance_pct:+.1f}%`
BTC: `${btc_price:,.2f}`

{detail}
"""
        return self.send_alert(title, message, level)

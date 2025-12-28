#!/usr/bin/env python3
"""
Directional Paper Trading Simulator

Simulates directional trading based on EMA/RSI/Bollinger signals.
Runs alongside the arb bot for performance comparison.

Usage:
    python -m src.paper_trading.directional_simulator --live
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    fetch_btc_ohlc_binance,
    fetch_fear_greed_history,
    calculate_indicators,
)
from src.paper_trading.directional_tracker import DirectionalTracker, PositionSide
from src.paper_trading.telegram_notifier import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("btc_trader.paper_trading.directional")


class DirectionalPaperTrader:
    """
    Paper trading simulator for directional strategy.

    Uses Elite strategy signals:
    - EMA 12/26 crossover for trend
    - RSI for momentum confirmation
    - Bollinger Bands for oversold/overbought
    - Fear & Greed for sentiment
    """

    # Configuration
    POLL_INTERVAL_SECONDS = 300  # Check every 5 minutes
    SIGNAL_CHECK_INTERVAL_MINUTES = 60  # Generate signals hourly
    STOP_LOSS_PCT = 0.05  # 5% trailing stop

    # Signal thresholds
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 75
    RSI_MOMENTUM_LOW = 45
    RSI_MOMENTUM_HIGH = 70
    FNG_FEAR = 30
    FNG_GREED = 70

    def __init__(
        self,
        notional_usd: float = 10000.0,
        telegram_enabled: bool = True,
    ):
        self.notional_usd = notional_usd

        # Components
        self.tracker = DirectionalTracker(notional_usd)
        self.telegram = TelegramNotifier() if telegram_enabled else None
        self._telegram_enabled = telegram_enabled

        # State
        self._running = False
        self._last_signal_check: Optional[datetime] = None
        self._current_indicators: Dict = {}

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            "DirectionalPaperTrader initialized: notional=$%.2f, telegram=%s",
            notional_usd, telegram_enabled
        )

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def start_live_simulation(self) -> None:
        """Start continuous simulation."""
        self._running = True

        # Get initial data
        indicators = self._fetch_indicators()
        if indicators is None:
            logger.error("Failed to fetch initial data")
            return

        self._current_indicators = indicators
        price = indicators["price"]

        # Print startup
        self._print_startup(price)

        if self.telegram:
            self._send_startup_notification(price)

        # Main loop
        try:
            while self._running:
                self._loop_iteration()
                time.sleep(self.POLL_INTERVAL_SECONDS)

        except Exception as e:
            logger.error("Error in main loop: %s", e)

        finally:
            self._shutdown()

    def _loop_iteration(self) -> None:
        """Single iteration of main loop."""
        now = datetime.now(timezone.utc)

        # Fetch latest data
        indicators = self._fetch_indicators()
        if not indicators:
            logger.warning("Failed to fetch indicators")
            return

        self._current_indicators = indicators
        price = indicators["price"]

        # Update trailing stop price
        self.tracker.update_price(price)

        # Check stop loss
        if self.tracker.check_stop_loss(price, self.STOP_LOSS_PCT):
            trade = self.tracker.close_position(price, "stop_loss")
            if trade and self.telegram:
                self._send_trade_notification(trade, "stop_loss")

        # Check for signals (hourly)
        if self._should_check_signals(now):
            self._check_signals(indicators)

    def _should_check_signals(self, now: datetime) -> bool:
        """Check if we should evaluate signals."""
        if self._last_signal_check is None:
            return True

        elapsed = (now - self._last_signal_check).total_seconds()
        return elapsed >= self.SIGNAL_CHECK_INTERVAL_MINUTES * 60

    def _fetch_indicators(self) -> Optional[Dict]:
        """Fetch price data and calculate indicators."""
        try:
            # Get OHLC data
            df = fetch_btc_ohlc_binance(days=30)
            if df.empty:
                logger.error("Empty OHLC data")
                return None

            # Calculate indicators
            df = calculate_indicators(df)

            # Get Fear & Greed
            fng_df = fetch_fear_greed_history(days=1)
            fng_value = fng_df["FNG_Value"].iloc[-1] if not fng_df.empty else 50

            # Get latest values
            latest = df.iloc[-1]

            return {
                "price": latest["Close"],
                "ema_12": latest["EMA_12"],
                "ema_26": latest["EMA_26"],
                "rsi": latest["RSI"],
                "bb_upper": latest["BB_Upper"],
                "bb_lower": latest["BB_Lower"],
                "bb_mid": latest["BB_Mid"],
                "atr": latest.get("ATR", 0),
                "fng": fng_value,
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            logger.error("Failed to fetch indicators: %s", e)
            return None

    def _check_signals(self, indicators: Dict) -> None:
        """Check for entry/exit signals."""
        now = datetime.now(timezone.utc)
        self._last_signal_check = now

        price = indicators["price"]
        ema_12 = indicators["ema_12"]
        ema_26 = indicators["ema_26"]
        rsi = indicators["rsi"]
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        fng = indicators["fng"]

        # Trend detection
        ema_bullish = ema_12 > ema_26
        ema_bearish = ema_12 < ema_26

        position = self.tracker.position

        # ==========================================
        # EXIT SIGNALS (check first)
        # ==========================================

        if position.is_long():
            # Exit LONG conditions
            exit_signal = False
            reason = ""

            # Trend reversal
            if ema_bearish and (ema_26 - ema_12) / ema_26 > 0.01:
                exit_signal = True
                reason = "EMA death cross"

            # Blow-off top
            elif price > bb_upper and rsi > self.RSI_OVERBOUGHT and fng > self.FNG_GREED:
                exit_signal = True
                reason = "Blow-off top"

            # RSI overbought
            elif rsi > 80:
                exit_signal = True
                reason = "RSI overbought"

            if exit_signal:
                trade = self.tracker.close_position(price, reason)
                if trade and self.telegram:
                    self._send_trade_notification(trade, reason)
                return

        elif position.is_short():
            # Exit SHORT conditions
            exit_signal = False
            reason = ""

            # Trend reversal
            if ema_bullish and (ema_12 - ema_26) / ema_26 > 0.01:
                exit_signal = True
                reason = "EMA golden cross"

            # Oversold bounce
            elif price < bb_lower and rsi < self.RSI_OVERSOLD and fng < self.FNG_FEAR:
                exit_signal = True
                reason = "Oversold bounce"

            if exit_signal:
                trade = self.tracker.close_position(price, reason)
                if trade and self.telegram:
                    self._send_trade_notification(trade, reason)
                return

        # ==========================================
        # ENTRY SIGNALS (only if FLAT)
        # ==========================================

        if position.is_flat():
            # LONG signals
            long_signal = False
            reason = ""

            # Smart trend entry
            if (ema_bullish and
                self.RSI_MOMENTUM_LOW < rsi < self.RSI_MOMENTUM_HIGH and
                fng < self.FNG_GREED):
                long_signal = True
                reason = f"Smart trend (EMA+, RSI={rsi:.0f}, FNG={fng})"

            # Contrarian entry (extreme fear)
            elif (price < bb_lower and
                  rsi < self.RSI_OVERSOLD and
                  fng < self.FNG_FEAR):
                long_signal = True
                reason = f"Contrarian (BB-, RSI={rsi:.0f}, FNG={fng})"

            if long_signal:
                if self.tracker.open_long(price, reason):
                    if self.telegram:
                        self._send_entry_notification("LONG", price, reason)
                return

            # SHORT signals (only in clear downtrend)
            short_signal = False

            if (ema_bearish and
                rsi > 60 and
                fng > 50):
                short_signal = True
                reason = f"Downtrend short (EMA-, RSI={rsi:.0f}, FNG={fng})"

            if short_signal:
                if self.tracker.open_short(price, reason):
                    if self.telegram:
                        self._send_entry_notification("SHORT", price, reason)

    def _print_startup(self, price: float) -> None:
        """Print startup banner."""
        pos = self.tracker.position
        stats = self.tracker.get_stats()

        print("\n" + "=" * 60)
        print("  DIRECTIONAL PAPER TRADER - STARTED")
        print("=" * 60)
        print(f"  Notional:     ${self.notional_usd:,.0f}")
        print(f"  BTC Price:    ${price:,.2f}")
        print(f"  Position:     {pos.side.value}")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Total P&L:    ${stats['total_pnl']:+,.2f}")
        print(f"  Win Rate:     {stats['win_rate']:.1f}%")
        print("=" * 60)
        print("  Strategy: EMA + RSI + BB + Fear&Greed")
        print("  Stop Loss: 5% trailing")
        print("=" * 60 + "\n")

    def _send_startup_notification(self, price: float) -> None:
        """Send Telegram startup notification."""
        if not self.telegram:
            return

        pos = self.tracker.position
        stats = self.tracker.get_stats()

        msg = f"""
*DIRECTIONAL BOT STARTED*

*Capital:* `${self.notional_usd:,.0f}`
*BTC:* `${price:,.2f}`
*Position:* `{pos.side.value}`

*Stats:*
  Trades: `{stats['total_trades']}`
  P&L: `${stats['total_pnl']:+,.2f}`
  Win Rate: `{stats['win_rate']:.1f}%`

_Strategy: EMA + RSI + BB + FNG_
        """.strip()

        self.telegram.send_message(msg)

    def _send_entry_notification(self, side: str, price: float, reason: str) -> None:
        """Send entry notification."""
        if not self.telegram:
            return

        emoji = "🟢" if side == "LONG" else "🔴"
        msg = f"""
{emoji} *{side} OPENED*

*Entry:* `${price:,.2f}`
*Size:* `${self.notional_usd:,.0f}`
*Reason:* {reason}
        """.strip()

        self.telegram.send_message(msg)

    def _send_trade_notification(self, trade, reason: str) -> None:
        """Send trade closed notification."""
        if not self.telegram:
            return

        emoji = "✅" if trade.pnl_usd > 0 else "❌"
        msg = f"""
{emoji} *{trade.side} CLOSED*

*Entry:* `${trade.entry_price:,.2f}`
*Exit:* `${trade.exit_price:,.2f}`
*P&L:* `${trade.pnl_usd:+,.2f}` (`{trade.pnl_pct:+.2f}%`)
*Reason:* {reason}
        """.strip()

        self.telegram.send_message(msg)

    def _shutdown(self) -> None:
        """Shutdown gracefully."""
        stats = self.tracker.get_stats()

        print("\n" + "=" * 60)
        print("  DIRECTIONAL PAPER TRADER - STOPPED")
        print("=" * 60)
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate:     {stats['win_rate']:.1f}%")
        print(f"  Total P&L:    ${stats['total_pnl']:+,.2f}")
        print("=" * 60 + "\n")

        if self.telegram:
            self.telegram.send_message(
                f"*DIRECTIONAL BOT STOPPED*\n\n"
                f"Trades: `{stats['total_trades']}`\n"
                f"P&L: `${stats['total_pnl']:+,.2f}`"
            )

    # ==========================================
    # API Methods for Telegram Commands
    # ==========================================

    def get_status_dict(self) -> dict:
        """Get status for /dir command."""
        pos = self.tracker.position
        indicators = self._current_indicators
        stats = self.tracker.get_stats()

        if not indicators:
            return {"active": True, "error": "No indicator data yet"}

        price = indicators.get("price", 0)
        unrealized = self.tracker.get_unrealized_pnl(price)

        return {
            "active": True,
            "position": {
                "side": pos.side.value,
                "entry_price": pos.entry_price,
                "quantity_btc": pos.quantity_btc,
                "notional_usd": pos.notional_usd,
            },
            "unrealized": unrealized,
            "btc_price": price,
            "stats": stats,
            "indicators": {
                "ema_12": indicators.get("ema_12", 0),
                "ema_26": indicators.get("ema_26", 0),
                "rsi": indicators.get("rsi", 50),
                "fng": indicators.get("fng", 50),
            },
        }

    def get_signals_dict(self) -> dict:
        """Get current signals for /signals command."""
        indicators = self._current_indicators

        if not indicators:
            return {"error": "No indicator data"}

        ema_12 = indicators.get("ema_12", 0)
        ema_26 = indicators.get("ema_26", 0)
        rsi = indicators.get("rsi", 50)
        fng = indicators.get("fng", 50)
        price = indicators.get("price", 0)
        bb_upper = indicators.get("bb_upper", 0)
        bb_lower = indicators.get("bb_lower", 0)

        # Determine trend
        if ema_12 > ema_26:
            trend = "BULLISH"
            ema_spread = (ema_12 - ema_26) / ema_26 * 100
        else:
            trend = "BEARISH"
            ema_spread = (ema_26 - ema_12) / ema_26 * 100

        # RSI status
        if rsi < 35:
            rsi_status = "OVERSOLD"
        elif rsi > 75:
            rsi_status = "OVERBOUGHT"
        elif 45 <= rsi <= 70:
            rsi_status = "MOMENTUM"
        else:
            rsi_status = "NEUTRAL"

        # FNG status
        if fng < 30:
            fng_status = "EXTREME FEAR"
        elif fng < 50:
            fng_status = "FEAR"
        elif fng < 70:
            fng_status = "GREED"
        else:
            fng_status = "EXTREME GREED"

        # BB position
        if price < bb_lower:
            bb_status = "BELOW LOWER"
        elif price > bb_upper:
            bb_status = "ABOVE UPPER"
        else:
            bb_status = "INSIDE BANDS"

        return {
            "price": price,
            "trend": trend,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "ema_spread_pct": ema_spread,
            "rsi": rsi,
            "rsi_status": rsi_status,
            "fng": fng,
            "fng_status": fng_status,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_status": bb_status,
            "timestamp": indicators.get("timestamp"),
        }

    def get_trades_dict(self, limit: int = 10) -> dict:
        """Get recent trades for /trades command."""
        trades = self.tracker.get_recent_trades(limit)
        stats = self.tracker.get_stats()

        return {
            "trades": [t.to_dict() for t in trades],
            "stats": stats,
        }

    def stop(self) -> None:
        """Stop the simulator."""
        self._running = False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Directional paper trading simulator"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Start live simulation",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=10000,
        help="Position size in USD",
    )

    args = parser.parse_args()

    trader = DirectionalPaperTrader(
        notional_usd=args.notional,
    )

    if args.live:
        trader.start_live_simulation()
    elif args.status:
        status = trader.get_status_dict()
        print(status)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

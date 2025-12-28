#!/usr/bin/env python3
"""
THE BEAST - Hybrid Paper Trading Simulator

The monster that combines ARB + DIRECTIONAL into one.

Features:
- Constant spot position (like arb)
- Dynamic perp hedge based on signals (directional)
- Collects funding AND captures directional moves
- 5 modes: FULL_LONG, HALF_LONG, NEUTRAL, HALF_SHORT, FULL_SHORT

Usage:
    python -m src.paper_trading.beast_simulator --live
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
from src.paper_trading.beast_tracker import BeastTracker, BeastMode, HEDGE_RATIOS
from src.paper_trading.funding_fetcher import BinanceFundingFetcher
from src.paper_trading.telegram_notifier import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("btc_trader.paper_trading.beast")


class BeastPaperTrader:
    """
    THE BEAST - Hybrid paper trading simulator.

    Combines arb (constant spot) with directional (variable perp).

    Signal -> Mode mapping:
    - VERY BULLISH: RSI<30, FNG<25, Price<BB_Lower -> FULL_LONG
    - BULLISH: EMA+, RSI 45-70, FNG<60 -> HALF_LONG
    - NEUTRAL: Mixed/unclear signals -> NEUTRAL
    - BEARISH: EMA-, RSI>65, FNG>60 -> HALF_SHORT
    - VERY BEARISH: RSI>80, FNG>80, Price>BB_Upper -> FULL_SHORT
    """

    # Configuration
    POLL_INTERVAL_SECONDS = 300  # Check every 5 minutes
    MODE_CHECK_INTERVAL_MINUTES = 30  # Check mode every 30 min
    FUNDING_CHECK_INTERVAL_HOURS = 8  # Funding every 8 hours

    # Signal thresholds
    RSI_VERY_OVERSOLD = 30
    RSI_OVERSOLD = 35
    RSI_MOMENTUM_LOW = 45
    RSI_MOMENTUM_HIGH = 70
    RSI_OVERBOUGHT = 75
    RSI_VERY_OVERBOUGHT = 80

    FNG_EXTREME_FEAR = 25
    FNG_FEAR = 40
    FNG_NEUTRAL = 50
    FNG_GREED = 60
    FNG_EXTREME_GREED = 80

    # Risk management
    DIRECTIONAL_STOP_PCT = 0.05  # 5% trailing stop
    DAILY_LOSS_LIMIT_PCT = 0.03  # 3% daily loss -> return to neutral

    def __init__(
        self,
        notional_usd: float = 10000.0,
        telegram_enabled: bool = True,
    ):
        self.notional_usd = notional_usd

        # Components
        self.tracker = BeastTracker(notional_usd)
        self.fetcher = BinanceFundingFetcher()
        self.telegram = TelegramNotifier() if telegram_enabled else None
        self._telegram_enabled = telegram_enabled

        # State
        self._running = False
        self._last_mode_check: Optional[datetime] = None
        self._last_funding_check: Optional[datetime] = None
        self._current_indicators: Dict = {}
        self._daily_start_pnl: float = 0.0
        self._daily_start_time: Optional[datetime] = None

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            "BeastPaperTrader initialized: notional=$%.2f, telegram=%s",
            notional_usd, telegram_enabled
        )

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def start_live_simulation(self) -> None:
        """Start continuous simulation."""
        self._running = True

        # Fetch initial data
        indicators = self._fetch_indicators()
        if indicators is None:
            logger.error("Failed to fetch initial data")
            return

        self._current_indicators = indicators
        price = indicators["price"]

        # Initialize position if needed
        if not self.tracker.position or not self.tracker.position.spot_btc:
            self.tracker.initialize_position(price)

        # Set daily tracking
        now = datetime.now(timezone.utc)
        self._daily_start_pnl = self.tracker.position.total_pnl
        self._daily_start_time = now

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

        # Update price tracking
        self.tracker.update_price(price)

        # Check daily loss limit
        if self._check_daily_loss_limit():
            return

        # Check directional stop loss
        if self.tracker.check_directional_stop(price, self.DIRECTIONAL_STOP_PCT):
            reason = "Trailing stop hit"
            logger.info("Directional stop triggered, returning to NEUTRAL")
            if self.tracker.change_mode(BeastMode.NEUTRAL, price, reason, indicators):
                if self.telegram:
                    self._send_mode_change_notification(BeastMode.NEUTRAL, price, reason)

        # Check for mode change
        if self._should_check_mode(now):
            self._check_mode(indicators)

        # Check for funding
        if self._should_check_funding(now):
            self._collect_funding(price)

    def _should_check_mode(self, now: datetime) -> bool:
        """Check if we should evaluate mode."""
        if self._last_mode_check is None:
            return True
        elapsed = (now - self._last_mode_check).total_seconds()
        return elapsed >= self.MODE_CHECK_INTERVAL_MINUTES * 60

    def _should_check_funding(self, now: datetime) -> bool:
        """Check if funding period has passed."""
        if self._last_funding_check is None:
            return True
        elapsed = (now - self._last_funding_check).total_seconds()
        return elapsed >= self.FUNDING_CHECK_INTERVAL_HOURS * 3600

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

    def _check_mode(self, indicators: Dict) -> None:
        """Determine optimal mode based on signals."""
        now = datetime.now(timezone.utc)
        self._last_mode_check = now

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

        current_mode = self.tracker.position.mode
        new_mode = current_mode
        reason = ""

        # ==========================================
        # VERY BULLISH -> FULL_LONG
        # ==========================================
        if (rsi < self.RSI_VERY_OVERSOLD and
            fng < self.FNG_EXTREME_FEAR and
            price < bb_lower):
            new_mode = BeastMode.FULL_LONG
            reason = f"Extreme fear + oversold (RSI={rsi:.0f}, FNG={fng})"

        # ==========================================
        # BULLISH -> HALF_LONG
        # ==========================================
        elif (ema_bullish and
              self.RSI_MOMENTUM_LOW < rsi < self.RSI_MOMENTUM_HIGH and
              fng < self.FNG_GREED):
            new_mode = BeastMode.HALF_LONG
            reason = f"Bullish trend (EMA+, RSI={rsi:.0f}, FNG={fng})"

        # ==========================================
        # VERY BEARISH -> FULL_SHORT
        # ==========================================
        elif (rsi > self.RSI_VERY_OVERBOUGHT and
              fng > self.FNG_EXTREME_GREED and
              price > bb_upper):
            new_mode = BeastMode.FULL_SHORT
            reason = f"Extreme greed + overbought (RSI={rsi:.0f}, FNG={fng})"

        # ==========================================
        # BEARISH -> HALF_SHORT
        # ==========================================
        elif (ema_bearish and
              rsi > 65 and
              fng > self.FNG_GREED):
            new_mode = BeastMode.HALF_SHORT
            reason = f"Bearish trend (EMA-, RSI={rsi:.0f}, FNG={fng})"

        # ==========================================
        # UNCLEAR -> NEUTRAL
        # ==========================================
        else:
            # Return to neutral if signals are mixed
            if current_mode != BeastMode.NEUTRAL:
                # Only return to neutral if we've been in mode for a while
                # and signals have become unclear
                new_mode = BeastMode.NEUTRAL
                reason = f"Mixed signals (RSI={rsi:.0f}, FNG={fng})"

        # Apply mode change
        if new_mode != current_mode:
            if self.tracker.change_mode(new_mode, price, reason, indicators):
                logger.info("Mode changed: %s -> %s", current_mode.value, new_mode.value)
                if self.telegram:
                    self._send_mode_change_notification(new_mode, price, reason)

    def _collect_funding(self, price: float) -> None:
        """Collect funding payment."""
        now = datetime.now(timezone.utc)
        self._last_funding_check = now

        try:
            rate_data = self.fetcher.get_current_funding_rate()
            if rate_data is None:
                logger.warning("Failed to fetch funding rate")
                return

            payment = self.tracker.record_funding(rate_data.funding_rate, price)

            if payment != 0:
                logger.info("Funding collected: $%.4f", payment)
                # Don't spam Telegram with every funding

        except Exception as e:
            logger.error("Failed to collect funding: %s", e)

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is hit."""
        if not self.tracker.position:
            return False

        daily_pnl = self.tracker.position.total_pnl - self._daily_start_pnl
        daily_loss_pct = abs(daily_pnl) / self.notional_usd

        if daily_pnl < 0 and daily_loss_pct >= self.DAILY_LOSS_LIMIT_PCT:
            if self.tracker.position.mode != BeastMode.NEUTRAL:
                price = self._current_indicators.get("price", 0)
                reason = f"Daily loss limit ({daily_loss_pct:.1%})"
                self.tracker.change_mode(BeastMode.NEUTRAL, price, reason)
                if self.telegram:
                    self._send_mode_change_notification(
                        BeastMode.NEUTRAL, price, reason
                    )
                return True

        return False

    def _print_startup(self, price: float) -> None:
        """Print startup banner."""
        pos = self.tracker.position
        stats = self.tracker.get_stats()

        print("\n" + "=" * 70)
        print("        THE BEAST - HYBRID PAPER TRADER - STARTED")
        print("=" * 70)
        print(f"  Notional:     ${self.notional_usd:,.0f}")
        print(f"  BTC Price:    ${price:,.2f}")
        print("-" * 70)
        print(f"  Mode:         {pos.mode.value}")
        print(f"  Spot:         +{pos.spot_btc:.6f} BTC")
        print(f"  Perp:         {pos.perp_btc:.6f} BTC")
        print(f"  Net Exposure: {pos.get_net_exposure():.6f} BTC")
        print("-" * 70)
        print(f"  Funding P&L:  ${pos.funding_collected:+,.2f}")
        print(f"  Direction P&L:${pos.directional_pnl:+,.2f}")
        print(f"  TOTAL P&L:    ${pos.total_pnl:+,.2f}")
        print("-" * 70)
        print(f"  Mode Changes: {stats['mode_changes']}")
        print(f"  Max Drawdown: ${stats['max_drawdown']:.2f}")
        print("=" * 70)
        print("  Strategy: ARB + DIRECTIONAL HYBRID")
        print("  Modes: FULL_LONG | HALF_LONG | NEUTRAL | HALF_SHORT | FULL_SHORT")
        print("=" * 70 + "\n")

    def _send_startup_notification(self, price: float) -> None:
        """Send Telegram startup notification."""
        if not self.telegram:
            return

        pos = self.tracker.position
        stats = self.tracker.get_stats()

        msg = f"""
*THE BEAST STARTED*

*Capital:* `${self.notional_usd:,.0f}`
*BTC:* `${price:,.2f}`

*Position:*
  Mode: `{pos.mode.value}`
  Spot: `+{pos.spot_btc:.6f} BTC`
  Perp: `{pos.perp_btc:.6f} BTC`
  Net: `{pos.get_net_exposure():.6f} BTC`

*P&L:*
  Funding: `${pos.funding_collected:+,.2f}`
  Direction: `${pos.directional_pnl:+,.2f}`
  Total: `${pos.total_pnl:+,.2f}`

_Hybrid Strategy: ARB + DIRECTIONAL_
        """.strip()

        self.telegram.send_message(msg)

    def _send_mode_change_notification(
        self, mode: BeastMode, price: float, reason: str
    ) -> None:
        """Send mode change notification."""
        if not self.telegram:
            return

        mode_emoji = {
            BeastMode.FULL_LONG: "+++",
            BeastMode.HALF_LONG: "++",
            BeastMode.NEUTRAL: "=",
            BeastMode.HALF_SHORT: "--",
            BeastMode.FULL_SHORT: "---",
        }

        pos = self.tracker.position

        msg = f"""
*BEAST MODE CHANGE*

*New Mode:* `{mode_emoji.get(mode, '')} {mode.value}`
*Reason:* {reason}

*Position:*
  Spot: `+{pos.spot_btc:.6f} BTC`
  Perp: `{pos.perp_btc:.6f} BTC`
  Net: `{pos.get_net_exposure():.6f} BTC`

*BTC:* `${price:,.2f}`
        """.strip()

        self.telegram.send_message(msg)

    def _shutdown(self) -> None:
        """Shutdown gracefully."""
        pos = self.tracker.position
        stats = self.tracker.get_stats()

        print("\n" + "=" * 70)
        print("        THE BEAST - STOPPED")
        print("=" * 70)
        print(f"  Mode Changes: {stats['mode_changes']}")
        print(f"  Funding P&L:  ${pos.funding_collected:+,.2f}")
        print(f"  Direction P&L:${pos.directional_pnl:+,.2f}")
        print(f"  TOTAL P&L:    ${pos.total_pnl:+,.2f}")
        print(f"  Max Drawdown: ${stats['max_drawdown']:.2f}")
        print("=" * 70 + "\n")

        if self.telegram:
            self.telegram.send_message(
                f"*THE BEAST STOPPED*\n\n"
                f"Mode Changes: `{stats['mode_changes']}`\n"
                f"Total P&L: `${pos.total_pnl:+,.2f}`"
            )

    # ==========================================
    # API Methods for Telegram Commands
    # ==========================================

    def get_status_dict(self) -> dict:
        """Get status for /beast command."""
        pos = self.tracker.position
        indicators = self._current_indicators
        stats = self.tracker.get_stats()

        if not pos:
            return {"active": True, "error": "No position initialized"}

        # Use last known price or entry price if no indicators yet
        price = indicators.get("price", 0) if indicators else pos.spot_entry_price

        # Calculate live P&L
        directional_pnl = 0
        if pos.mode != BeastMode.NEUTRAL:
            net_btc = pos.get_net_exposure()
            price_change = price - pos.mode_entry_price
            directional_pnl = net_btc * price_change

        total_pnl = pos.funding_collected + directional_pnl

        return {
            "active": True,
            "mode": pos.mode.value,
            "mode_reason": pos.mode_reason,
            "position": {
                "spot_btc": pos.spot_btc,
                "perp_btc": pos.perp_btc,
                "net_btc": pos.get_net_exposure(),
                "spot_value": pos.spot_btc * price,
                "perp_value": pos.perp_btc * price,
                "net_value": pos.get_net_exposure() * price,
                "hedge_ratio": HEDGE_RATIOS[pos.mode],
            },
            "pnl": {
                "funding": pos.funding_collected,
                "funding_payments": pos.funding_payments,
                "directional": directional_pnl,
                "total": total_pnl,
            },
            "btc_price": price,
            "stats": stats,
            "indicators": {
                "ema_12": indicators.get("ema_12", 0) if indicators else 0,
                "ema_26": indicators.get("ema_26", 0) if indicators else 0,
                "rsi": indicators.get("rsi", 50) if indicators else 50,
                "fng": indicators.get("fng", 50) if indicators else 50,
            },
        }

    def get_mode_dict(self) -> dict:
        """Get mode details for /mode command."""
        pos = self.tracker.position
        indicators = self._current_indicators

        if not pos:
            return {"error": "No position initialized"}

        mode_dist = self.tracker.get_mode_distribution()
        recent_changes = self.tracker.get_recent_mode_changes(5)

        return {
            "current_mode": pos.mode.value,
            "reason": pos.mode_reason,
            "hedge_ratio": HEDGE_RATIOS[pos.mode],
            "net_exposure_pct": pos.get_net_exposure_pct(),
            "mode_distribution": mode_dist,
            "recent_changes": [
                {
                    "from": mc.from_mode,
                    "to": mc.to_mode,
                    "reason": mc.reason,
                    "price": mc.price,
                    "time": mc.timestamp.isoformat(),
                }
                for mc in recent_changes
            ],
            "indicators": {
                "rsi": indicators.get("rsi", 50) if indicators else 50,
                "fng": indicators.get("fng", 50) if indicators else 50,
                "ema_trend": "BULLISH" if indicators and indicators.get("ema_12", 0) > indicators.get("ema_26", 0) else "UNKNOWN",
            },
        }

    def stop(self) -> None:
        """Stop the simulator."""
        self._running = False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="THE BEAST - Hybrid paper trading simulator"
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
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram notifications",
    )

    args = parser.parse_args()

    trader = BeastPaperTrader(
        notional_usd=args.notional,
        telegram_enabled=not args.no_telegram,
    )

    if args.live:
        trader.start_live_simulation()
    elif args.status:
        status = trader.get_status_dict()
        import json
        print(json.dumps(status, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

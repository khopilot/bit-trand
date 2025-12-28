#!/usr/bin/env python3
"""
Funding Arbitrage Paper Trader

Simulates the Always-In funding arbitrage strategy in real-time.
Fetches live funding rates from Binance and tracks simulated P&L.

Usage:
    python -m src.paper_trading.simulator --live      # Start live simulation
    python -m src.paper_trading.simulator --status    # Show current status
    python -m src.paper_trading.simulator --backfill 7  # Simulate last 7 days
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from .funding_fetcher import BinanceFundingFetcher, FundingRateData
from .position_tracker import PositionTracker, PaperPosition
from .logger import PaperTradingLogger
from .telegram_notifier import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("btc_trader.paper_trading.simulator")


class FundingArbPaperTrader:
    """
    Paper trading simulator for Always-In funding arbitrage strategy.

    Position: Long $10K spot BTC + Short $10K perpetual BTC
    Goal: Capture funding payments every 8 hours (00:00, 08:00, 16:00 UTC)

    Features:
    - Real-time funding rate fetching from Binance
    - Simulated position P&L tracking
    - CSV logging of all events
    - Validation against backtest expectations
    """

    # Expected average rate from backtest (0.0115% per 8h)
    EXPECTED_RATE_PER_PERIOD = 0.000115
    POLL_INTERVAL_SECONDS = 60
    STATUS_UPDATE_INTERVAL_MINUTES = 60
    RATE_CHECK_INTERVAL_MINUTES = 30
    AUTO_PAUSE_THRESHOLD = -0.0001  # -0.01% - pause if rate below this

    def __init__(
        self,
        notional_usd: float = 10000.0,
        symbol: str = "BTCUSDT",
        telegram_enabled: bool = True,
    ):
        self.notional_usd = notional_usd
        self.symbol = symbol

        # Components
        self.fetcher = BinanceFundingFetcher(symbol)
        self.tracker = PositionTracker(notional_usd)
        self.logger = PaperTradingLogger()

        # Telegram notifications
        self.telegram = TelegramNotifier() if telegram_enabled else None
        self._telegram_enabled = telegram_enabled

        # State
        self._running = False
        self._is_paused = False
        self._last_status_update: Optional[datetime] = None
        self._last_funding_time: Optional[datetime] = None
        self._last_daily_summary: Optional[datetime] = None
        self._last_rate_check: Optional[datetime] = None
        self._processed_funding_times: set = set()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            "FundingArbPaperTrader initialized: notional=$%.2f, symbol=%s, telegram=%s",
            notional_usd,
            symbol,
            telegram_enabled,
        )

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, stopping...")
        self._running = False

    def start_live_simulation(self) -> None:
        """
        Start continuous live simulation.

        Main loop:
        1. Fetch current funding rate
        2. Check if funding event occurred
        3. Update status hourly
        4. Sleep and repeat
        """
        self._running = True

        # Get current price
        price = self.fetcher.get_mark_price()
        if not price:
            logger.error("Failed to get initial price, aborting")
            return

        # Initialize or resume position
        is_new = False
        if not self.tracker.position or not self.tracker.position.is_active():
            self.tracker.initialize_position(price)
            is_new = True

        position = self.tracker.position

        # Print startup banner
        self.logger.print_startup_banner(position, price, is_new)
        self.logger.log_start(position, price)

        # Send Telegram startup notification
        if self.telegram:
            self.telegram.send_startup_notification(position, price, is_new)

        # Main loop
        try:
            while self._running:
                self._loop_iteration()
                time.sleep(self.POLL_INTERVAL_SECONDS)

        except Exception as e:
            logger.error("Unexpected error in main loop: %s", e)
            self.logger.log_error(str(e))

        finally:
            self._shutdown()

    def _loop_iteration(self) -> None:
        """Single iteration of the main loop."""
        now = datetime.now(timezone.utc)

        # Fetch current data
        rate_data = self.fetcher.get_current_funding_rate()
        if not rate_data:
            logger.warning("Failed to fetch funding rate, skipping iteration")
            return

        # Check predicted rate every 30 minutes for auto-pause
        if self._should_check_rate(now):
            self._check_predicted_rate()

        # Check for funding event (only if not paused)
        if not self._is_paused and self._should_process_funding(now, rate_data):
            self._process_funding_event(rate_data)

        # Hourly status update
        if self._should_update_status(now):
            self._update_status(rate_data)

        # Daily summary at 00:00 UTC
        if self._should_send_daily_summary(now):
            self._send_daily_summary(rate_data)

    def _should_process_funding(
        self,
        now: datetime,
        rate_data: FundingRateData,
    ) -> bool:
        """
        Check if we should process a funding event.

        Funding occurs at 00:00, 08:00, 16:00 UTC.
        We process if current hour is a funding hour and we haven't
        already processed this specific funding time.
        """
        funding_hours = [0, 8, 16]

        if now.hour not in funding_hours:
            return False

        # Check if we're within 5 minutes after funding time
        if now.minute > 5:
            return False

        # Create a key for this funding time
        funding_key = now.replace(minute=0, second=0, microsecond=0)

        if funding_key in self._processed_funding_times:
            return False

        return True

    def _process_funding_event(self, rate_data: FundingRateData) -> None:
        """Process a funding payment event."""
        now = datetime.now(timezone.utc)
        funding_key = now.replace(minute=0, second=0, microsecond=0)

        logger.info("Processing funding event at %s", funding_key)

        # Record the payment
        try:
            payment = self.tracker.record_funding_payment(
                rate=rate_data.funding_rate,
                current_price=rate_data.mark_price,
            )

            # Log and print
            self.logger.log_funding_event(payment)
            self.logger.print_funding_event(payment)

            # Send Telegram notification
            if self.telegram and self.tracker.position:
                all_time = self.tracker.get_all_time_summary(rate_data.mark_price)
                self.telegram.send_funding_notification(
                    payment=payment,
                    position=self.tracker.position,
                    all_time_summary=all_time,
                )

                # Check for rate anomalies
                self.telegram.send_rate_alert(
                    current_rate=rate_data.funding_rate,
                    expected_rate=self.EXPECTED_RATE_PER_PERIOD,
                    btc_price=rate_data.mark_price,
                )

            # Mark as processed
            self._processed_funding_times.add(funding_key)

            # Clean up old keys (keep last 10)
            if len(self._processed_funding_times) > 10:
                oldest = min(self._processed_funding_times)
                self._processed_funding_times.remove(oldest)

        except Exception as e:
            logger.error("Failed to process funding event: %s", e)
            self.logger.log_error(f"Funding processing failed: {e}")

    def _should_update_status(self, now: datetime) -> bool:
        """Check if we should print a status update."""
        if self._last_status_update is None:
            return True

        elapsed = (now - self._last_status_update).total_seconds()
        return elapsed >= self.STATUS_UPDATE_INTERVAL_MINUTES * 60

    def _should_send_daily_summary(self, now: datetime) -> bool:
        """Check if we should send daily summary (at 00:00 UTC)."""
        if now.hour != 0 or now.minute > 5:
            return False

        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self._last_daily_summary is None:
            return True

        return self._last_daily_summary.date() < today.date()

    def _send_daily_summary(self, rate_data: FundingRateData) -> None:
        """Send daily summary to Telegram."""
        now = datetime.now(timezone.utc)
        self._last_daily_summary = now

        if not self.telegram or not self.tracker.position:
            return

        daily_summary = self.tracker.get_daily_summary(rate_data.mark_price)
        all_time_summary = self.tracker.get_all_time_summary(rate_data.mark_price)

        self.telegram.send_daily_summary(
            position=self.tracker.position,
            daily_summary=daily_summary,
            all_time_summary=all_time_summary,
        )

        # Reset daily tracking
        self.tracker.reset_daily_tracking()

    def _should_check_rate(self, now: datetime) -> bool:
        """Check if we should check the predicted rate."""
        if self._last_rate_check is None:
            return True

        elapsed = (now - self._last_rate_check).total_seconds()
        return elapsed >= self.RATE_CHECK_INTERVAL_MINUTES * 60

    def _check_predicted_rate(self) -> None:
        """
        Check predicted funding rate and auto-pause if negative.

        Auto-pauses position if predicted rate is below threshold to avoid paying.
        Auto-resumes when rate turns positive again.
        """
        now = datetime.now(timezone.utc)
        self._last_rate_check = now

        predicted = self.fetcher.get_predicted_funding_rate()
        if predicted is None:
            logger.warning("Failed to get predicted rate")
            return

        logger.info("Predicted funding rate: %.4f%%", predicted * 100)

        # Auto-pause if rate is significantly negative
        if not self._is_paused and predicted < self.AUTO_PAUSE_THRESHOLD:
            self._pause_position(predicted)

        # Auto-resume if rate turned positive and we're paused
        elif self._is_paused and predicted > 0:
            self._resume_position(predicted)

    def _pause_position(self, rate: float) -> None:
        """Pause position to avoid negative funding."""
        self._is_paused = True
        logger.warning("Position PAUSED: predicted rate %.4f%% is negative", rate * 100)

        if self.telegram:
            self.telegram.send_alert(
                title="AUTO-PAUSE: NEGATIVE FUNDING",
                message=(
                    f"Predicted rate: `{rate*100:+.4f}%`\n"
                    f"Position PAUSED to avoid paying.\n"
                    f"Will resume when rate turns positive."
                ),
                level="warning",
            )

    def _resume_position(self, rate: float) -> None:
        """Resume position when rate turns positive."""
        self._is_paused = False
        logger.info("Position RESUMED: rate %.4f%% is positive", rate * 100)

        if self.telegram:
            self.telegram.send_alert(
                title="RESUMED: POSITIVE FUNDING",
                message=(
                    f"Rate now positive: `{rate*100:+.4f}%`\n"
                    f"Position RESUMED."
                ),
                level="info",
            )

    def _update_status(self, rate_data: FundingRateData) -> None:
        """Print and log status update."""
        now = datetime.now(timezone.utc)
        self._last_status_update = now

        position = self.tracker.position
        if not position:
            return

        # Get summaries
        daily_summary = self.tracker.get_daily_summary(rate_data.mark_price)
        all_time_summary = self.tracker.get_all_time_summary(rate_data.mark_price)

        # Print status
        self.logger.print_status_update(
            position=position,
            funding_rate=rate_data.funding_rate,
            btc_price=rate_data.mark_price,
            next_funding_time=rate_data.next_funding_time,
            daily_summary=daily_summary,
            all_time_summary=all_time_summary,
        )

        # Log status
        self.logger.log_status_update(
            position=position,
            current_price=rate_data.mark_price,
            daily_pnl=daily_summary,
            all_time_pnl=all_time_summary,
        )

    def _shutdown(self) -> None:
        """Shutdown the simulator gracefully."""
        position = self.tracker.position
        if not position:
            return

        price = self.fetcher.get_mark_price() or position.entry_price
        all_time_summary = self.tracker.get_all_time_summary(price)

        # Print final summary
        self.logger.print_final_summary(position, price, all_time_summary)
        self.logger.log_stop(position, price)

        # Send Telegram shutdown notification
        if self.telegram:
            self.telegram.send_shutdown_notification(position, all_time_summary)

        logger.info("Paper trading simulator stopped")

    def get_status_dict(self) -> dict:
        """
        Get current status as a dictionary (for Telegram callbacks).

        Returns:
            Status dictionary with position and P&L info
        """
        position = self.tracker.position

        if not position or not position.is_active():
            return {
                "active": False,
                "message": "No active paper trading position",
            }

        rate_data = self.fetcher.get_current_funding_rate()
        if not rate_data:
            return {
                "active": True,
                "error": "Could not fetch current data",
            }

        daily = self.tracker.get_daily_summary(rate_data.mark_price)
        all_time = self.tracker.get_all_time_summary(rate_data.mark_price)

        # Calculate time to next funding
        next_funding = rate_data.next_funding_time
        now = datetime.now(timezone.utc)
        time_to_funding = (next_funding - now).total_seconds() / 3600

        return {
            "active": True,
            "paused": self._is_paused,
            "position": {
                "notional_usd": position.notional_usd,
                "spot_qty": position.spot_qty,
                "perp_qty": position.perp_qty,
                "entry_price": position.entry_price,
            },
            "funding": {
                "current_rate": rate_data.funding_rate,
                "current_rate_pct": f"{rate_data.funding_rate * 100:+.4f}%",
                "next_funding_hours": time_to_funding,
            },
            "btc_price": rate_data.mark_price,
            "pnl": {
                "today": daily.get("total_today", 0),
                "all_time": all_time.get("total_funding", 0),
                "days_running": all_time.get("days_running", 0),
                "win_rate": all_time.get("win_rate_pct", 0),
            },
        }

    def stop(self) -> None:
        """Stop the simulator (can be called from Telegram)."""
        logger.info("Stop requested")
        self._running = False

    def get_position_dict(self) -> dict:
        """
        Get detailed position info for /pos command.

        Returns:
            Position details dictionary
        """
        position = self.tracker.position

        if not position or not position.is_active():
            return {"active": False, "message": "No active position"}

        rate_data = self.fetcher.get_current_funding_rate()
        if not rate_data:
            return {"active": True, "error": "Could not fetch current price"}

        current_price = rate_data.mark_price
        current_value = position.spot_qty * current_price
        entry_value = position.spot_qty * position.entry_price
        unrealized_pnl = current_value - entry_value

        # Calculate duration
        now = datetime.now(timezone.utc)
        duration = now - position.entry_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)

        return {
            "active": True,
            "entry_price": position.entry_price,
            "entry_time": position.entry_time,
            "current_price": current_price,
            "duration_hours": hours,
            "duration_minutes": minutes,
            "spot_qty": position.spot_qty,
            "perp_qty": position.perp_qty,
            "notional_usd": position.notional_usd,
            "spot_value_usd": current_value,
            "perp_value_usd": current_value,
            "unrealized_pnl": unrealized_pnl,
            "funding_pnl": position.total_funding_collected,
        }

    def get_history(self, limit: int = 10) -> dict:
        """
        Get funding payment history for /history command.

        Args:
            limit: Number of payments to return

        Returns:
            History dictionary with payments list
        """
        payments = self.tracker.get_recent_payments(limit)

        if not payments:
            return {"payments": [], "total": 0, "count": 0}

        payment_list = []
        for p in reversed(payments):  # Most recent first
            payment_list.append({
                "timestamp": p.timestamp,
                "rate": p.rate,
                "rate_pct": p.rate_pct,
                "payment_usd": p.payment_usd,
                "positive": p.payment_usd > 0,
            })

        total = sum(p.payment_usd for p in payments)

        return {
            "payments": payment_list,
            "total": total,
            "count": len(payments),
        }

    # Minimum data thresholds for projections
    MIN_DAYS_FOR_PROJECTION = 1.0
    MIN_PAYMENTS_FOR_PROJECTION = 3

    def get_stats(self) -> dict:
        """
        Get performance statistics for /stats command.

        Returns:
            Stats dictionary with metrics
        """
        position = self.tracker.position

        if not position or not position.is_active():
            return {"active": False, "message": "No active position"}

        rate_data = self.fetcher.get_current_funding_rate()
        current_price = rate_data.mark_price if rate_data else position.entry_price

        all_time = self.tracker.get_all_time_summary(current_price)

        # Calculate projections only with sufficient data
        days_running = all_time.get("days_running", 0)
        total_funding = all_time.get("total_funding", 0)
        total_payments = all_time.get("total_payments", 0)

        # Require minimum 1 day OR 3 payments before showing projections
        has_sufficient_data = (
            days_running >= self.MIN_DAYS_FOR_PROJECTION or
            total_payments >= self.MIN_PAYMENTS_FOR_PROJECTION
        )

        if has_sufficient_data and days_running > 0:
            daily_avg = total_funding / days_running
            monthly_projected = daily_avg * 30
            annual_projected = daily_avg * 365
            apy = (annual_projected / self.notional_usd) * 100
        else:
            # Insufficient data for meaningful projections
            daily_avg = None
            monthly_projected = None
            annual_projected = None
            apy = None

        # Calculate average rate
        payments = self.tracker.get_recent_payments(1000)
        if payments:
            avg_rate = sum(p.rate for p in payments) / len(payments)
        else:
            avg_rate = 0

        return {
            "active": True,
            "win_rate": all_time.get("win_rate_pct", 0),
            "total_payments": total_payments,
            "total_received": total_funding,
            "avg_rate": avg_rate,
            "days_running": days_running,
            "daily_avg": daily_avg,
            "monthly_projected": monthly_projected,
            "annual_projected": annual_projected,
            "apy": apy,
            "has_sufficient_data": has_sufficient_data,
        }

    def get_rate_info(self) -> dict:
        """
        Get next funding rate info for /rate command.

        Returns:
            Rate info dictionary
        """
        rate_data = self.fetcher.get_current_funding_rate()

        if not rate_data:
            return {"error": "Could not fetch rate data"}

        predicted = self.fetcher.get_predicted_funding_rate()
        expected_payment = self.notional_usd * (predicted or rate_data.funding_rate)

        # Calculate time to next funding
        now = datetime.now(timezone.utc)
        time_to_funding = (rate_data.next_funding_time - now).total_seconds()
        hours = int(time_to_funding // 3600)
        minutes = int((time_to_funding % 3600) // 60)

        # Determine status
        if predicted and predicted < self.AUTO_PAUSE_THRESHOLD:
            status = "warning"
            status_text = "Will PAY (rate negative)"
        elif self._is_paused:
            status = "paused"
            status_text = "PAUSED - waiting for positive rate"
        else:
            status = "ok"
            status_text = "Will collect (rate positive)"

        return {
            "predicted_rate": predicted or rate_data.funding_rate,
            "expected_payment": expected_payment,
            "hours": hours,
            "minutes": minutes,
            "next_funding_time": rate_data.next_funding_time,
            "status": status,
            "status_text": status_text,
        }

    def get_config(self) -> dict:
        """
        Get configuration for /config command.

        Returns:
            Config dictionary
        """
        return {
            "notional_usd": self.notional_usd,
            "symbol": self.symbol,
            "auto_pause_threshold": self.AUTO_PAUSE_THRESHOLD,
            "rate_check_interval_min": self.RATE_CHECK_INTERVAL_MINUTES,
            "poll_interval_sec": self.POLL_INTERVAL_SECONDS,
            "is_paused": self._is_paused,
            "is_running": self._running,
            "telegram_enabled": self._telegram_enabled,
        }

    def show_status(self) -> None:
        """Show current position status."""
        position = self.tracker.position

        if not position or not position.is_active():
            print("\n  No active paper trading position.\n")
            print("  Run with --live to start a new simulation.\n")
            return

        # Fetch current price
        rate_data = self.fetcher.get_current_funding_rate()
        if not rate_data:
            print("\n  Error: Could not fetch current price.\n")
            return

        # Get summaries
        daily_summary = self.tracker.get_daily_summary(rate_data.mark_price)
        all_time_summary = self.tracker.get_all_time_summary(rate_data.mark_price)

        # Print status
        self.logger.print_status_update(
            position=position,
            funding_rate=rate_data.funding_rate,
            btc_price=rate_data.mark_price,
            next_funding_time=rate_data.next_funding_time,
            daily_summary=daily_summary,
            all_time_summary=all_time_summary,
        )

    def backfill(self, days: int) -> None:
        """
        Backfill simulation using historical funding rates.

        This validates the simulator against actual historical data.

        Args:
            days: Number of days to backfill
        """
        print(f"\n  Backfilling {days} days of historical funding data...\n")

        # Fetch historical rates
        rates = self.fetcher.get_rates_for_days(days)

        if not rates:
            print("  Error: Could not fetch historical rates.\n")
            return

        print(f"  Fetched {len(rates)} funding periods.\n")

        # Get initial price (from first rate, approximate)
        initial_price = 50000  # Fallback
        current_rate = self.fetcher.get_current_funding_rate()
        if current_rate:
            initial_price = current_rate.mark_price

        # Clear any existing state and create fresh position
        self.tracker.clear_state()
        self.tracker.initialize_position(initial_price)

        # Process each historical rate
        total_funding = 0
        positive_count = 0

        for rate in rates:
            payment_usd = self.notional_usd * rate.funding_rate
            total_funding += payment_usd

            if payment_usd > 0:
                positive_count += 1

            # We don't use record_funding_payment here to avoid
            # persisting all historical payments to state

        # Calculate metrics
        num_payments = len(rates)
        win_rate = (positive_count / num_payments * 100) if num_payments > 0 else 0

        # Expected from backtest
        expected_per_payment = self.notional_usd * self.EXPECTED_RATE_PER_PERIOD
        expected_total = expected_per_payment * num_payments

        # Print summary
        self.logger.print_backfill_summary(
            days=days,
            total_funding=total_funding,
            num_payments=num_payments,
            win_rate=win_rate,
            expected=expected_total,
        )

        # Clear the temporary state
        self.tracker.clear_state()

    def reset(self) -> None:
        """Reset all state and start fresh."""
        self.tracker.clear_state()
        self._processed_funding_times.clear()
        self._last_status_update = None
        print("\n  Paper trading state cleared.\n")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Paper trading simulator for Always-In funding arbitrage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.paper_trading.simulator --live
  python -m src.paper_trading.simulator --status
  python -m src.paper_trading.simulator --backfill 7
  python -m src.paper_trading.simulator --reset
        """,
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Start live paper trading simulation",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current position status",
    )
    parser.add_argument(
        "--backfill",
        type=int,
        metavar="DAYS",
        help="Backfill simulation for N days using historical data",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all paper trading state",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=10000,
        help="Position size in USD (default: 10000)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )

    args = parser.parse_args()

    # Create trader
    trader = FundingArbPaperTrader(
        notional_usd=args.notional,
        symbol=args.symbol,
    )

    # Execute command
    if args.live:
        trader.start_live_simulation()
    elif args.status:
        trader.show_status()
    elif args.backfill:
        trader.backfill(args.backfill)
    elif args.reset:
        trader.reset()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

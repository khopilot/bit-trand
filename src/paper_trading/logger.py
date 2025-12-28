"""
Paper Trading Logger

Handles CSV logging and formatted console output for the paper trading simulator.
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .position_tracker import FundingPayment, PaperPosition

logger = logging.getLogger("btc_trader.paper_trading.logger")


class PaperTradingLogger:
    """
    Logger for paper trading simulation.

    Features:
    - CSV logging with daily rotation
    - Formatted console status updates
    - Funding event logging
    """

    CSV_HEADERS = [
        "timestamp",
        "event_type",
        "funding_rate",
        "payment_usd",
        "cumulative_pnl",
        "btc_price",
        "notes",
    ]

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: Optional[str] = None
        self._csv_path: Optional[Path] = None
        self._csv_initialized = False

    def _get_log_path(self, date: Optional[datetime] = None) -> Path:
        """Get log file path for given date."""
        if date is None:
            date = datetime.now(timezone.utc)
        date_str = date.strftime("%Y%m%d")
        return self.log_dir / f"paper_trading_{date_str}.csv"

    def _ensure_csv_initialized(self) -> None:
        """Ensure CSV file exists with headers."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        # Check for date rollover
        if today != self._current_date:
            self._current_date = today
            self._csv_path = self._get_log_path()
            self._csv_initialized = False

        if not self._csv_initialized:
            if not self._csv_path.exists():
                with open(self._csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.CSV_HEADERS)
                logger.info("Created new log file: %s", self._csv_path)
            self._csv_initialized = True

    def log_event(
        self,
        event_type: str,
        funding_rate: float = 0.0,
        payment_usd: float = 0.0,
        cumulative_pnl: float = 0.0,
        btc_price: float = 0.0,
        notes: str = "",
    ) -> None:
        """
        Log an event to CSV.

        Args:
            event_type: Type of event (funding, status, error, etc.)
            funding_rate: Funding rate as decimal
            payment_usd: Payment amount
            cumulative_pnl: Cumulative P&L
            btc_price: Current BTC price
            notes: Additional notes
        """
        self._ensure_csv_initialized()

        timestamp = datetime.now(timezone.utc).isoformat()

        row = [
            timestamp,
            event_type,
            f"{funding_rate:.6f}",
            f"{payment_usd:.4f}",
            f"{cumulative_pnl:.4f}",
            f"{btc_price:.2f}",
            notes,
        ]

        try:
            with open(self._csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except IOError as e:
            logger.error("Failed to write to CSV: %s", e)

    def log_funding_event(self, payment: FundingPayment) -> None:
        """Log a funding payment event."""
        sign = "+" if payment.payment_usd >= 0 else ""
        notes = f"Rate {payment.rate_pct}"

        self.log_event(
            event_type="funding",
            funding_rate=payment.rate,
            payment_usd=payment.payment_usd,
            cumulative_pnl=payment.cumulative_usd,
            btc_price=payment.btc_price,
            notes=notes,
        )

        logger.info(
            "FUNDING: %s rate -> %s$%.2f | Cumulative: $%.2f",
            payment.rate_pct,
            sign,
            payment.payment_usd,
            payment.cumulative_usd,
        )

    def log_status_update(
        self,
        position: PaperPosition,
        current_price: float,
        daily_pnl: Dict,
        all_time_pnl: Dict,
    ) -> None:
        """Log an hourly status update."""
        self.log_event(
            event_type="status",
            cumulative_pnl=position.total_funding_collected,
            btc_price=current_price,
            notes=f"Daily: ${daily_pnl.get('total_today', 0):.2f}",
        )

    def log_error(self, error_message: str, btc_price: float = 0.0) -> None:
        """Log an error event."""
        self.log_event(
            event_type="error",
            btc_price=btc_price,
            notes=error_message,
        )
        logger.error("Paper trading error: %s", error_message)

    def log_start(self, position: PaperPosition, btc_price: float) -> None:
        """Log simulator start."""
        self.log_event(
            event_type="start",
            btc_price=btc_price,
            cumulative_pnl=position.total_funding_collected,
            notes=f"Notional: ${position.notional_usd:.2f}",
        )

    def log_stop(self, position: PaperPosition, btc_price: float) -> None:
        """Log simulator stop."""
        self.log_event(
            event_type="stop",
            btc_price=btc_price,
            cumulative_pnl=position.total_funding_collected,
            notes=f"Payments: {len(position.funding_payments)}",
        )

    def print_status_update(
        self,
        position: PaperPosition,
        funding_rate: float,
        btc_price: float,
        next_funding_time: datetime,
        daily_summary: Dict,
        all_time_summary: Dict,
    ) -> None:
        """
        Print formatted status update to console.

        Format matches the user's specified output format.
        """
        now = datetime.now(timezone.utc)

        # Calculate time to next funding
        time_to_funding = next_funding_time - now
        hours = int(time_to_funding.total_seconds() // 3600)
        minutes = int((time_to_funding.total_seconds() % 3600) // 60)

        # Determine status
        status = "Running normally"
        status_icon = "OK"

        output = f"""
================================================================================
FUNDING ARB PAPER TRADER - Status Update
================================================================================
Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC
BTC Price: ${btc_price:,.2f}

POSITION:
  Spot:  {position.spot_qty:.4f} BTC (${position.notional_usd:,.2f})
  Perp: -{position.perp_qty:.4f} BTC (${position.notional_usd:,.2f})
  Delta: $0.00 (neutral)

FUNDING:
  Current Rate: {funding_rate * 100:.4f}% ({"positive = we receive" if funding_rate >= 0 else "negative = we pay"})
  Next Payment: {hours}h {minutes}m

P&L TODAY:
  Funding Received: ${daily_summary.get('funding_received', 0):,.2f}
  Unrealized P&L:   ${daily_summary.get('unrealized_pnl', 0):,.2f}
  Total Today:      ${daily_summary.get('total_today', 0):,.2f}

P&L ALL-TIME:
  Total Funding:    ${all_time_summary.get('total_funding', 0):,.2f} (started {all_time_summary.get('days_running', 0):.1f} days ago)
  Expected (backtest): ${all_time_summary.get('expected_total', 0):,.2f}
  Variance:         {all_time_summary.get('variance_pct', 0):+.1f}%
  Win Rate:         {all_time_summary.get('win_rate_pct', 0):.1f}%

STATUS: {status_icon} {status}
================================================================================
"""
        print(output)

    def print_funding_event(self, payment: FundingPayment) -> None:
        """Print funding event to console."""
        sign = "+" if payment.payment_usd >= 0 else ""
        icon = "RECV" if payment.payment_usd >= 0 else "PAID"

        print(f"""
--------------------------------------------------------------------------------
FUNDING EVENT @ {payment.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
Rate: {payment.rate_pct} | Payment: {sign}${abs(payment.payment_usd):.2f} [{icon}]
BTC Price: ${payment.btc_price:,.2f} | Cumulative: ${payment.cumulative_usd:,.2f}
--------------------------------------------------------------------------------
""")

    def print_startup_banner(
        self,
        position: PaperPosition,
        btc_price: float,
        is_new: bool = False,
    ) -> None:
        """Print startup banner."""
        mode = "NEW POSITION" if is_new else "RESUMING"

        print(f"""
================================================================================
        ALWAYS-IN FUNDING ARBITRAGE - PAPER TRADER
================================================================================

  Mode:     {mode}
  Notional: ${position.notional_usd:,.2f}
  BTC Price: ${btc_price:,.2f}
  BTC Qty:   {position.spot_qty:.6f} BTC

  Strategy: Long Spot + Short Perp (Delta Neutral)
  Goal:     Capture funding payments every 8 hours

================================================================================
  Press Ctrl+C to stop
================================================================================
""")

    def print_final_summary(
        self,
        position: PaperPosition,
        btc_price: float,
        all_time_summary: Dict,
    ) -> None:
        """Print final summary when stopping."""
        print(f"""
================================================================================
        PAPER TRADING SESSION ENDED
================================================================================

  Duration:        {all_time_summary.get('days_running', 0):.2f} days
  Total Payments:  {all_time_summary.get('total_payments', 0)}
  Win Rate:        {all_time_summary.get('win_rate_pct', 0):.1f}%

  FINAL P&L:
    Funding Collected: ${all_time_summary.get('total_funding', 0):,.2f}
    Unrealized P&L:    ${all_time_summary.get('unrealized_pnl', 0):,.2f}
    Total P&L:         ${all_time_summary.get('total_pnl', 0):,.2f}

  PERFORMANCE:
    Daily Average:     ${all_time_summary.get('daily_avg', 0):,.2f}
    Annualized:        {all_time_summary.get('annualized_pct', 0):.2f}%

  VALIDATION:
    Expected (backtest): ${all_time_summary.get('expected_total', 0):,.2f}
    Actual:              ${all_time_summary.get('total_funding', 0):,.2f}
    Variance:            {all_time_summary.get('variance_pct', 0):+.1f}%

================================================================================
""")

    def print_backfill_summary(
        self,
        days: int,
        total_funding: float,
        num_payments: int,
        win_rate: float,
        expected: float,
    ) -> None:
        """Print backfill summary."""
        variance_pct = ((total_funding - expected) / expected * 100) if expected > 0 else 0

        print(f"""
================================================================================
        BACKFILL COMPLETE - {days} Days
================================================================================

  Funding Periods: {num_payments}
  Total Funding:   ${total_funding:,.2f}
  Win Rate:        {win_rate:.1f}%

  VALIDATION:
    Expected (backtest): ${expected:,.2f}
    Actual:              ${total_funding:,.2f}
    Variance:            {variance_pct:+.1f}%

  Status: {"MATCH" if abs(variance_pct) < 10 else "DIVERGENCE DETECTED"}
================================================================================
""")

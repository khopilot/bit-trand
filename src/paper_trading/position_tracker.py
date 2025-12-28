"""
Position Tracker for Paper Trading

Tracks simulated position state, funding payments, and calculates P&L
for the Always-In funding arbitrage strategy.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("btc_trader.paper_trading.tracker")


@dataclass
class FundingPayment:
    """Record of a single funding payment."""
    timestamp: datetime
    rate: float
    payment_usd: float
    btc_price: float
    cumulative_usd: float = 0.0

    @property
    def rate_pct(self) -> str:
        """Rate as formatted percentage."""
        return f"{self.rate * 100:+.4f}%"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "rate": self.rate,
            "rate_pct": self.rate_pct,
            "payment_usd": self.payment_usd,
            "btc_price": self.btc_price,
            "cumulative_usd": self.cumulative_usd,
        }


@dataclass
class PaperPosition:
    """
    Simulated Always-In funding arbitrage position.

    Position structure:
    - Long spot BTC (notional_usd worth)
    - Short perpetual BTC (same notional, hedge)

    Net delta: ~0 (market neutral)
    Profit source: Funding payments when rate > 0
    """
    notional_usd: float = 10000.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    spot_qty: float = 0.0
    perp_qty: float = 0.0

    # P&L tracking
    funding_payments: List[FundingPayment] = field(default_factory=list)
    total_funding_collected: float = 0.0

    # Daily tracking
    daily_funding_start: float = 0.0
    daily_start_time: Optional[datetime] = None

    def is_active(self) -> bool:
        """Check if position is active."""
        return self.entry_time is not None and self.spot_qty > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "notional_usd": self.notional_usd,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "spot_qty": self.spot_qty,
            "perp_qty": self.perp_qty,
            "total_funding_collected": self.total_funding_collected,
            "num_payments": len(self.funding_payments),
        }


class PositionTracker:
    """
    Tracks paper trading position state and P&L.

    Features:
    - Initialize position at current price
    - Record funding payments
    - Calculate daily and all-time P&L
    - Persist state to disk
    """

    STATE_FILE = "logs/paper_trading_state.json"

    def __init__(self, notional_usd: float = 10000.0):
        self.notional_usd = notional_usd
        self.position: Optional[PaperPosition] = None
        self._state_path = Path(self.STATE_FILE)

        # Try to load existing state
        self._load_state()

    def initialize_position(self, current_price: float) -> PaperPosition:
        """
        Initialize a new paper trading position.

        Args:
            current_price: Current BTC price

        Returns:
            New PaperPosition
        """
        now = datetime.now(timezone.utc)

        # Calculate BTC quantity from notional
        btc_qty = self.notional_usd / current_price

        self.position = PaperPosition(
            notional_usd=self.notional_usd,
            entry_price=current_price,
            entry_time=now,
            spot_qty=btc_qty,
            perp_qty=btc_qty,  # Short the same amount
            daily_start_time=now,
            daily_funding_start=0.0,
        )

        logger.info(
            "Position initialized: $%.2f notional @ $%.2f = %.6f BTC",
            self.notional_usd,
            current_price,
            btc_qty,
        )

        self._save_state()
        return self.position

    def record_funding_payment(
        self,
        rate: float,
        current_price: float,
    ) -> FundingPayment:
        """
        Record a funding payment.

        Args:
            rate: Funding rate as decimal (e.g., 0.0001 = 0.01%)
            current_price: Current BTC price

        Returns:
            FundingPayment record
        """
        if not self.position or not self.position.is_active():
            raise ValueError("No active position to record funding for")

        # Calculate payment: position_size * rate
        # For short perp: positive rate = we receive, negative = we pay
        payment_usd = self.notional_usd * rate

        # Update cumulative
        self.position.total_funding_collected += payment_usd

        # Create payment record
        payment = FundingPayment(
            timestamp=datetime.now(timezone.utc),
            rate=rate,
            payment_usd=payment_usd,
            btc_price=current_price,
            cumulative_usd=self.position.total_funding_collected,
        )

        self.position.funding_payments.append(payment)

        logger.info(
            "Funding payment: %s rate = $%.2f | Cumulative: $%.2f",
            payment.rate_pct,
            payment_usd,
            self.position.total_funding_collected,
        )

        self._save_state()
        return payment

    def calculate_unrealized_pnl(self, current_price: float) -> Dict[str, float]:
        """
        Calculate unrealized P&L on spot and perp legs.

        For delta-neutral position, these should mostly cancel out.

        Args:
            current_price: Current BTC price

        Returns:
            Dict with spot_pnl, perp_pnl, net_pnl
        """
        if not self.position or not self.position.is_active():
            return {"spot_pnl": 0, "perp_pnl": 0, "net_pnl": 0}

        entry = self.position.entry_price
        qty = self.position.spot_qty

        # Spot P&L: long position
        spot_pnl = (current_price - entry) * qty

        # Perp P&L: short position (inverted)
        perp_pnl = (entry - current_price) * qty

        # Should net to ~0 (minor differences from fees/slippage)
        net_pnl = spot_pnl + perp_pnl

        return {
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "net_pnl": net_pnl,
        }

    def get_daily_summary(self, current_price: float) -> Dict:
        """
        Get P&L summary for today.

        Returns:
            Dict with daily metrics
        """
        if not self.position or not self.position.is_active():
            return {"error": "No active position"}

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Filter today's payments
        today_payments = [
            p for p in self.position.funding_payments
            if p.timestamp >= today_start
        ]

        daily_funding = sum(p.payment_usd for p in today_payments)
        unrealized = self.calculate_unrealized_pnl(current_price)

        return {
            "date": now.strftime("%Y-%m-%d"),
            "funding_payments": len(today_payments),
            "funding_received": daily_funding,
            "unrealized_pnl": unrealized["net_pnl"],
            "total_today": daily_funding + unrealized["net_pnl"],
        }

    def get_all_time_summary(self, current_price: float) -> Dict:
        """
        Get all-time P&L summary since position opened.

        Returns:
            Dict with all-time metrics
        """
        if not self.position or not self.position.is_active():
            return {"error": "No active position"}

        now = datetime.now(timezone.utc)
        duration = now - self.position.entry_time
        days = duration.total_seconds() / 86400

        total_funding = self.position.total_funding_collected
        unrealized = self.calculate_unrealized_pnl(current_price)
        total_pnl = total_funding + unrealized["net_pnl"]

        # Calculate rates
        daily_avg = total_funding / days if days > 0 else 0
        annualized_pct = (total_funding / self.notional_usd) * (365 / days) * 100 if days > 0 else 0

        # Win rate
        positive_payments = sum(
            1 for p in self.position.funding_payments if p.payment_usd > 0
        )
        total_payments = len(self.position.funding_payments)
        win_rate = (positive_payments / total_payments * 100) if total_payments > 0 else 0

        # Expected from backtest (~0.0115% per 8h = $1.15 per payment on $10K)
        expected_per_payment = self.notional_usd * 0.000115
        expected_total = expected_per_payment * total_payments
        variance_pct = ((total_funding - expected_total) / expected_total * 100) if expected_total > 0 else 0

        return {
            "start_time": self.position.entry_time.isoformat(),
            "days_running": days,
            "total_payments": total_payments,
            "positive_payments": positive_payments,
            "win_rate_pct": win_rate,
            "total_funding": total_funding,
            "unrealized_pnl": unrealized["net_pnl"],
            "total_pnl": total_pnl,
            "daily_avg": daily_avg,
            "annualized_pct": annualized_pct,
            "expected_total": expected_total,
            "variance_pct": variance_pct,
        }

    def get_recent_payments(self, limit: int = 10) -> List[FundingPayment]:
        """Get most recent funding payments."""
        if not self.position:
            return []
        return self.position.funding_payments[-limit:]

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking at start of new day."""
        if self.position:
            self.position.daily_start_time = datetime.now(timezone.utc)
            self.position.daily_funding_start = self.position.total_funding_collected
            self._save_state()

    def _save_state(self) -> None:
        """Save current state to disk."""
        if not self.position:
            return

        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "position": self.position.to_dict(),
            "payments": [p.to_dict() for p in self.position.funding_payments[-1000:]],  # Keep last 1000
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self._state_path.write_text(json.dumps(state, indent=2))
            logger.debug("State saved to %s", self._state_path)
        except IOError as e:
            logger.error("Failed to save state: %s", e)

    def _load_state(self) -> bool:
        """Load state from disk if exists."""
        if not self._state_path.exists():
            logger.info("No existing state file found")
            return False

        try:
            state = json.loads(self._state_path.read_text())
            pos_data = state["position"]

            # Reconstruct position
            self.position = PaperPosition(
                notional_usd=pos_data["notional_usd"],
                entry_price=pos_data["entry_price"],
                entry_time=datetime.fromisoformat(pos_data["entry_time"]) if pos_data["entry_time"] else None,
                spot_qty=pos_data["spot_qty"],
                perp_qty=pos_data["perp_qty"],
                total_funding_collected=pos_data["total_funding_collected"],
            )

            # Reconstruct payments
            for p_data in state.get("payments", []):
                payment = FundingPayment(
                    timestamp=datetime.fromisoformat(p_data["timestamp"]),
                    rate=p_data["rate"],
                    payment_usd=p_data["payment_usd"],
                    btc_price=p_data["btc_price"],
                    cumulative_usd=p_data.get("cumulative_usd", 0),
                )
                self.position.funding_payments.append(payment)

            logger.info(
                "Loaded existing state: %d payments, $%.2f funding collected",
                len(self.position.funding_payments),
                self.position.total_funding_collected,
            )
            return True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to load state: %s", e)
            return False

    def clear_state(self) -> None:
        """Clear saved state and reset position."""
        self.position = None
        if self._state_path.exists():
            self._state_path.unlink()
            logger.info("State cleared")

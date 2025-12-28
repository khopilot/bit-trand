"""
Position Manager for Funding Arbitrage

Manages synchronized spot and perpetual futures positions
to maintain delta-neutral exposure.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from .exchange_client import (
    BinanceClient,
    ExchangeClient,
    OrderResult,
    OrderSide,
    OrderType,
)
from .rate_monitor import FundingRateMonitor, FundingSignal

logger = logging.getLogger("btc_trader.funding_arb.position_manager")


class PositionStatus(Enum):
    """Status of the arbitrage position."""

    NONE = "none"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class ArbPosition:
    """
    Represents a funding arbitrage position.

    Consists of:
    - Long spot position (buy and hold BTC)
    - Short perpetual position (sell perp, same size)
    """

    status: PositionStatus = PositionStatus.NONE

    # Spot leg
    spot_quantity: float = 0.0
    spot_entry_price: float = 0.0
    spot_order_id: str = ""

    # Perpetual leg
    perp_quantity: float = 0.0
    perp_entry_price: float = 0.0
    perp_order_id: str = ""

    # Position metadata
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_funding_rate: float = 0.0

    # Funding payments collected
    funding_payments: List[Dict] = field(default_factory=list)
    total_funding_collected: float = 0.0

    # P&L tracking
    spot_pnl: float = 0.0
    perp_pnl: float = 0.0
    total_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        """Check if position is currently open."""
        return self.status == PositionStatus.OPEN

    @property
    def is_balanced(self) -> bool:
        """Check if spot and perp quantities match."""
        if self.spot_quantity == 0 and self.perp_quantity == 0:
            return True
        return abs(self.spot_quantity - self.perp_quantity) / max(self.spot_quantity, self.perp_quantity) < 0.01

    @property
    def notional_value(self) -> float:
        """Total notional value of the position."""
        avg_price = (self.spot_entry_price + self.perp_entry_price) / 2
        return self.spot_quantity * avg_price

    @property
    def delta(self) -> float:
        """Net BTC exposure (should be ~0 when balanced)."""
        return self.spot_quantity - self.perp_quantity

    def to_dict(self) -> Dict:
        """Convert position to dictionary for serialization."""
        return {
            "status": self.status.value,
            "spot_quantity": self.spot_quantity,
            "spot_entry_price": self.spot_entry_price,
            "perp_quantity": self.perp_quantity,
            "perp_entry_price": self.perp_entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_funding_rate": self.entry_funding_rate,
            "total_funding_collected": self.total_funding_collected,
            "funding_payment_count": len(self.funding_payments),
            "total_pnl": self.total_pnl,
            "is_balanced": self.is_balanced,
            "delta": self.delta,
        }


class PositionManager:
    """
    Manages funding arbitrage positions.

    Responsible for:
    - Opening synchronized spot/perp positions
    - Closing positions when funding turns negative
    - Tracking funding payments
    - Calculating P&L
    """

    def __init__(
        self,
        client: ExchangeClient,
        rate_monitor: FundingRateMonitor,
        symbol: str = "BTCUSDT",
        spot_symbol: str = "BTCUSDT",
        perp_symbol: str = "BTCUSDT",
        max_position_usd: float = 10000.0,
        leverage: int = 2,
        min_funding_rate: float = 0.0005,
    ):
        """
        Initialize position manager.

        Args:
            client: Exchange client for trading
            rate_monitor: Funding rate monitor
            symbol: Base trading symbol
            spot_symbol: Spot market symbol
            perp_symbol: Perpetual futures symbol
            max_position_usd: Maximum position size in USD
            leverage: Leverage for perpetual position
            min_funding_rate: Minimum funding rate to enter (0.0005 = 0.05%)
        """
        self.client = client
        self.rate_monitor = rate_monitor
        self.symbol = symbol
        self.spot_symbol = spot_symbol
        self.perp_symbol = perp_symbol
        self.max_position_usd = max_position_usd
        self.leverage = leverage
        self.min_funding_rate = min_funding_rate

        # Current position
        self.position = ArbPosition()

        # Historical positions for tracking
        self._position_history: List[ArbPosition] = []

        logger.info(
            "PositionManager initialized: max=$%.2f, leverage=%dx, min_rate=%.4f%%",
            max_position_usd,
            leverage,
            min_funding_rate * 100,
        )

    def should_enter(self) -> tuple[bool, str]:
        """
        Check if conditions are right to enter a position.

        Returns:
            Tuple of (should_enter, reason)
        """
        # Already have a position
        if self.position.is_open:
            return False, "Position already open"

        # Get current funding rate
        signal = self.rate_monitor.generate_signal(has_position=False)

        if signal.action == "enter":
            return True, signal.reason

        return False, signal.reason

    def should_exit(self) -> tuple[bool, str]:
        """
        Check if we should exit the current position.

        Returns:
            Tuple of (should_exit, reason)
        """
        if not self.position.is_open:
            return False, "No position to exit"

        # Get current funding rate
        signal = self.rate_monitor.generate_signal(has_position=True)

        if signal.action == "exit":
            return True, signal.reason

        return False, signal.reason

    def calculate_position_size(self, current_price: float) -> float:
        """
        Calculate position size based on available capital.

        Args:
            current_price: Current BTC price

        Returns:
            Position size in BTC
        """
        # Use max_position_usd to determine size
        btc_size = self.max_position_usd / current_price

        # Round to reasonable precision
        btc_size = round(btc_size, 4)

        logger.debug(
            "Position size: $%.2f / $%.2f = %.4f BTC",
            self.max_position_usd,
            current_price,
            btc_size,
        )

        return btc_size

    def open_position(self, size_btc: Optional[float] = None) -> bool:
        """
        Open a delta-neutral arbitrage position.

        Args:
            size_btc: Position size in BTC (calculates if None)

        Returns:
            True if successful, False otherwise
        """
        if self.position.is_open:
            logger.warning("Cannot open position: already have open position")
            return False

        try:
            self.position.status = PositionStatus.OPENING

            # Get current prices
            spot_price = self.client.get_spot_price(self.spot_symbol)
            if spot_price is None:
                logger.error("Failed to get spot price")
                self.position.status = PositionStatus.ERROR
                return False

            # Calculate size if not provided
            if size_btc is None:
                size_btc = self.calculate_position_size(spot_price)

            # Set leverage on futures account
            if isinstance(self.client, BinanceClient):
                self.client.set_futures_leverage(self.perp_symbol, self.leverage)

            # Get current funding rate for tracking
            funding_rate = self.rate_monitor.get_binance_funding_rate()
            entry_rate = funding_rate.rate if funding_rate else 0.0

            # Step 1: Buy spot BTC
            logger.info("Opening spot LONG: %.4f BTC at ~$%.2f", size_btc, spot_price)
            spot_order = self.client.place_spot_order(
                symbol=self.spot_symbol,
                side=OrderSide.BUY,
                quantity=size_btc,
                order_type=OrderType.MARKET,
            )

            if spot_order is None or spot_order.status not in ("FILLED", "NEW"):
                logger.error("Failed to open spot position")
                self.position.status = PositionStatus.ERROR
                return False

            self.position.spot_quantity = spot_order.filled_qty
            self.position.spot_entry_price = spot_order.price
            self.position.spot_order_id = spot_order.order_id

            # Step 2: Short perpetual futures (same size)
            perp_price = self.client.get_futures_price(self.perp_symbol)
            logger.info("Opening perp SHORT: %.4f BTC at ~$%.2f", size_btc, perp_price)

            perp_order = self.client.place_futures_order(
                symbol=self.perp_symbol,
                side=OrderSide.SELL,
                quantity=size_btc,
                order_type=OrderType.MARKET,
            )

            if perp_order is None or perp_order.status not in ("FILLED", "NEW"):
                logger.error("Failed to open perp position - need to close spot!")
                # TODO: Implement rollback - close spot position
                self.position.status = PositionStatus.ERROR
                return False

            self.position.perp_quantity = perp_order.filled_qty
            self.position.perp_entry_price = perp_order.price
            self.position.perp_order_id = perp_order.order_id

            # Mark position as open
            self.position.status = PositionStatus.OPEN
            self.position.entry_time = datetime.now(timezone.utc)
            self.position.entry_funding_rate = entry_rate

            logger.info(
                "Position opened successfully: %.4f BTC, spot=$%.2f, perp=$%.2f, rate=%.4f%%",
                self.position.spot_quantity,
                self.position.spot_entry_price,
                self.position.perp_entry_price,
                entry_rate * 100,
            )

            return True

        except Exception as e:
            logger.error("Error opening position: %s", e)
            self.position.status = PositionStatus.ERROR
            return False

    def close_position(self) -> bool:
        """
        Close the current arbitrage position.

        Returns:
            True if successful, False otherwise
        """
        if not self.position.is_open:
            logger.warning("No position to close")
            return False

        try:
            self.position.status = PositionStatus.CLOSING

            # Get current prices for P&L calculation
            spot_price = self.client.get_spot_price(self.spot_symbol)
            perp_price = self.client.get_futures_price(self.perp_symbol)

            # Step 1: Sell spot BTC
            logger.info(
                "Closing spot LONG: %.4f BTC at ~$%.2f",
                self.position.spot_quantity,
                spot_price,
            )
            spot_order = self.client.place_spot_order(
                symbol=self.spot_symbol,
                side=OrderSide.SELL,
                quantity=self.position.spot_quantity,
                order_type=OrderType.MARKET,
            )

            if spot_order is None:
                logger.error("Failed to close spot position")
                self.position.status = PositionStatus.ERROR
                return False

            # Step 2: Close perp short (buy to close)
            logger.info(
                "Closing perp SHORT: %.4f BTC at ~$%.2f",
                self.position.perp_quantity,
                perp_price,
            )
            perp_order = self.client.place_futures_order(
                symbol=self.perp_symbol,
                side=OrderSide.BUY,
                quantity=self.position.perp_quantity,
                order_type=OrderType.MARKET,
                reduce_only=True,
            )

            if perp_order is None:
                logger.error("Failed to close perp position")
                self.position.status = PositionStatus.ERROR
                return False

            # Calculate P&L
            self._calculate_pnl(spot_order.price, perp_order.price)

            # Mark position as closed
            self.position.status = PositionStatus.CLOSED
            self.position.exit_time = datetime.now(timezone.utc)

            logger.info(
                "Position closed: spot_pnl=$%.2f, perp_pnl=$%.2f, funding=$%.2f, total=$%.2f",
                self.position.spot_pnl,
                self.position.perp_pnl,
                self.position.total_funding_collected,
                self.position.total_pnl,
            )

            # Archive position and reset
            self._position_history.append(self.position)
            self.position = ArbPosition()

            return True

        except Exception as e:
            logger.error("Error closing position: %s", e)
            self.position.status = PositionStatus.ERROR
            return False

    def _calculate_pnl(self, spot_exit_price: float, perp_exit_price: float) -> None:
        """Calculate P&L for the current position."""
        # Spot P&L: (exit - entry) * quantity
        self.position.spot_pnl = (
            spot_exit_price - self.position.spot_entry_price
        ) * self.position.spot_quantity

        # Perp P&L: (entry - exit) * quantity (short position)
        self.position.perp_pnl = (
            self.position.perp_entry_price - perp_exit_price
        ) * self.position.perp_quantity

        # Total P&L = spot + perp + funding collected
        self.position.total_pnl = (
            self.position.spot_pnl
            + self.position.perp_pnl
            + self.position.total_funding_collected
        )

    def record_funding_payment(self, amount: float, rate: float) -> None:
        """
        Record a funding payment received.

        Args:
            amount: Payment amount in USD
            rate: Funding rate at the time
        """
        if not self.position.is_open:
            return

        payment = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "amount": amount,
            "rate": rate,
            "position_size": self.position.perp_quantity,
        }

        self.position.funding_payments.append(payment)
        self.position.total_funding_collected += amount

        logger.info(
            "Funding payment recorded: $%.4f at %.4f%% rate, total=$%.4f",
            amount,
            rate * 100,
            self.position.total_funding_collected,
        )

    def get_position_summary(self) -> Dict:
        """Get a summary of the current position."""
        if not self.position.is_open:
            return {
                "status": "no_position",
                "message": "No active position",
            }

        # Get current prices for unrealized P&L
        spot_price = self.client.get_spot_price(self.spot_symbol)
        perp_price = self.client.get_futures_price(self.perp_symbol)

        if spot_price and perp_price:
            unrealized_spot = (spot_price - self.position.spot_entry_price) * self.position.spot_quantity
            unrealized_perp = (self.position.perp_entry_price - perp_price) * self.position.perp_quantity
            unrealized_total = unrealized_spot + unrealized_perp
        else:
            unrealized_spot = 0
            unrealized_perp = 0
            unrealized_total = 0

        # Calculate duration
        if self.position.entry_time:
            duration = datetime.now(timezone.utc) - self.position.entry_time
            hours = duration.total_seconds() / 3600
        else:
            hours = 0

        return {
            "status": self.position.status.value,
            "spot_quantity": self.position.spot_quantity,
            "perp_quantity": self.position.perp_quantity,
            "spot_entry": self.position.spot_entry_price,
            "perp_entry": self.position.perp_entry_price,
            "current_spot": spot_price,
            "current_perp": perp_price,
            "unrealized_spot_pnl": unrealized_spot,
            "unrealized_perp_pnl": unrealized_perp,
            "unrealized_total": unrealized_total,
            "funding_collected": self.position.total_funding_collected,
            "funding_payments": len(self.position.funding_payments),
            "total_pnl": unrealized_total + self.position.total_funding_collected,
            "duration_hours": hours,
            "is_balanced": self.position.is_balanced,
            "delta": self.position.delta,
        }

    def get_historical_performance(self) -> Dict:
        """Get performance summary of all closed positions."""
        if not self._position_history:
            return {
                "total_positions": 0,
                "message": "No historical positions",
            }

        total_pnl = sum(p.total_pnl for p in self._position_history)
        total_funding = sum(p.total_funding_collected for p in self._position_history)
        total_spot_pnl = sum(p.spot_pnl for p in self._position_history)
        total_perp_pnl = sum(p.perp_pnl for p in self._position_history)

        return {
            "total_positions": len(self._position_history),
            "total_pnl": total_pnl,
            "total_funding": total_funding,
            "total_spot_pnl": total_spot_pnl,
            "total_perp_pnl": total_perp_pnl,
            "avg_pnl_per_position": total_pnl / len(self._position_history),
            "avg_funding_per_position": total_funding / len(self._position_history),
        }

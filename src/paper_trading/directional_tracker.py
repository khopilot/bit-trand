"""
Directional Position Tracker for Paper Trading

Tracks simulated directional trading positions (LONG/SHORT/FLAT)
based on EMA/RSI/Bollinger signals.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("btc_trader.paper_trading.directional")


class PositionSide(Enum):
    """Position direction."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class DirectionalTrade:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity_btc: float
    notional_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit"

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity_btc": self.quantity_btc,
            "notional_usd": self.notional_usd,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
        }


@dataclass
class DirectionalPosition:
    """
    Simulated directional trading position.

    Unlike arb which is always-in and neutral,
    this can be LONG, SHORT, or FLAT based on signals.
    """
    notional_usd: float = 10000.0

    # Current position state
    side: PositionSide = PositionSide.FLAT
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    quantity_btc: float = 0.0

    # Tracking
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float('inf')  # For short trailing stop

    # Trade history
    trades: List[DirectionalTrade] = field(default_factory=list)
    total_pnl: float = 0.0

    # Daily tracking
    daily_start_pnl: float = 0.0
    daily_start_time: Optional[datetime] = None

    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT

    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT

    def to_dict(self) -> dict:
        return {
            "notional_usd": self.notional_usd,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "quantity_btc": self.quantity_btc,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price if self.lowest_price != float('inf') else 0,
            "total_pnl": self.total_pnl,
            "num_trades": len(self.trades),
        }


class DirectionalTracker:
    """
    Tracks directional paper trading positions.

    Features:
    - Open LONG/SHORT positions based on signals
    - Close positions with P&L tracking
    - Track trade history and statistics
    - Persist state to disk
    """

    STATE_FILE = "logs/directional_state.json"

    def __init__(self, notional_usd: float = 10000.0):
        self.notional_usd = notional_usd
        self.position: Optional[DirectionalPosition] = None
        self._state_path = Path(self.STATE_FILE)

        # Initialize or load
        self._load_state()
        if not self.position:
            self.position = DirectionalPosition(notional_usd=notional_usd)

    def open_long(self, price: float, reason: str = "signal") -> bool:
        """
        Open a LONG position.

        Args:
            price: Entry price
            reason: Why we're entering

        Returns:
            True if position opened
        """
        if not self.position.is_flat():
            logger.warning("Cannot open LONG - already have position: %s", self.position.side)
            return False

        now = datetime.now(timezone.utc)
        quantity = self.notional_usd / price

        self.position.side = PositionSide.LONG
        self.position.entry_price = price
        self.position.entry_time = now
        self.position.quantity_btc = quantity
        self.position.highest_price = price
        self.position.lowest_price = price

        logger.info(
            "LONG opened: %.6f BTC @ $%.2f ($%.2f notional) - %s",
            quantity, price, self.notional_usd, reason
        )

        self._save_state()
        return True

    def open_short(self, price: float, reason: str = "signal") -> bool:
        """
        Open a SHORT position.

        Args:
            price: Entry price
            reason: Why we're entering

        Returns:
            True if position opened
        """
        if not self.position.is_flat():
            logger.warning("Cannot open SHORT - already have position: %s", self.position.side)
            return False

        now = datetime.now(timezone.utc)
        quantity = self.notional_usd / price

        self.position.side = PositionSide.SHORT
        self.position.entry_price = price
        self.position.entry_time = now
        self.position.quantity_btc = quantity
        self.position.highest_price = price
        self.position.lowest_price = price

        logger.info(
            "SHORT opened: %.6f BTC @ $%.2f ($%.2f notional) - %s",
            quantity, price, self.notional_usd, reason
        )

        self._save_state()
        return True

    def close_position(self, price: float, reason: str = "signal") -> Optional[DirectionalTrade]:
        """
        Close current position.

        Args:
            price: Exit price
            reason: Why we're exiting (signal, stop_loss, take_profit)

        Returns:
            Trade record if closed, None if no position
        """
        if self.position.is_flat():
            logger.warning("Cannot close - no open position")
            return None

        now = datetime.now(timezone.utc)

        # Calculate P&L
        if self.position.is_long():
            pnl_usd = (price - self.position.entry_price) * self.position.quantity_btc
        else:  # SHORT
            pnl_usd = (self.position.entry_price - price) * self.position.quantity_btc

        pnl_pct = (pnl_usd / self.notional_usd) * 100

        # Create trade record
        trade = DirectionalTrade(
            entry_time=self.position.entry_time,
            exit_time=now,
            side=self.position.side.value,
            entry_price=self.position.entry_price,
            exit_price=price,
            quantity_btc=self.position.quantity_btc,
            notional_usd=self.notional_usd,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )

        # Update totals
        self.position.trades.append(trade)
        self.position.total_pnl += pnl_usd

        logger.info(
            "%s closed @ $%.2f | P&L: $%.2f (%.2f%%) | Reason: %s",
            self.position.side.value, price, pnl_usd, pnl_pct, reason
        )

        # Reset position to FLAT
        self.position.side = PositionSide.FLAT
        self.position.entry_price = 0.0
        self.position.entry_time = None
        self.position.quantity_btc = 0.0
        self.position.highest_price = 0.0
        self.position.lowest_price = float('inf')

        self._save_state()
        return trade

    def update_price(self, price: float) -> None:
        """Update highest/lowest price for trailing stop calculation."""
        if self.position.is_long():
            self.position.highest_price = max(self.position.highest_price, price)
        elif self.position.is_short():
            self.position.lowest_price = min(self.position.lowest_price, price)

    def check_stop_loss(self, price: float, stop_pct: float = 0.05) -> bool:
        """
        Check if stop loss is hit.

        Args:
            price: Current price
            stop_pct: Stop loss percentage (default 5%)

        Returns:
            True if stop hit
        """
        if self.position.is_flat():
            return False

        if self.position.is_long():
            # Trailing stop from highest
            stop_price = self.position.highest_price * (1 - stop_pct)
            return price <= stop_price
        else:  # SHORT
            # Trailing stop from lowest
            stop_price = self.position.lowest_price * (1 + stop_pct)
            return price >= stop_price

    def get_unrealized_pnl(self, price: float) -> Dict[str, float]:
        """Get unrealized P&L for current position."""
        if self.position.is_flat():
            return {"pnl_usd": 0, "pnl_pct": 0}

        if self.position.is_long():
            pnl_usd = (price - self.position.entry_price) * self.position.quantity_btc
        else:
            pnl_usd = (self.position.entry_price - price) * self.position.quantity_btc

        pnl_pct = (pnl_usd / self.notional_usd) * 100

        return {"pnl_usd": pnl_usd, "pnl_pct": pnl_pct}

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        trades = self.position.trades

        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "avg_win": 0,
                "avg_loss": 0,
            }

        winning = [t for t in trades if t.pnl_usd > 0]
        losing = [t for t in trades if t.pnl_usd <= 0]

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades) * 100 if trades else 0,
            "total_pnl": self.position.total_pnl,
            "avg_pnl": self.position.total_pnl / len(trades) if trades else 0,
            "best_trade": max(t.pnl_usd for t in trades) if trades else 0,
            "worst_trade": min(t.pnl_usd for t in trades) if trades else 0,
            "avg_win": sum(t.pnl_usd for t in winning) / len(winning) if winning else 0,
            "avg_loss": sum(t.pnl_usd for t in losing) / len(losing) if losing else 0,
        }

    def get_recent_trades(self, limit: int = 10) -> List[DirectionalTrade]:
        """Get most recent trades."""
        return self.position.trades[-limit:] if self.position else []

    def _save_state(self) -> None:
        """Save state to disk."""
        if not self.position:
            return

        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "position": self.position.to_dict(),
            "trades": [t.to_dict() for t in self.position.trades[-100:]],  # Keep last 100
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self._state_path.write_text(json.dumps(state, indent=2))
            logger.debug("State saved to %s", self._state_path)
        except IOError as e:
            logger.error("Failed to save state: %s", e)

    def _load_state(self) -> bool:
        """Load state from disk."""
        if not self._state_path.exists():
            logger.info("No existing directional state file")
            return False

        try:
            state = json.loads(self._state_path.read_text())
            pos_data = state["position"]

            self.position = DirectionalPosition(
                notional_usd=pos_data["notional_usd"],
                side=PositionSide(pos_data["side"]),
                entry_price=pos_data["entry_price"],
                entry_time=datetime.fromisoformat(pos_data["entry_time"]) if pos_data["entry_time"] else None,
                quantity_btc=pos_data["quantity_btc"],
                highest_price=pos_data.get("highest_price", 0),
                lowest_price=pos_data.get("lowest_price", float('inf')) or float('inf'),
                total_pnl=pos_data["total_pnl"],
            )

            # Reconstruct trades
            for t_data in state.get("trades", []):
                trade = DirectionalTrade(
                    entry_time=datetime.fromisoformat(t_data["entry_time"]),
                    exit_time=datetime.fromisoformat(t_data["exit_time"]),
                    side=t_data["side"],
                    entry_price=t_data["entry_price"],
                    exit_price=t_data["exit_price"],
                    quantity_btc=t_data["quantity_btc"],
                    notional_usd=t_data["notional_usd"],
                    pnl_usd=t_data["pnl_usd"],
                    pnl_pct=t_data["pnl_pct"],
                    exit_reason=t_data["exit_reason"],
                )
                self.position.trades.append(trade)

            logger.info(
                "Loaded directional state: %s position, %d trades, $%.2f total P&L",
                self.position.side.value,
                len(self.position.trades),
                self.position.total_pnl,
            )
            return True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to load directional state: %s", e)
            return False

    def clear_state(self) -> None:
        """Clear state and reset."""
        self.position = DirectionalPosition(notional_usd=self.notional_usd)
        if self._state_path.exists():
            self._state_path.unlink()
            logger.info("Directional state cleared")

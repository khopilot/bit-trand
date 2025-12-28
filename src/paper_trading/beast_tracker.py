"""
Beast Tracker - Hybrid Strategy State Management

THE BEAST combines arb (constant spot) with directional (variable perp hedge).

Modes:
- FULL_LONG:   Spot=100%, Perp=0%   (Net: +100% exposure)
- HALF_LONG:   Spot=100%, Perp=50%  (Net: +50% exposure)
- NEUTRAL:     Spot=100%, Perp=100% (Net: 0% - pure arb)
- HALF_SHORT:  Spot=100%, Perp=150% (Net: -50% exposure)
- FULL_SHORT:  Spot=100%, Perp=200% (Net: -100% exposure)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("btc_trader.paper_trading.beast")


class BeastMode(Enum):
    """Beast operating modes with hedge ratios."""
    FULL_LONG = "FULL_LONG"      # Perp = 0% of spot
    HALF_LONG = "HALF_LONG"      # Perp = 50% of spot
    NEUTRAL = "NEUTRAL"          # Perp = 100% of spot (pure arb)
    HALF_SHORT = "HALF_SHORT"    # Perp = 150% of spot
    FULL_SHORT = "FULL_SHORT"    # Perp = 200% of spot


# Hedge ratios for each mode
HEDGE_RATIOS = {
    BeastMode.FULL_LONG: 0.0,
    BeastMode.HALF_LONG: 0.5,
    BeastMode.NEUTRAL: 1.0,
    BeastMode.HALF_SHORT: 1.5,
    BeastMode.FULL_SHORT: 2.0,
}


@dataclass
class ModeChange:
    """Record of a mode transition."""
    timestamp: datetime
    from_mode: str
    to_mode: str
    reason: str
    price: float
    indicators: Dict

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_mode": self.from_mode,
            "to_mode": self.to_mode,
            "reason": self.reason,
            "price": self.price,
            "indicators": self.indicators,
        }


@dataclass
class FundingPayment:
    """Record of a funding payment."""
    timestamp: datetime
    rate: float
    perp_notional: float
    payment_usd: float  # Positive = we received, negative = we paid


@dataclass
class BeastPosition:
    """
    Beast hybrid position state.

    Always maintains spot position.
    Dynamically adjusts perp based on mode.
    """
    notional_usd: float = 10000.0

    # Current state
    mode: BeastMode = BeastMode.NEUTRAL
    mode_reason: str = "Initial"

    # Spot position (constant)
    spot_btc: float = 0.0
    spot_entry_price: float = 0.0
    spot_entry_time: Optional[datetime] = None

    # Perp position (variable based on mode)
    perp_btc: float = 0.0  # Negative = short
    perp_notional: float = 0.0

    # P&L tracking
    funding_collected: float = 0.0
    directional_pnl: float = 0.0
    total_pnl: float = 0.0

    # Funding history
    funding_payments: int = 0

    # Mode tracking
    mode_changes: List[ModeChange] = field(default_factory=list)
    time_in_mode: Dict[str, float] = field(default_factory=dict)

    # Risk tracking
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0

    # For trailing stop on directional component
    mode_entry_price: float = 0.0
    highest_since_mode: float = 0.0
    lowest_since_mode: float = float('inf')

    def get_net_exposure(self) -> float:
        """Get net BTC exposure (spot + perp)."""
        return self.spot_btc + self.perp_btc

    def get_net_exposure_pct(self) -> float:
        """Get net exposure as % of notional."""
        return HEDGE_RATIOS[self.mode] * 100 - 100

    def get_hedge_ratio(self) -> float:
        """Get current hedge ratio."""
        return HEDGE_RATIOS[self.mode]

    def to_dict(self) -> dict:
        return {
            "notional_usd": self.notional_usd,
            "mode": self.mode.value,
            "mode_reason": self.mode_reason,
            "spot_btc": self.spot_btc,
            "spot_entry_price": self.spot_entry_price,
            "spot_entry_time": self.spot_entry_time.isoformat() if self.spot_entry_time else None,
            "perp_btc": self.perp_btc,
            "perp_notional": self.perp_notional,
            "funding_collected": self.funding_collected,
            "directional_pnl": self.directional_pnl,
            "total_pnl": self.total_pnl,
            "funding_payments": self.funding_payments,
            "mode_changes_count": len(self.mode_changes),
            "time_in_mode": self.time_in_mode,
            "max_drawdown": self.max_drawdown,
            "peak_pnl": self.peak_pnl,
        }


class BeastTracker:
    """
    Tracks the hybrid Beast position.

    Features:
    - Constant spot position
    - Dynamic perp hedge based on signals
    - Separate funding and directional P&L
    - Mode history and statistics
    """

    STATE_FILE = "logs/beast_state.json"

    def __init__(self, notional_usd: float = 10000.0):
        self.notional_usd = notional_usd
        self.position: Optional[BeastPosition] = None
        self._state_path = Path(self.STATE_FILE)
        self._mode_start_time: Optional[datetime] = None

        # Load or initialize
        self._load_state()
        if not self.position:
            self.position = BeastPosition(notional_usd=notional_usd)

    def initialize_position(self, price: float) -> None:
        """
        Initialize the beast position.

        Opens spot LONG and perp SHORT at 100% hedge (NEUTRAL mode).
        """
        now = datetime.now(timezone.utc)

        # Calculate spot position
        spot_btc = self.notional_usd / price

        # Initialize in NEUTRAL mode (100% hedged)
        perp_btc = -spot_btc  # Short perp
        perp_notional = self.notional_usd

        self.position = BeastPosition(
            notional_usd=self.notional_usd,
            mode=BeastMode.NEUTRAL,
            mode_reason="Initial entry",
            spot_btc=spot_btc,
            spot_entry_price=price,
            spot_entry_time=now,
            perp_btc=perp_btc,
            perp_notional=perp_notional,
            mode_entry_price=price,
            highest_since_mode=price,
            lowest_since_mode=price,
        )

        self._mode_start_time = now
        self.position.time_in_mode[BeastMode.NEUTRAL.value] = 0.0

        logger.info(
            "Beast initialized: Spot=+%.6f BTC @ $%.2f, Perp=-%.6f BTC (NEUTRAL)",
            spot_btc, price, abs(perp_btc)
        )

        self._save_state()

    def change_mode(
        self,
        new_mode: BeastMode,
        price: float,
        reason: str,
        indicators: Dict = None,
    ) -> bool:
        """
        Change the beast's mode.

        Adjusts perp position to match new hedge ratio.

        Returns:
            True if mode changed
        """
        if not self.position or not self.position.spot_btc:
            logger.warning("Cannot change mode - position not initialized")
            return False

        old_mode = self.position.mode
        if old_mode == new_mode:
            logger.debug("Mode unchanged: %s", new_mode.value)
            return False

        now = datetime.now(timezone.utc)

        # Update time in previous mode
        if self._mode_start_time:
            duration = (now - self._mode_start_time).total_seconds()
            old_mode_time = self.position.time_in_mode.get(old_mode.value, 0)
            self.position.time_in_mode[old_mode.value] = old_mode_time + duration

        # Calculate new perp position
        new_hedge_ratio = HEDGE_RATIOS[new_mode]
        new_perp_btc = -self.position.spot_btc * new_hedge_ratio
        new_perp_notional = abs(new_perp_btc * price)

        # Calculate directional P&L from mode change
        old_perp_value = abs(self.position.perp_btc * price)
        directional_pnl = self._calculate_mode_change_pnl(price)

        # Record mode change
        change = ModeChange(
            timestamp=now,
            from_mode=old_mode.value,
            to_mode=new_mode.value,
            reason=reason,
            price=price,
            indicators=indicators or {},
        )
        self.position.mode_changes.append(change)

        # Update position
        self.position.mode = new_mode
        self.position.mode_reason = reason
        self.position.perp_btc = new_perp_btc
        self.position.perp_notional = new_perp_notional
        self.position.directional_pnl += directional_pnl
        self.position.total_pnl = self.position.funding_collected + self.position.directional_pnl

        # Reset mode tracking
        self.position.mode_entry_price = price
        self.position.highest_since_mode = price
        self.position.lowest_since_mode = price
        self._mode_start_time = now

        # Update peak/drawdown
        self._update_drawdown()

        logger.info(
            "Mode changed: %s -> %s @ $%.2f | Perp: %.6f -> %.6f BTC | Reason: %s",
            old_mode.value, new_mode.value, price,
            -self.position.spot_btc * HEDGE_RATIOS[old_mode],
            new_perp_btc, reason
        )

        self._save_state()
        return True

    def _calculate_mode_change_pnl(self, current_price: float) -> float:
        """Calculate P&L from mode change based on net exposure."""
        if not self.position:
            return 0.0

        # Net exposure = spot + perp (perp is negative when short)
        net_btc = self.position.spot_btc + self.position.perp_btc

        # P&L = net exposure * price change since mode entry
        price_change = current_price - self.position.mode_entry_price
        pnl = net_btc * price_change

        return pnl

    def record_funding(self, rate: float, price: float) -> float:
        """
        Record funding payment.

        Args:
            rate: Funding rate (positive = longs pay shorts)
            price: Current BTC price

        Returns:
            Payment amount (positive = received)
        """
        if not self.position or self.position.perp_btc >= 0:
            return 0.0

        # We're short perp, so we receive when rate is positive
        perp_notional = abs(self.position.perp_btc * price)
        payment = perp_notional * rate

        self.position.funding_collected += payment
        self.position.funding_payments += 1
        self.position.total_pnl = self.position.funding_collected + self.position.directional_pnl

        self._update_drawdown()

        logger.info(
            "Funding received: $%.4f (rate=%.4f%%, notional=$%.2f)",
            payment, rate * 100, perp_notional
        )

        self._save_state()
        return payment

    def update_price(self, price: float) -> None:
        """Update price tracking for trailing stop."""
        if not self.position:
            return

        self.position.highest_since_mode = max(
            self.position.highest_since_mode, price
        )
        self.position.lowest_since_mode = min(
            self.position.lowest_since_mode, price
        )

        # Update directional P&L
        self.position.directional_pnl = self._calculate_mode_change_pnl(price)
        self.position.total_pnl = self.position.funding_collected + self.position.directional_pnl

        self._update_drawdown()

    def check_directional_stop(self, price: float, stop_pct: float = 0.05) -> bool:
        """
        Check if trailing stop hit on directional component.

        Only applies when net exposure != 0 (not in NEUTRAL).
        """
        if not self.position or self.position.mode == BeastMode.NEUTRAL:
            return False

        net_btc = self.position.get_net_exposure()

        if net_btc > 0:  # Net long
            # Stop if price drops 5% from highest
            stop_price = self.position.highest_since_mode * (1 - stop_pct)
            return price <= stop_price
        else:  # Net short
            # Stop if price rises 5% from lowest
            stop_price = self.position.lowest_since_mode * (1 + stop_pct)
            return price >= stop_price

    def _update_drawdown(self) -> None:
        """Update peak P&L and max drawdown."""
        if not self.position:
            return

        if self.position.total_pnl > self.position.peak_pnl:
            self.position.peak_pnl = self.position.total_pnl

        drawdown = self.position.peak_pnl - self.position.total_pnl
        self.position.max_drawdown = max(self.position.max_drawdown, drawdown)

    def get_unrealized_pnl(self, price: float) -> Dict[str, float]:
        """Get current unrealized P&L breakdown."""
        if not self.position:
            return {"funding": 0, "directional": 0, "total": 0}

        directional = self._calculate_mode_change_pnl(price)
        total = self.position.funding_collected + directional

        return {
            "funding": self.position.funding_collected,
            "directional": directional,
            "total": total,
        }

    def get_mode_distribution(self) -> Dict[str, float]:
        """Get percentage of time spent in each mode."""
        if not self.position or not self.position.time_in_mode:
            return {}

        # Add current mode time
        now = datetime.now(timezone.utc)
        current_time = self.position.time_in_mode.copy()

        if self._mode_start_time:
            duration = (now - self._mode_start_time).total_seconds()
            current_mode = self.position.mode.value
            current_time[current_mode] = current_time.get(current_mode, 0) + duration

        total_time = sum(current_time.values())
        if total_time == 0:
            return {}

        return {
            mode: (time / total_time * 100)
            for mode, time in current_time.items()
        }

    def get_stats(self) -> Dict:
        """Get beast statistics."""
        if not self.position:
            return {}

        mode_dist = self.get_mode_distribution()

        return {
            "mode": self.position.mode.value,
            "mode_reason": self.position.mode_reason,
            "mode_changes": len(self.position.mode_changes),
            "funding_collected": self.position.funding_collected,
            "funding_payments": self.position.funding_payments,
            "directional_pnl": self.position.directional_pnl,
            "total_pnl": self.position.total_pnl,
            "max_drawdown": self.position.max_drawdown,
            "peak_pnl": self.position.peak_pnl,
            "mode_distribution": mode_dist,
        }

    def get_recent_mode_changes(self, limit: int = 10) -> List[ModeChange]:
        """Get recent mode changes."""
        if not self.position:
            return []
        return self.position.mode_changes[-limit:]

    def _save_state(self) -> None:
        """Save state to disk."""
        if not self.position:
            return

        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "position": self.position.to_dict(),
            "mode_changes": [mc.to_dict() for mc in self.position.mode_changes[-50:]],
            "mode_start_time": self._mode_start_time.isoformat() if self._mode_start_time else None,
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
            logger.info("No existing beast state file")
            return False

        try:
            state = json.loads(self._state_path.read_text())
            pos_data = state["position"]

            self.position = BeastPosition(
                notional_usd=pos_data["notional_usd"],
                mode=BeastMode(pos_data["mode"]),
                mode_reason=pos_data.get("mode_reason", "Loaded"),
                spot_btc=pos_data["spot_btc"],
                spot_entry_price=pos_data["spot_entry_price"],
                spot_entry_time=datetime.fromisoformat(pos_data["spot_entry_time"]) if pos_data.get("spot_entry_time") else None,
                perp_btc=pos_data.get("perp_btc", -pos_data["spot_btc"]),
                perp_notional=pos_data.get("perp_notional", pos_data["notional_usd"]),
                funding_collected=pos_data.get("funding_collected", 0),
                directional_pnl=pos_data.get("directional_pnl", 0),
                total_pnl=pos_data.get("total_pnl", 0),
                funding_payments=pos_data.get("funding_payments", 0),
                time_in_mode=pos_data.get("time_in_mode", {}),
                max_drawdown=pos_data.get("max_drawdown", 0),
                peak_pnl=pos_data.get("peak_pnl", 0),
            )

            # Reconstruct mode changes
            for mc_data in state.get("mode_changes", []):
                change = ModeChange(
                    timestamp=datetime.fromisoformat(mc_data["timestamp"]),
                    from_mode=mc_data["from_mode"],
                    to_mode=mc_data["to_mode"],
                    reason=mc_data["reason"],
                    price=mc_data["price"],
                    indicators=mc_data.get("indicators", {}),
                )
                self.position.mode_changes.append(change)

            # Restore mode start time
            if state.get("mode_start_time"):
                self._mode_start_time = datetime.fromisoformat(state["mode_start_time"])

            logger.info(
                "Loaded beast state: %s mode, $%.2f funding, $%.2f directional, %d changes",
                self.position.mode.value,
                self.position.funding_collected,
                self.position.directional_pnl,
                len(self.position.mode_changes),
            )
            return True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to load beast state: %s", e)
            return False

    def clear_state(self) -> None:
        """Clear state and reset."""
        self.position = BeastPosition(notional_usd=self.notional_usd)
        self._mode_start_time = None
        if self._state_path.exists():
            self._state_path.unlink()
            logger.info("Beast state cleared")

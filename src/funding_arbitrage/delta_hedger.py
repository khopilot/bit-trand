"""
Delta Hedger for Funding Arbitrage

Monitors and rebalances positions to maintain delta-neutral exposure.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .exchange_client import ExchangeClient, OrderSide, OrderType
from .position_manager import ArbPosition, PositionManager

logger = logging.getLogger("btc_trader.funding_arb.delta_hedger")


@dataclass
class HedgeAction:
    """Represents a hedging action to be taken."""

    action_type: str  # "rebalance_spot", "rebalance_perp", "none"
    side: Optional[OrderSide] = None
    quantity: float = 0.0
    reason: str = ""
    urgency: str = "low"  # "low", "medium", "high", "critical"


@dataclass
class HedgeResult:
    """Result of a hedging operation."""

    success: bool
    action_taken: str
    quantity_adjusted: float
    old_delta: float
    new_delta: float
    message: str


class DeltaHedger:
    """
    Monitors position delta and rebalances when needed.

    Ensures spot and perp positions stay matched to maintain
    market-neutral exposure.

    Reasons for imbalance:
    - Partial fills on one leg
    - Funding affecting margin
    - Price divergence causing liquidation risk
    - Manual intervention
    """

    def __init__(
        self,
        client: ExchangeClient,
        position_manager: PositionManager,
        max_delta_pct: float = 0.02,  # 2% max imbalance
        rebalance_threshold: float = 0.01,  # 1% trigger rebalance
        margin_warning_pct: float = 0.30,  # Warn at 30% margin ratio
        margin_critical_pct: float = 0.20,  # Critical at 20%
    ):
        """
        Initialize delta hedger.

        Args:
            client: Exchange client for trading
            position_manager: Position manager to monitor
            max_delta_pct: Maximum allowed position imbalance
            rebalance_threshold: Threshold to trigger rebalance
            margin_warning_pct: Margin ratio warning level
            margin_critical_pct: Margin ratio critical level
        """
        self.client = client
        self.position_manager = position_manager
        self.max_delta_pct = max_delta_pct
        self.rebalance_threshold = rebalance_threshold
        self.margin_warning_pct = margin_warning_pct
        self.margin_critical_pct = margin_critical_pct

        # Tracking
        self._hedge_history: List[Dict] = []
        self._last_check: Optional[datetime] = None

        logger.info(
            "DeltaHedger initialized: max_delta=%.1f%%, rebalance=%.1f%%",
            max_delta_pct * 100,
            rebalance_threshold * 100,
        )

    def check_delta(self) -> HedgeAction:
        """
        Check current position delta and determine if rebalancing needed.

        Returns:
            HedgeAction with recommended action
        """
        self._last_check = datetime.now(timezone.utc)
        position = self.position_manager.position

        if not position.is_open:
            return HedgeAction(
                action_type="none",
                reason="No open position",
            )

        # Calculate delta
        spot_qty = position.spot_quantity
        perp_qty = position.perp_quantity

        if spot_qty == 0 and perp_qty == 0:
            return HedgeAction(
                action_type="none",
                reason="Position quantities are zero",
            )

        # Delta as percentage of larger position
        max_qty = max(spot_qty, perp_qty)
        delta = spot_qty - perp_qty
        delta_pct = abs(delta) / max_qty if max_qty > 0 else 0

        logger.debug(
            "Delta check: spot=%.4f, perp=%.4f, delta=%.4f (%.2f%%)",
            spot_qty,
            perp_qty,
            delta,
            delta_pct * 100,
        )

        # Determine action
        if delta_pct < self.rebalance_threshold:
            return HedgeAction(
                action_type="none",
                reason=f"Delta within threshold: {delta_pct*100:.2f}%",
            )

        if delta_pct > self.max_delta_pct:
            urgency = "critical"
        elif delta_pct > self.rebalance_threshold * 1.5:
            urgency = "high"
        else:
            urgency = "medium"

        # Determine which side to adjust
        if delta > 0:
            # Long delta - either sell spot or add to perp short
            return HedgeAction(
                action_type="rebalance_perp",
                side=OrderSide.SELL,
                quantity=abs(delta),
                reason=f"Long delta: {delta:.4f} BTC ({delta_pct*100:.2f}%)",
                urgency=urgency,
            )
        else:
            # Short delta - either buy spot or reduce perp short
            return HedgeAction(
                action_type="rebalance_spot",
                side=OrderSide.BUY,
                quantity=abs(delta),
                reason=f"Short delta: {delta:.4f} BTC ({delta_pct*100:.2f}%)",
                urgency=urgency,
            )

    def check_margin(self) -> Dict:
        """
        Check margin health of the perpetual position.

        Returns:
            Dictionary with margin status
        """
        perp_position = self.client.get_futures_position(
            self.position_manager.perp_symbol
        )

        if perp_position is None:
            return {
                "status": "no_position",
                "message": "No perpetual position found",
            }

        # Get current price for context
        current_price = self.client.get_futures_price(
            self.position_manager.perp_symbol
        )

        # Calculate margin ratio
        if perp_position.margin > 0:
            margin_ratio = (
                perp_position.margin + perp_position.unrealized_pnl
            ) / perp_position.margin
        else:
            margin_ratio = 0

        # Distance to liquidation
        if current_price and perp_position.liquidation_price > 0:
            liq_distance_pct = abs(
                current_price - perp_position.liquidation_price
            ) / current_price
        else:
            liq_distance_pct = 1.0  # No liquidation risk

        # Determine status
        if margin_ratio < self.margin_critical_pct:
            status = "critical"
            action_needed = True
        elif margin_ratio < self.margin_warning_pct:
            status = "warning"
            action_needed = False
        else:
            status = "healthy"
            action_needed = False

        return {
            "status": status,
            "margin_ratio": margin_ratio,
            "margin_usd": perp_position.margin,
            "unrealized_pnl": perp_position.unrealized_pnl,
            "liquidation_price": perp_position.liquidation_price,
            "current_price": current_price,
            "liq_distance_pct": liq_distance_pct,
            "leverage": perp_position.leverage,
            "action_needed": action_needed,
        }

    def rebalance(self, action: HedgeAction) -> HedgeResult:
        """
        Execute a rebalancing action.

        Args:
            action: HedgeAction to execute

        Returns:
            HedgeResult with outcome
        """
        if action.action_type == "none":
            return HedgeResult(
                success=True,
                action_taken="none",
                quantity_adjusted=0.0,
                old_delta=0.0,
                new_delta=0.0,
                message="No rebalancing needed",
            )

        position = self.position_manager.position
        old_delta = position.spot_quantity - position.perp_quantity

        try:
            if action.action_type == "rebalance_spot":
                # Adjust spot position
                order = self.client.place_spot_order(
                    symbol=self.position_manager.spot_symbol,
                    side=action.side,
                    quantity=action.quantity,
                    order_type=OrderType.MARKET,
                )

                if order is None:
                    return HedgeResult(
                        success=False,
                        action_taken="spot_order_failed",
                        quantity_adjusted=0.0,
                        old_delta=old_delta,
                        new_delta=old_delta,
                        message="Failed to place spot order",
                    )

                # Update position tracking
                if action.side == OrderSide.BUY:
                    position.spot_quantity += order.filled_qty
                else:
                    position.spot_quantity -= order.filled_qty

            elif action.action_type == "rebalance_perp":
                # Adjust perpetual position
                order = self.client.place_futures_order(
                    symbol=self.position_manager.perp_symbol,
                    side=action.side,
                    quantity=action.quantity,
                    order_type=OrderType.MARKET,
                )

                if order is None:
                    return HedgeResult(
                        success=False,
                        action_taken="perp_order_failed",
                        quantity_adjusted=0.0,
                        old_delta=old_delta,
                        new_delta=old_delta,
                        message="Failed to place perp order",
                    )

                # Update position tracking (short = negative adjustment for SELL)
                if action.side == OrderSide.SELL:
                    position.perp_quantity += order.filled_qty
                else:
                    position.perp_quantity -= order.filled_qty

            new_delta = position.spot_quantity - position.perp_quantity

            # Record hedge action
            self._hedge_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action.action_type,
                "side": action.side.value if action.side else None,
                "quantity": action.quantity,
                "old_delta": old_delta,
                "new_delta": new_delta,
                "reason": action.reason,
            })

            logger.info(
                "Rebalance executed: %s %.4f BTC, delta %.4f -> %.4f",
                action.action_type,
                action.quantity,
                old_delta,
                new_delta,
            )

            return HedgeResult(
                success=True,
                action_taken=action.action_type,
                quantity_adjusted=action.quantity,
                old_delta=old_delta,
                new_delta=new_delta,
                message=f"Successfully rebalanced {action.action_type}",
            )

        except Exception as e:
            logger.error("Rebalance failed: %s", e)
            return HedgeResult(
                success=False,
                action_taken="error",
                quantity_adjusted=0.0,
                old_delta=old_delta,
                new_delta=old_delta,
                message=str(e),
            )

    def auto_hedge(self) -> Optional[HedgeResult]:
        """
        Automatically check and rebalance if needed.

        Returns:
            HedgeResult if action taken, None otherwise
        """
        action = self.check_delta()

        if action.action_type == "none":
            return None

        # Only auto-rebalance if urgency is high enough
        if action.urgency in ("high", "critical"):
            logger.info("Auto-hedging: %s (urgency=%s)", action.reason, action.urgency)
            return self.rebalance(action)
        else:
            logger.debug("Rebalance needed but not urgent: %s", action.reason)
            return None

    def add_margin(self, amount_usd: float) -> bool:
        """
        Add margin to the perpetual position to reduce liquidation risk.

        Args:
            amount_usd: Amount of USDT to add as margin

        Returns:
            True if successful
        """
        # This would require exchange-specific implementation
        # for isolated margin adjustment
        logger.warning("add_margin not implemented - requires exchange-specific API")
        return False

    def get_status(self) -> Dict:
        """Get comprehensive hedger status."""
        position = self.position_manager.position

        if not position.is_open:
            return {
                "status": "no_position",
                "message": "No active position to hedge",
            }

        delta_action = self.check_delta()
        margin_status = self.check_margin()

        return {
            "position_status": position.status.value,
            "spot_quantity": position.spot_quantity,
            "perp_quantity": position.perp_quantity,
            "delta": position.delta,
            "is_balanced": position.is_balanced,
            "delta_action": {
                "type": delta_action.action_type,
                "quantity": delta_action.quantity,
                "urgency": delta_action.urgency,
                "reason": delta_action.reason,
            },
            "margin_status": margin_status,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "hedge_history_count": len(self._hedge_history),
        }

    def get_hedge_history(self, limit: int = 20) -> List[Dict]:
        """Get recent hedge actions."""
        return self._hedge_history[-limit:]

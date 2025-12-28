"""
Hybrid Funding Arbitrage Strategy

Always-in base position + dynamic boost on rate spikes.
Captures all positive funding while scaling up during high-rate periods.

Key improvements over threshold-based strategy:
- Base position always deployed (captures ~66% more funding)
- Dynamic Kelly sizing scales with rate magnitude
- Opportunistic directional overlay when trends are strong
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("btc_trader.funding_arb.hybrid")


class StrategyMode(Enum):
    """Operating modes for the hybrid strategy."""
    PURE_ARB = "pure_arb"           # Delta-neutral, collect funding only
    DIRECTIONAL = "directional"     # Trend following, no arb
    HYBRID = "hybrid"               # Arb base + directional overlay
    FLAT = "flat"                   # No position (unfavorable conditions)


@dataclass
class HybridStrategyConfig:
    """Configuration for hybrid strategy."""

    # Base position (always deployed)
    base_position_pct: float = 0.30  # 30% of capital always in arb

    # Boost position (added when rates spike)
    max_boost_pct: float = 0.70      # Up to 70% additional on spikes
    boost_rate_cap: float = 3.0       # Cap rate ratio at 3x average

    # Rate thresholds
    min_rate_for_boost: float = 0.0001   # 0.01% minimum to add boost
    exit_rate_threshold: float = -0.0001  # Exit boost below this

    # Directional overlay
    directional_max_pct: float = 0.30    # Max 30% in directional
    directional_confidence_min: float = 0.75  # Min confidence for directional
    directional_stop_loss: float = 0.05  # 5% stop loss

    # Risk limits
    max_drawdown_pct: float = 0.10   # 10% max drawdown
    max_single_exchange_pct: float = 0.60  # Max 60% on one exchange

    # Cost assumptions
    round_trip_cost_pct: float = 0.0036  # 0.36% round trip


@dataclass
class PositionState:
    """Current position state across all components."""

    # Base arb position
    base_spot_qty: float = 0.0
    base_perp_qty: float = 0.0
    base_entry_price: float = 0.0
    base_entry_time: Optional[datetime] = None

    # Boost position
    boost_spot_qty: float = 0.0
    boost_perp_qty: float = 0.0
    boost_entry_price: float = 0.0
    boost_entry_time: Optional[datetime] = None

    # Directional overlay
    directional_qty: float = 0.0
    directional_side: str = ""  # "long" or "short"
    directional_entry_price: float = 0.0
    directional_stop_price: float = 0.0

    # P&L tracking
    total_funding_collected: float = 0.0
    total_trading_pnl: float = 0.0
    funding_payments: List[Dict] = field(default_factory=list)

    @property
    def total_spot_qty(self) -> float:
        """Total spot position (base + boost)."""
        return self.base_spot_qty + self.boost_spot_qty

    @property
    def total_perp_qty(self) -> float:
        """Total perp position (base + boost)."""
        return self.base_perp_qty + self.boost_perp_qty

    @property
    def net_delta(self) -> float:
        """Net BTC exposure (should be ~0 for arb)."""
        arb_delta = self.total_spot_qty - self.total_perp_qty

        if self.directional_side == "long":
            return arb_delta + self.directional_qty
        elif self.directional_side == "short":
            return arb_delta - self.directional_qty
        return arb_delta

    @property
    def is_base_active(self) -> bool:
        """Whether base position is active."""
        return self.base_spot_qty > 0 or self.base_perp_qty > 0

    @property
    def is_boost_active(self) -> bool:
        """Whether boost position is active."""
        return self.boost_spot_qty > 0 or self.boost_perp_qty > 0

    @property
    def is_directional_active(self) -> bool:
        """Whether directional overlay is active."""
        return self.directional_qty > 0


@dataclass
class SizingDecision:
    """Position sizing decision from the strategy."""

    mode: StrategyMode
    base_size_usd: float
    boost_size_usd: float
    directional_size_usd: float
    directional_side: str  # "long", "short", or ""
    confidence: float
    reasoning: str


class HybridFundingStrategy:
    """
    Hybrid funding arbitrage strategy with always-in base + dynamic boost.

    Key Features:
    1. Base Position (30%): Always deployed to capture ALL positive funding
    2. Boost Position (0-70%): Scales dynamically with rate magnitude
    3. Directional Overlay (0-30%): Added during strong trends when arb unprofitable
    """

    def __init__(self, config: Optional[HybridStrategyConfig] = None):
        self.config = config or HybridStrategyConfig()
        self.position = PositionState()
        self.rate_history: List[Tuple[datetime, float]] = []
        self._avg_rate_30d: float = 0.0
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0

        logger.info(
            "HybridFundingStrategy initialized: base=%.0f%%, max_boost=%.0f%%",
            self.config.base_position_pct * 100,
            self.config.max_boost_pct * 100,
        )

    def update_rate_history(self, rate: float, timestamp: Optional[datetime] = None) -> None:
        """Add a new funding rate to history."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.rate_history.append((timestamp, rate))

        # Keep only last 30 days (3 periods/day * 30 days = 90 periods)
        max_history = 90
        if len(self.rate_history) > max_history:
            self.rate_history = self.rate_history[-max_history:]

        # Update 30-day average
        self._update_avg_rate()

    def _update_avg_rate(self) -> None:
        """Calculate rolling 30-day average rate."""
        if not self.rate_history:
            self._avg_rate_30d = 0.0001  # Default to 0.01%
            return

        positive_rates = [r for _, r in self.rate_history if r > 0]
        if positive_rates:
            self._avg_rate_30d = sum(positive_rates) / len(positive_rates)
        else:
            self._avg_rate_30d = 0.0001  # Minimum default

    def calculate_position_sizes(
        self,
        current_rate: float,
        capital: float,
        regime: str = "UNKNOWN",
        trend_confidence: float = 0.0,
    ) -> SizingDecision:
        """
        Calculate optimal position sizes for all components.

        Args:
            current_rate: Current funding rate (e.g., 0.0001 = 0.01%)
            capital: Available capital in USD
            regime: Market regime (STRONG_UPTREND, STRONG_DOWNTREND, etc.)
            trend_confidence: Confidence in trend signal (0-1)

        Returns:
            SizingDecision with sizes for base, boost, and directional
        """
        # Apply drawdown scaling
        dd_multiplier = self._get_drawdown_multiplier()

        # 1. BASE POSITION (always deployed if profitable or near-zero)
        base_size = capital * self.config.base_position_pct * dd_multiplier

        # 2. BOOST POSITION (dynamic Kelly based on rate)
        boost_size = self._calculate_boost_size(current_rate, capital, dd_multiplier)

        # 3. DIRECTIONAL OVERLAY (only in strong trends with low arb profitability)
        directional_size, directional_side = self._calculate_directional_size(
            current_rate, capital, regime, trend_confidence, dd_multiplier
        )

        # Determine overall mode
        if directional_size > 0:
            if base_size > 0 or boost_size > 0:
                mode = StrategyMode.HYBRID
            else:
                mode = StrategyMode.DIRECTIONAL
        elif base_size > 0 or boost_size > 0:
            mode = StrategyMode.PURE_ARB
        else:
            mode = StrategyMode.FLAT

        # Build reasoning
        reasoning = self._build_reasoning(
            current_rate, base_size, boost_size, directional_size, mode
        )

        return SizingDecision(
            mode=mode,
            base_size_usd=base_size,
            boost_size_usd=boost_size,
            directional_size_usd=directional_size,
            directional_side=directional_side,
            confidence=self._calculate_confidence(current_rate, regime, trend_confidence),
            reasoning=reasoning,
        )

    def _calculate_boost_size(
        self,
        current_rate: float,
        capital: float,
        dd_multiplier: float,
    ) -> float:
        """
        Calculate boost position size using dynamic Kelly.

        Formula: boost = max_boost * (rate_ratio - 1) / (cap - 1)
        Where rate_ratio = current_rate / avg_rate, capped at boost_rate_cap
        """
        if current_rate <= self.config.min_rate_for_boost:
            return 0.0

        if self._avg_rate_30d <= 0:
            return 0.0

        # Calculate rate ratio (how much above average)
        rate_ratio = current_rate / self._avg_rate_30d
        rate_ratio = min(rate_ratio, self.config.boost_rate_cap)

        if rate_ratio <= 1.0:
            return 0.0  # Below average, no boost

        # Scale boost linearly from 0% at ratio=1 to max at ratio=cap
        boost_fraction = (rate_ratio - 1.0) / (self.config.boost_rate_cap - 1.0)
        boost_pct = self.config.max_boost_pct * boost_fraction

        return capital * boost_pct * dd_multiplier

    def _calculate_directional_size(
        self,
        current_rate: float,
        capital: float,
        regime: str,
        trend_confidence: float,
        dd_multiplier: float,
    ) -> Tuple[float, str]:
        """
        Calculate directional overlay size.

        Only add directional when:
        1. Strong trend detected
        2. High confidence (>0.75)
        3. Arb is unprofitable (rate < breakeven)
        """
        # Check if arb is profitable
        breakeven_rate = self.config.round_trip_cost_pct / 3  # Per 8h period
        arb_profitable = current_rate > breakeven_rate

        # Check for strong trend
        strong_trend = regime in ("STRONG_UPTREND", "STRONG_DOWNTREND")
        high_confidence = trend_confidence >= self.config.directional_confidence_min

        if not (strong_trend and high_confidence and not arb_profitable):
            return 0.0, ""

        # Determine side
        if regime == "STRONG_UPTREND":
            side = "long"
        elif regime == "STRONG_DOWNTREND":
            side = "short"
        else:
            return 0.0, ""

        # Scale size by confidence
        confidence_scale = (trend_confidence - self.config.directional_confidence_min) / (
            1.0 - self.config.directional_confidence_min
        )
        confidence_scale = min(1.0, max(0.0, confidence_scale))

        size = capital * self.config.directional_max_pct * confidence_scale * dd_multiplier

        return size, side

    def _get_drawdown_multiplier(self) -> float:
        """
        Get position size multiplier based on current drawdown.

        Drawdown | Multiplier
        ---------|----------
        0-3%     | 1.0x
        3-6%     | 0.75x
        6-8%     | 0.50x
        8-10%    | 0.25x
        >10%     | 0.0x (HALT)
        """
        dd = self._current_drawdown

        if dd < 0.03:
            return 1.0
        elif dd < 0.06:
            return 0.75
        elif dd < 0.08:
            return 0.50
        elif dd < self.config.max_drawdown_pct:
            return 0.25
        else:
            return 0.0  # HALT

    def _calculate_confidence(
        self,
        current_rate: float,
        regime: str,
        trend_confidence: float,
    ) -> float:
        """Calculate overall strategy confidence."""
        base_confidence = 0.5

        # Rate above average boosts confidence
        if self._avg_rate_30d > 0:
            rate_ratio = current_rate / self._avg_rate_30d
            if rate_ratio > 1.5:
                base_confidence += 0.2
            elif rate_ratio > 1.0:
                base_confidence += 0.1

        # Strong regime boosts confidence
        if regime in ("STRONG_UPTREND", "STRONG_DOWNTREND"):
            base_confidence += 0.15

        # Trend confidence contributes
        base_confidence += trend_confidence * 0.15

        return min(1.0, base_confidence)

    def _build_reasoning(
        self,
        current_rate: float,
        base_size: float,
        boost_size: float,
        directional_size: float,
        mode: StrategyMode,
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts = [f"Mode: {mode.value}"]

        rate_pct = current_rate * 100
        avg_pct = self._avg_rate_30d * 100
        parts.append(f"Rate: {rate_pct:.4f}% (avg: {avg_pct:.4f}%)")

        if base_size > 0:
            parts.append(f"Base: ${base_size:,.0f}")
        if boost_size > 0:
            ratio = current_rate / self._avg_rate_30d if self._avg_rate_30d > 0 else 0
            parts.append(f"Boost: ${boost_size:,.0f} (rate {ratio:.1f}x avg)")
        if directional_size > 0:
            parts.append(f"Directional: ${directional_size:,.0f}")

        if self._current_drawdown > 0.03:
            mult = self._get_drawdown_multiplier()
            parts.append(f"DD scaling: {mult:.0%}")

        return " | ".join(parts)

    def update_equity(self, current_equity: float, initial_equity: float) -> None:
        """Update equity tracking for drawdown calculation."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        else:
            self._current_drawdown = 0.0

    def record_funding_payment(
        self,
        amount: float,
        rate: float,
        position_size: float,
    ) -> None:
        """Record a funding payment."""
        payment = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "amount": amount,
            "rate": rate,
            "position_size": position_size,
        }
        self.position.funding_payments.append(payment)
        self.position.total_funding_collected += amount

        logger.debug(
            "Funding payment recorded: $%.2f (rate: %.4f%%)",
            amount,
            rate * 100,
        )

    def should_rebalance_boost(self, current_rate: float) -> Tuple[bool, str]:
        """
        Check if boost position should be rebalanced.

        Returns:
            (should_rebalance, action) where action is "add", "reduce", or "none"
        """
        if not self.position.is_boost_active:
            # No boost - check if we should add
            if current_rate > self.config.min_rate_for_boost:
                if self._avg_rate_30d > 0 and current_rate > self._avg_rate_30d:
                    return True, "add"
            return False, "none"

        # Has boost - check if we should reduce
        if current_rate < self.config.exit_rate_threshold:
            return True, "reduce"

        if current_rate < self._avg_rate_30d * 0.5:
            return True, "reduce"

        return False, "none"

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        total_pnl = self.position.total_funding_collected + self.position.total_trading_pnl

        return {
            "total_funding_collected": self.position.total_funding_collected,
            "total_trading_pnl": self.position.total_trading_pnl,
            "total_pnl": total_pnl,
            "funding_payments_count": len(self.position.funding_payments),
            "current_drawdown": self._current_drawdown,
            "peak_equity": self._peak_equity,
            "avg_rate_30d": self._avg_rate_30d,
            "is_base_active": self.position.is_base_active,
            "is_boost_active": self.position.is_boost_active,
            "is_directional_active": self.position.is_directional_active,
            "net_delta": self.position.net_delta,
        }


# Convenience function for quick calculations
def calculate_hybrid_size(
    current_rate: float,
    avg_rate: float,
    capital: float,
    base_pct: float = 0.30,
    max_boost_pct: float = 0.70,
    rate_cap: float = 3.0,
) -> Tuple[float, float]:
    """
    Quick calculation of base + boost sizes.

    Args:
        current_rate: Current funding rate
        avg_rate: Average funding rate (30 day)
        capital: Total capital
        base_pct: Base position percentage
        max_boost_pct: Maximum boost percentage
        rate_cap: Cap on rate ratio

    Returns:
        (base_size, boost_size) in USD
    """
    base = capital * base_pct

    if current_rate <= 0 or avg_rate <= 0:
        return base, 0.0

    rate_ratio = min(current_rate / avg_rate, rate_cap)

    if rate_ratio <= 1.0:
        return base, 0.0

    boost_fraction = (rate_ratio - 1.0) / (rate_cap - 1.0)
    boost = capital * max_boost_pct * boost_fraction

    return base, boost

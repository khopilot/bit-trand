"""
Strategy Orchestrator

Unified controller that decides between:
1. Pure funding arb (delta-neutral)
2. Directional overlay (trend following)
3. Hybrid (arb base + directional boost)
4. Flat (no position)

Based on expert consultation:
- Opportunistic blend strategy
- Run arb when rates > costs
- Switch to directional when strong trend detected
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("btc_trader.strategy.orchestrator")


class StrategyMode(Enum):
    """Operating modes for the orchestrator."""
    PURE_ARB = "pure_arb"
    DIRECTIONAL = "directional"
    HYBRID = "hybrid"
    FLAT = "flat"


class TrendDirection(Enum):
    """Market trend direction."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class MarketState:
    """Current market state snapshot."""
    timestamp: datetime
    price: float
    funding_rate: float
    regime: TrendDirection
    trend_confidence: float
    volatility: float
    ema_spread_pct: float  # EMA12 vs EMA26 spread


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Breakeven rate (cost to enter/exit once)
    breakeven_rate: float = 0.0012  # 0.12% per period = 0.36% round trip / 3 periods

    # Arb thresholds
    arb_profitable_multiplier: float = 2.0  # Rate must be 2x breakeven for pure arb
    arb_minimum_rate: float = 0.0001        # Minimum rate to consider arb

    # Directional thresholds
    directional_confidence_min: float = 0.75
    directional_ema_spread_min: float = 0.02  # 2% EMA spread for strong trend

    # Risk limits
    max_drawdown_pct: float = 0.10
    max_position_pct: float = 1.00

    # Mode switching
    min_mode_duration_hours: float = 1.0  # Minimum time in a mode before switching
    cooldown_after_loss_hours: float = 4.0  # Cooldown after directional loss


@dataclass
class ModeDecision:
    """Decision on which mode to operate in."""
    mode: StrategyMode
    confidence: float
    reasoning: str
    arb_allocation: float  # 0-1
    directional_allocation: float  # 0-1
    directional_side: str  # "long", "short", or ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrchestratorState:
    """Internal state of the orchestrator."""
    current_mode: StrategyMode = StrategyMode.FLAT
    mode_entry_time: Optional[datetime] = None
    last_directional_loss_time: Optional[datetime] = None
    consecutive_losses: int = 0
    mode_history: List[Tuple[datetime, StrategyMode]] = field(default_factory=list)
    total_arb_pnl: float = 0.0
    total_directional_pnl: float = 0.0


class StrategyOrchestrator:
    """
    Master controller for all trading strategies.

    Decision Logic:
    1. If funding rate > 2x breakeven AND not strong trend:
       → PURE_ARB (capture high funding)

    2. If strong trend detected AND confidence > 0.75 AND rate < breakeven:
       → DIRECTIONAL (ride the trend)

    3. If funding rate > breakeven AND strong trend:
       → HYBRID (arb base + directional overlay)

    4. Otherwise:
       → FLAT (wait for better conditions)
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.state = OrchestratorState()

        logger.info(
            "StrategyOrchestrator initialized: breakeven=%.4f%%, dd_limit=%.0f%%",
            self.config.breakeven_rate * 100,
            self.config.max_drawdown_pct * 100,
        )

    def decide_mode(self, market: MarketState) -> ModeDecision:
        """
        Decide which strategy mode to operate in.

        Args:
            market: Current market state

        Returns:
            ModeDecision with mode and allocations
        """
        # Check if we can switch modes
        if not self._can_switch_mode():
            return self._continue_current_mode(market)

        # Check directional cooldown
        if self._in_directional_cooldown():
            # Only allow arb or flat
            return self._decide_arb_or_flat(market)

        # Get key metrics
        rate = market.funding_rate
        regime = market.regime
        confidence = market.trend_confidence
        ema_spread = abs(market.ema_spread_pct)

        is_strong_trend = regime in (
            TrendDirection.STRONG_UPTREND,
            TrendDirection.STRONG_DOWNTREND,
        )
        is_high_confidence = confidence >= self.config.directional_confidence_min
        is_arb_very_profitable = rate >= self.config.breakeven_rate * self.config.arb_profitable_multiplier
        is_arb_profitable = rate >= self.config.breakeven_rate
        is_arb_positive = rate >= self.config.arb_minimum_rate

        # Decision tree
        if is_arb_very_profitable and not is_strong_trend:
            # High funding rate, no strong trend → Pure arb
            return self._create_decision(
                mode=StrategyMode.PURE_ARB,
                confidence=0.85,
                reasoning=f"High funding rate ({rate*100:.4f}%) without strong trend",
                arb=1.0,
                directional=0.0,
            )

        elif is_strong_trend and is_high_confidence and not is_arb_profitable:
            # Strong trend, arb not profitable → Directional
            side = "long" if regime == TrendDirection.STRONG_UPTREND else "short"
            return self._create_decision(
                mode=StrategyMode.DIRECTIONAL,
                confidence=confidence,
                reasoning=f"Strong {regime.value} with {confidence:.0%} confidence, arb unprofitable",
                arb=0.0,
                directional=1.0,
                side=side,
            )

        elif is_strong_trend and is_high_confidence and is_arb_profitable:
            # Strong trend AND arb profitable → Hybrid
            side = "long" if regime == TrendDirection.STRONG_UPTREND else "short"

            # Allocate based on rate magnitude
            arb_pct = min(0.70, rate / (self.config.breakeven_rate * 3))
            directional_pct = 1.0 - arb_pct

            return self._create_decision(
                mode=StrategyMode.HYBRID,
                confidence=min(0.9, confidence),
                reasoning=f"Strong trend + profitable arb ({rate*100:.4f}%)",
                arb=arb_pct,
                directional=directional_pct,
                side=side,
            )

        elif is_arb_profitable:
            # Arb profitable, no strong trend → Pure arb
            return self._create_decision(
                mode=StrategyMode.PURE_ARB,
                confidence=0.70,
                reasoning=f"Arb profitable ({rate*100:.4f}%), no strong trend",
                arb=1.0,
                directional=0.0,
            )

        elif is_arb_positive:
            # Arb marginally positive → Small arb or flat
            return self._create_decision(
                mode=StrategyMode.PURE_ARB,
                confidence=0.50,
                reasoning=f"Marginal arb opportunity ({rate*100:.4f}%)",
                arb=0.5,  # Half size
                directional=0.0,
            )

        else:
            # Neither arb nor directional favorable → Flat
            return self._create_decision(
                mode=StrategyMode.FLAT,
                confidence=0.90,
                reasoning="No favorable conditions, staying flat",
                arb=0.0,
                directional=0.0,
            )

    def _create_decision(
        self,
        mode: StrategyMode,
        confidence: float,
        reasoning: str,
        arb: float,
        directional: float,
        side: str = "",
    ) -> ModeDecision:
        """Create a mode decision and update state."""
        decision = ModeDecision(
            mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            arb_allocation=arb,
            directional_allocation=directional,
            directional_side=side,
        )

        # Update state if mode changed
        if mode != self.state.current_mode:
            self.state.mode_history.append((datetime.now(timezone.utc), mode))
            self.state.current_mode = mode
            self.state.mode_entry_time = datetime.now(timezone.utc)

            logger.info(
                "Mode changed: %s → %s | Reason: %s",
                self.state.current_mode.value if self.state.current_mode else "none",
                mode.value,
                reasoning,
            )

        return decision

    def _continue_current_mode(self, market: MarketState) -> ModeDecision:
        """Continue in current mode without switching."""
        mode = self.state.current_mode

        if mode == StrategyMode.PURE_ARB:
            return self._create_decision(
                mode=mode,
                confidence=0.70,
                reasoning="Continuing arb (min duration not met)",
                arb=1.0,
                directional=0.0,
            )
        elif mode == StrategyMode.DIRECTIONAL:
            side = "long" if market.regime in (
                TrendDirection.STRONG_UPTREND,
                TrendDirection.WEAK_UPTREND,
            ) else "short"
            return self._create_decision(
                mode=mode,
                confidence=0.70,
                reasoning="Continuing directional (min duration not met)",
                arb=0.0,
                directional=1.0,
                side=side,
            )
        elif mode == StrategyMode.HYBRID:
            side = "long" if market.regime in (
                TrendDirection.STRONG_UPTREND,
                TrendDirection.WEAK_UPTREND,
            ) else "short"
            return self._create_decision(
                mode=mode,
                confidence=0.70,
                reasoning="Continuing hybrid (min duration not met)",
                arb=0.5,
                directional=0.5,
                side=side,
            )
        else:
            return self._create_decision(
                mode=StrategyMode.FLAT,
                confidence=0.90,
                reasoning="Staying flat (min duration not met)",
                arb=0.0,
                directional=0.0,
            )

    def _decide_arb_or_flat(self, market: MarketState) -> ModeDecision:
        """Decide between arb and flat (directional in cooldown)."""
        rate = market.funding_rate

        if rate >= self.config.breakeven_rate:
            return self._create_decision(
                mode=StrategyMode.PURE_ARB,
                confidence=0.75,
                reasoning="Arb profitable, directional in cooldown",
                arb=1.0,
                directional=0.0,
            )
        elif rate >= self.config.arb_minimum_rate:
            return self._create_decision(
                mode=StrategyMode.PURE_ARB,
                confidence=0.50,
                reasoning="Marginal arb, directional in cooldown",
                arb=0.5,
                directional=0.0,
            )
        else:
            return self._create_decision(
                mode=StrategyMode.FLAT,
                confidence=0.90,
                reasoning="No arb opportunity, directional in cooldown",
                arb=0.0,
                directional=0.0,
            )

    def _can_switch_mode(self) -> bool:
        """Check if enough time has passed to switch modes."""
        if self.state.mode_entry_time is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self.state.mode_entry_time).total_seconds()
        min_duration = self.config.min_mode_duration_hours * 3600

        return elapsed >= min_duration

    def _in_directional_cooldown(self) -> bool:
        """Check if in cooldown after directional loss."""
        if self.state.last_directional_loss_time is None:
            return False

        elapsed = (datetime.now(timezone.utc) - self.state.last_directional_loss_time).total_seconds()
        cooldown = self.config.cooldown_after_loss_hours * 3600

        return elapsed < cooldown

    def record_pnl(self, pnl: float, is_arb: bool) -> None:
        """Record P&L from a trade."""
        if is_arb:
            self.state.total_arb_pnl += pnl
        else:
            self.state.total_directional_pnl += pnl

            # Track directional losses
            if pnl < 0:
                self.state.consecutive_losses += 1
                self.state.last_directional_loss_time = datetime.now(timezone.utc)
                logger.warning(
                    "Directional loss: $%.2f (consecutive: %d)",
                    pnl,
                    self.state.consecutive_losses,
                )
            else:
                self.state.consecutive_losses = 0

    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        return {
            "current_mode": self.state.current_mode.value,
            "mode_entry_time": (
                self.state.mode_entry_time.isoformat()
                if self.state.mode_entry_time else None
            ),
            "in_directional_cooldown": self._in_directional_cooldown(),
            "consecutive_losses": self.state.consecutive_losses,
            "total_arb_pnl": f"${self.state.total_arb_pnl:.2f}",
            "total_directional_pnl": f"${self.state.total_directional_pnl:.2f}",
            "total_pnl": f"${self.state.total_arb_pnl + self.state.total_directional_pnl:.2f}",
            "mode_switches": len(self.state.mode_history),
        }

    def get_mode_history(self, limit: int = 10) -> List[Dict]:
        """Get recent mode history."""
        recent = self.state.mode_history[-limit:]
        return [
            {"timestamp": ts.isoformat(), "mode": mode.value}
            for ts, mode in recent
        ]

    def force_mode(self, mode: StrategyMode, reason: str = "Manual override") -> None:
        """Force a specific mode (for testing or manual control)."""
        self.state.mode_history.append((datetime.now(timezone.utc), mode))
        self.state.current_mode = mode
        self.state.mode_entry_time = datetime.now(timezone.utc)

        logger.warning("Mode FORCED: %s | Reason: %s", mode.value, reason)

    def reset_state(self) -> None:
        """Reset orchestrator state."""
        self.state = OrchestratorState()
        logger.info("Orchestrator state reset")


# Strategy mode init file
def create_strategy_init():
    """Helper to create init file."""
    pass

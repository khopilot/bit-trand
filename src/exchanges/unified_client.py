"""
Unified Exchange Client

Abstraction layer that routes to the best exchange based on funding rates.
Supports position splitting across multiple exchanges.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .okx_client import OKXClient, OKXConfig

logger = logging.getLogger("btc_trader.exchanges.unified")


class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    OKX = "okx"


@dataclass
class ExchangeRate:
    """Funding rate from an exchange."""
    exchange: Exchange
    rate: float
    timestamp: datetime
    next_funding_time: Optional[datetime] = None
    predicted_rate: Optional[float] = None


@dataclass
class UnifiedPosition:
    """Position on a specific exchange."""
    exchange: Exchange
    spot_qty: float
    perp_qty: float
    spot_entry_price: float
    perp_entry_price: float
    entry_time: datetime
    funding_collected: float = 0.0


@dataclass
class UnifiedBalance:
    """Balance on a specific exchange."""
    exchange: Exchange
    available_usdt: float
    total_usdt: float


@dataclass
class AllocationDecision:
    """Decision on how to allocate across exchanges."""
    allocations: Dict[Exchange, float]  # Exchange -> percentage (0-1)
    best_exchange: Exchange
    best_rate: float
    reasoning: str


@dataclass
class UnifiedClientConfig:
    """Configuration for unified client."""
    # Exchange enablement
    enable_binance: bool = True
    enable_okx: bool = True

    # Allocation rules
    max_single_exchange_pct: float = 0.60  # Max 60% on one exchange
    min_allocation_pct: float = 0.10       # Min 10% to allocate
    rebalance_rate_diff_pct: float = 0.20  # Rebalance if rate diff > 20%

    # Rate routing
    prefer_higher_rate: bool = True
    rate_staleness_seconds: int = 300  # Consider rate stale after 5 min

    # OKX config
    okx_config: Optional[OKXConfig] = None


class UnifiedExchangeClient:
    """
    Unified client that abstracts multiple exchanges.

    Features:
    1. Fetch funding rates from all enabled exchanges
    2. Route to best exchange based on rates
    3. Split positions across exchanges
    4. Track cross-exchange delta
    """

    def __init__(self, config: Optional[UnifiedClientConfig] = None):
        self.config = config or UnifiedClientConfig()
        self._clients: Dict[Exchange, object] = {}
        self._rate_cache: Dict[Exchange, ExchangeRate] = {}
        self._positions: Dict[Exchange, UnifiedPosition] = {}

        self._initialize_clients()

        logger.info(
            "UnifiedExchangeClient initialized: binance=%s, okx=%s",
            self.config.enable_binance,
            self.config.enable_okx,
        )

    def _initialize_clients(self) -> None:
        """Initialize exchange clients."""
        if self.config.enable_okx:
            okx_config = self.config.okx_config or OKXConfig()
            self._clients[Exchange.OKX] = OKXClient(okx_config)

        if self.config.enable_binance:
            # Import Binance client from existing module
            try:
                from ..funding_arbitrage.exchange_client import BinanceClient
                self._clients[Exchange.BINANCE] = BinanceClient(testnet=True)
            except ImportError:
                logger.warning("BinanceClient not available")

    # =========================================================================
    # RATE FETCHING
    # =========================================================================

    def get_all_rates(self) -> Dict[Exchange, ExchangeRate]:
        """
        Fetch current funding rates from all enabled exchanges.

        Returns:
            Dict mapping exchange to rate info
        """
        rates = {}
        now = datetime.now(timezone.utc)

        # OKX
        if Exchange.OKX in self._clients:
            try:
                okx_client: OKXClient = self._clients[Exchange.OKX]
                rate = okx_client.get_funding_rate()
                if rate is not None:
                    rates[Exchange.OKX] = ExchangeRate(
                        exchange=Exchange.OKX,
                        rate=rate,
                        timestamp=now,
                    )
                    self._rate_cache[Exchange.OKX] = rates[Exchange.OKX]
            except Exception as e:
                logger.warning("Failed to get OKX rate: %s", e)

        # Binance
        if Exchange.BINANCE in self._clients:
            try:
                from ..funding_arbitrage.rate_monitor import FundingRateMonitor
                monitor = FundingRateMonitor()
                binance_rate = monitor.get_binance_rate()
                if binance_rate:
                    rates[Exchange.BINANCE] = ExchangeRate(
                        exchange=Exchange.BINANCE,
                        rate=binance_rate.rate,
                        timestamp=now,
                        next_funding_time=binance_rate.next_funding_time,
                    )
                    self._rate_cache[Exchange.BINANCE] = rates[Exchange.BINANCE]
            except Exception as e:
                logger.warning("Failed to get Binance rate: %s", e)

        return rates

    def get_best_exchange(self) -> Tuple[Exchange, float]:
        """
        Get exchange with highest positive funding rate.

        Returns:
            (best_exchange, rate) tuple
        """
        rates = self.get_all_rates()

        if not rates:
            logger.warning("No rates available from any exchange")
            return Exchange.BINANCE, 0.0

        # Filter positive rates
        positive_rates = {
            ex: r for ex, r in rates.items()
            if r.rate > 0
        }

        if not positive_rates:
            # All negative - return least negative
            best = max(rates.items(), key=lambda x: x[1].rate)
            return best[0], best[1].rate

        # Return highest positive rate
        best = max(positive_rates.items(), key=lambda x: x[1].rate)
        return best[0], best[1].rate

    def get_rate_comparison(self) -> Dict:
        """
        Get detailed rate comparison across exchanges.

        Returns:
            Dict with comparison data
        """
        rates = self.get_all_rates()

        if not rates:
            return {"error": "No rates available"}

        comparison = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchanges": {},
            "best_exchange": None,
            "best_rate": 0.0,
            "rate_spread": 0.0,
        }

        for ex, rate in rates.items():
            comparison["exchanges"][ex.value] = {
                "rate": rate.rate,
                "rate_pct": f"{rate.rate * 100:.4f}%",
                "annualized_pct": f"{rate.rate * 3 * 365 * 100:.1f}%",
            }

        if rates:
            best_ex, best_rate = self.get_best_exchange()
            comparison["best_exchange"] = best_ex.value
            comparison["best_rate"] = best_rate

            # Calculate spread
            all_rates = [r.rate for r in rates.values()]
            comparison["rate_spread"] = max(all_rates) - min(all_rates)

        return comparison

    # =========================================================================
    # ALLOCATION DECISIONS
    # =========================================================================

    def calculate_allocation(
        self,
        total_capital: float,
        current_positions: Optional[Dict[Exchange, UnifiedPosition]] = None,
    ) -> AllocationDecision:
        """
        Calculate optimal allocation across exchanges.

        Strategy:
        1. Prefer exchange with higher rate
        2. Cap any single exchange at max_single_exchange_pct
        3. Rebalance if rate difference > threshold

        Args:
            total_capital: Total capital to allocate
            current_positions: Existing positions (if any)

        Returns:
            AllocationDecision with recommended allocations
        """
        rates = self.get_all_rates()

        if not rates:
            # Default to Binance only
            return AllocationDecision(
                allocations={Exchange.BINANCE: 1.0},
                best_exchange=Exchange.BINANCE,
                best_rate=0.0,
                reasoning="No rate data available, defaulting to Binance",
            )

        if len(rates) == 1:
            # Only one exchange available
            ex = list(rates.keys())[0]
            return AllocationDecision(
                allocations={ex: 1.0},
                best_exchange=ex,
                best_rate=rates[ex].rate,
                reasoning=f"Only {ex.value} available",
            )

        # Multiple exchanges - calculate optimal split
        best_ex, best_rate = self.get_best_exchange()
        sorted_rates = sorted(rates.items(), key=lambda x: x[1].rate, reverse=True)

        allocations = {}
        reasoning_parts = []

        # Calculate rate-weighted allocation
        total_positive_rate = sum(max(0, r.rate) for r in rates.values())

        if total_positive_rate > 0:
            for ex, rate in rates.items():
                if rate.rate > 0:
                    weight = rate.rate / total_positive_rate
                    # Apply caps
                    weight = min(weight, self.config.max_single_exchange_pct)
                    weight = max(weight, self.config.min_allocation_pct) if weight > 0 else 0
                    allocations[ex] = weight
        else:
            # All negative - equal split
            for ex in rates:
                allocations[ex] = 1.0 / len(rates)
            reasoning_parts.append("All rates negative, equal split")

        # Normalize to sum to 1.0
        total_weight = sum(allocations.values())
        if total_weight > 0:
            allocations = {ex: w / total_weight for ex, w in allocations.items()}

        # Build reasoning
        for ex, pct in allocations.items():
            if pct > 0:
                rate = rates[ex].rate
                reasoning_parts.append(
                    f"{ex.value}: {pct:.0%} (rate: {rate*100:.4f}%)"
                )

        return AllocationDecision(
            allocations=allocations,
            best_exchange=best_ex,
            best_rate=best_rate,
            reasoning=" | ".join(reasoning_parts),
        )

    def split_position(
        self,
        total_size_usd: float,
        allocation: Optional[AllocationDecision] = None,
    ) -> Dict[Exchange, float]:
        """
        Split a position across exchanges based on allocation.

        Args:
            total_size_usd: Total position size in USD
            allocation: Allocation decision (if None, calculates new one)

        Returns:
            Dict mapping exchange to position size in USD
        """
        if allocation is None:
            allocation = self.calculate_allocation(total_size_usd)

        return {
            ex: total_size_usd * pct
            for ex, pct in allocation.allocations.items()
            if pct > 0
        }

    # =========================================================================
    # POSITION TRACKING
    # =========================================================================

    def get_total_delta(self) -> float:
        """
        Get total delta across all exchanges.

        Returns:
            Net BTC exposure (should be ~0 for arb)
        """
        total_spot = sum(p.spot_qty for p in self._positions.values())
        total_perp = sum(p.perp_qty for p in self._positions.values())
        return total_spot - total_perp

    def get_total_notional(self, current_price: float) -> float:
        """Get total position notional across all exchanges."""
        total_spot = sum(p.spot_qty for p in self._positions.values())
        return total_spot * current_price

    def get_exchange_breakdown(self) -> Dict[str, Dict]:
        """Get position breakdown by exchange."""
        result = {}

        for ex, pos in self._positions.items():
            result[ex.value] = {
                "spot_qty": pos.spot_qty,
                "perp_qty": pos.perp_qty,
                "delta": pos.spot_qty - pos.perp_qty,
                "funding_collected": pos.funding_collected,
            }

        return result

    def record_position(
        self,
        exchange: Exchange,
        position: UnifiedPosition,
    ) -> None:
        """Record a position on an exchange."""
        self._positions[exchange] = position
        logger.info(
            "Position recorded on %s: spot=%.6f, perp=%.6f",
            exchange.value,
            position.spot_qty,
            position.perp_qty,
        )

    def remove_position(self, exchange: Exchange) -> None:
        """Remove position from tracking."""
        if exchange in self._positions:
            del self._positions[exchange]
            logger.info("Position removed from %s", exchange.value)

    # =========================================================================
    # REBALANCING
    # =========================================================================

    def should_rebalance(self) -> Tuple[bool, str]:
        """
        Check if positions should be rebalanced across exchanges.

        Rebalance triggers:
        1. Rate difference > threshold
        2. Single exchange > max allocation

        Returns:
            (should_rebalance, reason)
        """
        if len(self._positions) <= 1:
            return False, "Only one exchange has positions"

        rates = self.get_all_rates()

        if len(rates) < 2:
            return False, "Not enough rate data"

        # Check rate difference
        rate_values = [r.rate for r in rates.values()]
        rate_diff = max(rate_values) - min(rate_values)
        avg_rate = sum(rate_values) / len(rate_values)

        if avg_rate > 0 and rate_diff / avg_rate > self.config.rebalance_rate_diff_pct:
            return True, f"Rate difference {rate_diff/avg_rate:.0%} > threshold"

        # Check single exchange concentration
        total_notional = sum(
            p.spot_qty * p.spot_entry_price
            for p in self._positions.values()
        )

        if total_notional > 0:
            for ex, pos in self._positions.items():
                notional = pos.spot_qty * pos.spot_entry_price
                pct = notional / total_notional
                if pct > self.config.max_single_exchange_pct + 0.1:
                    return True, f"{ex.value} has {pct:.0%} > max {self.config.max_single_exchange_pct:.0%}"

        return False, "No rebalance needed"

    def calculate_rebalance_trades(
        self,
        target_allocation: AllocationDecision,
        current_price: float,
    ) -> List[Dict]:
        """
        Calculate trades needed to rebalance to target allocation.

        Returns:
            List of trade instructions
        """
        trades = []
        total_notional = self.get_total_notional(current_price)

        if total_notional == 0:
            return trades

        # Calculate target notional per exchange
        target_notionals = {
            ex: total_notional * pct
            for ex, pct in target_allocation.allocations.items()
        }

        # Calculate current notional per exchange
        current_notionals = {
            ex: pos.spot_qty * current_price
            for ex, pos in self._positions.items()
        }

        # Calculate differences
        for ex in set(target_notionals.keys()) | set(current_notionals.keys()):
            target = target_notionals.get(ex, 0)
            current = current_notionals.get(ex, 0)
            diff = target - current

            if abs(diff) > 100:  # Only trade if difference > $100
                trade = {
                    "exchange": ex.value,
                    "action": "increase" if diff > 0 else "decrease",
                    "amount_usd": abs(diff),
                    "amount_btc": abs(diff) / current_price,
                    "target_pct": target_allocation.allocations.get(ex, 0),
                    "current_pct": current / total_notional if total_notional > 0 else 0,
                }
                trades.append(trade)

        return trades


# Create exports init file
def create_exchanges_init():
    """Helper to check if __init__.py exists."""
    pass

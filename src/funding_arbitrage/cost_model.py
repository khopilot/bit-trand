"""
Cost Model for Funding Arbitrage

Models realistic trading costs including fees, slippage, and borrowing costs.
"""

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger("btc_trader.funding_arb.cost_model")


@dataclass
class CostParameters:
    """
    Trading cost parameters.

    Default values are for Binance VIP 0 tier.
    """

    # Trading fees (Binance default tier)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%

    # Slippage estimates
    entry_slippage: float = 0.0005  # 0.05%
    exit_slippage: float = 0.0005  # 0.05%

    # Borrowing costs (if using spot margin)
    spot_borrow_rate_daily: float = 0.0  # Usually 0 if holding spot outright

    # Funding-specific costs
    funding_fee_deduction: float = 0.0  # Some exchanges take a cut


@dataclass
class TradeCost:
    """Detailed cost breakdown for a trade."""

    entry_fee: float
    exit_fee: float
    entry_slippage: float
    exit_slippage: float
    holding_cost: float
    total_cost: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "entry_slippage": self.entry_slippage,
            "exit_slippage": self.exit_slippage,
            "holding_cost": self.holding_cost,
            "total_cost": self.total_cost,
        }


class CostCalculator:
    """
    Calculate realistic trading costs for funding arbitrage.

    For funding arbitrage, we have:
    - Spot leg: Buy spot BTC, later sell
    - Perp leg: Short perp, later close

    Each leg incurs fees and slippage on entry and exit.
    """

    def __init__(self, params: CostParameters = None):
        """
        Initialize cost calculator.

        Args:
            params: Cost parameters (uses defaults if None)
        """
        self.params = params or CostParameters()

        logger.info(
            "CostCalculator initialized: maker=%.2f%%, taker=%.2f%%, slippage=%.2f%%",
            self.params.maker_fee * 100,
            self.params.taker_fee * 100,
            self.params.entry_slippage * 100,
        )

    def calculate_entry_cost(
        self,
        notional_value: float,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate cost to enter a funding arbitrage position.

        For funding arb entry:
        - Spot buy: fee + slippage
        - Perp short: fee + slippage

        Args:
            notional_value: Position size in USD
            is_maker: Whether using maker orders (lower fees)

        Returns:
            Total entry cost in USD
        """
        fee_rate = self.params.maker_fee if is_maker else self.params.taker_fee

        # Both legs pay fees
        spot_fee = notional_value * fee_rate
        perp_fee = notional_value * fee_rate

        # Slippage on both legs
        spot_slippage = notional_value * self.params.entry_slippage
        perp_slippage = notional_value * self.params.entry_slippage

        total = spot_fee + perp_fee + spot_slippage + perp_slippage

        return total

    def calculate_exit_cost(
        self,
        notional_value: float,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate cost to exit a funding arbitrage position.

        For funding arb exit:
        - Spot sell: fee + slippage
        - Perp close (buy): fee + slippage

        Args:
            notional_value: Position size in USD
            is_maker: Whether using maker orders (lower fees)

        Returns:
            Total exit cost in USD
        """
        fee_rate = self.params.maker_fee if is_maker else self.params.taker_fee

        # Both legs pay fees
        spot_fee = notional_value * fee_rate
        perp_fee = notional_value * fee_rate

        # Slippage on both legs
        spot_slippage = notional_value * self.params.exit_slippage
        perp_slippage = notional_value * self.params.exit_slippage

        total = spot_fee + perp_fee + spot_slippage + perp_slippage

        return total

    def calculate_holding_cost(
        self,
        notional_value: float,
        days_held: float,
    ) -> float:
        """
        Calculate cost of holding position (borrowing costs if any).

        Args:
            notional_value: Position size in USD
            days_held: Number of days position is held

        Returns:
            Holding cost in USD
        """
        return notional_value * self.params.spot_borrow_rate_daily * days_held

    def calculate_round_trip_cost(
        self,
        notional_value: float,
        days_held: float = 0,
        is_maker: bool = False,
    ) -> TradeCost:
        """
        Calculate total cost for entering and exiting a position.

        Args:
            notional_value: Position size in USD
            days_held: Number of days position is held
            is_maker: Whether using maker orders

        Returns:
            TradeCost with detailed breakdown
        """
        fee_rate = self.params.maker_fee if is_maker else self.params.taker_fee

        # Entry costs (2 legs)
        entry_fee = notional_value * fee_rate * 2
        entry_slippage = notional_value * self.params.entry_slippage * 2

        # Exit costs (2 legs)
        exit_fee = notional_value * fee_rate * 2
        exit_slippage = notional_value * self.params.exit_slippage * 2

        # Holding costs
        holding_cost = self.calculate_holding_cost(notional_value, days_held)

        total = entry_fee + entry_slippage + exit_fee + exit_slippage + holding_cost

        return TradeCost(
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            holding_cost=holding_cost,
            total_cost=total,
        )

    def calculate_breakeven_rate(
        self,
        notional_value: float,
        expected_holding_days: float = 1.0,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate minimum funding rate needed to break even.

        This helps determine if a position is worth entering.

        Args:
            notional_value: Position size in USD
            expected_holding_days: Expected holding period
            is_maker: Whether using maker orders

        Returns:
            Minimum funding rate (as decimal) to break even
        """
        round_trip = self.calculate_round_trip_cost(
            notional_value, expected_holding_days, is_maker
        )

        # Funding periods in holding period (3 per day)
        funding_periods = expected_holding_days * 3

        # Minimum rate per period to cover costs
        if funding_periods > 0:
            breakeven_rate = round_trip.total_cost / notional_value / funding_periods
        else:
            breakeven_rate = float("inf")

        return breakeven_rate

    def estimate_net_yield(
        self,
        notional_value: float,
        funding_rate: float,
        funding_periods: int = 1,
        is_maker: bool = False,
    ) -> Dict:
        """
        Estimate net yield after costs.

        Args:
            notional_value: Position size in USD
            funding_rate: Expected funding rate per period
            funding_periods: Number of funding periods
            is_maker: Whether using maker orders

        Returns:
            Dictionary with gross yield, costs, and net yield
        """
        # Gross funding earned
        gross_funding = notional_value * funding_rate * funding_periods

        # If this is a single entry/exit cycle
        days_held = funding_periods / 3  # 3 periods per day
        costs = self.calculate_round_trip_cost(notional_value, days_held, is_maker)

        # Net yield
        net_yield = gross_funding - costs.total_cost

        return {
            "gross_funding": gross_funding,
            "total_costs": costs.total_cost,
            "cost_breakdown": costs.to_dict(),
            "net_yield": net_yield,
            "net_yield_pct": net_yield / notional_value * 100,
            "gross_yield_pct": gross_funding / notional_value * 100,
            "cost_pct": costs.total_cost / notional_value * 100,
            "profitable": net_yield > 0,
        }

    def get_cost_summary(self, notional_value: float = 10000.0) -> str:
        """
        Get a formatted summary of costs for a given notional.

        Args:
            notional_value: Position size to calculate costs for

        Returns:
            Formatted cost summary string
        """
        rt_taker = self.calculate_round_trip_cost(notional_value, is_maker=False)
        rt_maker = self.calculate_round_trip_cost(notional_value, is_maker=True)
        breakeven_taker = self.calculate_breakeven_rate(notional_value, 1.0, False)
        breakeven_maker = self.calculate_breakeven_rate(notional_value, 1.0, True)

        return f"""
Cost Summary for ${notional_value:,.2f} position:

Taker Orders (market orders):
  Entry: ${rt_taker.entry_fee + rt_taker.entry_slippage:.2f}
  Exit:  ${rt_taker.exit_fee + rt_taker.exit_slippage:.2f}
  Total: ${rt_taker.total_cost:.2f} ({rt_taker.total_cost/notional_value*100:.3f}%)
  Breakeven rate (1 day): {breakeven_taker*100:.4f}%

Maker Orders (limit orders):
  Entry: ${rt_maker.entry_fee + rt_maker.entry_slippage:.2f}
  Exit:  ${rt_maker.exit_fee + rt_maker.exit_slippage:.2f}
  Total: ${rt_maker.total_cost:.2f} ({rt_maker.total_cost/notional_value*100:.3f}%)
  Breakeven rate (1 day): {breakeven_maker*100:.4f}%

Fee Structure:
  Maker: {self.params.maker_fee*100:.2f}%
  Taker: {self.params.taker_fee*100:.2f}%
  Slippage: {self.params.entry_slippage*100:.2f}%
"""


# Pre-configured cost calculators for different scenarios
def get_binance_vip0_costs() -> CostCalculator:
    """Get cost calculator for Binance VIP 0 tier."""
    return CostCalculator(
        CostParameters(
            maker_fee=0.0002,
            taker_fee=0.0004,
            entry_slippage=0.0005,
            exit_slippage=0.0005,
        )
    )


def get_binance_vip1_costs() -> CostCalculator:
    """Get cost calculator for Binance VIP 1 tier."""
    return CostCalculator(
        CostParameters(
            maker_fee=0.00016,
            taker_fee=0.0004,
            entry_slippage=0.0003,
            exit_slippage=0.0003,
        )
    )


def get_low_slippage_costs() -> CostCalculator:
    """Get cost calculator for low slippage environment (high liquidity)."""
    return CostCalculator(
        CostParameters(
            maker_fee=0.0002,
            taker_fee=0.0004,
            entry_slippage=0.0002,
            exit_slippage=0.0002,
        )
    )


def get_conservative_costs() -> CostCalculator:
    """Get cost calculator with conservative (higher) cost estimates."""
    return CostCalculator(
        CostParameters(
            maker_fee=0.0002,
            taker_fee=0.0004,
            entry_slippage=0.001,  # 0.1% slippage
            exit_slippage=0.001,
        )
    )

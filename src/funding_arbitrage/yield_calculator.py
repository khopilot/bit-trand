"""
Yield Calculator for Funding Arbitrage

Tracks earnings, calculates APY, and provides performance metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .rate_monitor import FundingRateMonitor

logger = logging.getLogger("btc_trader.funding_arb.yield_calculator")


@dataclass
class FundingPayment:
    """Record of a single funding payment."""

    timestamp: datetime
    amount: float  # USD value of payment
    rate: float  # Funding rate at the time
    position_size: float  # BTC position size
    notional_value: float  # USD notional value


@dataclass
class YieldPeriod:
    """Yield statistics for a time period."""

    period_name: str
    start_date: datetime
    end_date: datetime
    total_funding: float
    payment_count: int
    avg_rate: float
    avg_position: float
    annualized_yield: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    total_funding_earned: float = 0.0
    total_trading_pnl: float = 0.0
    total_pnl: float = 0.0

    # Yield metrics
    current_apy: float = 0.0
    avg_funding_rate: float = 0.0
    best_rate: float = 0.0
    worst_rate: float = 0.0

    # Position metrics
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    total_duration_hours: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0


class YieldCalculator:
    """
    Calculates and tracks yield from funding arbitrage.

    Provides:
    - Real-time APY calculation
    - Historical yield analysis
    - Performance comparison vs benchmarks
    - Projections based on current rates
    """

    # Funding period is every 8 hours
    FUNDING_PERIODS_PER_DAY = 3
    FUNDING_PERIODS_PER_YEAR = 365 * 3

    def __init__(
        self,
        rate_monitor: Optional[FundingRateMonitor] = None,
        initial_capital: float = 10000.0,
    ):
        """
        Initialize yield calculator.

        Args:
            rate_monitor: Optional rate monitor for live data
            initial_capital: Starting capital in USD
        """
        self.rate_monitor = rate_monitor
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Payment tracking
        self._payments: List[FundingPayment] = []
        self._trading_pnl: List[Dict] = []

        # Position tracking for averaging
        self._position_history: List[Dict] = []

        logger.info(
            "YieldCalculator initialized: capital=$%.2f",
            initial_capital,
        )

    def record_funding_payment(
        self,
        amount: float,
        rate: float,
        position_size: float,
        notional_value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a funding payment.

        Args:
            amount: Payment amount in USD
            rate: Funding rate (decimal)
            position_size: Position size in BTC
            notional_value: Position value in USD
            timestamp: Payment time (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        payment = FundingPayment(
            timestamp=timestamp,
            amount=amount,
            rate=rate,
            position_size=position_size,
            notional_value=notional_value,
        )

        self._payments.append(payment)
        self.current_capital += amount

        logger.debug(
            "Funding payment recorded: $%.4f at %.4f%% rate",
            amount,
            rate * 100,
        )

    def record_trading_pnl(self, pnl: float, reason: str = "") -> None:
        """
        Record trading P&L (from opening/closing positions).

        Args:
            pnl: P&L amount in USD
            reason: Description of the P&L source
        """
        self._trading_pnl.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "amount": pnl,
            "reason": reason,
        })
        self.current_capital += pnl

    def record_position(self, position_size: float, notional_value: float) -> None:
        """Record position for averaging."""
        self._position_history.append({
            "timestamp": datetime.now(timezone.utc),
            "size": position_size,
            "notional": notional_value,
        })

    def get_total_funding(self) -> float:
        """Get total funding earned."""
        return sum(p.amount for p in self._payments)

    def get_total_trading_pnl(self) -> float:
        """Get total trading P&L."""
        return sum(t["amount"] for t in self._trading_pnl)

    def get_total_pnl(self) -> float:
        """Get total P&L (funding + trading)."""
        return self.get_total_funding() + self.get_total_trading_pnl()

    def get_return_pct(self) -> float:
        """Get total return as percentage."""
        if self.initial_capital <= 0:
            return 0.0
        return (self.current_capital - self.initial_capital) / self.initial_capital * 100

    def calculate_apy(self, lookback_days: int = 30) -> float:
        """
        Calculate APY based on recent performance.

        Args:
            lookback_days: Days to look back for calculation

        Returns:
            Annualized percentage yield
        """
        if not self._payments:
            return 0.0

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        recent_payments = [p for p in self._payments if p.timestamp >= cutoff]

        if not recent_payments:
            return 0.0

        # Sum funding earned
        total_funding = sum(p.amount for p in recent_payments)

        # Average notional value
        avg_notional = sum(p.notional_value for p in recent_payments) / len(recent_payments)

        if avg_notional <= 0:
            return 0.0

        # Calculate daily yield
        daily_yield = total_funding / lookback_days / avg_notional

        # Annualize
        apy = daily_yield * 365 * 100

        return apy

    def calculate_projected_yield(
        self,
        position_size_btc: float,
        current_rate: Optional[float] = None,
        days: int = 30,
    ) -> Dict:
        """
        Project future yield based on current rate.

        Args:
            position_size_btc: Position size in BTC
            current_rate: Current funding rate (fetches if None)
            days: Number of days to project

        Returns:
            Projection details
        """
        if current_rate is None and self.rate_monitor:
            rate_data = self.rate_monitor.get_binance_funding_rate()
            current_rate = rate_data.rate if rate_data else 0.0
        elif current_rate is None:
            current_rate = 0.0

        # Get current BTC price for notional calculation
        btc_price = 95000  # Default fallback
        if self.rate_monitor:
            from .exchange_client import BinanceClient
            client = BinanceClient(testnet=False)
            price = client.get_spot_price("BTCUSDT")
            if price:
                btc_price = price

        notional_value = position_size_btc * btc_price

        # Calculate projections
        funding_per_period = notional_value * current_rate
        periods_in_projection = days * self.FUNDING_PERIODS_PER_DAY
        total_projected = funding_per_period * periods_in_projection

        daily_yield = funding_per_period * self.FUNDING_PERIODS_PER_DAY
        monthly_yield = daily_yield * 30
        yearly_yield = daily_yield * 365

        apy = (daily_yield / notional_value * 365 * 100) if notional_value > 0 else 0

        return {
            "position_btc": position_size_btc,
            "notional_usd": notional_value,
            "current_rate": current_rate,
            "current_rate_pct": current_rate * 100,
            "projection_days": days,
            "projected_yield": total_projected,
            "daily_yield": daily_yield,
            "monthly_yield": monthly_yield,
            "yearly_yield": yearly_yield,
            "projected_apy": apy,
        }

    def get_yield_by_period(self, period: str = "daily") -> List[YieldPeriod]:
        """
        Get yield breakdown by time period.

        Args:
            period: "daily", "weekly", or "monthly"

        Returns:
            List of YieldPeriod objects
        """
        if not self._payments:
            return []

        # Determine period duration
        if period == "daily":
            delta = timedelta(days=1)
        elif period == "weekly":
            delta = timedelta(weeks=1)
        elif period == "monthly":
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)

        # Group payments by period
        periods: List[YieldPeriod] = []
        if not self._payments:
            return periods

        # Sort payments by timestamp
        sorted_payments = sorted(self._payments, key=lambda p: p.timestamp)
        start_date = sorted_payments[0].timestamp
        end_date = datetime.now(timezone.utc)

        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + delta, end_date)

            # Get payments in this period
            period_payments = [
                p for p in sorted_payments
                if current_start <= p.timestamp < current_end
            ]

            if period_payments:
                total_funding = sum(p.amount for p in period_payments)
                avg_rate = sum(p.rate for p in period_payments) / len(period_payments)
                avg_position = sum(p.notional_value for p in period_payments) / len(period_payments)

                # Calculate annualized yield for this period
                period_days = (current_end - current_start).days or 1
                daily_yield = total_funding / period_days
                annualized = (daily_yield / avg_position * 365 * 100) if avg_position > 0 else 0

                periods.append(YieldPeriod(
                    period_name=current_start.strftime("%Y-%m-%d"),
                    start_date=current_start,
                    end_date=current_end,
                    total_funding=total_funding,
                    payment_count=len(period_payments),
                    avg_rate=avg_rate,
                    avg_position=avg_position,
                    annualized_yield=annualized,
                ))

            current_start = current_end

        return periods

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        metrics = PerformanceMetrics()

        # Funding metrics
        metrics.total_funding_earned = self.get_total_funding()
        metrics.total_trading_pnl = self.get_total_trading_pnl()
        metrics.total_pnl = self.get_total_pnl()

        if self._payments:
            rates = [p.rate for p in self._payments]
            metrics.avg_funding_rate = sum(rates) / len(rates)
            metrics.best_rate = max(rates)
            metrics.worst_rate = min(rates)

            positions = [p.notional_value for p in self._payments]
            metrics.avg_position_size = sum(positions) / len(positions)
            metrics.max_position_size = max(positions)

        # Calculate current APY
        metrics.current_apy = self.calculate_apy(30)

        # Duration (if we have position history)
        if self._position_history:
            first = self._position_history[0]["timestamp"]
            last = datetime.now(timezone.utc)
            metrics.total_duration_hours = (last - first).total_seconds() / 3600

        # Calculate drawdown
        if self._payments:
            running_pnl = 0.0
            peak = 0.0
            max_dd = 0.0
            for payment in self._payments:
                running_pnl += payment.amount
                peak = max(peak, running_pnl)
                drawdown = peak - running_pnl
                max_dd = max(max_dd, drawdown)
            metrics.max_drawdown = max_dd

        return metrics

    def get_summary(self) -> Dict:
        """Get a formatted summary of yield performance."""
        metrics = self.get_performance_metrics()

        # Recent period analysis
        recent_apy = self.calculate_apy(7)  # Last week
        monthly_apy = self.calculate_apy(30)  # Last month

        return {
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "return_pct": self.get_return_pct(),
            },
            "funding": {
                "total_earned": metrics.total_funding_earned,
                "payment_count": len(self._payments),
                "avg_rate_pct": metrics.avg_funding_rate * 100,
                "best_rate_pct": metrics.best_rate * 100,
                "worst_rate_pct": metrics.worst_rate * 100,
            },
            "yield": {
                "current_apy": metrics.current_apy,
                "7d_apy": recent_apy,
                "30d_apy": monthly_apy,
            },
            "position": {
                "avg_size_usd": metrics.avg_position_size,
                "max_size_usd": metrics.max_position_size,
                "duration_hours": metrics.total_duration_hours,
            },
            "risk": {
                "max_drawdown": metrics.max_drawdown,
                "trading_pnl": metrics.total_trading_pnl,
            },
            "total_pnl": metrics.total_pnl,
        }

    def compare_to_benchmark(
        self,
        btc_start_price: float,
        btc_end_price: float,
    ) -> Dict:
        """
        Compare funding arb performance to buy-and-hold BTC.

        Args:
            btc_start_price: BTC price at strategy start
            btc_end_price: BTC price now

        Returns:
            Comparison metrics
        """
        # Our strategy return
        our_return_pct = self.get_return_pct()

        # BTC buy-and-hold return
        btc_return_pct = ((btc_end_price - btc_start_price) / btc_start_price) * 100

        # Alpha (our return - benchmark return)
        alpha = our_return_pct - btc_return_pct

        return {
            "strategy_return_pct": our_return_pct,
            "btc_return_pct": btc_return_pct,
            "alpha": alpha,
            "outperformed": our_return_pct > btc_return_pct,
            "message": (
                f"Strategy: {our_return_pct:.2f}%, BTC: {btc_return_pct:.2f}%, "
                f"Alpha: {alpha:+.2f}%"
            ),
        }

    def backtest_historical_rates(
        self,
        historical_rates: List[Dict],
        position_size_btc: float = 0.1,
        btc_price: float = 95000,
    ) -> Dict:
        """
        Backtest strategy on historical funding rates.

        Args:
            historical_rates: List of {timestamp, rate} dicts
            position_size_btc: Simulated position size
            btc_price: Average BTC price for simulation

        Returns:
            Backtest results
        """
        if not historical_rates:
            return {"error": "No historical rates provided"}

        notional = position_size_btc * btc_price
        total_funding = 0.0
        payments = []

        for rate_data in historical_rates:
            rate = rate_data.get("rate", 0)
            payment = notional * rate

            if rate > 0:  # Only count positive funding
                total_funding += payment
                payments.append({
                    "timestamp": rate_data.get("timestamp"),
                    "rate": rate,
                    "payment": payment,
                })

        # Calculate statistics
        num_periods = len(historical_rates)
        positive_periods = len(payments)
        positive_pct = (positive_periods / num_periods * 100) if num_periods > 0 else 0

        # Estimate duration in days
        days = num_periods / self.FUNDING_PERIODS_PER_DAY

        # Calculate APY
        daily_yield = total_funding / days if days > 0 else 0
        apy = (daily_yield / notional * 365 * 100) if notional > 0 else 0

        return {
            "position_btc": position_size_btc,
            "notional_usd": notional,
            "periods_analyzed": num_periods,
            "positive_periods": positive_periods,
            "positive_rate_pct": positive_pct,
            "days_simulated": days,
            "total_funding": total_funding,
            "avg_payment": total_funding / positive_periods if positive_periods > 0 else 0,
            "daily_yield": daily_yield,
            "monthly_yield": daily_yield * 30,
            "yearly_yield": daily_yield * 365,
            "simulated_apy": apy,
        }

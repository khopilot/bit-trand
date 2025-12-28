#!/usr/bin/env python3
"""
Comprehensive Funding Arbitrage Backtest

Multi-year backtest using ALL historical funding rate data from Binance
(September 2019 to present). Includes realistic cost modeling and
year-by-year performance analysis.

Usage:
    python comprehensive_backtest.py --fetch-all
    python comprehensive_backtest.py --use-cache
    python comprehensive_backtest.py --start 2020-01-01 --end 2024-12-31
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.funding_arbitrage.funding_rate_fetcher import (
    FundingRateFetcher,
    FundingRateRecord,
)
from src.funding_arbitrage.cost_model import CostCalculator, CostParameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class YearlyMetrics:
    """Metrics for a single year."""

    year: int
    total_funding_earned: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    apy: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    funding_periods: int = 0
    positive_periods: int = 0
    negative_periods: int = 0
    avg_positive_rate: float = 0.0
    avg_negative_rate: float = 0.0
    best_rate: float = 0.0
    worst_rate: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "year": self.year,
            "total_funding_earned": self.total_funding_earned,
            "total_costs": self.total_costs,
            "net_pnl": self.net_pnl,
            "apy": self.apy,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "funding_periods": self.funding_periods,
            "positive_periods": self.positive_periods,
            "negative_periods": self.negative_periods,
            "avg_positive_rate": self.avg_positive_rate,
            "avg_negative_rate": self.avg_negative_rate,
            "best_rate": self.best_rate,
            "worst_rate": self.worst_rate,
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""

    # Overall metrics
    total_funding_earned: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    overall_apy: float = 0.0
    overall_sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # Configuration
    initial_capital: float = 0.0
    final_capital: float = 0.0
    position_pct: float = 0.0

    # Data coverage
    start_date: str = ""
    end_date: str = ""
    total_periods: int = 0
    days_covered: int = 0

    # Year-by-year breakdown
    yearly_metrics: List[YearlyMetrics] = field(default_factory=list)

    # Market period analysis
    market_periods: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_funding_earned": self.total_funding_earned,
            "total_costs": self.total_costs,
            "net_pnl": self.net_pnl,
            "overall_apy": self.overall_apy,
            "overall_sharpe": self.overall_sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "position_pct": self.position_pct,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_periods": self.total_periods,
            "days_covered": self.days_covered,
            "yearly_metrics": [y.to_dict() for y in self.yearly_metrics],
            "market_periods": self.market_periods,
        }


class ComprehensiveBacktest:
    """
    Multi-year funding arbitrage backtest.

    Simulates the funding arbitrage strategy with:
    - Entry when funding rate > threshold
    - Exit when funding rate < 0 or duration limit
    - Realistic cost deduction
    - Year-by-year performance tracking
    """

    # Market period definitions
    MARKET_PERIODS = {
        "2019-2020 Early Era": (
            datetime(2019, 9, 1, tzinfo=timezone.utc),
            datetime(2020, 3, 31, tzinfo=timezone.utc),
        ),
        "2020-2021 Bull Run": (
            datetime(2020, 4, 1, tzinfo=timezone.utc),
            datetime(2021, 4, 30, tzinfo=timezone.utc),
        ),
        "2021-2022 Peak/Crash": (
            datetime(2021, 5, 1, tzinfo=timezone.utc),
            datetime(2022, 6, 30, tzinfo=timezone.utc),
        ),
        "2022-2023 Bear Market": (
            datetime(2022, 7, 1, tzinfo=timezone.utc),
            datetime(2023, 3, 31, tzinfo=timezone.utc),
        ),
        "2023-2024 Recovery": (
            datetime(2023, 4, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        ),
        "2024-2025 ETF Era": (
            datetime(2024, 4, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, tzinfo=timezone.utc),
        ),
    }

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_pct: float = 0.30,
        cost_params: CostParameters = None,
        entry_threshold: float = 0.0005,  # 0.05% min rate to enter
        exit_threshold: float = -0.0001,  # Exit on negative rate
    ):
        """
        Initialize backtest.

        Args:
            initial_capital: Starting capital in USD
            position_pct: Percentage of capital to allocate
            cost_params: Trading cost parameters
            entry_threshold: Minimum funding rate to enter position
            exit_threshold: Funding rate threshold to exit position
        """
        self.initial_capital = initial_capital
        self.position_pct = position_pct
        self.cost_calculator = CostCalculator(cost_params)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        logger.info(
            "ComprehensiveBacktest initialized: capital=$%.2f, position=%.0f%%, entry=%.4f%%",
            initial_capital,
            position_pct * 100,
            entry_threshold * 100,
        )

    def run_backtest(
        self,
        funding_rates: List[FundingRateRecord],
    ) -> BacktestResult:
        """
        Run comprehensive backtest on funding rate data.

        Args:
            funding_rates: List of historical funding rate records

        Returns:
            BacktestResult with all metrics
        """
        if not funding_rates:
            logger.warning("No funding rates provided")
            return BacktestResult(initial_capital=self.initial_capital)

        # Sort by timestamp
        sorted_rates = sorted(funding_rates, key=lambda x: x.timestamp)

        # Initialize result
        result = BacktestResult(
            initial_capital=self.initial_capital,
            position_pct=self.position_pct,
            start_date=sorted_rates[0].timestamp.isoformat(),
            end_date=sorted_rates[-1].timestamp.isoformat(),
            total_periods=len(sorted_rates),
            days_covered=(sorted_rates[-1].timestamp - sorted_rates[0].timestamp).days,
        )

        # Calculate position notional
        position_capital = self.initial_capital * self.position_pct

        # Simulate strategy
        simulation = self._simulate_strategy(sorted_rates, position_capital)

        # Populate result
        result.total_funding_earned = simulation["total_funding"]
        result.total_costs = simulation["total_costs"]
        result.net_pnl = simulation["net_pnl"]
        result.final_capital = self.initial_capital + simulation["net_pnl"]
        result.win_rate = simulation["win_rate"]
        result.max_drawdown = simulation["max_drawdown"]
        result.overall_sharpe = simulation["sharpe_ratio"]

        # Calculate overall APY
        years = result.days_covered / 365
        if years > 0 and self.initial_capital > 0:
            total_return = result.net_pnl / (self.initial_capital * self.position_pct)
            result.overall_apy = (total_return / years) * 100

        # Year-by-year analysis
        result.yearly_metrics = self._analyze_by_year(sorted_rates, position_capital)

        # Market period analysis
        result.market_periods = self._analyze_market_periods(
            sorted_rates, position_capital
        )

        return result

    def _simulate_strategy(
        self,
        rates: List[FundingRateRecord],
        position_capital: float,
    ) -> Dict:
        """
        Simulate the funding arbitrage strategy.

        Strategy:
        - Enter when rate > entry_threshold
        - Exit when rate < exit_threshold
        - Collect funding each period while in position
        """
        total_funding = 0.0
        total_costs = 0.0
        in_position = False
        position_entry_time = None
        equity_curve = []
        period_returns = []

        current_equity = 0.0

        for rate in rates:
            funding_amount = 0.0

            if not in_position:
                # Check entry
                if rate.funding_rate >= self.entry_threshold:
                    # Enter position
                    in_position = True
                    position_entry_time = rate.timestamp
                    entry_cost = self.cost_calculator.calculate_entry_cost(
                        position_capital
                    )
                    total_costs += entry_cost
                    current_equity -= entry_cost
            else:
                # In position - collect funding
                funding_amount = position_capital * rate.funding_rate
                total_funding += funding_amount
                current_equity += funding_amount

                # Check exit
                if rate.funding_rate < self.exit_threshold:
                    # Exit position
                    exit_cost = self.cost_calculator.calculate_exit_cost(
                        position_capital
                    )
                    total_costs += exit_cost
                    current_equity -= exit_cost
                    in_position = False

            equity_curve.append(current_equity)
            period_returns.append(funding_amount / position_capital if position_capital > 0 else 0)

        # Close any remaining position
        if in_position:
            exit_cost = self.cost_calculator.calculate_exit_cost(position_capital)
            total_costs += exit_cost
            current_equity -= exit_cost

        net_pnl = total_funding - total_costs

        # Calculate metrics
        win_rate = sum(1 for r in rates if r.funding_rate > 0) / len(rates) if rates else 0
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(period_returns)

        return {
            "total_funding": total_funding,
            "total_costs": total_costs,
            "net_pnl": net_pnl,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": equity_curve,
        }

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0

        equity = np.array(equity_curve)
        cummax = np.maximum.accumulate(equity)

        # Handle case where cummax is all zeros or negative
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(cummax > 0, (cummax - equity) / cummax, 0)

        return float(np.nanmax(drawdown)) if len(drawdown) > 0 else 0.0

    def _calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.05,
        periods_per_year: int = 1095,  # 3 per day * 365
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: List of period returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Annualized Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)
        rf_per_period = risk_free_rate / periods_per_year

        sharpe = (annualized_return - risk_free_rate) / annualized_std

        return float(sharpe)

    def _analyze_by_year(
        self,
        rates: List[FundingRateRecord],
        position_capital: float,
    ) -> List[YearlyMetrics]:
        """Break down results by calendar year."""
        # Group rates by year
        rates_by_year: Dict[int, List[FundingRateRecord]] = {}
        for rate in rates:
            year = rate.timestamp.year
            if year not in rates_by_year:
                rates_by_year[year] = []
            rates_by_year[year].append(rate)

        yearly_metrics = []

        for year in sorted(rates_by_year.keys()):
            year_rates = rates_by_year[year]

            # Simulate for this year
            sim = self._simulate_strategy(year_rates, position_capital)

            # Calculate metrics
            funding_rates = [r.funding_rate for r in year_rates]
            positive_rates = [r for r in funding_rates if r > 0]
            negative_rates = [r for r in funding_rates if r < 0]

            # APY calculation
            days_in_year = len(year_rates) / 3  # 3 periods per day
            if days_in_year > 0 and position_capital > 0:
                year_return = sim["net_pnl"] / position_capital
                apy = (year_return / days_in_year) * 365 * 100
            else:
                apy = 0

            metrics = YearlyMetrics(
                year=year,
                total_funding_earned=sim["total_funding"],
                total_costs=sim["total_costs"],
                net_pnl=sim["net_pnl"],
                apy=apy,
                max_drawdown=sim["max_drawdown"],
                sharpe_ratio=sim["sharpe_ratio"],
                win_rate=sim["win_rate"],
                funding_periods=len(year_rates),
                positive_periods=len(positive_rates),
                negative_periods=len(negative_rates),
                avg_positive_rate=np.mean(positive_rates) if positive_rates else 0,
                avg_negative_rate=np.mean(negative_rates) if negative_rates else 0,
                best_rate=max(funding_rates) if funding_rates else 0,
                worst_rate=min(funding_rates) if funding_rates else 0,
            )

            yearly_metrics.append(metrics)

        return yearly_metrics

    def _analyze_market_periods(
        self,
        rates: List[FundingRateRecord],
        position_capital: float,
    ) -> Dict:
        """Analyze performance by market period."""
        period_results = {}

        for period_name, (start, end) in self.MARKET_PERIODS.items():
            # Filter rates for this period
            period_rates = [
                r for r in rates if start <= r.timestamp <= end
            ]

            if not period_rates:
                continue

            # Simulate
            sim = self._simulate_strategy(period_rates, position_capital)

            # Calculate APY
            days = len(period_rates) / 3
            if days > 0 and position_capital > 0:
                period_return = sim["net_pnl"] / position_capital
                apy = (period_return / days) * 365 * 100
            else:
                apy = 0

            period_results[period_name] = {
                "periods": len(period_rates),
                "days": days,
                "total_funding": sim["total_funding"],
                "total_costs": sim["total_costs"],
                "net_pnl": sim["net_pnl"],
                "apy": apy,
                "win_rate": sim["win_rate"],
                "max_drawdown": sim["max_drawdown"],
            }

        return period_results

    def generate_report(
        self,
        result: BacktestResult,
        output_format: str = "console",
    ) -> str:
        """
        Generate formatted backtest report.

        Args:
            result: BacktestResult object
            output_format: "console", "markdown", or "json"

        Returns:
            Formatted report string
        """
        if output_format == "json":
            return json.dumps(result.to_dict(), indent=2, default=str)

        # Console/Markdown format
        separator = "=" * 64

        lines = [
            separator,
            "         COMPREHENSIVE FUNDING ARBITRAGE BACKTEST REPORT",
            separator,
            "",
            "DATA COVERAGE",
            f"   Period: {result.start_date[:10]} -> {result.end_date[:10]}",
            f"   Funding Periods: {result.total_periods:,}",
            f"   Days: {result.days_covered:,}",
            "",
            "OVERALL PERFORMANCE",
            f"   Initial Capital: ${result.initial_capital:,.2f}",
            f"   Position Allocation: {result.position_pct*100:.0f}%",
            f"   Final Value: ${result.final_capital:,.2f}",
            f"   Total Return: {(result.final_capital/result.initial_capital - 1)*100:+.1f}%",
            f"   Net Funding Earned: ${result.total_funding_earned:,.2f}",
            f"   Total Costs: ${result.total_costs:,.2f}",
            f"   Net P&L: ${result.net_pnl:,.2f}",
            "",
            "RISK METRICS",
            f"   Sharpe Ratio: {result.overall_sharpe:.2f}",
            f"   Max Drawdown: {result.max_drawdown*100:.1f}%",
            f"   Win Rate: {result.win_rate*100:.1f}%",
            f"   Overall APY: {result.overall_apy:.1f}%",
            "",
        ]

        # Year-by-year table
        if result.yearly_metrics:
            lines.extend([
                "YEAR-BY-YEAR BREAKDOWN",
                "-" * 64,
                f"{'Year':<6} {'APY':>10} {'Funding':>12} {'Costs':>10} {'Net P&L':>12} {'Win%':>8}",
                "-" * 64,
            ])

            for ym in result.yearly_metrics:
                lines.append(
                    f"{ym.year:<6} {ym.apy:>9.1f}% ${ym.total_funding_earned:>10,.0f} "
                    f"${ym.total_costs:>8,.0f} ${ym.net_pnl:>10,.0f} {ym.win_rate*100:>7.1f}%"
                )

            lines.append("-" * 64)
            lines.append("")

        # Market periods
        if result.market_periods:
            lines.extend([
                "MARKET PERIOD ANALYSIS",
                "-" * 64,
            ])

            for period, data in result.market_periods.items():
                lines.append(
                    f"{period}: APY={data['apy']:.1f}%, "
                    f"Net=${data['net_pnl']:,.0f}, "
                    f"Win={data['win_rate']*100:.0f}%"
                )

            lines.append("")

        lines.append(separator)

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive funding arbitrage backtest"
    )
    parser.add_argument(
        "--fetch-all",
        action="store_true",
        help="Fetch all historical data from Binance",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data if available",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital (default: 10000)",
    )
    parser.add_argument(
        "--position-pct",
        type=float,
        default=0.30,
        help="Position allocation (default: 0.30)",
    )
    parser.add_argument(
        "--format",
        choices=["console", "markdown", "json"],
        default="console",
        help="Output format (default: console)",
    )

    args = parser.parse_args()

    # Initialize fetcher
    fetcher = FundingRateFetcher()
    cache_path = "data/funding_rates/btcusdt_funding_history.csv"

    # Determine data source
    if args.use_cache:
        logger.info("Loading from cache...")
        records = fetcher.get_cached_or_fetch(cache_path)
    elif args.fetch_all:
        logger.info("Fetching ALL historical data from Binance...")

        def progress(fetched, total):
            print(f"\rFetched {fetched:,} / ~{total:,} records", end="", flush=True)

        records = fetcher.fetch_all_history(progress_callback=progress)
        print()  # Newline after progress
        fetcher.save_to_csv(records, cache_path)
    else:
        # Default: use cache or fetch
        records = fetcher.get_cached_or_fetch(cache_path)

    # Filter by date range if specified
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        records = [r for r in records if r.timestamp >= start_date]

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        records = [r for r in records if r.timestamp <= end_date]

    if not records:
        logger.error("No funding rate data available")
        return

    # Print statistics
    stats = fetcher.get_statistics(records)
    logger.info(
        "Data: %d periods, %s to %s, %.1f%% positive",
        stats["total_records"],
        stats["first_date"][:10],
        stats["last_date"][:10],
        stats["positive_pct"],
    )

    # Run backtest
    backtest = ComprehensiveBacktest(
        initial_capital=args.capital,
        position_pct=args.position_pct,
    )

    result = backtest.run_backtest(records)

    # Generate report
    report = backtest.generate_report(result, args.format)
    print(report)

    # Save JSON results
    if args.format != "json":
        json_path = "data/funding_rates/backtest_results.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Results saved to %s", json_path)


if __name__ == "__main__":
    main()

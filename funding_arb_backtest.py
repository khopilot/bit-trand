#!/usr/bin/env python3
"""
Funding Arbitrage Backtest

Backtests the funding rate arbitrage strategy using historical data
from Binance perpetual futures.

Usage:
    python funding_arb_backtest.py [--days 365] [--capital 10000] [--position-pct 0.3]
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.funding_arbitrage import (
    FundingRateMonitor,
    YieldCalculator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_backtest(
    days: int = 365,
    initial_capital: float = 10000.0,
    position_pct: float = 0.30,
    min_rate: float = 0.0005,
) -> dict:
    """
    Run funding arbitrage backtest.

    Args:
        days: Number of days to backtest
        initial_capital: Starting capital in USD
        position_pct: Percentage of capital to allocate
        min_rate: Minimum funding rate to enter position

    Returns:
        Backtest results dictionary
    """
    logger.info("=" * 60)
    logger.info("FUNDING ARBITRAGE BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Period: {days} days")
    logger.info(f"Capital: ${initial_capital:,.2f}")
    logger.info(f"Position allocation: {position_pct*100:.0f}%")
    logger.info(f"Min entry rate: {min_rate*100:.4f}%")
    logger.info("-" * 60)

    # Initialize components
    rate_monitor = FundingRateMonitor(
        min_funding_rate=min_rate,
        entry_threshold=min_rate * 2,  # Enter at 2x min rate
    )

    calculator = YieldCalculator(initial_capital=initial_capital)

    # Fetch historical funding rates
    logger.info("Fetching historical funding rates from Binance...")
    limit = min(days * 3, 1000)  # 3 funding periods per day, max 1000
    historical_rates = rate_monitor.get_historical_funding_rates(
        exchange="binance",
        symbol="BTCUSDT",
        limit=limit,
    )

    if not historical_rates:
        logger.error("Failed to fetch historical rates")
        return {"error": "No historical data"}

    logger.info(f"Fetched {len(historical_rates)} funding rate records")

    # Analyze funding rate distribution
    rates = [r["rate"] for r in historical_rates]
    positive_rates = [r for r in rates if r > 0]
    negative_rates = [r for r in rates if r < 0]

    logger.info("-" * 60)
    logger.info("FUNDING RATE ANALYSIS")
    logger.info("-" * 60)
    logger.info(f"Total periods: {len(rates)}")
    logger.info(f"Positive: {len(positive_rates)} ({len(positive_rates)/len(rates)*100:.1f}%)")
    logger.info(f"Negative: {len(negative_rates)} ({len(negative_rates)/len(rates)*100:.1f}%)")
    logger.info(f"Avg rate: {sum(rates)/len(rates)*100:.4f}%")
    logger.info(f"Max rate: {max(rates)*100:.4f}%")
    logger.info(f"Min rate: {min(rates)*100:.4f}%")

    # Simulate strategy
    position_capital = initial_capital * position_pct
    btc_price = 95000  # Approximate average price

    # Run backtest using YieldCalculator
    backtest = calculator.backtest_historical_rates(
        historical_rates=historical_rates,
        position_size_btc=position_capital / btc_price,
        btc_price=btc_price,
    )

    logger.info("-" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("-" * 60)
    logger.info(f"Position: {backtest['position_btc']:.4f} BTC (${backtest['notional_usd']:,.2f})")
    logger.info(f"Days simulated: {backtest['days_simulated']:.1f}")
    logger.info(f"Positive funding periods: {backtest['positive_rate_pct']:.1f}%")
    logger.info(f"Total funding earned: ${backtest['total_funding']:,.2f}")
    logger.info(f"Daily yield: ${backtest['daily_yield']:.2f}")
    logger.info(f"Monthly yield: ${backtest['monthly_yield']:.2f}")
    logger.info(f"Yearly yield (projected): ${backtest['yearly_yield']:,.2f}")
    logger.info(f"Simulated APY: {backtest['simulated_apy']:.2f}%")

    # Calculate ROI
    total_return = backtest["total_funding"]
    roi = total_return / position_capital * 100

    logger.info("-" * 60)
    logger.info("PERFORMANCE METRICS")
    logger.info("-" * 60)
    logger.info(f"Capital allocated: ${position_capital:,.2f}")
    logger.info(f"Total return: ${total_return:,.2f}")
    logger.info(f"ROI: {roi:.2f}%")

    # Compare to buy-and-hold
    # Simulate what BTC did during this period
    if len(historical_rates) >= 2:
        first_price = historical_rates[0].get("mark_price", btc_price)
        last_price = historical_rates[-1].get("mark_price", btc_price)
        if first_price > 0 and last_price > 0:
            btc_return_pct = (last_price - first_price) / first_price * 100
        else:
            btc_return_pct = 0
    else:
        btc_return_pct = 0

    logger.info("-" * 60)
    logger.info("COMPARISON TO BUY-AND-HOLD")
    logger.info("-" * 60)
    logger.info(f"Funding Arb ROI: {roi:.2f}%")
    logger.info(f"BTC Price ROI: {btc_return_pct:.2f}%")
    logger.info(f"Alpha (vs hold): {roi - btc_return_pct:.2f}%")

    # Strategy advantages
    logger.info("-" * 60)
    logger.info("STRATEGY CHARACTERISTICS")
    logger.info("-" * 60)
    logger.info("- Market-neutral: No directional BTC exposure")
    logger.info("- Consistent yield: Earn funding even in sideways markets")
    logger.info("- Lower volatility: P&L smoother than spot holding")
    logger.info("- Capital efficiency: Can use leverage on perp side")

    # Risks
    logger.info("-" * 60)
    logger.info("RISK FACTORS")
    logger.info("-" * 60)
    logger.info("- Negative funding: Rate can go negative (shorts pay longs)")
    logger.info("- Exchange risk: Counterparty risk on futures exchange")
    logger.info("- Liquidation: Perp position can get liquidated if underfunded")
    logger.info("- Basis risk: Spot/perp prices can diverge temporarily")

    logger.info("=" * 60)

    return {
        "days": days,
        "capital": initial_capital,
        "position_capital": position_capital,
        "periods_analyzed": len(historical_rates),
        "positive_rate_pct": backtest["positive_rate_pct"],
        "total_funding": backtest["total_funding"],
        "daily_yield": backtest["daily_yield"],
        "monthly_yield": backtest["monthly_yield"],
        "yearly_yield": backtest["yearly_yield"],
        "simulated_apy": backtest["simulated_apy"],
        "roi_pct": roi,
        "btc_return_pct": btc_return_pct,
        "alpha": roi - btc_return_pct,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest funding rate arbitrage strategy"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to backtest (default: 365)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital in USD (default: 10000)",
    )
    parser.add_argument(
        "--position-pct",
        type=float,
        default=0.30,
        help="Position allocation percentage (default: 0.30)",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=0.0005,
        help="Minimum funding rate to enter (default: 0.0005 = 0.05%%)",
    )

    args = parser.parse_args()

    results = run_backtest(
        days=args.days,
        initial_capital=args.capital,
        position_pct=args.position_pct,
        min_rate=args.min_rate,
    )

    # Print summary for copy/paste
    print("\n" + "=" * 60)
    print("SUMMARY (copy/paste friendly)")
    print("=" * 60)
    print(f"Simulated APY: {results.get('simulated_apy', 0):.1f}%")
    print(f"Daily yield: ${results.get('daily_yield', 0):.2f}")
    print(f"Monthly yield: ${results.get('monthly_yield', 0):.2f}")
    print(f"Alpha vs Hold: {results.get('alpha', 0):+.1f}%")


if __name__ == "__main__":
    main()

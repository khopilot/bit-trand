#!/usr/bin/env python3
"""
Run BTC Elite Trader in Paper Trading mode.

Paper trading simulates all trades without interacting with exchange.
Good for testing strategy without risking real funds.

Usage:
    python scripts/run_paper.py
    python scripts/run_paper.py --backtest      # Run backtest instead
    python scripts/run_paper.py --backtest 365  # 365-day backtest

Author: khopilot
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import Orchestrator


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def run_paper_trading(config_path: str):
    """Run paper trading mode."""
    orchestrator = Orchestrator(config_path=config_path, mode="paper")

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        asyncio.create_task(orchestrator.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await orchestrator.stop()


async def run_backtest(config_path: str, days: int):
    """Run backtest mode."""
    orchestrator = Orchestrator(config_path=config_path, mode="paper")

    print(f"\n{'='*50}")
    print(f"PAPER TRADING BACKTEST ({days} Days)")
    print("="*50)

    results = await orchestrator.run_backtest(days=days)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"\nInitial Capital:  ${results['initial_capital']:,.2f}")
    print(f"Final Value:      ${results['final_value']:,.2f}")
    print(f"ROI:              {results['roi']:+.2f}%")
    print(f"Max Drawdown:     {results['max_drawdown']:.2f}%")
    print(f"Total Trades:     {results['total_trades']}")

    print(f"\n{'-'*50}")
    print("Last 10 Trades:")
    for trade in results.get("ledger", []):
        print(f"  {trade}")

    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run BTC Elite Trader in Paper Trading mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--backtest",
        nargs="?",
        const=365,
        type=int,
        help="Run backtest instead of live paper trading (default: 365 days)",
    )

    args = parser.parse_args()

    setup_logging()

    if args.backtest:
        asyncio.run(run_backtest(args.config, args.backtest))
    else:
        print("\n" + "="*50)
        print("BTC ELITE TRADER - PAPER TRADING MODE")
        print("="*50)
        print("\nStarting paper trading...")
        print("Press Ctrl+C to stop.\n")
        asyncio.run(run_paper_trading(args.config))


if __name__ == "__main__":
    main()

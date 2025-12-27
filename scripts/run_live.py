#!/usr/bin/env python3
"""
Run BTC Elite Trader in Live Trading mode.

WARNING: This script trades with REAL FUNDS!
Make sure you understand the risks before running.

Required environment variables:
    EXCHANGE_API_KEY      - Binance API key
    EXCHANGE_API_SECRET   - Binance API secret
    TELEGRAM_BOT_TOKEN    - Telegram bot token (optional)
    TELEGRAM_CHAT_ID      - Telegram chat ID (optional)
    DATABASE_URL          - PostgreSQL URL (optional)

Usage:
    python scripts/run_live.py --testnet    # Testnet mode (recommended first)
    python scripts/run_live.py              # LIVE mode (real funds)

Author: khopilot
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import Orchestrator


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_prerequisites(mode: str) -> bool:
    """Check required environment variables and confirm live trading."""
    api_key = os.getenv("EXCHANGE_API_KEY")
    api_secret = os.getenv("EXCHANGE_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: Exchange API credentials not set!")
        print("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET environment variables.")
        return False

    if mode == "live":
        print("\n" + "!"*60)
        print("!!! WARNING: LIVE TRADING MODE !!!")
        print("!"*60)
        print("\nThis will trade with REAL FUNDS on the exchange.")
        print("Make sure you have:")
        print("  1. Tested on testnet first")
        print("  2. Reviewed risk management settings")
        print("  3. Set appropriate position limits")
        print("  4. API key has withdrawal DISABLED")
        print("  5. IP whitelist enabled on exchange")
        print("")

        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return False

    return True


async def run_trading(config_path: str, mode: str):
    """Run trading in specified mode."""
    orchestrator = Orchestrator(config_path=config_path, mode=mode)

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


def main():
    parser = argparse.ArgumentParser(
        description="Run BTC Elite Trader in Live Trading mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Run in testnet/sandbox mode (recommended for testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    setup_logging(args.debug)

    mode = "testnet" if args.testnet else "live"

    if not check_prerequisites(mode):
        sys.exit(1)

    print("\n" + "="*50)
    print(f"BTC ELITE TRADER - {mode.upper()} MODE")
    print("="*50)
    print("\nStarting trading bot...")
    print("Press Ctrl+C to stop.\n")

    asyncio.run(run_trading(args.config, mode))


if __name__ == "__main__":
    main()

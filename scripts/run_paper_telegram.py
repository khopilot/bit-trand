#!/usr/bin/env python3
"""
Run Paper Trading with Telegram Integration

Runs the funding arbitrage paper trader with full Telegram integration.
Sends notifications for:
- Funding payments (every 8 hours)
- Daily summaries (00:00 UTC)
- Startup/shutdown events
- Rate anomaly alerts

Usage:
    python scripts/run_paper_telegram.py
    python scripts/run_paper_telegram.py --notional 20000
    python scripts/run_paper_telegram.py --no-telegram

Environment Variables:
    TELEGRAM_BOT_TOKEN - Telegram bot API token
    TELEGRAM_CHAT_ID - Target chat ID for notifications
"""

import argparse
import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None):
        pass  # No-op if dotenv not installed

from src.paper_trading import FundingArbPaperTrader
from src.telegram_control import TelegramControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("paper_telegram")


def run_paper_trader(trader: FundingArbPaperTrader):
    """Run paper trader in a separate thread."""
    trader.start_live_simulation()


async def main_async(
    notional_usd: float = 10000.0,
    telegram_enabled: bool = True,
):
    """
    Main async entry point.

    Args:
        notional_usd: Position size in USD
        telegram_enabled: Whether to enable Telegram bot
    """
    # Load environment variables from project root
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if telegram_enabled and not (bot_token and chat_id):
        logger.warning(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. "
            "Telegram notifications will print to console."
        )

    # Initialize paper trader
    trader = FundingArbPaperTrader(
        notional_usd=notional_usd,
        telegram_enabled=telegram_enabled,
    )

    # Initialize Telegram control if enabled
    telegram = None
    if telegram_enabled and bot_token:
        telegram = TelegramControl(
            bot_token=bot_token,
            chat_id=chat_id,
        )

        # Set callbacks for arb commands
        telegram.set_arb_callbacks(
            status=trader.get_status_dict,
            start=lambda: None,  # Already running
            stop=trader.stop,
            position=trader.get_position_dict,
            history=trader.get_history,
            stats=trader.get_stats,
            rate=trader.get_rate_info,
            config=trader.get_config,
        )

        # Also set main status callback so /status works
        telegram.set_callbacks(
            status=trader.get_status_dict,
        )

    print("""
================================================================================
        PAPER TRADING WITH TELEGRAM INTEGRATION
================================================================================

Starting paper trading simulation with Telegram notifications.

Commands available in Telegram:
    /arb        - Show paper trading status
    /arb_stop   - Stop paper trading

Press Ctrl+C to stop.
================================================================================
""")

    try:
        if telegram:
            # Start Telegram bot
            await telegram.start()

            # Run paper trader in background thread
            trader_thread = threading.Thread(
                target=run_paper_trader,
                args=(trader,),
                daemon=True,
            )
            trader_thread.start()

            # Wait for trader to start
            await asyncio.sleep(2)

            # Keep running while trader thread is alive
            while trader_thread.is_alive():
                await asyncio.sleep(1)

        else:
            # Run paper trader directly (blocking)
            trader.start_live_simulation()

    except KeyboardInterrupt:
        logger.info("Shutdown requested...")

    finally:
        # Clean shutdown
        trader.stop()

        if telegram:
            await telegram.stop()

        logger.info("Paper trading stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run paper trading with Telegram integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_paper_telegram.py
    python scripts/run_paper_telegram.py --notional 20000
    python scripts/run_paper_telegram.py --no-telegram

Environment Variables:
    TELEGRAM_BOT_TOKEN  - Telegram bot API token
    TELEGRAM_CHAT_ID    - Target chat ID
        """,
    )

    parser.add_argument(
        "--notional",
        type=float,
        default=10000.0,
        help="Position size in USD (default: 10000)",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram integration (console only)",
    )

    args = parser.parse_args()

    asyncio.run(
        main_async(
            notional_usd=args.notional,
            telegram_enabled=not args.no_telegram,
        )
    )


if __name__ == "__main__":
    main()

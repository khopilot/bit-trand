#!/usr/bin/env python3
"""
Run THE TRINITY - All Three Paper Trading Bots

Runs ALL bots in parallel:
- BOT A (Arb): Long Spot + Short Perp (funding collection)
- BOT B (Directional): EMA/RSI/BB signal trading
- BOT C (Beast): Hybrid ARB + DIRECTIONAL combined

All report to the same Telegram chat with different commands:
- /arb       - Arb bot status
- /dir       - Directional bot status
- /beast     - Beast hybrid bot status
- /mode      - Beast mode details
- /trinity   - Compare all 3 strategies
- /compare   - Compare arb vs dir
- /signals   - Current indicators
- /trades    - Recent directional trades

Usage:
    python scripts/run_trinity.py
    python scripts/run_trinity.py --notional 10000

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
        pass

from src.paper_trading import FundingArbPaperTrader
from src.paper_trading.directional_simulator import DirectionalPaperTrader
from src.paper_trading.beast_simulator import BeastPaperTrader
from src.telegram_control import TelegramControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "trinity.log"),
    ],
)

logger = logging.getLogger("trinity")


def run_arb_trader(trader: FundingArbPaperTrader):
    """Run arb trader in a thread."""
    try:
        trader.start_live_simulation()
    except Exception as e:
        logger.error("Arb trader error: %s", e)


def run_dir_trader(trader: DirectionalPaperTrader):
    """Run directional trader in a thread."""
    try:
        trader.start_live_simulation()
    except Exception as e:
        logger.error("Directional trader error: %s", e)


def run_beast_trader(trader: BeastPaperTrader):
    """Run beast trader in a thread."""
    try:
        trader.start_live_simulation()
    except Exception as e:
        logger.error("Beast trader error: %s", e)


async def main_async(
    notional_usd: float = 10000.0,
    telegram_enabled: bool = True,
):
    """Main async entry point."""
    # Ensure logs directory exists
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    # Load environment
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if telegram_enabled and not (bot_token and chat_id):
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")

    # Initialize ALL THREE traders
    arb_trader = FundingArbPaperTrader(
        notional_usd=notional_usd,
        telegram_enabled=False,  # Use shared Telegram control
    )

    dir_trader = DirectionalPaperTrader(
        notional_usd=notional_usd,
        telegram_enabled=False,  # Use shared Telegram control
    )

    beast_trader = BeastPaperTrader(
        notional_usd=notional_usd,
        telegram_enabled=False,  # Use shared Telegram control
    )

    # Initialize shared Telegram control
    telegram = None
    if telegram_enabled and bot_token:
        telegram = TelegramControl(
            bot_token=bot_token,
            chat_id=chat_id,
        )

        # Set arb callbacks
        telegram.set_arb_callbacks(
            status=arb_trader.get_status_dict,
            start=lambda: None,
            stop=arb_trader.stop,
            position=arb_trader.get_position_dict,
            history=arb_trader.get_history,
            stats=arb_trader.get_stats,
            rate=arb_trader.get_rate_info,
            config=arb_trader.get_config,
        )

        # Set directional callbacks
        telegram.set_dir_callbacks(
            status=dir_trader.get_status_dict,
            signals=dir_trader.get_signals_dict,
            trades=dir_trader.get_trades_dict,
            stop=dir_trader.stop,
        )

        # Set beast callbacks
        telegram.set_beast_callbacks(
            status=beast_trader.get_status_dict,
            mode=beast_trader.get_mode_dict,
            stop=beast_trader.stop,
        )

        # Also set main status callback
        telegram.set_callbacks(
            status=arb_trader.get_status_dict,
        )

    print("""
================================================================================
                    THE TRINITY - THREE BOTS IN PARALLEL
================================================================================

Running THREE paper trading bots:

  BOT A (ARB):         Long Spot + Short Perp = Funding collection
  BOT B (DIRECTIONAL): EMA + RSI + BB signals = Trend trading
  BOT C (THE BEAST):   ARB + DIRECTIONAL hybrid = Dynamic hedge

Telegram Commands:
  /trinity   - Compare ALL 3 strategies (recommended!)
  /arb       - Arb bot status
  /dir       - Directional bot status
  /beast     - Beast hybrid status
  /mode      - Beast mode details
  /signals   - Current indicators
  /trades    - Recent directional trades

Press Ctrl+C to stop all bots.
================================================================================
""")

    try:
        if telegram:
            await telegram.start()

            # Start all traders in threads
            arb_thread = threading.Thread(
                target=run_arb_trader,
                args=(arb_trader,),
                daemon=True,
                name="ArbTrader",
            )

            dir_thread = threading.Thread(
                target=run_dir_trader,
                args=(dir_trader,),
                daemon=True,
                name="DirTrader",
            )

            beast_thread = threading.Thread(
                target=run_beast_trader,
                args=(beast_trader,),
                daemon=True,
                name="BeastTrader",
            )

            arb_thread.start()
            dir_thread.start()
            beast_thread.start()

            logger.info("All 3 traders started")

            # Wait for startup
            await asyncio.sleep(5)

            # Send startup notification
            await telegram.send_message(
                "*THE TRINITY STARTED*\n\n"
                "Running ALL 3 bots in parallel:\n"
                "- ARB (Safe)\n"
                "- DIRECTIONAL (Hunter)\n"
                "- BEAST (Monster)\n\n"
                "Use /trinity to compare all!"
            )

            # Keep running while threads alive
            while (arb_thread.is_alive() or
                   dir_thread.is_alive() or
                   beast_thread.is_alive()):
                await asyncio.sleep(1)

        else:
            # Run without telegram - just start all threads
            arb_thread = threading.Thread(
                target=run_arb_trader,
                args=(arb_trader,),
                daemon=True,
            )
            dir_thread = threading.Thread(
                target=run_dir_trader,
                args=(dir_trader,),
                daemon=True,
            )
            beast_thread = threading.Thread(
                target=run_beast_trader,
                args=(beast_trader,),
                daemon=True,
            )

            arb_thread.start()
            dir_thread.start()
            beast_thread.start()

            # Keep running
            while True:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown requested...")

    finally:
        arb_trader.stop()
        dir_trader.stop()
        beast_trader.stop()

        if telegram:
            await telegram.send_message(
                "*THE TRINITY STOPPED*\n\n"
                "All 3 bots have stopped.\n"
                "Use /trinity to see final results."
            )
            await telegram.stop()

        logger.info("Trinity stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run THE TRINITY - all 3 paper trading bots",
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
        help="Disable Telegram",
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

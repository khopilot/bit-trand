#!/usr/bin/env python3
"""
Run Dual Paper Trading Bots with Telegram Integration

Runs BOTH bots in parallel:
- Arb Bot: Funding arbitrage (Long Spot + Short Perp)
- Directional Bot: EMA/RSI/BB signal trading

Both report to the same Telegram chat with different commands:
- /arb, /pos, /stats - Arb bot commands
- /dir, /signals, /trades - Directional bot commands
- /compare - Side-by-side comparison

Usage:
    python scripts/run_dual_bots.py
    python scripts/run_dual_bots.py --notional 10000

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
from src.telegram_control import TelegramControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("dual_bots")


def run_arb_trader(trader: FundingArbPaperTrader):
    """Run arb trader in a thread."""
    trader.start_live_simulation()


def run_dir_trader(trader: DirectionalPaperTrader):
    """Run directional trader in a thread."""
    trader.start_live_simulation()


async def main_async(
    notional_usd: float = 10000.0,
    telegram_enabled: bool = True,
):
    """Main async entry point."""
    # Load environment
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if telegram_enabled and not (bot_token and chat_id):
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")

    # Initialize BOTH traders
    arb_trader = FundingArbPaperTrader(
        notional_usd=notional_usd,
        telegram_enabled=telegram_enabled,
    )

    dir_trader = DirectionalPaperTrader(
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

        # Also set main status callback
        telegram.set_callbacks(
            status=arb_trader.get_status_dict,
        )

    print("""
================================================================================
           DUAL BOT PAPER TRADING - ARB vs DIRECTIONAL
================================================================================

Running TWO paper trading bots in parallel:

  BOT A (ARB):         Long Spot + Short Perp = Funding collection
  BOT B (DIRECTIONAL): EMA + RSI + BB signals = Trend trading

Telegram Commands:
  /arb       - Arb bot status
  /dir       - Directional bot status
  /compare   - Side-by-side comparison
  /signals   - Current indicators
  /trades    - Recent directional trades

Press Ctrl+C to stop both bots.
================================================================================
""")

    try:
        if telegram:
            await telegram.start()

            # Start arb trader in thread
            arb_thread = threading.Thread(
                target=run_arb_trader,
                args=(arb_trader,),
                daemon=True,
                name="ArbTrader",
            )
            arb_thread.start()

            # Start directional trader in thread
            dir_thread = threading.Thread(
                target=run_dir_trader,
                args=(dir_trader,),
                daemon=True,
                name="DirTrader",
            )
            dir_thread.start()

            # Wait for startup
            await asyncio.sleep(3)

            # Send startup notification
            await telegram.send_message(
                "*DUAL BOTS STARTED*\n\n"
                "Running ARB + DIRECTIONAL in parallel.\n"
                "Use /compare to see both."
            )

            # Keep running while both threads alive
            while arb_thread.is_alive() or dir_thread.is_alive():
                await asyncio.sleep(1)

        else:
            # Run just arb trader without telegram
            arb_trader.start_live_simulation()

    except KeyboardInterrupt:
        logger.info("Shutdown requested...")

    finally:
        arb_trader.stop()
        dir_trader.stop()

        if telegram:
            await telegram.send_message(
                "*DUAL BOTS STOPPED*\n\n"
                "Use /compare to see final results."
            )
            await telegram.stop()

        logger.info("Dual bots stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run dual paper trading bots",
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

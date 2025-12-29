#!/usr/bin/env python3
"""
Run THE PREDATOR - Hunt Smart Money & Liquidations

Standalone runner for the Predator bot with Telegram integration.

The Predator predicts OTHER TRADERS' behavior, not price.
It fades the crowd and hunts liquidations using FREE Binance data.

Commands:
- /predator   - Full analysis report
- /hunt       - Active signals only
- /conviction - Quick score check

Usage:
    python scripts/run_predator.py
    python scripts/run_predator.py --interval 5
    python scripts/run_predator.py --once  # Single analysis

Environment Variables:
    TELEGRAM_BOT_TOKEN - Telegram bot API token
    TELEGRAM_CHAT_ID - Target chat ID for notifications
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None):
        pass

from src.predator import PredatorBot
from src.telegram_control import TelegramControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "predator.log"),
    ],
)

logger = logging.getLogger("predator")


async def main_async(
    interval_minutes: int = 5,
    telegram_enabled: bool = True,
    single_run: bool = False,
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
        telegram_enabled = False

    # Initialize Predator bot
    predator = PredatorBot(
        symbol="BTCUSDT",
        state_file=PROJECT_ROOT / "logs" / "predator_state.json",
    )

    # Signal handling
    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received...")
        predator.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Single run mode
    if single_run:
        print("\n" + "=" * 60)
        print("THE PREDATOR - Single Analysis")
        print("=" * 60 + "\n")

        try:
            score = predator.analyze()
            print(predator.get_detailed_report())
            return
        except Exception as e:
            logger.error("Analysis failed: %s", e)
            return

    # Initialize Telegram
    telegram = None
    if telegram_enabled and bot_token:
        telegram = TelegramControl(
            bot_token=bot_token,
            chat_id=chat_id,
        )

        # Set predator callbacks
        telegram.set_predator_callbacks(
            status=predator.get_status_dict,
            hunt=predator.get_hunt_report,
            analyze=predator.analyze,
        )

    print("""
================================================================================
                    THE PREDATOR - HUNT SMART MONEY
================================================================================

Predicting OTHER TRADERS' behavior using FREE Binance data:
- Funding Rate Extremes (30% weight)
- Open Interest Divergence (25% weight)
- Long/Short Account Ratio (25% weight)
- Taker Buy/Sell Volume (20% weight)

Telegram Commands:
  /predator   - Full analysis report
  /hunt       - Active signals only
  /conviction - Quick score check

Press Ctrl+C to stop.
================================================================================
""")

    try:
        if telegram:
            await telegram.start()

            # Send startup notification
            await telegram.send_message(
                "🦅 *THE PREDATOR STARTED*\n\n"
                "Hunting smart money & liquidations...\n\n"
                "Use /predator for full analysis\n"
                "Use /hunt for active signals\n"
                "Use /conviction for quick check"
            )

        # Run initial analysis
        logger.info("Running initial analysis...")
        try:
            score = predator.analyze()
            logger.info(
                "Initial: %s (conviction: %.0f)",
                score.signal, score.conviction
            )

            if telegram and score.action in ("TRADE", "CONSIDER"):
                await telegram.send_message(
                    f"🦅 *INITIAL SCAN COMPLETE*\n\n"
                    f"Signal: *{score.signal}*\n"
                    f"Conviction: `{score.conviction:.0f}/100`\n"
                    f"Action: *{score.action}*\n\n"
                    f"Use /predator for details"
                )
        except Exception as e:
            logger.error("Initial analysis failed: %s", e)

        # Continuous loop
        while predator.running or not single_run:
            predator.running = True

            # Wait for interval
            for _ in range(interval_minutes * 60):
                if not predator.running:
                    break
                await asyncio.sleep(1)

            if not predator.running:
                break

            # Run analysis
            try:
                score = predator.analyze()

                # Alert on high conviction
                if telegram and score.action == "TRADE":
                    emoji = "🟢" if score.signal == "LONG" else "🔴"
                    await telegram.send_message(
                        f"🦅 *PREDATOR ALERT*\n\n"
                        f"{emoji} *{score.signal}* Signal!\n"
                        f"Conviction: `{score.conviction:.0f}/100`\n\n"
                        f"{score.trade_rationale}\n\n"
                        f"Use /predator for full analysis"
                    )

            except Exception as e:
                logger.error("Analysis error: %s", e)

    except KeyboardInterrupt:
        logger.info("Shutdown requested...")

    finally:
        predator.stop()

        if telegram:
            await telegram.send_message(
                "🦅 *THE PREDATOR STOPPED*\n\n"
                "No longer hunting."
            )
            await telegram.stop()

        logger.info("Predator stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run THE PREDATOR - Hunt smart money & liquidations",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Minutes between analyses (default: 5)",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single analysis and exit",
    )

    args = parser.parse_args()

    asyncio.run(
        main_async(
            interval_minutes=args.interval,
            telegram_enabled=not args.no_telegram,
            single_run=args.once,
        )
    )


if __name__ == "__main__":
    main()

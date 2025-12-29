"""
Predator Bot - Main Orchestrator

Coordinates all analyzers and produces actionable signals.
Designed to run standalone or integrate with Trinity.

Features:
- Fetches all market data via data_fetcher
- Runs all analyzers (funding, OI, sentiment)
- Produces conviction score
- Tracks signal history
- Can send alerts via callback
"""

import logging
import time
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import FETCH_INTERVAL_MINUTES, DEFAULT_SYMBOL
from .data_fetcher import PredatorDataFetcher
from .funding_analyzer import FundingAnalyzer
from .oi_analyzer import OIAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from .conviction_scorer import ConvictionScorer, ConvictionScore

logger = logging.getLogger(__name__)


class PredatorBot:
    """
    The Predator - Hunt Smart Money & Liquidations

    This bot predicts OTHER TRADERS' behavior, not price.
    It fades the crowd and hunts liquidations.
    """

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        state_file: Optional[Path] = None,
        alert_callback: Optional[Callable[[ConvictionScore], None]] = None,
    ):
        """
        Initialize Predator Bot.

        Args:
            symbol: Trading pair (default: BTCUSDT)
            state_file: Path to save state (optional)
            alert_callback: Function to call on high conviction signals
        """
        self.symbol = symbol
        self.state_file = state_file or Path("logs/predator_state.json")
        self.alert_callback = alert_callback

        # Initialize components
        self.fetcher = PredatorDataFetcher(symbol=symbol)
        self.funding_analyzer = FundingAnalyzer()
        self.oi_analyzer = OIAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conviction_scorer = ConvictionScorer()

        # State
        self.running = False
        self.last_analysis: Optional[ConvictionScore] = None
        self.analysis_count = 0
        self.last_alert_signal: Optional[str] = None

        # Load state if exists
        self._load_state()

    def analyze(self) -> ConvictionScore:
        """
        Run full analysis and return conviction score.

        This is the main entry point for getting signals.
        """
        try:
            # Fetch all data
            logger.info("Fetching market data for %s...", self.symbol)
            data = self.fetcher.get_all_data()

            # Get current price for OI analysis
            current_price = self.fetcher.get_current_price()

            # Calculate 4h price change for OI divergence
            # TODO: Add price history tracking
            price_change_4h = 0.0  # Placeholder

            # Run analyzers
            funding_signal = self.funding_analyzer.analyze(data["funding"])
            oi_signal = self.oi_analyzer.analyze(
                data["oi"], price_change_4h=price_change_4h
            )
            sentiment_signal = self.sentiment_analyzer.analyze(
                data["ls_ratio"], data["taker"]
            )

            # Calculate conviction score
            score = self.conviction_scorer.score(
                funding_signal, oi_signal, sentiment_signal
            )

            self.last_analysis = score
            self.analysis_count += 1

            # Log summary
            logger.info(
                "Analysis #%d: %s (conviction: %.0f, action: %s)",
                self.analysis_count,
                score.signal,
                score.conviction,
                score.action,
            )

            # Check for alert condition
            self._check_alert(score)

            # Save state
            self._save_state()

            return score

        except Exception as e:
            logger.error("Analysis failed: %s", e)
            raise

    def _check_alert(self, score: ConvictionScore) -> None:
        """Check if we should send an alert."""
        # Only alert on high conviction trades
        if score.action not in ("TRADE", "CONSIDER"):
            return

        # Don't repeat the same alert
        if score.signal == self.last_alert_signal:
            return

        # Send alert
        if self.alert_callback:
            try:
                self.alert_callback(score)
                self.last_alert_signal = score.signal
                logger.info("Alert sent: %s", score.signal)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

    def run_continuous(
        self,
        interval_minutes: int = FETCH_INTERVAL_MINUTES,
    ) -> None:
        """
        Run continuous analysis loop.

        Args:
            interval_minutes: Minutes between analyses
        """
        self.running = True
        logger.info(
            "Starting Predator Bot (interval: %d min)", interval_minutes
        )

        while self.running:
            try:
                self.analyze()

                # Wait for next interval
                for _ in range(interval_minutes * 60):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error("Analysis error: %s", e)
                # Wait before retry
                time.sleep(60)

        logger.info("Predator Bot stopped")

    def stop(self) -> None:
        """Stop the continuous loop."""
        self.running = False
        self._save_state()

    def get_status_dict(self) -> dict:
        """Get current status as dict (for Telegram)."""
        if not self.last_analysis:
            return {
                "status": "No analysis yet",
                "analysis_count": self.analysis_count,
            }

        score = self.last_analysis
        return {
            "timestamp": score.timestamp.isoformat(),
            "signal": score.signal,
            "conviction": score.conviction,
            "strength": score.strength,
            "action": score.action,
            "funding_score": score.funding_score,
            "oi_score": score.oi_score,
            "ls_ratio_score": score.ls_ratio_score,
            "taker_score": score.taker_score,
            "signals_long": score.signals_long,
            "signals_short": score.signals_short,
            "signals_neutral": score.signals_neutral,
            "agreement_pct": score.agreement_pct,
            "summary": score.summary,
            "analysis_count": self.analysis_count,
        }

    def get_detailed_report(self) -> str:
        """Get full detailed report."""
        if not self.last_analysis:
            return "No analysis available. Run analyze() first."

        score = self.last_analysis

        # Build full report
        report = self.conviction_scorer.format_report(score)

        # Add component details
        report += "\n\n" + self.funding_analyzer.format_report(
            score.funding_signal
        )
        report += "\n" + self.oi_analyzer.format_report(score.oi_signal)
        report += "\n" + self.sentiment_analyzer.format_report(
            score.sentiment_signal
        )

        return report

    def get_hunt_report(self) -> str:
        """Get active signals only (for /hunt command)."""
        if not self.last_analysis:
            return "No active hunt. Run analyze() first."

        score = self.last_analysis

        # Only show if there's an actionable signal
        if score.action in ("WAIT", "NO_TRADE"):
            return (
                f"🦅 *NO ACTIVE HUNT*\n\n"
                f"Conviction: {score.conviction:.0f}/100 ({score.strength})\n"
                f"Signals: {score.signals_long}L/{score.signals_short}S/{score.signals_neutral}N\n\n"
                f"Waiting for stronger signals..."
            )

        emoji = "🟢" if score.signal == "LONG" else "🔴"
        return (
            f"🦅 *ACTIVE HUNT*\n\n"
            f"{emoji} *{score.signal}* Signal Detected!\n\n"
            f"Conviction: `{score.conviction:.0f}/100` ({score.strength})\n"
            f"Action: *{score.action}*\n\n"
            f"📋 {score.summary}\n\n"
            f"💡 {score.trade_rationale}"
        )

    def _save_state(self) -> None:
        """Save state to file."""
        if not self.state_file:
            return

        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "symbol": self.symbol,
                "analysis_count": self.analysis_count,
                "last_alert_signal": self.last_alert_signal,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            if self.last_analysis:
                state["last_analysis"] = {
                    "timestamp": self.last_analysis.timestamp.isoformat(),
                    "signal": self.last_analysis.signal,
                    "conviction": self.last_analysis.conviction,
                    "action": self.last_analysis.action,
                }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.warning("Failed to save state: %s", e)

    def _load_state(self) -> None:
        """Load state from file."""
        if not self.state_file or not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            self.analysis_count = state.get("analysis_count", 0)
            self.last_alert_signal = state.get("last_alert_signal")

            logger.info(
                "Loaded state: %d previous analyses", self.analysis_count
            )

        except Exception as e:
            logger.warning("Failed to load state: %s", e)

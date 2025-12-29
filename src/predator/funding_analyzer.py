"""
Funding Rate Analyzer - Hunt Overleveraged Positions

Funding rate logic:
- Positive funding = Longs pay shorts = Market is overleveraged LONG
- Negative funding = Shorts pay longs = Market is overleveraged SHORT

Trading signals:
- Funding > 0.05% (extreme high) = Contrarian SHORT
- Funding < -0.02% (extreme low) = Contrarian LONG
- Funding in middle = NEUTRAL

Weight: 30% of total conviction score
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .config import (
    FUNDING_EXTREME_HIGH,
    FUNDING_EXTREME_LOW,
    FUNDING_VERY_HIGH,
    FUNDING_VERY_LOW,
    FUNDING_PERCENTILE_HIGH,
    FUNDING_PERCENTILE_LOW,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NEUTRAL,
    WEIGHT_FUNDING,
)
from .data_fetcher import FundingData

logger = logging.getLogger(__name__)


@dataclass
class FundingSignal:
    """Signal from funding rate analysis."""
    signal: str  # LONG, SHORT, or NEUTRAL
    score: float  # 0-100 contribution to conviction
    strength: str  # "extreme", "high", "moderate", "weak"
    reason: str  # Human-readable explanation
    current_rate: float
    current_rate_pct: str  # Formatted as percentage
    percentile: float
    annualized_rate: float  # Annual % based on current


class FundingAnalyzer:
    """
    Analyzes funding rates for contrarian signals.

    High positive funding = Overleveraged longs = FADE with SHORT
    High negative funding = Overleveraged shorts = FADE with LONG
    """

    def __init__(self):
        self.last_signal: Optional[FundingSignal] = None

    def analyze(self, data: FundingData) -> FundingSignal:
        """
        Analyze funding rate data and generate signal.

        Args:
            data: FundingData from data_fetcher

        Returns:
            FundingSignal with direction and score
        """
        rate = data.current_rate
        percentile = data.percentile_current

        # Annualized rate (3 payments/day * 365 days)
        annualized = rate * 3 * 365 * 100  # As percentage

        # Determine signal based on rate extremes
        signal, strength, score, reason = self._evaluate_rate(
            rate, percentile, annualized
        )

        result = FundingSignal(
            signal=signal,
            score=score,
            strength=strength,
            reason=reason,
            current_rate=rate,
            current_rate_pct=f"{rate * 100:.4f}%",
            percentile=percentile,
            annualized_rate=annualized,
        )

        self.last_signal = result
        return result

    def _evaluate_rate(
        self, rate: float, percentile: float, annualized: float
    ) -> tuple[str, str, float, str]:
        """
        Evaluate funding rate and return (signal, strength, score, reason).

        Scoring:
        - Extreme (>90th or <10th percentile): 80-100 pts
        - High (>75th or <25th percentile): 50-80 pts
        - Moderate: 20-50 pts
        - Neutral: 0-20 pts
        """
        # Very extreme positive = STRONG SHORT signal
        if rate >= FUNDING_VERY_HIGH:
            return (
                SIGNAL_SHORT,
                "extreme",
                95.0,
                f"EXTREME overleveraged longs! Rate {rate*100:.4f}% "
                f"({annualized:.1f}% APY). Liquidation cascade likely."
            )

        # Extreme positive = SHORT signal
        if rate >= FUNDING_EXTREME_HIGH:
            score = 70 + (rate - FUNDING_EXTREME_HIGH) / (FUNDING_VERY_HIGH - FUNDING_EXTREME_HIGH) * 25
            return (
                SIGNAL_SHORT,
                "high",
                min(score, 90),
                f"High funding {rate*100:.4f}% = longs paying {annualized:.1f}% APY. "
                f"Crowd is LONG, consider SHORT."
            )

        # Very extreme negative = STRONG LONG signal
        if rate <= FUNDING_VERY_LOW:
            return (
                SIGNAL_LONG,
                "extreme",
                95.0,
                f"EXTREME overleveraged shorts! Rate {rate*100:.4f}% "
                f"({annualized:.1f}% APY). Short squeeze likely."
            )

        # Extreme negative = LONG signal
        if rate <= FUNDING_EXTREME_LOW:
            score = 70 + (FUNDING_EXTREME_LOW - rate) / (FUNDING_EXTREME_LOW - FUNDING_VERY_LOW) * 25
            return (
                SIGNAL_LONG,
                "high",
                min(score, 90),
                f"Negative funding {rate*100:.4f}% = shorts paying. "
                f"Crowd is SHORT, consider LONG."
            )

        # Check percentile for moderate signals
        if percentile >= FUNDING_PERCENTILE_HIGH:
            score = 30 + (percentile - FUNDING_PERCENTILE_HIGH) / (100 - FUNDING_PERCENTILE_HIGH) * 40
            return (
                SIGNAL_SHORT,
                "moderate",
                score,
                f"Funding at {percentile:.0f}th percentile. "
                f"Above average bullish positioning."
            )

        if percentile <= FUNDING_PERCENTILE_LOW:
            score = 30 + (FUNDING_PERCENTILE_LOW - percentile) / FUNDING_PERCENTILE_LOW * 40
            return (
                SIGNAL_LONG,
                "moderate",
                score,
                f"Funding at {percentile:.0f}th percentile. "
                f"Below average bearish positioning."
            )

        # Neutral zone
        return (
            SIGNAL_NEUTRAL,
            "weak",
            10.0,
            f"Funding neutral at {rate*100:.4f}% ({percentile:.0f}th percentile). "
            f"No extreme positioning detected."
        )

    def get_weighted_score(self, signal: FundingSignal) -> float:
        """
        Get the weighted score contribution (0-30 for 30% weight).

        Args:
            signal: FundingSignal from analyze()

        Returns:
            Score contribution (0 to WEIGHT_FUNDING * 100)
        """
        return signal.score * WEIGHT_FUNDING

    def format_report(self, signal: FundingSignal) -> str:
        """Format signal as readable report."""
        emoji = {
            SIGNAL_LONG: "🟢",
            SIGNAL_SHORT: "🔴",
            SIGNAL_NEUTRAL: "⚪",
        }

        return f"""
📊 *FUNDING ANALYSIS*
━━━━━━━━━━━━━━━━━━━━━━
Signal: {emoji.get(signal.signal, '❓')} *{signal.signal}* ({signal.strength})
Score: `{signal.score:.1f}/100` (weighted: `{self.get_weighted_score(signal):.1f}`)

Current Rate: `{signal.current_rate_pct}`
Annualized: `{signal.annualized_rate:.1f}%`
Percentile: `{signal.percentile:.0f}th`

💡 {signal.reason}
"""

"""
Conviction Scorer - Aggregate All Signals Into Actionable Score

Combines all analyzer outputs into a single conviction score (0-100):
- Funding Rate: 30% weight
- Open Interest: 25% weight
- L/S Account Ratio: 25% weight
- Taker Volume: 20% weight

Score interpretation:
- 75-100: HIGH conviction - Strong trade signal
- 50-74: MODERATE conviction - Consider trading
- 25-49: MIXED signals - Wait for clarity
- 0-24: NO edge - Do not trade

The final signal is determined by signal agreement, not just score.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import (
    CONVICTION_STRONG,
    CONVICTION_MODERATE,
    CONVICTION_WEAK,
    WEIGHT_FUNDING,
    WEIGHT_OI,
    WEIGHT_LS_RATIO,
    WEIGHT_TAKER,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NEUTRAL,
)
from .funding_analyzer import FundingSignal
from .oi_analyzer import OISignal
from .sentiment_analyzer import SentimentSignal

logger = logging.getLogger(__name__)


@dataclass
class ConvictionScore:
    """Final conviction score with all components."""
    timestamp: datetime

    # Overall verdict
    signal: str  # LONG, SHORT, or NEUTRAL
    conviction: float  # 0-100 total score
    strength: str  # "high", "moderate", "mixed", "none"
    action: str  # "TRADE", "CONSIDER", "WAIT", "NO_TRADE"

    # Component scores (already weighted)
    funding_score: float  # 0-30
    oi_score: float  # 0-25
    ls_ratio_score: float  # 0-25
    taker_score: float  # 0-20

    # Signal agreement
    signals_long: int  # How many signals say LONG
    signals_short: int  # How many signals say SHORT
    signals_neutral: int  # How many signals are neutral
    agreement_pct: float  # % of non-neutral signals agreeing

    # Human-readable summary
    summary: str
    trade_rationale: str

    # Raw signals for reference
    funding_signal: FundingSignal
    oi_signal: OISignal
    sentiment_signal: SentimentSignal


class ConvictionScorer:
    """
    Aggregates all signals into a single actionable conviction score.

    Scoring philosophy:
    1. Weight each component according to its predictive power
    2. Require signal agreement for high conviction
    3. Conflicting signals reduce conviction
    4. Strong signals in one area can override weak signals elsewhere
    """

    def __init__(self):
        self.last_score: Optional[ConvictionScore] = None
        self.history: list[ConvictionScore] = []

    def score(
        self,
        funding_signal: FundingSignal,
        oi_signal: OISignal,
        sentiment_signal: SentimentSignal,
    ) -> ConvictionScore:
        """
        Calculate conviction score from all signals.

        Args:
            funding_signal: From FundingAnalyzer
            oi_signal: From OIAnalyzer
            sentiment_signal: From SentimentAnalyzer (includes L/S and taker)

        Returns:
            ConvictionScore with overall verdict
        """
        # Calculate weighted component scores
        funding_score = funding_signal.score * WEIGHT_FUNDING
        oi_score = oi_signal.score * WEIGHT_OI
        ls_ratio_score = sentiment_signal.ls_signal.score * WEIGHT_LS_RATIO
        taker_score = sentiment_signal.taker_signal.score * WEIGHT_TAKER

        # Total conviction (0-100)
        total_conviction = funding_score + oi_score + ls_ratio_score + taker_score

        # Count signal directions
        signals = [
            funding_signal.signal,
            oi_signal.signal,
            sentiment_signal.ls_signal.signal,
            sentiment_signal.taker_signal.signal,
        ]

        signals_long = sum(1 for s in signals if s == SIGNAL_LONG)
        signals_short = sum(1 for s in signals if s == SIGNAL_SHORT)
        signals_neutral = sum(1 for s in signals if s == SIGNAL_NEUTRAL)

        # Calculate agreement percentage
        non_neutral = signals_long + signals_short
        if non_neutral > 0:
            max_direction = max(signals_long, signals_short)
            agreement_pct = (max_direction / non_neutral) * 100
        else:
            agreement_pct = 0

        # Determine overall signal direction
        final_signal = self._determine_signal(
            signals_long, signals_short, signals_neutral, total_conviction
        )

        # Determine strength and action
        strength, action = self._determine_strength(
            total_conviction, agreement_pct, final_signal
        )

        # Generate summary and rationale
        summary = self._generate_summary(
            final_signal, total_conviction, strength,
            signals_long, signals_short, signals_neutral
        )

        trade_rationale = self._generate_rationale(
            funding_signal, oi_signal, sentiment_signal, final_signal
        )

        result = ConvictionScore(
            timestamp=datetime.now(timezone.utc),
            signal=final_signal,
            conviction=total_conviction,
            strength=strength,
            action=action,
            funding_score=funding_score,
            oi_score=oi_score,
            ls_ratio_score=ls_ratio_score,
            taker_score=taker_score,
            signals_long=signals_long,
            signals_short=signals_short,
            signals_neutral=signals_neutral,
            agreement_pct=agreement_pct,
            summary=summary,
            trade_rationale=trade_rationale,
            funding_signal=funding_signal,
            oi_signal=oi_signal,
            sentiment_signal=sentiment_signal,
        )

        self.last_score = result
        self.history.append(result)

        # Keep only last 100 scores
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return result

    def _determine_signal(
        self,
        longs: int,
        shorts: int,
        neutrals: int,
        conviction: float,
    ) -> str:
        """Determine the final signal direction."""
        # Need at least 2 non-neutral signals agreeing
        if longs >= 2 and longs > shorts:
            return SIGNAL_LONG
        if shorts >= 2 and shorts > longs:
            return SIGNAL_SHORT

        # If tied or mostly neutral, stay neutral
        return SIGNAL_NEUTRAL

    def _determine_strength(
        self,
        conviction: float,
        agreement: float,
        signal: str,
    ) -> tuple[str, str]:
        """Determine signal strength and recommended action."""

        # High conviction + good agreement = strong trade
        if conviction >= CONVICTION_STRONG and agreement >= 75:
            return ("high", "TRADE")

        if conviction >= CONVICTION_STRONG and agreement >= 50:
            return ("high", "CONSIDER")

        # Moderate conviction
        if conviction >= CONVICTION_MODERATE:
            if agreement >= 75:
                return ("moderate", "CONSIDER")
            else:
                return ("moderate", "WAIT")

        # Mixed/weak
        if conviction >= CONVICTION_WEAK:
            return ("mixed", "WAIT")

        # No edge
        return ("none", "NO_TRADE")

    def _generate_summary(
        self,
        signal: str,
        conviction: float,
        strength: str,
        longs: int,
        shorts: int,
        neutrals: int,
    ) -> str:
        """Generate human-readable summary."""
        if signal == SIGNAL_NEUTRAL:
            return (
                f"NEUTRAL - No clear edge. "
                f"Signals: {longs}L/{shorts}S/{neutrals}N. "
                f"Conviction: {conviction:.0f}/100 ({strength})"
            )

        return (
            f"{signal} SIGNAL - {strength.upper()} conviction. "
            f"Signals: {longs}L/{shorts}S/{neutrals}N. "
            f"Score: {conviction:.0f}/100"
        )

    def _generate_rationale(
        self,
        funding: FundingSignal,
        oi: OISignal,
        sentiment: SentimentSignal,
        final_signal: str,
    ) -> str:
        """Generate trading rationale from component signals."""
        reasons = []

        if funding.signal == final_signal and funding.signal != SIGNAL_NEUTRAL:
            reasons.append(f"Funding: {funding.reason}")

        if oi.signal == final_signal and oi.signal != SIGNAL_NEUTRAL:
            reasons.append(f"OI: {oi.reason}")

        if sentiment.signal == final_signal and sentiment.signal != SIGNAL_NEUTRAL:
            reasons.append(f"Sentiment: {sentiment.reason}")

        if not reasons:
            return "No strong confirmations. Mixed signals suggest waiting."

        return " | ".join(reasons)

    def format_report(self, score: ConvictionScore) -> str:
        """Format conviction score as readable report."""
        emoji = {
            SIGNAL_LONG: "🟢",
            SIGNAL_SHORT: "🔴",
            SIGNAL_NEUTRAL: "⚪",
        }

        action_emoji = {
            "TRADE": "🎯",
            "CONSIDER": "🤔",
            "WAIT": "⏳",
            "NO_TRADE": "🚫",
        }

        # Component bar visualization
        def bar(value: float, max_val: float) -> str:
            filled = int((value / max_val) * 10)
            return "█" * filled + "░" * (10 - filled)

        return f"""
🦅 *PREDATOR CONVICTION SCORE*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{emoji.get(score.signal, '❓')} *{score.signal}* | {action_emoji.get(score.action, '❓')} *{score.action}*

*CONVICTION: `{score.conviction:.0f}/100`* ({score.strength})

*Component Scores:*
Funding  [{bar(score.funding_score, 30)}] `{score.funding_score:.1f}/30`
OI       [{bar(score.oi_score, 25)}] `{score.oi_score:.1f}/25`
L/S Ratio[{bar(score.ls_ratio_score, 25)}] `{score.ls_ratio_score:.1f}/25`
Taker    [{bar(score.taker_score, 20)}] `{score.taker_score:.1f}/20`

*Signal Agreement:*
🟢 LONG: {score.signals_long} | 🔴 SHORT: {score.signals_short} | ⚪ NEUTRAL: {score.signals_neutral}
Agreement: `{score.agreement_pct:.0f}%`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 *{score.summary}*

💡 *Rationale:* {score.trade_rationale}
"""

    def format_short_report(self, score: ConvictionScore) -> str:
        """Short format for quick glance."""
        emoji = {
            SIGNAL_LONG: "🟢",
            SIGNAL_SHORT: "🔴",
            SIGNAL_NEUTRAL: "⚪",
        }

        return (
            f"{emoji.get(score.signal, '❓')} {score.signal} | "
            f"Score: {score.conviction:.0f} | "
            f"{score.action} | "
            f"{score.signals_long}L/{score.signals_short}S/{score.signals_neutral}N"
        )

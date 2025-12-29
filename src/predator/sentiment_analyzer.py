"""
Sentiment Analyzer - Read the Crowd & Fade Them

Two data sources:
1. Long/Short Account Ratio (25% weight)
   - Shows % of accounts that are long vs short
   - Crowd is usually wrong at extremes
   - L/S > 2.0 = 66%+ long = Contrarian SHORT
   - L/S < 0.5 = 66%+ short = Contrarian LONG

2. Taker Buy/Sell Volume (20% weight)
   - Shows who is aggressively crossing the spread
   - Taker buy surge = Retail FOMO (often near tops)
   - Taker sell surge = Retail panic (often near bottoms)

Combined: 45% of total conviction score
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .config import (
    LS_RATIO_EXTREME_HIGH,
    LS_RATIO_EXTREME_LOW,
    LS_RATIO_MODERATE_HIGH,
    LS_RATIO_MODERATE_LOW,
    TAKER_IMBALANCE_HIGH,
    TAKER_IMBALANCE_LOW,
    TAKER_EXTREME_HIGH,
    TAKER_EXTREME_LOW,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NEUTRAL,
    WEIGHT_LS_RATIO,
    WEIGHT_TAKER,
)
from .data_fetcher import LongShortData, TakerVolumeData

logger = logging.getLogger(__name__)


@dataclass
class LSRatioSignal:
    """Signal from Long/Short ratio analysis."""
    signal: str
    score: float
    strength: str
    reason: str
    current_ratio: float
    long_pct: float
    short_pct: float
    crowd_position: str  # "LONG", "SHORT", or "BALANCED"


@dataclass
class TakerSignal:
    """Signal from Taker volume analysis."""
    signal: str
    score: float
    strength: str
    reason: str
    buy_sell_ratio: float
    market_action: str  # "BUYING", "SELLING", or "BALANCED"


@dataclass
class SentimentSignal:
    """Combined sentiment signal."""
    signal: str  # Overall direction
    score: float  # Combined score (0-100)
    strength: str
    reason: str
    ls_signal: LSRatioSignal
    taker_signal: TakerSignal


class SentimentAnalyzer:
    """
    Analyzes crowd sentiment via L/S ratio and taker volume.

    Philosophy: Fade the crowd at extremes.
    - When everyone is long, go short
    - When everyone is short, go long
    - When retail is panic buying, prepare to sell
    - When retail is panic selling, prepare to buy
    """

    def __init__(self):
        self.last_signal: Optional[SentimentSignal] = None

    def analyze_ls_ratio(self, data: LongShortData) -> LSRatioSignal:
        """
        Analyze Long/Short Account Ratio.

        Note: This is ACCOUNT ratio, not VALUE ratio.
        Many small longs vs few big shorts = high ratio but shorts may win.
        """
        ratio = data.current_ratio
        long_pct = data.long_pct
        short_pct = data.short_pct

        # Determine crowd position
        if ratio > 1.2:
            crowd = "LONG"
        elif ratio < 0.8:
            crowd = "SHORT"
        else:
            crowd = "BALANCED"

        # Generate contrarian signal
        signal, strength, score, reason = self._evaluate_ls_ratio(
            ratio, long_pct, short_pct
        )

        return LSRatioSignal(
            signal=signal,
            score=score,
            strength=strength,
            reason=reason,
            current_ratio=ratio,
            long_pct=long_pct,
            short_pct=short_pct,
            crowd_position=crowd,
        )

    def _evaluate_ls_ratio(
        self, ratio: float, long_pct: float, short_pct: float
    ) -> tuple[str, str, float, str]:
        """Evaluate L/S ratio for contrarian signal."""

        # EXTREME: >66% accounts on one side
        if ratio >= LS_RATIO_EXTREME_HIGH:
            return (
                SIGNAL_SHORT,
                "extreme",
                90.0,
                f"EXTREME crowd long! {long_pct:.1f}% accounts long (ratio {ratio:.2f}). "
                f"Fade the herd - prepare to SHORT."
            )

        if ratio <= LS_RATIO_EXTREME_LOW:
            return (
                SIGNAL_LONG,
                "extreme",
                90.0,
                f"EXTREME crowd short! {short_pct:.1f}% accounts short (ratio {ratio:.2f}). "
                f"Fade the herd - prepare to LONG."
            )

        # HIGH: >60% accounts on one side
        if ratio >= LS_RATIO_MODERATE_HIGH:
            score = 50 + (ratio - LS_RATIO_MODERATE_HIGH) / (LS_RATIO_EXTREME_HIGH - LS_RATIO_MODERATE_HIGH) * 40
            return (
                SIGNAL_SHORT,
                "high",
                min(score, 85),
                f"Crowd leaning long ({long_pct:.1f}% of accounts). "
                f"Contrarian SHORT signal."
            )

        if ratio <= LS_RATIO_MODERATE_LOW:
            score = 50 + (LS_RATIO_MODERATE_LOW - ratio) / (LS_RATIO_MODERATE_LOW - LS_RATIO_EXTREME_LOW) * 40
            return (
                SIGNAL_LONG,
                "high",
                min(score, 85),
                f"Crowd leaning short ({short_pct:.1f}% of accounts). "
                f"Contrarian LONG signal."
            )

        # BALANCED: No clear crowd bias
        return (
            SIGNAL_NEUTRAL,
            "weak",
            15.0,
            f"Crowd balanced (L/S ratio {ratio:.2f}). "
            f"No extreme positioning to fade."
        )

    def analyze_taker_volume(self, data: TakerVolumeData) -> TakerSignal:
        """
        Analyze Taker Buy/Sell Volume.

        Taker = market order that crosses the spread.
        High taker buy = aggressive retail FOMO (usually near tops).
        High taker sell = aggressive retail panic (usually near bottoms).
        """
        ratio = data.buy_sell_ratio

        # Determine market action
        if ratio > 1.1:
            action = "BUYING"
        elif ratio < 0.9:
            action = "SELLING"
        else:
            action = "BALANCED"

        signal, strength, score, reason = self._evaluate_taker(ratio)

        return TakerSignal(
            signal=signal,
            score=score,
            strength=strength,
            reason=reason,
            buy_sell_ratio=ratio,
            market_action=action,
        )

    def _evaluate_taker(
        self, ratio: float
    ) -> tuple[str, str, float, str]:
        """Evaluate taker volume for contrarian signal."""

        # EXTREME: 2x more buyers or sellers
        if ratio >= TAKER_EXTREME_HIGH:
            return (
                SIGNAL_SHORT,
                "extreme",
                85.0,
                f"EXTREME taker buying! Ratio {ratio:.2f} = retail FOMO. "
                f"Smart money likely distributing. Fade with SHORT."
            )

        if ratio <= TAKER_EXTREME_LOW:
            return (
                SIGNAL_LONG,
                "extreme",
                85.0,
                f"EXTREME taker selling! Ratio {ratio:.2f} = retail panic. "
                f"Smart money likely accumulating. Fade with LONG."
            )

        # HIGH: 50% more buyers or sellers
        if ratio >= TAKER_IMBALANCE_HIGH:
            score = 40 + (ratio - TAKER_IMBALANCE_HIGH) / (TAKER_EXTREME_HIGH - TAKER_IMBALANCE_HIGH) * 45
            return (
                SIGNAL_SHORT,
                "high",
                min(score, 80),
                f"Heavy taker buying (ratio {ratio:.2f}). "
                f"Late buyers entering - distribution phase."
            )

        if ratio <= TAKER_IMBALANCE_LOW:
            score = 40 + (TAKER_IMBALANCE_LOW - ratio) / (TAKER_IMBALANCE_LOW - TAKER_EXTREME_LOW) * 45
            return (
                SIGNAL_LONG,
                "high",
                min(score, 80),
                f"Heavy taker selling (ratio {ratio:.2f}). "
                f"Panic sellers exiting - accumulation phase."
            )

        # BALANCED
        return (
            SIGNAL_NEUTRAL,
            "weak",
            10.0,
            f"Taker volume balanced (ratio {ratio:.2f}). "
            f"No clear aggression detected."
        )

    def analyze(
        self, ls_data: LongShortData, taker_data: TakerVolumeData
    ) -> SentimentSignal:
        """
        Analyze both L/S ratio and taker volume.

        Returns combined sentiment signal.
        """
        ls_signal = self.analyze_ls_ratio(ls_data)
        taker_signal = self.analyze_taker_volume(taker_data)

        # Combine signals
        combined_signal, combined_strength, combined_score, combined_reason = \
            self._combine_signals(ls_signal, taker_signal)

        result = SentimentSignal(
            signal=combined_signal,
            score=combined_score,
            strength=combined_strength,
            reason=combined_reason,
            ls_signal=ls_signal,
            taker_signal=taker_signal,
        )

        self.last_signal = result
        return result

    def _combine_signals(
        self, ls: LSRatioSignal, taker: TakerSignal
    ) -> tuple[str, str, float, str]:
        """Combine L/S ratio and taker signals."""

        # Both agree on direction = strong signal
        if ls.signal == taker.signal and ls.signal != SIGNAL_NEUTRAL:
            avg_score = (ls.score + taker.score) / 2
            return (
                ls.signal,
                "extreme" if avg_score > 80 else "high",
                avg_score,
                f"CONFLUENCE! {ls.signal}: L/S + Taker both confirm. "
                f"Crowd and retail on same side to fade."
            )

        # Only L/S has signal
        if ls.signal != SIGNAL_NEUTRAL and taker.signal == SIGNAL_NEUTRAL:
            return (
                ls.signal,
                ls.strength,
                ls.score * 0.8,  # Slightly lower without confirmation
                f"L/S ratio {ls.signal} signal. {ls.reason}"
            )

        # Only taker has signal
        if taker.signal != SIGNAL_NEUTRAL and ls.signal == SIGNAL_NEUTRAL:
            return (
                taker.signal,
                taker.strength,
                taker.score * 0.8,
                f"Taker volume {taker.signal} signal. {taker.reason}"
            )

        # Conflicting signals - be cautious
        if ls.signal != SIGNAL_NEUTRAL and taker.signal != SIGNAL_NEUTRAL:
            return (
                SIGNAL_NEUTRAL,
                "weak",
                20.0,
                f"Conflicting signals: L/S says {ls.signal}, Taker says {taker.signal}. "
                f"Wait for alignment."
            )

        # Both neutral
        return (
            SIGNAL_NEUTRAL,
            "weak",
            10.0,
            "Sentiment balanced. No clear crowd extreme to fade."
        )

    def get_weighted_score(self, signal: SentimentSignal) -> float:
        """
        Get the weighted score contribution.

        L/S Ratio: 25% weight
        Taker Volume: 20% weight
        Total: 45%
        """
        ls_contribution = signal.ls_signal.score * WEIGHT_LS_RATIO
        taker_contribution = signal.taker_signal.score * WEIGHT_TAKER
        return ls_contribution + taker_contribution

    def format_report(self, signal: SentimentSignal) -> str:
        """Format signal as readable report."""
        emoji = {
            SIGNAL_LONG: "🟢",
            SIGNAL_SHORT: "🔴",
            SIGNAL_NEUTRAL: "⚪",
        }

        ls = signal.ls_signal
        taker = signal.taker_signal

        return f"""
📊 *SENTIMENT ANALYSIS*
━━━━━━━━━━━━━━━━━━━━━━
Overall: {emoji.get(signal.signal, '❓')} *{signal.signal}* ({signal.strength})
Combined Score: `{signal.score:.1f}/100`

*L/S Account Ratio:*
  Signal: {emoji.get(ls.signal, '❓')} {ls.signal}
  Ratio: `{ls.current_ratio:.2f}` ({ls.long_pct:.1f}% L / {ls.short_pct:.1f}% S)
  Crowd: *{ls.crowd_position}*
  Score: `{ls.score:.1f}` (weighted: `{ls.score * WEIGHT_LS_RATIO:.1f}`)

*Taker Volume:*
  Signal: {emoji.get(taker.signal, '❓')} {taker.signal}
  Ratio: `{taker.buy_sell_ratio:.2f}`
  Action: *{taker.market_action}*
  Score: `{taker.score:.1f}` (weighted: `{taker.score * WEIGHT_TAKER:.1f}`)

💡 {signal.reason}
"""

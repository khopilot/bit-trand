"""
Open Interest Analyzer - Detect Position Building & Liquidations

Open Interest logic:
- OI increasing + Price up = New longs entering (trend continuation)
- OI increasing + Price down = New shorts entering (trend continuation)
- OI decreasing + Price down = Longs liquidating (potential bottom)
- OI decreasing + Price up = Shorts liquidating (potential top)

Key signals:
- Sharp OI drop = Liquidation event (trade the reversal)
- OI divergence from price = Positions getting trapped
- Sudden OI spike = New leverage entering (watch for squeeze)

Weight: 25% of total conviction score
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .config import (
    OI_CHANGE_SIGNIFICANT,
    OI_CHANGE_MAJOR,
    OI_LOOKBACK_HOURS,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NEUTRAL,
    WEIGHT_OI,
)
from .data_fetcher import OpenInterestData

logger = logging.getLogger(__name__)


@dataclass
class OISignal:
    """Signal from Open Interest analysis."""
    signal: str  # LONG, SHORT, or NEUTRAL
    score: float  # 0-100 contribution to conviction
    strength: str  # "extreme", "high", "moderate", "weak"
    reason: str  # Human-readable explanation
    current_oi_value: float  # In USDT
    change_1h: float
    change_4h: float
    change_24h: float
    interpretation: str  # What the OI movement means


class OIAnalyzer:
    """
    Analyzes Open Interest for position flow signals.

    Looks for:
    1. Sharp OI drops = Liquidation events
    2. OI divergences from price
    3. Major OI buildups = Leverage accumulating
    """

    def __init__(self):
        self.last_signal: Optional[OISignal] = None
        self.last_price_direction: Optional[str] = None

    def analyze(
        self,
        data: OpenInterestData,
        price_change_4h: float = 0.0,
    ) -> OISignal:
        """
        Analyze Open Interest data and generate signal.

        Args:
            data: OpenInterestData from data_fetcher
            price_change_4h: % price change over 4 hours (for divergence)

        Returns:
            OISignal with direction and score
        """
        change_4h = data.change_4h
        change_24h = data.change_24h

        # Determine what the OI movement means
        interpretation = self._interpret_oi_movement(change_4h, price_change_4h)

        # Generate signal
        signal, strength, score, reason = self._evaluate_oi(
            change_4h, change_24h, price_change_4h, interpretation
        )

        result = OISignal(
            signal=signal,
            score=score,
            strength=strength,
            reason=reason,
            current_oi_value=data.current_oi_value,
            change_1h=data.change_1h,
            change_4h=change_4h,
            change_24h=change_24h,
            interpretation=interpretation,
        )

        self.last_signal = result
        return result

    def _interpret_oi_movement(
        self, oi_change: float, price_change: float
    ) -> str:
        """
        Interpret what OI + price movement combination means.

        The 4 scenarios:
        1. OI↑ + Price↑ = New longs (bullish continuation)
        2. OI↑ + Price↓ = New shorts (bearish continuation)
        3. OI↓ + Price↓ = Long liquidations (potential reversal UP)
        4. OI↓ + Price↑ = Short liquidations (potential reversal DOWN)
        """
        oi_up = oi_change > 0.01  # 1% threshold
        oi_down = oi_change < -0.01
        price_up = price_change > 0.005  # 0.5% threshold
        price_down = price_change < -0.005

        if oi_up and price_up:
            return "NEW_LONGS"
        elif oi_up and price_down:
            return "NEW_SHORTS"
        elif oi_down and price_down:
            return "LONG_LIQUIDATION"
        elif oi_down and price_up:
            return "SHORT_LIQUIDATION"
        else:
            return "CONSOLIDATION"

    def _evaluate_oi(
        self,
        change_4h: float,
        change_24h: float,
        price_change: float,
        interpretation: str,
    ) -> tuple[str, str, float, str]:
        """
        Evaluate OI data and return (signal, strength, score, reason).

        Liquidation events are the key trading signals:
        - Long liquidation (OI↓ + Price↓) = BUY signal (bottom)
        - Short liquidation (OI↓ + Price↑) = SELL signal (top)

        New position accumulation is a continuation signal but less reliable.
        """
        abs_change_4h = abs(change_4h)
        abs_change_24h = abs(change_24h)

        # MAJOR EVENT: Large OI drop = Liquidation cascade
        if change_4h < -OI_CHANGE_MAJOR:
            if interpretation == "LONG_LIQUIDATION":
                return (
                    SIGNAL_LONG,
                    "extreme",
                    90.0,
                    f"MAJOR long liquidation! OI dropped {change_4h*100:.1f}% in 4h. "
                    f"Overleveraged longs flushed. Reversal UP likely."
                )
            elif interpretation == "SHORT_LIQUIDATION":
                return (
                    SIGNAL_SHORT,
                    "extreme",
                    90.0,
                    f"MAJOR short liquidation! OI dropped {change_4h*100:.1f}% in 4h. "
                    f"Shorts squeezed out. Reversal DOWN possible."
                )

        # SIGNIFICANT EVENT: Notable OI drop
        if change_4h < -OI_CHANGE_SIGNIFICANT:
            if interpretation == "LONG_LIQUIDATION":
                score = 50 + (abs(change_4h) / OI_CHANGE_MAJOR) * 40
                return (
                    SIGNAL_LONG,
                    "high",
                    min(score, 85),
                    f"Long liquidation detected. OI down {change_4h*100:.1f}% in 4h "
                    f"while price falling. Capitulation = buying opportunity."
                )
            elif interpretation == "SHORT_LIQUIDATION":
                score = 50 + (abs(change_4h) / OI_CHANGE_MAJOR) * 40
                return (
                    SIGNAL_SHORT,
                    "high",
                    min(score, 85),
                    f"Short squeeze detected. OI down {change_4h*100:.1f}% in 4h "
                    f"while price rising. Exhaustion = selling opportunity."
                )

        # DIVERGENCE: OI building in one direction
        if change_4h > OI_CHANGE_SIGNIFICANT:
            if interpretation == "NEW_LONGS":
                # New longs = bearish contrarian (they'll get liquidated)
                return (
                    SIGNAL_SHORT,
                    "moderate",
                    45.0,
                    f"New longs entering ({change_4h*100:.1f}% OI increase). "
                    f"Fresh leverage = potential liquidation fuel."
                )
            elif interpretation == "NEW_SHORTS":
                # New shorts = bullish contrarian (they'll get squeezed)
                return (
                    SIGNAL_LONG,
                    "moderate",
                    45.0,
                    f"New shorts entering ({change_4h*100:.1f}% OI increase). "
                    f"Fresh short leverage = squeeze fuel."
                )

        # ACCUMULATION: 24h OI building significantly
        if abs_change_24h > OI_CHANGE_MAJOR:
            direction = "up" if change_24h > 0 else "down"
            return (
                SIGNAL_NEUTRAL,
                "weak",
                25.0,
                f"OI {direction} {abs_change_24h*100:.1f}% over 24h. "
                f"Leverage accumulating, watch for flush."
            )

        # NEUTRAL: No significant OI movement
        return (
            SIGNAL_NEUTRAL,
            "weak",
            10.0,
            f"OI stable (4h: {change_4h*100:+.1f}%, 24h: {change_24h*100:+.1f}%). "
            f"No significant position changes detected."
        )

    def get_weighted_score(self, signal: OISignal) -> float:
        """
        Get the weighted score contribution (0-25 for 25% weight).

        Args:
            signal: OISignal from analyze()

        Returns:
            Score contribution (0 to WEIGHT_OI * 100)
        """
        return signal.score * WEIGHT_OI

    def format_report(self, signal: OISignal) -> str:
        """Format signal as readable report."""
        emoji = {
            SIGNAL_LONG: "🟢",
            SIGNAL_SHORT: "🔴",
            SIGNAL_NEUTRAL: "⚪",
        }

        interp_emoji = {
            "NEW_LONGS": "📈",
            "NEW_SHORTS": "📉",
            "LONG_LIQUIDATION": "💥",
            "SHORT_LIQUIDATION": "💥",
            "CONSOLIDATION": "↔️",
        }

        return f"""
📊 *OPEN INTEREST ANALYSIS*
━━━━━━━━━━━━━━━━━━━━━━
Signal: {emoji.get(signal.signal, '❓')} *{signal.signal}* ({signal.strength})
Score: `{signal.score:.1f}/100` (weighted: `{self.get_weighted_score(signal):.1f}`)

OI Value: `${signal.current_oi_value/1e9:.2f}B`
1h Change: `{signal.change_1h*100:+.2f}%`
4h Change: `{signal.change_4h*100:+.2f}%`
24h Change: `{signal.change_24h*100:+.2f}%`

{interp_emoji.get(signal.interpretation, '❓')} *{signal.interpretation.replace('_', ' ')}*

💡 {signal.reason}
"""

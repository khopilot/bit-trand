"""
Strategy Engine for BTC Elite Trader - EXPERT EDITION

Production-grade signal generation with:
- Regime-aware trading (only trade favorable conditions)
- Multi-timeframe confirmation
- Fractional Kelly position sizing
- Dynamic confidence scoring

Expert traders adapt to market conditions. This engine does too.

Author: khopilot
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .models import Position, Signal, SignalType, StrategyConfig
from .regime_detector import RegimeDetector, RegimeState, TrendRegime, VolatilityRegime

logger = logging.getLogger("btc_trader.strategy_engine")


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    position_usd: float
    btc_quantity: float
    stop_price: float
    kelly_fraction: float
    size_multiplier: float
    risk_amount: float
    reason: str


class StrategyEngine:
    """
    Expert trading signal generator with regime awareness.

    Enhancements over basic version:
    - Regime filtering (don't trade choppy/ranging markets)
    - Multi-timeframe trend confirmation
    - Fractional Kelly position sizing
    - Dynamic confidence scoring based on conditions
    - FNG used for size adjustment, not entry/exit
    """

    def __init__(
        self,
        config: StrategyConfig,
        kelly_fraction: float = 0.25,  # Quarter Kelly
        use_regime_filter: bool = True,
        use_mtf_filter: bool = True,
        min_confidence: float = 0.6,
    ):
        """
        Initialize StrategyEngine.

        Args:
            config: Strategy configuration parameters
            kelly_fraction: Fraction of Kelly to use (0.25 = Quarter Kelly)
            use_regime_filter: Enable regime-based filtering
            use_mtf_filter: Enable multi-timeframe confirmation
            min_confidence: Minimum confidence to generate signal
        """
        self.config = config
        self.kelly_fraction = kelly_fraction
        self.use_regime_filter = use_regime_filter
        self.use_mtf_filter = use_mtf_filter
        self.min_confidence = min_confidence

        # Initialize regime detector
        self.regime_detector = RegimeDetector()

        # Track recent performance for dynamic Kelly
        self._recent_wins: int = 0
        self._recent_trades: int = 0
        self._rolling_win_rate: float = 0.5

        # Multi-timeframe state
        self._daily_trend: Optional[str] = None
        self._4h_trend: Optional[str] = None

        logger.info(
            "StrategyEngine initialized: kelly=%.2f, regime_filter=%s, mtf_filter=%s",
            kelly_fraction,
            use_regime_filter,
            use_mtf_filter,
        )

    def update_mtf_trends(
        self,
        daily_ema_fast: float,
        daily_ema_slow: float,
        h4_ema_fast: Optional[float] = None,
        h4_ema_slow: Optional[float] = None,
    ) -> None:
        """
        Update multi-timeframe trend information.

        Args:
            daily_ema_fast: Daily EMA fast
            daily_ema_slow: Daily EMA slow
            h4_ema_fast: 4H EMA fast (optional)
            h4_ema_slow: 4H EMA slow (optional)
        """
        self._daily_trend = "up" if daily_ema_fast > daily_ema_slow else "down"

        if h4_ema_fast is not None and h4_ema_slow is not None:
            self._4h_trend = "up" if h4_ema_fast > h4_ema_slow else "down"

    def update_performance(self, win: bool) -> None:
        """Update rolling win rate for dynamic Kelly."""
        self._recent_trades += 1
        if win:
            self._recent_wins += 1

        # Use last 20 trades for rolling win rate
        if self._recent_trades >= 20:
            self._rolling_win_rate = self._recent_wins / self._recent_trades
            # Reset counters
            self._recent_wins = int(self._rolling_win_rate * 10)
            self._recent_trades = 10

    def generate_signal(
        self,
        df: pd.DataFrame,
        position: Optional[Position],
        fng_value: int = 50,
        regime_state: Optional[RegimeState] = None,
    ) -> Signal:
        """
        Generate trading signal with regime awareness.

        Args:
            df: DataFrame with OHLCV and indicator data
            position: Current open position (if any)
            fng_value: Fear & Greed Index value (0-100)
            regime_state: Pre-computed regime state (optional)

        Returns:
            Signal with type and metadata
        """
        if df.empty or len(df) < 2:
            return Signal(
                signal_type=SignalType.NONE,
                price=0.0,
                timestamp=datetime.utcnow(),
                reason="Insufficient data",
            )

        # Get current row data
        current = df.iloc[-1]
        previous = df.iloc[-2]

        price = float(current["Close"])
        ema_fast = float(current["EMA_12"])
        ema_slow = float(current["EMA_26"])
        rsi = float(current["RSI"])
        bb_upper = float(current.get("BB_Upper", price * 1.02))
        bb_lower = float(current.get("BB_Lower", price * 0.98))
        atr = float(current.get("ATR", price * 0.02))

        prev_ema_fast = float(previous["EMA_12"])
        prev_ema_slow = float(previous["EMA_26"])

        timestamp = datetime.utcnow()

        # Detect regime if not provided
        if regime_state is None and self.use_regime_filter:
            regime_state = self.regime_detector.detect(df)

        indicators = {
            "price": price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "atr": atr,
            "fng": fng_value,
        }

        if regime_state:
            indicators["regime"] = regime_state.to_dict()

        # === REGIME FILTER ===
        # Always allow exits, but filter entries
        if position is None or not position.is_open:
            if self.use_regime_filter and regime_state and not regime_state.should_trade:
                return Signal(
                    signal_type=SignalType.NONE,
                    price=price,
                    timestamp=timestamp,
                    reason=f"Regime filter: {regime_state.reason}",
                    indicators=indicators,
                )

        # === EXIT SIGNALS ===
        if position and position.is_open:
            signal = self._check_exit_signals(
                price=price,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                prev_ema_fast=prev_ema_fast,
                prev_ema_slow=prev_ema_slow,
                rsi=rsi,
                bb_upper=bb_upper,
                fng_value=fng_value,
                position=position,
                atr=atr,
                timestamp=timestamp,
                indicators=indicators,
            )
            if signal.signal_type != SignalType.NONE:
                return signal

        # === ENTRY SIGNALS ===
        if position is None or not position.is_open:
            signal = self._check_entry_signals(
                price=price,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rsi=rsi,
                bb_lower=bb_lower,
                fng_value=fng_value,
                atr=atr,
                timestamp=timestamp,
                indicators=indicators,
                regime_state=regime_state,
            )
            if signal.signal_type != SignalType.NONE:
                # Apply MTF filter
                if self.use_mtf_filter and self._daily_trend:
                    if signal.is_buy and self._daily_trend == "down":
                        return Signal(
                            signal_type=SignalType.NONE,
                            price=price,
                            timestamp=timestamp,
                            reason="MTF filter: Daily trend is down, avoiding long entry",
                            indicators=indicators,
                        )

                # Check minimum confidence
                if signal.confidence < self.min_confidence:
                    return Signal(
                        signal_type=SignalType.NONE,
                        price=price,
                        timestamp=timestamp,
                        reason=f"Confidence too low: {signal.confidence:.2f} < {self.min_confidence}",
                        indicators=indicators,
                    )

                return signal

        return Signal(
            signal_type=SignalType.NONE,
            price=price,
            timestamp=timestamp,
            reason="No signal",
            indicators=indicators,
        )

    def _check_exit_signals(
        self,
        price: float,
        ema_fast: float,
        ema_slow: float,
        prev_ema_fast: float,
        prev_ema_slow: float,
        rsi: float,
        bb_upper: float,
        fng_value: int,
        position: Position,
        atr: float,
        timestamp: datetime,
        indicators: dict,
    ) -> Signal:
        """Check for exit conditions."""

        # Update trailing stop
        stop_price = position.update_trailing_stop(
            current_price=price,
            atr=atr,
            multiplier=self.config.trailing_stop_atr_multiplier,
            min_pct=self.config.min_stop_pct,
        )

        # 1. Trailing stop hit (highest priority)
        if price < stop_price:
            return Signal(
                signal_type=SignalType.SELL_TRAILING_STOP,
                price=price,
                timestamp=timestamp,
                confidence=1.0,
                reason=f"Trailing stop hit at ${stop_price:,.0f}",
                indicators={**indicators, "stop_price": stop_price},
            )

        # 2. Trend reversal (death cross)
        trend_reversal = (prev_ema_fast >= prev_ema_slow) and (ema_fast < ema_slow)
        if trend_reversal:
            return Signal(
                signal_type=SignalType.SELL_REVERSAL,
                price=price,
                timestamp=timestamp,
                confidence=0.9,
                reason="EMA trend reversal (death cross)",
                indicators=indicators,
            )

        # 3. Blow-off top (extreme conditions)
        blow_off_top = (
            (price > bb_upper)
            and (rsi > self.config.rsi_overbought)
            and (fng_value > self.config.fng_greed_threshold)
        )
        if blow_off_top:
            return Signal(
                signal_type=SignalType.SELL_BLOWOFF,
                price=price,
                timestamp=timestamp,
                confidence=0.95,
                reason=f"Blow-off top: price>${bb_upper:,.0f}, RSI={rsi:.0f}, FNG={fng_value}",
                indicators=indicators,
            )

        return Signal(
            signal_type=SignalType.NONE,
            price=price,
            timestamp=timestamp,
            indicators=indicators,
        )

    def _check_entry_signals(
        self,
        price: float,
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        bb_lower: float,
        fng_value: int,
        atr: float,
        timestamp: datetime,
        indicators: dict,
        regime_state: Optional[RegimeState],
    ) -> Signal:
        """Check for entry conditions with dynamic confidence."""

        is_uptrend = ema_fast > ema_slow
        base_confidence = 0.5

        # Regime boosts
        regime_boost = 0.0
        if regime_state:
            if regime_state.trend in (TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND):
                regime_boost = 0.15
            elif regime_state.volatility == VolatilityRegime.NORMAL:
                regime_boost += 0.05

        # 1. Smart Trend Entry
        smart_trend_conditions = (
            is_uptrend
            and (self.config.rsi_momentum_low < rsi < self.config.rsi_momentum_high)
            and (fng_value < self.config.fng_greed_threshold)
        )

        if smart_trend_conditions:
            # Calculate confidence based on conditions
            confidence = base_confidence + regime_boost

            # RSI in sweet spot (55-65) = higher confidence
            if 55 <= rsi <= 65:
                confidence += 0.1

            # EMA spread (strong trend) = higher confidence
            ema_spread = (ema_fast - ema_slow) / ema_slow * 100
            if ema_spread > 2:
                confidence += 0.1

            # ATR reasonable = higher confidence
            atr_pct = atr / price * 100
            if 1 < atr_pct < 4:
                confidence += 0.05

            confidence = min(confidence, 1.0)

            return Signal(
                signal_type=SignalType.BUY_TREND,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Smart trend: EMA up, RSI={rsi:.0f}, FNG={fng_value}, conf={confidence:.2f}",
                indicators=indicators,
            )

        # 2. Contrarian Sniper Entry (higher bar)
        contrarian_conditions = (
            (price < bb_lower)
            and (rsi < self.config.rsi_oversold)
            and (fng_value < self.config.fng_fear_threshold)
        )

        if contrarian_conditions:
            # Contrarian entries need stronger confirmation
            confidence = base_confidence - 0.1 + regime_boost

            # Deep oversold = higher confidence
            if rsi < 25:
                confidence += 0.15
            elif rsi < 30:
                confidence += 0.1

            # Extreme fear = higher confidence
            if fng_value < 15:
                confidence += 0.15
            elif fng_value < 20:
                confidence += 0.1

            # Price significantly below BB = higher confidence
            bb_distance = (bb_lower - price) / price * 100
            if bb_distance > 3:
                confidence += 0.1

            confidence = min(max(confidence, 0.0), 1.0)

            return Signal(
                signal_type=SignalType.BUY_CONTRARIAN,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Contrarian: price<BB, RSI={rsi:.0f}, FNG={fng_value}, conf={confidence:.2f}",
                indicators=indicators,
            )

        return Signal(
            signal_type=SignalType.NONE,
            price=price,
            timestamp=timestamp,
            indicators=indicators,
        )

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        signal_confidence: float = 0.8,
        regime_multiplier: float = 1.0,
        fng_value: int = 50,
    ) -> PositionSizeResult:
        """
        Calculate position size with Fractional Kelly.

        Expert position sizing:
        1. Base size from risk per trade
        2. Adjusted by Kelly fraction
        3. Scaled by regime multiplier
        4. FNG-adjusted (reduce in greed, increase in fear)
        5. Capped at maximum position

        Args:
            capital: Total available capital
            entry_price: Entry price
            atr: Current ATR value
            signal_confidence: Signal confidence (0-1)
            regime_multiplier: Regime-based multiplier (0-1.5)
            fng_value: Fear & Greed Index (0-100)

        Returns:
            PositionSizeResult with all sizing details
        """
        # Calculate stop distance
        atr_stop_distance = atr * self.config.trailing_stop_atr_multiplier
        min_stop_distance = entry_price * self.config.min_stop_pct
        stop_distance = max(atr_stop_distance, min_stop_distance)
        stop_price = entry_price - stop_distance
        stop_pct = stop_distance / entry_price

        # Base risk amount
        base_risk = capital * self.config.risk_per_trade_pct

        # === FRACTIONAL KELLY ===
        # Kelly = (win_rate * avg_win_ratio - loss_rate) / avg_win_ratio
        # We use a simplified version based on rolling win rate

        win_rate = self._rolling_win_rate
        loss_rate = 1 - win_rate
        avg_win_ratio = 2.0  # Assume 2:1 reward/risk

        kelly_optimal = (win_rate * avg_win_ratio - loss_rate) / avg_win_ratio
        kelly_optimal = max(0, min(kelly_optimal, 0.5))  # Cap at 50%

        # Apply kelly fraction (Quarter Kelly by default)
        kelly_adjusted = kelly_optimal * self.kelly_fraction

        # === CONFIDENCE ADJUSTMENT ===
        # Scale by signal confidence
        confidence_mult = 0.5 + (signal_confidence * 0.5)  # Range: 0.5-1.0

        # === REGIME ADJUSTMENT ===
        # From regime detector
        regime_mult = regime_multiplier

        # === FNG ADJUSTMENT ===
        # Fear = increase size, Greed = decrease size
        if fng_value < 25:
            fng_mult = 1.2  # Extreme fear: 20% larger
        elif fng_value < 40:
            fng_mult = 1.1  # Fear: 10% larger
        elif fng_value > 75:
            fng_mult = 0.7  # Extreme greed: 30% smaller
        elif fng_value > 60:
            fng_mult = 0.85  # Greed: 15% smaller
        else:
            fng_mult = 1.0  # Neutral

        # === COMBINE ALL FACTORS ===
        combined_mult = confidence_mult * regime_mult * fng_mult

        # Position size from risk
        if stop_pct > 0:
            position_by_risk = base_risk / stop_pct
        else:
            position_by_risk = capital * self.config.max_position_pct * 0.5

        # Apply Kelly and multipliers
        if kelly_adjusted > 0:
            kelly_position = capital * kelly_adjusted
            # Blend risk-based and Kelly-based (favor risk-based for safety)
            position_usd = min(position_by_risk, kelly_position) * combined_mult
        else:
            position_usd = position_by_risk * combined_mult * 0.5  # Reduce if Kelly is negative

        # Apply maximum position limit
        max_position = capital * self.config.max_position_pct
        position_usd = min(position_usd, max_position)

        # Minimum viable position
        position_usd = max(position_usd, 10.0)

        # Calculate BTC quantity
        btc_quantity = position_usd / entry_price

        # Build reason string
        reason_parts = [
            f"Kelly={kelly_optimal:.2%}→{kelly_adjusted:.2%}",
            f"Conf={confidence_mult:.2f}",
            f"Regime={regime_mult:.2f}",
            f"FNG={fng_mult:.2f}",
        ]

        result = PositionSizeResult(
            position_usd=position_usd,
            btc_quantity=btc_quantity,
            stop_price=stop_price,
            kelly_fraction=kelly_adjusted,
            size_multiplier=combined_mult,
            risk_amount=base_risk,
            reason=" | ".join(reason_parts),
        )

        logger.debug(
            "Position sizing: capital=$%.0f → size=$%.0f (%.2f%%) | %s",
            capital,
            position_usd,
            position_usd / capital * 100,
            result.reason,
        )

        return result

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to execution price."""
        if is_buy:
            return price * (1 + self.config.slippage)
        else:
            return price * (1 - self.config.slippage)

    def get_strategy_state(self) -> dict:
        """Get current strategy state for debugging/monitoring."""
        return {
            "kelly_fraction": self.kelly_fraction,
            "rolling_win_rate": self._rolling_win_rate,
            "recent_trades": self._recent_trades,
            "recent_wins": self._recent_wins,
            "daily_trend": self._daily_trend,
            "4h_trend": self._4h_trend,
            "use_regime_filter": self.use_regime_filter,
            "use_mtf_filter": self.use_mtf_filter,
            "min_confidence": self.min_confidence,
        }

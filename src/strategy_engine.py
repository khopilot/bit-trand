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

# Optional ML regime detector
try:
    from .ml_regime_detector import MLRegimeDetector, ML_AVAILABLE
except ImportError:
    MLRegimeDetector = None
    ML_AVAILABLE = False

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
        use_mtf_filter: bool = False,  # DISABLED - was blocking valid entries
        min_confidence: float = 0.3,   # Lowered from 0.5 for more trades
        ml_config: Optional[dict] = None,  # ML regime detector config
    ):
        """
        Initialize StrategyEngine.

        Args:
            config: Strategy configuration parameters
            kelly_fraction: Fraction of Kelly to use (0.25 = Quarter Kelly)
            use_regime_filter: Enable regime-based filtering
            use_mtf_filter: Enable multi-timeframe confirmation
            min_confidence: Minimum confidence to generate signal
            ml_config: ML regime detector config (from config.yaml ml section)
        """
        self.config = config
        self.kelly_fraction = kelly_fraction
        self.use_regime_filter = use_regime_filter
        self.use_mtf_filter = use_mtf_filter
        self.min_confidence = min_confidence

        # Initialize regime detector (ML or rule-based)
        self.ml_enabled = False
        ml_config = ml_config or {}

        if ml_config.get("enabled", False) and MLRegimeDetector and ML_AVAILABLE:
            self.regime_detector = MLRegimeDetector(
                model_path=ml_config.get("model_path", "models/regime_classifier.pkl"),
                confidence_threshold=ml_config.get("confidence_threshold", 0.6),
                enabled=True,
            )
            self.ml_enabled = self.regime_detector.is_ml_active
            logger.info("ML Regime Detector initialized (active: %s)", self.ml_enabled)
        else:
            self.regime_detector = RegimeDetector()

        # Track recent performance for dynamic Kelly
        self._recent_wins: int = 0
        self._recent_trades: int = 0
        self._rolling_win_rate: float = 0.5

        # Multi-timeframe state
        self._daily_trend: Optional[str] = None
        self._4h_trend: Optional[str] = None

        # Smart cooldown tracking (loss-streak based, not calendar)
        self._consecutive_losses: int = 0
        self._max_consecutive_losses: int = 3  # Pause after 3 consecutive losses
        self._last_trade_pnl_pct: float = 0.0

        # ALPHA MAXIMIZER: Re-entry tracking after stop-outs
        self._last_exit_price: float = 0.0
        self._should_seek_reentry: bool = False
        self._days_since_exit: int = 0

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

    def update_performance(self, win: bool, pnl_pct: float = 0.0) -> None:
        """Update rolling win rate and consecutive loss tracking for smart cooldown."""
        self._recent_trades += 1
        if win:
            self._recent_wins += 1
            self._consecutive_losses = 0  # Reset on win
        else:
            self._consecutive_losses += 1  # Track consecutive losses

        self._last_trade_pnl_pct = pnl_pct

        # Use last 20 trades for rolling win rate
        if self._recent_trades >= 20:
            self._rolling_win_rate = self._recent_wins / self._recent_trades
            # Reset counters
            self._recent_wins = int(self._rolling_win_rate * 10)
            self._recent_trades = 10

    def record_exit(self, exit_price: float, exit_reason: str = "") -> None:
        """
        ALPHA MAXIMIZER: Record an exit for re-entry logic.

        After a stop-out or trailing stop exit, we want to re-enter
        if the trend resumes instead of sitting in cash.
        """
        self._last_exit_price = exit_price
        self._should_seek_reentry = True
        self._days_since_exit = 0
        logger.info(f"Exit recorded at ${exit_price:,.0f} ({exit_reason}) - seeking re-entry")

    def increment_days_since_exit(self) -> None:
        """Increment days counter for re-entry timing."""
        if self._should_seek_reentry:
            self._days_since_exit += 1

    def _check_reentry_signals(
        self,
        price: float,
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        regime_state: Optional["RegimeState"],
        timestamp: datetime,
        indicators: dict,
    ) -> Optional[Signal]:
        """
        ALPHA MAXIMIZER: Check for re-entry after a stop-out.

        Don't sit in cash - re-enter when trend resumes.
        """
        if not self._should_seek_reentry:
            return None

        # Don't re-enter immediately - wait at least 2 days
        if self._days_since_exit < 2:
            return None

        # Check if we've been waiting too long (30 days max)
        if self._days_since_exit > 30:
            self._should_seek_reentry = False
            logger.info("Re-entry expired after 30 days")
            return None

        # PULLBACK RE-ENTRY: Price pulled back 5%+ and is recovering
        if self._last_exit_price > 0:
            pullback_from_exit = (self._last_exit_price - price) / self._last_exit_price

            if pullback_from_exit > 0.05:  # 5%+ pullback from exit
                if ema_fast > ema_slow and rsi > 45:
                    self._should_seek_reentry = False
                    return Signal(
                        signal_type=SignalType.BUY_TREND,
                        price=price,
                        timestamp=timestamp,
                        confidence=0.70,
                        reason=f"Re-entry: {pullback_from_exit*100:.1f}% pullback, RSI={rsi:.0f}",
                        indicators=indicators,
                    )

        # BREAKOUT RE-ENTRY: Price exceeded previous exit (trend resumed)
        if price > self._last_exit_price * 1.02:  # 2% above exit
            if regime_state and regime_state.trend in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND):
                self._should_seek_reentry = False
                return Signal(
                    signal_type=SignalType.BUY_TREND,
                    price=price,
                    timestamp=timestamp,
                    confidence=0.80,
                    reason=f"Re-entry: Breakout above exit ${self._last_exit_price:,.0f}",
                    indicators=indicators,
                )

        # V-BOTTOM RE-ENTRY: Fast RSI recovery
        if rsi > 50 and self._days_since_exit < 10:
            if ema_fast > ema_slow:
                self._should_seek_reentry = False
                return Signal(
                    signal_type=SignalType.BUY_TREND,
                    price=price,
                    timestamp=timestamp,
                    confidence=0.65,
                    reason=f"Re-entry: RSI recovery to {rsi:.0f}",
                    indicators=indicators,
                )

        return None

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

        prev_close = float(previous["Close"])

        indicators = {
            "price": price,
            "prev_close": prev_close,
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
                regime_state=regime_state,
            )
            if signal.signal_type != SignalType.NONE:
                return signal

        # === ENTRY SIGNALS ===
        if position is None or not position.is_open:
            # ALPHA MAXIMIZER: Recovery mode should REDUCE entries, not BLOCK them entirely
            # Otherwise we miss bull runs after a few losses
            in_recovery = self._consecutive_losses >= self._max_consecutive_losses

            # In recovery: Only take HIGH confidence entries (regime-based)
            if in_recovery:
                # Allow entry if strong uptrend (don't miss bull runs!)
                if regime_state and regime_state.trend == TrendRegime.STRONG_UPTREND:
                    logger.info("Recovery mode: Allowing STRONG_UPTREND entry despite losses")
                    # Reset consecutive losses on strong signal
                    self._consecutive_losses = 0
                else:
                    # Increment days counter for time-based recovery
                    self._days_since_exit += 1
                    # After 30 days, auto-reset recovery mode
                    if self._days_since_exit > 30:
                        logger.info("Recovery mode: Auto-reset after 30 days")
                        self._consecutive_losses = 0
                    else:
                        return Signal(
                            signal_type=SignalType.NONE,
                            price=price,
                            timestamp=timestamp,
                            reason=f"Recovery mode: {self._consecutive_losses} losses, wait {30 - self._days_since_exit} days",
                            indicators=indicators,
                        )

            # ALPHA MAXIMIZER: Check for re-entry after stop-out (priority over normal entries)
            reentry_signal = self._check_reentry_signals(
                price=price,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rsi=rsi,
                regime_state=regime_state,
                timestamp=timestamp,
                indicators=indicators,
            )
            if reentry_signal:
                return reentry_signal

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
                    # Block LONG entries in downtrend
                    if signal.is_buy and self._daily_trend == "down":
                        return Signal(
                            signal_type=SignalType.NONE,
                            price=price,
                            timestamp=timestamp,
                            reason="MTF filter: Daily trend is down, avoiding long entry",
                            indicators=indicators,
                        )
                    # Block SHORT entries in uptrend
                    if signal.is_short and self._daily_trend == "up":
                        return Signal(
                            signal_type=SignalType.NONE,
                            price=price,
                            timestamp=timestamp,
                            reason="MTF filter: Daily trend is up, avoiding short entry",
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

    def _should_persist_position(
        self,
        price: float,
        rsi: float,
        ema_fast: float,
        ema_slow: float,
        regime_state: Optional["RegimeState"],
        unrealized_pct: float,
    ) -> tuple[bool, str]:
        """
        ALPHA MAXIMIZER: Check if position should persist despite stop being hit.

        Don't exit just because price touched stop - check if underlying trend is intact.
        If momentum is strong, widen the stop instead of exiting.
        """
        # Rule 1: Strong momentum - don't exit
        if rsi > 55 and ema_fast > ema_slow:
            return True, "Momentum intact (RSI>55, EMA bullish)"

        # Rule 2: Strong uptrend regime - only exit on extreme conditions
        if regime_state and regime_state.trend == TrendRegime.STRONG_UPTREND:
            if rsi > 45:
                return True, "Strong uptrend regime, RSI>45"

        # Rule 3: Profitable position - trail more loosely
        if unrealized_pct > 0.30 and rsi > 40:
            return True, f"Profitable (+{unrealized_pct*100:.0f}%), RSI>40"

        # Rule 4: Price above entry AND EMAs still bullish
        if unrealized_pct > 0 and ema_fast > ema_slow * 1.01:
            return True, "Above entry with bullish EMA spread"

        return False, "Exit confirmed - momentum weak"

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
        regime_state: Optional["RegimeState"] = None,
    ) -> Signal:
        """Check for exit conditions."""

        # ALPHA MAXIMIZER: Trend-adaptive trailing stops
        # Wide stops in uptrends (ride winners), tight in downtrends (protect capital)
        if regime_state:
            trend = regime_state.trend

            # UPTREND: Wide stops - ride the trend, tolerate pullbacks
            if trend in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND):
                atr_mult = 8.0   # Very wide (was 5.0)
                min_stop = 0.25  # 25% drawdown tolerance (was 0.15)

            # DOWNTREND: Tight stops - protect capital, exit quickly
            elif trend in (TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND):
                atr_mult = 3.0   # Tight
                min_stop = 0.10  # 10% max drawdown

            # RANGING: Medium stops
            else:
                atr_mult = 4.0
                min_stop = 0.15

            # MOMENTUM BOOST: If RSI > 60, widen stops even more (strong momentum)
            if rsi > 60:
                min_stop *= 1.25  # 25% wider
                atr_mult *= 1.25
        else:
            atr_mult = self.config.trailing_stop_atr_multiplier
            min_stop = self.config.min_stop_pct

        # Update trailing stop with dynamic values
        stop_price = position.update_trailing_stop(
            current_price=price,
            atr=atr,
            multiplier=atr_mult,
            min_pct=min_stop,
        )

        # === LONG POSITION EXITS ===
        if position.side == "long":
            # 0. Take-profit partial exits (REGIME-ADAPTIVE)
            unrealized_pct = (price - position.entry_price) / position.entry_price

            # Select TP levels based on market regime
            is_strong_uptrend = regime_state and regime_state.trend == TrendRegime.STRONG_UPTREND
            is_weak_uptrend = regime_state and regime_state.trend == TrendRegime.WEAK_UPTREND
            is_downtrend = regime_state and regime_state.trend in (
                TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND, TrendRegime.RANGING
            )

            if is_strong_uptrend:
                # ALPHA MAXIMIZER: In strong uptrend, HOLD 75%+ of position
                # Only take small profits at extreme levels to ride the full trend
                tp_levels = [
                    (1.00, 0.10, "TP1"),  # +100% profit = sell 10% only
                    (2.00, 0.15, "TP2"),  # +200% profit = sell 15%
                    (3.00, 0.25, "TP3"),  # +300% profit = sell 25%, keep 50% running
                ]
            elif is_downtrend:
                # BEAR MARKET: Aggressive TP to lock in gains quickly
                tp_levels = [
                    (0.08, 0.40, "TP1"),  # +8% profit = sell 40%
                    (0.15, 0.35, "TP2"),  # +15% profit = sell 35%
                    (0.25, 0.25, "TP3"),  # +25% profit = sell all remaining
                ]
            else:
                # WEAK UPTREND: Still ride the trend (was 15/30/50 - too early!)
                # Higher thresholds to capture more of bull market moves
                tp_levels = [
                    (0.40, 0.20, "TP1"),  # +40% profit = sell 20%
                    (0.80, 0.30, "TP2"),  # +80% profit = sell 30%
                    (1.20, 0.25, "TP3"),  # +120% profit = sell 25%, keep 25% running
                ]

            for tp_target, tp_qty_pct, tp_name in tp_levels:
                if unrealized_pct >= tp_target and not position.tp_executed.get(tp_name):
                    return Signal(
                        signal_type=SignalType.SELL_PARTIAL,
                        price=price,
                        timestamp=timestamp,
                        confidence=1.0,
                        reason=f"LONG {tp_name}: +{tp_target*100:.0f}% profit, sell {tp_qty_pct*100:.0f}%",
                        indicators={**indicators, "tp_quantity_pct": tp_qty_pct, "tp_name": tp_name},
                    )

            # 1. Trailing stop from HIGHS (not entry) - WITH MOMENTUM PERSISTENCE
            if price < stop_price:
                # ALPHA MAXIMIZER: Check if we should persist despite stop being hit
                should_persist, persist_reason = self._should_persist_position(
                    price=price,
                    rsi=rsi,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    regime_state=regime_state,
                    unrealized_pct=unrealized_pct,
                )

                if should_persist:
                    # Widen the stop instead of exiting - give 10% more room
                    new_stop = price * 0.90
                    position.stop_price = new_stop
                    # Log persistence (don't return signal - continue holding)
                    logger.info(f"PERSISTING position: {persist_reason}, widening stop to ${new_stop:,.0f}")
                else:
                    # Momentum weak - execute the stop
                    drawdown_from_high = (position.highest_price - price) / position.highest_price * 100
                    return Signal(
                        signal_type=SignalType.SELL_TRAILING_STOP,
                        price=price,
                        timestamp=timestamp,
                        confidence=1.0,
                        reason=f"Trailing stop hit: ${stop_price:,.0f} (high was ${position.highest_price:,.0f}, -{drawdown_from_high:.1f}%)",
                        indicators={**indicators, "stop_price": stop_price},
                    )

            # 2. Trend reversal - ALPHA MAXIMIZER: Very conservative in uptrends
            trend_reversal = (prev_ema_fast >= prev_ema_slow) and (ema_fast < ema_slow)
            ema_spread_pct = (ema_slow - ema_fast) / ema_slow * 100
            is_bear_regime = regime_state and regime_state.trend in (
                TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND
            )
            is_uptrend = regime_state and regime_state.trend in (
                TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND
            )

            # ALPHA MAXIMIZER: Only exit on reversal if conditions are EXTREME
            # In uptrends: require STRONG death cross AND weak RSI (don't exit on noise)
            if is_uptrend:
                # Much stricter: Need 3% EMA spread AND RSI < 30 to exit in uptrend
                should_exit_reversal = (
                    trend_reversal and ema_spread_pct > 3.0 and rsi < 30
                )
            else:
                # Normal conditions in non-uptrend
                should_exit_reversal = (
                    (trend_reversal and ema_spread_pct > 1.5)  # Death cross with 1.5% spread
                    or (ema_spread_pct > 3.0 and rsi < 30)     # Very strong weakness
                    or (is_bear_regime and rsi < 30)          # Bear regime with very weak RSI
                )
            if should_exit_reversal:
                return Signal(
                    signal_type=SignalType.SELL_REVERSAL,
                    price=price,
                    timestamp=timestamp,
                    confidence=0.9,
                    reason=f"Trend reversal: EMA spread={ema_spread_pct:.1f}%, RSI={rsi:.0f}, bear={is_bear_regime}",
                    indicators=indicators,
                )

            # 3. Blow-off top - RE-ENABLED with balanced thresholds
            blow_off_top = (
                (price > bb_upper * 1.02)  # 2% above upper BB
                and (rsi > 80)             # Strong overbought
                and (fng_value > 75)       # Extreme greed
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

        # === SHORT POSITION EXITS ===
        elif position.side == "short":
            bb_lower = float(indicators.get("bb_lower", price * 0.98))

            # 0. Take-profit partial exits for SHORT (profit when price drops)
            unrealized_pct = (position.entry_price - price) / position.entry_price
            tp_levels = [
                (0.01, 0.20, "TP1"),  # +1% profit = cover 20%
                (0.03, 0.30, "TP2"),  # +3% profit = cover 30%
                (0.05, 0.30, "TP3"),  # +5% profit = cover 30%, keep 20% for trailing
            ]
            for tp_target, tp_qty_pct, tp_name in tp_levels:
                if unrealized_pct >= tp_target and not position.tp_executed.get(tp_name):
                    return Signal(
                        signal_type=SignalType.SELL_PARTIAL,  # Reuse for partial cover
                        price=price,
                        timestamp=timestamp,
                        confidence=1.0,
                        reason=f"SHORT {tp_name}: +{tp_target*100:.0f}% profit, cover {tp_qty_pct*100:.0f}%",
                        indicators={**indicators, "tp_quantity_pct": tp_qty_pct, "tp_name": tp_name},
                    )

            # 1. Trailing stop hit for SHORT (price went UP above stop)
            if price > stop_price:
                return Signal(
                    signal_type=SignalType.COVER_TRAILING_STOP,
                    price=price,
                    timestamp=timestamp,
                    confidence=1.0,
                    reason=f"SHORT trailing stop hit at ${stop_price:,.0f}",
                    indicators={**indicators, "stop_price": stop_price},
                )

            # 2. Trend reversal for SHORT (golden cross = cover)
            golden_cross = (prev_ema_fast <= prev_ema_slow) and (ema_fast > ema_slow)
            if golden_cross:
                return Signal(
                    signal_type=SignalType.COVER_REVERSAL,
                    price=price,
                    timestamp=timestamp,
                    confidence=0.9,
                    reason="EMA trend reversal (golden cross) - cover short",
                    indicators=indicators,
                )

            # 3. Capitulation bottom (extreme fear = cover short)
            capitulation_bottom = (
                (price < bb_lower)
                and (rsi < self.config.rsi_oversold)
                and (fng_value < 25)  # Extreme fear
            )
            if capitulation_bottom:
                return Signal(
                    signal_type=SignalType.COVER_REVERSAL,
                    price=price,
                    timestamp=timestamp,
                    confidence=0.95,
                    reason=f"Capitulation bottom: price<${bb_lower:,.0f}, RSI={rsi:.0f}, FNG={fng_value}",
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
        """Check for entry conditions - MAXIMUM TRADES MODE."""

        is_uptrend = ema_fast > ema_slow
        base_confidence = 0.5

        # Regime boosts (optional)
        regime_boost = 0.0
        in_uptrend_regime = False
        if regime_state:
            if regime_state.trend in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND):
                regime_boost = 0.2
                in_uptrend_regime = True
            elif regime_state.trend in (TrendRegime.STRONG_DOWNTREND,):
                regime_boost = 0.1
            if regime_state.volatility == VolatilityRegime.NORMAL:
                regime_boost += 0.05

        # 1. ENTER when in uptrend regime OR price shows momentum
        # Use multiple conditions to enter EARLY in trends:
        # - Regime uptrend (price > 200 EMA)
        # - EMA crossover (12 > 26)
        # - RSI recovering from oversold (> 40 from < 30)

        # Check if price is above EMA (short or long term momentum)
        price_momentum = is_uptrend  # EMA_12 > EMA_26

        # Enter on ANY uptrend signal
        if in_uptrend_regime or price_momentum:
            confidence = 0.5 + regime_boost
            # RSI not extremely overbought
            if rsi < 80:
                confidence += 0.1
            elif rsi < 85:
                confidence += 0.05
            # Reasonable volatility
            atr_pct = atr / price * 100
            if atr_pct < 5:
                confidence += 0.05

            confidence = min(confidence, 1.0)
            regime_info = regime_state.trend.value if regime_state else "EMA-only"
            entry_reason = "regime_uptrend" if in_uptrend_regime else "EMA_crossover"
            return Signal(
                signal_type=SignalType.BUY_TREND,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Uptrend entry: {entry_reason}, {regime_info}, RSI={rsi:.0f}",
                indicators=indicators,
            )

        # 2. Smart Trend Entry (wider RSI range)
        smart_trend_conditions = (
            is_uptrend
            and (self.config.rsi_momentum_low < rsi < self.config.rsi_momentum_high)
        )
        # Note: Removed FNG filter - we want trades in all sentiment conditions

        if smart_trend_conditions:
            confidence = base_confidence + regime_boost

            # RSI in sweet spot = higher confidence
            if 45 <= rsi <= 65:
                confidence += 0.15
            elif 35 <= rsi <= 75:
                confidence += 0.1

            # EMA spread (strong trend) = higher confidence
            ema_spread = (ema_fast - ema_slow) / ema_slow * 100
            if ema_spread > 1:
                confidence += 0.1

            confidence = min(confidence, 1.0)

            return Signal(
                signal_type=SignalType.BUY_TREND,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Smart trend: EMA up, RSI={rsi:.0f}, conf={confidence:.2f}",
                indicators=indicators,
            )

        # 3. Contrarian Entry - RELAXED (ANY oversold condition triggers)
        # Enter if ANY of: price < BB, RSI < oversold, or FNG extreme fear
        price_below_bb = price < bb_lower
        rsi_oversold = rsi < self.config.rsi_oversold
        extreme_fear = fng_value < 30

        # Count how many oversold signals we have
        oversold_count = sum([price_below_bb, rsi_oversold, extreme_fear])

        if oversold_count >= 1:  # Changed from requiring ALL to requiring ANY
            confidence = 0.35 + (oversold_count * 0.15) + regime_boost

            # Deep oversold boosts
            if rsi < 30:
                confidence += 0.1
            if fng_value < 20:
                confidence += 0.1
            if price_below_bb:
                bb_distance = (bb_lower - price) / price * 100
                if bb_distance > 2:
                    confidence += 0.1

            confidence = min(max(confidence, 0.0), 1.0)

            return Signal(
                signal_type=SignalType.BUY_CONTRARIAN,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Contrarian: signals={oversold_count}/3, RSI={rsi:.0f}, FNG={fng_value}, conf={confidence:.2f}",
                indicators=indicators,
            )

        # 3. PULLBACK TO EMA Entry (high probability in trends - PRO TRADER)
        # Price pulls back to EMA in uptrend = great entry point
        pullback_to_ema = (
            is_uptrend  # EMA12 > EMA26 (confirmed uptrend)
            and price < ema_fast * 1.03  # Price within 3% of fast EMA
            and price > ema_slow * 0.97  # Price above slow EMA (not crashed)
            and 35 < rsi < 60  # Healthy pullback zone (not overbought)
        )

        if pullback_to_ema:
            confidence = 0.70 + regime_boost  # High base confidence for pullbacks

            # RSI sweet spot boosts
            if 40 <= rsi <= 55:
                confidence += 0.1  # Optimal pullback zone

            # Tight to EMA = stronger signal
            ema_distance = (ema_fast - price) / ema_fast * 100
            if ema_distance > 1:  # 1-3% pullback
                confidence += 0.1

            confidence = min(confidence, 1.0)

            return Signal(
                signal_type=SignalType.BUY_TREND,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Pullback to EMA: price {ema_distance:.1f}% from EMA12, RSI={rsi:.0f}",
                indicators=indicators,
            )

        # 4. VOLATILITY BREAKOUT Entry (catch big moves - PRO TRADER)
        prev_price = float(indicators.get("prev_close", price))
        price_change_pct = abs(price - prev_price) / prev_price if prev_price > 0 else 0
        atr_pct = atr / price

        vol_breakout = (
            price_change_pct > atr_pct * 2  # Move > 2x ATR (significant)
            and is_uptrend  # In direction of trend
            and rsi < 75  # Not overbought
            and price > prev_price  # Upward breakout
        )

        if vol_breakout:
            confidence = 0.65 + regime_boost

            # Strong breakout boosts
            if price_change_pct > atr_pct * 3:
                confidence += 0.15  # Very strong move
            elif price_change_pct > atr_pct * 2.5:
                confidence += 0.1

            # RSI confirmation
            if 50 <= rsi <= 65:
                confidence += 0.1  # Momentum confirmation

            confidence = min(confidence, 1.0)

            return Signal(
                signal_type=SignalType.BUY_TREND,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Volatility breakout: {price_change_pct*100:.1f}% move (>{atr_pct*200:.1f}% ATR)",
                indicators=indicators,
            )

        # 5. SHORT Trend Entry (downtrend with relief rally opportunity)
        is_downtrend = ema_fast < ema_slow
        bb_upper = float(indicators.get("bb_upper", price * 1.02))

        # In downtrends, short on relief rallies (FNG > 35 = not extreme fear)
        # or when RSI shows momentum recovery that's likely to fail
        short_trend_conditions = (
            is_downtrend
            and (self.config.rsi_momentum_low < rsi < self.config.rsi_momentum_high)
            and (fng_value > 35)  # Not extreme fear (relief rally)
        )

        if short_trend_conditions:
            confidence = base_confidence + regime_boost

            # RSI in upper-middle range = stronger short (overextended bounce)
            if 55 <= rsi <= 65:
                confidence += 0.15
            elif 50 <= rsi <= 55:
                confidence += 0.1

            # EMA spread (strong downtrend) = higher confidence
            ema_spread = (ema_slow - ema_fast) / ema_slow * 100
            if ema_spread > 2:
                confidence += 0.1

            # FNG recovery from fear = better short opportunity
            if 40 <= fng_value <= 55:
                confidence += 0.1  # Relief rally territory

            confidence = min(confidence, 1.0)

            return Signal(
                signal_type=SignalType.SHORT_TREND,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Short trend: EMA down, RSI={rsi:.0f}, FNG={fng_value}, conf={confidence:.2f}",
                indicators=indicators,
            )

        # 4. SHORT Contrarian Entry (overbought conditions - lower FNG threshold)
        short_contrarian_conditions = (
            (price > bb_upper)
            and (rsi > self.config.rsi_overbought)
            and (fng_value > 50)  # Neutral-to-greed (not deep fear)
        )

        if short_contrarian_conditions:
            confidence = base_confidence - 0.1 + regime_boost

            # Very overbought = higher confidence
            if rsi > 80:
                confidence += 0.15
            elif rsi > 75:
                confidence += 0.1

            # Higher greed = higher confidence for short
            if fng_value > 75:
                confidence += 0.15
            elif fng_value > 60:
                confidence += 0.1

            confidence = min(max(confidence, 0.0), 1.0)

            return Signal(
                signal_type=SignalType.SHORT_CONTRARIAN,
                price=price,
                timestamp=timestamp,
                confidence=confidence,
                reason=f"Short contrarian: price>BB, RSI={rsi:.0f}, FNG={fng_value}, conf={confidence:.2f}",
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
        Calculate position size - SIMPLIFIED for trend following.

        Uses large fixed position to capture major moves instead of Kelly.

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

        # Use config max_position_pct with regime and confidence adjustments
        base_position_pct = self.config.max_position_pct  # From config (e.g., 0.25)

        # Apply regime multiplier (0.5 to 1.25 based on market conditions)
        position_pct = base_position_pct * regime_multiplier

        # Apply confidence scaling (higher confidence = larger position)
        position_pct *= (0.5 + signal_confidence * 0.5)  # Scale 50-100% based on confidence

        # Cap at max_position_pct to prevent over-sizing
        position_pct = min(position_pct, self.config.max_position_pct)

        # Calculate position
        position_usd = capital * position_pct

        # Minimum viable position
        position_usd = max(position_usd, 10.0)

        # Calculate BTC quantity
        btc_quantity = position_usd / entry_price

        # Build reason string
        reason_parts = [
            f"Base={base_position_pct:.0%}",
            f"Regime={regime_multiplier:.2f}x",
            f"Conf={signal_confidence:.2f}",
            f"Final={position_pct:.1%}",
        ]

        result = PositionSizeResult(
            position_usd=position_usd,
            btc_quantity=btc_quantity,
            stop_price=stop_price,
            kelly_fraction=0.0,  # Not using Kelly
            size_multiplier=1.0,  # Fixed
            risk_amount=position_usd * self.config.min_stop_pct,  # Risk if stopped out
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
            "ml_enabled": self.ml_enabled,
            "ml_active": getattr(self.regime_detector, 'is_ml_active', False),
        }

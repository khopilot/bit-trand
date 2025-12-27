"""
Regime Detector for BTC Elite Trader

Advanced market regime classification using:
- ADX (Average Directional Index) for trend strength
- Volatility regime (ATR percentile ranking)
- Market structure (higher highs/lows detection)
- Choppiness Index for ranging detection

Expert traders don't trade the same way in all conditions.
This module tells you WHEN to trade and WHEN to sit out.

Author: khopilot
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("btc_trader.regime")


class TrendRegime(Enum):
    """Trend classification."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class VolatilityRegime(Enum):
    """Volatility classification."""
    EXTREME_LOW = "extreme_low"      # < 10th percentile - breakout likely
    LOW = "low"                       # 10-30th percentile
    NORMAL = "normal"                 # 30-70th percentile
    HIGH = "high"                     # 70-90th percentile
    EXTREME_HIGH = "extreme_high"    # > 90th percentile - reduce size or sit out


class MarketStructure(Enum):
    """Market structure classification."""
    BULLISH = "bullish"       # Higher highs and higher lows
    BEARISH = "bearish"       # Lower highs and lower lows
    CONSOLIDATION = "consolidation"  # Mixed structure


@dataclass
class RegimeState:
    """Complete market regime state."""
    trend: TrendRegime
    volatility: VolatilityRegime
    structure: MarketStructure
    adx: float
    adx_trend: str  # "rising", "falling", "flat"
    atr_percentile: float
    choppiness: float
    should_trade: bool
    position_size_multiplier: float  # 0.0 to 1.5
    reason: str

    def to_dict(self) -> dict:
        return {
            "trend": self.trend.value,
            "volatility": self.volatility.value,
            "structure": self.structure.value,
            "adx": self.adx,
            "adx_trend": self.adx_trend,
            "atr_percentile": self.atr_percentile,
            "choppiness": self.choppiness,
            "should_trade": self.should_trade,
            "position_size_multiplier": self.position_size_multiplier,
            "reason": self.reason,
        }


class RegimeDetector:
    """
    Market regime detection for adaptive trading.

    Uses multiple indicators to classify:
    1. Trend strength (ADX)
    2. Trend direction (+DI/-DI)
    3. Volatility regime (ATR percentile)
    4. Market structure (swing highs/lows)
    5. Choppiness (ranging detection)
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_strong_threshold: float = 25.0,
        adx_weak_threshold: float = 20.0,
        atr_lookback: int = 100,
        chop_period: int = 14,
        chop_threshold: float = 61.8,  # Fibonacci level
        swing_lookback: int = 10,
    ):
        """
        Initialize RegimeDetector.

        Args:
            adx_period: Period for ADX calculation
            adx_strong_threshold: ADX above this = strong trend
            adx_weak_threshold: ADX below this = no trend (ranging)
            atr_lookback: Days to look back for ATR percentile
            chop_period: Period for Choppiness Index
            chop_threshold: Above this = choppy/ranging market
            swing_lookback: Bars to look back for swing detection
        """
        self.adx_period = adx_period
        self.adx_strong_threshold = adx_strong_threshold
        self.adx_weak_threshold = adx_weak_threshold
        self.atr_lookback = atr_lookback
        self.chop_period = chop_period
        self.chop_threshold = chop_threshold
        self.swing_lookback = swing_lookback

    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index) with +DI and -DI.

        ADX measures trend strength (not direction):
        - ADX > 25: Strong trend
        - ADX 20-25: Weak trend
        - ADX < 20: No trend (ranging)
        """
        df = df.copy()

        # True Range
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(alpha=1/self.adx_period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/self.adx_period, adjust=False).mean() / atr

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.finfo(float).eps)
        adx = dx.ewm(alpha=1/self.adx_period, adjust=False).mean()

        df["ADX"] = adx
        df["Plus_DI"] = plus_di
        df["Minus_DI"] = minus_di

        return df

    def calculate_choppiness(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Choppiness Index.

        Measures if market is trending or ranging:
        - > 61.8: Choppy/ranging (Fibonacci level)
        - < 38.2: Trending
        - Between: Transitional
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Sum of TR over period
        atr_sum = tr.rolling(self.chop_period).sum()

        # Highest high and lowest low over period
        highest = high.rolling(self.chop_period).max()
        lowest = low.rolling(self.chop_period).min()

        # Choppiness Index
        chop = 100 * np.log10(atr_sum / (highest - lowest).replace(0, np.finfo(float).eps)) / np.log10(self.chop_period)

        return chop

    def detect_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """
        Detect market structure from swing highs/lows.

        Bullish: Higher highs AND higher lows
        Bearish: Lower highs AND lower lows
        Consolidation: Mixed
        """
        if len(df) < self.swing_lookback * 3:
            return MarketStructure.CONSOLIDATION

        highs = df["High"].values
        lows = df["Low"].values

        # Find swing highs (local maxima)
        swing_highs = []
        swing_lows = []

        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            # Swing high: higher than surrounding bars
            if highs[i] == max(highs[i-self.swing_lookback:i+self.swing_lookback+1]):
                swing_highs.append((i, highs[i]))
            # Swing low: lower than surrounding bars
            if lows[i] == min(lows[i-self.swing_lookback:i+self.swing_lookback+1]):
                swing_lows.append((i, lows[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketStructure.CONSOLIDATION

        # Compare last two swing points
        last_two_highs = [sh[1] for sh in swing_highs[-2:]]
        last_two_lows = [sl[1] for sl in swing_lows[-2:]]

        higher_highs = last_two_highs[-1] > last_two_highs[-2]
        higher_lows = last_two_lows[-1] > last_two_lows[-2]
        lower_highs = last_two_highs[-1] < last_two_highs[-2]
        lower_lows = last_two_lows[-1] < last_two_lows[-2]

        if higher_highs and higher_lows:
            return MarketStructure.BULLISH
        elif lower_highs and lower_lows:
            return MarketStructure.BEARISH
        else:
            return MarketStructure.CONSOLIDATION

    def get_atr_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current ATR as percentile of historical ATR."""
        if "ATR" not in df.columns or len(df) < self.atr_lookback:
            return 50.0

        current_atr = df["ATR"].iloc[-1]
        historical_atr = df["ATR"].tail(self.atr_lookback)

        percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
        return float(percentile)

    def classify_trend(self, adx: float, plus_di: float, minus_di: float) -> TrendRegime:
        """Classify trend based on ADX and DI values."""
        is_bullish = plus_di > minus_di

        if adx >= self.adx_strong_threshold:
            return TrendRegime.STRONG_UPTREND if is_bullish else TrendRegime.STRONG_DOWNTREND
        elif adx >= self.adx_weak_threshold:
            return TrendRegime.WEAK_UPTREND if is_bullish else TrendRegime.WEAK_DOWNTREND
        else:
            return TrendRegime.RANGING

    def classify_volatility(self, atr_percentile: float) -> VolatilityRegime:
        """Classify volatility based on ATR percentile."""
        if atr_percentile < 10:
            return VolatilityRegime.EXTREME_LOW
        elif atr_percentile < 30:
            return VolatilityRegime.LOW
        elif atr_percentile < 70:
            return VolatilityRegime.NORMAL
        elif atr_percentile < 90:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME_HIGH

    def get_adx_trend(self, df: pd.DataFrame, lookback: int = 5) -> str:
        """Determine if ADX is rising, falling, or flat."""
        if "ADX" not in df.columns or len(df) < lookback:
            return "flat"

        recent_adx = df["ADX"].tail(lookback).values
        slope = (recent_adx[-1] - recent_adx[0]) / lookback

        if slope > 0.5:
            return "rising"
        elif slope < -0.5:
            return "falling"
        else:
            return "flat"

    def should_trade(
        self,
        trend: TrendRegime,
        volatility: VolatilityRegime,
        structure: MarketStructure,
        choppiness: float,
    ) -> Tuple[bool, float, str]:
        """
        Determine if we should trade and position size multiplier.

        Returns:
            Tuple of (should_trade, size_multiplier, reason)
        """
        # Extreme volatility - sit out
        if volatility == VolatilityRegime.EXTREME_HIGH:
            return False, 0.0, "Extreme volatility - sitting out"

        # Very choppy market - sit out
        if choppiness > 70:
            return False, 0.0, f"Very choppy market (CHOP={choppiness:.1f}) - sitting out"

        # No trend (ranging) - reduce size significantly or sit out
        if trend == TrendRegime.RANGING:
            if choppiness > self.chop_threshold:
                return False, 0.0, "Ranging market with high choppiness"
            else:
                return True, 0.5, "Ranging but not choppy - reduced size"

        # Strong trend aligned with structure - full size or more
        if trend in (TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND):
            if structure == MarketStructure.BULLISH and trend == TrendRegime.STRONG_UPTREND:
                return True, 1.25, "Strong uptrend confirmed by structure"
            elif structure == MarketStructure.BEARISH and trend == TrendRegime.STRONG_DOWNTREND:
                return True, 1.25, "Strong downtrend confirmed by structure"
            else:
                return True, 1.0, "Strong trend"

        # Weak trend - normal size if structure confirms
        if trend in (TrendRegime.WEAK_UPTREND, TrendRegime.WEAK_DOWNTREND):
            if volatility == VolatilityRegime.HIGH:
                return True, 0.75, "Weak trend with high volatility - reduced size"
            else:
                return True, 1.0, "Weak trend - normal size"

        # Default
        return True, 1.0, "Normal conditions"

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            RegimeState with complete classification
        """
        if len(df) < max(self.adx_period * 2, self.atr_lookback, self.chop_period * 2):
            return RegimeState(
                trend=TrendRegime.RANGING,
                volatility=VolatilityRegime.NORMAL,
                structure=MarketStructure.CONSOLIDATION,
                adx=0.0,
                adx_trend="flat",
                atr_percentile=50.0,
                choppiness=50.0,
                should_trade=False,
                position_size_multiplier=0.0,
                reason="Insufficient data for regime detection",
            )

        # Calculate indicators
        df = self.calculate_adx(df)
        choppiness = self.calculate_choppiness(df)

        # Get current values
        current_adx = float(df["ADX"].iloc[-1])
        current_plus_di = float(df["Plus_DI"].iloc[-1])
        current_minus_di = float(df["Minus_DI"].iloc[-1])
        current_chop = float(choppiness.iloc[-1]) if not pd.isna(choppiness.iloc[-1]) else 50.0

        # Classify
        trend = self.classify_trend(current_adx, current_plus_di, current_minus_di)
        atr_percentile = self.get_atr_percentile(df)
        volatility = self.classify_volatility(atr_percentile)
        structure = self.detect_market_structure(df)
        adx_trend = self.get_adx_trend(df)

        # Determine trading decision
        should_trade, size_mult, reason = self.should_trade(trend, volatility, structure, current_chop)

        # Adjust for volatility
        if volatility == VolatilityRegime.HIGH:
            size_mult *= 0.75
            reason += " (volatility adjustment)"
        elif volatility == VolatilityRegime.EXTREME_LOW:
            # Low volatility often precedes breakout - be ready but careful
            size_mult *= 0.9
            reason += " (pre-breakout volatility)"

        return RegimeState(
            trend=trend,
            volatility=volatility,
            structure=structure,
            adx=current_adx,
            adx_trend=adx_trend,
            atr_percentile=atr_percentile,
            choppiness=current_chop,
            should_trade=should_trade,
            position_size_multiplier=size_mult,
            reason=reason,
        )

    def get_regime_summary(self, state: RegimeState) -> str:
        """Get human-readable regime summary."""
        trade_status = "✅ TRADE" if state.should_trade else "❌ SIT OUT"

        return (
            f"{trade_status}\n"
            f"Trend: {state.trend.value} (ADX={state.adx:.1f} {state.adx_trend})\n"
            f"Volatility: {state.volatility.value} ({state.atr_percentile:.0f}th percentile)\n"
            f"Structure: {state.structure.value}\n"
            f"Choppiness: {state.choppiness:.1f}\n"
            f"Size Multiplier: {state.position_size_multiplier:.2f}x\n"
            f"Reason: {state.reason}"
        )

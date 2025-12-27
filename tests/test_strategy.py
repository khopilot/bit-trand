"""
Unit tests for Strategy Engine.

Tests signal generation, position sizing, and indicator calculations.

Author: khopilot
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Position, PositionStatus, SignalType, StrategyConfig
from src.strategy_engine import StrategyEngine
from src.data_service import calculate_indicators


@pytest.fixture
def default_config():
    """Default strategy configuration."""
    return StrategyConfig(
        ema_fast=12,
        ema_slow=26,
        rsi_period=14,
        rsi_momentum_low=50,
        rsi_momentum_high=70,
        rsi_oversold=35,
        rsi_overbought=75,
        bb_period=20,
        bb_std=2.0,
        trailing_stop_atr_multiplier=3.0,
        min_stop_pct=0.08,
        slippage=0.001,
        max_position_pct=0.25,
        risk_per_trade_pct=0.01,
        fng_greed_threshold=80,
        fng_fear_threshold=25,
        fng_default=50,
    )


@pytest.fixture
def strategy_engine(default_config):
    """Strategy engine with default config."""
    return StrategyEngine(default_config)


@pytest.fixture
def sample_df():
    """Sample OHLCV DataFrame with indicators."""
    np.random.seed(42)

    # Generate 100 days of synthetic price data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = 40000 + np.cumsum(np.random.randn(100) * 500)

    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": prices * 0.998,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, 100),
    })

    # Calculate indicators
    config = {"strategy": {"ema_fast": 12, "ema_slow": 26, "rsi_period": 14, "bb_period": 20, "bb_std": 2}}
    return calculate_indicators(df, config)


class TestStrategyEngine:
    """Tests for StrategyEngine."""

    def test_no_signal_with_empty_df(self, strategy_engine):
        """Should return NONE signal for empty DataFrame."""
        signal = strategy_engine.generate_signal(pd.DataFrame(), None)
        assert signal.signal_type == SignalType.NONE

    def test_no_signal_with_insufficient_data(self, strategy_engine):
        """Should return NONE signal with insufficient data."""
        df = pd.DataFrame({"Close": [40000], "EMA_12": [40000], "EMA_26": [40000], "RSI": [50]})
        signal = strategy_engine.generate_signal(df, None)
        assert signal.signal_type == SignalType.NONE

    def test_buy_signal_uptrend(self, strategy_engine, sample_df):
        """Should generate BUY signal in uptrend with valid RSI."""
        # Modify data to create buy conditions
        df = sample_df.copy()
        df.loc[df.index[-1], "EMA_12"] = 45000
        df.loc[df.index[-1], "EMA_26"] = 44000
        df.loc[df.index[-1], "RSI"] = 60  # Between 50-70
        df.loc[df.index[-2], "EMA_12"] = 44500
        df.loc[df.index[-2], "EMA_26"] = 44000

        signal = strategy_engine.generate_signal(df, None, fng_value=50)

        assert signal.signal_type == SignalType.BUY_TREND
        assert signal.is_buy

    def test_no_buy_when_fng_too_high(self, strategy_engine, sample_df):
        """Should not buy when Fear & Greed is too high."""
        df = sample_df.copy()
        df.loc[df.index[-1], "EMA_12"] = 45000
        df.loc[df.index[-1], "EMA_26"] = 44000
        df.loc[df.index[-1], "RSI"] = 60
        df.loc[df.index[-2], "EMA_12"] = 44500
        df.loc[df.index[-2], "EMA_26"] = 44000

        signal = strategy_engine.generate_signal(df, None, fng_value=85)  # Extreme greed

        assert signal.signal_type == SignalType.NONE

    def test_sell_signal_trend_reversal(self, strategy_engine, sample_df):
        """Should generate SELL signal on trend reversal."""
        df = sample_df.copy()
        # Previous: EMA12 > EMA26
        df.loc[df.index[-2], "EMA_12"] = 45000
        df.loc[df.index[-2], "EMA_26"] = 44000
        # Current: EMA12 < EMA26 (death cross)
        df.loc[df.index[-1], "EMA_12"] = 43500
        df.loc[df.index[-1], "EMA_26"] = 44000
        df.loc[df.index[-1], "Close"] = 43500
        df.loc[df.index[-1], "ATR"] = 1000

        position = Position(
            entry_price=42000,
            quantity=0.1,
            highest_price=45000,
            status=PositionStatus.OPEN,
        )

        signal = strategy_engine.generate_signal(df, position)

        assert signal.signal_type == SignalType.SELL_REVERSAL
        assert signal.is_sell

    def test_trailing_stop_hit(self, strategy_engine, sample_df):
        """Should generate SELL signal when trailing stop is hit."""
        df = sample_df.copy()
        df.loc[df.index[-1], "Close"] = 40000
        df.loc[df.index[-1], "ATR"] = 1000
        df.loc[df.index[-1], "EMA_12"] = 41000
        df.loc[df.index[-1], "EMA_26"] = 40500
        df.loc[df.index[-2], "EMA_12"] = 41500
        df.loc[df.index[-2], "EMA_26"] = 41000

        position = Position(
            entry_price=45000,
            quantity=0.1,
            highest_price=47000,  # Was at 47000, now at 40000
            stop_price=44000,
            status=PositionStatus.OPEN,
        )

        signal = strategy_engine.generate_signal(df, position)

        assert signal.signal_type == SignalType.SELL_TRAILING_STOP
        assert signal.is_sell


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_position_size_respects_max(self, strategy_engine):
        """Position size should not exceed max percentage."""
        capital = 10000
        entry = 50000
        atr = 2000

        position_usd, quantity, stop = strategy_engine.calculate_position_size(
            capital=capital,
            entry_price=entry,
            atr=atr,
        )

        max_allowed = capital * strategy_engine.config.max_position_pct
        assert position_usd <= max_allowed

    def test_position_size_with_tight_stop(self, strategy_engine):
        """Tight stop should result in smaller position."""
        capital = 10000
        entry = 50000
        atr = 500  # Small ATR = tight stop

        position_usd, quantity, stop = strategy_engine.calculate_position_size(
            capital=capital,
            entry_price=entry,
            atr=atr,
        )

        # With min_stop_pct of 8%, stop should be at least 8% below entry
        min_stop_distance = entry * strategy_engine.config.min_stop_pct
        actual_stop_distance = entry - stop
        assert actual_stop_distance >= min_stop_distance

    def test_slippage_applied(self, strategy_engine):
        """Slippage should increase buy price, decrease sell price."""
        price = 50000

        buy_price = strategy_engine.apply_slippage(price, is_buy=True)
        sell_price = strategy_engine.apply_slippage(price, is_buy=False)

        assert buy_price > price
        assert sell_price < price
        assert abs(buy_price - price) == abs(sell_price - price)


class TestIndicators:
    """Tests for indicator calculations."""

    def test_ema_calculation(self, sample_df):
        """EMA columns should be present and valid."""
        assert "EMA_12" in sample_df.columns
        assert "EMA_26" in sample_df.columns
        assert not sample_df["EMA_12"].isna().all()

    def test_rsi_bounds(self, sample_df):
        """RSI should be between 0 and 100."""
        assert sample_df["RSI"].min() >= 0
        assert sample_df["RSI"].max() <= 100

    def test_bollinger_bands(self, sample_df):
        """Bollinger Bands should have proper relationship."""
        # Upper > Mid > Lower
        assert (sample_df["BB_Upper"] >= sample_df["BB_Mid"]).all()
        assert (sample_df["BB_Mid"] >= sample_df["BB_Lower"]).all()

    def test_atr_positive(self, sample_df):
        """ATR should be positive."""
        assert (sample_df["ATR"].dropna() >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

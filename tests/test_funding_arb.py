"""
Unit tests for Funding Arbitrage Module.

Tests rate monitoring, yield calculation, position management, and delta hedging.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.funding_arbitrage import (
    FundingRateMonitor,
    FundingRate,
    FundingSignal,
    YieldCalculator,
    PositionManager,
    ArbPosition,
    PositionStatus,
    DeltaHedger,
    HedgeAction,
    OrderSide,
)


class TestFundingRateMonitor:
    """Tests for FundingRateMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a funding rate monitor."""
        return FundingRateMonitor(
            min_funding_rate=0.0005,
            entry_threshold=0.001,
            exit_threshold=-0.0001,
        )

    def test_initialization(self, monitor):
        """Should initialize with correct thresholds."""
        assert monitor.min_funding_rate == 0.0005
        assert monitor.entry_threshold == 0.001
        assert monitor.exit_threshold == -0.0001

    def test_funding_rate_properties(self):
        """FundingRate should calculate rate_pct and annualized correctly."""
        rate = FundingRate(
            exchange="binance",
            symbol="BTCUSDT",
            rate=0.0001,  # 0.01%
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc),
        )

        assert rate.rate_pct == 0.01
        # Annualized: 0.0001 * 3 * 365 * 100 = 10.95%
        assert abs(rate.annualized_rate - 10.95) < 0.01

    def test_generate_signal_no_position_high_rate(self, monitor):
        """Should generate ENTER signal when rate is above threshold."""
        rate = FundingRate(
            exchange="binance",
            symbol="BTCUSDT",
            rate=0.0015,  # 0.15% - above entry threshold
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc),
        )

        signal = monitor.generate_signal(current_rate=rate, has_position=False)

        assert signal.action == "enter"
        assert signal.rate == 0.0015
        assert signal.confidence > 0.5

    def test_generate_signal_no_position_low_rate(self, monitor):
        """Should generate HOLD signal when rate is below entry threshold."""
        rate = FundingRate(
            exchange="binance",
            symbol="BTCUSDT",
            rate=0.0003,  # 0.03% - below minimum
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc),
        )

        signal = monitor.generate_signal(current_rate=rate, has_position=False)

        assert signal.action == "hold"
        assert "too low" in signal.reason.lower()

    def test_generate_signal_with_position_positive_rate(self, monitor):
        """Should generate HOLD signal when position open and rate is positive."""
        rate = FundingRate(
            exchange="binance",
            symbol="BTCUSDT",
            rate=0.0008,  # 0.08% - above min
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc),
        )

        signal = monitor.generate_signal(current_rate=rate, has_position=True)

        assert signal.action == "hold"
        assert "holding" in signal.reason.lower()

    def test_generate_signal_with_position_negative_rate(self, monitor):
        """Should generate EXIT signal when rate is negative."""
        rate = FundingRate(
            exchange="binance",
            symbol="BTCUSDT",
            rate=-0.0002,  # -0.02% - below exit threshold
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc),
        )

        signal = monitor.generate_signal(current_rate=rate, has_position=True)

        assert signal.action == "exit"
        assert "negative" in signal.reason.lower()

    def test_rate_trend_calculation(self, monitor):
        """Should correctly identify rate trends."""
        # Add increasing rates to history
        now = datetime.now(timezone.utc)
        for i in range(10):
            rate = FundingRate(
                exchange="binance",
                symbol="BTCUSDT",
                rate=0.0001 + (i * 0.0001),  # Increasing
                next_funding_time=now,
                timestamp=now + timedelta(hours=i * 8),
            )
            monitor._add_to_history(rate)

        trend, avg = monitor.get_rate_trend(lookback=10)

        assert trend == "increasing"
        assert avg > 0


class TestYieldCalculator:
    """Tests for YieldCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a yield calculator."""
        return YieldCalculator(initial_capital=10000.0)

    def test_initialization(self, calculator):
        """Should initialize with correct capital."""
        assert calculator.initial_capital == 10000.0
        assert calculator.current_capital == 10000.0

    def test_record_funding_payment(self, calculator):
        """Should correctly record funding payments."""
        calculator.record_funding_payment(
            amount=10.0,
            rate=0.001,
            position_size=0.1,
            notional_value=9500,
        )

        assert len(calculator._payments) == 1
        assert calculator.get_total_funding() == 10.0
        assert calculator.current_capital == 10010.0

    def test_get_return_pct(self, calculator):
        """Should calculate return percentage correctly."""
        # Add some earnings
        calculator.record_funding_payment(
            amount=100.0,
            rate=0.001,
            position_size=0.1,
            notional_value=9500,
        )

        assert calculator.get_return_pct() == 1.0  # 100/10000 = 1%

    def test_calculate_projected_yield(self, calculator):
        """Should project yield correctly."""
        projection = calculator.calculate_projected_yield(
            position_size_btc=0.1,
            current_rate=0.001,  # 0.1%
            days=30,
        )

        assert projection["position_btc"] == 0.1
        assert projection["current_rate"] == 0.001
        assert projection["projection_days"] == 30
        assert projection["projected_yield"] > 0
        assert projection["projected_apy"] > 0

    def test_backtest_historical_rates(self, calculator):
        """Should backtest on historical rates."""
        # Create sample historical rates
        historical = [
            {"timestamp": datetime.now(timezone.utc), "rate": 0.0005},
            {"timestamp": datetime.now(timezone.utc), "rate": 0.0008},
            {"timestamp": datetime.now(timezone.utc), "rate": 0.0010},
            {"timestamp": datetime.now(timezone.utc), "rate": -0.0002},  # Negative
            {"timestamp": datetime.now(timezone.utc), "rate": 0.0006},
        ]

        result = calculator.backtest_historical_rates(
            historical_rates=historical,
            position_size_btc=0.1,
            btc_price=95000,
        )

        assert result["periods_analyzed"] == 5
        assert result["positive_periods"] == 4  # 4 positive, 1 negative
        assert result["total_funding"] > 0

    def test_compare_to_benchmark(self, calculator):
        """Should compare to BTC buy-and-hold."""
        # Add some earnings (5% return)
        calculator.current_capital = 10500

        comparison = calculator.compare_to_benchmark(
            btc_start_price=90000,
            btc_end_price=99000,  # 10% increase
        )

        assert comparison["strategy_return_pct"] == 5.0
        assert comparison["btc_return_pct"] == 10.0
        assert comparison["alpha"] == -5.0  # Underperformed
        assert comparison["outperformed"] is False

    def test_get_performance_metrics(self, calculator):
        """Should return comprehensive metrics."""
        # Add some payments
        for i in range(5):
            calculator.record_funding_payment(
                amount=10.0,
                rate=0.001,
                position_size=0.1,
                notional_value=9500,
            )

        metrics = calculator.get_performance_metrics()

        assert metrics.total_funding_earned == 50.0
        assert metrics.avg_funding_rate == 0.001


class TestArbPosition:
    """Tests for ArbPosition dataclass."""

    def test_is_balanced(self):
        """Should correctly check if position is balanced."""
        position = ArbPosition(
            status=PositionStatus.OPEN,
            spot_quantity=0.1,
            perp_quantity=0.1,
        )
        assert position.is_balanced is True

        # Unbalanced
        position.perp_quantity = 0.09
        assert position.is_balanced is False

    def test_delta_calculation(self):
        """Should calculate delta correctly."""
        position = ArbPosition(
            spot_quantity=0.1,
            perp_quantity=0.095,
        )

        assert abs(position.delta - 0.005) < 0.0001  # 0.1 - 0.095

    def test_notional_value(self):
        """Should calculate notional value correctly."""
        position = ArbPosition(
            spot_quantity=0.1,
            spot_entry_price=95000,
            perp_entry_price=95100,
        )

        # (95000 + 95100) / 2 * 0.1 = 9505
        expected = 0.1 * ((95000 + 95100) / 2)
        assert abs(position.notional_value - expected) < 1


class TestDeltaHedger:
    """Tests for DeltaHedger."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock exchange client."""
        client = MagicMock()
        client.get_spot_price.return_value = 95000
        client.get_futures_price.return_value = 95100
        client.get_futures_position.return_value = None
        return client

    @pytest.fixture
    def mock_position_manager(self, mock_client):
        """Create a mock position manager."""
        rate_monitor = FundingRateMonitor()
        manager = PositionManager(
            client=mock_client,
            rate_monitor=rate_monitor,
        )
        return manager

    @pytest.fixture
    def hedger(self, mock_client, mock_position_manager):
        """Create a delta hedger."""
        return DeltaHedger(
            client=mock_client,
            position_manager=mock_position_manager,
            max_delta_pct=0.02,
            rebalance_threshold=0.01,
        )

    def test_check_delta_no_position(self, hedger):
        """Should return none action when no position."""
        action = hedger.check_delta()

        assert action.action_type == "none"
        assert "no open position" in action.reason.lower()

    def test_check_delta_balanced(self, hedger, mock_position_manager):
        """Should return none action when balanced."""
        mock_position_manager.position = ArbPosition(
            status=PositionStatus.OPEN,
            spot_quantity=0.1,
            perp_quantity=0.1,
        )

        action = hedger.check_delta()

        assert action.action_type == "none"
        assert "within threshold" in action.reason.lower()

    def test_check_delta_long_imbalance(self, hedger, mock_position_manager):
        """Should detect long delta imbalance."""
        mock_position_manager.position = ArbPosition(
            status=PositionStatus.OPEN,
            spot_quantity=0.1,
            perp_quantity=0.08,  # 20% less
        )

        action = hedger.check_delta()

        assert action.action_type == "rebalance_perp"
        assert action.side == OrderSide.SELL
        assert action.quantity > 0
        assert action.urgency in ("high", "critical")

    def test_check_delta_short_imbalance(self, hedger, mock_position_manager):
        """Should detect short delta imbalance."""
        mock_position_manager.position = ArbPosition(
            status=PositionStatus.OPEN,
            spot_quantity=0.08,  # 20% less
            perp_quantity=0.1,
        )

        action = hedger.check_delta()

        assert action.action_type == "rebalance_spot"
        assert action.side == OrderSide.BUY
        assert action.quantity > 0

    def test_get_status(self, hedger, mock_position_manager):
        """Should return comprehensive status."""
        mock_position_manager.position = ArbPosition(
            status=PositionStatus.OPEN,
            spot_quantity=0.1,
            perp_quantity=0.1,
        )

        status = hedger.get_status()

        assert status["position_status"] == "open"
        assert status["is_balanced"] is True
        assert "delta_action" in status


class TestPositionManager:
    """Tests for PositionManager."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock exchange client."""
        client = MagicMock()
        client.get_spot_price.return_value = 95000
        client.get_futures_price.return_value = 95100
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create a position manager."""
        rate_monitor = FundingRateMonitor()
        return PositionManager(
            client=mock_client,
            rate_monitor=rate_monitor,
            max_position_usd=10000,
            leverage=2,
        )

    def test_calculate_position_size(self, manager):
        """Should calculate position size correctly."""
        size = manager.calculate_position_size(current_price=95000)

        # 10000 / 95000 = 0.1052...
        expected = round(10000 / 95000, 4)
        assert size == expected

    def test_should_enter_no_position(self, manager):
        """Should determine entry based on funding rate."""
        # Mock the rate monitor
        with patch.object(
            manager.rate_monitor,
            "generate_signal",
            return_value=FundingSignal(
                action="enter",
                rate=0.001,
                expected_yield_24h=0.3,
                confidence=0.8,
                reason="High funding rate",
            ),
        ):
            should_enter, reason = manager.should_enter()
            assert should_enter is True

    def test_should_enter_already_open(self, manager):
        """Should not enter when position already open."""
        manager.position.status = PositionStatus.OPEN

        should_enter, reason = manager.should_enter()

        assert should_enter is False
        assert "already open" in reason.lower()

    def test_record_funding_payment(self, manager):
        """Should record funding payments."""
        manager.position = ArbPosition(
            status=PositionStatus.OPEN,
            perp_quantity=0.1,
        )

        manager.record_funding_payment(amount=5.0, rate=0.0005)

        assert len(manager.position.funding_payments) == 1
        assert manager.position.total_funding_collected == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

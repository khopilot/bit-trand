"""
Comprehensive Tests for Funding Rate Arbitrage System

Tests:
1. FundingRateFetcher - Pagination, CSV, caching
2. CostCalculator - Fee calculations
3. ComprehensiveBacktest - Multi-year analysis
4. Integration - End-to-end tests
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import os

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.funding_arbitrage.funding_rate_fetcher import (
    FundingRateFetcher,
    FundingRateRecord,
)
from src.funding_arbitrage.cost_model import (
    CostCalculator,
    CostParameters,
    TradeCost,
    get_binance_vip0_costs,
)


# ============== Fixtures ==============


@pytest.fixture
def sample_funding_rates():
    """Generate sample funding rate data for testing."""
    rates = []
    base_time = datetime(2021, 1, 1, tzinfo=timezone.utc)

    for i in range(100):
        hours_offset = i * 8  # 8-hour periods
        timestamp = base_time + timedelta(hours=hours_offset)

        # Simulate realistic rates (mostly positive, occasionally negative)
        if i % 10 == 0:
            rate = -0.0002  # Occasional negative
        elif i % 5 == 0:
            rate = 0.001  # High rate
        else:
            rate = 0.0003  # Normal positive

        rates.append(
            FundingRateRecord(
                timestamp=timestamp,
                funding_rate=rate,
                mark_price=40000 + (i * 10),
                symbol="BTCUSDT",
            )
        )

    return rates


@pytest.fixture
def multi_year_rates():
    """Generate multi-year funding rate data."""
    rates = []
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)

    # Simulate 3 years of data (3 * 365 * 3 = 3285 periods)
    for i in range(3285):
        timestamp = start + timedelta(hours=i * 8)

        # Vary rates by "era"
        year = timestamp.year
        if year == 2020:
            base_rate = 0.0005  # Bull market
        elif year == 2021:
            base_rate = 0.0008  # Peak bull
        else:
            base_rate = 0.0002  # Bear market

        # Add noise
        noise = np.random.normal(0, 0.0002)
        rate = max(-0.001, min(0.003, base_rate + noise))

        rates.append(
            FundingRateRecord(
                timestamp=timestamp,
                funding_rate=rate,
                mark_price=30000 + (i * 5),
                symbol="BTCUSDT",
            )
        )

    return rates


@pytest.fixture
def cost_params():
    """Standard cost parameters for testing."""
    return CostParameters(
        maker_fee=0.0002,
        taker_fee=0.0004,
        entry_slippage=0.0005,
        exit_slippage=0.0005,
    )


@pytest.fixture
def mock_binance_response():
    """Mock Binance API response."""
    return [
        {
            "symbol": "BTCUSDT",
            "fundingTime": 1609459200000,  # 2021-01-01 00:00:00 UTC
            "fundingRate": "0.00030000",
            "markPrice": "29000.00000000",
        },
        {
            "symbol": "BTCUSDT",
            "fundingTime": 1609488000000,  # 2021-01-01 08:00:00 UTC
            "fundingRate": "0.00025000",
            "markPrice": "29100.00000000",
        },
    ]


# ============== FundingRateFetcher Tests ==============


class TestFundingRateFetcher:
    """Tests for FundingRateFetcher class."""

    def test_init_default_values(self):
        """Test fetcher initializes with default values."""
        fetcher = FundingRateFetcher()
        assert fetcher.symbol == "BTCUSDT"
        assert fetcher.api_timeout == 15
        assert fetcher.rate_limit_delay == 0.5

    def test_init_custom_values(self):
        """Test fetcher initializes with custom values."""
        fetcher = FundingRateFetcher(
            symbol="ETHUSDT",
            api_timeout=30,
            rate_limit_delay=1.0,
        )
        assert fetcher.symbol == "ETHUSDT"
        assert fetcher.api_timeout == 30

    def test_funding_rate_record_from_binance(self, mock_binance_response):
        """Test creating record from Binance response."""
        record = FundingRateRecord.from_binance(mock_binance_response[0])

        assert record.funding_rate == 0.0003
        assert record.mark_price == 29000.0
        assert record.symbol == "BTCUSDT"
        assert record.timestamp.year == 2021

    def test_funding_rate_record_to_dict(self):
        """Test converting record to dictionary."""
        record = FundingRateRecord(
            timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc),
            funding_rate=0.0003,
            mark_price=40000.0,
        )

        d = record.to_dict()

        assert "timestamp" in d
        assert d["funding_rate"] == 0.0003
        assert d["mark_price"] == 40000.0

    def test_funding_rate_record_from_dict(self):
        """Test creating record from dictionary."""
        d = {
            "timestamp": "2021-01-01T00:00:00+00:00",
            "funding_rate": "0.0003",
            "mark_price": "40000.0",
            "symbol": "BTCUSDT",
        }

        record = FundingRateRecord.from_dict(d)

        assert record.funding_rate == 0.0003
        assert record.mark_price == 40000.0

    @patch("requests.Session.get")
    def test_fetch_single_page(self, mock_get, mock_binance_response):
        """Test fetching a single page of rates."""
        mock_response = Mock()
        mock_response.json.return_value = mock_binance_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = FundingRateFetcher()
        records = fetcher.fetch_date_range(
            start_date=datetime(2021, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2021, 1, 2, tzinfo=timezone.utc),
        )

        assert len(records) == 2
        assert records[0].funding_rate == 0.0003

    @patch("requests.Session.get")
    def test_pagination_multiple_pages(self, mock_get):
        """Test pagination logic with multiple API calls."""
        # First call returns 1000 records, second returns 500, third empty
        page1 = [
            {
                "fundingTime": 1609459200000 + i * 28800000,
                "fundingRate": "0.0003",
                "markPrice": "30000",
                "symbol": "BTCUSDT",
            }
            for i in range(1000)
        ]
        page2 = [
            {
                "fundingTime": 1609459200000 + (1000 + i) * 28800000,
                "fundingRate": "0.0003",
                "markPrice": "30000",
                "symbol": "BTCUSDT",
            }
            for i in range(500)
        ]

        mock_response = Mock()
        mock_response.json.side_effect = [page1, page2, []]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = FundingRateFetcher(rate_limit_delay=0)  # No delay for testing
        records = fetcher.fetch_all_history(
            start_date=datetime(2021, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
        )

        assert len(records) == 1500

    def test_save_and_load_csv(self, sample_funding_rates, tmp_path):
        """Test saving and loading from CSV."""
        csv_path = tmp_path / "test_rates.csv"

        fetcher = FundingRateFetcher()
        fetcher.save_to_csv(sample_funding_rates, str(csv_path))

        assert csv_path.exists()

        loaded = fetcher.load_from_csv(str(csv_path))

        assert len(loaded) == len(sample_funding_rates)
        assert loaded[0].funding_rate == sample_funding_rates[0].funding_rate

    def test_load_nonexistent_csv(self):
        """Test loading from non-existent CSV returns empty list."""
        fetcher = FundingRateFetcher()
        records = fetcher.load_from_csv("/nonexistent/path.csv")

        assert records == []

    def test_get_statistics(self, sample_funding_rates):
        """Test statistics calculation."""
        fetcher = FundingRateFetcher()
        stats = fetcher.get_statistics(sample_funding_rates)

        assert stats["total_records"] == 100
        assert stats["positive_periods"] > 0
        assert stats["negative_periods"] > 0
        assert "avg_rate" in stats
        assert "max_rate" in stats
        assert "min_rate" in stats

    def test_get_statistics_empty(self):
        """Test statistics with empty data."""
        fetcher = FundingRateFetcher()
        stats = fetcher.get_statistics([])

        assert "error" in stats


# ============== CostModel Tests ==============


class TestCostModel:
    """Tests for CostCalculator class."""

    def test_default_parameters(self):
        """Test default cost parameters."""
        params = CostParameters()
        assert params.maker_fee == 0.0002
        assert params.taker_fee == 0.0004
        assert params.entry_slippage == 0.0005

    def test_calculator_initialization(self, cost_params):
        """Test calculator initialization."""
        calc = CostCalculator(cost_params)
        assert calc.params == cost_params

    def test_entry_cost_taker(self, cost_params):
        """Test entry cost with taker orders."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        cost = calc.calculate_entry_cost(notional, is_maker=False)

        # 2 legs * 0.04% fee + 2 legs * 0.05% slippage
        # = 10000 * 0.0004 * 2 + 10000 * 0.0005 * 2
        # = 8 + 10 = 18
        assert cost == pytest.approx(18.0, rel=0.01)

    def test_entry_cost_maker(self, cost_params):
        """Test entry cost with maker orders."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        cost = calc.calculate_entry_cost(notional, is_maker=True)

        # 2 legs * 0.02% fee + 2 legs * 0.05% slippage
        # = 10000 * 0.0002 * 2 + 10000 * 0.0005 * 2
        # = 4 + 10 = 14
        assert cost == pytest.approx(14.0, rel=0.01)

    def test_exit_cost(self, cost_params):
        """Test exit cost calculation."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        cost = calc.calculate_exit_cost(notional, is_maker=False)

        assert cost == pytest.approx(18.0, rel=0.01)

    def test_round_trip_cost(self, cost_params):
        """Test complete round trip cost."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        result = calc.calculate_round_trip_cost(notional)

        # Entry + Exit = 18 + 18 = 36
        assert result.total_cost == pytest.approx(36.0, rel=0.01)
        assert isinstance(result, TradeCost)

    def test_maker_vs_taker_fees(self, cost_params):
        """Test maker fees are lower than taker."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        maker_cost = calc.calculate_entry_cost(notional, is_maker=True)
        taker_cost = calc.calculate_entry_cost(notional, is_maker=False)

        assert maker_cost < taker_cost

    def test_breakeven_rate(self, cost_params):
        """Test breakeven rate calculation."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        breakeven = calc.calculate_breakeven_rate(notional, expected_holding_days=1.0)

        # Round trip cost ~36, 3 funding periods per day
        # Breakeven per period = 36 / 10000 / 3 = 0.0012
        assert breakeven > 0
        assert breakeven < 0.01  # Should be reasonable

    def test_estimate_net_yield_profitable(self, cost_params):
        """Test net yield estimation for profitable scenario."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        # Use higher rate and more periods to overcome costs
        result = calc.estimate_net_yield(
            notional_value=notional,
            funding_rate=0.002,  # 0.2% per period (very high)
            funding_periods=10,  # ~3 days
        )

        # Gross: 10000 * 0.002 * 10 = $200
        # Costs: ~$36 round trip
        # Net: ~$164
        assert result["gross_funding"] == 200.0
        assert result["net_yield"] > 0
        assert result["profitable"] is True

    def test_estimate_net_yield_unprofitable(self, cost_params):
        """Test net yield estimation for unprofitable scenario."""
        calc = CostCalculator(cost_params)
        notional = 10000.0

        result = calc.estimate_net_yield(
            notional_value=notional,
            funding_rate=0.0001,  # 0.01% per period (too low)
            funding_periods=3,
        )

        assert result["gross_funding"] == 3.0  # 10000 * 0.0001 * 3
        assert result["profitable"] is False  # Costs exceed funding

    def test_get_binance_vip0_costs(self):
        """Test pre-configured VIP 0 costs."""
        calc = get_binance_vip0_costs()
        assert calc.params.maker_fee == 0.0002
        assert calc.params.taker_fee == 0.0004

    def test_cost_summary_format(self, cost_params):
        """Test cost summary formatting."""
        calc = CostCalculator(cost_params)
        summary = calc.get_cost_summary(10000.0)

        assert "Taker Orders" in summary
        assert "Maker Orders" in summary
        assert "$" in summary


# ============== ComprehensiveBacktest Tests ==============


class TestComprehensiveBacktest:
    """Tests for ComprehensiveBacktest class."""

    def test_import(self):
        """Test that ComprehensiveBacktest can be imported."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest(initial_capital=10000.0)
        assert bt.initial_capital == 10000.0

    def test_backtest_basic(self, sample_funding_rates):
        """Test basic backtest execution."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest(initial_capital=10000.0, position_pct=0.30)
        result = bt.run_backtest(sample_funding_rates)

        assert result.initial_capital == 10000.0
        assert result.total_periods == len(sample_funding_rates)
        assert result.net_pnl != 0 or result.total_funding_earned == 0

    def test_backtest_empty_data(self):
        """Test backtest with empty data."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest([])

        assert result.total_periods == 0
        assert result.net_pnl == 0

    def test_yearly_breakdown(self, multi_year_rates):
        """Test year-by-year analysis."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest(initial_capital=10000.0)
        result = bt.run_backtest(multi_year_rates)

        # Should have metrics for each year
        years = {ym.year for ym in result.yearly_metrics}
        assert 2020 in years
        assert 2021 in years
        assert 2022 in years

    def test_sharpe_ratio_calculation(self, sample_funding_rates):
        """Test Sharpe ratio is calculated."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(sample_funding_rates)

        # Sharpe should be a reasonable number
        assert result.overall_sharpe != float("inf")
        assert result.overall_sharpe != float("-inf")

    def test_max_drawdown_calculation(self, sample_funding_rates):
        """Test max drawdown is calculated."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(sample_funding_rates)

        # Max drawdown should be between 0 and 1
        assert 0 <= result.max_drawdown <= 1

    def test_win_rate_calculation(self, sample_funding_rates):
        """Test win rate calculation."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(sample_funding_rates)

        # Win rate should be between 0 and 1
        assert 0 <= result.win_rate <= 1

    def test_market_period_analysis(self, multi_year_rates):
        """Test market period breakdown."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(multi_year_rates)

        # Should have some market periods analyzed
        assert len(result.market_periods) > 0

    def test_report_generation_console(self, sample_funding_rates):
        """Test console report generation."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(sample_funding_rates)
        report = bt.generate_report(result, "console")

        assert "COMPREHENSIVE" in report
        assert "PERFORMANCE" in report

    def test_report_generation_json(self, sample_funding_rates):
        """Test JSON report generation."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(sample_funding_rates)
        report = bt.generate_report(result, "json")

        # Should be valid JSON
        data = json.loads(report)
        assert "net_pnl" in data

    def test_result_to_dict(self, sample_funding_rates):
        """Test result serialization."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(sample_funding_rates)
        d = result.to_dict()

        assert "initial_capital" in d
        assert "yearly_metrics" in d
        assert isinstance(d["yearly_metrics"], list)


# ============== Market Regime Tests ==============


class TestMarketRegimes:
    """Test different market conditions."""

    def test_all_positive_rates(self):
        """Test with all positive funding rates."""
        from comprehensive_backtest import ComprehensiveBacktest

        rates = [
            FundingRateRecord(
                timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc)
                + timedelta(hours=i * 8),
                funding_rate=0.0005,
                mark_price=40000,
            )
            for i in range(100)
        ]

        bt = ComprehensiveBacktest(initial_capital=10000.0)
        result = bt.run_backtest(rates)

        assert result.win_rate == 1.0
        assert result.total_funding_earned > 0

    def test_all_negative_rates(self):
        """Test with all negative funding rates."""
        from comprehensive_backtest import ComprehensiveBacktest

        rates = [
            FundingRateRecord(
                timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc)
                + timedelta(hours=i * 8),
                funding_rate=-0.0002,
                mark_price=40000,
            )
            for i in range(100)
        ]

        bt = ComprehensiveBacktest(initial_capital=10000.0)
        result = bt.run_backtest(rates)

        assert result.win_rate == 0.0

    def test_high_volatility_rates(self):
        """Test with highly volatile funding rates."""
        from comprehensive_backtest import ComprehensiveBacktest

        rates = []
        for i in range(100):
            # Alternate between high positive and negative
            rate = 0.002 if i % 2 == 0 else -0.001
            rates.append(
                FundingRateRecord(
                    timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc)
                    + timedelta(hours=i * 8),
                    funding_rate=rate,
                    mark_price=40000,
                )
            )

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(rates)

        # Should still produce valid results
        assert result.total_periods == 100


# ============== Integration Tests ==============


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_full_workflow_mock(self, tmp_path, sample_funding_rates):
        """Test full workflow with mock data."""
        from comprehensive_backtest import ComprehensiveBacktest

        # Save to CSV
        fetcher = FundingRateFetcher()
        csv_path = tmp_path / "test_funding.csv"
        fetcher.save_to_csv(sample_funding_rates, str(csv_path))

        # Load and backtest
        loaded = fetcher.load_from_csv(str(csv_path))
        bt = ComprehensiveBacktest(initial_capital=10000.0)
        result = bt.run_backtest(loaded)

        # Verify full workflow
        assert result.total_periods == len(sample_funding_rates)
        assert result.final_capital > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_real_data_limited(self):
        """Test fetching real data (limited to avoid rate limits)."""
        fetcher = FundingRateFetcher()

        # Only fetch last 7 days to be quick
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)

        try:
            records = fetcher.fetch_date_range(start_date, end_date)
            # Should get ~21 records (3 per day * 7 days)
            assert len(records) >= 15  # Some buffer
        except Exception:
            pytest.skip("API not available")


# ============== Edge Cases ==============


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_record(self):
        """Test with single funding rate record."""
        from comprehensive_backtest import ComprehensiveBacktest

        rates = [
            FundingRateRecord(
                timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc),
                funding_rate=0.0003,
                mark_price=40000,
            )
        ]

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(rates)

        assert result.total_periods == 1

    def test_zero_capital(self):
        """Test with zero initial capital."""
        from comprehensive_backtest import ComprehensiveBacktest

        bt = ComprehensiveBacktest(initial_capital=0.0)
        result = bt.run_backtest([])

        assert result.initial_capital == 0.0

    def test_very_small_rates(self):
        """Test with very small funding rates."""
        from comprehensive_backtest import ComprehensiveBacktest

        rates = [
            FundingRateRecord(
                timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc)
                + timedelta(hours=i * 8),
                funding_rate=0.00001,  # 0.001%
                mark_price=40000,
            )
            for i in range(100)
        ]

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(rates)

        # Should still work, likely unprofitable due to costs
        assert result.total_periods == 100

    def test_extreme_rates(self):
        """Test with extreme funding rates."""
        from comprehensive_backtest import ComprehensiveBacktest

        rates = [
            FundingRateRecord(
                timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc)
                + timedelta(hours=i * 8),
                funding_rate=0.01,  # 1% - very high
                mark_price=40000,
            )
            for i in range(10)
        ]

        bt = ComprehensiveBacktest()
        result = bt.run_backtest(rates)

        assert result.total_funding_earned > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

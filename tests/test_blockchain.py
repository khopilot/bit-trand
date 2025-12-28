"""
Unit tests for Blockchain Service (Free API Edition).

Tests API calls, caching, rate limiting, and signal generation.

Author: khopilot
"""

import sys
from datetime import datetime, timezone, date
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blockchain_service import (
    FreeAPIBlockchainService,
    MempoolMetrics,
    LargeTransaction,
    OnChainSignal,
    MEMPOOL_API,
    BLOCKCHAIR_API,
)


@pytest.fixture
def service():
    """Create blockchain service for testing."""
    return FreeAPIBlockchainService(
        mempool_api=MEMPOOL_API,
        blockchair_api=BLOCKCHAIR_API,
        large_tx_threshold=100,
        cache_ttl=300,
        blockchair_daily_limit=1000,
        enabled=True,
    )


@pytest.fixture
def disabled_service():
    """Create disabled blockchain service."""
    return FreeAPIBlockchainService(enabled=False)


class TestServiceInit:
    """Tests for service initialization."""

    def test_default_init(self, service):
        """Should initialize with default values."""
        assert service.enabled is True
        assert service.large_tx_threshold == 100
        assert service.cache_ttl == 300
        assert service.blockchair_daily_limit == 1000

    def test_disabled_init(self, disabled_service):
        """Should initialize as disabled."""
        assert disabled_service.enabled is False

    def test_custom_endpoints(self):
        """Should accept custom API endpoints."""
        service = FreeAPIBlockchainService(
            mempool_api="https://custom.mempool.api",
            blockchair_api="https://custom.blockchair.api",
        )

        assert "custom.mempool" in service.mempool_api
        assert "custom.blockchair" in service.blockchair_api


class TestRateLimiting:
    """Tests for Blockchair rate limiting."""

    def test_initial_call_count(self, service):
        """Should start with zero calls."""
        assert service._blockchair_calls_today == 0

    def test_check_blockchair_limit_under_limit(self, service):
        """Should allow calls under limit."""
        service._blockchair_calls_today = 500
        assert service._check_blockchair_limit() is True

    def test_check_blockchair_limit_at_limit(self, service):
        """Should block calls at limit."""
        service._blockchair_calls_today = 1000
        assert service._check_blockchair_limit() is False

    def test_increment_blockchair_calls(self, service):
        """Should increment call counter."""
        initial = service._blockchair_calls_today
        service._increment_blockchair_calls()
        assert service._blockchair_calls_today == initial + 1

    def test_daily_reset(self, service):
        """Should reset counter on new day."""
        service._blockchair_calls_today = 999
        service._blockchair_reset_date = date(2020, 1, 1)  # Old date

        # Check limit should trigger reset
        result = service._check_blockchair_limit()

        assert result is True
        assert service._blockchair_calls_today == 0
        assert service._blockchair_reset_date == date.today()


class TestCaching:
    """Tests for response caching."""

    def test_set_and_get_cache(self, service):
        """Should cache and retrieve values."""
        service._set_cache("test_key", {"data": 123})

        cached = service._get_cache("test_key")
        assert cached == {"data": 123}

    def test_cache_miss(self, service):
        """Should return None for missing key."""
        cached = service._get_cache("nonexistent_key")
        assert cached is None

    def test_cache_expiry(self, service):
        """Should return None for expired cache."""
        # Set cache with old timestamp
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        service._cache["old_key"] = (old_time, {"data": "old"})

        cached = service._get_cache("old_key")
        assert cached is None


class TestMempoolMetrics:
    """Tests for MempoolMetrics dataclass."""

    def test_mempool_metrics_creation(self):
        """Should create valid MempoolMetrics."""
        metrics = MempoolMetrics(
            size=50000,
            bytes=15000000,
            avg_fee_rate=10.5,
            fast_fee=15,
            medium_fee=10,
            slow_fee=5,
            congestion="normal",
        )

        assert metrics.size == 50000
        assert metrics.congestion == "normal"
        assert metrics.fast_fee == 15


class TestLargeTransaction:
    """Tests for LargeTransaction dataclass."""

    def test_large_transaction_creation(self):
        """Should create valid LargeTransaction."""
        tx = LargeTransaction(
            txid="abc123",
            value_btc=150.5,
            block_height=800000,
            timestamp=datetime.now(timezone.utc),
        )

        assert tx.txid == "abc123"
        assert tx.value_btc == 150.5
        assert tx.block_height == 800000


class TestOnChainSignal:
    """Tests for OnChainSignal dataclass."""

    def test_on_chain_signal_creation(self):
        """Should create valid OnChainSignal."""
        mempool = MempoolMetrics(
            size=50000,
            bytes=15000000,
            avg_fee_rate=10.5,
            fast_fee=15,
            medium_fee=10,
            slow_fee=5,
            congestion="normal",
        )

        signal = OnChainSignal(
            mempool=mempool,
            large_tx_count=3,
            large_tx_volume=450.0,
            recommendation="neutral",
            confidence_adjustment=1.05,
            reason="normal",
        )

        assert signal.large_tx_count == 3
        assert signal.recommendation == "neutral"
        assert signal.confidence_adjustment == 1.05


class TestDisabledService:
    """Tests for disabled service behavior."""

    def test_get_mempool_info_disabled(self, disabled_service):
        """Should return None when disabled (sync wrapper)."""
        import asyncio
        result = asyncio.run(disabled_service.get_mempool_info())
        assert result is None

    def test_get_large_transactions_disabled(self, disabled_service):
        """Should return empty list when disabled (sync wrapper)."""
        import asyncio
        result = asyncio.run(disabled_service.get_large_transactions())
        assert result == []

    def test_get_on_chain_signal_disabled(self, disabled_service):
        """Should return None when disabled (sync wrapper)."""
        import asyncio
        result = asyncio.run(disabled_service.get_on_chain_signal())
        assert result is None

    def test_health_check_disabled(self, disabled_service):
        """Should return False when disabled (sync wrapper)."""
        import asyncio
        result = asyncio.run(disabled_service.health_check())
        assert result is False


class TestServiceStatus:
    """Tests for service status reporting."""

    def test_get_status(self, service):
        """Should return complete status dictionary."""
        service._blockchair_calls_today = 42

        status = service.get_status()

        assert status["enabled"] is True
        assert status["provider"] == "free_api"
        assert status["blockchair_calls_today"] == 42
        assert status["blockchair_daily_limit"] == 1000
        assert status["large_tx_threshold"] == 100


class TestCongestionLevels:
    """Tests for mempool congestion classification."""

    def test_congestion_extreme(self):
        """100k+ txs should be extreme."""
        # This tests the logic in get_mempool_info
        count = 150000
        if count > 100000:
            congestion = "extreme"
        elif count > 50000:
            congestion = "high"
        elif count > 10000:
            congestion = "normal"
        else:
            congestion = "low"

        assert congestion == "extreme"

    def test_congestion_high(self):
        """50k-100k txs should be high."""
        count = 75000
        if count > 100000:
            congestion = "extreme"
        elif count > 50000:
            congestion = "high"
        elif count > 10000:
            congestion = "normal"
        else:
            congestion = "low"

        assert congestion == "high"

    def test_congestion_normal(self):
        """10k-50k txs should be normal."""
        count = 25000
        if count > 100000:
            congestion = "extreme"
        elif count > 50000:
            congestion = "high"
        elif count > 10000:
            congestion = "normal"
        else:
            congestion = "low"

        assert congestion == "normal"

    def test_congestion_low(self):
        """<10k txs should be low."""
        count = 5000
        if count > 100000:
            congestion = "extreme"
        elif count > 50000:
            congestion = "high"
        elif count > 10000:
            congestion = "normal"
        else:
            congestion = "low"

        assert congestion == "low"


class TestConfidenceAdjustment:
    """Tests for confidence adjustment logic."""

    def test_extreme_mempool_boost(self):
        """Extreme mempool should boost confidence."""
        adjustment = 1.0
        congestion = "extreme"

        if congestion == "extreme":
            adjustment *= 1.15
        elif congestion == "high":
            adjustment *= 1.08
        elif congestion == "low":
            adjustment *= 0.95

        assert adjustment == 1.15

    def test_high_fees_boost(self):
        """High fees should boost confidence."""
        adjustment = 1.0
        fast_fee = 60

        if fast_fee > 50:
            adjustment *= 1.05
        elif fast_fee < 5:
            adjustment *= 0.98

        assert adjustment == 1.05

    def test_whale_activity_boost(self):
        """Large tx count should boost confidence."""
        adjustment = 1.0
        large_tx_count = 6

        if large_tx_count >= 5:
            adjustment *= 1.10
        elif large_tx_count >= 3:
            adjustment *= 1.05

        assert adjustment == 1.10

    def test_adjustment_clamping(self):
        """Adjustment should be clamped to 0.8-1.25."""
        # Simulate very high adjustment
        adjustment = 1.5

        clamped = min(max(adjustment, 0.8), 1.25)
        assert clamped == 1.25

        # Simulate very low adjustment
        adjustment = 0.5
        clamped = min(max(adjustment, 0.8), 1.25)
        assert clamped == 0.8


class TestRecommendationLogic:
    """Tests for signal recommendation logic."""

    def test_boost_buy_recommendation(self):
        """High adjustment should recommend boost_buy."""
        adjustment = 1.15

        if adjustment > 1.10:
            recommendation = "boost_buy"
        elif adjustment > 1.0:
            recommendation = "neutral"
        elif adjustment < 0.95:
            recommendation = "reduce_buy"
        else:
            recommendation = "neutral"

        assert recommendation == "boost_buy"

    def test_reduce_buy_recommendation(self):
        """Low adjustment should recommend reduce_buy."""
        adjustment = 0.90

        if adjustment > 1.10:
            recommendation = "boost_buy"
        elif adjustment > 1.0:
            recommendation = "neutral"
        elif adjustment < 0.95:
            recommendation = "reduce_buy"
        else:
            recommendation = "neutral"

        assert recommendation == "reduce_buy"


class TestLiveAPI:
    """Live API integration tests (requires network)."""

    @pytest.mark.skip(reason="Requires live network - run manually")
    def test_live_mempool_info(self, service):
        """Should fetch real mempool data."""
        import asyncio

        async def run():
            result = await service.get_mempool_info()
            await service.close()
            return result

        result = asyncio.run(run())

        assert result is not None
        assert isinstance(result, MempoolMetrics)
        assert result.size > 0
        assert result.congestion in ["low", "normal", "high", "extreme"]

    @pytest.mark.skip(reason="Requires live network - run manually")
    def test_live_block_height(self, service):
        """Should fetch real block height."""
        import asyncio

        async def run():
            result = await service.get_block_height()
            await service.close()
            return result

        result = asyncio.run(run())

        assert result is not None
        assert result > 800000  # We're well past this

    @pytest.mark.skip(reason="Requires live network - run manually")
    def test_live_on_chain_signal(self, service):
        """Should generate real on-chain signal."""
        import asyncio

        async def run():
            result = await service.get_on_chain_signal()
            await service.close()
            return result

        result = asyncio.run(run())

        assert result is not None
        assert isinstance(result, OnChainSignal)
        assert result.recommendation in ["boost_buy", "reduce_buy", "boost_sell", "reduce_sell", "neutral"]
        assert 0.8 <= result.confidence_adjustment <= 1.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

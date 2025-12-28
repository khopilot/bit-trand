"""
Funding Rate Arbitrage Module

A market-neutral strategy that earns yield by:
1. Going LONG on spot BTC
2. Going SHORT on perpetual futures (same size)
3. Collecting funding payments when rate > 0

Realistic expected returns (based on 6-year backtest):
- Bull market: 20-30% APY
- Normal market: 5-12% APY
- Bear market: 0-5% APY
"""

from .rate_monitor import FundingRateMonitor, FundingRate, FundingSignal
from .exchange_client import (
    ExchangeClient,
    BinanceClient,
    OrderSide,
    OrderType,
    OrderResult,
    AccountBalance,
    FuturesPosition,
)
from .position_manager import PositionManager, ArbPosition, PositionStatus
from .delta_hedger import DeltaHedger, HedgeAction, HedgeResult
from .yield_calculator import YieldCalculator, FundingPayment, PerformanceMetrics
from .funding_rate_fetcher import FundingRateFetcher, FundingRateRecord
from .cost_model import CostCalculator, CostParameters, TradeCost
from .hybrid_strategy import (
    HybridFundingStrategy,
    HybridStrategyConfig,
    StrategyMode,
    SizingDecision,
    PositionState,
)
from .atomic_executor import (
    AtomicExecutor,
    ExecutionConfig,
    ExecutionResult,
    ExecutionStatus,
    ExecutionMetrics,
)

__all__ = [
    # Rate monitoring
    "FundingRateMonitor",
    "FundingRate",
    "FundingSignal",
    # Exchange client
    "ExchangeClient",
    "BinanceClient",
    "OrderSide",
    "OrderType",
    "OrderResult",
    "AccountBalance",
    "FuturesPosition",
    # Position management
    "PositionManager",
    "ArbPosition",
    "PositionStatus",
    # Delta hedging
    "DeltaHedger",
    "HedgeAction",
    "HedgeResult",
    # Yield calculation
    "YieldCalculator",
    "FundingPayment",
    "PerformanceMetrics",
    # Paginated fetcher
    "FundingRateFetcher",
    "FundingRateRecord",
    # Cost model
    "CostCalculator",
    "CostParameters",
    "TradeCost",
    # Hybrid strategy (NEW)
    "HybridFundingStrategy",
    "HybridStrategyConfig",
    "StrategyMode",
    "SizingDecision",
    "PositionState",
    # Atomic executor (NEW)
    "AtomicExecutor",
    "ExecutionConfig",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionMetrics",
]

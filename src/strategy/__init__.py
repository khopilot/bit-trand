"""
Strategy Module

Contains the unified strategy orchestrator that decides between:
- Pure funding arbitrage
- Directional trading
- Hybrid (arb + directional overlay)
"""

from .orchestrator import (
    StrategyOrchestrator,
    OrchestratorConfig,
    OrchestratorState,
    StrategyMode,
    TrendDirection,
    MarketState,
    ModeDecision,
)

__all__ = [
    "StrategyOrchestrator",
    "OrchestratorConfig",
    "OrchestratorState",
    "StrategyMode",
    "TrendDirection",
    "MarketState",
    "ModeDecision",
]

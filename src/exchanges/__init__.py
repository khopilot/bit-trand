"""
Multi-Exchange Trading Module

Provides abstraction layer for trading across multiple exchanges.
Currently supports: Binance, OKX
"""

from .okx_client import (
    OKXClient,
    OKXConfig,
    OKXOrderResult,
    OKXPosition,
    OKXBalance,
)
from .unified_client import (
    UnifiedExchangeClient,
    UnifiedClientConfig,
    UnifiedPosition,
    UnifiedBalance,
    AllocationDecision,
    ExchangeRate,
    Exchange,
)

__all__ = [
    # OKX
    "OKXClient",
    "OKXConfig",
    "OKXOrderResult",
    "OKXPosition",
    "OKXBalance",
    # Unified
    "UnifiedExchangeClient",
    "UnifiedClientConfig",
    "UnifiedPosition",
    "UnifiedBalance",
    "AllocationDecision",
    "ExchangeRate",
    "Exchange",
]

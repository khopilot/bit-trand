"""
Paper Trading Module

Provides a paper trading simulator for the Always-In funding arbitrage strategy.
Fetches live funding rates from Binance and tracks simulated P&L.

Usage:
    python -m src.paper_trading.simulator --live      # Start live simulation
    python -m src.paper_trading.simulator --status    # Show current status
    python -m src.paper_trading.simulator --backfill 7  # Simulate last 7 days
"""

from .simulator import FundingArbPaperTrader
from .funding_fetcher import (
    BinanceFundingFetcher,
    FundingRateData,
    HistoricalFundingRate,
)
from .position_tracker import (
    PositionTracker,
    PaperPosition,
    FundingPayment,
)
from .logger import PaperTradingLogger
from .telegram_notifier import TelegramNotifier

__all__ = [
    # Main class
    "FundingArbPaperTrader",
    # Fetcher
    "BinanceFundingFetcher",
    "FundingRateData",
    "HistoricalFundingRate",
    # Tracker
    "PositionTracker",
    "PaperPosition",
    "FundingPayment",
    # Logger
    "PaperTradingLogger",
    # Telegram
    "TelegramNotifier",
]

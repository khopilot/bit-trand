"""
THE PREDATOR BOT - Hunt Smart Money & Liquidations

This module predicts OTHER TRADERS' behavior, not price.
It fades the crowd and hunts liquidations using FREE Binance data.

Components:
- data_fetcher: Binance Futures API wrapper
- funding_analyzer: Funding rate extremes (30% weight)
- oi_analyzer: Open Interest divergence (25% weight)
- sentiment_analyzer: L/S ratio + taker volume (45% weight)
- conviction_scorer: Aggregate signals (0-100 score)
- bot: Main orchestrator
"""

from .config import *
from .data_fetcher import PredatorDataFetcher
from .funding_analyzer import FundingAnalyzer
from .oi_analyzer import OIAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from .conviction_scorer import ConvictionScorer
from .bot import PredatorBot

__all__ = [
    "PredatorDataFetcher",
    "FundingAnalyzer",
    "OIAnalyzer",
    "SentimentAnalyzer",
    "ConvictionScorer",
    "PredatorBot",
]

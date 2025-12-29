"""
Predator Data Fetcher - Binance Futures API Wrapper

Fetches FREE data from Binance Futures for predicting trader behavior:
- Funding rates (how much longs/shorts are paying)
- Open Interest (total leveraged positions)
- Long/Short Account Ratio (crowd positioning)
- Taker Buy/Sell Volume (who's market buying/selling)
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from .config import (
    BINANCE_FUTURES_BASE,
    BINANCE_DATA_BASE,
    DEFAULT_SYMBOL,
    API_TIMEOUT,
    HISTORY_DAYS,
)

logger = logging.getLogger(__name__)


@dataclass
class FundingData:
    """Current and historical funding rate data."""
    symbol: str
    current_rate: float  # Current funding rate
    next_funding_time: datetime
    rates_history: list[dict]  # [{time, rate}, ...]
    avg_rate_24h: float
    avg_rate_7d: float
    percentile_current: float  # Where current rate sits in history


@dataclass
class OpenInterestData:
    """Open Interest data."""
    symbol: str
    current_oi: float  # In contracts
    current_oi_value: float  # In USDT
    history: list[dict]  # [{time, oi, oi_value}, ...]
    change_1h: float  # % change
    change_4h: float
    change_24h: float


@dataclass
class LongShortData:
    """Long/Short account ratio data."""
    symbol: str
    current_ratio: float  # > 1 = more longs, < 1 = more shorts
    long_pct: float  # % of accounts long
    short_pct: float  # % of accounts short
    history: list[dict]  # [{time, ratio, long_pct, short_pct}, ...]
    avg_ratio_24h: float


@dataclass
class TakerVolumeData:
    """Taker buy/sell volume data."""
    symbol: str
    buy_sell_ratio: float  # > 1 = more taker buys
    buy_volume: float
    sell_volume: float
    history: list[dict]  # [{time, ratio, buy_vol, sell_vol}, ...]
    avg_ratio_24h: float


class PredatorDataFetcher:
    """
    Fetches market data from Binance Futures API.

    All endpoints are FREE and don't require API keys.
    """

    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        self.symbol = symbol
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PredatorBot/1.0",
        })
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    def _rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None) -> dict | list:
        """Make GET request with error handling."""
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error("API request failed: %s - %s", url, e)
            raise

    def get_funding_rate(self) -> FundingData:
        """
        Get funding rate data.

        Endpoints:
        - /fapi/v1/premiumIndex - Current funding rate & next time
        - /fapi/v1/fundingRate - Historical funding rates
        """
        # Current funding info
        premium_url = f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex"
        premium_data = self._get(premium_url, {"symbol": self.symbol})

        current_rate = float(premium_data["lastFundingRate"])
        next_funding_time = datetime.fromtimestamp(
            premium_data["nextFundingTime"] / 1000,
            tz=timezone.utc
        )

        # Historical funding rates (last 30 days = ~90 payments)
        history_url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
        limit = HISTORY_DAYS * 3  # 3 payments per day
        history_data = self._get(history_url, {
            "symbol": self.symbol,
            "limit": min(limit, 1000),  # API max is 1000
        })

        rates_history = []
        for item in history_data:
            rates_history.append({
                "time": datetime.fromtimestamp(
                    item["fundingTime"] / 1000, tz=timezone.utc
                ),
                "rate": float(item["fundingRate"]),
            })

        # Calculate averages
        rates = [r["rate"] for r in rates_history]
        avg_24h = sum(rates[:3]) / 3 if len(rates) >= 3 else current_rate
        avg_7d = sum(rates[:21]) / 21 if len(rates) >= 21 else avg_24h

        # Calculate percentile of current rate
        if rates:
            sorted_rates = sorted(rates)
            position = sum(1 for r in sorted_rates if r <= current_rate)
            percentile = (position / len(sorted_rates)) * 100
        else:
            percentile = 50.0

        return FundingData(
            symbol=self.symbol,
            current_rate=current_rate,
            next_funding_time=next_funding_time,
            rates_history=rates_history,
            avg_rate_24h=avg_24h,
            avg_rate_7d=avg_7d,
            percentile_current=percentile,
        )

    def get_open_interest(self) -> OpenInterestData:
        """
        Get Open Interest data.

        Endpoints:
        - /fapi/v1/openInterest - Current OI
        - /futures/data/openInterestHist - Historical OI (5m intervals)
        """
        # Current OI
        current_url = f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
        current_data = self._get(current_url, {"symbol": self.symbol})
        current_oi = float(current_data["openInterest"])

        # Get current price for value calculation
        price_url = f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/price"
        price_data = self._get(price_url, {"symbol": self.symbol})
        current_price = float(price_data["price"])
        current_oi_value = current_oi * current_price

        # Historical OI (5m intervals, last 24h = 288 periods)
        history_url = f"{BINANCE_DATA_BASE}/openInterestHist"
        history_data = self._get(history_url, {
            "symbol": self.symbol,
            "period": "5m",
            "limit": 500,  # ~42 hours of data
        })

        history = []
        for item in history_data:
            history.append({
                "time": datetime.fromtimestamp(
                    item["timestamp"] / 1000, tz=timezone.utc
                ),
                "oi": float(item["sumOpenInterest"]),
                "oi_value": float(item["sumOpenInterestValue"]),
            })

        # Calculate % changes
        def calc_change(periods_back: int) -> float:
            if len(history) > periods_back:
                old_oi = history[-periods_back - 1]["oi"]
                if old_oi > 0:
                    return (current_oi - old_oi) / old_oi
            return 0.0

        change_1h = calc_change(12)   # 12 x 5min = 1h
        change_4h = calc_change(48)   # 48 x 5min = 4h
        change_24h = calc_change(288) # 288 x 5min = 24h

        return OpenInterestData(
            symbol=self.symbol,
            current_oi=current_oi,
            current_oi_value=current_oi_value,
            history=history,
            change_1h=change_1h,
            change_4h=change_4h,
            change_24h=change_24h,
        )

    def get_long_short_ratio(self) -> LongShortData:
        """
        Get Long/Short Account Ratio.

        This shows what % of accounts are long vs short.
        NOT the same as position value - a few whales can offset many small accounts.

        Endpoint: /futures/data/globalLongShortAccountRatio
        """
        url = f"{BINANCE_DATA_BASE}/globalLongShortAccountRatio"
        data = self._get(url, {
            "symbol": self.symbol,
            "period": "5m",
            "limit": 500,
        })

        if not data:
            raise ValueError("No L/S ratio data returned")

        # Most recent is first after sorting by timestamp
        latest = max(data, key=lambda x: x["timestamp"])
        current_ratio = float(latest["longShortRatio"])
        long_pct = float(latest["longAccount"]) * 100
        short_pct = float(latest["shortAccount"]) * 100

        history = []
        for item in data:
            history.append({
                "time": datetime.fromtimestamp(
                    item["timestamp"] / 1000, tz=timezone.utc
                ),
                "ratio": float(item["longShortRatio"]),
                "long_pct": float(item["longAccount"]) * 100,
                "short_pct": float(item["shortAccount"]) * 100,
            })

        # Sort by time descending
        history.sort(key=lambda x: x["time"], reverse=True)

        # Average ratio over 24h (288 periods at 5min)
        ratios = [h["ratio"] for h in history[:288]]
        avg_ratio_24h = sum(ratios) / len(ratios) if ratios else current_ratio

        return LongShortData(
            symbol=self.symbol,
            current_ratio=current_ratio,
            long_pct=long_pct,
            short_pct=short_pct,
            history=history,
            avg_ratio_24h=avg_ratio_24h,
        )

    def get_taker_volume(self) -> TakerVolumeData:
        """
        Get Taker Buy/Sell Volume Ratio.

        Taker = someone who market buys/sells (crosses the spread).
        High taker buy = aggressive buying (bullish short-term).
        High taker sell = aggressive selling (bearish short-term).

        Endpoint: /futures/data/takerlongshortRatio
        """
        url = f"{BINANCE_DATA_BASE}/takerlongshortRatio"
        data = self._get(url, {
            "symbol": self.symbol,
            "period": "5m",
            "limit": 500,
        })

        if not data:
            raise ValueError("No taker volume data returned")

        latest = max(data, key=lambda x: x["timestamp"])
        buy_sell_ratio = float(latest["buySellRatio"])
        buy_volume = float(latest["buyVol"])
        sell_volume = float(latest["sellVol"])

        history = []
        for item in data:
            history.append({
                "time": datetime.fromtimestamp(
                    item["timestamp"] / 1000, tz=timezone.utc
                ),
                "ratio": float(item["buySellRatio"]),
                "buy_vol": float(item["buyVol"]),
                "sell_vol": float(item["sellVol"]),
            })

        history.sort(key=lambda x: x["time"], reverse=True)

        # Average over 24h
        ratios = [h["ratio"] for h in history[:288]]
        avg_ratio_24h = sum(ratios) / len(ratios) if ratios else buy_sell_ratio

        return TakerVolumeData(
            symbol=self.symbol,
            buy_sell_ratio=buy_sell_ratio,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            history=history,
            avg_ratio_24h=avg_ratio_24h,
        )

    def get_all_data(self) -> dict:
        """
        Fetch all data in one call.

        Returns dict with keys: funding, oi, ls_ratio, taker
        """
        return {
            "funding": self.get_funding_rate(),
            "oi": self.get_open_interest(),
            "ls_ratio": self.get_long_short_ratio(),
            "taker": self.get_taker_volume(),
        }

    def get_current_price(self) -> float:
        """Get current BTC price."""
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/price"
        data = self._get(url, {"symbol": self.symbol})
        return float(data["price"])

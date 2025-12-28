"""
Binance Funding Rate Fetcher

Fetches live funding rates from Binance public API (no authentication required).
Used by the paper trading simulator to simulate funding payments.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import requests

logger = logging.getLogger("btc_trader.paper_trading.fetcher")


@dataclass
class FundingRateData:
    """Funding rate data from Binance."""
    symbol: str
    funding_rate: float
    funding_time: datetime
    mark_price: float
    next_funding_time: datetime

    @property
    def rate_pct(self) -> float:
        """Funding rate as percentage."""
        return self.funding_rate * 100

    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (3 periods/day * 365 days)."""
        return self.funding_rate * 3 * 365 * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "funding_rate": self.funding_rate,
            "funding_rate_pct": f"{self.rate_pct:.4f}%",
            "funding_time": self.funding_time.isoformat(),
            "mark_price": self.mark_price,
            "next_funding_time": self.next_funding_time.isoformat(),
            "annualized_pct": f"{self.annualized_rate:.2f}%",
        }


@dataclass
class HistoricalFundingRate:
    """Historical funding rate record."""
    symbol: str
    funding_rate: float
    funding_time: datetime

    @property
    def rate_pct(self) -> float:
        """Funding rate as percentage."""
        return self.funding_rate * 100


class BinanceFundingFetcher:
    """
    Fetches funding rates from Binance Futures API.

    Uses public endpoints - no API key required.
    """

    BASE_URL = "https://fapi.binance.com"
    DEFAULT_SYMBOL = "BTCUSDT"
    TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        self.symbol = symbol
        self._session = requests.Session()
        self._last_rate: Optional[FundingRateData] = None
        self._last_fetch_time: Optional[datetime] = None

        logger.info("BinanceFundingFetcher initialized for %s", symbol)

    def _request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Make a request to Binance API with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response or None on failure
        """
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.TIMEOUT,
                )
                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                delay = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "API request failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    self.MAX_RETRIES,
                    str(e),
                    delay,
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(delay)

        logger.error("API request failed after %d attempts", self.MAX_RETRIES)
        return None

    def get_current_funding_rate(self) -> Optional[FundingRateData]:
        """
        Get current funding rate and next funding time.

        Uses GET /fapi/v1/premiumIndex which provides:
        - lastFundingRate: Current funding rate
        - nextFundingTime: Next funding settlement time
        - markPrice: Current mark price

        Returns:
            FundingRateData or None on failure
        """
        data = self._request(
            "/fapi/v1/premiumIndex",
            params={"symbol": self.symbol},
        )

        if not data:
            # Return cached rate if available
            if self._last_rate:
                logger.warning("Using cached funding rate")
                return self._last_rate
            return None

        try:
            rate = FundingRateData(
                symbol=data["symbol"],
                funding_rate=float(data["lastFundingRate"]),
                funding_time=datetime.fromtimestamp(
                    int(data["time"]) / 1000, tz=timezone.utc
                ),
                mark_price=float(data["markPrice"]),
                next_funding_time=datetime.fromtimestamp(
                    int(data["nextFundingTime"]) / 1000, tz=timezone.utc
                ),
            )

            # Cache the rate
            self._last_rate = rate
            self._last_fetch_time = datetime.now(timezone.utc)

            logger.debug(
                "Funding rate: %.4f%% | Mark: $%.2f | Next: %s",
                rate.rate_pct,
                rate.mark_price,
                rate.next_funding_time.strftime("%H:%M UTC"),
            )

            return rate

        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse funding rate response: %s", e)
            return self._last_rate

    def get_mark_price(self) -> Optional[float]:
        """
        Get current mark price for the symbol.

        Returns:
            Mark price in USD or None on failure
        """
        rate = self.get_current_funding_rate()
        return rate.mark_price if rate else None

    def get_next_funding_time(self) -> Optional[datetime]:
        """
        Get next funding settlement time.

        Returns:
            Datetime of next funding or None on failure
        """
        rate = self.get_current_funding_rate()
        return rate.next_funding_time if rate else None

    def get_time_to_funding(self) -> Optional[float]:
        """
        Get time remaining until next funding in seconds.

        Returns:
            Seconds until next funding or None on failure
        """
        next_time = self.get_next_funding_time()
        if not next_time:
            return None

        now = datetime.now(timezone.utc)
        delta = (next_time - now).total_seconds()
        return max(0, delta)

    def get_historical_rates(
        self,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[HistoricalFundingRate]:
        """
        Get historical funding rates.

        Args:
            limit: Number of records (max 1000)
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of historical funding rates
        """
        params = {
            "symbol": self.symbol,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        data = self._request("/fapi/v1/fundingRate", params=params)

        if not data:
            return []

        rates = []
        for record in data:
            try:
                rates.append(HistoricalFundingRate(
                    symbol=record["symbol"],
                    funding_rate=float(record["fundingRate"]),
                    funding_time=datetime.fromtimestamp(
                        int(record["fundingTime"]) / 1000, tz=timezone.utc
                    ),
                ))
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse historical rate: %s", e)

        logger.debug("Fetched %d historical funding rates", len(rates))
        return rates

    def get_rates_for_days(self, days: int) -> List[HistoricalFundingRate]:
        """
        Get funding rates for the last N days.

        Args:
            days: Number of days to fetch

        Returns:
            List of historical funding rates (3 per day)
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Each day has 3 funding periods, paginate if needed
        all_rates = []
        current_end = end_time

        while current_end > start_time:
            rates = self.get_historical_rates(
                limit=1000,
                start_time=start_time,
                end_time=current_end,
            )

            if not rates:
                break

            all_rates.extend(rates)

            # Move to earlier time
            current_end = rates[-1].funding_time - timedelta(seconds=1)

            # Rate limit protection
            time.sleep(0.2)

        # Sort by time ascending and deduplicate
        seen = set()
        unique_rates = []
        for rate in sorted(all_rates, key=lambda r: r.funding_time):
            key = (rate.symbol, rate.funding_time)
            if key not in seen:
                seen.add(key)
                unique_rates.append(rate)

        logger.info(
            "Fetched %d funding rates for last %d days",
            len(unique_rates),
            days,
        )

        return unique_rates

    def get_statistics(self, days: int = 30) -> dict:
        """
        Calculate funding rate statistics for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        rates = self.get_rates_for_days(days)

        if not rates:
            return {"error": "No data available"}

        funding_rates = [r.funding_rate for r in rates]

        positive_count = sum(1 for r in funding_rates if r > 0)
        negative_count = sum(1 for r in funding_rates if r < 0)

        return {
            "period_days": days,
            "total_periods": len(rates),
            "positive_periods": positive_count,
            "negative_periods": negative_count,
            "positive_pct": positive_count / len(rates) * 100,
            "mean_rate": sum(funding_rates) / len(funding_rates),
            "mean_rate_pct": sum(funding_rates) / len(funding_rates) * 100,
            "max_rate": max(funding_rates),
            "min_rate": min(funding_rates),
            "cumulative_rate": sum(funding_rates),
            "annualized_pct": sum(funding_rates) / days * 365 * 100,
        }

    def is_funding_time(self, tolerance_seconds: int = 60) -> bool:
        """
        Check if current time is within funding settlement window.

        Funding occurs at 00:00, 08:00, 16:00 UTC.

        Args:
            tolerance_seconds: Seconds after funding time to still consider valid

        Returns:
            True if within funding window
        """
        now = datetime.now(timezone.utc)

        # Funding hours
        funding_hours = [0, 8, 16]

        if now.hour in funding_hours and now.minute == 0:
            if now.second <= tolerance_seconds:
                return True

        return False

    def get_predicted_funding_rate(self) -> Optional[float]:
        """
        Get predicted funding rate for the next settlement.

        Uses the premiumIndex endpoint which provides the current funding rate
        that will be applied at the next funding time.

        Returns:
            Predicted funding rate as decimal, or None on failure
        """
        data = self._request(
            "/fapi/v1/premiumIndex",
            params={"symbol": self.symbol},
        )

        if not data:
            return None

        try:
            # lastFundingRate is the rate that will be applied at nextFundingTime
            return float(data["lastFundingRate"])
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse predicted rate: %s", e)
            return None

    def get_cached_rate(self) -> Optional[FundingRateData]:
        """Get the last cached funding rate."""
        return self._last_rate

    def get_cache_age_seconds(self) -> Optional[float]:
        """Get age of cached rate in seconds."""
        if not self._last_fetch_time:
            return None

        now = datetime.now(timezone.utc)
        return (now - self._last_fetch_time).total_seconds()


# Import here to avoid circular imports
from datetime import timedelta


def format_funding_rate(rate: float) -> str:
    """Format funding rate for display."""
    pct = rate * 100
    sign = "+" if rate >= 0 else ""
    return f"{sign}{pct:.4f}%"


def calculate_funding_payment(rate: float, position_size: float) -> float:
    """
    Calculate funding payment amount.

    Args:
        rate: Funding rate as decimal
        position_size: Position size in USD

    Returns:
        Payment amount in USD (positive = received, negative = paid)
    """
    return position_size * rate

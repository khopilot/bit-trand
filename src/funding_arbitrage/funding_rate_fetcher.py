"""
Funding Rate Fetcher with Pagination

Fetches ALL historical funding rates from Binance perpetual futures,
overcoming the 1000 record API limit through pagination.
"""

import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger("btc_trader.funding_arb.fetcher")


@dataclass
class FundingRateRecord:
    """Single funding rate record."""

    timestamp: datetime
    funding_rate: float
    mark_price: float
    symbol: str = "BTCUSDT"

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "funding_rate": self.funding_rate,
            "mark_price": self.mark_price,
            "symbol": self.symbol,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FundingRateRecord":
        """Create from dictionary (CSV import)."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            funding_rate=float(data["funding_rate"]),
            mark_price=float(data["mark_price"]),
            symbol=data.get("symbol", "BTCUSDT"),
        )

    @classmethod
    def from_binance(cls, data: Dict) -> "FundingRateRecord":
        """Create from Binance API response."""
        # Handle empty or missing markPrice
        mark_price_str = data.get("markPrice", "0")
        try:
            mark_price = float(mark_price_str) if mark_price_str else 0.0
        except (ValueError, TypeError):
            mark_price = 0.0

        return cls(
            timestamp=datetime.fromtimestamp(
                data["fundingTime"] / 1000, tz=timezone.utc
            ),
            funding_rate=float(data["fundingRate"]),
            mark_price=mark_price,
            symbol=data.get("symbol", "BTCUSDT"),
        )


class FundingRateFetcher:
    """
    Paginated historical funding rate fetcher.

    Fetches ALL available funding rates from Binance perpetual futures
    since launch (September 13, 2019) using pagination.
    """

    # Binance perpetual futures launched September 13, 2019
    BINANCE_PERPS_LAUNCH = datetime(2019, 9, 13, tzinfo=timezone.utc)

    # Funding settlement every 8 hours
    FUNDING_PERIOD_HOURS = 8
    FUNDING_PERIODS_PER_DAY = 3

    # API limits
    MAX_RECORDS_PER_CALL = 1000
    BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        api_timeout: int = 15,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ):
        """
        Initialize the fetcher.

        Args:
            symbol: Trading pair symbol
            api_timeout: Request timeout in seconds
            rate_limit_delay: Delay between API calls to avoid rate limits
            max_retries: Number of retries for failed requests
        """
        self.symbol = symbol
        self.api_timeout = api_timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._session = requests.Session()

        logger.info(
            "FundingRateFetcher initialized: symbol=%s, timeout=%ds",
            symbol,
            api_timeout,
        )

    def fetch_all_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[FundingRateRecord]:
        """
        Fetch complete funding rate history with pagination.

        Args:
            start_date: Start date (default: BINANCE_PERPS_LAUNCH)
            end_date: End date (default: now)
            progress_callback: Optional callback(fetched_count, estimated_total)

        Returns:
            List of all FundingRateRecord objects
        """
        if start_date is None:
            start_date = self.BINANCE_PERPS_LAUNCH
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Ensure timezone awareness
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        # Estimate total records
        days = (end_date - start_date).days
        estimated_total = days * self.FUNDING_PERIODS_PER_DAY

        logger.info(
            "Fetching funding rates: %s to %s (~%d records expected)",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            estimated_total,
        )

        records = self._paginate_requests(
            start_ms, end_ms, estimated_total, progress_callback
        )

        logger.info("Fetched %d funding rate records", len(records))
        return records

    def _paginate_requests(
        self,
        start_ms: int,
        end_ms: int,
        estimated_total: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[FundingRateRecord]:
        """
        Internal pagination logic.

        Strategy:
        1. Start from start_ms
        2. Fetch 1000 records
        3. Get last timestamp, use as new startTime + 1
        4. Repeat until no more records or endTime reached
        """
        all_records: List[FundingRateRecord] = []
        current_start = start_ms
        page = 0

        while current_start < end_ms:
            page += 1
            params = {
                "symbol": self.symbol,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": self.MAX_RECORDS_PER_CALL,
            }

            try:
                response = self._fetch_with_retry(params)
                data = response.json()

                if not data:
                    logger.debug("No more records (page %d)", page)
                    break

                # Convert to records
                page_records = [FundingRateRecord.from_binance(d) for d in data]
                all_records.extend(page_records)

                # Progress callback
                if progress_callback:
                    progress_callback(len(all_records), estimated_total)

                logger.debug(
                    "Page %d: fetched %d records (total: %d)",
                    page,
                    len(page_records),
                    len(all_records),
                )

                # Move start to after last record
                last_timestamp = data[-1]["fundingTime"]
                current_start = last_timestamp + 1

                # If we got less than max, we're done
                if len(data) < self.MAX_RECORDS_PER_CALL:
                    break

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except requests.RequestException as e:
                logger.error("Failed to fetch page %d: %s", page, e)
                raise

        return all_records

    def _fetch_with_retry(self, params: Dict) -> requests.Response:
        """Fetch with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(
                    self.BINANCE_API_URL,
                    params=params,
                    timeout=self.api_timeout,
                )
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise last_error

    def fetch_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[FundingRateRecord]:
        """Fetch funding rates for a specific date range."""
        return self.fetch_all_history(start_date=start_date, end_date=end_date)

    def save_to_csv(
        self,
        records: List[FundingRateRecord],
        filepath: str = "data/funding_rates/btcusdt_funding_history.csv",
    ) -> None:
        """
        Save records to CSV for offline analysis.

        Args:
            records: List of FundingRateRecord objects
            filepath: Output file path
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "funding_rate", "mark_price", "symbol"]
            )
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())

        logger.info("Saved %d records to %s", len(records), filepath)

    def load_from_csv(
        self,
        filepath: str = "data/funding_rates/btcusdt_funding_history.csv",
    ) -> List[FundingRateRecord]:
        """
        Load previously fetched records from CSV.

        Args:
            filepath: Input file path

        Returns:
            List of FundingRateRecord objects
        """
        if not os.path.exists(filepath):
            logger.warning("CSV file not found: %s", filepath)
            return []

        records = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(FundingRateRecord.from_dict(row))

        logger.info("Loaded %d records from %s", len(records), filepath)
        return records

    def get_cached_or_fetch(
        self,
        cache_filepath: str = "data/funding_rates/btcusdt_funding_history.csv",
        refresh_if_older_than_hours: int = 24,
    ) -> List[FundingRateRecord]:
        """
        Smart caching: Load from CSV if exists and fresh, otherwise fetch.

        Args:
            cache_filepath: Path to cache file
            refresh_if_older_than_hours: Refresh cache if older than this

        Returns:
            List of FundingRateRecord objects
        """
        cache_path = Path(cache_filepath)

        # Check if cache exists and is fresh
        if cache_path.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(
                cache_path.stat().st_mtime
            )
            cache_age_hours = cache_age.total_seconds() / 3600

            if cache_age_hours < refresh_if_older_than_hours:
                logger.info("Using cached data (%.1f hours old)", cache_age_hours)
                records = self.load_from_csv(cache_filepath)

                # Check if we need to append recent data
                if records:
                    last_timestamp = records[-1].timestamp
                    now = datetime.now(timezone.utc)

                    # If last record is more than 8 hours old, fetch new data
                    if (now - last_timestamp) > timedelta(hours=8):
                        logger.info("Fetching new records since %s", last_timestamp)
                        new_records = self.fetch_all_history(
                            start_date=last_timestamp + timedelta(seconds=1),
                            end_date=now,
                        )
                        if new_records:
                            records.extend(new_records)
                            self.save_to_csv(records, cache_filepath)

                return records

        # No cache or stale - fetch all
        logger.info("Cache not found or stale - fetching all history")
        records = self.fetch_all_history()
        self.save_to_csv(records, cache_filepath)
        return records

    def get_statistics(self, records: List[FundingRateRecord]) -> Dict:
        """
        Calculate statistics from funding rate records.

        Args:
            records: List of FundingRateRecord objects

        Returns:
            Statistics dictionary
        """
        if not records:
            return {"error": "No records provided"}

        rates = [r.funding_rate for r in records]
        positive_rates = [r for r in rates if r > 0]
        negative_rates = [r for r in rates if r < 0]

        # Time span
        first_date = records[0].timestamp
        last_date = records[-1].timestamp
        days_covered = (last_date - first_date).days

        return {
            "total_records": len(records),
            "first_date": first_date.isoformat(),
            "last_date": last_date.isoformat(),
            "days_covered": days_covered,
            "positive_periods": len(positive_rates),
            "negative_periods": len(negative_rates),
            "positive_pct": len(positive_rates) / len(rates) * 100,
            "avg_rate": sum(rates) / len(rates),
            "avg_rate_pct": sum(rates) / len(rates) * 100,
            "max_rate": max(rates),
            "max_rate_pct": max(rates) * 100,
            "min_rate": min(rates),
            "min_rate_pct": min(rates) * 100,
            "avg_positive_rate": (
                sum(positive_rates) / len(positive_rates) if positive_rates else 0
            ),
            "avg_negative_rate": (
                sum(negative_rates) / len(negative_rates) if negative_rates else 0
            ),
        }

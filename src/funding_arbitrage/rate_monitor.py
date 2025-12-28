"""
Funding Rate Monitor

Tracks funding rates across exchanges and provides entry/exit signals
for the funding arbitrage strategy.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("btc_trader.funding_arb.rate_monitor")


@dataclass
class FundingRate:
    """Funding rate data from an exchange."""

    exchange: str
    symbol: str
    rate: float  # Funding rate as decimal (0.0001 = 0.01%)
    next_funding_time: datetime
    timestamp: datetime

    @property
    def rate_pct(self) -> float:
        """Rate as percentage."""
        return self.rate * 100

    @property
    def annualized_rate(self) -> float:
        """Annualized rate assuming 3 funding periods per day."""
        return self.rate * 3 * 365 * 100


@dataclass
class FundingSignal:
    """Signal for funding arbitrage entry/exit."""

    action: str  # "enter", "exit", "hold"
    rate: float
    expected_yield_24h: float
    confidence: float
    reason: str


class FundingRateMonitor:
    """
    Monitor funding rates and generate entry/exit signals.

    Funding arbitrage is profitable when:
    - Funding rate is positive (longs pay shorts)
    - Rate is above minimum threshold (covers fees + slippage)
    - Rate is expected to stay positive for sufficient duration
    """

    def __init__(
        self,
        min_funding_rate: float = 0.0005,  # 0.05% per 8h minimum
        entry_threshold: float = 0.001,  # 0.1% per 8h to enter
        exit_threshold: float = -0.0001,  # -0.01% per 8h to exit
        api_timeout: int = 10,
    ):
        """
        Initialize the funding rate monitor.

        Args:
            min_funding_rate: Minimum rate to consider entering (covers fees)
            entry_threshold: Rate threshold to trigger entry signal
            exit_threshold: Rate threshold to trigger exit signal
            api_timeout: API request timeout in seconds
        """
        self.min_funding_rate = min_funding_rate
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.api_timeout = api_timeout

        # Historical rates for trend analysis
        self._rate_history: List[FundingRate] = []
        self._max_history = 100  # Keep last 100 rates

        logger.info(
            "FundingRateMonitor initialized: min=%.4f%%, entry=%.4f%%, exit=%.4f%%",
            min_funding_rate * 100,
            entry_threshold * 100,
            exit_threshold * 100,
        )

    def get_binance_funding_rate(self, symbol: str = "BTCUSDT") -> Optional[FundingRate]:
        """
        Get current funding rate from Binance Futures.

        Args:
            symbol: Trading pair symbol

        Returns:
            FundingRate if successful, None otherwise
        """
        try:
            # Get current funding rate
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            response = requests.get(url, timeout=self.api_timeout)
            response.raise_for_status()
            data = response.json()

            rate = float(data["lastFundingRate"])
            next_time = datetime.fromtimestamp(
                data["nextFundingTime"] / 1000, tz=timezone.utc
            )

            funding_rate = FundingRate(
                exchange="binance",
                symbol=symbol,
                rate=rate,
                next_funding_time=next_time,
                timestamp=datetime.now(timezone.utc),
            )

            # Store in history
            self._add_to_history(funding_rate)

            logger.debug(
                "Binance funding rate: %.4f%% (annualized: %.1f%%)",
                funding_rate.rate_pct,
                funding_rate.annualized_rate,
            )

            return funding_rate

        except requests.RequestException as e:
            logger.error("Failed to fetch Binance funding rate: %s", e)
            return None

    def get_okx_funding_rate(self, symbol: str = "BTC-USDT-SWAP") -> Optional[FundingRate]:
        """
        Get current funding rate from OKX.

        Args:
            symbol: Trading pair symbol

        Returns:
            FundingRate if successful, None otherwise
        """
        try:
            url = f"https://www.okx.com/api/v5/public/funding-rate?instId={symbol}"
            response = requests.get(url, timeout=self.api_timeout)
            response.raise_for_status()
            data = response.json()

            if data["code"] != "0" or not data["data"]:
                logger.warning("OKX API returned no data")
                return None

            rate_data = data["data"][0]
            rate = float(rate_data["fundingRate"])
            next_time = datetime.fromtimestamp(
                int(rate_data["nextFundingTime"]) / 1000, tz=timezone.utc
            )

            funding_rate = FundingRate(
                exchange="okx",
                symbol=symbol,
                rate=rate,
                next_funding_time=next_time,
                timestamp=datetime.now(timezone.utc),
            )

            self._add_to_history(funding_rate)

            logger.debug(
                "OKX funding rate: %.4f%% (annualized: %.1f%%)",
                funding_rate.rate_pct,
                funding_rate.annualized_rate,
            )

            return funding_rate

        except requests.RequestException as e:
            logger.error("Failed to fetch OKX funding rate: %s", e)
            return None

    def get_historical_funding_rates(
        self, exchange: str = "binance", symbol: str = "BTCUSDT", limit: int = 100
    ) -> List[Dict]:
        """
        Get historical funding rates for backtesting.

        Args:
            exchange: Exchange name
            symbol: Trading pair
            limit: Number of historical rates to fetch

        Returns:
            List of historical funding rate dictionaries
        """
        try:
            if exchange == "binance":
                url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
                response = requests.get(url, timeout=self.api_timeout)
                response.raise_for_status()
                data = response.json()

                return [
                    {
                        "timestamp": datetime.fromtimestamp(
                            d["fundingTime"] / 1000, tz=timezone.utc
                        ),
                        "rate": float(d["fundingRate"]),
                        "mark_price": float(d.get("markPrice", 0)),
                    }
                    for d in data
                ]

            logger.warning("Historical rates not implemented for %s", exchange)
            return []

        except requests.RequestException as e:
            logger.error("Failed to fetch historical rates: %s", e)
            return []

    def _add_to_history(self, rate: FundingRate) -> None:
        """Add rate to history, maintaining max size."""
        self._rate_history.append(rate)
        if len(self._rate_history) > self._max_history:
            self._rate_history = self._rate_history[-self._max_history :]

    def get_rate_trend(self, lookback: int = 10) -> Tuple[str, float]:
        """
        Analyze recent rate trend.

        Args:
            lookback: Number of recent rates to analyze

        Returns:
            Tuple of (trend direction, average rate)
        """
        if len(self._rate_history) < 3:
            return "unknown", 0.0

        recent = self._rate_history[-lookback:]
        rates = [r.rate for r in recent]
        avg_rate = sum(rates) / len(rates)

        # Simple trend detection
        if len(rates) >= 3:
            first_half_avg = sum(rates[: len(rates) // 2]) / (len(rates) // 2)
            second_half_avg = sum(rates[len(rates) // 2 :]) / (
                len(rates) - len(rates) // 2
            )

            if second_half_avg > first_half_avg * 1.2:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        return trend, avg_rate

    def generate_signal(
        self, current_rate: Optional[FundingRate] = None, has_position: bool = False
    ) -> FundingSignal:
        """
        Generate entry/exit signal based on current funding rate.

        Args:
            current_rate: Current funding rate (fetches if None)
            has_position: Whether we currently have an arb position

        Returns:
            FundingSignal with recommended action
        """
        if current_rate is None:
            current_rate = self.get_binance_funding_rate()

        if current_rate is None:
            return FundingSignal(
                action="hold",
                rate=0.0,
                expected_yield_24h=0.0,
                confidence=0.0,
                reason="Failed to fetch funding rate",
            )

        rate = current_rate.rate
        trend, avg_rate = self.get_rate_trend()

        # Calculate expected 24h yield (3 funding periods)
        expected_yield_24h = rate * 3 * 100  # As percentage

        # Decision logic
        if has_position:
            # Check for exit
            if rate < self.exit_threshold:
                return FundingSignal(
                    action="exit",
                    rate=rate,
                    expected_yield_24h=expected_yield_24h,
                    confidence=0.9,
                    reason=f"Funding rate negative: {rate*100:.4f}%",
                )
            elif rate < self.min_funding_rate and trend == "decreasing":
                return FundingSignal(
                    action="exit",
                    rate=rate,
                    expected_yield_24h=expected_yield_24h,
                    confidence=0.7,
                    reason=f"Rate declining below minimum: {rate*100:.4f}%",
                )
            else:
                return FundingSignal(
                    action="hold",
                    rate=rate,
                    expected_yield_24h=expected_yield_24h,
                    confidence=0.8,
                    reason=f"Holding position, rate: {rate*100:.4f}%",
                )
        else:
            # Check for entry
            if rate >= self.entry_threshold:
                confidence = min(0.9, 0.5 + (rate / self.entry_threshold) * 0.2)
                if trend == "increasing":
                    confidence = min(1.0, confidence + 0.1)
                elif trend == "decreasing":
                    confidence = max(0.5, confidence - 0.1)

                return FundingSignal(
                    action="enter",
                    rate=rate,
                    expected_yield_24h=expected_yield_24h,
                    confidence=confidence,
                    reason=f"High funding rate: {rate*100:.4f}% ({trend} trend)",
                )
            elif rate >= self.min_funding_rate:
                return FundingSignal(
                    action="hold",
                    rate=rate,
                    expected_yield_24h=expected_yield_24h,
                    confidence=0.5,
                    reason=f"Rate above minimum but below entry: {rate*100:.4f}%",
                )
            else:
                return FundingSignal(
                    action="hold",
                    rate=rate,
                    expected_yield_24h=expected_yield_24h,
                    confidence=0.3,
                    reason=f"Rate too low for entry: {rate*100:.4f}%",
                )

    def get_best_exchange(self) -> Tuple[str, float]:
        """
        Find which exchange has the best funding rate for shorting.

        Returns:
            Tuple of (exchange name, rate)
        """
        rates = {}

        binance_rate = self.get_binance_funding_rate()
        if binance_rate:
            rates["binance"] = binance_rate.rate

        okx_rate = self.get_okx_funding_rate()
        if okx_rate:
            rates["okx"] = okx_rate.rate

        if not rates:
            return "binance", 0.0

        best_exchange = max(rates, key=rates.get)
        return best_exchange, rates[best_exchange]

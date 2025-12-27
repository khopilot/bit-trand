"""
Data Service for BTC Elite Trader

Provides real-time and historical market data using ccxt/ccxt.pro.
Supports WebSocket streaming and REST fallback.

Author: khopilot
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt.pro as ccxtpro

    HAS_CCXT_PRO = True
except ImportError:
    HAS_CCXT_PRO = False

import ccxt

from .models import MarketData

logger = logging.getLogger("btc_trader.data_service")


class DataService:
    """
    Market data service with WebSocket streaming and REST fallback.

    Provides:
    - Real-time OHLCV data via WebSocket
    - Historical data loading
    - Automatic reconnection
    - Indicator calculation on new candles
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        sandbox: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Initialize DataService.

        Args:
            exchange_id: Exchange name (binance, bybit, etc.)
            symbol: Trading pair
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            sandbox: Use testnet if True
            api_key: API key for authenticated endpoints
            api_secret: API secret
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.sandbox = sandbox

        # Exchange configuration
        exchange_config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if api_key:
            exchange_config["apiKey"] = api_key
        if api_secret:
            exchange_config["secret"] = api_secret

        # Initialize REST exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(exchange_config)

        if sandbox:
            self.exchange.set_sandbox_mode(True)
            logger.info("DataService initialized in SANDBOX mode")

        # Initialize WebSocket exchange if available
        self.ws_exchange: Optional[ccxtpro.Exchange] = None
        if HAS_CCXT_PRO:
            ws_exchange_class = getattr(ccxtpro, exchange_id, None)
            if ws_exchange_class:
                self.ws_exchange = ws_exchange_class(exchange_config)
                if sandbox:
                    self.ws_exchange.set_sandbox_mode(True)

        # Data storage
        self.ohlcv_data: List[MarketData] = []
        self.df: Optional[pd.DataFrame] = None
        self.last_candle_time: Optional[datetime] = None

        # Callbacks
        self._on_candle_callbacks: List[Callable[[pd.DataFrame], None]] = []

        # State
        self._running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60

    def add_candle_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Register callback for new candle events."""
        self._on_candle_callbacks.append(callback)

    async def load_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Load historical OHLCV data.

        Args:
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data and calculated indicators
        """
        try:
            logger.info("Loading %d historical candles for %s", limit, self.symbol)
            ohlcv = await self._fetch_ohlcv_async(limit)

            if not ohlcv:
                logger.warning("No historical data received")
                return pd.DataFrame()

            self.ohlcv_data = [MarketData.from_list(candle) for candle in ohlcv]
            self.df = self._build_dataframe()
            self.last_candle_time = self.ohlcv_data[-1].timestamp

            logger.info(
                "Loaded %d candles, last: %s at $%.2f",
                len(self.ohlcv_data),
                self.last_candle_time,
                self.ohlcv_data[-1].close,
            )

            return self.df

        except Exception as e:
            logger.error("Failed to load history: %s", e)
            return pd.DataFrame()

    async def _fetch_ohlcv_async(self, limit: int) -> list:
        """Fetch OHLCV data asynchronously."""
        if self.ws_exchange:
            return await self.ws_exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        else:
            # Sync fallback in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            )

    def load_history_sync(self, limit: int = 100) -> pd.DataFrame:
        """Synchronous version of load_history."""
        try:
            logger.info("Loading %d historical candles (sync)", limit)
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)

            if not ohlcv:
                return pd.DataFrame()

            self.ohlcv_data = [MarketData.from_list(candle) for candle in ohlcv]
            self.df = self._build_dataframe()
            self.last_candle_time = self.ohlcv_data[-1].timestamp

            return self.df

        except Exception as e:
            logger.error("Failed to load history (sync): %s", e)
            return pd.DataFrame()

    def _build_dataframe(self) -> pd.DataFrame:
        """Build DataFrame from OHLCV data."""
        df = pd.DataFrame(
            [
                {
                    "Date": candle.timestamp.strftime("%Y-%m-%d"),
                    "Timestamp": candle.timestamp,
                    "Open": candle.open,
                    "High": candle.high,
                    "Low": candle.low,
                    "Close": candle.close,
                    "Volume": candle.volume,
                }
                for candle in self.ohlcv_data
            ]
        )
        return df

    async def start_streaming(self) -> None:
        """Start WebSocket streaming for real-time data."""
        if not self.ws_exchange:
            logger.warning("WebSocket not available, using polling fallback")
            await self._start_polling()
            return

        self._running = True
        self._reconnect_delay = 1

        logger.info("Starting WebSocket stream for %s %s", self.symbol, self.timeframe)

        while self._running:
            try:
                ohlcv = await self.ws_exchange.watch_ohlcv(self.symbol, self.timeframe)

                for candle in ohlcv:
                    candle_data = MarketData.from_list(candle)

                    # Check if new candle
                    if self.last_candle_time and candle_data.timestamp > self.last_candle_time:
                        self._on_new_candle(candle_data)

                self._reconnect_delay = 1  # Reset on successful receive

            except Exception as e:
                if not self._running:
                    break

                logger.error("WebSocket error: %s, reconnecting in %ds", e, self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _start_polling(self) -> None:
        """Fallback polling when WebSocket unavailable."""
        self._running = True

        # Calculate polling interval from timeframe
        intervals = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        poll_interval = intervals.get(self.timeframe, 3600)

        logger.info("Starting polling with %ds interval", poll_interval)

        while self._running:
            try:
                await self.load_history(limit=2)

                if self.ohlcv_data:
                    latest = self.ohlcv_data[-1]
                    if self.last_candle_time and latest.timestamp > self.last_candle_time:
                        self._on_new_candle(latest)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error("Polling error: %s", e)
                await asyncio.sleep(60)

    def _on_new_candle(self, candle: MarketData) -> None:
        """Handle new candle event."""
        logger.info(
            "New candle: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.2f",
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )

        self.ohlcv_data.append(candle)
        self.last_candle_time = candle.timestamp

        # Keep only last 200 candles
        if len(self.ohlcv_data) > 200:
            self.ohlcv_data = self.ohlcv_data[-200:]

        self.df = self._build_dataframe()

        # Notify callbacks
        for callback in self._on_candle_callbacks:
            try:
                callback(self.df)
            except Exception as e:
                logger.error("Candle callback error: %s", e)

    async def stop(self) -> None:
        """Stop streaming."""
        self._running = False

        if self.ws_exchange:
            await self.ws_exchange.close()

        logger.info("DataService stopped")

    def get_current_price(self) -> float:
        """Get current price from latest candle."""
        if self.ohlcv_data:
            return self.ohlcv_data[-1].close
        return 0.0

    def get_ticker(self) -> dict:
        """Get current ticker (sync)."""
        try:
            return self.exchange.fetch_ticker(self.symbol)
        except Exception as e:
            logger.error("Failed to fetch ticker: %s", e)
            return {}

    async def get_ticker_async(self) -> dict:
        """Get current ticker (async)."""
        try:
            if self.ws_exchange:
                return await self.ws_exchange.fetch_ticker(self.symbol)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.exchange.fetch_ticker, self.symbol)
        except Exception as e:
            logger.error("Failed to fetch ticker: %s", e)
            return {}


def calculate_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculate technical indicators on OHLCV DataFrame.

    Reuses logic from common.py but adapted for live data.

    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary

    Returns:
        DataFrame with added indicator columns
    """
    if df.empty:
        return df

    df = df.copy()
    strat = config.get("strategy", {})

    # EMA
    ema_fast = strat.get("ema_fast", 12)
    ema_slow = strat.get("ema_slow", 26)
    df["EMA_12"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()

    # RSI
    rsi_period = strat.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_period).mean()
    loss = loss.replace(0, np.finfo(float).eps)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_period = strat.get("bb_period", 20)
    bb_std = strat.get("bb_std", 2)
    df["BB_Mid"] = df["Close"].rolling(window=bb_period).mean()
    bb_rolling_std = df["Close"].rolling(window=bb_period).std()
    df["BB_Upper"] = df["BB_Mid"] + (bb_std * bb_rolling_std)
    df["BB_Lower"] = df["BB_Mid"] - (bb_std * bb_rolling_std)

    # ATR (True Range based)
    if "High" in df.columns and "Low" in df.columns:
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift()).abs()
        tr3 = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean()
    else:
        # Fallback: close-to-close volatility proxy
        df["ATR"] = df["Close"].pct_change().abs().rolling(window=14).mean() * df["Close"]

    # Forward fill NaN values for indicators
    df = df.ffill().bfill()

    return df

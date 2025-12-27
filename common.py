"""
BTC Trading Bot - Common Module

Production-grade shared utilities for data fetching, indicator calculation,
position sizing, and messaging. Designed for real trading signals.

Author: khopilot
"""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import requests
import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class Config(TypedDict):
    strategy: Dict[str, Any]
    fng: Dict[str, Any]
    api: Dict[str, Any]
    market: Dict[str, Any]
    logging: Dict[str, Any]


class LedgerEntry(TypedDict):
    date: str
    action: str  # BUY or SELL
    price: float
    reason: str
    fng_value: Optional[int]


# -----------------------------------------------------------------------------
# Module-level Logger
# -----------------------------------------------------------------------------

logger = logging.getLogger("btc_trader")


# -----------------------------------------------------------------------------
# Configuration Management
# -----------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.

    Returns:
        Config dictionary with all settings.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    if config_path is None:
        # Default to config.yaml in same directory as this module
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("Configuration loaded from %s", config_path)
    return config


def setup_logging(config: Optional[Config] = None) -> logging.Logger:
    """
    Configure centralized logging for the trading bot.

    Args:
        config: Configuration dictionary. If None, uses defaults.

    Returns:
        Configured root logger for btc_trader.
    """
    if config is None:
        log_level = "INFO"
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
    else:
        log_cfg = config.get("logging", {})
        log_level = log_cfg.get("level", "INFO")
        log_format = log_cfg.get(
            "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        date_format = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")

    # Configure root logger for btc_trader namespace
    root_logger = logging.getLogger("btc_trader")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


# -----------------------------------------------------------------------------
# Retry Decorator for API Calls
# -----------------------------------------------------------------------------

def create_retry_decorator(max_retries: int = 3):
    """
    Create a retry decorator with configurable attempts.

    Args:
        max_retries: Maximum number of retry attempts.

    Returns:
        Tenacity retry decorator.
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
        reraise=True,
    )


def fetch_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
    max_retries: int = 3,
) -> requests.Response:
    """
    Fetch URL with retry logic and exponential backoff.

    Args:
        url: URL to fetch.
        params: Query parameters.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        Response object.

    Raises:
        requests.RequestException: If all retries fail.
    """
    retry_decorator = create_retry_decorator(max_retries)

    @retry_decorator
    def _fetch() -> requests.Response:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response

    return _fetch()


# -----------------------------------------------------------------------------
# Data Fetching Functions
# -----------------------------------------------------------------------------

def fetch_btc_ohlc_binance(
    days: int = 365,
    timeout: int = 15,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch OHLC data from Binance API for ATR calculation.

    Args:
        days: Number of days of historical data.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close.
        Empty DataFrame on failure.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": min(days, 1000),  # Binance limit is 1000
    }

    try:
        response = fetch_with_retry(url, params, timeout, max_retries)
        data = response.json()

        ohlc_data = []
        for candle in data:
            # Binance kline format: [open_time, open, high, low, close, ...]
            open_time = candle[0]
            date_str = datetime.fromtimestamp(
                open_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            ohlc_data.append({
                "Date": date_str,
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
            })

        df = pd.DataFrame(ohlc_data)
        logger.info("Fetched %d days of OHLC data from Binance", len(df))
        return df

    except requests.RequestException as e:
        logger.error("Failed to fetch Binance OHLC data: %s", e)
        return pd.DataFrame()
    except (KeyError, IndexError, ValueError) as e:
        logger.error("Failed to parse Binance response: %s", e)
        return pd.DataFrame()


def fetch_btc_history(
    days: int = 365,
    timeout: int = 15,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch Bitcoin price history from CoinGecko API.

    Args:
        days: Number of days of historical data.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        DataFrame with columns: Date, Close.
        Empty DataFrame on failure.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily",
    }

    try:
        response = fetch_with_retry(url, params, timeout, max_retries)
        data = response.json()
        prices = data.get("prices", [])

        price_data = []
        for ts, price in prices:
            date_str = datetime.fromtimestamp(
                ts / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            price_data.append({"Date": date_str, "Close": price})

        df = pd.DataFrame(price_data)
        logger.info("Fetched %d days of price data from CoinGecko", len(df))
        return df

    except requests.RequestException as e:
        logger.error("Failed to fetch CoinGecko price data: %s", e)
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        logger.error("Failed to parse CoinGecko response: %s", e)
        return pd.DataFrame()


def fetch_fear_greed_history(
    days: int = 365,
    timeout: int = 10,
    max_retries: int = 3,
    default_value: int = 50,
) -> pd.DataFrame:
    """
    Fetch Fear & Greed Index history from Alternative.me API.

    Args:
        days: Number of days of historical data.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        default_value: Default FNG value used on API failure (logged as warning).

    Returns:
        DataFrame with columns: Date, FNG_Value, FNG_Class.
        Empty DataFrame on failure (caller should handle fallback).
    """
    url = f"https://api.alternative.me/fng/?limit={days}"

    try:
        response = fetch_with_retry(url, None, timeout, max_retries)
        data = response.json()
        fng_data = data.get("data", [])

        formatted_data = []
        for entry in fng_data:
            ts = int(entry["timestamp"])
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            formatted_data.append({
                "Date": date_str,
                "FNG_Value": int(entry["value"]),
                "FNG_Class": entry["value_classification"],
            })

        df = pd.DataFrame(formatted_data)
        logger.info("Fetched %d days of Fear & Greed data", len(df))
        return df

    except requests.RequestException as e:
        logger.warning(
            "Failed to fetch Fear & Greed data: %s. Using default value %d (neutral).",
            e,
            default_value,
        )
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        logger.warning(
            "Failed to parse Fear & Greed response: %s. Using default value %d (neutral).",
            e,
            default_value,
        )
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Technical Indicators
# -----------------------------------------------------------------------------

def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility-based stops.

    Requires DataFrame with High, Low, Close columns.

    Args:
        df: DataFrame with OHLC data.
        period: ATR lookback period.

    Returns:
        Series with ATR values.
    """
    if not all(col in df.columns for col in ["High", "Low", "Close"]):
        logger.warning("ATR calculation requires High, Low, Close columns. Returning zeros.")
        return pd.Series(0.0, index=df.index)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    # True Range: max of (H-L, |H-PC|, |L-PC|)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is smoothed average of True Range
    atr = true_range.rolling(window=period, min_periods=1).mean()

    return atr


def calculate_indicators(
    df: pd.DataFrame,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Calculate all technical indicators: EMA, RSI, Bollinger Bands, ATR.

    Args:
        df: DataFrame with at least Close column. If High/Low present, ATR is calculated.
        config: Configuration dictionary for indicator parameters.

    Returns:
        DataFrame with added indicator columns:
        - EMA_12, EMA_26 (or configured values)
        - RSI
        - BB_Upper, BB_Mid, BB_Lower
        - ATR (if OHLC data available)
    """
    if config is None:
        ema_fast = 12
        ema_slow = 26
        rsi_period = 14
        bb_period = 20
        bb_std = 2
    else:
        strat = config.get("strategy", {})
        ema_fast = strat.get("ema_fast", 12)
        ema_slow = strat.get("ema_slow", 26)
        rsi_period = strat.get("rsi_period", 14)
        bb_period = strat.get("bb_period", 20)
        bb_std = strat.get("bb_std", 2)

    df = df.copy()

    # EMA Crossover
    df["EMA_12"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()

    # RSI (Wilder's smoothing)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_period).mean()

    # Avoid division by zero
    rs = gain / loss.replace(0, np.finfo(float).eps)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # Neutral default during warmup

    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(window=bb_period).mean()
    df["BB_Std"] = df["Close"].rolling(window=bb_period).std()
    df["BB_Upper"] = df["BB_Mid"] + (bb_std * df["BB_Std"])
    df["BB_Lower"] = df["BB_Mid"] - (bb_std * df["BB_Std"])

    # ATR (if OHLC data available)
    if all(col in df.columns for col in ["High", "Low"]):
        df["ATR"] = calculate_atr(df, period=rsi_period)
    else:
        # Fallback: estimate ATR from close-to-close volatility
        # This is less accurate but allows basic functionality
        df["ATR"] = df["Close"].diff().abs().rolling(window=rsi_period).mean()
        logger.debug("Using close-to-close ATR estimate (no OHLC data)")

    return df


# -----------------------------------------------------------------------------
# Position Sizing
# -----------------------------------------------------------------------------

def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.02,
    max_pct: float = 0.25,
) -> Tuple[float, float]:
    """
    Calculate position size based on risk management rules.

    Uses fixed-fractional position sizing: risk a fixed percentage of capital
    per trade, with maximum position size cap.

    Args:
        capital: Available capital in USD.
        entry_price: Planned entry price.
        stop_price: Stop loss price.
        risk_pct: Percentage of capital to risk per trade (default 2%).
        max_pct: Maximum percentage of capital per position (default 25%).

    Returns:
        Tuple of (position_size_usd, btc_quantity).

    Raises:
        ValueError: If entry_price <= stop_price (invalid stop).
    """
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("Prices must be positive")

    if entry_price <= stop_price:
        raise ValueError(
            f"Entry price ({entry_price}) must be greater than stop price ({stop_price})"
        )

    # Calculate risk per unit
    risk_per_unit = entry_price - stop_price
    risk_pct_per_unit = risk_per_unit / entry_price

    # Amount to risk in USD
    risk_amount = capital * risk_pct

    # Position size based on risk
    position_size_usd = risk_amount / risk_pct_per_unit

    # Apply maximum position constraint
    max_position_usd = capital * max_pct
    position_size_usd = min(position_size_usd, max_position_usd)

    # Calculate BTC quantity
    btc_quantity = position_size_usd / entry_price

    logger.debug(
        "Position sizing: capital=%.2f, entry=%.2f, stop=%.2f -> size=%.2f USD (%.6f BTC)",
        capital,
        entry_price,
        stop_price,
        position_size_usd,
        btc_quantity,
    )

    return position_size_usd, btc_quantity


# -----------------------------------------------------------------------------
# Telegram Messaging
# -----------------------------------------------------------------------------

def send_telegram_message(
    message: str,
    timeout: int = 10,
) -> bool:
    """
    Send message to Telegram with console fallback.

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.

    Args:
        message: Message text (Markdown format supported).
        timeout: Request timeout in seconds.

    Returns:
        True if message sent successfully, False otherwise.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.info("Telegram credentials not configured. Printing to console.")
        print("-" * 60)
        print(message)
        print("-" * 60)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except requests.RequestException as e:
        logger.error("Failed to send Telegram message: %s", e)
        # Fallback to console
        print("-" * 60)
        print(message)
        print("-" * 60)
        return False


# -----------------------------------------------------------------------------
# Ledger Parsing
# -----------------------------------------------------------------------------

def parse_ledger_entry(entry: str) -> Optional[LedgerEntry]:
    """
    Universal ledger entry parser supporting both Elite and Pro formats.

    Supported formats:
    - Elite: "2025-01-01: BUY at $50,000.00 (Smart Trend | FNG: 45)"
    - Elite: "2025-01-01: SELL at $55,000.00 (Trailing Stop | FNG: 60)"
    - Pro: "Day 0 (2025-01-01): BUY at $50,000.00 (EMA Up + RSI 55.0)"
    - Pro: "Day 5 (2025-01-06): SELL at $52,000.00 (Trend Reversal)"

    Args:
        entry: Raw ledger string.

    Returns:
        Parsed LedgerEntry or None if parsing fails.
    """
    # Pattern 1: Elite format - "YYYY-MM-DD: ACTION at $PRICE (REASON | FNG: VALUE)"
    elite_pattern = r"(\d{4}-\d{2}-\d{2}):\s+(BUY|SELL)\s+at\s+\$([0-9,]+\.?\d*)\s+\(([^|]+?)(?:\s*\|\s*FNG:\s*(\d+))?\)"

    # Pattern 2: Pro format - "Day N (YYYY-MM-DD): ACTION at $PRICE (REASON)"
    pro_pattern = r"Day\s+\d+\s+\((\d{4}-\d{2}-\d{2})\):\s+(BUY|SELL)\s+at\s+\$([0-9,]+\.?\d*)\s+\(([^)]+)\)"

    # Try Elite format first
    match = re.match(elite_pattern, entry.strip())
    if match:
        date_str, action, price_str, reason, fng_str = match.groups()
        price = float(price_str.replace(",", ""))
        fng_value = int(fng_str) if fng_str else None
        return LedgerEntry(
            date=date_str,
            action=action,
            price=price,
            reason=reason.strip(),
            fng_value=fng_value,
        )

    # Try Pro format
    match = re.match(pro_pattern, entry.strip())
    if match:
        date_str, action, price_str, reason = match.groups()
        price = float(price_str.replace(",", ""))
        return LedgerEntry(
            date=date_str,
            action=action,
            price=price,
            reason=reason.strip(),
            fng_value=None,
        )

    logger.warning("Failed to parse ledger entry: %s", entry[:50])
    return None


def parse_ledger(ledger: List[str]) -> List[LedgerEntry]:
    """
    Parse a list of ledger entries.

    Args:
        ledger: List of raw ledger strings.

    Returns:
        List of parsed LedgerEntry objects (skips unparseable entries).
    """
    parsed = []
    for entry in ledger:
        result = parse_ledger_entry(entry)
        if result is not None:
            parsed.append(result)
    return parsed


# -----------------------------------------------------------------------------
# Module Initialization
# -----------------------------------------------------------------------------

def init_module(config_path: Optional[str] = None) -> Tuple[Config, logging.Logger]:
    """
    Initialize the common module with configuration and logging.

    Args:
        config_path: Optional path to config file.

    Returns:
        Tuple of (config, logger).
    """
    config = load_config(config_path)
    module_logger = setup_logging(config)
    return config, module_logger


# -----------------------------------------------------------------------------
# Convenience Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Types
    "Config",
    "LedgerEntry",
    # Configuration
    "load_config",
    "setup_logging",
    "init_module",
    # Data Fetching
    "fetch_btc_ohlc_binance",
    "fetch_btc_history",
    "fetch_fear_greed_history",
    "fetch_with_retry",
    # Indicators
    "calculate_atr",
    "calculate_indicators",
    # Position Sizing
    "calculate_position_size",
    # Messaging
    "send_telegram_message",
    # Ledger
    "parse_ledger_entry",
    "parse_ledger",
    # Logger
    "logger",
]

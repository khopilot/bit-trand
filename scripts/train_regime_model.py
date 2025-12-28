#!/usr/bin/env python3
"""
Train ML Regime Classifier for BTC Elite Trader

Fetches historical BTC data and trains XGBoost model to predict
market regimes (uptrend, downtrend, ranging).

Usage:
    python scripts/train_regime_model.py
    python scripts/train_regime_model.py --days 1095  # 3 years
    python scripts/train_regime_model.py --from-db    # Use database history

Author: khopilot
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_regime_detector import MLRegimeDetector, ML_AVAILABLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_regime")


def fetch_coingecko_history(days: int = 1095) -> pd.DataFrame:
    """
    Fetch BTC/USD OHLCV data from CoinGecko.

    Args:
        days: Number of days of history (max ~1095 for free tier)

    Returns:
        DataFrame with OHLCV columns
    """
    import requests

    logger.info(f"Fetching {days} days of BTC history from CoinGecko...")

    # CoinGecko OHLC endpoint (free tier)
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {
        "vs_currency": "usd",
        "days": str(min(days, 365)),  # CoinGecko limits OHLC to 365 days
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("Empty response from CoinGecko")

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
        df = df.set_index("Date")
        df = df.drop("Timestamp", axis=1)

        # Add volume (CoinGecko OHLC doesn't include it, so we fake it)
        df["Volume"] = np.random.uniform(10000, 50000, len(df))

        logger.info(f"Fetched {len(df)} OHLC candles from CoinGecko")
        return df

    except Exception as e:
        logger.error(f"CoinGecko fetch failed: {e}")
        raise


def fetch_binance_history(days: int = 1095) -> pd.DataFrame:
    """
    Fetch BTC/USDT OHLCV data from Binance.

    Args:
        days: Number of days of history

    Returns:
        DataFrame with OHLCV columns
    """
    import requests
    from datetime import timedelta

    logger.info(f"Fetching {days} days of BTC history from Binance...")

    url = "https://api.binance.com/api/v3/klines"
    all_data = []

    # Binance returns max 1000 candles per request
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    while start_time < end_time:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Move start time forward
            start_time = data[-1][0] + 1

            logger.debug(f"Fetched {len(data)} candles, total: {len(all_data)}")

        except Exception as e:
            logger.error(f"Binance fetch failed: {e}")
            break

    if not all_data:
        raise ValueError("No data fetched from Binance")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"
    ])

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.set_index("Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]

    logger.info(f"Fetched {len(df)} daily candles from Binance")
    return df


def fetch_from_database() -> pd.DataFrame:
    """
    Fetch historical data from PostgreSQL database.

    Returns:
        DataFrame with OHLCV columns
    """
    import asyncio
    import asyncpg
    import yaml

    logger.info("Fetching historical data from database...")

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_config = config.get("database", {})

    async def _fetch():
        conn = await asyncpg.connect(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            user=db_config.get("user", "postgres"),
            password=db_config.get("password", ""),
            database=db_config.get("database", "btc_trader"),
        )

        try:
            # Query historical OHLCV data
            rows = await conn.fetch("""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_history
                WHERE symbol = 'BTC/USDT'
                ORDER BY timestamp
            """)
            return rows
        finally:
            await conn.close()

    try:
        rows = asyncio.run(_fetch())

        if not rows:
            raise ValueError("No OHLCV history in database")

        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.set_index("Date")
        df = df.astype(float)

        logger.info(f"Fetched {len(df)} rows from database")
        return df

    except Exception as e:
        logger.error(f"Database fetch failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train ML Regime Classifier")
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Days of historical data (default: 1095 = 3 years)",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Fetch data from database instead of API",
    )
    parser.add_argument(
        "--source",
        choices=["binance", "coingecko"],
        default="binance",
        help="API source for historical data (default: binance)",
    )
    parser.add_argument(
        "--forward-days",
        type=int,
        default=5,
        help="Days ahead to predict (default: 5)",
    )
    parser.add_argument(
        "--up-threshold",
        type=float,
        default=0.02,
        help="Return threshold for uptrend label (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--down-threshold",
        type=float,
        default=-0.02,
        help="Return threshold for downtrend label (default: -0.02 = -2%%)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/regime_classifier.pkl",
        help="Path to save trained model",
    )
    args = parser.parse_args()

    if not ML_AVAILABLE:
        logger.error("ML libraries not installed. Run: pip install scikit-learn xgboost joblib")
        sys.exit(1)

    # Fetch historical data
    try:
        if args.from_db:
            df = fetch_from_database()
        elif args.source == "binance":
            df = fetch_binance_history(args.days)
        else:
            df = fetch_coingecko_history(args.days)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        sys.exit(1)

    if len(df) < 365:
        logger.error(f"Insufficient data for training: {len(df)} rows (need at least 365)")
        sys.exit(1)

    logger.info(f"Training data: {df.index[0]} to {df.index[-1]} ({len(df)} days)")

    # Initialize and train ML detector
    detector = MLRegimeDetector(
        model_path=args.model_path,
        enabled=True,
    )

    try:
        metrics = detector.train(
            df,
            forward_days=args.forward_days,
            up_threshold=args.up_threshold,
            down_threshold=args.down_threshold,
        )

        # Print results
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Training samples: {metrics['train_samples']}")
        print(f"Test samples: {metrics['test_samples']}")
        print(f"Train accuracy: {metrics['train_accuracy']:.3f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.3f}")

        print("\nFeature Importance:")
        importance = sorted(
            metrics['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, imp in importance:
            bar = "#" * int(imp * 50)
            print(f"  {feature:20s}: {imp:.3f} {bar}")

        print("\nClassification Report:")
        report = metrics['classification_report']
        for label in ['uptrend', 'downtrend', 'ranging']:
            if label in report:
                r = report[label]
                print(f"  {label:12s}: precision={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1-score']:.3f}")

        print(f"\nModel saved to: {args.model_path}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BTC Trading Bot Utilities
Common operations for testing and validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
from typing import Tuple, Optional

# Constants
USD_KHR_RATE = 4050
API_TIMEOUT = 10


def validate_data(df: pd.DataFrame) -> dict:
    """Validate DataFrame has required columns and no critical issues."""
    issues = []
    
    required_cols = ['Date', 'Close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    if df['Close'].isnull().any():
        null_count = df['Close'].isnull().sum()
        issues.append(f"Null prices: {null_count}")
    
    if (df['Close'] <= 0).any():
        issues.append("Non-positive prices detected")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'rows': len(df),
        'date_range': (df['Date'].min(), df['Date'].max()) if 'Date' in df.columns else None
    }


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators used in Elite strategy."""
    df = df.copy()
    
    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    avg_loss = avg_loss.replace(0, np.finfo(float).eps)
    df['RSI'] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Mid'] - (2 * bb_std)
    
    # ATR (recommended addition)
    if 'High' in df.columns and 'Low' in df.columns:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
    
    return df


def test_api_connectivity() -> dict:
    """Test connectivity to all external APIs."""
    results = {}
    
    # CoinGecko
    try:
        r = requests.get(
            'https://api.coingecko.com/api/v3/ping',
            timeout=API_TIMEOUT
        )
        results['coingecko'] = {
            'status': 'ok' if r.status_code == 200 else 'error',
            'code': r.status_code
        }
    except Exception as e:
        results['coingecko'] = {'status': 'error', 'message': str(e)}
    
    # Fear & Greed
    try:
        r = requests.get(
            'https://api.alternative.me/fng/?limit=1',
            timeout=API_TIMEOUT
        )
        results['fear_greed'] = {
            'status': 'ok' if r.status_code == 200 else 'error',
            'code': r.status_code
        }
    except Exception as e:
        results['fear_greed'] = {'status': 'error', 'message': str(e)}
    
    return results


def convert_to_khr(usd_amount: float) -> str:
    """Convert USD to KHR with Khmer formatting."""
    khr = usd_amount * USD_KHR_RATE
    return f"៛{khr:,.0f}"


def calculate_metrics(portfolio_values: list, initial_capital: float = 10000) -> dict:
    """Calculate performance metrics from portfolio value series."""
    if not portfolio_values:
        return {}
    
    final = portfolio_values[-1]
    returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
               for i in range(1, len(portfolio_values))]
    
    # Sharpe (annualized, assuming daily)
    if returns:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0
    else:
        sharpe = 0
    
    # Max Drawdown
    peak = initial_capital
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    return {
        'total_return': (final - initial_capital) / initial_capital * 100,
        'final_value_usd': final,
        'final_value_khr': final * USD_KHR_RATE,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd * 100,
        'days': len(portfolio_values)
    }


def parse_ledger_entry(entry: str) -> Optional[dict]:
    """Parse ledger entry (handles both Elite and Pro formats)."""
    import re
    
    # Elite format: "2025-01-15: BUY at $42,500.00 (Portfolio: $10,000.00)"
    elite_pattern = r'(\d{4}-\d{2}-\d{2}): (BUY|SELL) at \$([\d,]+\.?\d*)'
    
    # Pro format: "Day 0 (2025-01-01): BUY at $42,500.00"
    pro_pattern = r'Day \d+ \((\d{4}-\d{2}-\d{2})\): (BUY|SELL) at \$([\d,]+\.?\d*)'
    
    for pattern in [elite_pattern, pro_pattern]:
        match = re.match(pattern, entry)
        if match:
            return {
                'date': match.group(1),
                'action': match.group(2),
                'price': float(match.group(3).replace(',', ''))
            }
    
    return None


if __name__ == '__main__':
    print("Testing API connectivity...")
    results = test_api_connectivity()
    for api, status in results.items():
        print(f"  {api}: {status}")
    
    print(f"\nUSD/KHR conversion: $100 = {convert_to_khr(100)}")

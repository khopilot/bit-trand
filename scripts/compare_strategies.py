#!/usr/bin/env python3
"""
Strategy Comparison Script

Compares OLD strategy (1 trade/year, hold forever) vs NEW strategy (TP + exits).
Also tests regime-adaptive improvements.

Author: khopilot
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Position, StrategyConfig
from src.regime_detector import RegimeDetector, TrendRegime


@dataclass
class StrategyResult:
    """Results from a strategy backtest."""
    name: str
    initial_capital: float
    final_value: float
    roi_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[dict]


def fetch_historical_data(days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV data from Binance."""
    print(f"Fetching {days} days of historical data...")

    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )

    all_ohlcv = []
    current_since = since

    while True:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', since=current_since, limit=200)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        current_since = ohlcv[-1][0] + 86400000
        if len(ohlcv) < 200:
            break

    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df = df.drop('Timestamp', axis=1).drop_duplicates(subset=['Date']).reset_index(drop=True)

    print(f"Fetched {len(df)} days of data")
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = df.copy()

    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Mid'] - (2 * bb_std)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()

    return df.dropna()


def calculate_metrics(portfolio_values: List[float], trades: List[dict]) -> Tuple[float, float]:
    """Calculate max drawdown and Sharpe ratio."""
    if len(portfolio_values) < 2:
        return 0.0, 0.0

    # Max drawdown
    peak = portfolio_values[0]
    max_dd = 0.0
    for val in portfolio_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (daily returns)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    return max_dd * 100, sharpe


def run_old_strategy(df: pd.DataFrame, initial_capital: float = 10000) -> StrategyResult:
    """
    OLD Strategy: Buy on first EMA cross, hold forever (never exit).

    This simulates the original strategy that generated only 1 trade/year.
    """
    capital = initial_capital
    position = None
    portfolio_values = [initial_capital]
    trades = []
    wins = 0
    losses = 0

    for i in range(30, len(df)):  # Skip warmup
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        price = float(row['Close'])

        # Entry: EMA crossover (and no position)
        if position is None:
            ema_12 = float(row['EMA_12'])
            ema_26 = float(row['EMA_26'])
            prev_ema_12 = float(prev_row['EMA_12'])
            prev_ema_26 = float(prev_row['EMA_26'])
            rsi = float(row['RSI'])

            # BUY: EMA12 crosses above EMA26, RSI 50-70
            if ema_12 > ema_26 and prev_ema_12 <= prev_ema_26 and 50 <= rsi <= 70:
                quantity = (capital * 0.9) / price  # 90% of capital
                position = {
                    'entry_price': price,
                    'quantity': quantity,
                    'entry_date': str(row['Date']),
                }
                capital -= quantity * price
                trades.append({
                    'type': 'buy',
                    'date': str(row['Date']),
                    'price': price,
                    'quantity': quantity,
                })

        # NO EXIT - hold forever (the original bug)

        # Update portfolio value
        if position:
            portfolio_values.append(capital + position['quantity'] * price)
        else:
            portfolio_values.append(capital)

    # Final value
    if position:
        final_value = capital + position['quantity'] * df.iloc[-1]['Close']
    else:
        final_value = capital

    roi = (final_value - initial_capital) / initial_capital * 100
    max_dd, sharpe = calculate_metrics(portfolio_values, trades)

    return StrategyResult(
        name="OLD Strategy (Hold Forever)",
        initial_capital=initial_capital,
        final_value=final_value,
        roi_pct=roi,
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        win_rate=0.0,  # No exits = no win/loss
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        trades=trades,
    )


def run_new_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000,
    tp_levels: List[Tuple[float, float, str]] = None,
    trailing_stop_pct: float = 0.15,
) -> StrategyResult:
    """
    NEW Strategy: Buy on EMA cross, exit on TP or trailing stop.

    Args:
        tp_levels: List of (target_pct, sell_pct, name) tuples
        trailing_stop_pct: Trailing stop percentage from highs
    """
    if tp_levels is None:
        tp_levels = [
            (0.15, 0.33, "TP1"),  # +15% = sell 33%
            (0.30, 0.33, "TP2"),  # +30% = sell 33%
            (0.50, 0.34, "TP3"),  # +50% = sell 34%
        ]

    capital = initial_capital
    position = None
    portfolio_values = [initial_capital]
    trades = []
    wins = 0
    losses = 0

    for i in range(30, len(df)):  # Skip warmup
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        price = float(row['Close'])

        # Update highest price if in position
        if position:
            if price > position['highest_price']:
                position['highest_price'] = price

        # Entry logic
        if position is None:
            ema_12 = float(row['EMA_12'])
            ema_26 = float(row['EMA_26'])
            prev_ema_12 = float(prev_row['EMA_12'])
            prev_ema_26 = float(prev_row['EMA_26'])
            rsi = float(row['RSI'])

            # BUY: EMA12 crosses above EMA26, RSI 45-75
            if ema_12 > ema_26 and prev_ema_12 <= prev_ema_26 and 45 <= rsi <= 75:
                quantity = (capital * 0.9) / price
                position = {
                    'entry_price': price,
                    'quantity': quantity,
                    'initial_quantity': quantity,
                    'highest_price': price,
                    'entry_date': str(row['Date']),
                    'tp_executed': {"TP1": False, "TP2": False, "TP3": False},
                }
                capital -= quantity * price
                trades.append({
                    'type': 'buy',
                    'date': str(row['Date']),
                    'price': price,
                    'quantity': quantity,
                })

        # Exit logic
        elif position:
            unrealized_pct = (price - position['entry_price']) / position['entry_price']

            # Check take-profit levels
            exit_reason = None
            for tp_target, tp_sell_pct, tp_name in tp_levels:
                if unrealized_pct >= tp_target and not position['tp_executed'][tp_name]:
                    # Partial exit
                    sell_qty = position['initial_quantity'] * tp_sell_pct
                    if sell_qty > position['quantity']:
                        sell_qty = position['quantity']

                    position['tp_executed'][tp_name] = True
                    position['quantity'] -= sell_qty
                    pnl = (price - position['entry_price']) * sell_qty
                    capital += sell_qty * price

                    trades.append({
                        'type': 'sell_partial',
                        'date': str(row['Date']),
                        'price': price,
                        'quantity': sell_qty,
                        'pnl': pnl,
                        'reason': tp_name,
                    })

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    # If position fully closed
                    if position['quantity'] <= 0.0001:
                        position = None
                        break

            # Check trailing stop
            if position:
                stop_price = position['highest_price'] * (1 - trailing_stop_pct)
                if price < stop_price:
                    # Full exit
                    pnl = (price - position['entry_price']) * position['quantity']
                    capital += position['quantity'] * price

                    trades.append({
                        'type': 'sell',
                        'date': str(row['Date']),
                        'price': price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'reason': 'trailing_stop',
                    })

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    position = None

            # Check trend reversal
            if position:
                ema_12 = float(row['EMA_12'])
                ema_26 = float(row['EMA_26'])
                prev_ema_12 = float(prev_row['EMA_12'])
                prev_ema_26 = float(prev_row['EMA_26'])

                # Death cross
                if ema_12 < ema_26 and prev_ema_12 >= prev_ema_26:
                    pnl = (price - position['entry_price']) * position['quantity']
                    capital += position['quantity'] * price

                    trades.append({
                        'type': 'sell',
                        'date': str(row['Date']),
                        'price': price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'reason': 'death_cross',
                    })

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    position = None

        # Update portfolio value
        if position:
            portfolio_values.append(capital + position['quantity'] * price)
        else:
            portfolio_values.append(capital)

    # Final value
    if position:
        final_value = capital + position['quantity'] * df.iloc[-1]['Close']
    else:
        final_value = capital

    roi = (final_value - initial_capital) / initial_capital * 100
    max_dd, sharpe = calculate_metrics(portfolio_values, trades)
    total_trades = len([t for t in trades if t['type'] in ('sell', 'sell_partial')])

    return StrategyResult(
        name="NEW Strategy (TP + Trailing Stop)",
        initial_capital=initial_capital,
        final_value=final_value,
        roi_pct=roi,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=wins / max(1, wins + losses) * 100,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        trades=trades,
    )


def run_regime_adaptive_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000,
) -> StrategyResult:
    """
    REGIME-ADAPTIVE Strategy: Adjust TP levels based on market regime.

    - Strong uptrend: Higher TP (50/100/150%)
    - Weak uptrend: Standard TP (15/30/50%)
    - Ranging/Down: Aggressive TP (10/20/30%)
    """
    regime_detector = RegimeDetector()

    capital = initial_capital
    position = None
    portfolio_values = [initial_capital]
    trades = []
    wins = 0
    losses = 0
    cooldown_until = None  # Cooldown after trades

    for i in range(100, len(df)):  # Need more warmup for regime detection
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        price = float(row['Close'])
        current_date = row['Date']

        # Update highest price if in position
        if position:
            if price > position['highest_price']:
                position['highest_price'] = price

        # Detect regime
        current_df = df.iloc[:i+1]
        regime = regime_detector.detect(current_df)

        # Select TP levels based on regime
        if regime.trend == TrendRegime.STRONG_UPTREND:
            tp_levels = [
                (0.50, 0.33, "TP1"),  # +50% = sell 33%
                (1.00, 0.33, "TP2"),  # +100% = sell 33%
                (1.50, 0.34, "TP3"),  # +150% = sell 34%
            ]
            trailing_stop_pct = 0.20  # Wider stop in strong trend
        elif regime.trend == TrendRegime.WEAK_UPTREND:
            tp_levels = [
                (0.15, 0.33, "TP1"),
                (0.30, 0.33, "TP2"),
                (0.50, 0.34, "TP3"),
            ]
            trailing_stop_pct = 0.15
        else:  # Ranging or downtrend
            tp_levels = [
                (0.10, 0.33, "TP1"),  # Tighter TP
                (0.20, 0.33, "TP2"),
                (0.30, 0.34, "TP3"),
            ]
            trailing_stop_pct = 0.10  # Tighter stop

        # Check cooldown
        if cooldown_until is not None and current_date < cooldown_until:
            # Update portfolio value and continue
            if position:
                portfolio_values.append(capital + position['quantity'] * price)
            else:
                portfolio_values.append(capital)
            continue

        # Entry logic
        if position is None:
            ema_12 = float(row['EMA_12'])
            ema_26 = float(row['EMA_26'])
            prev_ema_12 = float(prev_row['EMA_12'])
            prev_ema_26 = float(prev_row['EMA_26'])
            rsi = float(row['RSI'])

            # Adjust entry criteria based on regime
            rsi_low = 45 if regime.trend in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND) else 50
            rsi_high = 80 if regime.trend == TrendRegime.STRONG_UPTREND else 75

            # BUY: EMA12 crosses above EMA26
            if ema_12 > ema_26 and prev_ema_12 <= prev_ema_26 and rsi_low <= rsi <= rsi_high:
                # Position size based on regime
                size_mult = regime.position_size_multiplier if hasattr(regime, 'position_size_multiplier') else 1.0
                position_pct = min(0.9, 0.7 * size_mult)  # Base 70%, scaled by regime

                quantity = (capital * position_pct) / price
                position = {
                    'entry_price': price,
                    'quantity': quantity,
                    'initial_quantity': quantity,
                    'highest_price': price,
                    'entry_date': str(row['Date']),
                    'tp_executed': {"TP1": False, "TP2": False, "TP3": False},
                    'regime': regime.trend.value,
                }
                capital -= quantity * price
                trades.append({
                    'type': 'buy',
                    'date': str(row['Date']),
                    'price': price,
                    'quantity': quantity,
                    'regime': regime.trend.value,
                })

        # Exit logic
        elif position:
            unrealized_pct = (price - position['entry_price']) / position['entry_price']

            # Check take-profit levels
            for tp_target, tp_sell_pct, tp_name in tp_levels:
                if unrealized_pct >= tp_target and not position['tp_executed'].get(tp_name, False):
                    sell_qty = position['initial_quantity'] * tp_sell_pct
                    if sell_qty > position['quantity']:
                        sell_qty = position['quantity']

                    position['tp_executed'][tp_name] = True
                    position['quantity'] -= sell_qty
                    pnl = (price - position['entry_price']) * sell_qty
                    capital += sell_qty * price

                    trades.append({
                        'type': 'sell_partial',
                        'date': str(row['Date']),
                        'price': price,
                        'quantity': sell_qty,
                        'pnl': pnl,
                        'reason': tp_name,
                    })

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    if position['quantity'] <= 0.0001:
                        position = None
                        # Set cooldown (5 days)
                        cooldown_until = current_date + pd.Timedelta(days=5)
                        break

            # Check trailing stop
            if position:
                stop_price = position['highest_price'] * (1 - trailing_stop_pct)
                if price < stop_price:
                    pnl = (price - position['entry_price']) * position['quantity']
                    capital += position['quantity'] * price

                    trades.append({
                        'type': 'sell',
                        'date': str(row['Date']),
                        'price': price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'reason': 'trailing_stop',
                    })

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    position = None
                    cooldown_until = current_date + pd.Timedelta(days=5)

        # Update portfolio value
        if position:
            portfolio_values.append(capital + position['quantity'] * price)
        else:
            portfolio_values.append(capital)

    # Final value
    if position:
        final_value = capital + position['quantity'] * df.iloc[-1]['Close']
    else:
        final_value = capital

    roi = (final_value - initial_capital) / initial_capital * 100
    max_dd, sharpe = calculate_metrics(portfolio_values, trades)
    total_trades = len([t for t in trades if t['type'] in ('sell', 'sell_partial')])

    return StrategyResult(
        name="REGIME-ADAPTIVE Strategy",
        initial_capital=initial_capital,
        final_value=final_value,
        roi_pct=roi,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=wins / max(1, wins + losses) * 100,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        trades=trades,
    )


def print_comparison(results: List[StrategyResult], buyhold_roi: float):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON REPORT")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<25} | ", end="")
    for r in results:
        print(f"{r.name[:20]:<22} | ", end="")
    print("Buy & Hold")

    print("-" * 80)

    # ROI
    print(f"{'ROI (%)':<25} | ", end="")
    for r in results:
        print(f"{r.roi_pct:>+21.2f}% | ", end="")
    print(f"{buyhold_roi:>+.2f}%")

    # Alpha vs B&H
    print(f"{'Alpha vs B&H (%)':<25} | ", end="")
    for r in results:
        alpha = r.roi_pct - buyhold_roi
        print(f"{alpha:>+21.2f}% | ", end="")
    print("0.00%")

    # Final Value
    print(f"{'Final Value ($)':<25} | ", end="")
    for r in results:
        print(f"${r.final_value:>19,.0f} | ", end="")
    bh_final = results[0].initial_capital * (1 + buyhold_roi / 100)
    print(f"${bh_final:>,.0f}")

    # Trades
    print(f"{'Total Trades':<25} | ", end="")
    for r in results:
        print(f"{r.total_trades:>22} | ", end="")
    print("1")

    # Win Rate
    print(f"{'Win Rate (%)':<25} | ", end="")
    for r in results:
        print(f"{r.win_rate:>21.1f}% | ", end="")
    print("N/A")

    # Max Drawdown
    print(f"{'Max Drawdown (%)':<25} | ", end="")
    for r in results:
        print(f"{r.max_drawdown_pct:>21.1f}% | ", end="")
    print("N/A")

    # Sharpe Ratio
    print(f"{'Sharpe Ratio':<25} | ", end="")
    for r in results:
        print(f"{r.sharpe_ratio:>22.2f} | ", end="")
    print("N/A")

    print("=" * 80)

    # Winner
    best = max(results, key=lambda x: x.roi_pct)
    if best.roi_pct > buyhold_roi:
        print(f"\nWINNER: {best.name} (+{best.roi_pct - buyhold_roi:.2f}% alpha)")
    else:
        print(f"\nWINNER: Buy & Hold (+{buyhold_roi - best.roi_pct:.2f}% over best strategy)")


def main():
    print("=" * 80)
    print("STRATEGY COMPARISON: OLD vs NEW vs REGIME-ADAPTIVE")
    print("=" * 80)

    # Fetch data
    df = fetch_historical_data(days=365)
    df = calculate_indicators(df)

    # Calculate buy-and-hold
    start_price = df.iloc[30]['Close']  # After warmup
    end_price = df.iloc[-1]['Close']
    buyhold_roi = (end_price - start_price) / start_price * 100

    print(f"\nBuy-and-Hold: ${start_price:,.0f} -> ${end_price:,.0f} ({buyhold_roi:+.2f}%)")

    # Run strategies
    print("\nRunning OLD strategy (hold forever)...")
    old_result = run_old_strategy(df)
    print(f"  -> {old_result.total_trades} trades, {old_result.roi_pct:+.2f}% ROI")

    print("\nRunning NEW strategy (TP + trailing stop)...")
    new_result = run_new_strategy(df)
    print(f"  -> {new_result.total_trades} trades, {new_result.roi_pct:+.2f}% ROI")

    print("\nRunning REGIME-ADAPTIVE strategy...")
    adaptive_result = run_regime_adaptive_strategy(df)
    print(f"  -> {adaptive_result.total_trades} trades, {adaptive_result.roi_pct:+.2f}% ROI")

    # Print comparison
    results = [old_result, new_result, adaptive_result]
    print_comparison(results, buyhold_roi)

    # Print trade details for best strategy
    best = max(results, key=lambda x: x.roi_pct)
    print(f"\n{best.name} Trade Log:")
    print("-" * 60)
    for trade in best.trades[:10]:  # First 10 trades
        t_type = trade['type']
        t_date = trade['date'][:10]
        t_price = trade['price']
        pnl = trade.get('pnl', 0)
        reason = trade.get('reason', '')
        if pnl:
            print(f"  {t_date}: {t_type:<12} @ ${t_price:>10,.0f}  P&L: ${pnl:>+8,.0f}  ({reason})")
        else:
            print(f"  {t_date}: {t_type:<12} @ ${t_price:>10,.0f}")

    if len(best.trades) > 10:
        print(f"  ... and {len(best.trades) - 10} more trades")

    return results


if __name__ == "__main__":
    results = main()

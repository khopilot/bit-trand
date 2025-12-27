"""
BTC Professional Backtest Runner

Imports core strategy from btc_trader and calculates advanced metrics.
Uses shared ledger parsing from common.py to support both Elite and Pro formats.

Author: khopilot
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from btc_trader import run_elite_strategy
from common import (
    calculate_indicators,
    fetch_btc_history,
    fetch_btc_ohlc_binance,
    fetch_fear_greed_history,
)
from common import (
    LedgerEntry,
    load_config,
    parse_ledger,
    setup_logging,
    Config,
)

# Module logger
logger = logging.getLogger("btc_trader.backtest")


def calculate_advanced_metrics(
    df: pd.DataFrame,
    ledger: List[str],
) -> Dict[str, Any]:
    """
    Calculate advanced trading metrics from backtest results.

    Metrics:
    - Total ROI
    - Max Drawdown
    - Sharpe Ratio (annualized, 365 trading days)
    - Win Rate
    - Profit Factor

    Args:
        df: DataFrame with 'Portfolio Value' and 'Close' columns.
        ledger: List of raw ledger entry strings.

    Returns:
        Dictionary of calculated metrics.
    """
    if df.empty:
        logger.warning("Empty DataFrame provided, returning zero metrics")
        return {
            "ROI": 0.0,
            "Max Drawdown": 0.0,
            "Sharpe Ratio": 0.0,
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
            "Total Trades": 0,
        }

    df = df.copy()

    # 1. Total ROI
    start_val = df["Portfolio Value"].iloc[0]
    final_val = df["Portfolio Value"].iloc[-1]
    total_roi = ((final_val - start_val) / start_val) * 100

    # 2. Max Drawdown
    df["Peak"] = df["Portfolio Value"].cummax()
    df["Drawdown"] = (df["Portfolio Value"] - df["Peak"]) / df["Peak"]
    max_drawdown = df["Drawdown"].min() * 100

    # 3. Sharpe Ratio (annualized)
    df["Daily_Return"] = df["Portfolio Value"].pct_change()
    mean_daily_return = df["Daily_Return"].mean()
    std_daily_return = df["Daily_Return"].std()

    if std_daily_return > 0:
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(365)
    else:
        sharpe_ratio = 0.0

    # 4. Win Rate & Profit Factor from Ledger
    # Use universal parser from common.py
    parsed_entries: List[LedgerEntry] = parse_ledger(ledger)

    trades: List[float] = []
    current_buy_price = 0.0
    final_price = df["Close"].iloc[-1] if not df.empty else 0.0

    for entry in parsed_entries:
        if entry["action"] == "BUY":
            current_buy_price = entry["price"]
        elif entry["action"] == "SELL":
            if current_buy_price > 0:
                pnl = (entry["price"] - current_buy_price) / current_buy_price
                trades.append(pnl)
                current_buy_price = 0.0

    # Handle open position at end of backtest
    if current_buy_price > 0 and final_price > 0:
        unrealized_pnl = (final_price - current_buy_price) / current_buy_price
        trades.append(unrealized_pnl)
        logger.debug(
            "Open position closed at final price: $%.2f, unrealized PnL: %.2f%%",
            final_price,
            unrealized_pnl * 100,
        )

    # Calculate win/loss metrics
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0

    # Handle division by zero: if no losses, profit factor is infinity (capped for display)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        # All winning trades, no losses
        profit_factor = float("inf")
    else:
        # No trades or all breakeven
        profit_factor = 0.0

    return {
        "ROI": total_roi,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Total Trades": len(trades),
    }


def print_backtest_report(metrics: Dict[str, Any], ledger: List[str]) -> None:
    """
    Print formatted backtest report to console.

    Args:
        metrics: Dictionary of calculated metrics.
        ledger: List of trade entries.
    """
    print("\n" + "=" * 50)
    print("PROFESSIONAL BACKTEST REPORT (Last 365 Days)")
    print("=" * 50)

    print("\nOverall Performance:")
    print(f"  Total ROI:       {metrics['ROI']:>8.2f}%")
    print(f"  Max Drawdown:    {metrics['Max Drawdown']:>8.2f}%")
    print(f"  Sharpe Ratio:    {metrics['Sharpe Ratio']:>8.2f} (>1 good, >2 very good)")

    print("\nTrade Statistics:")
    print(f"  Total Trades:    {metrics['Total Trades']:>8}")
    print(f"  Win Rate:        {metrics['Win Rate']:>8.2f}%")

    # Format profit factor (handle infinity)
    pf = metrics["Profit Factor"]
    if pf == float("inf"):
        pf_str = "     inf (no losses)"
    else:
        pf_str = f"{pf:>8.2f}"
    print(f"  Profit Factor:   {pf_str}")

    print("\n" + "-" * 50)
    print("Latest 5 Trades:")
    for entry in ledger[-5:]:
        print(f"  {entry}")
    print("=" * 50 + "\n")


def main() -> None:
    """
    Run professional backtest with advanced metrics.
    """
    # Load configuration and setup logging
    config = load_config()
    setup_logging(config)

    market_cfg = config.get("market", {})
    backtest_days = market_cfg.get("backtest_days", 365)

    logger.info("=== Professional Backtest Runner ===")

    # Step 1: Fetch OHLC data from Binance for proper ATR
    logger.info("Fetching %d days of OHLC data from Binance...", backtest_days)
    df_ohlc = fetch_btc_ohlc_binance(days=backtest_days)

    if df_ohlc.empty:
        logger.warning("Binance OHLC failed, falling back to CoinGecko...")
        df_prices = fetch_btc_history(days=backtest_days)
        if df_prices.empty:
            logger.error("Failed to fetch price data. Aborting backtest.")
            return
        df_ohlc = df_prices

    logger.info("Fetching Fear & Greed data...")
    df_fng = fetch_fear_greed_history(days=backtest_days)

    # Merge OHLC and FNG data
    if not df_fng.empty:
        df = pd.merge(df_ohlc, df_fng, on="Date", how="left")
    else:
        df = df_ohlc
        df["FNG_Value"] = 50  # Neutral default

    # Step 2: Calculate indicators
    logger.info("Calculating technical indicators...")
    df = calculate_indicators(df, config)

    # Step 3: Run strategy (now returns 3 values)
    logger.info("Executing Elite strategy...")
    df, ledger, stats = run_elite_strategy(df, config)

    # Step 4: Calculate metrics
    logger.info("Calculating advanced metrics...")
    metrics = calculate_advanced_metrics(df, ledger)

    # Step 5: Print report
    print_backtest_report(metrics, ledger)

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()

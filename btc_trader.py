"""
BTC Elite Trading Strategy with Production Risk Management

Implements the Elite trading strategy combining:
- EMA crossovers (trend)
- RSI (momentum)
- Bollinger Bands (volatility)
- Fear & Greed Index (sentiment)
- ATR-based dynamic trailing stops
- Position sizing with risk management
- Transaction cost (slippage) modeling

Author: khopilot
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (
    Config,
    calculate_indicators,
    calculate_position_size,
    fetch_btc_history,
    fetch_btc_ohlc_binance,
    fetch_fear_greed_history,
    load_config,
    send_telegram_message,
    setup_logging,
)

# Module logger
logger = logging.getLogger("btc_trader.elite")


def apply_slippage(price: float, is_buy: bool, slippage_pct: float) -> float:
    """
    Apply slippage to execution price.

    Args:
        price: Base price.
        is_buy: True if buying (slippage increases price).
        slippage_pct: Slippage as decimal (e.g., 0.001 for 0.1%).

    Returns:
        Adjusted price after slippage.
    """
    if is_buy:
        return price * (1 + slippage_pct)
    else:
        return price * (1 - slippage_pct)


def run_elite_strategy(
    df: pd.DataFrame,
    config: Optional[Config] = None,
) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Execute Elite trading strategy with ATR-based stops and position sizing.

    Strategy:
    - Smart Trend Entry: EMA12 > EMA26 + RSI 50-70 + FNG < 80
    - Contrarian Entry: Price < BB_Lower + RSI < 35 + FNG < 25
    - Exit: Trend reversal OR blow-off top OR ATR trailing stop

    Args:
        df: DataFrame with OHLC data, indicators, and FNG values.
        config: Configuration dictionary. If None, uses defaults.

    Returns:
        Tuple of (DataFrame with Portfolio Value, ledger list, stats dict).
    """
    if config is None:
        config = load_config()

    strat = config.get("strategy", {})
    fng_cfg = config.get("fng", {})
    market_cfg = config.get("market", {})

    # Strategy parameters from config
    rsi_momentum_low = strat.get("rsi_momentum_low", 50)
    rsi_momentum_high = strat.get("rsi_momentum_high", 70)
    rsi_oversold = strat.get("rsi_oversold", 35)
    rsi_overbought = strat.get("rsi_overbought", 75)
    atr_multiplier = strat.get("trailing_stop_atr_multiplier", 3.0)
    min_stop_pct = strat.get("min_stop_pct", 0.05)  # Minimum 5% stop
    slippage_pct = strat.get("slippage", 0.001)
    risk_per_trade = strat.get("risk_per_trade_pct", 0.01)
    max_position_pct = strat.get("max_position_pct", 0.25)

    fng_greed = fng_cfg.get("greed_threshold", 80)
    fng_fear = fng_cfg.get("fear_threshold", 25)
    fng_default = fng_cfg.get("default_value", 50)

    initial_capital = market_cfg.get("initial_capital", 10000)
    warmup_days = market_cfg.get("warmup_days", 30)

    # Initialize state
    cash = float(initial_capital)
    btc = 0.0
    portfolio_values = []
    ledger: List[str] = []

    position = None
    entry_price = 0.0
    highest_price_since_buy = 0.0

    # Track FNG data quality
    fng_missing_count = 0
    total_days = len(df)

    # Fill missing values
    df = df.copy()
    df = df.ffill()

    # Check FNG quality before filling with default
    if "FNG_Value" in df.columns:
        fng_missing_count = df["FNG_Value"].isna().sum()
        df["FNG_Value"] = df["FNG_Value"].fillna(fng_default)
    else:
        df["FNG_Value"] = fng_default
        fng_missing_count = total_days

    # Log FNG quality warning
    if fng_missing_count > 0:
        fng_quality_pct = ((total_days - fng_missing_count) / total_days) * 100
        logger.warning(
            "FNG data quality: %.1f%% available (%d/%d days missing, using default=%d)",
            fng_quality_pct,
            fng_missing_count,
            total_days,
            fng_default,
        )

    # Skip warmup period
    start_idx = min(warmup_days, len(df) - 1)
    logger.info("Skipping first %d days for indicator warmup", start_idx)

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["Close"]
        date = row["Date"]

        # During warmup, just track portfolio value
        if i < start_idx:
            portfolio_values.append(cash + (btc * price))
            continue

        # Technical indicators
        ema_12 = row["EMA_12"]
        ema_26 = row["EMA_26"]
        rsi = row["RSI"]
        bb_upper = row.get("BB_Upper", price * 1.02)
        bb_lower = row.get("BB_Lower", price * 0.98)
        atr = row.get("ATR", price * 0.02)  # Default 2% ATR if missing

        # Sentiment
        fng_val = row.get("FNG_Value", fng_default)

        # Previous EMAs for crossover detection
        if i > 0:
            prev_ema_12 = df["EMA_12"].iloc[i - 1]
            prev_ema_26 = df["EMA_26"].iloc[i - 1]
        else:
            prev_ema_12 = prev_ema_26 = 0

        # --- SELL LOGIC ---
        if position == "LONG":
            if price > highest_price_since_buy:
                highest_price_since_buy = price

            # ATR-based trailing stop with minimum
            atr_stop = atr * atr_multiplier
            min_stop = highest_price_since_buy * min_stop_pct
            stop_price = highest_price_since_buy - max(atr_stop, min_stop)

            # Exit conditions
            trend_reversal = (prev_ema_12 >= prev_ema_26) and (ema_12 < ema_26)
            blow_off_top = (price > bb_upper) and (rsi > rsi_overbought) and (fng_val > fng_greed)
            stop_hit = price < stop_price

            if trend_reversal or blow_off_top or stop_hit:
                exec_price = apply_slippage(price, is_buy=False, slippage_pct=slippage_pct)
                cash = btc * exec_price
                btc = 0.0
                position = None

                if stop_hit:
                    reason = f"ATR Stop (${stop_price:,.0f})"
                elif blow_off_top:
                    reason = "Blow-off Top"
                else:
                    reason = "Trend Reversal"

                pnl_pct = ((exec_price - entry_price) / entry_price) * 100
                ledger.append(
                    f"{date}: SELL at ${exec_price:,.2f} ({reason} | FNG: {int(fng_val)} | PnL: {pnl_pct:+.1f}%)"
                )
                logger.debug(
                    "%s: SELL at $%.2f, reason=%s, PnL=%.1f%%",
                    date, exec_price, reason, pnl_pct,
                )

        # --- BUY LOGIC ---
        elif position is None:
            is_uptrend = ema_12 > ema_26

            # Smart Trend Entry
            smart_trend_entry = (
                is_uptrend
                and (rsi_momentum_low < rsi < rsi_momentum_high)
                and (fng_val < fng_greed)
            )

            # Contrarian Sniper Entry
            contrarian_entry = (
                (price < bb_lower)
                and (rsi < rsi_oversold)
                and (fng_val < fng_fear)
            )

            if smart_trend_entry or contrarian_entry:
                # Calculate position size with risk management
                # Use total portfolio value (cash + unrealized), not just cash
                total_portfolio = cash + (btc * price)

                # Stop distance: max of ATR-based or minimum percentage
                atr_stop_distance = atr * atr_multiplier
                min_stop_distance = price * min_stop_pct
                stop_distance = max(atr_stop_distance, min_stop_distance)
                stop_price = price - stop_distance

                try:
                    position_usd, btc_qty = calculate_position_size(
                        capital=total_portfolio,  # Use total portfolio value
                        entry_price=price,
                        stop_price=stop_price,
                        risk_pct=risk_per_trade,
                        max_pct=max_position_pct,
                    )
                    # Don't exceed available cash
                    position_usd = min(position_usd, cash)
                except ValueError as e:
                    logger.warning("Position sizing error: %s, using 25%% of capital", e)
                    position_usd = min(cash, total_portfolio * max_position_pct)

                # Skip if position too small
                if position_usd < 10:
                    portfolio_values.append(cash + (btc * price))
                    continue

                exec_price = apply_slippage(price, is_buy=True, slippage_pct=slippage_pct)
                btc = position_usd / exec_price
                cash = cash - position_usd
                position = "LONG"
                entry_price = exec_price
                highest_price_since_buy = price

                entry_type = "Smart Trend" if smart_trend_entry else "Sniper Bottom"
                ledger.append(
                    f"{date}: BUY at ${exec_price:,.2f} ({entry_type} | FNG: {int(fng_val)} | Size: ${position_usd:,.0f})"
                )
                logger.debug(
                    "%s: BUY at $%.2f, type=%s, size=$%.0f",
                    date, exec_price, entry_type, position_usd,
                )

        current_value = cash + (btc * price)
        portfolio_values.append(current_value)

    df["Portfolio Value"] = portfolio_values

    # Stats for reporting
    stats = {
        "fng_missing_days": fng_missing_count,
        "fng_quality_pct": ((total_days - fng_missing_count) / total_days) * 100 if total_days > 0 else 0,
        "warmup_days": start_idx,
    }

    return df, ledger, stats


def generate_report(
    df: pd.DataFrame,
    ledger: List[str],
    config: Config,
    stats: dict,
) -> str:
    """
    Generate formatted trading report.

    Args:
        df: DataFrame with portfolio values.
        ledger: List of trade entries.
        config: Configuration dictionary.
        stats: Strategy execution stats.

    Returns:
        Formatted report string.
    """
    market_cfg = config.get("market", {})
    usd_khr_rate = market_cfg.get("usd_khr_rate", 4050)
    initial_capital = market_cfg.get("initial_capital", 10000)

    start_value = initial_capital
    final_value = df["Portfolio Value"].iloc[-1]
    roi = ((final_value - start_value) / start_value) * 100
    final_value_khr = final_value * usd_khr_rate

    start_date = df["Date"].iloc[0]
    end_date = df["Date"].iloc[-1]

    # Calculate max drawdown
    peak = df["Portfolio Value"].cummax()
    drawdown = (df["Portfolio Value"] - peak) / peak
    max_dd = drawdown.min() * 100

    report_lines = [
        "*Bitcoin ELITE-BOT Report (Cambodia Edition)*",
        "",
        f"*Strategy:* Trend + Bollinger + RSI + Sentiment",
        f"*Period:* {start_date} to {end_date}",
        f"*Start Capital:* ${start_value:,.2f}",
        f"*Final Value:* ${final_value:,.2f}",
        f"*Final Value (KHR):* {final_value_khr:,.0f}",
        f"*ROI:* {roi:+.2f}%",
        f"*Max Drawdown:* {max_dd:.2f}%",
        "",
    ]

    # FNG data quality indicator
    fng_quality = stats.get("fng_quality_pct", 100)
    if fng_quality < 100:
        report_lines.append(f"*FNG Data Quality:* {fng_quality:.0f}% (some days used default)")
        report_lines.append("")

    report_lines.append("*Trade Ledger (Last 10 Trades):*")
    if not ledger:
        report_lines.append("No trades executed.")
    else:
        for entry in ledger[-10:]:
            report_lines.append(f"- {entry}")

    return "\n".join(report_lines)


def main() -> None:
    """
    Run Elite trading strategy backtest.
    """
    # Load configuration and setup logging
    config = load_config()
    setup_logging(config)

    market_cfg = config.get("market", {})
    backtest_days = market_cfg.get("backtest_days", 365)

    logger.info("=== Bitcoin Elite Trading Bot ===")

    # Step 1: Fetch OHLC data from Binance for ATR
    logger.info("Fetching %d days of OHLC data from Binance...", backtest_days)
    df_ohlc = fetch_btc_ohlc_binance(days=backtest_days)

    if df_ohlc.empty:
        # Fallback to CoinGecko (no OHLC, will use close-to-close ATR)
        logger.warning("Binance OHLC failed, falling back to CoinGecko...")
        df_prices = fetch_btc_history(days=backtest_days)
        if df_prices.empty:
            logger.error("Failed to fetch price data. Aborting.")
            return
        df_ohlc = df_prices

    # Step 2: Fetch Fear & Greed data
    logger.info("Fetching Fear & Greed Index...")
    df_fng = fetch_fear_greed_history(days=backtest_days)

    # Merge OHLC and FNG data
    if not df_fng.empty:
        df = pd.merge(df_ohlc, df_fng, on="Date", how="left")
    else:
        df = df_ohlc
        logger.warning("FNG data unavailable, using default neutral value")

    # Step 3: Calculate indicators
    logger.info("Calculating technical indicators...")
    df = calculate_indicators(df, config)

    # Step 4: Run strategy
    logger.info("Running Elite strategy with ATR stops...")
    df, ledger, stats = run_elite_strategy(df, config)

    # Step 5: Generate and send report
    report = generate_report(df, ledger, config, stats)
    logger.info("Strategy complete. Sending report...")
    send_telegram_message(report)


if __name__ == "__main__":
    main()

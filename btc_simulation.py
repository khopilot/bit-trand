"""
BTC Monte Carlo Simulation with ATR-based Risk Management

Simulates future Bitcoin price paths using Geometric Brownian Motion,
calibrated from real market data. Applies Pro trading strategy with
dynamic ATR-based trailing stops and slippage modeling.

Author: khopilot
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (
    Config,
    calculate_indicators,
    fetch_btc_history,
    fetch_btc_ohlc_binance,
    load_config,
    send_telegram_message,
    setup_logging,
)

# Module logger
logger = logging.getLogger("btc_trader.simulation")

# Default simulation seed for reproducibility
DEFAULT_SEED = 42


def calibrate_gbm_parameters(prices: np.ndarray) -> Tuple[float, float]:
    """
    Calibrate drift and volatility from historical prices.

    Uses log returns to estimate parameters for Geometric Brownian Motion.

    Args:
        prices: Array of historical prices.

    Returns:
        Tuple of (drift, volatility) as daily values.
    """
    if len(prices) < 2:
        logger.warning("Insufficient prices for calibration, using defaults")
        return 0.0, 0.02  # Default: 0% drift, 2% daily volatility

    log_returns = np.log(prices[1:] / prices[:-1])
    drift = float(np.mean(log_returns))
    volatility = float(np.std(log_returns))

    logger.info(
        "Calibrated GBM parameters: drift=%.6f, volatility=%.6f",
        drift,
        volatility,
    )
    return drift, volatility


def simulate_gbm_path(
    start_price: float,
    drift: float,
    volatility: float,
    days: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate future price path using Geometric Brownian Motion.

    Args:
        start_price: Starting price for simulation.
        drift: Daily drift (mean log return).
        volatility: Daily volatility (std of log returns).
        days: Number of days to simulate.
        seed: Random seed for reproducibility. If None, uses DEFAULT_SEED.

    Returns:
        Array of simulated prices including start price.
    """
    actual_seed = seed if seed is not None else DEFAULT_SEED
    np.random.seed(actual_seed)
    logger.info("Simulation using seed: %d", actual_seed)

    prices = [start_price]
    for _ in range(days):
        shock = np.random.normal(0, 1)
        log_return = drift + volatility * shock
        new_price = prices[-1] * np.exp(log_return)
        prices.append(new_price)

    return np.array(prices)


def generate_simulated_ohlc(
    close_prices: np.ndarray,
    volatility: float,
) -> pd.DataFrame:
    """
    Generate synthetic OHLC data from close prices for ATR calculation.

    Estimates High/Low based on volatility assumptions.

    Args:
        close_prices: Array of simulated close prices.
        volatility: Daily volatility for High/Low estimation.

    Returns:
        DataFrame with Date, Open, High, Low, Close columns.
    """
    start_date = datetime.now(tz=timezone.utc)
    dates = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(len(close_prices))
    ]

    # Estimate OHLC: Open = previous close, High/Low = close +/- volatility factor
    opens = np.roll(close_prices, 1)
    opens[0] = close_prices[0]

    # High/Low estimated as +/- 0.5-1.5x daily volatility
    high_factor = 1 + volatility * 0.75
    low_factor = 1 - volatility * 0.75

    highs = np.maximum(close_prices, opens) * high_factor
    lows = np.minimum(close_prices, opens) * low_factor

    df = pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": close_prices,
    })

    return df


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


def run_pro_strategy_with_atr(
    df: pd.DataFrame,
    config: Config,
    initial_capital: float = 10000.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Execute Pro trading strategy with ATR-based trailing stops and slippage.

    Strategy:
    - Buy: EMA12 > EMA26 (uptrend) + RSI in momentum zone
    - Sell: Trend reversal OR ATR-based trailing stop hit

    Args:
        df: DataFrame with OHLC data and indicators.
        config: Configuration dictionary.
        initial_capital: Starting capital in USD.

    Returns:
        Tuple of (DataFrame with Portfolio Value, ledger list).
    """
    strat_cfg = config.get("strategy", {})
    rsi_momentum_low = strat_cfg.get("rsi_momentum_low", 50)
    rsi_momentum_high = strat_cfg.get("rsi_momentum_high", 75)
    atr_multiplier = strat_cfg.get("trailing_stop_atr_multiplier", 2.0)
    slippage_pct = strat_cfg.get("slippage", 0.001)

    cash = initial_capital
    btc = 0.0
    portfolio_values = []
    ledger: List[str] = []

    position = None
    entry_price = 0.0
    highest_price_since_buy = 0.0

    df = df.copy()

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        date = df["Date"].iloc[i]
        ema_12 = df["EMA_12"].iloc[i]
        ema_26 = df["EMA_26"].iloc[i]
        rsi = df["RSI"].iloc[i]
        atr = df["ATR"].iloc[i] if "ATR" in df.columns else price * 0.02

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

            # ATR-based trailing stop
            stop_price = highest_price_since_buy - (atr * atr_multiplier)
            trend_reversal = (prev_ema_12 >= prev_ema_26) and (ema_12 < ema_26)
            stop_hit = price < stop_price

            if trend_reversal or stop_hit:
                exec_price = apply_slippage(price, is_buy=False, slippage_pct=slippage_pct)
                cash = btc * exec_price
                btc = 0.0
                position = None

                reason = "Trend Reversal" if trend_reversal else f"ATR Stop (${stop_price:,.0f})"
                pnl_pct = ((exec_price - entry_price) / entry_price) * 100
                ledger.append(
                    f"Day {i} ({date}): SELL at ${exec_price:,.2f} ({reason}, PnL: {pnl_pct:+.1f}%)"
                )
                logger.debug(
                    "Day %d: SELL at $%.2f, reason=%s, PnL=%.1f%%",
                    i, exec_price, reason, pnl_pct,
                )

        # --- BUY LOGIC ---
        elif position is None:
            is_uptrend = ema_12 > ema_26
            valid_momentum = rsi_momentum_low < rsi < rsi_momentum_high

            if is_uptrend and valid_momentum:
                exec_price = apply_slippage(price, is_buy=True, slippage_pct=slippage_pct)
                btc = cash / exec_price
                cash = 0.0
                position = "LONG"
                entry_price = exec_price
                highest_price_since_buy = price

                ledger.append(
                    f"Day {i} ({date}): BUY at ${exec_price:,.2f} (EMA Up + RSI {rsi:.1f})"
                )
                logger.debug("Day %d: BUY at $%.2f, RSI=%.1f", i, exec_price, rsi)

        current_value = cash + (btc * price)
        portfolio_values.append(current_value)

    df["Portfolio Value"] = portfolio_values
    return df, ledger


def generate_simulation_report(
    df: pd.DataFrame,
    ledger: List[str],
    config: Config,
    seed: int,
) -> str:
    """
    Generate formatted simulation report.

    Args:
        df: DataFrame with portfolio values.
        ledger: List of trade entries.
        config: Configuration dictionary.
        seed: Random seed used for simulation.

    Returns:
        Formatted report string.
    """
    market_cfg = config.get("market", {})
    usd_khr_rate = market_cfg.get("usd_khr_rate", 4050)

    start_val = df["Portfolio Value"].iloc[0]
    final_val = df["Portfolio Value"].iloc[-1]
    roi = ((final_val - start_val) / start_val) * 100
    final_khr = final_val * usd_khr_rate

    # Calculate max drawdown
    peak = df["Portfolio Value"].cummax()
    drawdown = (df["Portfolio Value"] - peak) / peak
    max_dd = drawdown.min() * 100

    report_lines = [
        "*Bitcoin Monte Carlo Simulation (60 Days)*",
        "",
        f"Seed: {seed} (reproducible)",
        "Based on last 60 days of market conditions.",
        "",
        f"*Start Capital:* ${start_val:,.2f}",
        f"*Final Value:* ${final_val:,.2f}",
        f"*Final Value (KHR):* {final_khr:,.0f}",
        f"*ROI:* {roi:+.2f}%",
        f"*Max Drawdown:* {max_dd:.2f}%",
        "",
        "*Simulated Trades:*",
    ]

    if ledger:
        for entry in ledger:
            report_lines.append(f"- {entry}")
    else:
        report_lines.append("- No trades executed")

    return "\n".join(report_lines)


def main(seed: Optional[int] = None) -> None:
    """
    Run Monte Carlo simulation with Pro strategy.

    Args:
        seed: Random seed for reproducibility. If None, uses DEFAULT_SEED.
    """
    # Load configuration and setup logging
    config = load_config()
    setup_logging(config)

    actual_seed = seed if seed is not None else DEFAULT_SEED
    market_cfg = config.get("market", {})
    initial_capital = market_cfg.get("initial_capital", 10000)

    logger.info("=== Bitcoin Monte Carlo Simulation ===")

    # Step 1: Fetch real price data
    logger.info("Fetching last 60 days of real Bitcoin data...")
    df_prices = fetch_btc_history(days=60)

    if df_prices.empty:
        logger.error("Failed to fetch price data. Aborting simulation.")
        return

    real_prices = df_prices["Close"].values

    # Step 2: Calibrate GBM parameters
    logger.info("Calibrating simulation parameters...")
    drift, volatility = calibrate_gbm_parameters(real_prices)

    # Step 3: Simulate future prices
    logger.info("Running Monte Carlo simulation (60 days)...")
    start_price = real_prices[-1]
    sim_prices = simulate_gbm_path(
        start_price=start_price,
        drift=drift,
        volatility=volatility,
        days=60,
        seed=actual_seed,
    )

    # Step 4: Generate synthetic OHLC for ATR calculation
    logger.info("Generating synthetic OHLC data...")
    df_sim = generate_simulated_ohlc(sim_prices, volatility)

    # Step 5: Calculate indicators
    logger.info("Calculating technical indicators...")
    df_sim = calculate_indicators(df_sim, config)

    # Step 6: Run strategy
    logger.info("Executing Pro strategy with ATR stops...")
    df_sim, ledger = run_pro_strategy_with_atr(df_sim, config, initial_capital)

    # Step 7: Generate and send report
    report = generate_simulation_report(df_sim, ledger, config, actual_seed)
    logger.info("Simulation complete. Sending report...")
    send_telegram_message(report)


if __name__ == "__main__":
    main()

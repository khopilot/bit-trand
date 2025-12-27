# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin trading simulation bot with Telegram notifications, targeting the Cambodian market (USD/KHR conversion at 4050 rate). Implements technical analysis strategies using real market data from CoinGecko and Fear & Greed Index.

## Commands

```bash
# Setup
pip install -r requirements.txt

# Run live trading report (365-day backtest + Telegram notification)
python btc_trader.py

# Run Monte Carlo future simulation (60-day projection)
python btc_simulation.py

# Run professional backtest with advanced metrics
python backtest_runner.py
```

## Environment Variables

For Telegram notifications (optional - prints to console if not set):
- `TELEGRAM_BOT_TOKEN` - Telegram bot API token
- `TELEGRAM_CHAT_ID` - Target chat ID for notifications

## Architecture

**Module Dependencies:**
- `btc_trader.py` - Core module with reusable functions (fetch data, calculate indicators, run strategy)
- `backtest_runner.py` - Imports from btc_trader, adds advanced metrics calculation
- `btc_simulation.py` - Self-contained, duplicates some functions (simplified versions for simulation context)

**Entry Points:**

1. **btc_trader.py** - Main "Elite" strategy combining EMA crossovers, RSI, Bollinger Bands, and Fear & Greed sentiment. Fetches 365 days of real data, runs backtest, sends Telegram report.

2. **btc_simulation.py** - Monte Carlo simulation using Geometric Brownian Motion. Calibrates drift/volatility from 60 days of real prices, then projects 60 simulated future days. Uses simplified "Pro" strategy (no Bollinger Bands or FNG).

3. **backtest_runner.py** - Imports core functions from btc_trader, adds advanced metrics (Sharpe ratio, max drawdown, win rate, profit factor).

**Data Sources (public APIs, may have rate limits):**
- CoinGecko API: `api.coingecko.com` - BTC/USD daily prices
- Alternative.me API: `api.alternative.me/fng` - Fear & Greed Index

**Indicator Parameters:**
- EMA: 12 and 26 period
- RSI: 14 period
- Bollinger Bands: 20 period, 2 standard deviations
- Trailing Stop: 5%

**Trading Strategy ("Elite"):**
- Buy signals: EMA12 > EMA26 uptrend + RSI 50-70 + FNG < 80 (smart trend), or contrarian entry when price < lower BB + RSI < 35 + FNG < 25
- Sell signals: EMA death cross, blow-off top (price > upper BB + RSI > 75 + FNG > 80), or 5% trailing stop

## Key Functions (btc_trader.py)

- `fetch_btc_history(days)` - Returns DataFrame with Date/Close columns
- `fetch_fear_greed_history(days)` - Returns DataFrame with Date/FNG_Value/FNG_Class
- `calculate_indicators(df)` - Adds EMA_12, EMA_26, RSI, BB_Upper/Lower/Mid columns
- `run_elite_strategy(df)` - Returns (df with Portfolio Value, ledger list)
- `send_telegram_message(message)` - Sends Markdown-formatted message or prints to console

## Known Issues & Limitations

### Bugs (Fixed)

- ~~**RSI division by zero**~~: Fixed - uses `loss.replace(0, np.finfo(float).eps)` to avoid crash
- ~~**Open positions ignored**~~: Fixed - unrealized trades now counted in metrics at backtest end
- ~~**Timezone bug**~~: Fixed - all timestamps use `tz=timezone.utc` for consistent date alignment
- ~~**No request timeouts**~~: Fixed - all API calls now have 10-15s timeout

### Remaining Issues

- **Ledger parser breaks on Pro strategy**: `backtest_runner.py:47-60` - Hard-coded indices expect Elite format (`"2025-01-01: BUY..."`) but Pro uses (`"Day 0 (2025-01-01): BUY..."`)
- **No shared module**: Functions duplicated between `btc_trader.py` and `btc_simulation.py` instead of imported
- **No configuration system**: All parameters (EMA periods, thresholds, stop %) hardcoded inline
- **Silent FNG fallback**: When Alternative.me API fails, FNG defaults to 50 (neutral) without warning

### Strategy Notes

- **5% trailing stop**: May be too tight for crypto volatility (2-3% daily swings trigger false stops)
- **RSI threshold mismatch**: Elite uses 50-70, Pro uses 50-75 for momentum entry
- **No position sizing**: Contrarian entries (higher risk) use same allocation as trend entries

## Future Roadmap

### v1.1 - Code Quality
- Extract shared module (`common.py`) for duplicated functions
- Add config file (YAML/JSON) for strategy parameters
- Fix ledger parser to handle both Elite and Pro formats
- Add FNG fallback warning log

### v1.2 - Strategy Improvements
- Dynamic trailing stop based on ATR/volatility
- Position sizing based on signal confidence
- Align RSI thresholds across strategies

### v1.3 - Robustness
- Add retry logic with exponential backoff for API calls
- CLI arguments for days, capital, strategy selection
- Unit tests for indicator calculations

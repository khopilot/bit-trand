# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a **skill pack** for the BTC Trading Bot project. It provides reusable documentation, utilities, and reference materials. The main bot code lives in the parent directory.

## Quick Start

```bash
# Test API connectivity
python utils.py

# Main bot commands (run from parent directory)
python btc_trader.py         # 365d backtest + Telegram
python btc_simulation.py     # 60d Monte Carlo
python backtest_runner.py    # Advanced metrics
```

## File Reference

| File | Purpose |
|------|---------|
| [SKILL.md](SKILL.md) | **Primary reference** - architecture, strategies, bugs, fixes |
| [apis.md](apis.md) | API endpoints, response schemas, rate limits |
| [indicators.md](indicators.md) | EMA, RSI, Bollinger formulas and edge cases |
| [strategies.md](strategies.md) | Elite vs Pro logic, entry/exit conditions |
| [utils.py](utils.py) | Data validation, metrics, ledger parsing |

## When to Consult This Skill Pack

- Modifying indicator calculations → [indicators.md](indicators.md)
- Changing strategy thresholds → [strategies.md](strategies.md) or [SKILL.md](SKILL.md#strategy-matrix)
- Debugging API issues → [apis.md](apis.md)
- Understanding architecture → [SKILL.md](SKILL.md#architecture)
- Fixing known bugs → [SKILL.md](SKILL.md#critical-bugs)

## Key Utilities (utils.py)

```python
from utils import (
    validate_data,           # Check DataFrame quality
    calculate_all_indicators, # EMA, RSI, BB, optional ATR
    test_api_connectivity,   # Verify CoinGecko/FNG APIs
    calculate_metrics,       # Sharpe, drawdown, returns
    parse_ledger_entry,      # Handles Elite + Pro formats
    convert_to_khr,          # USD → KHR (rate: 4050)
)
```

## Environment

```bash
# Optional Telegram (falls back to stdout)
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

## Critical Notes

1. **SKILL.md is the source of truth** for strategy logic and known issues
2. **5% trailing stop is too tight** for crypto volatility - see ATR-based alternative in SKILL.md
3. **FNG API silently fails** to neutral (50) without logging - needs fix
4. **btc_simulation.py duplicates code** from btc_trader.py - refactor target

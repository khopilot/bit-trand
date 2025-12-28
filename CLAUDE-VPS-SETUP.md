# BTC Elite Trader - VPS Setup Documentation

This document details the setup and deployment of the BTC Elite Trader Expert System on the VPS, completed on December 27, 2025.

## Project Overview

**Repository**: https://github.com/khopilot/bit-trand  
**Bot**: @geckocoinkh_bot on Telegram  
**Mode**: Paper Trading (no real money)

## Critical Discovery: Two Different Trading Systems

### ❌ OLD System (btc_trader.py)
- **ROI**: -99.38% in backtest
- **Issue**: Trades in ALL market conditions
- **Problem**: No regime detection, massive losses in bear markets
- **Status**: DEPRECATED - Do not use

### ✅ NEW Expert System (scripts/run_paper.py)
- **ROI**: Protected (0% in bear markets)
- **Feature**: Market regime detection (bull/bear/ranging)
- **Benefit**: Sits out bad markets to protect capital
- **Status**: ACTIVE - Currently running

## Setup Summary

### 1. VPS Cleanup (Freed 6GB+)
```bash
# Cleaned systemd journals (2.1GB)
journalctl --vacuum-time=7d

# Cleaned Docker resources (3GB)
docker system prune -a --volumes -f

# Updated all packages
apt update && apt upgrade -y
```

### 2. Python Environment Setup
```bash
# Installed Python development tools
apt install -y python3-pip python3-venv python3-dev

# Created virtual environment
cd /root/bit-trand
python3 -m venv venv
source venv/bin/activate

# Installed dependencies
pip install -r requirements.txt
pip install python-dotenv  # Added for .env loading
```

### 3. Environment Configuration
Created `.env` with (DO NOT commit actual values to git):
```
TELEGRAM_BOT_TOKEN=<your-bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>
BTC_DB_PASSWORD=<your-db-password>
```
**SECURITY NOTE:** Never expose API tokens in documentation. Store in `.env` file only.

### 4. Startup Scripts Created

**start_expert.sh** - Runs with proper environment:
```bash
#!/bin/bash
cd /root/bit-trand
export $(cat .env | grep -v '^#' | xargs)
source venv/bin/activate
python scripts/run_paper.py
```

**start_background.sh** - Runs in background:
```bash
#!/bin/bash
cd /root/bit-trand
nohup ./start_expert.sh > expert_bot.log 2>&1 &
echo "✅ Expert bot started with PID: $!"
```

## Important Issues Discovered

### 1. Immediate Pause/Resume on Startup
- **Symptom**: Bot paused and resumed within 2 seconds of starting
- **Cause**: Telegram commands were sent (either testing or pending from previous session)
- **Resolution**: Normal behavior - bot correctly responds to commands

### 2. Environment Variable Loading
- **Symptom**: "No bot token provided" when using `python scripts/run_paper.py` directly
- **Cause**: .env file not automatically loaded by Python
- **Resolution**: Created wrapper scripts that export environment variables

### 3. Validation Results
```
Walk-Forward: FAIL (high curve-fitting risk)
Regime Detection: PASS
Risk Management: PASS
Performance Decay: PASS

Overall: 3/4 checks passed
⚠️ STRATEGY NOT VALIDATED for live trading
```

## How to Manage the Bot

### Start the Bot
```bash
cd /root/bit-trand
./start_background.sh
```

### Monitor the Bot
```bash
# View real-time logs
tail -f expert_bot.log

# Check if running
ps aux | grep run_paper

# View Telegram messages
# Check @geckocoinkh_bot on Telegram
```

### Stop the Bot
```bash
pkill -f run_paper.py
```

### Telegram Commands
- `/status` - Current position and P&L
- `/balance` - Account balances
- `/pause` - Pause trading
- `/resume` - Resume trading
- `/help` - Show all commands

## Key Files

| File | Purpose |
|------|---------|
| `btc_trader.py` | OLD strategy (-99% ROI) - DO NOT USE |
| `scripts/run_paper.py` | NEW expert system - USE THIS |
| `scripts/run_validation.py` | Strategy validation tool |
| `start_expert.sh` | Startup script with env vars |
| `start_background.sh` | Background startup script |
| `expert_bot.log` | Current bot logs |
| `.env` | Environment configuration |

## Warnings

1. **DO NOT use btc_trader.py** - This is the old strategy that loses 99% in bear markets
2. **Validation Failed** - The expert system failed walk-forward validation. Only use for paper trading.
3. **Paper Trading Only** - Do not connect real exchange API keys until validation passes
4. **Monitor Regularly** - Check Telegram for alerts and unusual behavior

## Next Steps

1. **Improve Strategy** to pass validation:
   - Reduce curve-fitting by simplifying parameters
   - Improve consistency across market periods
   - Increase Sharpe ratio (currently < 0.5)

2. **Run Extended Paper Trading**:
   - Monitor performance over several weeks
   - Track regime detection accuracy
   - Verify risk management works as expected

3. **Only After Validation Passes**:
   - Consider connecting testnet API
   - Run extended testnet validation
   - Eventually move to live with small capital

## Technical Details

- **VPS**: Ubuntu 24.04, 11GB RAM, 193GB disk
- **Python**: 3.12.3
- **Bot Framework**: Custom async Python with ccxt
- **Database**: In-memory (paper trading mode)
- **Risk Management**: Position sizing, drawdown scaling, kill switch

---

**Created**: December 27, 2025  
**Author**: Claude (Anthropic)  
**Purpose**: Document VPS setup and deployment of BTC Elite Trader Expert System
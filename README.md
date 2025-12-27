# BTC Elite Trader

Production-grade expert crypto trading bot with Telegram control.

## Features

- **Expert Strategy**: Regime-aware trading with ADX, Choppiness Index
- **Position Sizing**: Quarter Kelly with drawdown scaling
- **Risk Management**: ATR trailing stops, consecutive loss protection
- **Validation**: Walk-forward testing with Monte Carlo simulation
- **Control**: Telegram bot for monitoring and commands

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your credentials

# Run validation
python scripts/run_validation.py

# Run paper trading
python scripts/run_paper.py
```

### VPS Deployment (Docker)

```bash
# Clone repo on VPS
git clone <your-repo-url>
cd btc-elite-trader

# Setup environment
cp .env.example .env
nano .env  # Add your credentials

# Deploy
./deploy.sh paper     # Paper trading
./deploy.sh testnet   # Binance testnet
./deploy.sh live      # Live trading (requires confirmation)
```

## Telegram Commands

| Command | Action |
|---------|--------|
| `/status` | View position, P&L, regime |
| `/balance` | Account balances |
| `/pause` | Pause trading |
| `/resume` | Resume trading |
| `/kill confirm` | Emergency stop |

## Configuration

Edit `config/config.yaml`:

```yaml
position_sizing:
  kelly_fraction: 0.25      # Quarter Kelly
  use_regime_filter: true   # Only trade good conditions
  use_mtf_filter: true      # Daily trend confirmation

risk:
  max_daily_loss_pct: 0.05  # 5% daily loss limit
  max_drawdown_pct: 0.15    # 15% max drawdown
  consecutive_loss_limit: 3 # Recovery mode after 3 losses
```

## Architecture

```
DataService → StrategyEngine → RiskManager → Executor → Exchange
                   ↓                              ↓
            RegimeDetector                   Telegram
                   ↓
         PerformanceTracker → Database
```

## Safety Checklist

Before live trading:
- [ ] Run validation: `python scripts/run_validation.py`
- [ ] Paper trade 30+ days
- [ ] Testnet trade 7+ days
- [ ] Binance API: withdrawal DISABLED
- [ ] IP whitelist on exchange
- [ ] 2FA enabled

## Files

```
src/
├── strategy_engine.py    # Signal generation
├── risk_manager.py       # Safety controls
├── regime_detector.py    # Market classification
├── performance_tracker.py # Decay detection
├── executor_service.py   # Order execution
├── telegram_control.py   # Bot commands
└── orchestrator.py       # Main loop

scripts/
├── run_paper.py          # Paper trading
├── run_live.py           # Live trading
└── run_validation.py     # Strategy validation

config/
├── config.yaml           # Production config
└── config.testnet.yaml   # Conservative testnet
```

## License

Private - khopilot

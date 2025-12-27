# Trading Strategies Reference

## Elite Strategy (btc_trader.py)

Primary production strategy combining multiple signals.

### Entry Conditions

**Smart Trend Entry:**
```python
if (ema_12 > ema_26 and           # Uptrend confirmed
    50 <= rsi <= 70 and            # Healthy momentum
    fng < 80):                     # Not extreme greed
    signal = "BUY"
```

**Contrarian Entry:**
```python
if (price < bb_lower and           # Below lower band
    rsi < 35 and                   # Oversold
    fng < 25):                     # Extreme fear
    signal = "BUY"
```

### Exit Conditions

**Death Cross:**
```python
if ema_12 < ema_26:                # Trend reversal
    signal = "SELL"
```

**Blow-off Top:**
```python
if (price > bb_upper and           # Above upper band
    rsi > 75 and                   # Overbought
    fng > 80):                     # Extreme greed
    signal = "SELL"
```

**Trailing Stop:**
```python
trailing_stop_pct = 0.05  # 5%
if price < highest_since_buy * (1 - trailing_stop_pct):
    signal = "SELL"
```

## Pro Strategy (btc_simulation.py)

Simplified strategy for Monte Carlo simulation context.

### Differences from Elite

| Aspect | Elite | Pro |
|--------|-------|-----|
| Bollinger Bands | Yes | No |
| Fear & Greed | Yes | No |
| RSI Upper Bound | 70 | 75 |
| Context | Real backtest | Simulation |

### Pro Entry Logic
```python
if (ema_12 > ema_26 and
    50 <= rsi <= 75):              # Wider RSI range
    signal = "BUY"
```

## Strategy Parameters

### Current Defaults

```python
# Indicator periods
EMA_FAST = 12
EMA_SLOW = 26
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2

# Thresholds
RSI_MOMENTUM_LOW = 50
RSI_MOMENTUM_HIGH = 70  # 75 for Pro
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 75

FNG_GREED = 80
FNG_FEAR = 25

TRAILING_STOP = 0.05  # 5%
```

### Recommended Adjustments

**For High Volatility:**
- Increase trailing stop to 8-10%
- Consider ATR-based dynamic stop

**For Range-Bound Markets:**
- Widen RSI bands (40-80)
- Reduce position size on contrarian entries

## Position Sizing (Future)

Currently: 100% allocation per trade

Recommended improvement:
```python
# Risk-based sizing
position_size = min(
    capital * 0.02 / stop_distance,  # 2% risk per trade
    capital * 0.25                    # Max 25% position
)
```

## Signal Priority

When multiple conditions are met:
1. Exit signals take priority over entries
2. Trailing stop checked before other exits
3. Contrarian entries only when no position held

## Ledger Format

**Elite Format:**
```
2025-01-15: BUY at $42,500.00 (Portfolio: $10,000.00)
2025-01-20: SELL at $45,000.00 (Portfolio: $10,588.24)
```

**Pro Format (simulation):**
```
Day 0 (2025-01-01): BUY at $42,500.00
Day 5 (2025-01-06): SELL at $45,000.00
```

Note: `backtest_runner.py` parser expects Elite format only.

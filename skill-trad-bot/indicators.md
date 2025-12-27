# Technical Indicators Reference

## Exponential Moving Average (EMA)

### Formula
```
EMA_t = α × Price_t + (1 - α) × EMA_{t-1}
where α = 2 / (period + 1)
```

### Implementation
```python
def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()

# Usage
df['EMA_12'] = calculate_ema(df['Close'], 12)
df['EMA_26'] = calculate_ema(df['Close'], 26)
```

### Signals
- **Bullish crossover**: EMA_12 crosses above EMA_26
- **Bearish crossover (death cross)**: EMA_12 crosses below EMA_26

## Relative Strength Index (RSI)

### Formula
```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss over N periods
```

### Implementation
```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Prevent division by zero
    avg_loss = avg_loss.replace(0, np.finfo(float).eps)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Close'], 14)
```

### Interpretation
| RSI Range | Condition | Action |
|-----------|-----------|--------|
| > 70 | Overbought | Consider sell |
| 50-70 | Healthy momentum | Hold/Buy trend |
| 30-50 | Neutral | Wait |
| < 30 | Oversold | Consider buy |

## Bollinger Bands

### Formula
```
Middle Band = SMA(Close, 20)
Upper Band = Middle + (2 × σ)
Lower Band = Middle - (2 × σ)
where σ = rolling standard deviation
```

### Implementation
```python
def calculate_bollinger(prices: pd.Series, period: int = 20, 
                        num_std: float = 2.0) -> tuple:
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return upper, middle, lower

df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = calculate_bollinger(df['Close'])
```

### Signals
- **Price > Upper**: Potential overbought / trend strength
- **Price < Lower**: Potential oversold / mean reversion
- **Band squeeze**: Low volatility, breakout expected

## Average True Range (ATR) - Recommended Addition

### Formula
```
TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
ATR = SMA(TR, 14)
```

### Implementation
```python
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr
```

### Use Case
Dynamic trailing stop based on volatility:
```python
trailing_stop = current_price - (2 * atr)  # 2 ATR below price
```

## Combined Indicator Function

Current implementation in `btc_trader.py`:

```python
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    return df
```

## Edge Cases

### NaN Values
First N-1 values will be NaN for rolling calculations. Handle by:
```python
df = df.dropna()
# or
df['indicator'].fillna(method='bfill')  # Backfill for simulation start
```

### Zero Division
Always protect RSI calculation:
```python
avg_loss = avg_loss.replace(0, np.finfo(float).eps)
```

### Data Alignment
Ensure all DataFrames use consistent timezone:
```python
df['Date'] = pd.to_datetime(df['Date'], utc=True)
```

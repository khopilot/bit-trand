# API Reference

## CoinGecko API

Public API for cryptocurrency market data.

### Endpoint: Historical Prices
```
GET https://api.coingecko.com/api/v3/coins/bitcoin/market_chart
```

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| vs_currency | string | Target currency (usd) |
| days | integer | Days of history (1-365) |

### Response Schema
```json
{
  "prices": [
    [1704067200000, 42500.50],  // [timestamp_ms, price_usd]
    [1704153600000, 43100.25]
  ],
  "market_caps": [...],
  "total_volumes": [...]
}
```

### Implementation
```python
def fetch_btc_history(days: int = 365) -> pd.DataFrame:
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {'vs_currency': 'usd', 'days': days}
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['Timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df = df[['Date', 'Close']]
    
    return df
```

### Rate Limits
- Public API: ~30 requests/minute
- Consider caching for development

## Alternative.me Fear & Greed Index

Crypto market sentiment indicator.

### Endpoint: Historical FNG
```
GET https://api.alternative.me/fng/
```

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| limit | integer | Number of days (max 365) |

### Response Schema
```json
{
  "name": "Fear and Greed Index",
  "data": [
    {
      "value": "25",
      "value_classification": "Extreme Fear",
      "timestamp": "1704067200",
      "time_until_update": "..."
    }
  ]
}
```

### Classification Values
| Value | Classification |
|-------|----------------|
| 0-24 | Extreme Fear |
| 25-44 | Fear |
| 45-55 | Neutral |
| 56-75 | Greed |
| 76-100 | Extreme Greed |

### Implementation
```python
def fetch_fear_greed_history(days: int = 365) -> pd.DataFrame:
    url = f'https://api.alternative.me/fng/?limit={days}'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        records = []
        for item in data['data']:
            records.append({
                'Date': datetime.fromtimestamp(
                    int(item['timestamp']), 
                    tz=timezone.utc
                ),
                'FNG_Value': int(item['value']),
                'FNG_Class': item['value_classification']
            })
        
        return pd.DataFrame(records)
    
    except Exception as e:
        # Silent fallback - returns neutral value
        # TODO: Add warning log
        return create_default_fng(days)
```

### Fallback Behavior
When API fails, current implementation defaults to FNG=50 (neutral).

**Issue**: No warning logged. Recommended fix:
```python
import logging
logging.warning(f"FNG API failed: {e}. Using default value 50.")
```

## Telegram Bot API

### Endpoint: Send Message
```
POST https://api.telegram.org/bot{token}/sendMessage
```

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| chat_id | string | Target chat |
| text | string | Message content |
| parse_mode | string | "Markdown" or "HTML" |

### Implementation
```python
def send_telegram_message(message: str) -> bool:
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print(message)  # Fallback to console
        return False
    
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    response = requests.post(url, json=payload, timeout=15)
    return response.status_code == 200
```

### Markdown Formatting
```python
message = f"""
🤖 *BTC Elite Trading Report*
📅 Period: {days} days
💰 Final Portfolio: ${value:,.2f}
📈 Return: {return_pct:.2f}%
🇰🇭 KHR: ៛{value * 4050:,.0f}

📊 *Trades*
• Total: {num_trades}
• Win Rate: {win_rate:.1f}%
"""
```

## Error Handling Pattern

```python
def api_call_with_retry(url: str, params: dict = None, 
                        max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                time.sleep(60)
                continue
            raise
```

## Data Alignment

Both APIs return daily data but timestamps may not align perfectly.

```python
def merge_price_fng(price_df: pd.DataFrame, 
                    fng_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize dates to date-only for merging
    price_df['Date'] = price_df['Date'].dt.date
    fng_df['Date'] = fng_df['Date'].dt.date
    
    merged = price_df.merge(fng_df, on='Date', how='left')
    merged['FNG_Value'] = merged['FNG_Value'].fillna(50)  # Neutral default
    
    return merged
```

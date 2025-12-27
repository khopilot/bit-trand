# BTC Trading Bot: Simulation → Production Architecture

## Executive Summary

This document outlines the transformation from a backtesting simulation to a production live-trading system. The architecture prioritizes **safety over speed** — a bug in a trading bot is a direct debit from your bank account.

---

## Current State vs Target State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT (Simulation)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  CoinGecko API ──► btc_trader.py ──► Calculate ──► Simulate ──► Telegram    │
│  (daily, delayed)     (monolith)      Indicators    Trades       Report     │
│                                                                              │
│  Problems:                                                                   │
│  • No exchange connection          • No persistent state                    │
│  • Delayed data (5-10 min)         • No order management                    │
│  • Single-run execution            • No risk controls                       │
│  • Simulated P&L only              • No recovery from failures              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TARGET (Production)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │ Exchange │◄──│   Executor   │◄──│   Strategy   │◄──│    Data      │      │
│  │   API    │   │   Service    │   │    Engine    │   │   Service    │      │
│  └──────────┘   └──────────────┘   └──────────────┘   └──────────────┘      │
│       │               │                   │                   │              │
│       │               ▼                   ▼                   │              │
│       │         ┌──────────────────────────────────┐         │              │
│       │         │         PostgreSQL               │         │              │
│       │         │  • Positions  • Orders  • Trades │         │              │
│       │         └──────────────────────────────────┘         │              │
│       │                          │                            │              │
│       │                          ▼                            │              │
│       │         ┌──────────────────────────────────┐         │              │
│       └────────►│       Risk Manager (Watchdog)    │◄────────┘              │
│                 │  • Position limits • Kill switch │                        │
│                 └──────────────────────────────────┘                        │
│                                  │                                           │
│                                  ▼                                           │
│                 ┌──────────────────────────────────┐                        │
│                 │   Control Plane (Telegram/API)   │                        │
│                 │  • Start/Stop  • Status  • Alerts│                        │
│                 └──────────────────────────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Data Service

**Purpose**: Provide real-time and historical price data with calculated indicators.

```python
# data_service.py
import asyncio
import ccxt.pro as ccxtpro  # WebSocket support
import pandas as pd
from collections import deque
from typing import Callable
import logging

class DataService:
    """
    Real-time OHLCV data with indicator calculations.
    Replaces CoinGecko with exchange WebSocket feeds.
    """
    
    def __init__(
        self,
        exchange_id: str = 'binance',
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        history_length: int = 100  # Bars to keep in memory
    ):
        self.exchange = getattr(ccxtpro, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles = deque(maxlen=history_length)
        self.callbacks: list[Callable] = []
        self._running = False
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start WebSocket connection and begin streaming."""
        self._running = True
        self.logger.info(f"Starting data feed for {self.symbol} @ {self.timeframe}")
        
        # Load initial history
        ohlcv = await self.exchange.fetch_ohlcv(
            self.symbol, self.timeframe, limit=100
        )
        for candle in ohlcv:
            self.candles.append(candle)
        
        # Stream new candles
        while self._running:
            try:
                candle = await self.exchange.watch_ohlcv(
                    self.symbol, self.timeframe
                )
                self._process_candle(candle[0])
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)
                
    def _process_candle(self, candle: list):
        """Process new candle and trigger callbacks."""
        # candle = [timestamp, open, high, low, close, volume]
        if self.candles and self.candles[-1][0] == candle[0]:
            # Update existing candle (still forming)
            self.candles[-1] = candle
        else:
            # New candle
            self.candles.append(candle)
            
        df = self.to_dataframe()
        df = self.calculate_indicators(df)
        
        for callback in self.callbacks:
            callback(df)
            
    def to_dataframe(self) -> pd.DataFrame:
        """Convert candles to DataFrame."""
        df = pd.DataFrame(
            list(self.candles),
            columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
        return df
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators used by Elite strategy.
        IMPORTANT: This is called on EVERY new candle - must be fast.
        """
        # EMA
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI (with div/0 protection)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Mid'] - (2 * bb_std)
        
        # ATR for dynamic trailing stop
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(span=14, adjust=False).mean()
        
        return df
        
    def on_update(self, callback: Callable):
        """Register callback for data updates."""
        self.callbacks.append(callback)
        
    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        await self.exchange.close()
```

### 2. Strategy Engine

**Purpose**: Generate trading signals based on Elite strategy logic.

```python
# strategy_engine.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import logging

class Signal(Enum):
    NONE = "NONE"
    BUY_TREND = "BUY_TREND"
    BUY_CONTRARIAN = "BUY_CONTRARIAN"
    SELL_REVERSAL = "SELL_REVERSAL"
    SELL_BLOWOFF = "SELL_BLOWOFF"
    SELL_TRAILING_STOP = "SELL_TRAILING_STOP"


@dataclass
class StrategyConfig:
    """
    All strategy parameters in one place.
    Load from YAML in production.
    """
    # EMA
    ema_fast: int = 12
    ema_slow: int = 26
    
    # RSI
    rsi_momentum_low: float = 50
    rsi_momentum_high: float = 70
    rsi_oversold: float = 35
    rsi_overbought: float = 75
    
    # Fear & Greed
    fng_greed_threshold: float = 80
    fng_fear_threshold: float = 25
    
    # Trailing Stop
    trailing_stop_pct: float = 0.05  # 5%
    use_atr_stop: bool = True
    atr_multiplier: float = 2.0
    
    # Position sizing
    max_position_pct: float = 0.25  # Max 25% of capital per trade
    risk_per_trade_pct: float = 0.02  # Risk 2% per trade


@dataclass
class Position:
    """Current position state."""
    is_open: bool = False
    side: str = "none"  # "long" or "short"
    entry_price: float = 0.0
    quantity: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float('inf')


class StrategyEngine:
    """
    Elite strategy implementation.
    Stateless signal generation - state managed externally.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fng_value = 50  # Default neutral, updated async
        
    def update_fng(self, value: float):
        """Update Fear & Greed index (fetched separately)."""
        self.fng_value = value
        
    def generate_signal(
        self,
        df: pd.DataFrame,
        position: Position
    ) -> tuple[Signal, dict]:
        """
        Generate trading signal based on current market state.
        
        Returns:
            (Signal, metadata dict with reasoning)
        """
        if len(df) < 26:  # Need enough data for EMA_26
            return Signal.NONE, {"reason": "Insufficient data"}
            
        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else row
        
        price = row['Close']
        ema_12 = row['EMA_12']
        ema_26 = row['EMA_26']
        rsi = row['RSI']
        bb_upper = row['BB_Upper']
        bb_lower = row['BB_Lower']
        atr = row['ATR']
        
        meta = {
            "price": price,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "rsi": rsi,
            "fng": self.fng_value,
            "atr": atr
        }
        
        # === EXIT SIGNALS (check first) ===
        if position.is_open:
            # Update trailing stop reference
            if price > position.highest_since_entry:
                position.highest_since_entry = price
                
            # Calculate stop price
            if self.config.use_atr_stop:
                stop_distance = atr * self.config.atr_multiplier
            else:
                stop_distance = position.highest_since_entry * self.config.trailing_stop_pct
                
            stop_price = position.highest_since_entry - stop_distance
            meta["stop_price"] = stop_price
            
            # Trailing stop hit
            if price < stop_price:
                meta["reason"] = f"Trailing stop triggered at {stop_price:.2f}"
                return Signal.SELL_TRAILING_STOP, meta
                
            # Death cross
            if ema_12 < ema_26 and prev['EMA_12'] >= prev['EMA_26']:
                meta["reason"] = "EMA death cross"
                return Signal.SELL_REVERSAL, meta
                
            # Blow-off top
            if (price > bb_upper and 
                rsi > self.config.rsi_overbought and 
                self.fng_value > self.config.fng_greed_threshold):
                meta["reason"] = "Blow-off top detected"
                return Signal.SELL_BLOWOFF, meta
                
        # === ENTRY SIGNALS (only if no position) ===
        if not position.is_open:
            # Smart trend entry
            if (ema_12 > ema_26 and
                self.config.rsi_momentum_low <= rsi <= self.config.rsi_momentum_high and
                self.fng_value < self.config.fng_greed_threshold):
                meta["reason"] = "Trend entry: EMA bullish + RSI healthy + FNG not greedy"
                return Signal.BUY_TREND, meta
                
            # Contrarian entry
            if (price < bb_lower and
                rsi < self.config.rsi_oversold and
                self.fng_value < self.config.fng_fear_threshold):
                meta["reason"] = "Contrarian entry: Oversold + Extreme fear"
                return Signal.BUY_CONTRARIAN, meta
                
        return Signal.NONE, {"reason": "No signal conditions met"}
        
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Calculate position size based on risk per trade.
        
        Risk-based sizing: position_size = (capital * risk%) / stop_distance
        """
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            stop_distance = entry_price * 0.05  # Default 5% if no stop
            
        risk_amount = capital * self.config.risk_per_trade_pct
        max_position = capital * self.config.max_position_pct
        
        position_value = risk_amount / (stop_distance / entry_price)
        position_value = min(position_value, max_position)
        
        quantity = position_value / entry_price
        return quantity
```

### 3. Executor Service

**Purpose**: Execute trades on exchange with proper order management.

```python
# executor_service.py
import asyncio
import ccxt.pro as ccxtpro
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import logging
import uuid

@dataclass
class Order:
    """Order representation."""
    id: str
    client_id: str
    symbol: str
    side: str  # "buy" or "sell"
    type: str  # "market", "limit", "stop_market"
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"  # pending, open, filled, cancelled, failed
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at


class ExecutorService:
    """
    Handles order execution with safety checks.
    
    Key responsibilities:
    - Place orders on exchange
    - Track order lifecycle
    - Handle partial fills
    - Implement idempotency
    """
    
    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        symbol: str = 'BTC/USDT',
        testnet: bool = True,  # START WITH TESTNET!
        db = None  # Database connection for persistence
    ):
        exchange_class = getattr(ccxtpro, exchange_id)
        
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            
        self.symbol = symbol
        self.db = db
        self.pending_orders: dict[str, Order] = {}
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Load markets and validate symbol."""
        await self.exchange.load_markets()
        if self.symbol not in self.exchange.markets:
            raise ValueError(f"Symbol {self.symbol} not found on exchange")
        self.market = self.exchange.markets[self.symbol]
        self.logger.info(f"Executor initialized for {self.symbol}")
        
    async def execute_market_order(
        self,
        side: str,
        quantity: float,
        client_id: Optional[str] = None
    ) -> Order:
        """
        Execute market order with slippage protection.
        
        Args:
            side: "buy" or "sell"
            quantity: Amount in base currency
            client_id: Idempotency key (generate if not provided)
        """
        if client_id is None:
            client_id = f"elite_{uuid.uuid4().hex[:8]}"
            
        # Check for existing order with same client_id (idempotency)
        if client_id in self.pending_orders:
            self.logger.warning(f"Duplicate order {client_id}, returning existing")
            return self.pending_orders[client_id]
            
        # Validate quantity against exchange limits
        quantity = self._adjust_quantity(quantity)
        
        order = Order(
            id="",  # Filled by exchange
            client_id=client_id,
            symbol=self.symbol,
            side=side,
            type="market",
            quantity=quantity
        )
        
        self.pending_orders[client_id] = order
        
        try:
            self.logger.info(f"Placing {side} market order for {quantity} {self.symbol}")
            
            result = await self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=quantity,
                params={'clientOrderId': client_id}
            )
            
            order.id = result['id']
            order.status = result['status']
            order.filled_quantity = result.get('filled', 0)
            order.filled_price = result.get('average', result.get('price', 0))
            order.updated_at = datetime.now(timezone.utc)
            
            self.logger.info(
                f"Order {order.id} {order.status}: "
                f"filled {order.filled_quantity} @ {order.filled_price}"
            )
            
            # Persist to database
            if self.db:
                await self._save_order(order)
                
            return order
            
        except ccxt.InsufficientFunds as e:
            order.status = "failed"
            order.updated_at = datetime.now(timezone.utc)
            self.logger.error(f"Insufficient funds: {e}")
            raise
            
        except ccxt.ExchangeError as e:
            order.status = "failed"
            order.updated_at = datetime.now(timezone.utc)
            self.logger.error(f"Exchange error: {e}")
            raise
            
    async def place_stop_loss(
        self,
        quantity: float,
        stop_price: float,
        client_id: Optional[str] = None
    ) -> Order:
        """
        Place stop-loss order on exchange (as backup to local trailing stop).
        
        Note: Exchange stop-loss is a safety net. Primary stop logic is local.
        """
        if client_id is None:
            client_id = f"elite_sl_{uuid.uuid4().hex[:8]}"
            
        quantity = self._adjust_quantity(quantity)
        stop_price = self._adjust_price(stop_price)
        
        order = Order(
            id="",
            client_id=client_id,
            symbol=self.symbol,
            side="sell",
            type="stop_market",
            quantity=quantity,
            stop_price=stop_price
        )
        
        try:
            # Different exchanges have different stop order syntax
            result = await self.exchange.create_order(
                symbol=self.symbol,
                type='stop_market',
                side='sell',
                amount=quantity,
                params={
                    'stopPrice': stop_price,
                    'clientOrderId': client_id
                }
            )
            
            order.id = result['id']
            order.status = result['status']
            order.updated_at = datetime.now(timezone.utc)
            
            self.logger.info(f"Stop-loss placed: {quantity} @ {stop_price}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place stop-loss: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            await self.exchange.cancel_order(order_id, self.symbol)
            self.logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel {order_id}: {e}")
            return False
            
    async def get_balance(self, asset: str = 'USDT') -> dict:
        """Get available balance."""
        balance = await self.exchange.fetch_balance()
        return {
            'free': balance[asset]['free'],
            'used': balance[asset]['used'],
            'total': balance[asset]['total']
        }
        
    def _adjust_quantity(self, quantity: float) -> float:
        """Adjust quantity to exchange precision."""
        precision = self.market.get('precision', {}).get('amount', 8)
        min_amount = self.market.get('limits', {}).get('amount', {}).get('min', 0)
        
        quantity = round(quantity, precision)
        if quantity < min_amount:
            raise ValueError(f"Quantity {quantity} below minimum {min_amount}")
        return quantity
        
    def _adjust_price(self, price: float) -> float:
        """Adjust price to exchange precision."""
        precision = self.market.get('precision', {}).get('price', 2)
        return round(price, precision)
        
    async def _save_order(self, order: Order):
        """Persist order to database."""
        # Implement based on your DB choice
        pass
        
    async def close(self):
        """Graceful shutdown."""
        await self.exchange.close()
```

### 4. Risk Manager (Watchdog)

**Purpose**: Independent safety layer that can kill trading if things go wrong.

```python
# risk_manager.py
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_usd: float = 10000  # Max position size
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    max_trades_per_day: int = 10
    heartbeat_timeout_seconds: int = 60
    

class RiskManager:
    """
    Independent risk management layer.
    
    Runs as separate process/thread to ensure it can kill trading
    even if main strategy has bugs.
    
    Key responsibilities:
    - Monitor position sizes
    - Track daily P&L
    - Enforce trading limits
    - Kill switch for emergencies
    - Heartbeat monitoring
    """
    
    def __init__(
        self,
        config: RiskConfig,
        executor: 'ExecutorService',
        notifier: 'TelegramNotifier'
    ):
        self.config = config
        self.executor = executor
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)
        
        # State
        self.trading_enabled = True
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.trades_today = 0
        self.last_heartbeat = datetime.now(timezone.utc)
        self.day_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
    async def check_trade(
        self,
        side: str,
        quantity: float,
        price: float
    ) -> tuple[bool, str]:
        """
        Pre-trade risk check.
        
        Returns:
            (allowed, reason)
        """
        if not self.trading_enabled:
            return False, "Trading disabled by kill switch"
            
        # Check daily trade count
        if self.trades_today >= self.config.max_trades_per_day:
            return False, f"Daily trade limit ({self.config.max_trades_per_day}) reached"
            
        # Check position size
        position_value = quantity * price
        if position_value > self.config.max_position_usd:
            return False, f"Position ${position_value:.2f} exceeds max ${self.config.max_position_usd}"
            
        # Check daily loss limit
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl) / self.peak_equity
            if loss_pct >= self.config.max_daily_loss_pct:
                await self.trigger_kill_switch("Daily loss limit reached")
                return False, f"Daily loss {loss_pct:.1%} >= limit {self.config.max_daily_loss_pct:.1%}"
                
        # Check max drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown >= self.config.max_drawdown_pct:
                await self.trigger_kill_switch("Max drawdown reached")
                return False, f"Drawdown {drawdown:.1%} >= limit {self.config.max_drawdown_pct:.1%}"
                
        return True, "OK"
        
    async def record_trade(self, pnl: float):
        """Record completed trade for risk tracking."""
        self.trades_today += 1
        self.daily_pnl += pnl
        
        # Update equity tracking
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            
    async def heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.now(timezone.utc)
        
    async def monitor_heartbeat(self):
        """Background task to check heartbeat."""
        while True:
            await asyncio.sleep(10)
            
            elapsed = (datetime.now(timezone.utc) - self.last_heartbeat).seconds
            if elapsed > self.config.heartbeat_timeout_seconds:
                await self.trigger_kill_switch(
                    f"Heartbeat timeout ({elapsed}s > {self.config.heartbeat_timeout_seconds}s)"
                )
                
    async def trigger_kill_switch(self, reason: str):
        """Emergency stop all trading."""
        self.trading_enabled = False
        self.logger.critical(f"🚨 KILL SWITCH: {reason}")
        
        # Close all positions
        try:
            await self._close_all_positions()
        except Exception as e:
            self.logger.error(f"Failed to close positions: {e}")
            
        # Alert
        await self.notifier.send_alert(
            f"🚨 *KILL SWITCH ACTIVATED*\n"
            f"Reason: {reason}\n"
            f"All positions closed. Trading disabled."
        )
        
    async def _close_all_positions(self):
        """Emergency close all open positions."""
        balance = await self.executor.get_balance('BTC')
        if balance['total'] > 0:
            await self.executor.execute_market_order(
                side='sell',
                quantity=balance['total'],
                client_id=f"emergency_close_{datetime.now().timestamp()}"
            )
            
    async def enable_trading(self):
        """Re-enable trading after manual review."""
        self.trading_enabled = True
        self.logger.info("Trading re-enabled")
        await self.notifier.send_alert("✅ Trading re-enabled by operator")
        
    async def reset_daily_counters(self):
        """Reset daily counters at midnight UTC."""
        while True:
            now = datetime.now(timezone.utc)
            midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            await asyncio.sleep((midnight - now).seconds)
            
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.day_start = midnight
            self.logger.info("Daily counters reset")
```

### 5. Telegram Control Plane

**Purpose**: Bidirectional control and monitoring via Telegram.

```python
# telegram_control.py
import asyncio
from telegram import Update, Bot
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters
)
import logging

class TelegramControl:
    """
    Bidirectional Telegram bot for control and alerts.
    
    Commands:
    /status - Current position, P&L, system health
    /balance - Account balance
    /pause - Pause trading
    /resume - Resume trading
    /kill - Emergency close all positions
    /config - View/update config
    """
    
    def __init__(
        self,
        token: str,
        chat_id: str,
        orchestrator: 'Orchestrator'
    ):
        self.token = token
        self.chat_id = chat_id
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.bot = Bot(token)
        
    async def start(self):
        """Start Telegram bot."""
        app = Application.builder().token(self.token).build()
        
        # Command handlers
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("balance", self.cmd_balance))
        app.add_handler(CommandHandler("pause", self.cmd_pause))
        app.add_handler(CommandHandler("resume", self.cmd_resume))
        app.add_handler(CommandHandler("kill", self.cmd_kill))
        app.add_handler(CommandHandler("help", self.cmd_help))
        
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        
        self.logger.info("Telegram bot started")
        
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current status."""
        if str(update.effective_chat.id) != self.chat_id:
            return  # Ignore unauthorized
            
        status = await self.orchestrator.get_status()
        
        msg = f"""
📊 *BTC Elite Status*

*Position:* {status['position']}
*Entry Price:* ${status['entry_price']:,.2f}
*Current Price:* ${status['current_price']:,.2f}
*Unrealized P&L:* ${status['unrealized_pnl']:,.2f} ({status['unrealized_pnl_pct']:.2f}%)

*Today:*
• Trades: {status['trades_today']}
• P&L: ${status['daily_pnl']:,.2f}

*System:*
• Trading: {'✅ Enabled' if status['trading_enabled'] else '❌ Disabled'}
• Last Signal: {status['last_signal']}
• Uptime: {status['uptime']}
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get account balance."""
        if str(update.effective_chat.id) != self.chat_id:
            return
            
        balance = await self.orchestrator.get_balance()
        
        msg = f"""
💰 *Account Balance*

*USDT:* ${balance['USDT']['total']:,.2f}
  • Free: ${balance['USDT']['free']:,.2f}
  • In Orders: ${balance['USDT']['used']:,.2f}

*BTC:* {balance['BTC']['total']:.8f}
  • Free: {balance['BTC']['free']:.8f}
  • In Orders: {balance['BTC']['used']:.8f}

🇰🇭 *KHR:* ៛{balance['USDT']['total'] * 4050:,.0f}
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause trading."""
        if str(update.effective_chat.id) != self.chat_id:
            return
            
        await self.orchestrator.pause_trading()
        await update.message.reply_text("⏸️ Trading paused. Use /resume to continue.")
        
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume trading."""
        if str(update.effective_chat.id) != self.chat_id:
            return
            
        await self.orchestrator.resume_trading()
        await update.message.reply_text("▶️ Trading resumed.")
        
    async def cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency kill switch."""
        if str(update.effective_chat.id) != self.chat_id:
            return
            
        await update.message.reply_text(
            "⚠️ *KILL SWITCH*\n\n"
            "This will close ALL positions and disable trading.\n"
            "Type `/kill confirm` to proceed.",
            parse_mode='Markdown'
        )
        
        # Check for confirmation
        if context.args and context.args[0] == 'confirm':
            await self.orchestrator.emergency_kill("Manual kill via Telegram")
            await update.message.reply_text("🚨 Kill switch activated. All positions closed.")
            
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help."""
        msg = """
🤖 *BTC Elite Trading Bot*

*Commands:*
/status - Position & P&L
/balance - Account balance
/pause - Pause trading
/resume - Resume trading
/kill - Emergency close all
/help - This message

Bot monitors BTC/USDT using Elite strategy.
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    async def send_alert(self, message: str):
        """Send alert message."""
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode='Markdown'
        )
        
    async def send_trade_notification(
        self,
        side: str,
        quantity: float,
        price: float,
        reason: str
    ):
        """Notify on trade execution."""
        emoji = "🟢" if side == "buy" else "🔴"
        
        msg = f"""
{emoji} *{side.upper()} Executed*

*Quantity:* {quantity:.8f} BTC
*Price:* ${price:,.2f}
*Value:* ${quantity * price:,.2f}

*Reason:* {reason}
"""
        await self.send_alert(msg)
```

---

## Database Schema

```sql
-- PostgreSQL schema for trade persistence

CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_price DECIMAL(20, 8),
    exit_time TIMESTAMPTZ,
    pnl DECIMAL(20, 8),
    pnl_pct DECIMAL(10, 4),
    status VARCHAR(20) DEFAULT 'open',
    signal_type VARCHAR(30),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    exchange_id VARCHAR(100),
    client_id VARCHAR(100) UNIQUE NOT NULL,
    position_id INTEGER REFERENCES positions(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    filled_price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(30) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    ema_12 DECIMAL(20, 8),
    ema_26 DECIMAL(20, 8),
    rsi DECIMAL(10, 4),
    fng INTEGER,
    atr DECIMAL(20, 8),
    metadata JSONB,
    executed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE daily_stats (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    starting_equity DECIMAL(20, 8),
    ending_equity DECIMAL(20, 8),
    pnl DECIMAL(20, 8),
    pnl_pct DECIMAL(10, 4),
    trades_count INTEGER,
    win_count INTEGER,
    loss_count INTEGER,
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_client_id ON orders(client_id);
CREATE INDEX idx_signals_timestamp ON signals(timestamp DESC);
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VPS (Singapore/Tokyo)                            │
│                     Low latency to Asian exchanges                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │   Main Bot Process  │    │   Watchdog Process  │                     │
│  │   (systemd service) │    │   (systemd service) │                     │
│  │                     │    │                     │                     │
│  │  • Data Service     │    │  • Monitors main    │                     │
│  │  • Strategy Engine  │    │  • Can kill main    │                     │
│  │  • Executor         │    │  • Independent DB   │                     │
│  │  • Telegram         │    │    connection       │                     │
│  └──────────┬──────────┘    └──────────┬──────────┘                     │
│             │                          │                                 │
│             ▼                          ▼                                 │
│  ┌───────────────────────────────────────────────────────┐              │
│  │                    PostgreSQL                          │              │
│  │           (positions, orders, signals)                 │              │
│  └───────────────────────────────────────────────────────┘              │
│                                                                          │
│  ┌───────────────────────────────────────────────────────┐              │
│  │                      Redis                             │              │
│  │        (real-time state, pub/sub for events)          │              │
│  └───────────────────────────────────────────────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS / WSS
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Exchanges                                      │
│                                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │  Binance │    │   Bybit  │    │   OKX    │    │ Gate.io  │         │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

External Services:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Telegram   │    │ Alternative  │    │   Grafana    │
│   (control)  │    │   .me (FNG)  │    │  (metrics)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Configuration

```yaml
# config.yaml

exchange:
  id: binance
  testnet: true  # START WITH TESTNET!
  symbol: BTC/USDT
  
strategy:
  ema_fast: 12
  ema_slow: 26
  rsi_period: 14
  rsi_momentum: [50, 70]
  rsi_oversold: 35
  rsi_overbought: 75
  bb_period: 20
  bb_std: 2
  trailing_stop_pct: 0.05
  use_atr_stop: true
  atr_multiplier: 2.0
  
risk:
  max_position_usd: 10000
  max_daily_loss_pct: 0.05
  max_drawdown_pct: 0.15
  max_trades_per_day: 10
  risk_per_trade_pct: 0.02
  max_position_pct: 0.25
  
data:
  timeframe: 1h
  history_length: 100
  fng_update_interval: 3600  # 1 hour
  
database:
  host: localhost
  port: 5432
  name: btc_elite
  
redis:
  host: localhost
  port: 6379
  
telegram:
  enabled: true
  
logging:
  level: INFO
  file: /var/log/btc_elite/bot.log
  
khr_rate: 4050
```

---

## Migration Plan

### Phase 1: Refactor (Week 1-2)

```bash
# Directory structure
btc-elite-trader/
├── src/
│   ├── __init__.py
│   ├── data_service.py
│   ├── strategy_engine.py
│   ├── executor_service.py
│   ├── risk_manager.py
│   ├── telegram_control.py
│   ├── orchestrator.py
│   ├── database.py
│   └── config.py
├── tests/
│   ├── test_strategy.py
│   ├── test_indicators.py
│   └── test_executor.py
├── config/
│   ├── config.yaml
│   └── config.testnet.yaml
├── scripts/
│   ├── run_backtest.py      # Existing btc_trader.py logic
│   ├── run_paper.py         # Paper trading mode
│   └── run_live.py          # Production mode
├── migrations/
│   └── 001_initial.sql
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

**Tasks:**
- [ ] Extract shared module from btc_trader/btc_simulation
- [ ] Add YAML config system
- [ ] Add proper logging with rotation
- [ ] Write unit tests for indicators
- [ ] Set up pytest with coverage

### Phase 2: Exchange Integration (Week 3-4)

**Tasks:**
- [ ] Implement ExecutorService with ccxt
- [ ] Test on Binance testnet
- [ ] Handle order lifecycle
- [ ] Implement position tracking
- [ ] Add PostgreSQL persistence

### Phase 3: Real-Time Data (Week 5)

**Tasks:**
- [ ] Replace CoinGecko with exchange WebSocket
- [ ] Implement DataService
- [ ] Handle reconnection logic
- [ ] Add FNG async fetcher

### Phase 4: Risk & Safety (Week 6)

**Tasks:**
- [ ] Implement RiskManager
- [ ] Add heartbeat monitoring
- [ ] Implement kill switch
- [ ] Add daily loss limits
- [ ] Test failure scenarios

### Phase 5: Control Plane (Week 7-8)

**Tasks:**
- [ ] Implement bidirectional Telegram
- [ ] Add /status, /pause, /kill commands
- [ ] Trade notifications
- [ ] Alert system

### Phase 6: Deployment (Week 9)

**Tasks:**
- [ ] Set up VPS (Vultr Singapore)
- [ ] Docker containerization
- [ ] systemd services
- [ ] Log rotation
- [ ] Monitoring (Prometheus + Grafana)
- [ ] Backup strategy

### Phase 7: Go Live (Week 10+)

**Tasks:**
- [ ] Start with $100 USD
- [ ] Monitor for 1 week
- [ ] Fix bugs
- [ ] Gradually increase position size
- [ ] Document runbook

---

## Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| VPS (4GB RAM, Singapore) | $24 |
| Domain + SSL | ~$1 |
| PostgreSQL (on VPS) | $0 |
| Redis (on VPS) | $0 |
| Monitoring (self-hosted) | $0 |
| Telegram | Free |
| **Total Infrastructure** | **~$25/month** |

| Trading Costs | Rate |
|---------------|------|
| Binance Spot | 0.1% maker/taker |
| Bybit Spot | 0.1% maker/taker |
| Slippage (estimated) | 0.05-0.1% |

---

## Critical Safety Checklist

Before going live, verify:

- [ ] Testnet trading works correctly
- [ ] Paper trading matches expectations
- [ ] Kill switch tested and functional
- [ ] Daily loss limit triggers correctly
- [ ] Heartbeat monitoring alerts on failure
- [ ] All API keys have withdrawal disabled
- [ ] IP whitelist enabled on exchange
- [ ] 2FA enabled on exchange account
- [ ] Telegram alerts working
- [ ] Database backups configured
- [ ] Log rotation working
- [ ] Runbook documented
- [ ] Emergency contacts listed

---

## Quick Start (After Implementation)

```bash
# 1. Clone and setup
git clone <repo>
cd btc-elite-trader
pip install -r requirements.txt

# 2. Configure
cp config/config.example.yaml config/config.yaml
# Edit with your API keys (testnet first!)

# 3. Initialize database
python -m scripts.init_db

# 4. Run backtest (existing logic)
python -m scripts.run_backtest --days 365

# 5. Paper trading
python -m scripts.run_paper

# 6. Live trading (after extensive testing!)
python -m scripts.run_live
```

---

## Final Notes

The jump from simulation to production is significant. Key mindset shifts:

1. **Simulation**: "Let's see what would have happened"
2. **Production**: "A bug is a direct debit from my bank account"

Start small. $100 for the first month. Increase only after proving stability.

The architecture above is designed for a **solo operator**. If scaling to multiple users or strategies, consider:
- Message queues (Kafka/RabbitMQ) for event processing
- Kubernetes for horizontal scaling
- Separate databases per strategy
- More sophisticated position management

But that's future scope. Get the single-strategy system bulletproof first.

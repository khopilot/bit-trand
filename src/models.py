"""
Data Models for BTC Elite Trader

Defines core dataclasses for positions, orders, signals, and configuration.

Author: khopilot
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalType(Enum):
    """Trading signal types."""

    NONE = "none"
    BUY_TREND = "buy_trend"
    BUY_CONTRARIAN = "buy_contrarian"
    SELL_REVERSAL = "sell_reversal"
    SELL_BLOWOFF = "sell_blowoff"
    SELL_TRAILING_STOP = "sell_trailing_stop"
    SELL_PARTIAL = "sell_partial"  # Partial take-profit exit

    # SHORT position signals
    SHORT_TREND = "short_trend"
    SHORT_CONTRARIAN = "short_contrarian"
    COVER_REVERSAL = "cover_reversal"
    COVER_TRAILING_STOP = "cover_trailing_stop"


class OrderStatus(Enum):
    """Order lifecycle status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side (buy/sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class PositionStatus(Enum):
    """Position status."""

    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Signal:
    """Trading signal with metadata."""

    signal_type: SignalType
    price: float
    timestamp: datetime
    confidence: float = 1.0
    reason: str = ""
    indicators: dict = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.signal_type in (SignalType.BUY_TREND, SignalType.BUY_CONTRARIAN)

    @property
    def is_sell(self) -> bool:
        return self.signal_type in (
            SignalType.SELL_REVERSAL,
            SignalType.SELL_BLOWOFF,
            SignalType.SELL_TRAILING_STOP,
            SignalType.SELL_PARTIAL,
        )

    @property
    def is_short(self) -> bool:
        return self.signal_type in (SignalType.SHORT_TREND, SignalType.SHORT_CONTRARIAN)

    @property
    def is_cover(self) -> bool:
        return self.signal_type in (SignalType.COVER_REVERSAL, SignalType.COVER_TRAILING_STOP)


@dataclass
class Order:
    """Order representation with tracking."""

    id: Optional[str] = None
    client_order_id: str = ""
    symbol: str = "BTC/USDT"
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    exchange_response: dict = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)


@dataclass
class Position:
    """Position tracking with P&L calculation."""

    id: Optional[int] = None
    symbol: str = "BTC/USDT"
    side: str = "long"
    entry_price: float = 0.0
    quantity: float = 0.0
    initial_quantity: float = 0.0  # Track original size for partial exits
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    stop_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0  # For SHORT trailing stops
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    signal_type: SignalType = SignalType.NONE
    metadata: dict = field(default_factory=dict)
    tp_executed: dict = field(default_factory=dict)  # Track which TP levels hit {price_mult: True}

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    @property
    def notional_value(self) -> float:
        return self.quantity * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if not self.is_open:
            return 0.0
        if self.side == "short":
            # SHORT: profit when price drops
            return (self.entry_price - current_price) * self.quantity
        return (current_price - self.entry_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "short":
            # SHORT: profit when price drops
            return ((self.entry_price - current_price) / self.entry_price) * 100
        return ((current_price - self.entry_price) / self.entry_price) * 100

    def update_trailing_stop(self, current_price: float, atr: float, multiplier: float, min_pct: float) -> float:
        """Update trailing stop based on highest/lowest price and ATR."""
        if self.side == "short":
            return self._update_short_trailing_stop(current_price, atr, multiplier, min_pct)
        return self._update_long_trailing_stop(current_price, atr, multiplier, min_pct)

    def _update_long_trailing_stop(self, current_price: float, atr: float, multiplier: float, min_pct: float) -> float:
        """Update trailing stop for LONG positions (stop below price)."""
        if current_price > self.highest_price:
            self.highest_price = current_price

        atr_stop = self.highest_price - (atr * multiplier)
        min_stop = self.highest_price * (1 - min_pct)
        self.stop_price = max(atr_stop, min_stop)
        return self.stop_price

    def _update_short_trailing_stop(self, current_price: float, atr: float, multiplier: float, min_pct: float) -> float:
        """Update trailing stop for SHORT positions (stop above price)."""
        # Initialize lowest_price on first update
        if self.lowest_price == 0.0 or current_price < self.lowest_price:
            self.lowest_price = current_price

        # Stop is ABOVE the lowest price for shorts
        atr_stop = self.lowest_price + (atr * multiplier)
        min_stop = self.lowest_price * (1 + min_pct)
        self.stop_price = min(atr_stop, min_stop)  # Use min for shorts (tighter stop)
        return self.stop_price


@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""

    # EMA settings
    ema_fast: int = 12
    ema_slow: int = 26

    # RSI settings (MAXIMUM TRADES MODE)
    rsi_period: int = 14
    rsi_momentum_low: int = 30   # Was 45 - enter much earlier in trends
    rsi_momentum_high: int = 85  # Was 75 - capture strong momentum phases
    rsi_oversold: int = 45       # Was 40 - more contrarian opportunities
    rsi_overbought: int = 80     # Was 75 - stay in longer

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Risk management (MAXIMUM TREND CAPTURE)
    trailing_stop_atr_multiplier: float = 10.0  # Very wide - stay in trends
    min_stop_pct: float = 0.30  # Tolerate 30% drawdown for major moves
    slippage: float = 0.001
    max_position_pct: float = 0.95  # Near 100% - maximize trend capture
    risk_per_trade_pct: float = 0.02  # Double the risk per trade

    # Sentiment thresholds (loosened for more trades)
    fng_greed_threshold: int = 100  # Was 80 - blocked trend entries in bull markets
    fng_fear_threshold: int = 50    # Was 25 - contrarian only triggered 0.25% of days
    fng_default: int = 50

    @classmethod
    def from_config(cls, config: dict) -> "StrategyConfig":
        """Create from config dictionary."""
        strat = config.get("strategy", {})
        fng = config.get("fng", {})

        return cls(
            ema_fast=strat.get("ema_fast", 12),
            ema_slow=strat.get("ema_slow", 26),
            rsi_period=strat.get("rsi_period", 14),
            rsi_momentum_low=strat.get("rsi_momentum_low", 45),
            rsi_momentum_high=strat.get("rsi_momentum_high", 75),
            rsi_oversold=strat.get("rsi_oversold", 40),
            rsi_overbought=strat.get("rsi_overbought", 75),
            bb_period=strat.get("bb_period", 20),
            bb_std=strat.get("bb_std", 2.0),
            trailing_stop_atr_multiplier=strat.get("trailing_stop_atr_multiplier", 5.0),
            min_stop_pct=strat.get("min_stop_pct", 0.15),
            slippage=strat.get("slippage", 0.001),
            max_position_pct=strat.get("max_position_pct", 0.25),
            risk_per_trade_pct=strat.get("risk_per_trade_pct", 0.01),
            fng_greed_threshold=fng.get("greed_threshold", 100),
            fng_fear_threshold=fng.get("fear_threshold", 50),
            fng_default=fng.get("default_value", 50),
        )


@dataclass
class RiskLimits:
    """Risk management limits."""

    max_position_usd: float = 10000.0
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_trades_per_day: int = 10

    # Current state
    daily_pnl: float = 0.0
    daily_trades: int = 0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0

    @classmethod
    def from_config(cls, config: dict) -> "RiskLimits":
        """Create from config dictionary."""
        risk = config.get("risk", {})
        market = config.get("market", {})

        return cls(
            max_position_usd=risk.get("max_position_usd", 10000.0),
            max_daily_loss_pct=risk.get("max_daily_loss_pct", 0.05),
            max_drawdown_pct=risk.get("max_drawdown_pct", 0.15),
            max_trades_per_day=risk.get("max_trades_per_day", 10),
            peak_equity=market.get("initial_capital", 10000.0),
        )


@dataclass
class MarketData:
    """OHLCV market data point."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_list(cls, data: list) -> "MarketData":
        """Create from CCXT OHLCV list format [timestamp, O, H, L, C, V]."""
        return cls(
            timestamp=datetime.utcfromtimestamp(data[0] / 1000),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
        )


@dataclass
class AccountBalance:
    """Account balance information."""

    currency: str = "USDT"
    total: float = 0.0
    free: float = 0.0
    used: float = 0.0

    @property
    def available(self) -> float:
        return self.free


@dataclass
class SystemStatus:
    """System health status."""

    is_running: bool = True
    is_paused: bool = False
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    last_signal: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    error_count: int = 0
    last_error: str = ""
    mode: str = "paper"  # paper, testnet, live

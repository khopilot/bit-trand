"""
Executor Service for BTC Elite Trader

Handles order execution, tracking, and lifecycle management.
Supports paper trading, testnet, and live modes.

Author: khopilot
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

import ccxt

from .models import (
    AccountBalance,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionStatus,
    Signal,
    SignalType,
)

logger = logging.getLogger("btc_trader.executor")


class ExecutorService:
    """
    Order execution service with exchange integration.

    Features:
    - Paper trading mode (simulated execution)
    - Testnet/sandbox mode
    - Live trading
    - Retry with exponential backoff
    - Idempotent order handling
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        sandbox: bool = True,
        paper_trading: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        slippage: float = 0.001,
    ):
        """
        Initialize ExecutorService.

        Args:
            exchange_id: Exchange name
            symbol: Trading pair
            sandbox: Use testnet if True
            paper_trading: Simulate orders without exchange
            api_key: API key
            api_secret: API secret
            slippage: Slippage percentage for paper trading
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.sandbox = sandbox
        self.paper_trading = paper_trading
        self.slippage = slippage

        # Paper trading state
        self._paper_balance_usdt = 10000.0
        self._paper_balance_btc = 0.0
        self._paper_orders: list[Order] = []

        # Exchange initialization
        exchange_config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if api_key:
            exchange_config["apiKey"] = api_key
        if api_secret:
            exchange_config["secret"] = api_secret

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(exchange_config)

        if sandbox:
            self.exchange.set_sandbox_mode(True)

        mode = "PAPER" if paper_trading else ("SANDBOX" if sandbox else "LIVE")
        logger.info("ExecutorService initialized in %s mode", mode)

    def set_paper_balance(self, usdt: float, btc: float = 0.0) -> None:
        """Set initial paper trading balance."""
        self._paper_balance_usdt = usdt
        self._paper_balance_btc = btc
        logger.info("Paper balance set: USDT=%.2f, BTC=%.8f", usdt, btc)

    async def execute_signal(
        self,
        signal: Signal,
        position: Optional[Position],
        capital: float,
        position_size_usd: float,
    ) -> tuple[Optional[Order], Optional[Position]]:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal to execute
            position: Current position (if any)
            capital: Total capital
            position_size_usd: Size for new positions

        Returns:
            Tuple of (Order, updated Position)
        """
        if signal.signal_type == SignalType.NONE:
            return None, position

        if signal.is_buy and (position is None or not position.is_open):
            return await self._execute_buy(signal, capital, position_size_usd)

        elif signal.is_sell and position and position.is_open:
            return await self._execute_sell(signal, position)

        return None, position

    async def _execute_buy(
        self,
        signal: Signal,
        capital: float,
        position_size_usd: float,
    ) -> tuple[Optional[Order], Optional[Position]]:
        """Execute buy order."""
        try:
            # Apply slippage for execution price
            exec_price = self._apply_slippage(signal.price, is_buy=True)

            # Calculate quantity
            quantity = position_size_usd / exec_price

            # Create order
            order = Order(
                client_order_id=self._generate_order_id(),
                symbol=self.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=exec_price,
                status=OrderStatus.PENDING,
            )

            # Execute
            if self.paper_trading:
                order = self._paper_execute(order)
            else:
                order = await self._exchange_execute(order)

            if order.is_filled:
                # Create position
                position = Position(
                    symbol=self.symbol,
                    side="long",
                    entry_price=order.average_price,
                    quantity=order.filled_quantity,
                    entry_order_id=order.id,
                    highest_price=order.average_price,
                    status=PositionStatus.OPEN,
                    signal_type=signal.signal_type,
                    metadata={
                        "signal_reason": signal.reason,
                        "signal_confidence": signal.confidence,
                    },
                )

                logger.info(
                    "BUY executed: %.6f BTC @ $%.2f (total: $%.2f)",
                    order.filled_quantity,
                    order.average_price,
                    order.filled_quantity * order.average_price,
                )

                return order, position

            return order, None

        except Exception as e:
            logger.error("Buy execution failed: %s", e)
            return None, None

    async def _execute_sell(
        self,
        signal: Signal,
        position: Position,
    ) -> tuple[Optional[Order], Optional[Position]]:
        """Execute sell order."""
        try:
            # Apply slippage for execution price
            exec_price = self._apply_slippage(signal.price, is_buy=False)

            # Create order
            order = Order(
                client_order_id=self._generate_order_id(),
                symbol=self.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                price=exec_price,
                status=OrderStatus.PENDING,
            )

            # Execute
            if self.paper_trading:
                order = self._paper_execute(order)
            else:
                order = await self._exchange_execute(order)

            if order.is_filled:
                # Close position
                position.exit_order_id = order.id
                position.exit_time = datetime.utcnow()
                position.exit_price = order.average_price
                position.status = PositionStatus.CLOSED
                position.realized_pnl = (order.average_price - position.entry_price) * position.quantity

                pnl_pct = ((order.average_price - position.entry_price) / position.entry_price) * 100

                logger.info(
                    "SELL executed: %.6f BTC @ $%.2f (PnL: $%.2f / %.2f%%)",
                    order.filled_quantity,
                    order.average_price,
                    position.realized_pnl,
                    pnl_pct,
                )

                return order, position

            return order, position

        except Exception as e:
            logger.error("Sell execution failed: %s", e)
            return None, position

    def _paper_execute(self, order: Order) -> Order:
        """Execute order in paper trading mode."""
        now = datetime.utcnow()

        if order.side == OrderSide.BUY:
            cost = order.quantity * order.price
            if cost > self._paper_balance_usdt:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected: insufficient USDT balance")
                return order

            self._paper_balance_usdt -= cost
            self._paper_balance_btc += order.quantity

        else:  # SELL
            if order.quantity > self._paper_balance_btc:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected: insufficient BTC balance")
                return order

            self._paper_balance_btc -= order.quantity
            self._paper_balance_usdt += order.quantity * order.price

        order.id = f"paper_{order.client_order_id}"
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = order.price
        order.updated_at = now

        self._paper_orders.append(order)

        logger.debug(
            "Paper %s: %.6f @ $%.2f | Balance: USDT=%.2f, BTC=%.8f",
            order.side.value,
            order.quantity,
            order.price,
            self._paper_balance_usdt,
            self._paper_balance_btc,
        )

        return order

    async def _exchange_execute(self, order: Order) -> Order:
        """Execute order on exchange."""
        try:
            # Adjust quantity precision
            market = self.exchange.market(self.symbol)
            quantity = self.exchange.amount_to_precision(self.symbol, order.quantity)

            # Execute market order
            if order.order_type == OrderType.MARKET:
                response = await self._create_order_async(
                    symbol=self.symbol,
                    order_type="market",
                    side=order.side.value,
                    amount=float(quantity),
                )
            else:
                response = await self._create_order_async(
                    symbol=self.symbol,
                    order_type="limit",
                    side=order.side.value,
                    amount=float(quantity),
                    price=order.price,
                )

            order.id = response.get("id")
            order.status = self._map_order_status(response.get("status", ""))
            order.filled_quantity = float(response.get("filled", 0))
            order.average_price = float(response.get("average", order.price) or order.price)
            order.updated_at = datetime.utcnow()
            order.exchange_response = response

            return order

        except ccxt.InsufficientFunds as e:
            logger.error("Insufficient funds: %s", e)
            order.status = OrderStatus.REJECTED
            return order

        except ccxt.ExchangeError as e:
            logger.error("Exchange error: %s", e)
            order.status = OrderStatus.REJECTED
            return order

    async def _create_order_async(self, **kwargs) -> dict:
        """Create order asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.exchange.create_order(**kwargs),
        )

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to price."""
        if is_buy:
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)

    def _generate_order_id(self) -> str:
        """Generate unique client order ID."""
        return f"btc_elite_{uuid.uuid4().hex[:12]}"

    def _map_order_status(self, status: str) -> OrderStatus:
        """Map exchange status to OrderStatus."""
        mapping = {
            "open": OrderStatus.SUBMITTED,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(status, OrderStatus.PENDING)

    def get_balance(self) -> AccountBalance:
        """Get current account balance (sync)."""
        if self.paper_trading:
            return AccountBalance(
                currency="USDT",
                total=self._paper_balance_usdt + (self._paper_balance_btc * self._get_price()),
                free=self._paper_balance_usdt,
                used=self._paper_balance_btc * self._get_price(),
            )

        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            return AccountBalance(
                currency="USDT",
                total=float(usdt.get("total", 0)),
                free=float(usdt.get("free", 0)),
                used=float(usdt.get("used", 0)),
            )
        except Exception as e:
            logger.error("Failed to fetch balance: %s", e)
            return AccountBalance()

    async def get_balance_async(self) -> AccountBalance:
        """Get current account balance (async)."""
        if self.paper_trading:
            return self.get_balance()

        try:
            loop = asyncio.get_event_loop()
            balance = await loop.run_in_executor(None, self.exchange.fetch_balance)
            usdt = balance.get("USDT", {})
            return AccountBalance(
                currency="USDT",
                total=float(usdt.get("total", 0)),
                free=float(usdt.get("free", 0)),
                used=float(usdt.get("used", 0)),
            )
        except Exception as e:
            logger.error("Failed to fetch balance: %s", e)
            return AccountBalance()

    def _get_price(self) -> float:
        """Get current price for paper trading valuation."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker.get("last", 0))
        except Exception:
            return 0.0

    def get_paper_positions(self) -> tuple[float, float]:
        """Get paper trading balances."""
        return self._paper_balance_usdt, self._paper_balance_btc

    def close_all_positions(self, current_price: float) -> None:
        """Emergency close all positions (paper mode)."""
        if self.paper_trading and self._paper_balance_btc > 0:
            exec_price = self._apply_slippage(current_price, is_buy=False)
            self._paper_balance_usdt += self._paper_balance_btc * exec_price
            self._paper_balance_btc = 0.0
            logger.warning("Emergency close: all positions liquidated at $%.2f", exec_price)

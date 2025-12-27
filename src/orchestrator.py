"""
Orchestrator for BTC Elite Trader

Main coordination layer that ties together all services:
- DataService for market data
- StrategyEngine for signal generation
- ExecutorService for order execution
- RiskManager for safety controls
- TelegramControl for notifications
- Database for persistence

Author: khopilot
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

from .data_service import DataService, calculate_indicators
from .database import Database, InMemoryDatabase
from .executor_service import ExecutorService
from .models import Position, RiskLimits, Signal, SignalType, StrategyConfig
from .risk_manager import RiskManager, Watchdog
from .strategy_engine import StrategyEngine
from .telegram_control import TelegramControl

logger = logging.getLogger("btc_trader.orchestrator")


class Orchestrator:
    """
    Main trading orchestrator.

    Coordinates all services and implements the main trading loop.

    Modes:
    - paper: Simulated trading with no exchange interaction
    - testnet: Real exchange API but sandbox/testnet mode
    - live: Production trading with real funds
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        mode: str = "paper",
    ):
        """
        Initialize Orchestrator.

        Args:
            config_path: Path to configuration file
            mode: Trading mode (paper, testnet, live)
        """
        self.mode = mode
        self.config = self._load_config(config_path)
        self._running = False
        self._position: Optional[Position] = None

        # Initialize services
        self._init_services()

        logger.info("Orchestrator initialized in %s mode", mode.upper())

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return self._default_config()

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "strategy": {
                "ema_fast": 12,
                "ema_slow": 26,
                "rsi_period": 14,
                "rsi_momentum_low": 50,
                "rsi_momentum_high": 70,
                "rsi_oversold": 35,
                "rsi_overbought": 75,
                "bb_period": 20,
                "bb_std": 2,
                "trailing_stop_atr_multiplier": 3.0,
                "min_stop_pct": 0.08,
                "slippage": 0.001,
                "max_position_pct": 0.25,
                "risk_per_trade_pct": 0.01,
            },
            "fng": {
                "greed_threshold": 80,
                "fear_threshold": 25,
                "default_value": 50,
            },
            "market": {
                "initial_capital": 10000,
                "usd_khr_rate": 4050,
            },
            "risk": {
                "max_position_usd": 10000,
                "max_daily_loss_pct": 0.05,
                "max_drawdown_pct": 0.15,
                "max_trades_per_day": 10,
            },
        }

    def _init_services(self) -> None:
        """Initialize all services."""
        # Get credentials from environment
        api_key = os.getenv("EXCHANGE_API_KEY")
        api_secret = os.getenv("EXCHANGE_API_SECRET")
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        db_url = os.getenv("DATABASE_URL")

        # Determine sandbox mode
        sandbox = self.mode in ("paper", "testnet")
        paper_trading = self.mode == "paper"

        # Strategy configuration
        self.strategy_config = StrategyConfig.from_config(self.config)

        # Data service
        self.data_service = DataService(
            exchange_id="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            sandbox=sandbox,
            api_key=api_key if not paper_trading else None,
            api_secret=api_secret if not paper_trading else None,
        )

        # Strategy engine
        self.strategy_engine = StrategyEngine(self.strategy_config)

        # Executor service
        self.executor = ExecutorService(
            exchange_id="binance",
            symbol="BTC/USDT",
            sandbox=sandbox,
            paper_trading=paper_trading,
            api_key=api_key,
            api_secret=api_secret,
            slippage=self.strategy_config.slippage,
        )

        # Set initial paper balance
        initial_capital = self.config.get("market", {}).get("initial_capital", 10000)
        self.executor.set_paper_balance(initial_capital)

        # Risk manager
        risk_limits = RiskLimits.from_config(self.config)
        self.risk_manager = RiskManager(risk_limits, initial_equity=initial_capital)

        # Watchdog
        self.watchdog = Watchdog(self.risk_manager, heartbeat_timeout=300)

        # Telegram
        self.telegram = TelegramControl(
            bot_token=bot_token,
            chat_id=chat_id,
        )
        self._setup_telegram_callbacks()

        # Database
        if db_url:
            self.database = Database(db_url)
        else:
            self.database = InMemoryDatabase()

        # Register data callback
        self.data_service.add_candle_callback(self._on_new_candle)

    def _setup_telegram_callbacks(self) -> None:
        """Configure Telegram bot callbacks."""
        self.telegram.set_callbacks(
            status=self._get_status,
            balance=self._get_balance,
            pause=self.risk_manager.pause,
            resume=self.risk_manager.resume,
            kill=lambda: self.risk_manager.kill("Telegram command"),
        )

        self.watchdog.set_alert_callback(
            lambda msg: asyncio.create_task(
                self.telegram.send_alert("Watchdog Alert", msg, "critical")
            )
        )

    def _get_status(self) -> dict:
        """Get current system status."""
        status = {
            "position": None,
            "risk": self.risk_manager.get_status(),
            "mode": self.mode,
            "running": self._running,
        }

        if self._position and self._position.is_open:
            current_price = self.data_service.get_current_price()
            status["position"] = {
                "quantity": self._position.quantity,
                "entry_price": self._position.entry_price,
                "current_price": current_price,
                "unrealized_pnl_pct": self._position.unrealized_pnl_pct(current_price),
            }

        return status

    def _get_balance(self) -> dict:
        """Get current balance."""
        balance = self.executor.get_balance()
        usdt, btc = self.executor.get_paper_positions()

        return {
            "total": balance.total,
            "free": balance.free,
            "used": balance.used,
            "btc": btc,
        }

    async def start(self) -> None:
        """Start the trading system."""
        logger.info("Starting BTC Elite Trader in %s mode...", self.mode.upper())

        self._running = True

        # Connect to database
        await self.database.connect()
        await self.database.run_migrations()

        # Load historical data
        df = await self.data_service.load_history(limit=100)
        if df.empty:
            logger.error("Failed to load historical data")
            return

        # Calculate indicators
        self.df = calculate_indicators(df, self.config)
        logger.info("Loaded %d candles with indicators", len(self.df))

        # Check for existing position
        self._position = await self.database.get_open_position()
        if self._position:
            logger.info("Resumed open position: %s", self._position)

        # Start Telegram bot
        await self.telegram.start()

        # Send startup notification
        await self.telegram.send_alert(
            "Bot Started",
            f"BTC Elite Trader started in {self.mode.upper()} mode\n"
            f"Capital: ${self.config.get('market', {}).get('initial_capital', 10000):,.2f}",
            "info",
        )

        # Start main loop
        await self._main_loop()

    async def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Entering main trading loop...")

        # Use hourly interval for 1h timeframe
        interval = 3600  # 1 hour

        while self._running:
            try:
                # Record heartbeat
                self.watchdog.heartbeat()

                # Check watchdog
                if not self.watchdog.check():
                    logger.warning("Watchdog check failed")

                # Fetch latest data
                df = self.data_service.load_history_sync(limit=100)
                if df.empty:
                    logger.warning("Failed to fetch data, retrying...")
                    await asyncio.sleep(60)
                    continue

                # Calculate indicators
                self.df = calculate_indicators(df, self.config)

                # Process latest candle
                await self._process_candle(self.df)

                # Update risk manager with current equity
                balance = self.executor.get_balance()
                self.risk_manager.update_equity(balance.total)

                # Update daily stats
                unrealized = 0.0
                if self._position and self._position.is_open:
                    current_price = self.data_service.get_current_price()
                    unrealized = self._position.unrealized_pnl(current_price)

                await self.database.update_daily_stats(
                    equity=balance.total,
                    unrealized_pnl=unrealized,
                )

                # Wait for next interval
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error("Error in main loop: %s", e)
                await self.telegram.send_alert("Error", str(e), "error")
                await asyncio.sleep(60)

    def _on_new_candle(self, df: pd.DataFrame) -> None:
        """Callback for new candle (from WebSocket)."""
        # Calculate indicators
        df = calculate_indicators(df, self.config)
        self.df = df

        # Process in async context
        asyncio.create_task(self._process_candle(df))

    async def _process_candle(self, df: pd.DataFrame) -> None:
        """Process a candle and potentially execute trades."""
        if not self.risk_manager.is_trading_allowed:
            logger.debug("Trading not allowed, skipping candle")
            return

        # Get Fear & Greed (default to neutral for now)
        fng_value = self.config.get("fng", {}).get("default_value", 50)

        # Generate signal
        signal = self.strategy_engine.generate_signal(
            df=df,
            position=self._position,
            fng_value=fng_value,
        )

        if signal.signal_type == SignalType.NONE:
            return

        logger.info(
            "Signal generated: %s at $%.2f (%s)",
            signal.signal_type.value,
            signal.price,
            signal.reason,
        )

        # Calculate position size for buys
        position_size_usd = 0.0
        if signal.is_buy:
            balance = self.executor.get_balance()
            atr = df["ATR"].iloc[-1] if "ATR" in df.columns else signal.price * 0.02

            position_size_usd, _, stop_price = self.strategy_engine.calculate_position_size(
                capital=balance.total,
                entry_price=signal.price,
                atr=atr,
            )

        # Check risk limits
        risk_check = self.risk_manager.check_trade(
            signal=signal,
            position_size_usd=position_size_usd,
            current_position=self._position,
        )

        if not risk_check.approved:
            logger.warning("Trade rejected: %s", risk_check.reason)
            await self.database.save_signal(signal, executed=False)
            return

        if risk_check.adjusted_size:
            position_size_usd = risk_check.adjusted_size

        # Execute trade
        balance = self.executor.get_balance()
        order, updated_position = await self.executor.execute_signal(
            signal=signal,
            position=self._position,
            capital=balance.total,
            position_size_usd=position_size_usd,
        )

        if order and order.is_filled:
            # Update position
            self._position = updated_position

            # Save to database
            position_id = await self.database.save_position(self._position) if self._position else None
            await self.database.save_order(order, position_id)
            await self.database.save_signal(signal, executed=True)

            # Record trade in risk manager
            pnl = 0.0
            trade_result = None
            if signal.is_sell and updated_position:
                pnl = updated_position.realized_pnl
                trade_result = "win" if pnl > 0 else "loss"

            self.risk_manager.record_trade(pnl)

            # Update daily stats
            await self.database.update_daily_stats(
                equity=self.executor.get_balance().total,
                realized_pnl=pnl,
                trade_result=trade_result,
            )

            # Send notification
            await self.telegram.send_trade_notification(
                action="BUY" if signal.is_buy else "SELL",
                price=order.average_price,
                quantity=order.filled_quantity,
                pnl=pnl if signal.is_sell else None,
                reason=signal.reason,
            )

            logger.info(
                "Trade executed: %s %.6f BTC @ $%.2f",
                "BUY" if signal.is_buy else "SELL",
                order.filled_quantity,
                order.average_price,
            )

    async def stop(self) -> None:
        """Stop the trading system."""
        logger.info("Stopping BTC Elite Trader...")
        self._running = False

        # Close all positions in paper mode
        if self.mode == "paper" and self._position and self._position.is_open:
            current_price = self.data_service.get_current_price()
            self.executor.close_all_positions(current_price)
            logger.info("Paper positions closed at $%.2f", current_price)

        # Send shutdown notification
        await self.telegram.send_alert(
            "Bot Stopped",
            "BTC Elite Trader has been stopped",
            "info",
        )

        # Stop services
        await self.telegram.stop()
        await self.data_service.stop()
        await self.database.close()

        logger.info("Shutdown complete")

    async def run_backtest(self, days: int = 365) -> dict:
        """
        Run backtest on historical data.

        Args:
            days: Number of days to backtest

        Returns:
            Dictionary with backtest results
        """
        logger.info("Running backtest for %d days...", days)

        # Load historical data
        df = self.data_service.load_history_sync(limit=days)
        if df.empty:
            return {"error": "Failed to load data"}

        df = calculate_indicators(df, self.config)

        # Initialize state
        initial_capital = self.config.get("market", {}).get("initial_capital", 10000)
        self.executor.set_paper_balance(initial_capital)
        self._position = None

        ledger = []
        portfolio_values = []

        # Simulate trading
        for i in range(30, len(df)):  # Skip warmup
            row_df = df.iloc[: i + 1]
            current_price = float(df.iloc[i]["Close"])

            signal = self.strategy_engine.generate_signal(
                df=row_df,
                position=self._position,
                fng_value=50,
            )

            if signal.signal_type != SignalType.NONE:
                atr = float(df.iloc[i].get("ATR", current_price * 0.02))
                balance = self.executor.get_balance()

                if signal.is_buy:
                    position_size, _, _ = self.strategy_engine.calculate_position_size(
                        capital=balance.total,
                        entry_price=signal.price,
                        atr=atr,
                    )
                else:
                    position_size = 0

                order, self._position = await self.executor.execute_signal(
                    signal=signal,
                    position=self._position,
                    capital=balance.total,
                    position_size_usd=position_size,
                )

                if order and order.is_filled:
                    date_str = df.iloc[i]["Date"]
                    action = "BUY" if signal.is_buy else "SELL"
                    ledger.append(f"{date_str}: {action} at ${order.average_price:,.2f}")

            # Track portfolio value
            usdt, btc = self.executor.get_paper_positions()
            portfolio_value = usdt + (btc * current_price)
            portfolio_values.append(portfolio_value)

        # Calculate metrics
        final_value = portfolio_values[-1] if portfolio_values else initial_capital
        roi = ((final_value - initial_capital) / initial_capital) * 100

        # Max drawdown
        peak = initial_capital
        max_dd = 0
        for pv in portfolio_values:
            if pv > peak:
                peak = pv
            dd = (peak - pv) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "roi": roi,
            "max_drawdown": max_dd * 100,
            "total_trades": len(ledger),
            "ledger": ledger[-10:],  # Last 10 trades
        }

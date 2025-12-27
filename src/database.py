"""
Database Layer for BTC Elite Trader

PostgreSQL persistence with asyncpg for positions, orders, and stats.
Includes optional SQLite fallback for development.

Author: khopilot
"""

import asyncio
import json
import logging
from datetime import date, datetime
from typing import Any, Optional

from .models import Order, OrderStatus, Position, PositionStatus, Signal, SignalType

logger = logging.getLogger("btc_trader.database")

# Try to import asyncpg
try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    logger.warning("asyncpg not installed, database features limited")


class Database:
    """
    PostgreSQL database layer for trading data persistence.

    Tables:
    - positions: Open and closed positions
    - orders: Order history
    - signals: Signal audit trail
    - daily_stats: Daily performance metrics
    - system_state: Key-value store for state
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize Database.

        Args:
            connection_string: PostgreSQL connection string
                Example: postgresql://user:pass@localhost:5432/btc_trader
        """
        self.connection_string = connection_string
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> bool:
        """
        Connect to database.

        Returns:
            True if connected successfully
        """
        if not HAS_ASYNCPG:
            logger.warning("asyncpg not available, using in-memory storage")
            return False

        if not self.connection_string:
            logger.warning("No connection string provided")
            return False

        try:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
            )
            logger.info("Database connected")
            return True

        except Exception as e:
            logger.error("Database connection failed: %s", e)
            return False

    async def close(self) -> None:
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            logger.info("Database connection closed")

    async def run_migrations(self, migration_path: str = "migrations/001_initial.sql") -> bool:
        """
        Run database migrations.

        Args:
            migration_path: Path to migration SQL file

        Returns:
            True if successful
        """
        if not self._pool:
            return False

        try:
            with open(migration_path, "r") as f:
                sql = f.read()

            async with self._pool.acquire() as conn:
                await conn.execute(sql)

            logger.info("Migrations applied: %s", migration_path)
            return True

        except FileNotFoundError:
            logger.error("Migration file not found: %s", migration_path)
            return False
        except Exception as e:
            logger.error("Migration failed: %s", e)
            return False

    # Position operations
    async def save_position(self, position: Position) -> Optional[int]:
        """Save position to database."""
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                if position.id:
                    # Update
                    await conn.execute(
                        """
                        UPDATE positions SET
                            stop_price = $1,
                            highest_price = $2,
                            status = $3,
                            exit_time = $4,
                            exit_price = $5,
                            exit_order_id = $6,
                            realized_pnl = $7,
                            metadata = $8
                        WHERE id = $9
                        """,
                        position.stop_price,
                        position.highest_price,
                        position.status.value,
                        position.exit_time,
                        position.exit_price,
                        position.exit_order_id,
                        position.realized_pnl,
                        json.dumps(position.metadata),
                        position.id,
                    )
                    return position.id
                else:
                    # Insert
                    result = await conn.fetchval(
                        """
                        INSERT INTO positions (
                            symbol, side, entry_price, quantity, entry_order_id,
                            stop_price, highest_price, status, entry_time,
                            signal_type, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        RETURNING id
                        """,
                        position.symbol,
                        position.side,
                        position.entry_price,
                        position.quantity,
                        position.entry_order_id,
                        position.stop_price,
                        position.highest_price,
                        position.status.value,
                        position.entry_time,
                        position.signal_type.value,
                        json.dumps(position.metadata),
                    )
                    return result

        except Exception as e:
            logger.error("Failed to save position: %s", e)
            return None

    async def get_open_position(self, symbol: str = "BTC/USDT") -> Optional[Position]:
        """Get current open position."""
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM positions
                    WHERE symbol = $1 AND status = 'open'
                    ORDER BY entry_time DESC LIMIT 1
                    """,
                    symbol,
                )

            if row:
                return self._row_to_position(row)
            return None

        except Exception as e:
            logger.error("Failed to get open position: %s", e)
            return None

    async def get_positions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[Position]:
        """Get positions with optional status filter."""
        if not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                if status:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM positions
                        WHERE status = $1
                        ORDER BY entry_time DESC LIMIT $2
                        """,
                        status,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM positions
                        ORDER BY entry_time DESC LIMIT $1
                        """,
                        limit,
                    )

            return [self._row_to_position(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get positions: %s", e)
            return []

    def _row_to_position(self, row: asyncpg.Record) -> Position:
        """Convert database row to Position."""
        return Position(
            id=row["id"],
            symbol=row["symbol"],
            side=row["side"],
            entry_price=float(row["entry_price"]),
            quantity=float(row["quantity"]),
            entry_order_id=row["entry_order_id"],
            exit_order_id=row["exit_order_id"],
            stop_price=float(row["stop_price"]) if row["stop_price"] else 0.0,
            highest_price=float(row["highest_price"]) if row["highest_price"] else 0.0,
            status=PositionStatus(row["status"]),
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            exit_price=float(row["exit_price"]) if row["exit_price"] else 0.0,
            realized_pnl=float(row["realized_pnl"]) if row["realized_pnl"] else 0.0,
            signal_type=SignalType(row["signal_type"]) if row["signal_type"] else SignalType.NONE,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # Order operations
    async def save_order(self, order: Order, position_id: Optional[int] = None) -> Optional[int]:
        """Save order to database."""
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    """
                    INSERT INTO orders (
                        exchange_order_id, client_order_id, symbol, side, order_type,
                        quantity, price, stop_price, status, filled_quantity,
                        average_price, position_id, exchange_response
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (client_order_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        filled_quantity = EXCLUDED.filled_quantity,
                        average_price = EXCLUDED.average_price,
                        exchange_response = EXCLUDED.exchange_response
                    RETURNING id
                    """,
                    order.id,
                    order.client_order_id,
                    order.symbol,
                    order.side.value,
                    order.order_type.value,
                    order.quantity,
                    order.price,
                    order.stop_price,
                    order.status.value,
                    order.filled_quantity,
                    order.average_price,
                    position_id,
                    json.dumps(order.exchange_response),
                )
                return result

        except Exception as e:
            logger.error("Failed to save order: %s", e)
            return None

    # Signal operations
    async def save_signal(
        self,
        signal: Signal,
        executed: bool = False,
        order_id: Optional[int] = None,
    ) -> Optional[int]:
        """Save signal to database."""
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    """
                    INSERT INTO signals (
                        signal_type, price, confidence, reason, indicators,
                        executed, order_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """,
                    signal.signal_type.value,
                    signal.price,
                    signal.confidence,
                    signal.reason,
                    json.dumps(signal.indicators),
                    executed,
                    order_id,
                )
                return result

        except Exception as e:
            logger.error("Failed to save signal: %s", e)
            return None

    # Daily stats operations
    async def update_daily_stats(
        self,
        equity: float,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        trade_result: Optional[str] = None,
    ) -> None:
        """Update daily statistics."""
        if not self._pool:
            return

        today = date.today()

        try:
            async with self._pool.acquire() as conn:
                # Get or create today's stats
                row = await conn.fetchrow(
                    "SELECT * FROM daily_stats WHERE date = $1",
                    today,
                )

                if row:
                    # Update existing
                    high = max(float(row["high_equity"]), equity)
                    low = min(float(row["low_equity"]), equity)
                    trades = row["total_trades"]
                    wins = row["winning_trades"]
                    losses = row["losing_trades"]

                    if trade_result == "win":
                        trades += 1
                        wins += 1
                    elif trade_result == "loss":
                        trades += 1
                        losses += 1

                    await conn.execute(
                        """
                        UPDATE daily_stats SET
                            ending_equity = $1,
                            high_equity = $2,
                            low_equity = $3,
                            realized_pnl = realized_pnl + $4,
                            unrealized_pnl = $5,
                            total_trades = $6,
                            winning_trades = $7,
                            losing_trades = $8
                        WHERE date = $9
                        """,
                        equity,
                        high,
                        low,
                        realized_pnl,
                        unrealized_pnl,
                        trades,
                        wins,
                        losses,
                        today,
                    )
                else:
                    # Insert new
                    await conn.execute(
                        """
                        INSERT INTO daily_stats (
                            date, starting_equity, ending_equity,
                            high_equity, low_equity,
                            realized_pnl, unrealized_pnl
                        ) VALUES ($1, $2, $2, $2, $2, $3, $4)
                        """,
                        today,
                        equity,
                        realized_pnl,
                        unrealized_pnl,
                    )

        except Exception as e:
            logger.error("Failed to update daily stats: %s", e)

    # System state operations
    async def get_state(self, key: str) -> Any:
        """Get system state value."""
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT value FROM system_state WHERE key = $1",
                    key,
                )
                return json.loads(result) if result else None

        except Exception as e:
            logger.error("Failed to get state: %s", e)
            return None

    async def set_state(self, key: str, value: Any) -> bool:
        """Set system state value."""
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO system_state (key, value)
                    VALUES ($1, $2)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                    """,
                    key,
                    json.dumps(value),
                )
                return True

        except Exception as e:
            logger.error("Failed to set state: %s", e)
            return False


class InMemoryDatabase:
    """
    In-memory database for development/testing.

    Provides same interface as Database but stores data in memory.
    """

    def __init__(self):
        self._positions: list[Position] = []
        self._orders: list[Order] = []
        self._signals: list[Signal] = []
        self._state: dict[str, Any] = {}
        self._next_id = 1

    async def connect(self) -> bool:
        logger.info("Using in-memory database")
        return True

    async def close(self) -> None:
        pass

    async def run_migrations(self, *args) -> bool:
        return True

    async def save_position(self, position: Position) -> Optional[int]:
        if not position.id:
            position.id = self._next_id
            self._next_id += 1
            self._positions.append(position)
        else:
            for i, p in enumerate(self._positions):
                if p.id == position.id:
                    self._positions[i] = position
                    break
        return position.id

    async def get_open_position(self, symbol: str = "BTC/USDT") -> Optional[Position]:
        for p in reversed(self._positions):
            if p.symbol == symbol and p.is_open:
                return p
        return None

    async def get_positions(self, status: Optional[str] = None, limit: int = 100) -> list[Position]:
        positions = self._positions
        if status:
            positions = [p for p in positions if p.status.value == status]
        return list(reversed(positions))[:limit]

    async def save_order(self, order: Order, position_id: Optional[int] = None) -> Optional[int]:
        self._orders.append(order)
        return len(self._orders)

    async def save_signal(self, signal: Signal, executed: bool = False, order_id: Optional[int] = None) -> Optional[int]:
        self._signals.append(signal)
        return len(self._signals)

    async def update_daily_stats(self, *args, **kwargs) -> None:
        pass

    async def get_state(self, key: str) -> Any:
        return self._state.get(key)

    async def set_state(self, key: str, value: Any) -> bool:
        self._state[key] = value
        return True

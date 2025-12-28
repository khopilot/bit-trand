"""
Atomic Executor with Rollback

Executes spot + perpetual orders as an atomic unit.
If the second leg fails, automatically rolls back the first leg.

This prevents leaving unhedged positions that would expose
the strategy to directional risk.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .exchange_client import (
    ExchangeClient,
    OrderResult,
    OrderSide,
    OrderType,
)
from .position_manager import ArbPosition, PositionStatus

logger = logging.getLogger("btc_trader.funding_arb.atomic")


class ExecutionStatus(Enum):
    """Status of atomic execution."""
    SUCCESS = "success"
    SPOT_FAILED = "spot_failed"
    PERP_FAILED = "perp_failed"
    ROLLBACK_SUCCESS = "rollback_success"
    ROLLBACK_FAILED = "rollback_failed"
    PARTIAL_FILL = "partial_fill"


@dataclass
class ExecutionResult:
    """Result of atomic execution attempt."""
    status: ExecutionStatus
    position: Optional[ArbPosition]
    spot_order: Optional[OrderResult]
    perp_order: Optional[OrderResult]
    rollback_order: Optional[OrderResult]
    slippage_cost: float = 0.0
    execution_time_ms: float = 0.0
    error_message: str = ""


@dataclass
class ExecutionConfig:
    """Configuration for atomic execution."""
    max_retries: int = 3
    retry_delay_seconds: float = 0.5
    max_slippage_pct: float = 0.002  # 0.2% max slippage
    rollback_timeout_seconds: float = 5.0
    log_all_orders: bool = True


@dataclass
class ExecutionMetrics:
    """Metrics tracking for executions."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    rollbacks_triggered: int = 0
    rollbacks_successful: int = 0
    rollbacks_failed: int = 0
    total_slippage_cost: float = 0.0
    total_rollback_cost: float = 0.0
    execution_times: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def avg_execution_time_ms(self) -> float:
        if not self.execution_times:
            return 0.0
        return sum(self.execution_times) / len(self.execution_times)


class AtomicExecutor:
    """
    Execute spot + perp orders atomically with rollback on failure.

    Execution Flow:
    1. Execute spot order (buy for long arb, sell for close)
    2. If spot succeeds, execute perp order
    3. If perp fails, IMMEDIATELY rollback spot
    4. Track all slippage and rollback costs

    This ensures we never have unhedged exposure.
    """

    def __init__(
        self,
        client: ExchangeClient,
        config: Optional[ExecutionConfig] = None,
    ):
        self.client = client
        self.config = config or ExecutionConfig()
        self.metrics = ExecutionMetrics()
        self._execution_history: List[ExecutionResult] = []

        logger.info(
            "AtomicExecutor initialized: max_retries=%d, max_slippage=%.2f%%",
            self.config.max_retries,
            self.config.max_slippage_pct * 100,
        )

    def open_position_atomic(
        self,
        size_btc: float,
        current_price: float,
        funding_rate: float,
    ) -> ExecutionResult:
        """
        Open a delta-neutral position atomically.

        Steps:
        1. Buy spot BTC
        2. Short perpetual BTC (same size)
        3. If perp fails, sell spot immediately

        Args:
            size_btc: Position size in BTC
            current_price: Current BTC price for slippage calculation
            funding_rate: Current funding rate (for position record)

        Returns:
            ExecutionResult with status and details
        """
        start_time = time.time()
        self.metrics.total_executions += 1

        logger.info(
            "Opening atomic position: %.6f BTC @ $%.2f",
            size_btc,
            current_price,
        )

        # Step 1: Buy Spot
        spot_order = self._execute_with_retry(
            side=OrderSide.BUY,
            quantity=size_btc,
            is_spot=True,
        )

        if not spot_order or spot_order.status != "FILLED":
            self.metrics.failed_executions += 1
            return ExecutionResult(
                status=ExecutionStatus.SPOT_FAILED,
                position=None,
                spot_order=spot_order,
                perp_order=None,
                rollback_order=None,
                error_message="Spot order failed to fill",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        logger.info(
            "Spot order filled: %.6f BTC @ $%.2f",
            spot_order.filled_qty,
            spot_order.price,
        )

        # Step 2: Short Perpetual
        perp_order = self._execute_with_retry(
            side=OrderSide.SELL,
            quantity=spot_order.filled_qty,  # Match spot fill exactly
            is_spot=False,
        )

        if not perp_order or perp_order.status != "FILLED":
            # ROLLBACK: Sell spot immediately
            logger.warning(
                "Perp order failed! Initiating rollback of spot position..."
            )
            self.metrics.rollbacks_triggered += 1

            rollback_result = self._execute_rollback(spot_order)
            execution_time = (time.time() - start_time) * 1000

            if rollback_result.status == ExecutionStatus.ROLLBACK_SUCCESS:
                self.metrics.rollbacks_successful += 1
                self.metrics.total_rollback_cost += rollback_result.slippage_cost
            else:
                self.metrics.rollbacks_failed += 1
                logger.error("CRITICAL: Rollback failed! Manual intervention required!")

            self.metrics.failed_executions += 1
            return ExecutionResult(
                status=rollback_result.status,
                position=None,
                spot_order=spot_order,
                perp_order=perp_order,
                rollback_order=rollback_result.rollback_order,
                slippage_cost=rollback_result.slippage_cost,
                execution_time_ms=execution_time,
                error_message=f"Perp failed, rollback {'succeeded' if rollback_result.status == ExecutionStatus.ROLLBACK_SUCCESS else 'FAILED'}",
            )

        # Step 3: Success - Create position record
        execution_time = (time.time() - start_time) * 1000
        self.metrics.successful_executions += 1
        self.metrics.execution_times.append(execution_time)

        # Calculate slippage
        spot_slippage = abs(spot_order.price - current_price) / current_price
        perp_slippage = abs(perp_order.price - current_price) / current_price
        total_slippage_cost = (spot_slippage + perp_slippage) * size_btc * current_price
        self.metrics.total_slippage_cost += total_slippage_cost

        position = ArbPosition(
            status=PositionStatus.OPEN,
            spot_quantity=spot_order.filled_qty,
            spot_entry_price=spot_order.price,
            spot_order_id=spot_order.order_id,
            perp_quantity=perp_order.filled_qty,
            perp_entry_price=perp_order.price,
            perp_order_id=perp_order.order_id,
            entry_time=datetime.now(timezone.utc),
            entry_funding_rate=funding_rate,
        )

        logger.info(
            "Atomic position opened: spot=%.6f@$%.2f, perp=%.6f@$%.2f, slippage=$%.2f",
            spot_order.filled_qty,
            spot_order.price,
            perp_order.filled_qty,
            perp_order.price,
            total_slippage_cost,
        )

        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            position=position,
            spot_order=spot_order,
            perp_order=perp_order,
            rollback_order=None,
            slippage_cost=total_slippage_cost,
            execution_time_ms=execution_time,
        )

        self._execution_history.append(result)
        return result

    def close_position_atomic(
        self,
        position: ArbPosition,
        current_price: float,
    ) -> ExecutionResult:
        """
        Close an existing delta-neutral position atomically.

        Steps:
        1. Sell spot BTC
        2. Close perp short (buy to close)
        3. If perp close fails, buy back spot

        Args:
            position: Current position to close
            current_price: Current BTC price

        Returns:
            ExecutionResult with status and details
        """
        start_time = time.time()
        self.metrics.total_executions += 1

        logger.info(
            "Closing atomic position: spot=%.6f, perp=%.6f",
            position.spot_quantity,
            position.perp_quantity,
        )

        # Step 1: Sell Spot
        spot_order = self._execute_with_retry(
            side=OrderSide.SELL,
            quantity=position.spot_quantity,
            is_spot=True,
        )

        if not spot_order or spot_order.status != "FILLED":
            self.metrics.failed_executions += 1
            return ExecutionResult(
                status=ExecutionStatus.SPOT_FAILED,
                position=position,
                spot_order=spot_order,
                perp_order=None,
                rollback_order=None,
                error_message="Spot sell order failed",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 2: Close Perp Short (Buy to close)
        perp_order = self._execute_with_retry(
            side=OrderSide.BUY,
            quantity=position.perp_quantity,
            is_spot=False,
            reduce_only=True,
        )

        if not perp_order or perp_order.status != "FILLED":
            # ROLLBACK: Buy back spot
            logger.warning(
                "Perp close failed! Initiating rollback - buying back spot..."
            )
            self.metrics.rollbacks_triggered += 1

            # Create a fake "spot order" to represent what we need to rollback
            rollback_target = OrderResult(
                order_id="rollback",
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                quantity=spot_order.filled_qty,
                price=spot_order.price,
                filled_qty=spot_order.filled_qty,
                status="FILLED",
                timestamp=datetime.now(timezone.utc),
            )

            # Buy back spot (opposite of the sell we did)
            rollback_order = self._execute_with_retry(
                side=OrderSide.BUY,
                quantity=spot_order.filled_qty,
                is_spot=True,
            )

            execution_time = (time.time() - start_time) * 1000

            if rollback_order and rollback_order.status == "FILLED":
                self.metrics.rollbacks_successful += 1
                slippage_cost = abs(rollback_order.price - spot_order.price) * spot_order.filled_qty
                self.metrics.total_rollback_cost += slippage_cost

                return ExecutionResult(
                    status=ExecutionStatus.ROLLBACK_SUCCESS,
                    position=position,  # Position still open
                    spot_order=spot_order,
                    perp_order=perp_order,
                    rollback_order=rollback_order,
                    slippage_cost=slippage_cost,
                    execution_time_ms=execution_time,
                    error_message="Perp close failed, spot rollback succeeded",
                )
            else:
                self.metrics.rollbacks_failed += 1
                logger.error("CRITICAL: Close rollback failed! Manual intervention required!")
                return ExecutionResult(
                    status=ExecutionStatus.ROLLBACK_FAILED,
                    position=position,
                    spot_order=spot_order,
                    perp_order=perp_order,
                    rollback_order=rollback_order,
                    execution_time_ms=execution_time,
                    error_message="CRITICAL: Perp close failed AND rollback failed",
                )

        # Success - both closed
        execution_time = (time.time() - start_time) * 1000
        self.metrics.successful_executions += 1
        self.metrics.execution_times.append(execution_time)

        # Calculate closing slippage
        spot_slippage = abs(spot_order.price - current_price) / current_price
        perp_slippage = abs(perp_order.price - current_price) / current_price
        total_slippage_cost = (spot_slippage + perp_slippage) * position.spot_quantity * current_price
        self.metrics.total_slippage_cost += total_slippage_cost

        # Update position status
        position.status = PositionStatus.CLOSED
        position.exit_time = datetime.now(timezone.utc)

        logger.info(
            "Atomic position closed: spot@$%.2f, perp@$%.2f, slippage=$%.2f",
            spot_order.price,
            perp_order.price,
            total_slippage_cost,
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            position=position,
            spot_order=spot_order,
            perp_order=perp_order,
            rollback_order=None,
            slippage_cost=total_slippage_cost,
            execution_time_ms=execution_time,
        )

    def _execute_with_retry(
        self,
        side: OrderSide,
        quantity: float,
        is_spot: bool,
        reduce_only: bool = False,
    ) -> Optional[OrderResult]:
        """Execute order with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                if is_spot:
                    order = self.client.place_spot_order(
                        symbol="BTCUSDT",
                        side=side,
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                    )
                else:
                    order = self.client.place_futures_order(
                        symbol="BTCUSDT",
                        side=side,
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                        reduce_only=reduce_only,
                    )

                if order and order.status == "FILLED":
                    return order

                if order:
                    logger.warning(
                        "Order not filled (attempt %d/%d): status=%s",
                        attempt + 1,
                        self.config.max_retries,
                        order.status,
                    )

            except Exception as e:
                last_error = e
                logger.warning(
                    "Order failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay_seconds)

        if last_error:
            logger.error("All order attempts failed: %s", last_error)

        return None

    def _execute_rollback(self, spot_order: OrderResult) -> ExecutionResult:
        """
        Execute rollback of spot order.

        Immediately sells the spot position that was bought.
        """
        logger.info(
            "Executing rollback: selling %.6f BTC",
            spot_order.filled_qty,
        )

        rollback_order = self._execute_with_retry(
            side=OrderSide.SELL,
            quantity=spot_order.filled_qty,
            is_spot=True,
        )

        if rollback_order and rollback_order.status == "FILLED":
            slippage_cost = abs(rollback_order.price - spot_order.price) * spot_order.filled_qty

            logger.info(
                "Rollback successful: sold @ $%.2f (entry was $%.2f), cost=$%.2f",
                rollback_order.price,
                spot_order.price,
                slippage_cost,
            )

            return ExecutionResult(
                status=ExecutionStatus.ROLLBACK_SUCCESS,
                position=None,
                spot_order=spot_order,
                perp_order=None,
                rollback_order=rollback_order,
                slippage_cost=slippage_cost,
            )
        else:
            logger.error(
                "CRITICAL: Rollback FAILED! Unhedged position of %.6f BTC remains!",
                spot_order.filled_qty,
            )

            return ExecutionResult(
                status=ExecutionStatus.ROLLBACK_FAILED,
                position=None,
                spot_order=spot_order,
                perp_order=None,
                rollback_order=rollback_order,
                slippage_cost=0.0,
                error_message="Rollback order failed to fill",
            )

    def get_metrics(self) -> Dict:
        """Get execution metrics summary."""
        return {
            "total_executions": self.metrics.total_executions,
            "successful_executions": self.metrics.successful_executions,
            "failed_executions": self.metrics.failed_executions,
            "success_rate": f"{self.metrics.success_rate:.1%}",
            "rollbacks_triggered": self.metrics.rollbacks_triggered,
            "rollbacks_successful": self.metrics.rollbacks_successful,
            "rollbacks_failed": self.metrics.rollbacks_failed,
            "total_slippage_cost": f"${self.metrics.total_slippage_cost:.2f}",
            "total_rollback_cost": f"${self.metrics.total_rollback_cost:.2f}",
            "avg_execution_time_ms": f"{self.metrics.avg_execution_time_ms:.1f}ms",
        }

    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent execution history."""
        recent = self._execution_history[-limit:]
        return [
            {
                "status": r.status.value,
                "slippage_cost": r.slippage_cost,
                "execution_time_ms": r.execution_time_ms,
                "error": r.error_message,
            }
            for r in recent
        ]

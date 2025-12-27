"""
Performance Tracker for BTC Elite Trader

Real-time performance monitoring and decay detection:
- Expected vs actual slippage tracking
- Rolling win rate and profit factor
- Strategy decay detection
- Automatic alerts and pause triggers

Expert traders constantly monitor for edge decay.

Author: khopilot
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("btc_trader.performance")


@dataclass
class TradeRecord:
    """Record of a single trade for tracking."""
    timestamp: datetime
    side: str  # "buy" or "sell"
    expected_price: float
    actual_price: float
    quantity: float
    slippage_pct: float
    pnl: float  # For closed trades
    pnl_pct: float
    signal_type: str
    regime: str


@dataclass
class PerformanceMetrics:
    """Current performance metrics snapshot."""
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float  # (win_rate * avg_win) - (loss_rate * avg_loss)

    # Slippage
    avg_slippage: float
    max_slippage: float
    slippage_cost_total: float

    # Rolling performance
    rolling_roi: float
    rolling_sharpe: float
    rolling_max_dd: float

    # Decay indicators
    recent_win_rate: float  # Last N trades
    win_rate_trend: str  # "improving", "stable", "declining"
    is_decaying: bool
    decay_severity: str  # "none", "mild", "moderate", "severe"

    # Timestamps
    last_trade: Optional[datetime]
    last_win: Optional[datetime]
    losing_streak: int


@dataclass
class DecayAlert:
    """Alert for strategy decay detection."""
    timestamp: datetime
    severity: str
    metric: str
    current_value: float
    threshold: float
    message: str


class PerformanceTracker:
    """
    Real-time performance monitoring and decay detection.

    Tracks:
    - Win rate trends
    - Slippage vs expected
    - Profit factor decay
    - Consecutive losses
    - Strategy edge decay
    """

    def __init__(
        self,
        rolling_window: int = 20,
        decay_threshold_win_rate: float = 0.10,  # 10% decline triggers alert
        decay_threshold_pf: float = 0.20,  # 20% profit factor decline
        max_losing_streak: int = 5,
        max_slippage_pct: float = 0.005,  # 0.5% max acceptable slippage
        alert_callback: Optional[Callable[[DecayAlert], None]] = None,
    ):
        """
        Initialize PerformanceTracker.

        Args:
            rolling_window: Number of trades for rolling metrics
            decay_threshold_win_rate: Win rate decline that triggers decay alert
            decay_threshold_pf: Profit factor decline that triggers decay alert
            max_losing_streak: Consecutive losses before alert
            max_slippage_pct: Maximum acceptable slippage
            alert_callback: Function to call on alerts
        """
        self.rolling_window = rolling_window
        self.decay_threshold_win_rate = decay_threshold_win_rate
        self.decay_threshold_pf = decay_threshold_pf
        self.max_losing_streak = max_losing_streak
        self.max_slippage_pct = max_slippage_pct
        self.alert_callback = alert_callback

        # Trade history
        self.trades: Deque[TradeRecord] = deque(maxlen=1000)
        self.closed_trades: List[TradeRecord] = []

        # Baseline metrics (set after initial period)
        self.baseline_win_rate: Optional[float] = None
        self.baseline_profit_factor: Optional[float] = None
        self.baseline_expectancy: Optional[float] = None

        # Current state
        self.current_streak: int = 0  # Positive = wins, negative = losses
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0

        # Alerts history
        self.alerts: List[DecayAlert] = []

        logger.info(
            "PerformanceTracker initialized: window=%d, decay_win_rate=%.1f%%, decay_pf=%.1f%%",
            rolling_window,
            decay_threshold_win_rate * 100,
            decay_threshold_pf * 100,
        )

    def record_trade(
        self,
        side: str,
        expected_price: float,
        actual_price: float,
        quantity: float,
        pnl: float = 0.0,
        signal_type: str = "",
        regime: str = "",
    ) -> None:
        """
        Record a trade execution.

        Args:
            side: "buy" or "sell"
            expected_price: Price signal was generated at
            actual_price: Actual execution price
            quantity: Trade quantity
            pnl: Realized P&L (for sells)
            signal_type: Type of signal that triggered trade
            regime: Market regime at time of trade
        """
        # Calculate slippage
        if side == "buy":
            slippage_pct = (actual_price - expected_price) / expected_price
        else:
            slippage_pct = (expected_price - actual_price) / expected_price

        pnl_pct = pnl / (actual_price * quantity) * 100 if quantity > 0 else 0.0

        record = TradeRecord(
            timestamp=datetime.utcnow(),
            side=side,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            slippage_pct=slippage_pct,
            pnl=pnl,
            pnl_pct=pnl_pct,
            signal_type=signal_type,
            regime=regime,
        )

        self.trades.append(record)

        # Track closed trades separately
        if side == "sell" and pnl != 0:
            self.closed_trades.append(record)

            # Update streak
            if pnl > 0:
                self.current_streak = max(1, self.current_streak + 1)
            else:
                self.current_streak = min(-1, self.current_streak - 1)

            # Check for losing streak alert
            if self.current_streak <= -self.max_losing_streak:
                self._trigger_alert(
                    severity="warning",
                    metric="losing_streak",
                    current_value=abs(self.current_streak),
                    threshold=self.max_losing_streak,
                    message=f"Losing streak of {abs(self.current_streak)} trades",
                )

        # Check slippage
        if abs(slippage_pct) > self.max_slippage_pct:
            self._trigger_alert(
                severity="warning",
                metric="slippage",
                current_value=slippage_pct * 100,
                threshold=self.max_slippage_pct * 100,
                message=f"High slippage: {slippage_pct*100:.2f}% on {side}",
            )

        # Check for decay after sufficient trades
        if len(self.closed_trades) >= self.rolling_window * 2:
            self._check_decay()

        logger.debug(
            "Trade recorded: %s %.4f @ $%.2f (slippage: %.3f%%)",
            side, quantity, actual_price, slippage_pct * 100,
        )

    def update_equity(self, equity: float) -> None:
        """Update current equity for drawdown tracking."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def set_baseline(self) -> bool:
        """
        Set baseline metrics from current performance.

        Call this after initial trading period to establish baseline.

        Returns:
            True if baseline was set successfully
        """
        if len(self.closed_trades) < self.rolling_window:
            logger.warning(
                "Not enough trades for baseline: %d < %d",
                len(self.closed_trades),
                self.rolling_window,
            )
            return False

        metrics = self.get_metrics()
        self.baseline_win_rate = metrics.win_rate
        self.baseline_profit_factor = metrics.profit_factor
        self.baseline_expectancy = metrics.expectancy

        logger.info(
            "Baseline set: WR=%.1f%%, PF=%.2f, Exp=%.4f",
            self.baseline_win_rate * 100,
            self.baseline_profit_factor,
            self.baseline_expectancy,
        )

        return True

    def _check_decay(self) -> None:
        """Check for strategy decay."""
        if self.baseline_win_rate is None:
            return

        metrics = self.get_metrics()

        # Win rate decay
        if self.baseline_win_rate > 0:
            wr_decline = (self.baseline_win_rate - metrics.recent_win_rate) / self.baseline_win_rate
            if wr_decline > self.decay_threshold_win_rate:
                severity = "critical" if wr_decline > 0.25 else "warning"
                self._trigger_alert(
                    severity=severity,
                    metric="win_rate",
                    current_value=metrics.recent_win_rate * 100,
                    threshold=self.baseline_win_rate * 100,
                    message=f"Win rate declined {wr_decline*100:.1f}% from baseline",
                )

        # Profit factor decay
        if self.baseline_profit_factor and self.baseline_profit_factor > 0:
            current_pf = metrics.profit_factor
            if current_pf > 0:
                pf_decline = (self.baseline_profit_factor - current_pf) / self.baseline_profit_factor
                if pf_decline > self.decay_threshold_pf:
                    severity = "critical" if pf_decline > 0.40 else "warning"
                    self._trigger_alert(
                        severity=severity,
                        metric="profit_factor",
                        current_value=current_pf,
                        threshold=self.baseline_profit_factor,
                        message=f"Profit factor declined {pf_decline*100:.1f}% from baseline",
                    )

    def _trigger_alert(
        self,
        severity: str,
        metric: str,
        current_value: float,
        threshold: float,
        message: str,
    ) -> None:
        """Trigger a performance alert."""
        alert = DecayAlert(
            timestamp=datetime.utcnow(),
            severity=severity,
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            message=message,
        )

        self.alerts.append(alert)

        logger.warning("PERFORMANCE ALERT [%s]: %s", severity.upper(), message)

        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        closed = self.closed_trades

        if not closed:
            return self._empty_metrics()

        # Basic statistics
        total = len(closed)
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]

        winning = len(wins)
        losing = len(losses)
        win_rate = winning / total if total > 0 else 0.0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0.0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        # Expectancy
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Slippage
        slippages = [t.slippage_pct for t in self.trades]
        avg_slippage = np.mean(slippages) if slippages else 0.0
        max_slippage = max(slippages) if slippages else 0.0
        slippage_cost = sum(t.slippage_pct * t.actual_price * t.quantity for t in self.trades)

        # Rolling metrics (last N trades)
        recent = list(closed)[-self.rolling_window:]
        recent_wins = sum(1 for t in recent if t.pnl > 0)
        recent_win_rate = recent_wins / len(recent) if recent else 0.0

        # Win rate trend
        if len(closed) >= self.rolling_window * 2:
            older = list(closed)[-self.rolling_window*2:-self.rolling_window]
            older_wins = sum(1 for t in older if t.pnl > 0)
            older_wr = older_wins / len(older) if older else 0.0

            if recent_win_rate > older_wr + 0.05:
                wr_trend = "improving"
            elif recent_win_rate < older_wr - 0.05:
                wr_trend = "declining"
            else:
                wr_trend = "stable"
        else:
            wr_trend = "insufficient_data"

        # Decay detection
        is_decaying = False
        decay_severity = "none"

        if self.baseline_win_rate and self.baseline_win_rate > 0:
            wr_decline = (self.baseline_win_rate - recent_win_rate) / self.baseline_win_rate
            if wr_decline > 0.25:
                is_decaying = True
                decay_severity = "severe"
            elif wr_decline > 0.15:
                is_decaying = True
                decay_severity = "moderate"
            elif wr_decline > 0.10:
                is_decaying = True
                decay_severity = "mild"

        # Rolling ROI and Sharpe
        pnls = [t.pnl for t in recent]
        rolling_roi = sum(pnls) / self.current_equity * 100 if self.current_equity > 0 else 0.0

        if len(pnls) > 1 and np.std(pnls) > 0:
            rolling_sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(365 / max(1, len(pnls)))
        else:
            rolling_sharpe = 0.0

        # Max drawdown
        if self.peak_equity > 0:
            rolling_max_dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        else:
            rolling_max_dd = 0.0

        # Losing streak
        losing_streak = abs(self.current_streak) if self.current_streak < 0 else 0

        # Last trade times
        last_trade = closed[-1].timestamp if closed else None
        last_win = next((t.timestamp for t in reversed(closed) if t.pnl > 0), None)

        return PerformanceMetrics(
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_slippage=avg_slippage,
            max_slippage=max_slippage,
            slippage_cost_total=slippage_cost,
            rolling_roi=rolling_roi,
            rolling_sharpe=rolling_sharpe,
            rolling_max_dd=rolling_max_dd,
            recent_win_rate=recent_win_rate,
            win_rate_trend=wr_trend,
            is_decaying=is_decaying,
            decay_severity=decay_severity,
            last_trade=last_trade,
            last_win=last_win,
            losing_streak=losing_streak,
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_slippage=0.0,
            max_slippage=0.0,
            slippage_cost_total=0.0,
            rolling_roi=0.0,
            rolling_sharpe=0.0,
            rolling_max_dd=0.0,
            recent_win_rate=0.0,
            win_rate_trend="insufficient_data",
            is_decaying=False,
            decay_severity="none",
            last_trade=None,
            last_win=None,
            losing_streak=0,
        )

    def get_regime_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by market regime."""
        breakdown = {}

        for trade in self.closed_trades:
            regime = trade.regime or "unknown"

            if regime not in breakdown:
                breakdown[regime] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0,
                }

            breakdown[regime]["trades"] += 1
            if trade.pnl > 0:
                breakdown[regime]["wins"] += 1
            breakdown[regime]["total_pnl"] += trade.pnl

        # Calculate win rates
        for regime in breakdown:
            trades = breakdown[regime]["trades"]
            wins = breakdown[regime]["wins"]
            breakdown[regime]["win_rate"] = wins / trades if trades > 0 else 0.0

        return breakdown

    def format_metrics(self) -> str:
        """Format current metrics for display."""
        m = self.get_metrics()

        decay_icon = "🔴" if m.is_decaying else "🟢"
        trend_icon = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(m.win_rate_trend, "❓")

        lines = [
            "=" * 50,
            "PERFORMANCE METRICS",
            "=" * 50,
            "",
            f"Total Trades: {m.total_trades} ({m.winning_trades}W / {m.losing_trades}L)",
            f"Win Rate: {m.win_rate*100:.1f}% {trend_icon}",
            f"Recent Win Rate: {m.recent_win_rate*100:.1f}%",
            f"Avg Win: ${m.avg_win:,.2f}  |  Avg Loss: ${m.avg_loss:,.2f}",
            f"Profit Factor: {m.profit_factor:.2f}",
            f"Expectancy: ${m.expectancy:,.2f}",
            "",
            f"Rolling ROI: {m.rolling_roi:+.2f}%",
            f"Rolling Sharpe: {m.rolling_sharpe:.2f}",
            f"Max Drawdown: {m.rolling_max_dd:.2f}%",
            "",
            f"Avg Slippage: {m.avg_slippage*100:.3f}%",
            f"Max Slippage: {m.max_slippage*100:.3f}%",
            f"Total Slippage Cost: ${m.slippage_cost_total:,.2f}",
            "",
            f"Strategy Health: {decay_icon} {m.decay_severity.upper()}",
            f"Losing Streak: {m.losing_streak}",
            "",
        ]

        if m.last_trade:
            lines.append(f"Last Trade: {m.last_trade.strftime('%Y-%m-%d %H:%M')}")
        if m.last_win:
            lines.append(f"Last Win: {m.last_win.strftime('%Y-%m-%d %H:%M')}")

        lines.append("=" * 50)

        return "\n".join(lines)

    def should_pause_trading(self) -> Tuple[bool, str]:
        """
        Determine if trading should be paused based on performance.

        Returns:
            Tuple of (should_pause, reason)
        """
        m = self.get_metrics()

        # Severe decay
        if m.decay_severity == "severe":
            return True, "Severe strategy decay detected"

        # Long losing streak
        if m.losing_streak >= self.max_losing_streak:
            return True, f"Losing streak of {m.losing_streak} trades"

        # Very low recent win rate
        if m.total_trades >= self.rolling_window and m.recent_win_rate < 0.25:
            return True, f"Recent win rate too low: {m.recent_win_rate*100:.1f}%"

        # Negative expectancy
        if m.total_trades >= self.rolling_window and m.expectancy < -50:
            return True, f"Negative expectancy: ${m.expectancy:.2f}"

        return False, ""

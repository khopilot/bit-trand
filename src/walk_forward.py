"""
Walk-Forward Validation for BTC Elite Trader

Implements proper out-of-sample testing to detect curve-fitting:
- Rolling window optimization
- Train/test splits
- Monte Carlo validation
- Statistical significance testing

If your strategy works in-sample but fails out-of-sample, it's curve-fitted.

Author: khopilot
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("btc_trader.validation")


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward period."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_roi: float
    test_roi: float
    train_sharpe: float
    test_sharpe: float
    train_max_dd: float
    test_max_dd: float
    train_trades: int
    test_trades: int
    train_win_rate: float
    test_win_rate: float
    is_profitable: bool
    degradation: float  # How much worse is test vs train


@dataclass
class ValidationReport:
    """Complete walk-forward validation report."""
    periods: List[WalkForwardResult]
    overall_oos_roi: float  # Out-of-sample ROI
    overall_is_roi: float   # In-sample ROI
    oos_sharpe: float
    is_sharpe: float
    avg_degradation: float
    consistency_ratio: float  # % of periods where OOS was profitable
    is_robust: bool
    statistical_significance: float  # p-value
    monte_carlo_percentile: float
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "overall_oos_roi": self.overall_oos_roi,
            "overall_is_roi": self.overall_is_roi,
            "oos_sharpe": self.oos_sharpe,
            "is_sharpe": self.is_sharpe,
            "avg_degradation": self.avg_degradation,
            "consistency_ratio": self.consistency_ratio,
            "is_robust": self.is_robust,
            "statistical_significance": self.statistical_significance,
            "monte_carlo_percentile": self.monte_carlo_percentile,
            "num_periods": len(self.periods),
            "recommendations": self.recommendations,
        }


class WalkForwardValidator:
    """
    Walk-forward validation engine.

    Proper backtesting methodology:
    1. Split data into rolling windows
    2. Optimize on train window
    3. Test on unseen test window
    4. Roll forward and repeat
    5. Aggregate out-of-sample results
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        min_train_days: int = 180,
        min_test_days: int = 30,
        num_periods: int = 5,
        monte_carlo_runs: int = 1000,
    ):
        """
        Initialize WalkForwardValidator.

        Args:
            train_ratio: Ratio of data for training (0.7 = 70% train, 30% test)
            min_train_days: Minimum days in training window
            min_test_days: Minimum days in test window
            num_periods: Number of walk-forward periods
            monte_carlo_runs: Number of Monte Carlo simulations
        """
        self.train_ratio = train_ratio
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days
        self.num_periods = num_periods
        self.monte_carlo_runs = monte_carlo_runs

    def create_windows(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create train/test windows for walk-forward analysis.

        Returns:
            List of (train_df, test_df) tuples
        """
        total_days = len(df)
        min_required = self.min_train_days + self.min_test_days

        if total_days < min_required:
            logger.warning(
                "Insufficient data: %d days < %d minimum",
                total_days,
                min_required,
            )
            return []

        windows = []

        # Calculate window sizes
        window_size = total_days // self.num_periods
        train_size = int(window_size * self.train_ratio)
        test_size = window_size - train_size

        # Ensure minimums
        train_size = max(train_size, self.min_train_days)
        test_size = max(test_size, self.min_test_days)

        for i in range(self.num_periods):
            start_idx = i * window_size

            # Expanding window: train on all previous data + current train
            train_end_idx = start_idx + train_size
            test_end_idx = min(train_end_idx + test_size, total_days)

            if test_end_idx > total_days:
                break

            # Use expanding window for training (more data = better)
            train_df = df.iloc[:train_end_idx].copy()
            test_df = df.iloc[train_end_idx:test_end_idx].copy()

            if len(train_df) >= self.min_train_days and len(test_df) >= self.min_test_days:
                windows.append((train_df, test_df))
                logger.debug(
                    "Window %d: Train[0:%d] Test[%d:%d]",
                    i, train_end_idx, train_end_idx, test_end_idx,
                )

        logger.info("Created %d walk-forward windows", len(windows))
        return windows

    def calculate_metrics(
        self,
        portfolio_values: List[float],
        trades: List[dict],
    ) -> dict:
        """Calculate performance metrics from portfolio values."""
        if not portfolio_values or len(portfolio_values) < 2:
            return {
                "roi": 0.0,
                "sharpe": 0.0,
                "max_dd": 0.0,
                "trades": 0,
                "win_rate": 0.0,
            }

        pv = np.array(portfolio_values)

        # ROI
        roi = (pv[-1] - pv[0]) / pv[0] * 100

        # Sharpe Ratio (annualized)
        returns = np.diff(pv) / pv[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365)
        else:
            sharpe = 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(pv)
        drawdown = (peak - pv) / peak
        max_dd = np.max(drawdown) * 100

        # Win Rate
        if trades:
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            win_rate = wins / len(trades) * 100
        else:
            win_rate = 0.0

        return {
            "roi": roi,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "trades": len(trades),
            "win_rate": win_rate,
        }

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], Tuple[List[float], List[dict]]],
    ) -> Tuple[List[float], List[dict]]:
        """
        Run strategy on data.

        Args:
            df: Price data with indicators
            strategy_fn: Function that takes df and returns (portfolio_values, trades)

        Returns:
            Tuple of (portfolio_values, trades)
        """
        return strategy_fn(df)

    def validate(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], Tuple[List[float], List[dict]]],
    ) -> ValidationReport:
        """
        Run complete walk-forward validation.

        Args:
            df: Full price data with indicators
            strategy_fn: Strategy function to validate

        Returns:
            ValidationReport with all results
        """
        logger.info("Starting walk-forward validation...")

        windows = self.create_windows(df)

        if not windows:
            return self._empty_report("Insufficient data for validation")

        results = []

        for i, (train_df, test_df) in enumerate(windows):
            logger.info(
                "Validating period %d/%d: Train=%d days, Test=%d days",
                i + 1, len(windows), len(train_df), len(test_df),
            )

            # Run on training data (in-sample)
            train_pv, train_trades = self.run_strategy(train_df, strategy_fn)
            train_metrics = self.calculate_metrics(train_pv, train_trades)

            # Run on test data (out-of-sample)
            test_pv, test_trades = self.run_strategy(test_df, strategy_fn)
            test_metrics = self.calculate_metrics(test_pv, test_trades)

            # Calculate degradation
            if train_metrics["roi"] != 0:
                degradation = (train_metrics["roi"] - test_metrics["roi"]) / abs(train_metrics["roi"]) * 100
            else:
                degradation = 0.0

            result = WalkForwardResult(
                train_start=str(train_df["Date"].iloc[0]),
                train_end=str(train_df["Date"].iloc[-1]),
                test_start=str(test_df["Date"].iloc[0]),
                test_end=str(test_df["Date"].iloc[-1]),
                train_roi=train_metrics["roi"],
                test_roi=test_metrics["roi"],
                train_sharpe=train_metrics["sharpe"],
                test_sharpe=test_metrics["sharpe"],
                train_max_dd=train_metrics["max_dd"],
                test_max_dd=test_metrics["max_dd"],
                train_trades=train_metrics["trades"],
                test_trades=test_metrics["trades"],
                train_win_rate=train_metrics["win_rate"],
                test_win_rate=test_metrics["win_rate"],
                is_profitable=test_metrics["roi"] > 0,
                degradation=degradation,
            )
            results.append(result)

        # Aggregate results
        report = self._aggregate_results(results, df, strategy_fn)

        return report

    def _aggregate_results(
        self,
        results: List[WalkForwardResult],
        df: pd.DataFrame,
        strategy_fn: Callable,
    ) -> ValidationReport:
        """Aggregate walk-forward results into final report."""

        # Overall metrics
        oos_rois = [r.test_roi for r in results]
        is_rois = [r.train_roi for r in results]
        oos_sharpes = [r.test_sharpe for r in results]
        is_sharpes = [r.train_sharpe for r in results]
        degradations = [r.degradation for r in results]

        overall_oos_roi = np.mean(oos_rois) if oos_rois else 0.0
        overall_is_roi = np.mean(is_rois) if is_rois else 0.0
        oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0
        is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_degradation = np.mean(degradations) if degradations else 0.0

        # Consistency ratio
        profitable_periods = sum(1 for r in results if r.is_profitable)
        consistency_ratio = profitable_periods / len(results) if results else 0.0

        # Monte Carlo simulation for statistical significance
        mc_percentile, p_value = self._monte_carlo_test(df, strategy_fn, overall_oos_roi)

        # Determine if strategy is robust
        is_robust = (
            consistency_ratio >= 0.6 and  # At least 60% profitable periods
            avg_degradation < 50 and       # Less than 50% degradation
            oos_sharpe > 0.5 and           # Positive risk-adjusted returns
            p_value < 0.05                  # Statistically significant
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            results, overall_oos_roi, overall_is_roi, avg_degradation,
            consistency_ratio, oos_sharpe, p_value,
        )

        return ValidationReport(
            periods=results,
            overall_oos_roi=overall_oos_roi,
            overall_is_roi=overall_is_roi,
            oos_sharpe=oos_sharpe,
            is_sharpe=is_sharpe,
            avg_degradation=avg_degradation,
            consistency_ratio=consistency_ratio,
            is_robust=is_robust,
            statistical_significance=p_value,
            monte_carlo_percentile=mc_percentile,
            recommendations=recommendations,
        )

    def _monte_carlo_test(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable,
        actual_roi: float,
    ) -> Tuple[float, float]:
        """
        Monte Carlo simulation to test statistical significance.

        Shuffles returns to create random strategies and compares
        actual performance to random distribution.

        Returns:
            Tuple of (percentile, p_value)
        """
        logger.info("Running Monte Carlo simulation (%d runs)...", self.monte_carlo_runs)

        if len(df) < 50:
            return 50.0, 0.5

        # Calculate actual returns
        returns = df["Close"].pct_change().dropna().values

        random_rois = []

        for i in range(self.monte_carlo_runs):
            # Shuffle returns
            shuffled_returns = np.random.permutation(returns)

            # Reconstruct prices
            shuffled_prices = [df["Close"].iloc[0]]
            for r in shuffled_returns:
                shuffled_prices.append(shuffled_prices[-1] * (1 + r))

            # Calculate random ROI
            random_roi = (shuffled_prices[-1] - shuffled_prices[0]) / shuffled_prices[0] * 100
            random_rois.append(random_roi)

        random_rois = np.array(random_rois)

        # Calculate percentile
        percentile = np.sum(random_rois < actual_roi) / len(random_rois) * 100

        # P-value (two-tailed)
        p_value = 2 * min(percentile, 100 - percentile) / 100

        logger.info(
            "Monte Carlo: Actual ROI=%.2f%%, Percentile=%.1f%%, p-value=%.4f",
            actual_roi, percentile, p_value,
        )

        return percentile, p_value

    def _generate_recommendations(
        self,
        results: List[WalkForwardResult],
        oos_roi: float,
        is_roi: float,
        degradation: float,
        consistency: float,
        sharpe: float,
        p_value: float,
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Curve-fitting detection
        if degradation > 75:
            recommendations.append(
                "⚠️ HIGH CURVE-FITTING RISK: OOS performance degrades >75% from IS. "
                "Simplify strategy parameters."
            )
        elif degradation > 50:
            recommendations.append(
                "⚡ MODERATE CURVE-FITTING: Consider reducing number of parameters "
                "or using more robust indicators."
            )

        # Consistency
        if consistency < 0.4:
            recommendations.append(
                "❌ LOW CONSISTENCY: Strategy profitable in <40% of periods. "
                "May not be robust across market regimes."
            )
        elif consistency < 0.6:
            recommendations.append(
                "⚠️ MODERATE CONSISTENCY: Consider adding regime detection "
                "to avoid unfavorable market conditions."
            )

        # Statistical significance
        if p_value > 0.1:
            recommendations.append(
                "❌ NOT STATISTICALLY SIGNIFICANT (p>0.1): Results could be due to luck. "
                "Need more data or better edge."
            )
        elif p_value > 0.05:
            recommendations.append(
                "⚠️ MARGINAL SIGNIFICANCE (0.05<p<0.1): Results are weak. "
                "Collect more out-of-sample data before live trading."
            )

        # Sharpe ratio
        if sharpe < 0:
            recommendations.append(
                "❌ NEGATIVE SHARPE: Risk-adjusted returns are negative. "
                "Do not trade this strategy."
            )
        elif sharpe < 0.5:
            recommendations.append(
                "⚠️ LOW SHARPE (<0.5): Poor risk-adjusted returns. "
                "Consider improving entry/exit timing."
            )
        elif sharpe > 1.5:
            recommendations.append(
                "✅ EXCELLENT SHARPE (>1.5): Strong risk-adjusted returns."
            )

        # Overall assessment
        if oos_roi > 0 and consistency >= 0.6 and p_value < 0.05:
            recommendations.append(
                "✅ STRATEGY APPEARS ROBUST: Consider paper trading for 30+ days "
                "before committing real capital."
            )
        else:
            recommendations.append(
                "❌ STRATEGY NEEDS IMPROVEMENT: Do not proceed to live trading "
                "until issues are addressed."
            )

        return recommendations

    def _empty_report(self, reason: str) -> ValidationReport:
        """Create empty validation report."""
        return ValidationReport(
            periods=[],
            overall_oos_roi=0.0,
            overall_is_roi=0.0,
            oos_sharpe=0.0,
            is_sharpe=0.0,
            avg_degradation=0.0,
            consistency_ratio=0.0,
            is_robust=False,
            statistical_significance=1.0,
            monte_carlo_percentile=50.0,
            recommendations=[f"❌ Validation failed: {reason}"],
        )

    def format_report(self, report: ValidationReport) -> str:
        """Format validation report for display."""
        lines = [
            "=" * 60,
            "WALK-FORWARD VALIDATION REPORT",
            "=" * 60,
            "",
            f"Periods Tested: {len(report.periods)}",
            f"Consistency Ratio: {report.consistency_ratio:.1%}",
            "",
            "PERFORMANCE COMPARISON:",
            f"  In-Sample ROI:      {report.overall_is_roi:+.2f}%",
            f"  Out-of-Sample ROI:  {report.overall_oos_roi:+.2f}%",
            f"  Degradation:        {report.avg_degradation:.1f}%",
            "",
            f"  In-Sample Sharpe:   {report.is_sharpe:.2f}",
            f"  Out-of-Sample Sharpe: {report.oos_sharpe:.2f}",
            "",
            "STATISTICAL TESTS:",
            f"  Monte Carlo Percentile: {report.monte_carlo_percentile:.1f}%",
            f"  P-Value: {report.statistical_significance:.4f}",
            "",
            f"VERDICT: {'✅ ROBUST' if report.is_robust else '❌ NOT ROBUST'}",
            "",
            "RECOMMENDATIONS:",
        ]

        for rec in report.recommendations:
            lines.append(f"  {rec}")

        lines.append("")
        lines.append("PERIOD DETAILS:")

        for i, period in enumerate(report.periods):
            status = "✅" if period.is_profitable else "❌"
            lines.append(
                f"  {status} Period {i+1}: "
                f"Train {period.train_roi:+.1f}% → Test {period.test_roi:+.1f}% "
                f"(degradation: {period.degradation:.0f}%)"
            )

        lines.append("=" * 60)

        return "\n".join(lines)

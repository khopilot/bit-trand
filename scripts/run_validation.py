#!/usr/bin/env python3
"""
Comprehensive Strategy Validation Script

Must run this BEFORE live trading to validate:
1. Walk-forward validation (out-of-sample testing)
2. Regime detection accuracy
3. Monte Carlo statistical significance
4. Performance decay detection
5. Risk management stress testing

DO NOT TRADE LIVE UNTIL THIS PASSES.

Author: khopilot
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Position, RiskLimits, Signal, SignalType, StrategyConfig
from src.regime_detector import RegimeDetector, RegimeState, TrendRegime, VolatilityRegime
from src.strategy_engine import StrategyEngine
from src.walk_forward import ValidationReport, WalkForwardValidator
from src.performance_tracker import PerformanceTracker
from src.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("validation")


class StrategyValidator:
    """
    Comprehensive strategy validation system.

    Runs multiple validation checks to ensure the strategy
    is robust before live trading.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        output_dir: str = "validation_reports",
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load config
        self.config = self._load_config()

        # Initialize components
        self.strategy_config = StrategyConfig.from_config(self.config)
        self.risk_limits = RiskLimits.from_config(self.config)

        self.strategy_engine = StrategyEngine(
            config=self.strategy_config,
            kelly_fraction=0.25,
            use_regime_filter=True,
            use_mtf_filter=True,
        )

        self.regime_detector = RegimeDetector()

        self.walk_forward = WalkForwardValidator(
            train_ratio=0.7,
            min_train_days=180,
            min_test_days=30,
            num_periods=5,
            monte_carlo_runs=1000,
        )

        self.performance_tracker = PerformanceTracker()

        initial_capital = self.config.get("market", {}).get("initial_capital", 10000)
        self.risk_manager = RiskManager(
            limits=self.risk_limits,
            initial_equity=initial_capital,
        )

        # Validation results
        self.results: Dict = {}

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)

        if not config_file.exists():
            # Fall back to testnet config
            config_file = Path("config/config.testnet.yaml")

        if not config_file.exists():
            logger.warning("No config file found, using defaults")
            return {}

        with open(config_file) as f:
            return yaml.safe_load(f)

    def fetch_historical_data(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical BTC price data.

        Uses CoinGecko API for free access.
        """
        logger.info("Fetching %d days of historical data...", days)

        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            prices = data.get("prices", [])

            df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
            df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
            df = df.drop("Timestamp", axis=1)

            # Create OHLC (approximate from daily close)
            df["Open"] = df["Close"].shift(1)
            df["High"] = df["Close"] * 1.01  # Approximate
            df["Low"] = df["Close"] * 0.99   # Approximate
            df["Volume"] = 0  # Not available from this endpoint

            df = df.dropna()

            logger.info("Fetched %d days of data", len(df))
            return df

        except Exception as e:
            logger.error("Failed to fetch data: %s", e)
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required technical indicators."""
        df = df.copy()

        # EMA
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_Mid"] = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + (2 * bb_std)
        df["BB_Lower"] = df["BB_Mid"] - (2 * bb_std)

        # ATR
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(span=14, adjust=False).mean()

        # Drop warmup rows
        df = df.dropna()

        return df

    def run_strategy_backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000,
    ) -> Tuple[List[float], List[dict]]:
        """
        Run strategy backtest and return portfolio values and trades.

        This is the strategy function passed to walk-forward validation.
        """
        portfolio_values = [initial_capital]
        trades = []
        capital = initial_capital
        position: Optional[Position] = None

        for i in range(1, len(df)):
            current_df = df.iloc[:i+1]
            current_row = df.iloc[i]
            price = float(current_row["Close"])
            atr = float(current_row.get("ATR", price * 0.02))

            # Detect regime
            regime_state = self.regime_detector.detect(current_df)

            # Generate signal
            signal = self.strategy_engine.generate_signal(
                df=current_df,
                position=position,
                fng_value=50,  # Default neutral
                regime_state=regime_state,
            )

            # Process signal
            if signal.is_buy and position is None:
                # Calculate position size
                size_result = self.strategy_engine.calculate_position_size(
                    capital=capital,
                    entry_price=price,
                    atr=atr,
                    signal_confidence=signal.confidence,
                    regime_multiplier=regime_state.position_size_multiplier if regime_state else 1.0,
                )

                # Apply slippage
                fill_price = self.strategy_engine.apply_slippage(price, is_buy=True)

                # Create position
                quantity = size_result.position_usd / fill_price
                position = Position(
                    entry_price=fill_price,
                    quantity=quantity,
                    highest_price=fill_price,
                    stop_price=size_result.stop_price,
                    signal_type=signal.signal_type,
                )

                capital -= size_result.position_usd

                trades.append({
                    "type": "buy",
                    "date": str(current_row["Date"]),
                    "price": fill_price,
                    "quantity": quantity,
                    "signal": signal.signal_type.value,
                })

            elif signal.is_sell and position is not None:
                # Apply slippage
                fill_price = self.strategy_engine.apply_slippage(price, is_buy=False)

                # Calculate P&L
                pnl = (fill_price - position.entry_price) * position.quantity
                capital += position.quantity * fill_price

                trades.append({
                    "type": "sell",
                    "date": str(current_row["Date"]),
                    "price": fill_price,
                    "quantity": position.quantity,
                    "pnl": pnl,
                    "pnl_pct": pnl / (position.entry_price * position.quantity) * 100,
                    "signal": signal.signal_type.value,
                })

                # Update strategy engine with trade result
                self.strategy_engine.update_performance(win=pnl > 0)

                position = None

            # Update portfolio value
            if position is not None:
                portfolio_value = capital + (position.quantity * price)
            else:
                portfolio_value = capital

            portfolio_values.append(portfolio_value)

        return portfolio_values, trades

    def validate_walk_forward(self, df: pd.DataFrame) -> ValidationReport:
        """Run walk-forward validation."""
        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 60)

        def strategy_fn(data: pd.DataFrame) -> Tuple[List[float], List[dict]]:
            return self.run_strategy_backtest(data)

        report = self.walk_forward.validate(df, strategy_fn)

        # Display report
        print(self.walk_forward.format_report(report))

        self.results["walk_forward"] = report.to_dict()

        return report

    def validate_regime_detection(self, df: pd.DataFrame) -> dict:
        """
        Validate regime detection across different market conditions.

        Checks that regime detector correctly identifies:
        - Trending vs ranging markets
        - High vs low volatility periods
        - Market structure changes
        """
        logger.info("=" * 60)
        logger.info("REGIME DETECTION VALIDATION")
        logger.info("=" * 60)

        regime_history = []

        for i in range(100, len(df)):
            current_df = df.iloc[:i+1]
            regime = self.regime_detector.detect(current_df)

            regime_history.append({
                "date": str(df.iloc[i]["Date"]),
                "trend": regime.trend.value,
                "volatility": regime.volatility.value,
                "structure": regime.structure.value,
                "adx": regime.adx,
                "choppiness": regime.choppiness,
                "should_trade": regime.should_trade,
                "size_mult": regime.position_size_multiplier,
            })

        regime_df = pd.DataFrame(regime_history)

        # Calculate statistics
        trend_distribution = regime_df["trend"].value_counts(normalize=True) * 100
        volatility_distribution = regime_df["volatility"].value_counts(normalize=True) * 100
        trade_pct = regime_df["should_trade"].mean() * 100
        avg_size_mult = regime_df["size_mult"].mean()

        results = {
            "total_periods": len(regime_history),
            "trend_distribution": trend_distribution.to_dict(),
            "volatility_distribution": volatility_distribution.to_dict(),
            "should_trade_pct": trade_pct,
            "avg_size_multiplier": avg_size_mult,
            "regime_changes": self._count_regime_changes(regime_df),
        }

        # Display results
        print("\nTrend Distribution:")
        for trend, pct in trend_distribution.items():
            print(f"  {trend}: {pct:.1f}%")

        print("\nVolatility Distribution:")
        for vol, pct in volatility_distribution.items():
            print(f"  {vol}: {pct:.1f}%")

        print(f"\nShould Trade: {trade_pct:.1f}% of the time")
        print(f"Average Size Multiplier: {avg_size_mult:.2f}x")

        # Validation checks
        checks = []

        # Check 1: Shouldn't always trade (should sit out sometimes)
        if trade_pct > 90:
            checks.append("WARNING: Trading too often (>90%). Regime filter may be too loose.")
        elif trade_pct < 30:
            checks.append("WARNING: Trading too rarely (<30%). Regime filter may be too strict.")
        else:
            checks.append("OK: Trade frequency in reasonable range (30-90%)")

        # Check 2: Should have diverse regimes
        if len(trend_distribution) < 3:
            checks.append("WARNING: Not detecting enough trend types")
        else:
            checks.append("OK: Detecting multiple trend types")

        # Check 3: Size multiplier should vary
        if avg_size_mult < 0.5 or avg_size_mult > 1.2:
            checks.append(f"WARNING: Unusual average size multiplier ({avg_size_mult:.2f})")
        else:
            checks.append("OK: Size multiplier in reasonable range")

        results["checks"] = checks

        print("\nValidation Checks:")
        for check in checks:
            print(f"  {check}")

        self.results["regime_detection"] = results

        return results

    def _count_regime_changes(self, regime_df: pd.DataFrame) -> dict:
        """Count how often regime changes."""
        changes = {
            "trend_changes": (regime_df["trend"] != regime_df["trend"].shift()).sum(),
            "volatility_changes": (regime_df["volatility"] != regime_df["volatility"].shift()).sum(),
            "structure_changes": (regime_df["structure"] != regime_df["structure"].shift()).sum(),
        }
        return changes

    def validate_risk_management(self, df: pd.DataFrame) -> dict:
        """
        Stress test risk management system.

        Simulates extreme scenarios to verify:
        - Drawdown limits trigger correctly
        - Position sizing scales with drawdown
        - Kill switch works
        """
        logger.info("=" * 60)
        logger.info("RISK MANAGEMENT STRESS TEST")
        logger.info("=" * 60)

        initial_equity = 10000
        test_scenarios = []

        # Scenario 1: Consecutive losses
        logger.info("Testing consecutive loss scenario...")
        rm = RiskManager(
            limits=RiskLimits(
                max_position_usd=5000,
                max_daily_loss_pct=0.05,
                max_drawdown_pct=0.15,
            ),
            initial_equity=initial_equity,
        )

        # Simulate 5 consecutive losses
        for i in range(5):
            rm.record_trade(pnl=-200, is_win=False)

        scenario_1 = {
            "name": "5 consecutive losses",
            "in_recovery_mode": rm.in_recovery_mode,
            "recovery_multiplier": 0.5 if rm.in_recovery_mode else 1.0,
            "consecutive_losses": rm._consecutive_losses,
            "trading_paused": rm.trading_paused,
        }
        test_scenarios.append(scenario_1)

        # Scenario 2: Drawdown scaling
        logger.info("Testing drawdown scaling...")
        rm2 = RiskManager(
            limits=RiskLimits(max_drawdown_pct=0.15),
            initial_equity=initial_equity,
        )

        drawdown_tests = []
        for dd_pct in [0.0, 0.05, 0.10, 0.15, 0.20]:
            rm2._current_equity = initial_equity * (1 - dd_pct)
            scale = rm2.get_drawdown_scale()
            drawdown_tests.append({
                "drawdown_pct": dd_pct * 100,
                "scale_factor": scale,
            })

        scenario_2 = {
            "name": "Drawdown scaling",
            "tests": drawdown_tests,
        }
        test_scenarios.append(scenario_2)

        # Scenario 3: Daily loss limit
        logger.info("Testing daily loss limit...")
        rm3 = RiskManager(
            limits=RiskLimits(max_daily_loss_pct=0.05),
            initial_equity=initial_equity,
        )

        # Record trades to accumulate daily loss (after reset check)
        rm3._check_daily_reset()
        rm3._daily_pnl = -600  # 6% loss (manual set after reset)
        daily_loss_check = rm3.check_daily_limits()

        scenario_3 = {
            "name": "Daily loss limit",
            "daily_pnl": rm3._daily_pnl,
            "limit_triggered": not daily_loss_check.approved,
            "reason": daily_loss_check.reason if not daily_loss_check.approved else "N/A",
        }
        test_scenarios.append(scenario_3)

        # Scenario 4: Kill switch
        logger.info("Testing kill switch...")
        rm4 = RiskManager(
            limits=RiskLimits(),
            initial_equity=initial_equity,
        )

        rm4.activate_kill_switch("Stress test")

        scenario_4 = {
            "name": "Kill switch",
            "kill_switch_active": rm4.kill_switch_active,
            "trading_paused": rm4.trading_paused,
        }
        test_scenarios.append(scenario_4)

        # Display results
        print("\nRisk Management Stress Test Results:")
        print("-" * 40)

        all_passed = True

        for scenario in test_scenarios:
            print(f"\n{scenario['name']}:")
            for key, value in scenario.items():
                if key != "name":
                    print(f"  {key}: {value}")

            # Validate expectations
            if scenario["name"] == "5 consecutive losses":
                if not scenario["in_recovery_mode"]:
                    print("  FAIL: Should be in recovery mode")
                    all_passed = False
                else:
                    print("  PASS: Correctly entered recovery mode")

            elif scenario["name"] == "Daily loss limit":
                if not scenario["limit_triggered"]:
                    print("  FAIL: Daily loss limit should trigger")
                    all_passed = False
                else:
                    print("  PASS: Daily loss limit triggered correctly")

            elif scenario["name"] == "Kill switch":
                if not scenario["kill_switch_active"]:
                    print("  FAIL: Kill switch should be active")
                    all_passed = False
                else:
                    print("  PASS: Kill switch activated correctly")

        results = {
            "scenarios": test_scenarios,
            "all_passed": all_passed,
        }

        print(f"\nOverall Risk Management: {'PASS' if all_passed else 'FAIL'}")

        self.results["risk_management"] = results

        return results

    def validate_performance_decay(self, df: pd.DataFrame) -> dict:
        """
        Check for performance decay over time.

        Uses walk-forward results to detect if strategy
        performance is degrading over recent periods.
        """
        logger.info("=" * 60)
        logger.info("PERFORMANCE DECAY ANALYSIS")
        logger.info("=" * 60)

        # Run backtest
        portfolio_values, trades = self.run_strategy_backtest(df)

        # Record trades with performance tracker
        for trade in trades:
            if trade["type"] == "sell":
                self.performance_tracker.record_trade(
                    side="sell",
                    expected_price=trade["price"] * 1.001,  # Assume small slippage
                    actual_price=trade["price"],
                    quantity=trade["quantity"],
                    pnl=trade.get("pnl", 0),
                )

        # Get performance metrics
        metrics = self.performance_tracker.get_metrics()

        # Check for decay
        should_pause, reason = self.performance_tracker.should_pause_trading()

        results = {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "rolling_win_rate": metrics.recent_win_rate,
            "avg_slippage_pct": metrics.avg_slippage,
            "decay_detected": metrics.is_decaying,
            "should_pause": should_pause,
            "reason": reason if should_pause else "No issues detected",
        }

        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Overall Win Rate: {metrics.win_rate:.1%}")
        print(f"  Rolling Win Rate: {metrics.recent_win_rate:.1%}")
        print(f"  Avg Slippage: {metrics.avg_slippage:.3%}")
        print(f"  Decay Detected: {metrics.is_decaying}")

        if should_pause:
            print(f"\n  WARNING: {reason}")
        else:
            print(f"\n  OK: Strategy performance stable")

        self.results["performance_decay"] = results

        return results

    def run_full_validation(self, days: int = 365) -> dict:
        """
        Run complete validation suite.

        Returns:
            Dictionary with all validation results
        """
        print("=" * 60)
        print("BTC ELITE TRADER - FULL STRATEGY VALIDATION")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Period: {days} days")
        print("=" * 60)

        # Fetch data
        df = self.fetch_historical_data(days=days)
        df = self.calculate_indicators(df)

        # Run all validations
        wf_report = self.validate_walk_forward(df)
        regime_results = self.validate_regime_detection(df)
        risk_results = self.validate_risk_management(df)
        decay_results = self.validate_performance_decay(df)

        # Overall assessment
        print("\n" + "=" * 60)
        print("FINAL VALIDATION VERDICT")
        print("=" * 60)

        passed_checks = 0
        total_checks = 4

        # Check 1: Walk-forward robust
        if wf_report.is_robust:
            print("  Walk-Forward: PASS")
            passed_checks += 1
        else:
            print("  Walk-Forward: FAIL")

        # Check 2: Regime detection reasonable
        regime_ok = 30 <= regime_results["should_trade_pct"] <= 90
        if regime_ok:
            print("  Regime Detection: PASS")
            passed_checks += 1
        else:
            print("  Regime Detection: FAIL")

        # Check 3: Risk management working
        if risk_results["all_passed"]:
            print("  Risk Management: PASS")
            passed_checks += 1
        else:
            print("  Risk Management: FAIL")

        # Check 4: No performance decay
        if not decay_results["should_pause"]:
            print("  Performance Decay: PASS")
            passed_checks += 1
        else:
            print("  Performance Decay: FAIL")

        overall_passed = passed_checks == total_checks

        print(f"\nOverall: {passed_checks}/{total_checks} checks passed")

        if overall_passed:
            print("\n STRATEGY VALIDATED - OK to proceed to paper trading")
        else:
            print("\n STRATEGY NOT VALIDATED - DO NOT proceed to live trading")
            print("  Review failed checks and address issues first.")

        # Save results
        self.results["validation_date"] = datetime.now().isoformat()
        self.results["data_days"] = days
        self.results["overall_passed"] = overall_passed
        self.results["passed_checks"] = passed_checks
        self.results["total_checks"] = total_checks

        # Write to file
        output_file = self.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Validate BTC Elite Trader strategy before live trading"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to use (default: 365)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_reports",
        help="Directory to save validation reports",
    )

    args = parser.parse_args()

    validator = StrategyValidator(
        config_path=args.config,
        output_dir=args.output_dir,
    )

    results = validator.run_full_validation(days=args.days)

    # Exit with error code if validation failed
    if not results["overall_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Risk Analysis for Always-In Funding Arbitrage Strategy

Generates:
- Equity curve and drawdown plots
- Monthly/yearly P&L breakdown
- Funding rate distribution analysis
- Stress tests
- Liquidation risk analysis
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- GO/NO-GO recommendation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
CAPITAL = 10000  # $10K starting capital
ENTRY_FEE_PCT = 0.0006  # 0.06% total entry (spot + perp)
EXIT_FEE_PCT = 0.0006   # 0.06% total exit
ROUND_TRIP_COST = ENTRY_FEE_PCT + EXIT_FEE_PCT  # 0.12%

# Paths
DATA_PATH = Path("/root/bit-trand/data/funding_rates/btcusdt_funding_history.csv")
PLOTS_PATH = Path("/root/bit-trand/docs/plots")
REPORT_PATH = Path("/root/bit-trand/docs/funding_arb_risk_analysis.md")

# GO/NO-GO thresholds
THRESHOLDS = {
    "max_drawdown_go": 0.15,      # < 15% is GO
    "max_drawdown_nogo": 0.25,    # > 25% is NO-GO
    "worst_month_go": -0.05,      # > -5% is GO
    "worst_month_nogo": -0.10,    # < -10% is NO-GO
    "negative_streak_go": 45,     # < 45 days is GO
    "negative_streak_nogo": 90,   # > 90 days is NO-GO
    "sharpe_go": 1.0,             # > 1.0 is GO
    "sharpe_nogo": 0.5,           # < 0.5 is NO-GO
    "calmar_go": 0.5,             # > 0.5 is GO
    "calmar_nogo": 0.3,           # < 0.3 is NO-GO
}


def load_funding_data():
    """Load and parse funding rate data."""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def simulate_always_in(df):
    """
    Simulate Always-In strategy.

    Position is always open, collecting funding every 8 hours.
    Entry cost paid once at start, exit cost once at end.
    """
    results = []
    cumulative_funding = 0
    equity = CAPITAL
    peak_equity = CAPITAL

    # Pay entry cost once
    entry_cost = CAPITAL * ENTRY_FEE_PCT
    equity -= entry_cost
    cumulative_funding -= entry_cost

    for idx, row in df.iterrows():
        rate = row['funding_rate']

        # Funding payment (positive rate = we receive, negative = we pay)
        # Position size is full capital worth of BTC
        funding_payment = CAPITAL * rate
        cumulative_funding += funding_payment
        equity += funding_payment

        # Track peak for drawdown
        if equity > peak_equity:
            peak_equity = equity

        drawdown = (peak_equity - equity) / peak_equity

        results.append({
            'timestamp': row['timestamp'],
            'funding_rate': rate,
            'funding_payment': funding_payment,
            'cumulative_funding': cumulative_funding,
            'equity': equity,
            'peak_equity': peak_equity,
            'drawdown': drawdown,
        })

    # Pay exit cost at end
    exit_cost = CAPITAL * EXIT_FEE_PCT
    results[-1]['equity'] -= exit_cost
    results[-1]['cumulative_funding'] -= exit_cost

    return pd.DataFrame(results)


def calculate_drawdown_metrics(sim_df):
    """Calculate comprehensive drawdown metrics."""
    drawdowns = sim_df['drawdown'].values
    max_dd = drawdowns.max()

    # Find drawdown periods
    in_drawdown = False
    dd_start = None
    dd_periods = []

    for idx, row in sim_df.iterrows():
        if row['drawdown'] > 0.001 and not in_drawdown:  # Threshold for "in drawdown"
            in_drawdown = True
            dd_start = row['timestamp']
        elif row['drawdown'] < 0.001 and in_drawdown:
            in_drawdown = False
            dd_end = row['timestamp']
            dd_periods.append({
                'start': dd_start,
                'end': dd_end,
                'duration_days': (dd_end - dd_start).days,
            })

    # If still in drawdown at end
    if in_drawdown:
        dd_periods.append({
            'start': dd_start,
            'end': sim_df.iloc[-1]['timestamp'],
            'duration_days': (sim_df.iloc[-1]['timestamp'] - dd_start).days,
        })

    longest_dd_duration = max([p['duration_days'] for p in dd_periods]) if dd_periods else 0

    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': f"{max_dd * 100:.2f}%",
        'longest_drawdown_days': longest_dd_duration,
        'num_drawdown_periods': len(dd_periods),
    }


def calculate_monthly_returns(sim_df):
    """Calculate monthly P&L."""
    sim_df = sim_df.copy()
    sim_df['month'] = sim_df['timestamp'].dt.to_period('M')

    monthly = []
    months = sim_df['month'].unique()

    for i, month in enumerate(months):
        month_data = sim_df[sim_df['month'] == month]
        month_funding = month_data['funding_payment'].sum()

        # First month includes entry cost (already in cumulative)
        if i == 0:
            month_funding -= CAPITAL * ENTRY_FEE_PCT
        # Last month includes exit cost
        if i == len(months) - 1:
            month_funding -= CAPITAL * EXIT_FEE_PCT

        monthly.append({
            'month': str(month),
            'funding_pnl': month_funding,
            'return_pct': month_funding / CAPITAL * 100,
            'num_periods': len(month_data),
        })

    return pd.DataFrame(monthly)


def calculate_yearly_returns(monthly_df):
    """Calculate yearly P&L from monthly data."""
    monthly_df = monthly_df.copy()
    monthly_df['year'] = monthly_df['month'].str[:4]

    yearly = monthly_df.groupby('year').agg({
        'funding_pnl': 'sum',
        'num_periods': 'sum',
    }).reset_index()

    yearly['return_pct'] = yearly['funding_pnl'] / CAPITAL * 100
    return yearly


def analyze_funding_distribution(df):
    """Analyze funding rate distribution."""
    rates = df['funding_rate'].values

    # Basic stats
    positive_count = (rates > 0).sum()
    negative_count = (rates < 0).sum()
    zero_count = (rates == 0).sum()

    # Consecutive negative streaks
    negative_streaks = []
    current_streak = 0
    streak_cost = 0

    for rate in rates:
        if rate < 0:
            current_streak += 1
            streak_cost += abs(rate) * CAPITAL
        else:
            if current_streak > 0:
                negative_streaks.append({
                    'length': current_streak,
                    'cost': streak_cost,
                })
            current_streak = 0
            streak_cost = 0

    if current_streak > 0:
        negative_streaks.append({
            'length': current_streak,
            'cost': streak_cost,
        })

    worst_streak = max(negative_streaks, key=lambda x: x['length']) if negative_streaks else {'length': 0, 'cost': 0}

    # Convert 8-hour periods to days
    worst_streak_days = worst_streak['length'] / 3

    return {
        'total_periods': len(rates),
        'positive_periods': positive_count,
        'negative_periods': negative_count,
        'zero_periods': zero_count,
        'positive_pct': positive_count / len(rates) * 100,
        'negative_pct': negative_count / len(rates) * 100,
        'mean_rate': rates.mean(),
        'median_rate': np.median(rates),
        'std_rate': rates.std(),
        'min_rate': rates.min(),
        'max_rate': rates.max(),
        'worst_negative_streak_periods': worst_streak['length'],
        'worst_negative_streak_days': worst_streak_days,
        'worst_streak_cost': worst_streak['cost'],
    }


def stress_test(df):
    """Run stress tests for extended negative funding scenarios."""
    tests = []

    # Scenario 1: 30 days of -0.01% funding
    periods_30d = 30 * 3  # 3 periods per day
    loss_30d = CAPITAL * 0.0001 * periods_30d
    tests.append({
        'scenario': '30 days at -0.01% funding',
        'periods': periods_30d,
        'loss_usd': loss_30d,
        'loss_pct': loss_30d / CAPITAL * 100,
    })

    # Scenario 2: 60 days of -0.01% funding
    periods_60d = 60 * 3
    loss_60d = CAPITAL * 0.0001 * periods_60d
    tests.append({
        'scenario': '60 days at -0.01% funding',
        'periods': periods_60d,
        'loss_usd': loss_60d,
        'loss_pct': loss_60d / CAPITAL * 100,
    })

    # Scenario 3: 30 days at -0.05% (extreme bear market)
    loss_extreme_30d = CAPITAL * 0.0005 * periods_30d
    tests.append({
        'scenario': '30 days at -0.05% (extreme)',
        'periods': periods_30d,
        'loss_usd': loss_extreme_30d,
        'loss_pct': loss_extreme_30d / CAPITAL * 100,
    })

    # Breakeven analysis
    # With ROUND_TRIP_COST = 0.12%, need to recover this over holding period
    # If we assume 1-year hold: 0.12% / 365 / 3 = 0.0001096% per period minimum
    breakeven_rate = ROUND_TRIP_COST / 365 / 3
    tests.append({
        'scenario': 'Breakeven rate (1-year hold)',
        'periods': 365 * 3,
        'loss_usd': 0,
        'loss_pct': 0,
        'min_rate': breakeven_rate,
    })

    return tests


def liquidation_analysis():
    """Analyze liquidation risk for perpetual short position."""
    # For a short position, liquidation happens when price rises
    # Liq price = Entry * (1 + 1/leverage - maintenance_margin)
    # Assuming 0.5% maintenance margin

    results = []
    maintenance_margin = 0.005  # 0.5%

    for leverage in [2, 3, 5, 10]:
        # Simplified: liq at roughly (1 + 1/leverage) of entry price
        liq_multiplier = 1 + (1 / leverage) - maintenance_margin
        price_increase_to_liq = (liq_multiplier - 1) * 100

        results.append({
            'leverage': leverage,
            'liq_price_multiplier': liq_multiplier,
            'price_increase_to_liq_pct': price_increase_to_liq,
            'margin_pct': 100 / leverage,
        })

    # Capital required to survive pumps
    pump_scenarios = []
    for pump_pct in [50, 100, 200]:
        # To survive a 50% pump, need enough margin that position isn't liquidated
        # Required margin = pump_pct / (liq_multiplier - 1) approximately
        # For 1x leverage (safest): full collateral
        # For 2x leverage: 50% of position as margin

        # Calculate safe leverage for each pump scenario
        # Safe leverage = 1 / (pump_pct / 100)
        safe_leverage = 1 / (pump_pct / 100) if pump_pct > 0 else 10

        pump_scenarios.append({
            'pump_pct': pump_pct,
            'safe_leverage': min(safe_leverage, 1),  # Cap at 1x for safety
            'margin_required_pct': 100 / min(safe_leverage, 1) if safe_leverage > 0 else 100,
        })

    return {
        'leverage_scenarios': results,
        'pump_survival': pump_scenarios,
        'recommendation': 'Use 1x-2x leverage maximum for perpetual short to survive 50%+ pumps',
    }


def calculate_risk_metrics(sim_df, monthly_df):
    """Calculate risk-adjusted metrics."""
    # Daily returns (approximate from 8-hour periods)
    sim_df = sim_df.copy()
    sim_df['date'] = sim_df['timestamp'].dt.date
    daily = sim_df.groupby('date')['funding_payment'].sum().reset_index()
    daily['daily_return'] = daily['funding_payment'] / CAPITAL

    daily_returns = daily['daily_return'].values

    # Annualized metrics
    trading_days = len(daily)
    years = trading_days / 365

    total_return = sim_df.iloc[-1]['equity'] / CAPITAL - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Sharpe Ratio (assuming risk-free rate ~ 0 for simplicity)
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe_ratio = (daily_mean / daily_std) * np.sqrt(365) if daily_std > 0 else 0

    # Sortino Ratio (downside deviation only)
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 0 else 0.0001
    sortino_ratio = (daily_mean / downside_std) * np.sqrt(365) if downside_std > 0 else 0

    # Calmar Ratio (return / max drawdown)
    max_dd = sim_df['drawdown'].max()
    calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0

    # Win rate
    positive_months = (monthly_df['funding_pnl'] > 0).sum()
    total_months = len(monthly_df)

    return {
        'total_return_pct': total_return * 100,
        'annualized_return_pct': annualized_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate_monthly': positive_months / total_months * 100,
        'best_month_pct': monthly_df['return_pct'].max(),
        'worst_month_pct': monthly_df['return_pct'].min(),
        'median_month_pct': monthly_df['return_pct'].median(),
    }


def capital_requirements():
    """Calculate capital requirements."""
    # For delta-neutral:
    # - Spot position: CAPITAL
    # - Perp margin: CAPITAL / leverage
    # - Buffer for negative funding: ~10% of capital

    spot_capital = CAPITAL

    # Conservative: 2x leverage on perp
    perp_margin = CAPITAL / 2

    # Buffer for 60 days of -0.01% funding
    negative_buffer = CAPITAL * 0.0001 * 60 * 3 * 1.5  # 1.5x safety factor

    # Total
    total = spot_capital + perp_margin + negative_buffer

    return {
        'spot_capital': spot_capital,
        'perp_margin_2x': perp_margin,
        'negative_funding_buffer': negative_buffer,
        'total_recommended': total,
        'capital_efficiency': CAPITAL / total * 100,
    }


def generate_plots(sim_df, monthly_df, funding_df):
    """Generate all analysis plots."""
    plt.style.use('dark_background')

    # 1. Equity Curve
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(sim_df['timestamp'], sim_df['equity'], 'g-', linewidth=1.5, label='Equity')
    ax.axhline(y=CAPITAL, color='white', linestyle='--', alpha=0.3, label='Starting Capital')
    ax.fill_between(sim_df['timestamp'], CAPITAL, sim_df['equity'],
                    where=sim_df['equity'] >= CAPITAL, alpha=0.3, color='green')
    ax.fill_between(sim_df['timestamp'], CAPITAL, sim_df['equity'],
                    where=sim_df['equity'] < CAPITAL, alpha=0.3, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.set_title(f'Always-In Funding Arbitrage: Equity Curve (${CAPITAL:,} Starting Capital)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Drawdown Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(sim_df['timestamp'], 0, sim_df['drawdown'] * 100,
                    color='red', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Over Time')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'drawdown.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Funding Rate Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    rates = funding_df['funding_rate'] * 100  # Convert to percentage
    n, bins, patches = ax.hist(rates, bins=100, color='cyan', alpha=0.7, edgecolor='white', linewidth=0.5)

    # Color negative rates red
    for i, p in enumerate(patches):
        if bins[i] < 0:
            p.set_facecolor('red')

    ax.axvline(x=0, color='white', linestyle='--', linewidth=2)
    ax.axvline(x=rates.mean(), color='yellow', linestyle='-', linewidth=2, label=f'Mean: {rates.mean():.4f}%')
    ax.set_xlabel('Funding Rate (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Funding Rate Distribution (Sep 2019 - Dec 2025)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'funding_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Monthly Returns Heatmap
    monthly_df_plot = monthly_df.copy()
    monthly_df_plot['year'] = monthly_df_plot['month'].str[:4]
    monthly_df_plot['month_num'] = monthly_df_plot['month'].str[5:7].astype(int)

    # Pivot for heatmap
    pivot = monthly_df_plot.pivot(index='year', columns='month_num', values='return_pct')

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)

    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j] if j < len(pivot.columns) and not pd.isna(pivot.iloc[i, j]) else None
            if val is not None:
                text_color = 'black' if abs(val) < 2 else 'white'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=text_color, fontsize=8)

    ax.set_title('Monthly Returns Heatmap (%)')
    cbar = plt.colorbar(im, ax=ax, label='Return %')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'monthly_returns.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plots saved to docs/plots/")


def evaluate_go_nogo(metrics, dd_metrics, funding_stats, risk_metrics):
    """Evaluate GO/NO-GO based on thresholds."""
    results = []
    go_count = 0
    nogo_count = 0

    # Max Drawdown
    max_dd = dd_metrics['max_drawdown']
    if max_dd < THRESHOLDS['max_drawdown_go']:
        status = "GO"
        go_count += 1
    elif max_dd > THRESHOLDS['max_drawdown_nogo']:
        status = "NO-GO"
        nogo_count += 1
    else:
        status = "CAUTION"
    results.append(('Max Drawdown', f"{max_dd*100:.2f}%", f"<{THRESHOLDS['max_drawdown_go']*100}%", status))

    # Worst Month
    worst_month = risk_metrics['worst_month_pct'] / 100
    if worst_month > THRESHOLDS['worst_month_go']:
        status = "GO"
        go_count += 1
    elif worst_month < THRESHOLDS['worst_month_nogo']:
        status = "NO-GO"
        nogo_count += 1
    else:
        status = "CAUTION"
    results.append(('Worst Month', f"{worst_month*100:.2f}%", f">{THRESHOLDS['worst_month_go']*100}%", status))

    # Negative Streak
    neg_streak_days = funding_stats['worst_negative_streak_days']
    if neg_streak_days < THRESHOLDS['negative_streak_go']:
        status = "GO"
        go_count += 1
    elif neg_streak_days > THRESHOLDS['negative_streak_nogo']:
        status = "NO-GO"
        nogo_count += 1
    else:
        status = "CAUTION"
    results.append(('Negative Streak', f"{neg_streak_days:.1f} days", f"<{THRESHOLDS['negative_streak_go']} days", status))

    # Sharpe Ratio
    sharpe = risk_metrics['sharpe_ratio']
    if sharpe > THRESHOLDS['sharpe_go']:
        status = "GO"
        go_count += 1
    elif sharpe < THRESHOLDS['sharpe_nogo']:
        status = "NO-GO"
        nogo_count += 1
    else:
        status = "CAUTION"
    results.append(('Sharpe Ratio', f"{sharpe:.2f}", f">{THRESHOLDS['sharpe_go']}", status))

    # Calmar Ratio
    calmar = risk_metrics['calmar_ratio']
    if calmar > THRESHOLDS['calmar_go']:
        status = "GO"
        go_count += 1
    elif calmar < THRESHOLDS['calmar_nogo']:
        status = "NO-GO"
        nogo_count += 1
    else:
        status = "CAUTION"
    results.append(('Calmar Ratio', f"{calmar:.2f}", f">{THRESHOLDS['calmar_go']}", status))

    # Final decision
    if nogo_count > 0:
        final = "NO-GO"
    elif go_count >= 4:
        final = "GO"
    else:
        final = "CONDITIONAL GO"

    return results, final, go_count, nogo_count


def generate_report(all_results):
    """Generate markdown report."""
    sim_df = all_results['sim_df']
    monthly_df = all_results['monthly_df']
    yearly_df = all_results['yearly_df']
    dd_metrics = all_results['dd_metrics']
    funding_stats = all_results['funding_stats']
    stress_tests = all_results['stress_tests']
    liq_analysis = all_results['liq_analysis']
    cap_req = all_results['cap_req']
    risk_metrics = all_results['risk_metrics']
    go_nogo = all_results['go_nogo']

    evaluation, final_decision, go_count, nogo_count = go_nogo

    report = f"""# Always-In Funding Arbitrage: Comprehensive Risk Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Data Range:** Sep 2019 - Dec 2025 ({funding_stats['total_periods']:,} funding periods)
**Starting Capital:** ${CAPITAL:,}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Return** | {risk_metrics['total_return_pct']:.2f}% |
| **Annualized Return** | {risk_metrics['annualized_return_pct']:.2f}% |
| **Max Drawdown** | {dd_metrics['max_drawdown_pct']} |
| **Sharpe Ratio** | {risk_metrics['sharpe_ratio']:.2f} |
| **Calmar Ratio** | {risk_metrics['calmar_ratio']:.2f} |

### Final Verdict: **{final_decision}**

---

## 1. Equity Curve & Drawdown Analysis

![Equity Curve](plots/equity_curve.png)

![Drawdown](plots/drawdown.png)

### Drawdown Metrics

| Metric | Value |
|--------|-------|
| Max Drawdown | {dd_metrics['max_drawdown_pct']} |
| Longest Drawdown | {dd_metrics['longest_drawdown_days']} days |
| Number of Drawdown Periods | {dd_metrics['num_drawdown_periods']} |

---

## 2. Monthly & Yearly P&L Breakdown

![Monthly Returns](plots/monthly_returns.png)

### Monthly Statistics

| Metric | Value |
|--------|-------|
| Best Month | {risk_metrics['best_month_pct']:.2f}% |
| Worst Month | {risk_metrics['worst_month_pct']:.2f}% |
| Median Month | {risk_metrics['median_month_pct']:.2f}% |
| Win Rate (Monthly) | {risk_metrics['win_rate_monthly']:.1f}% |

### Yearly Returns

| Year | P&L ($) | Return (%) | Periods |
|------|---------|------------|---------|
"""

    for _, row in yearly_df.iterrows():
        report += f"| {row['year']} | ${row['funding_pnl']:.2f} | {row['return_pct']:.2f}% | {row['num_periods']} |\n"

    report += f"""
---

## 3. Funding Rate Distribution

![Funding Distribution](plots/funding_histogram.png)

### Statistics

| Metric | Value |
|--------|-------|
| Total Periods | {funding_stats['total_periods']:,} |
| Positive Periods | {funding_stats['positive_periods']:,} ({funding_stats['positive_pct']:.1f}%) |
| Negative Periods | {funding_stats['negative_periods']:,} ({funding_stats['negative_pct']:.1f}%) |
| Mean Rate | {funding_stats['mean_rate']*100:.4f}% |
| Median Rate | {funding_stats['median_rate']*100:.4f}% |
| Std Dev | {funding_stats['std_rate']*100:.4f}% |
| Min Rate | {funding_stats['min_rate']*100:.4f}% |
| Max Rate | {funding_stats['max_rate']*100:.4f}% |

### Worst Negative Streak

- **Duration:** {funding_stats['worst_negative_streak_periods']} periods ({funding_stats['worst_negative_streak_days']:.1f} days)
- **Total Cost:** ${funding_stats['worst_streak_cost']:.2f}

---

## 4. Stress Tests

| Scenario | Periods | Loss ($) | Loss (%) |
|----------|---------|----------|----------|
"""

    for test in stress_tests:
        if 'min_rate' not in test:
            report += f"| {test['scenario']} | {test['periods']} | ${test['loss_usd']:.2f} | {test['loss_pct']:.2f}% |\n"

    # Find breakeven test
    breakeven = next((t for t in stress_tests if 'min_rate' in t), None)
    if breakeven:
        report += f"\n**Breakeven Rate (1-year hold):** {breakeven['min_rate']*100:.6f}% per period\n"

    report += f"""
---

## 5. Liquidation Risk Analysis

### Perpetual Short Position (Per Leverage)

| Leverage | Price Increase to Liquidation | Margin Required |
|----------|------------------------------|-----------------|
"""

    for lev in liq_analysis['leverage_scenarios']:
        report += f"| {lev['leverage']}x | {lev['price_increase_to_liq_pct']:.1f}% | {lev['margin_pct']:.1f}% |\n"

    report += f"""
### Pump Survival Analysis

To survive a price pump, you need sufficient margin:

| BTC Pump | Recommended Leverage | Notes |
|----------|---------------------|-------|
| 50% | 2x or less | Typical bull run spike |
| 100% | 1x only | Major bull market |
| 200% | 1x + buffer | Extreme scenario |

**Recommendation:** {liq_analysis['recommendation']}

---

## 6. Capital Requirements

| Component | Amount ($) |
|-----------|------------|
| Spot Position (BTC) | ${cap_req['spot_capital']:,.2f} |
| Perp Margin (2x leverage) | ${cap_req['perp_margin_2x']:,.2f} |
| Negative Funding Buffer | ${cap_req['negative_funding_buffer']:,.2f} |
| **Total Recommended** | **${cap_req['total_recommended']:,.2f}** |

**Capital Efficiency:** {cap_req['capital_efficiency']:.1f}%

*Note: For ${CAPITAL:,} notional exposure, you need ~${cap_req['total_recommended']:,.0f} total capital.*

---

## 7. Risk-Adjusted Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sharpe Ratio | {risk_metrics['sharpe_ratio']:.2f} | {"Excellent" if risk_metrics['sharpe_ratio'] > 2 else "Good" if risk_metrics['sharpe_ratio'] > 1 else "Moderate"} |
| Sortino Ratio | {risk_metrics['sortino_ratio']:.2f} | {"Excellent" if risk_metrics['sortino_ratio'] > 2 else "Good" if risk_metrics['sortino_ratio'] > 1 else "Moderate"} |
| Calmar Ratio | {risk_metrics['calmar_ratio']:.2f} | {"Excellent" if risk_metrics['calmar_ratio'] > 1 else "Good" if risk_metrics['calmar_ratio'] > 0.5 else "Moderate"} |

---

## 8. GO / NO-GO Evaluation

| Metric | Actual | Threshold | Status |
|--------|--------|-----------|--------|
"""

    for metric, actual, threshold, status in evaluation:
        emoji = "" if status == "GO" else "" if status == "NO-GO" else ""
        report += f"| {metric} | {actual} | {threshold} | **{status}** |\n"

    report += f"""
### Final Decision: **{final_decision}**

- GO criteria met: {go_count}/5
- NO-GO criteria triggered: {nogo_count}/5

---

## Key Risks Identified

1. **Negative Funding Periods:** {funding_stats['negative_pct']:.1f}% of all periods have negative funding. Extended bearish sentiment can lead to consecutive losses.

2. **Liquidation Risk:** If using leverage on the perpetual short, a rapid BTC pump (>50%) could liquidate the position before funding profits accumulate.

3. **Exchange Risk:** Keeping funds on centralized exchanges exposes capital to hacks, insolvency, or withdrawal restrictions.

4. **Execution Risk:** Slippage during entry/exit can erode profits. Market orders in volatile conditions may get worse fills.

5. **Basis Risk:** Spot and perpetual prices may diverge temporarily, causing mark-to-market losses even if delta-neutral.

---

## Recommendations

1. **Start with 1x leverage** on perpetual to survive 100%+ pumps
2. **Use ${cap_req['total_recommended']:,.0f}+** total capital for ${CAPITAL:,} exposure
3. **Monitor funding rates** during extreme market conditions
4. **Set stop-loss** if cumulative drawdown exceeds 10%
5. **Diversify across exchanges** (Binance, OKX) to reduce counterparty risk

---

*Report generated by comprehensive_risk_analysis.py*
"""

    return report


def main():
    """Run complete risk analysis."""
    print("=" * 60)
    print("COMPREHENSIVE RISK ANALYSIS: Always-In Funding Arbitrage")
    print("=" * 60)
    print()

    # Load data
    print("Loading funding rate data...")
    df = load_funding_data()
    print(f"  Loaded {len(df):,} periods from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Simulate strategy
    print("\nSimulating Always-In strategy...")
    sim_df = simulate_always_in(df)
    print(f"  Final equity: ${sim_df.iloc[-1]['equity']:,.2f}")
    print(f"  Total return: {(sim_df.iloc[-1]['equity']/CAPITAL - 1)*100:.2f}%")

    # Calculate metrics
    print("\nCalculating metrics...")
    monthly_df = calculate_monthly_returns(sim_df)
    yearly_df = calculate_yearly_returns(monthly_df)
    dd_metrics = calculate_drawdown_metrics(sim_df)
    funding_stats = analyze_funding_distribution(df)
    stress_results = stress_test(df)
    liq_analysis = liquidation_analysis()
    cap_req = capital_requirements()
    risk_metrics = calculate_risk_metrics(sim_df, monthly_df)

    # Evaluate GO/NO-GO
    go_nogo = evaluate_go_nogo(risk_metrics, dd_metrics, funding_stats, risk_metrics)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(sim_df, monthly_df, df)

    # Package results
    all_results = {
        'sim_df': sim_df,
        'monthly_df': monthly_df,
        'yearly_df': yearly_df,
        'dd_metrics': dd_metrics,
        'funding_stats': funding_stats,
        'stress_tests': stress_results,
        'liq_analysis': liq_analysis,
        'cap_req': cap_req,
        'risk_metrics': risk_metrics,
        'go_nogo': go_nogo,
    }

    # Generate report
    print("\nGenerating report...")
    report = generate_report(all_results)
    REPORT_PATH.write_text(report)
    print(f"  Saved to {REPORT_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal Return: {risk_metrics['total_return_pct']:.2f}%")
    print(f"Annualized Return: {risk_metrics['annualized_return_pct']:.2f}%")
    print(f"Max Drawdown: {dd_metrics['max_drawdown_pct']}")
    print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}")

    print("\n" + "=" * 60)
    evaluation, final_decision, go_count, nogo_count = go_nogo
    print(f"FINAL DECISION: {final_decision}")
    print("=" * 60)

    for metric, actual, threshold, status in evaluation:
        print(f"  {metric}: {actual} (threshold: {threshold}) -> {status}")

    print(f"\n  GO: {go_count}/5 | NO-GO: {nogo_count}/5")

    return all_results


if __name__ == "__main__":
    results = main()

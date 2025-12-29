"""
Predator Bot Configuration

Thresholds and weights for signal detection.
"""

# =============================================================================
# FUNDING RATE THRESHOLDS
# =============================================================================
# Funding > 0.05% = Overleveraged longs, consider SHORT
# Funding < -0.02% = Overleveraged shorts, consider LONG
FUNDING_EXTREME_HIGH = 0.0005   # 0.05%
FUNDING_EXTREME_LOW = -0.0002   # -0.02%
FUNDING_VERY_HIGH = 0.001       # 0.1% = extreme greed
FUNDING_VERY_LOW = -0.0005      # -0.05% = extreme fear

# Percentile thresholds for historical comparison
FUNDING_PERCENTILE_HIGH = 90    # Top 10% = extreme
FUNDING_PERCENTILE_LOW = 10     # Bottom 10% = extreme

# =============================================================================
# LONG/SHORT RATIO THRESHOLDS
# =============================================================================
# L/S > 2.0 = 66%+ accounts long, crowded trade -> SHORT
# L/S < 0.5 = 66%+ accounts short, crowded trade -> LONG
LS_RATIO_EXTREME_HIGH = 2.0
LS_RATIO_EXTREME_LOW = 0.5
LS_RATIO_MODERATE_HIGH = 1.5    # 60% long
LS_RATIO_MODERATE_LOW = 0.67    # 60% short

# =============================================================================
# OPEN INTEREST THRESHOLDS
# =============================================================================
# Significant OI changes indicate new positions
OI_CHANGE_SIGNIFICANT = 0.03    # 3% change
OI_CHANGE_MAJOR = 0.05          # 5% change = major event
OI_LOOKBACK_HOURS = 4           # Compare to 4 hours ago

# =============================================================================
# TAKER VOLUME THRESHOLDS
# =============================================================================
# Taker buy/sell ratio extremes
TAKER_IMBALANCE_HIGH = 1.5      # 50% more buyers
TAKER_IMBALANCE_LOW = 0.67      # 50% more sellers
TAKER_EXTREME_HIGH = 2.0        # 100% more buyers
TAKER_EXTREME_LOW = 0.5         # 100% more sellers

# =============================================================================
# CONVICTION SCORING WEIGHTS
# =============================================================================
WEIGHT_FUNDING = 0.30           # 30% - funding extremes
WEIGHT_OI = 0.25                # 25% - OI divergence
WEIGHT_LS_RATIO = 0.25          # 25% - L/S ratio
WEIGHT_TAKER = 0.20             # 20% - taker volume

# =============================================================================
# TRADE DECISION THRESHOLDS
# =============================================================================
CONVICTION_STRONG = 75          # Score >= 75 = strong signal
CONVICTION_MODERATE = 50        # Score 50-74 = moderate signal
CONVICTION_WEAK = 25            # Score 25-49 = wait
# Score < 25 = no trade

# =============================================================================
# API CONFIGURATION
# =============================================================================
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_DATA_BASE = "https://fapi.binance.com/futures/data"
DEFAULT_SYMBOL = "BTCUSDT"
API_TIMEOUT = 10  # seconds

# Data fetch intervals
FETCH_INTERVAL_MINUTES = 5      # Check every 5 minutes
HISTORY_DAYS = 30               # How much history to fetch

# =============================================================================
# SIGNAL DEFINITIONS
# =============================================================================
SIGNAL_LONG = "LONG"
SIGNAL_SHORT = "SHORT"
SIGNAL_NEUTRAL = "NEUTRAL"

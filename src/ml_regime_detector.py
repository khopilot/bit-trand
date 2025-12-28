"""
ML-Enhanced Regime Detector for BTC Elite Trader

Wraps the rule-based RegimeDetector with XGBoost classification for
better trend prediction based on historical patterns.

Features:
- Uses same indicators as rule-based detector (RSI, ADX, ATR, etc.)
- Trains on historical data to predict future regime
- Falls back to rule-based detection when ML unavailable
- Toggle between ML and rule-based via config

Author: khopilot
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from src.regime_detector import (
    RegimeDetector,
    RegimeState,
    TrendRegime,
    VolatilityRegime,
    MarketStructure,
)

logger = logging.getLogger("btc_trader.ml_regime")


@dataclass
class MLPrediction:
    """ML model prediction with confidence."""
    predicted_regime: TrendRegime
    confidence: float
    probabilities: Dict[str, float]
    features_used: Dict[str, float]


class MLRegimeDetector:
    """
    ML-enhanced regime detection using XGBoost.

    Falls back to rule-based RegimeDetector when:
    - ML libraries not installed
    - Model not trained
    - Confidence below threshold
    """

    # Feature columns for ML model
    FEATURE_COLS = [
        "rsi",
        "adx",
        "atr_pct",
        "ema_ratio",  # EMA fast/slow ratio
        "bb_position",  # Price position in Bollinger Bands (0-1)
        "volume_ratio",  # Current volume vs 20-day average
        "di_diff",  # +DI - -DI (trend direction strength)
        "choppiness",
        "price_vs_200ema",  # Price vs 200 EMA ratio
        "momentum_5d",  # 5-day price momentum
        "momentum_20d",  # 20-day price momentum
    ]

    # Regime labels for training
    REGIME_LABELS = {
        "uptrend": 0,
        "downtrend": 1,
        "ranging": 2,
    }
    REGIME_NAMES = {v: k for k, v in REGIME_LABELS.items()}

    def __init__(
        self,
        model_path: str = "models/regime_classifier.pkl",
        confidence_threshold: float = 0.6,
        enabled: bool = True,
        rule_based_detector: Optional[RegimeDetector] = None,
    ):
        """
        Initialize ML Regime Detector.

        Args:
            model_path: Path to saved XGBoost model
            confidence_threshold: Minimum confidence to use ML prediction
            enabled: Whether to use ML (False = always use rule-based)
            rule_based_detector: Existing RegimeDetector instance to wrap
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled and ML_AVAILABLE

        # Rule-based fallback
        self.rule_detector = rule_based_detector or RegimeDetector()

        # ML components
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._load_model()

        if not ML_AVAILABLE:
            logger.warning("ML libraries not installed. Using rule-based detection only.")
        elif not self.model:
            logger.info("No trained model found. Using rule-based detection until model is trained.")

    def _load_model(self) -> bool:
        """Load trained model from disk."""
        if not ML_AVAILABLE or not self.model_path.exists():
            return False

        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data.get("model")
            self.scaler = model_data.get("scaler")
            logger.info(f"Loaded ML model from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def save_model(self) -> bool:
        """Save trained model to disk."""
        if not self.model or not self.scaler:
            logger.error("No model to save")
            return False

        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.FEATURE_COLS,
                "trained_at": datetime.utcnow().isoformat(),
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Saved ML model to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def extract_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract ML features from OHLCV DataFrame.

        Requires columns: Open, High, Low, Close, Volume
        Adds indicators if not present.
        """
        if len(df) < 200:
            logger.warning(f"Need at least 200 rows for features, got {len(df)}")
            return None

        df = df.copy()

        # Ensure we have required columns (case-insensitive)
        col_map = {}
        for col in df.columns:
            col_map[col.lower()] = col

        # Calculate ADX if not present
        if "ADX" not in df.columns:
            df = self.rule_detector.calculate_adx(df)

        # Calculate ATR if not present
        if "ATR" not in df.columns:
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["ATR"] = tr.ewm(span=14, adjust=False).mean()

        # RSI
        if "RSI" not in df.columns:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)
            df["RSI"] = 100 - (100 / (1 + rs))

        # EMAs
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        # Bollinger Bands
        df["BB_Mid"] = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

        # Volume average
        df["Vol_Avg_20"] = df["Volume"].rolling(20).mean()

        # Choppiness
        chop = self.rule_detector.calculate_choppiness(df)
        df["Choppiness"] = chop

        # Build features DataFrame
        features = pd.DataFrame(index=df.index)

        # RSI (0-100)
        features["rsi"] = df["RSI"]

        # ADX (0-100)
        features["adx"] = df["ADX"]

        # ATR as percentage of price
        features["atr_pct"] = (df["ATR"] / df["Close"]) * 100

        # EMA ratio (fast/slow) - above 1 = bullish
        features["ema_ratio"] = df["EMA_12"] / df["EMA_26"].replace(0, np.finfo(float).eps)

        # BB position (0 = at lower, 1 = at upper)
        bb_range = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.finfo(float).eps)
        features["bb_position"] = (df["Close"] - df["BB_Lower"]) / bb_range

        # Volume ratio (current vs 20-day avg)
        features["volume_ratio"] = df["Volume"] / df["Vol_Avg_20"].replace(0, np.finfo(float).eps)

        # DI difference (+DI - -DI)
        features["di_diff"] = df["Plus_DI"] - df["Minus_DI"]

        # Choppiness
        features["choppiness"] = df["Choppiness"]

        # Price vs 200 EMA ratio
        features["price_vs_200ema"] = df["Close"] / df["EMA_200"].replace(0, np.finfo(float).eps)

        # Price momentum (returns)
        features["momentum_5d"] = df["Close"].pct_change(5) * 100
        features["momentum_20d"] = df["Close"].pct_change(20) * 100

        # Drop rows with NaN (warmup period)
        features = features.dropna()

        return features

    def create_labels(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        up_threshold: float = 0.02,
        down_threshold: float = -0.02,
    ) -> pd.Series:
        """
        Create regime labels from future returns.

        Args:
            df: DataFrame with Close prices
            forward_days: Days ahead to look for return
            up_threshold: Return threshold for uptrend (e.g., 0.02 = 2%)
            down_threshold: Return threshold for downtrend (e.g., -0.02 = -2%)

        Returns:
            Series of labels (0=uptrend, 1=downtrend, 2=ranging)
        """
        future_return = df["Close"].shift(-forward_days) / df["Close"] - 1

        labels = pd.Series(index=df.index, dtype=int)
        labels[future_return > up_threshold] = self.REGIME_LABELS["uptrend"]
        labels[future_return < down_threshold] = self.REGIME_LABELS["downtrend"]
        labels[(future_return >= down_threshold) & (future_return <= up_threshold)] = self.REGIME_LABELS["ranging"]

        return labels

    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        up_threshold: float = 0.02,
        down_threshold: float = -0.02,
        test_size: float = 0.2,
    ) -> Dict:
        """
        Train XGBoost model on historical data.

        Args:
            df: DataFrame with OHLCV data
            forward_days: Days ahead to predict
            up_threshold: Return threshold for uptrend label
            down_threshold: Return threshold for downtrend label
            test_size: Fraction of data for testing

        Returns:
            Dictionary with training metrics
        """
        if not ML_AVAILABLE:
            raise RuntimeError("ML libraries not installed. Run: pip install scikit-learn xgboost joblib")

        logger.info(f"Training ML regime model on {len(df)} rows...")

        # Extract features
        features = self.extract_features(df)
        if features is None or len(features) < 100:
            raise ValueError("Insufficient data for training")

        # Create labels
        labels = self.create_labels(df.loc[features.index], forward_days, up_threshold, down_threshold)

        # Align and drop NaN labels (last forward_days rows)
        valid_idx = labels.dropna().index
        X = features.loc[valid_idx][self.FEATURE_COLS]
        y = labels.loc[valid_idx].astype(int)

        logger.info(f"Training data: {len(X)} samples")
        logger.info(f"Label distribution: {dict(y.value_counts())}")

        # Train/test split (time-series aware - no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )

        # Evaluate
        from sklearn.metrics import accuracy_score, classification_report

        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Feature importance
        importance = dict(zip(self.FEATURE_COLS, self.model.feature_importances_))

        metrics = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "feature_importance": importance,
            "classification_report": classification_report(
                y_test, y_pred_test,
                target_names=["uptrend", "downtrend", "ranging"],
                output_dict=True,
            ),
        }

        logger.info(f"Training complete. Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")

        # Save model
        self.save_model()

        return metrics

    def predict(self, df: pd.DataFrame) -> Optional[MLPrediction]:
        """
        Predict regime using ML model.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            MLPrediction or None if prediction fails
        """
        if not self.enabled or not self.model or not self.scaler:
            return None

        features = self.extract_features(df)
        if features is None or len(features) == 0:
            return None

        # Get latest features
        latest = features.iloc[-1:][self.FEATURE_COLS]

        try:
            # Scale and predict
            scaled = self.scaler.transform(latest)
            proba = self.model.predict_proba(scaled)[0]
            predicted_class = int(np.argmax(proba))
            confidence = float(proba[predicted_class])

            # Map to TrendRegime
            regime_name = self.REGIME_NAMES[predicted_class]
            if regime_name == "uptrend":
                predicted_regime = TrendRegime.STRONG_UPTREND if confidence > 0.7 else TrendRegime.WEAK_UPTREND
            elif regime_name == "downtrend":
                predicted_regime = TrendRegime.STRONG_DOWNTREND if confidence > 0.7 else TrendRegime.WEAK_DOWNTREND
            else:
                predicted_regime = TrendRegime.RANGING

            return MLPrediction(
                predicted_regime=predicted_regime,
                confidence=confidence,
                probabilities={
                    "uptrend": float(proba[0]),
                    "downtrend": float(proba[1]),
                    "ranging": float(proba[2]),
                },
                features_used=dict(latest.iloc[0]),
            )
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect market regime with ML enhancement.

        Falls back to rule-based when:
        - ML disabled
        - No trained model
        - ML confidence below threshold

        Args:
            df: DataFrame with OHLCV data

        Returns:
            RegimeState with ML or rule-based classification
        """
        # Always get rule-based as baseline
        rule_state = self.rule_detector.detect(df)

        # Try ML prediction
        ml_pred = self.predict(df)

        if ml_pred is None:
            logger.debug("Using rule-based detection (ML unavailable)")
            return rule_state

        if ml_pred.confidence < self.confidence_threshold:
            logger.debug(
                f"Using rule-based detection (ML confidence {ml_pred.confidence:.2f} < {self.confidence_threshold})"
            )
            return rule_state

        # Use ML prediction for trend, keep other rule-based values
        logger.debug(
            f"Using ML detection: {ml_pred.predicted_regime.value} "
            f"(confidence: {ml_pred.confidence:.2f})"
        )

        # Determine trading decision with ML trend
        should_trade, size_mult, reason = self.rule_detector.should_trade(
            ml_pred.predicted_regime,
            rule_state.volatility,
            rule_state.structure,
            rule_state.choppiness,
        )

        # Boost or reduce confidence based on ML
        if ml_pred.confidence > 0.8:
            size_mult *= 1.1
            reason = f"[ML High Conf] {reason}"
        elif ml_pred.confidence < 0.65:
            size_mult *= 0.9
            reason = f"[ML Low Conf] {reason}"
        else:
            reason = f"[ML] {reason}"

        return RegimeState(
            trend=ml_pred.predicted_regime,
            volatility=rule_state.volatility,
            structure=rule_state.structure,
            adx=rule_state.adx,
            adx_trend=rule_state.adx_trend,
            atr_percentile=rule_state.atr_percentile,
            choppiness=rule_state.choppiness,
            should_trade=should_trade,
            position_size_multiplier=min(size_mult, 1.5),  # Cap at 1.5x
            reason=reason,
        )

    def get_regime_summary(self, state: RegimeState) -> str:
        """Get human-readable regime summary."""
        return self.rule_detector.get_regime_summary(state)

    @property
    def is_ml_active(self) -> bool:
        """Check if ML model is loaded and enabled."""
        return self.enabled and self.model is not None

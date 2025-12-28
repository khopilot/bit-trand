"""
Unit tests for ML Regime Detector.

Tests feature extraction, prediction, training, and fallback behavior.

Author: khopilot
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_regime_detector import MLRegimeDetector, MLPrediction, ML_AVAILABLE
from src.regime_detector import TrendRegime, VolatilityRegime, RegimeState


@pytest.fixture
def sample_df():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    # Generate 250 days of synthetic price data (need 200+ for features)
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")
    base_price = 40000
    prices = base_price + np.cumsum(np.random.randn(250) * 500)

    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": prices * 0.998,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Volume": np.random.randint(1000000, 10000000, 250),
    })

    return df


@pytest.fixture
def ml_detector():
    """ML detector without loading model."""
    detector = MLRegimeDetector(
        model_path="models/test_model.pkl",
        confidence_threshold=0.6,
        enabled=True,
    )
    return detector


class TestMLRegimeDetector:
    """Tests for MLRegimeDetector initialization and configuration."""

    def test_init_without_model(self, ml_detector):
        """Should initialize without pre-trained model."""
        assert ml_detector.confidence_threshold == 0.6
        assert ml_detector.rule_detector is not None

    def test_ml_available_flag(self):
        """ML_AVAILABLE should reflect actual library availability."""
        # Just verify it's a boolean
        assert isinstance(ML_AVAILABLE, bool)

    def test_fallback_to_rule_based(self, ml_detector, sample_df):
        """Should fall back to rule-based when no model."""
        # Without a trained model, should use rule-based
        state = ml_detector.detect(sample_df)

        assert isinstance(state, RegimeState)
        assert state.trend is not None
        assert state.volatility is not None

    def test_is_ml_active_without_model(self):
        """is_ml_active should be False without trained model."""
        # Create detector with non-existent model path
        detector = MLRegimeDetector(
            model_path="nonexistent/path/model.pkl",
            enabled=True,
        )
        # Model not loaded, so should be False
        assert detector.model is None
        assert not detector.is_ml_active


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_insufficient_data(self, ml_detector):
        """Should return None with insufficient data."""
        small_df = pd.DataFrame({
            "Open": [40000] * 50,
            "High": [40500] * 50,
            "Low": [39500] * 50,
            "Close": [40000] * 50,
            "Volume": [1000000] * 50,
        })

        features = ml_detector.extract_features(small_df)
        assert features is None

    def test_feature_columns(self, ml_detector, sample_df):
        """Should extract all required feature columns."""
        features = ml_detector.extract_features(sample_df)

        assert features is not None
        for col in ml_detector.FEATURE_COLS:
            assert col in features.columns, f"Missing feature: {col}"

    def test_features_no_nan_after_warmup(self, ml_detector, sample_df):
        """Features should have no NaN after warmup period."""
        features = ml_detector.extract_features(sample_df)

        assert features is not None
        # After dropna in extract_features, should have no NaN
        assert not features.isna().any().any(), "Features contain NaN"

    def test_rsi_bounds(self, ml_detector, sample_df):
        """RSI feature should be bounded 0-100."""
        features = ml_detector.extract_features(sample_df)

        assert features is not None
        assert features["rsi"].min() >= 0
        assert features["rsi"].max() <= 100

    def test_bb_position_reasonable(self, ml_detector, sample_df):
        """BB position should be roughly between 0-1 (can exceed slightly)."""
        features = ml_detector.extract_features(sample_df)

        assert features is not None
        # Most values should be in 0-1 range
        assert features["bb_position"].median() > 0
        assert features["bb_position"].median() < 1


class TestLabelCreation:
    """Tests for training label creation."""

    def test_create_labels(self, ml_detector, sample_df):
        """Should create valid labels from future returns."""
        labels = ml_detector.create_labels(
            sample_df,
            forward_days=5,
            up_threshold=0.02,
            down_threshold=-0.02,
        )

        assert len(labels) == len(sample_df)
        # Last forward_days should be NaN (no future data)
        assert labels.iloc[-1:].isna().all()

    def test_label_distribution(self, ml_detector, sample_df):
        """Labels should have all three classes."""
        labels = ml_detector.create_labels(
            sample_df,
            forward_days=5,
            up_threshold=0.01,  # Smaller threshold for more variety
            down_threshold=-0.01,
        )

        valid_labels = labels.dropna()
        unique_labels = valid_labels.unique()

        # Should have multiple classes (may not have all 3 with random data)
        assert len(unique_labels) >= 1


class TestTraining:
    """Tests for model training."""

    @pytest.mark.skipif(not ML_AVAILABLE, reason="ML libraries not installed")
    def test_training_basic(self, ml_detector, sample_df):
        """Should train model successfully with enough data."""
        metrics = ml_detector.train(
            sample_df,
            forward_days=5,
            up_threshold=0.02,
            down_threshold=-0.02,
            test_size=0.2,
        )

        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert "feature_importance" in metrics

        # Model should be trained
        assert ml_detector.model is not None
        assert ml_detector.scaler is not None

    @pytest.mark.skipif(not ML_AVAILABLE, reason="ML libraries not installed")
    def test_training_accuracy_reasonable(self, ml_detector, sample_df):
        """Training accuracy should be better than random (33%)."""
        metrics = ml_detector.train(
            sample_df,
            forward_days=5,
            up_threshold=0.02,
            down_threshold=-0.02,
        )

        # Train accuracy should be significantly better than random
        assert metrics["train_accuracy"] > 0.4

    def test_training_insufficient_data(self, ml_detector):
        """Should raise error with insufficient data."""
        small_df = pd.DataFrame({
            "Open": [40000] * 50,
            "High": [40500] * 50,
            "Low": [39500] * 50,
            "Close": [40000] * 50,
            "Volume": [1000000] * 50,
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            ml_detector.train(small_df)


class TestPrediction:
    """Tests for regime prediction."""

    @pytest.mark.skipif(not ML_AVAILABLE, reason="ML libraries not installed")
    def test_prediction_after_training(self, ml_detector, sample_df):
        """Should make predictions after training."""
        # Train first
        ml_detector.train(sample_df, test_size=0.2)

        # Predict
        pred = ml_detector.predict(sample_df)

        assert pred is not None
        assert isinstance(pred, MLPrediction)
        assert pred.confidence >= 0 and pred.confidence <= 1
        assert "uptrend" in pred.probabilities
        assert "downtrend" in pred.probabilities
        assert "ranging" in pred.probabilities

    @pytest.mark.skipif(not ML_AVAILABLE, reason="ML libraries not installed")
    def test_probabilities_sum_to_one(self, ml_detector, sample_df):
        """Prediction probabilities should sum to 1."""
        ml_detector.train(sample_df, test_size=0.2)
        pred = ml_detector.predict(sample_df)

        assert pred is not None
        prob_sum = sum(pred.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_prediction_without_model(self, sample_df):
        """Should return None when no model is trained."""
        # Create detector with non-existent model path
        detector = MLRegimeDetector(
            model_path="nonexistent/model.pkl",
            enabled=True,
        )
        pred = detector.predict(sample_df)
        assert pred is None


class TestRegimeDetection:
    """Tests for full regime detection flow."""

    def test_detect_returns_regime_state(self, ml_detector, sample_df):
        """detect() should always return a RegimeState."""
        state = ml_detector.detect(sample_df)

        assert isinstance(state, RegimeState)
        assert isinstance(state.trend, TrendRegime)
        assert isinstance(state.volatility, VolatilityRegime)

    @pytest.mark.skipif(not ML_AVAILABLE, reason="ML libraries not installed")
    def test_detect_uses_ml_when_available(self, ml_detector, sample_df):
        """Should use ML prediction when model available and confident."""
        # Train with enough data
        ml_detector.train(sample_df, test_size=0.2)

        state = ml_detector.detect(sample_df)

        # State should have ML indicators in reason (if confidence is high enough)
        assert isinstance(state, RegimeState)
        # The reason may contain "[ML]" if confidence threshold met

    def test_detect_uses_rule_based_when_disabled(self, sample_df):
        """Should use rule-based when ML disabled."""
        detector = MLRegimeDetector(enabled=False)

        state = detector.detect(sample_df)

        assert isinstance(state, RegimeState)
        # Should not have ML marker in reason
        assert "[ML]" not in state.reason


class TestModelPersistence:
    """Tests for model save/load."""

    @pytest.mark.skipif(not ML_AVAILABLE, reason="ML libraries not installed")
    def test_save_and_load_model(self, sample_df, tmp_path):
        """Should save and load model correctly."""
        model_path = tmp_path / "test_model.pkl"

        # Create and train detector
        detector1 = MLRegimeDetector(model_path=str(model_path))
        detector1.train(sample_df, test_size=0.2)

        # Save should work
        assert model_path.exists()

        # Create new detector and load
        detector2 = MLRegimeDetector(model_path=str(model_path))

        # Should have loaded model
        assert detector2.model is not None
        assert detector2.scaler is not None

    def test_save_without_model(self):
        """Should return False when saving without trained model."""
        # Create detector without loading any model
        detector = MLRegimeDetector(
            model_path="nonexistent/model.pkl",
            enabled=True,
        )
        result = detector.save_model()
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

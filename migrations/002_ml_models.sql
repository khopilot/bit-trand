-- BTC Elite Trader - ML Models Schema
-- Migration: 002_ml_models
-- Created: 2025-01-01

-- ML models metadata table
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'regime_classifier', 'price_predictor', etc.
    version INTEGER NOT NULL DEFAULT 1,
    model_path VARCHAR(255) NOT NULL,
    model_blob BYTEA,  -- Optional: store model binary in DB
    feature_columns JSONB NOT NULL,
    hyperparameters JSONB DEFAULT '{}',
    training_metrics JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    trained_at TIMESTAMP WITH TIME ZONE NOT NULL,
    training_data_start DATE,
    training_data_end DATE,
    training_samples INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(name, version)
);

-- Index for model lookups
CREATE INDEX IF NOT EXISTS idx_ml_models_name ON ml_models(name);
CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active);
CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type);

-- ML predictions log (for backtesting and analysis)
CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    prediction VARCHAR(50) NOT NULL,  -- 'uptrend', 'downtrend', 'ranging'
    confidence DECIMAL(5, 4) NOT NULL,
    probabilities JSONB NOT NULL,  -- {"uptrend": 0.6, "downtrend": 0.2, "ranging": 0.2}
    features JSONB NOT NULL,
    actual_outcome VARCHAR(50),  -- Filled later when we know the result
    actual_return DECIMAL(10, 6),  -- Actual return over prediction window
    was_correct BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for prediction analysis
CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_correct ON ml_predictions(was_correct) WHERE was_correct IS NOT NULL;

-- OHLCV history table (for ML training)
CREATE TABLE IF NOT EXISTS ohlcv_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT 'BTC/USDT',
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL DEFAULT '1d',  -- '1d', '4h', '1h'
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(symbol, timestamp, timeframe)
);

-- Index for OHLCV lookups
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_history(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON ohlcv_history(timeframe);

-- Apply updated_at trigger to new tables
DROP TRIGGER IF EXISTS update_ml_models_updated_at ON ml_models;
CREATE TRIGGER update_ml_models_updated_at
    BEFORE UPDATE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to mark a model as active (deactivates others of same type)
CREATE OR REPLACE FUNCTION activate_ml_model(p_model_id INTEGER)
RETURNS VOID AS $$
BEGIN
    -- Deactivate all models of the same type
    UPDATE ml_models SET is_active = FALSE
    WHERE model_type = (SELECT model_type FROM ml_models WHERE id = p_model_id);

    -- Activate the specified model
    UPDATE ml_models SET is_active = TRUE WHERE id = p_model_id;
END;
$$ LANGUAGE plpgsql;

-- View for latest active models
CREATE OR REPLACE VIEW active_ml_models AS
SELECT * FROM ml_models WHERE is_active = TRUE;

-- Comments
COMMENT ON TABLE ml_models IS 'Machine learning model metadata and versioning';
COMMENT ON TABLE ml_predictions IS 'ML prediction log for analysis and backtesting';
COMMENT ON TABLE ohlcv_history IS 'Historical OHLCV data for ML training';
COMMENT ON FUNCTION activate_ml_model IS 'Activates a model and deactivates others of same type';

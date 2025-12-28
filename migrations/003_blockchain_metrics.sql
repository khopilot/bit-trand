-- BTC Elite Trader - Blockchain Metrics Schema
-- Migration: 003_blockchain_metrics
-- Created: 2025-01-01

-- Exchange flow metrics (hourly aggregates)
CREATE TABLE IF NOT EXISTS exchange_flows (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    block_height_start INTEGER NOT NULL,
    block_height_end INTEGER NOT NULL,
    inflow_btc DECIMAL(20, 8) NOT NULL DEFAULT 0,
    outflow_btc DECIMAL(20, 8) NOT NULL DEFAULT 0,
    net_flow DECIMAL(20, 8) NOT NULL DEFAULT 0,
    large_tx_count INTEGER NOT NULL DEFAULT 0,
    signal VARCHAR(30) NOT NULL,  -- 'selling_pressure', 'accumulation', 'neutral'
    confidence DECIMAL(5, 4) NOT NULL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(block_height_start, block_height_end)
);

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_exchange_flows_timestamp ON exchange_flows(timestamp);
CREATE INDEX IF NOT EXISTS idx_exchange_flows_signal ON exchange_flows(signal);

-- Mempool metrics (periodic snapshots)
CREATE TABLE IF NOT EXISTS mempool_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    size INTEGER NOT NULL,  -- Number of transactions
    bytes BIGINT NOT NULL,  -- Total size in bytes
    avg_fee_rate DECIMAL(10, 4) NOT NULL,  -- sat/vbyte
    min_fee_rate DECIMAL(10, 4),
    max_fee_rate DECIMAL(10, 4),
    congestion VARCHAR(20) NOT NULL,  -- 'low', 'normal', 'high', 'extreme'
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_mempool_snapshots_timestamp ON mempool_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_mempool_snapshots_congestion ON mempool_snapshots(congestion);

-- On-chain signals log (trading decision adjustments)
CREATE TABLE IF NOT EXISTS onchain_signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    exchange_flow_id INTEGER REFERENCES exchange_flows(id),
    mempool_snapshot_id INTEGER REFERENCES mempool_snapshots(id),
    recommendation VARCHAR(30) NOT NULL,  -- 'boost_buy', 'reduce_buy', 'boost_sell', 'reduce_sell', 'neutral'
    confidence_adjustment DECIMAL(5, 4) NOT NULL DEFAULT 1.0,
    was_applied BOOLEAN DEFAULT FALSE,
    signal_id INTEGER REFERENCES signals(id),  -- Link to trading signal if applied
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for signal analysis
CREATE INDEX IF NOT EXISTS idx_onchain_signals_timestamp ON onchain_signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_onchain_signals_recommendation ON onchain_signals(recommendation);
CREATE INDEX IF NOT EXISTS idx_onchain_signals_applied ON onchain_signals(was_applied);

-- Known exchange addresses table (for updates without code changes)
CREATE TABLE IF NOT EXISTS exchange_addresses (
    id SERIAL PRIMARY KEY,
    address VARCHAR(100) NOT NULL UNIQUE,
    exchange_name VARCHAR(100),
    address_type VARCHAR(30),  -- 'hot', 'cold', 'deposit', 'withdrawal'
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for address lookups
CREATE INDEX IF NOT EXISTS idx_exchange_addresses_address ON exchange_addresses(address);
CREATE INDEX IF NOT EXISTS idx_exchange_addresses_exchange ON exchange_addresses(exchange_name);
CREATE INDEX IF NOT EXISTS idx_exchange_addresses_active ON exchange_addresses(is_active);

-- Apply updated_at trigger
DROP TRIGGER IF EXISTS update_exchange_addresses_updated_at ON exchange_addresses;
CREATE TRIGGER update_exchange_addresses_updated_at
    BEFORE UPDATE ON exchange_addresses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- View for recent exchange flow summary
CREATE OR REPLACE VIEW recent_exchange_flows AS
SELECT
    date_trunc('hour', timestamp) as hour,
    SUM(inflow_btc) as total_inflow,
    SUM(outflow_btc) as total_outflow,
    SUM(net_flow) as total_net_flow,
    SUM(large_tx_count) as large_tx_count,
    MODE() WITHIN GROUP (ORDER BY signal) as dominant_signal,
    AVG(confidence) as avg_confidence
FROM exchange_flows
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', timestamp)
ORDER BY hour DESC;

-- View for mempool trends
CREATE OR REPLACE VIEW mempool_trends AS
SELECT
    date_trunc('hour', timestamp) as hour,
    AVG(size) as avg_size,
    AVG(avg_fee_rate) as avg_fee_rate,
    MAX(avg_fee_rate) as max_fee_rate,
    MODE() WITHIN GROUP (ORDER BY congestion) as dominant_congestion
FROM mempool_snapshots
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', timestamp)
ORDER BY hour DESC;

-- Comments
COMMENT ON TABLE exchange_flows IS 'Exchange inflow/outflow metrics from on-chain analysis';
COMMENT ON TABLE mempool_snapshots IS 'Bitcoin mempool state snapshots';
COMMENT ON TABLE onchain_signals IS 'On-chain derived trading signal adjustments';
COMMENT ON TABLE exchange_addresses IS 'Known exchange Bitcoin addresses';

-- BTC Elite Trader - Initial Schema
-- Migration: 001_initial
-- Created: 2025-01-01

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT 'BTC/USDT',
    side VARCHAR(10) NOT NULL DEFAULT 'long',
    entry_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_order_id VARCHAR(100),
    exit_order_id VARCHAR(100),
    stop_price DECIMAL(20, 8),
    highest_price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    exit_time TIMESTAMP WITH TIME ZONE,
    exit_price DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    signal_type VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for status lookups
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_time);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    exchange_order_id VARCHAR(100),
    client_order_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT 'BTC/USDT',
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    average_price DECIMAL(20, 8),
    position_id INTEGER REFERENCES positions(id),
    exchange_response JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for order lookups
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_client_id ON orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_position ON orders(position_id);

-- Signals table (for audit trail)
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    signal_type VARCHAR(50) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    confidence DECIMAL(5, 4),
    reason TEXT,
    indicators JSONB DEFAULT '{}',
    executed BOOLEAN DEFAULT FALSE,
    order_id INTEGER REFERENCES orders(id),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for signal analysis
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);

-- Daily stats table
CREATE TABLE IF NOT EXISTS daily_stats (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    starting_equity DECIMAL(20, 8),
    ending_equity DECIMAL(20, 8),
    high_equity DECIMAL(20, 8),
    low_equity DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    max_drawdown DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 6),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for date lookups
CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date);

-- System state table
CREATE TABLE IF NOT EXISTS system_state (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Insert default system state
INSERT INTO system_state (key, value) VALUES
    ('trading_mode', '"paper"'),
    ('is_paused', 'false'),
    ('is_killed', 'false'),
    ('last_heartbeat', 'null')
ON CONFLICT (key) DO NOTHING;

-- Trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
DROP TRIGGER IF EXISTS update_positions_updated_at ON positions;
CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_daily_stats_updated_at ON daily_stats;
CREATE TRIGGER update_daily_stats_updated_at
    BEFORE UPDATE ON daily_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_system_state_updated_at ON system_state;
CREATE TRIGGER update_system_state_updated_at
    BEFORE UPDATE ON system_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE positions IS 'Trading positions with entry/exit tracking';
COMMENT ON TABLE orders IS 'Order history and tracking';
COMMENT ON TABLE signals IS 'Signal audit trail for analysis';
COMMENT ON TABLE daily_stats IS 'Daily performance metrics';
COMMENT ON TABLE system_state IS 'Key-value store for system state';

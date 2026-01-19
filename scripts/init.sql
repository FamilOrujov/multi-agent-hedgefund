-- Alpha-Council Database Initialization

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Checkpoints table for LangGraph state persistence
CREATE TABLE IF NOT EXISTS checkpoints (
    id VARCHAR(255) PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    ticker VARCHAR(20),
    step_name VARCHAR(100),
    state JSONB NOT NULL,
    iteration INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_ticker ON checkpoints(ticker);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at DESC);

-- Analysis history table
CREATE TABLE IF NOT EXISTS analysis_history (
    id VARCHAR(255) PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    confidence INTEGER NOT NULL,
    position_size VARCHAR(50),
    final_thesis TEXT,
    state_snapshot JSONB,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analysis_history_ticker ON analysis_history(ticker);
CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at ON analysis_history(created_at DESC);

-- News cache table for Tavily results
CREATE TABLE IF NOT EXISTS news_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    query TEXT NOT NULL,
    results JSONB NOT NULL,
    source VARCHAR(50) DEFAULT 'tavily',
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '1 hour')
);

CREATE INDEX IF NOT EXISTS idx_news_cache_ticker ON news_cache(ticker);
CREATE INDEX IF NOT EXISTS idx_news_cache_expires ON news_cache(expires_at);

-- Financial data cache
CREATE TABLE IF NOT EXISTS financial_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    source VARCHAR(50) NOT NULL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours')
);

CREATE INDEX IF NOT EXISTS idx_financial_cache_ticker ON financial_cache(ticker);
CREATE INDEX IF NOT EXISTS idx_financial_cache_type ON financial_cache(data_type);

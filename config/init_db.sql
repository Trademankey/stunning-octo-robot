-- ═══════════════════════════════════════════════════════════
-- TimescaleDB Schema — Institutional Crypto Trading Bot
-- ═══════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── OHLCV Candles ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION
);

SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv (symbol, timeframe, time DESC);

-- ── Trades ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,       -- 'buy' | 'sell'
    order_type      TEXT NOT NULL,       -- 'market' | 'limit' | 'stop'
    quantity        DOUBLE PRECISION,
    price           DOUBLE PRECISION,
    fee             DOUBLE PRECISION DEFAULT 0,
    pnl             DOUBLE PRECISION DEFAULT 0,
    strategy        TEXT,
    model_signal    DOUBLE PRECISION,
    regime          TEXT,
    notes           JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, time DESC);

-- ── Positions ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id              BIGSERIAL PRIMARY KEY,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    quantity        DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    pnl             DOUBLE PRECISION DEFAULT 0,
    pnl_pct         DOUBLE PRECISION DEFAULT 0,
    max_drawdown    DOUBLE PRECISION DEFAULT 0,
    strategy        TEXT,
    status          TEXT DEFAULT 'open'  -- 'open' | 'closed' | 'stopped'
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status, symbol);

-- ── Features Snapshot (for replay) ────────────────────────
CREATE TABLE IF NOT EXISTS feature_snapshots (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    features    JSONB NOT NULL
);

SELECT create_hypertable('feature_snapshots', 'time', if_not_exists => TRUE);

-- ── Model Metrics ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_metrics (
    id              BIGSERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name      TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    DOUBLE PRECISION,
    metadata        JSONB DEFAULT '{}'
);

-- ── Equity Curve ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS equity_curve (
    time        TIMESTAMPTZ NOT NULL,
    equity      DOUBLE PRECISION,
    drawdown    DOUBLE PRECISION,
    daily_pnl   DOUBLE PRECISION
);

SELECT create_hypertable('equity_curve', 'time', if_not_exists => TRUE);

-- ── Continuous Aggregates for fast dashboard queries ───────
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    first(open, time)  AS open,
    max(high)           AS high,
    min(low)            AS low,
    last(close, time)   AS close,
    sum(volume)         AS volume
FROM ohlcv
WHERE timeframe = '1m'
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset    => INTERVAL '3 hours',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- ── Data Retention ─────────────────────────────────────────
SELECT add_retention_policy('ohlcv', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('feature_snapshots', INTERVAL '30 days', if_not_exists => TRUE);

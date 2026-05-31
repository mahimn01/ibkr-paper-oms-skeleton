-- PIT bitemporal metadata schema (SQLite + WAL).
-- See PLAN.md §2.1.
--
-- Conventions:
--   * `internal_id` is a permanent surrogate key; never reused.
--   * `valid_*` columns are valid-time (when fact was true in the world).
--   * `known_*` columns are transaction-time (when our system learned it).
--     Restatements append a new row; the old row gets `known_to = restatement_date`.
--   * Dates use ISO YYYY-MM-DD; '9999-12-31' = "still in effect".

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS securities (
    internal_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    primary_ticker TEXT    NOT NULL,
    cusip         TEXT,
    figi          TEXT,
    list_date     TEXT,
    delist_date   TEXT,
    delist_reason TEXT,
    known_from    TEXT    NOT NULL,
    known_to      TEXT    NOT NULL DEFAULT '9999-12-31'
);
CREATE INDEX IF NOT EXISTS idx_sec_ticker  ON securities(primary_ticker);
CREATE INDEX IF NOT EXISTS idx_sec_cusip   ON securities(cusip);

CREATE TABLE IF NOT EXISTS ticker_history (
    internal_id INTEGER NOT NULL,
    ticker      TEXT    NOT NULL,
    valid_from  TEXT    NOT NULL,
    valid_to    TEXT    NOT NULL DEFAULT '9999-12-31',
    PRIMARY KEY (internal_id, valid_from),
    FOREIGN KEY (internal_id) REFERENCES securities(internal_id)
);
CREATE INDEX IF NOT EXISTS idx_th_ticker ON ticker_history(ticker, valid_from, valid_to);

CREATE TABLE IF NOT EXISTS splits (
    internal_id INTEGER NOT NULL,
    ex_date     TEXT    NOT NULL,
    ratio       REAL    NOT NULL,         -- new/old (4-for-1 = 4.0)
    PRIMARY KEY (internal_id, ex_date),
    FOREIGN KEY (internal_id) REFERENCES securities(internal_id)
);

CREATE TABLE IF NOT EXISTS dividends (
    internal_id INTEGER NOT NULL,
    ex_date     TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    div_type    TEXT    NOT NULL,         -- regular/special/roc/stock
    PRIMARY KEY (internal_id, ex_date, div_type),
    FOREIGN KEY (internal_id) REFERENCES securities(internal_id)
);

CREATE TABLE IF NOT EXISTS mergers (
    source_id      INTEGER NOT NULL,
    target_id      INTEGER,               -- nullable for cash mergers
    effective_date TEXT    NOT NULL,
    cash_per_share REAL    NOT NULL DEFAULT 0,
    share_ratio    REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (source_id, effective_date),
    FOREIGN KEY (source_id) REFERENCES securities(internal_id),
    FOREIGN KEY (target_id) REFERENCES securities(internal_id)
);

CREATE TABLE IF NOT EXISTS spinoffs (
    parent_id      INTEGER NOT NULL,
    child_id       INTEGER NOT NULL,
    ex_date        TEXT    NOT NULL,
    ratio          REAL    NOT NULL,
    cost_basis_pct REAL    NOT NULL,
    PRIMARY KEY (parent_id, child_id, ex_date),
    FOREIGN KEY (parent_id) REFERENCES securities(internal_id),
    FOREIGN KEY (child_id)  REFERENCES securities(internal_id)
);

CREATE TABLE IF NOT EXISTS index_membership (
    index_name    TEXT    NOT NULL,        -- SP500, R1000, R2000, R3000, NDX, etc.
    internal_id   INTEGER NOT NULL,
    added_date    TEXT    NOT NULL,
    removed_date  TEXT,                    -- NULL = currently a member
    announce_date TEXT,
    PRIMARY KEY (index_name, internal_id, added_date),
    FOREIGN KEY (internal_id) REFERENCES securities(internal_id)
);
CREATE INDEX IF NOT EXISTS idx_im_active ON index_membership(index_name, removed_date);
CREATE INDEX IF NOT EXISTS idx_im_dates  ON index_membership(index_name, added_date, removed_date);

-- Risk-state table is a single-row latch for RiskManager persistence.
-- Lives in the same DB so order-log + risk-state writes can share a transaction.
CREATE TABLE IF NOT EXISTS risk_state (
    id                       INTEGER PRIMARY KEY CHECK (id = 1),
    session_start_net_liq    REAL,
    orders_today             INTEGER NOT NULL DEFAULT 0,
    orders_today_date        TEXT,
    last_updated             TEXT    NOT NULL
);

-- Cache for daily effective spread / impact estimates (Corwin-Schultz, etc.).
-- Keyed by (symbol, date, estimator).
CREATE TABLE IF NOT EXISTS spread_cache (
    internal_id INTEGER NOT NULL,
    date        TEXT    NOT NULL,
    estimator   TEXT    NOT NULL,            -- corwin_schultz / abdi_ranaldo / roll
    spread_bps  REAL    NOT NULL,
    computed_at TEXT    NOT NULL,
    PRIMARY KEY (internal_id, date, estimator),
    FOREIGN KEY (internal_id) REFERENCES securities(internal_id)
);

-- Borrow rates for shorts. NULL rate = unknown / HTB; strategy must skip.
CREATE TABLE IF NOT EXISTS borrow_rates (
    internal_id INTEGER NOT NULL,
    date        TEXT    NOT NULL,
    rate_bps    REAL,                        -- annualized
    shortable_shares INTEGER,
    source      TEXT,                        -- ibkr / markit / default_tier
    PRIMARY KEY (internal_id, date),
    FOREIGN KEY (internal_id) REFERENCES securities(internal_id)
);

-- Migration log: every bulk import / restatement leaves a trail.
CREATE TABLE IF NOT EXISTS migration_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    source       TEXT NOT NULL,              -- legacy_json_cache / norgate / ibkr / ...
    rows_in      INTEGER,
    rows_written INTEGER,
    notes        TEXT
);

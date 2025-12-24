# Workflows

## Run unit tests

`python3 -m unittest discover -s tests`

## Run paper integration tests (requires TWS running)

`RUN_IBKR_INTEGRATION=1 IBKR_PORT=7497 python3 -m unittest discover -s tests/integration`

## Place an order (paper, gated)

```bash
export TRADING_DB_PATH=trading_audit.sqlite3
export TRADING_DRY_RUN=false
export TRADING_LIVE_ENABLED=true
export TRADING_ORDER_TOKEN=I_UNDERSTAND_PAPER_TRADING

python3 -m trading_algo.cli --confirm-token "$TRADING_ORDER_TOKEN" --ibkr-port 7497 \
  place-order --broker ibkr --kind STK --symbol AAPL --qty 1 --side BUY --type MKT
```

## Modify an order

`python3 -m trading_algo.cli --confirm-token "$TRADING_ORDER_TOKEN" modify-order --broker ibkr --order-id 123 --kind STK --symbol AAPL --side BUY --qty 1 --type LMT --limit-price 99`

## Reconcile + track after restart

```bash
python3 -m trading_algo.cli oms-reconcile --broker ibkr
python3 -m trading_algo.cli oms-track --broker ibkr --poll-seconds 1 --timeout-seconds 300
```

## Run AutoRunner

Sim:
- `python3 -m trading_algo.autorun --broker sim --max-ticks 5 --sleep-seconds 0`

IBKR (paper):
- `python3 -m trading_algo.autorun --broker ibkr --ibkr-port 7497 --confirm-token "$TRADING_ORDER_TOKEN"`

## IBKR historical backtests (recommended workflow)

1) Export historical bars from IBKR to a CSV:

```bash
python3 -m trading_algo.cli export-history --broker ibkr --kind STK --symbol AAPL \
  --bar-size "5 mins" --duration-per-call "30 D" --out-csv data/AAPL_5m.csv --validate
```

2) Run a deterministic backtest from the CSV:

```bash
python3 -m trading_algo.cli backtest --csv data/AAPL_5m.csv --kind STK --symbol AAPL
```

Notes:
- IBKR historical data availability depends on your permissions/subscriptions and pacing limits.
- Export once, then iterate on your strategy using the same CSV for repeatable results.

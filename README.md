# IBKR Trading Algo Skeleton (Paper/Sim)

Minimal, extensible starter structure for an algorithmic trading bot that connects to Interactive Brokers (IBKR) paper trading via TWS or IB Gateway.

## What this gives you

- A clean project layout (`trading_algo/`) with an execution engine, strategy hooks, and a broker abstraction.
- An `IBKRBroker` implementation (uses `ib_insync`) for paper trading.
- A `SimBroker` so you can run the bot without TWS/Gateway running.
- A CLI to place basic orders (stocks/futures/forex), fetch a market data snapshot, and run a simple example strategy loop.

## Prereqs (IBKR)

1. Install and run **Trader Workstation (TWS)** or **IB Gateway**
2. Log into **Paper Trading**
3. Enable API access:
   - TWS: `File -> Global Configuration -> API -> Settings -> Enable ActiveX and Socket Clients`
   - Set an API port:
     - TWS paper default: `7497` (live: `7496`)
     - Gateway paper default: `4002` (live: `4001`)
   - Optional: add trusted IPs / disable read-only API

## Install (Python)

This skeleton expects Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` and edit if you want:

```bash
cp .env.example .env
```

Safety defaults:
- `TRADING_REQUIRE_PAPER` is enforced in code (cannot be disabled): the bot refuses to run if the connected session doesn't look like Paper Trading (managed accounts not starting with `DU`).
- `TRADING_DRY_RUN=true` stages orders only (no orders sent).
- `TRADING_LIVE_ENABLED=false` blocks sending orders to IBKR unless you explicitly enable it.
- `TRADING_ORDER_TOKEN` + `--confirm-token` is a second gate required to send IBKR orders.
- `TRADING_DB_PATH` enables a sqlite audit log for decisions and orders.

## Run (no IBKR required)

```bash
python3 -m trading_algo.cli place-order --broker sim --symbol AAPL --qty 1 --side BUY --type MKT
python3 -m trading_algo.cli run --broker sim
```

## Run (paper trading)

Make sure TWS/Gateway is running and API is enabled, then:

```bash
export TRADING_DRY_RUN=false
export TRADING_LIVE_ENABLED=true
export TRADING_ORDER_TOKEN=I_UNDERSTAND_PAPER_TRADING
python3 -m trading_algo.cli --confirm-token "$TRADING_ORDER_TOKEN" place-order --broker ibkr --kind STK --symbol AAPL --qty 1 --side BUY --type MKT
python3 -m trading_algo.cli --confirm-token "$TRADING_ORDER_TOKEN" place-order --broker ibkr --kind FUT --symbol ES --exchange CME --expiry 202503 --qty 1 --side BUY --type MKT
python3 -m trading_algo.cli --confirm-token "$TRADING_ORDER_TOKEN" place-order --broker ibkr --kind FX --symbol EURUSD --qty 10000 --side BUY --type MKT
python3 -m trading_algo.cli run --broker ibkr
```

## Market data snapshot

```bash
python3 -m trading_algo.cli snapshot --broker ibkr --kind STK --symbol AAPL
python3 -m trading_algo.cli snapshot --broker ibkr --kind FUT --symbol ES --exchange CME --expiry 202503
python3 -m trading_algo.cli snapshot --broker ibkr --kind FX --symbol EURUSD
```

## Historical bars

```bash
python3 -m trading_algo.cli history --broker ibkr --kind STK --symbol AAPL --duration "1 D" --bar-size "5 mins"
```

## Paper smoke test

```bash
python3 -m trading_algo.cli --ibkr-port 7497 paper-smoke
python3 -m trading_algo.cli --ibkr-port 7497 --confirm-token "$TRADING_ORDER_TOKEN" paper-smoke --order-test
```

## Order utilities

```bash
python3 -m trading_algo.cli order-status --broker ibkr --order-id 4
python3 -m trading_algo.cli cancel-order --broker ibkr --order-id 4
python3 -m trading_algo.cli place-bracket --broker ibkr --kind STK --symbol AAPL --side BUY --qty 1 --entry-limit 100 --take-profit 110 --stop-loss 95
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## SQLite audit trail

Set `TRADING_DB_PATH=trading_audit.sqlite3` and the CLI/Engine will log:
- `runs` (config snapshot + start/end)
- `decisions` (strategy intents accepted/rejected)
- `orders` (submitted orders)
- `order_status_events` (status snapshots like Submitted/Filled/Cancelled)
- `errors` (exceptions + key failures)

## Docs

- `docs/ARCHITECTURE.md`
- `docs/SAFETY.md`
- `docs/WORKFLOWS.md`
- `docs/DB_SCHEMA.md`

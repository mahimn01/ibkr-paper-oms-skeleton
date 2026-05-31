# Trading System

## Stack
Python 3.11+, IBKR TWS API, asyncio, click CLI

## Commands
python -m trading_algo --help    # Show all commands
python -m trading_algo --paper   # Paper trading (port 4002)
python -m trading_algo --live    # Live trading (port 4001)
pytest                           # Run tests

# PIT data layer (Wave T5)
python scripts/migrate_to_pit.py --cache-dir <legacy> --pit-root <new> [--reconcile]

## Architecture
- Entry: trading_algo/cli.py
- Broker: trading_algo/broker/ibkr.py
- Config: trading_algo/config.py (TradingConfig with safety rails)
- Strategies: trading_algo/strategies/ (14 equity + 9 crypto)
- PIT data: trading_algo/data/ (PITStore, UniverseResolver, AdjustmentEngine)
- Cost model: trading_algo/backtest_v2/cost_model.py
- Validator: trading_algo/quant_core/validation/{pbo,stationary_bootstrap,report_card}.py

## Rules
- ALWAYS use IBKR TWS API first for live data before WorldMonitor or web searches
- Paper port: 4002, Live port: 4001 — NEVER mix these up
- IB Gateway runs 24/7 via IBC in tmux, auto-restart at 23:55 ET
- All risk limits enforced at config level, not strategy level
- Type hints on ALL function signatures
- --paper and --live are mutually exclusive CLI flags

### Wave T5 rules (enterprise foundation)
- **NEVER hardcode a symbol list for a backtest universe.** Use
  `UniverseResolver.get_universe('SP500', as_of=date(...))`. Hardcoded
  lists are survivorship-biased; the resolver enforces this with
  `SurvivorshipBiasError` unless `allow_dev=True` is set explicitly for
  exploratory work.
- **Bars in the PIT store are stored UNADJUSTED.** Apply splits at
  query time via `AdjustmentEngine.factor(internal_id, bar_date, as_of)`
  or `adjust_series`. Forward-adjusting at write time is permanently
  wrong — a future split silently rewrites every prior backtest.
- **Realistic backtests must set `BacktestConfig(cost_model_config=
  CostModelConfig(), execution_policy='next_bar_open')`.** The legacy
  flat-bps + same-bar-close defaults exist only for backwards
  compatibility with existing tests; they overstate Sharpe by ~50–150
  bps/year on liquid equity strategies.
- **Every order to the broker carries a `client_order_id`.**
  `TradeIntent` populates one via uuid4 default factory;
  `OrderRequest.normalized()` auto-fills if a caller bypasses
  TradeIntent. The OMS asserts non-empty before submit. If you ever
  see "OrderRequest.order_ref must be set", a caller is bypassing
  normalised — fix the caller, don't disable the assertion.
- **`oms.submit()` and `oms.modify()` are halt-gated.** Even dry-runs
  block while `data/HALTED` exists. `oms.cancel()` is intentionally
  NOT halt-gated (cancels reduce exposure). To cut all broker traffic,
  kill the IB Gateway process.
- **RiskManager state persists across restarts when `db_path` is set.**
  daily-loss circuit breaker and orders/day cap survive crashes. Never
  pass a fresh in-memory RiskManager to live; always wire it to the
  trading SQLite path so a 2 PM crash doesn't reset the day's risk
  budget.
- **Validator threshold gates (PLAN.md §2.7):** PBO < 0.5, DSR > 0.95,
  lower 95% CI on annualised Sharpe > 0.3, walk-forward 12m Sharpe > 0
  in >= 75% of windows, MinTRL <= years available, cost-adjusted Sharpe
  > 0.3. Run `build_report_card(...)` on every strategy before
  deployment; status must be `APPROVED`.

## Self-Improvement
After every bug fix or correction, add a rule here to prevent repeating it.

- **IBKR order submission requires explicit account on multi-account logins.** Error 435 "You must specify an account" fires silently within ~500ms of placeOrder when multiple accounts are linked. The `cmd_place`/`cmd_combo` now call `_resolve_account()` (checks --account → IBKR_ACCOUNT env → sole managed account, else fail loudly) and `_wait_for_order_ack()` (waits up to 15s for terminal/stable state instead of a blind 2s sleep). Never reintroduce blind post-placeOrder sleeps shorter than IBKR's rejection-notification window.
- **CSCV PBO formula must threshold at canonical N/2 OOS rank, not the inverted complement.** The bug returned `mean(logits < 0)` which under-reported PBO by an order of magnitude (0.13 vs 0.87 on pure noise). Fixed in `quant_core/validation/pbo.py`. If you ever rewrite a PBO routine: with N=50 pure-noise trials, the answer must be in [0.5, 1.0]; if it's near 0.1 you have the inverted formula. Reference: Bailey-Borwein-LdP-Zhu 2017, J. Comput. Finance 20(4).
- **PyArrow nanosecond timestamps cap at year 2262.** `datetime(9999, 12, 31)` as a far-future bitemporal sentinel is out of range. Use `pa.timestamp('us')` (microsecond resolution, range to 294247) instead. Also strip tzinfo from datetimes before passing to pyarrow, or naive vs aware comparisons fail at read time.

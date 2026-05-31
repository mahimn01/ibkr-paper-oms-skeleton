# Trading-Algo Enterprise Foundation — PLAN

**Branch:** `enterprise-foundation-v1`
**Mission:** Take trading-algo from "sophisticated infrastructure with unverified strategies and silent backtest fiction" to "institutional-grade research + execution platform on which a real edge can be built and trusted."
**Authored:** 2026-05-01

---

## 0. Executive summary

Six parallel agents read the codebase and the literature in depth. The findings, reduced to one paragraph each:

- **Architecture is the strong layer.** The IBKR broker, circuit breaker, contract cache, and order-ack waiting are well done. The 5-layer quant orchestrator (data → signal → risk → portfolio → execution) is a clean skeleton. ATLAS / RAT / LLM are research artifacts of unverified value.
- **The data layer is broken.** No point-in-time correctness, no survivorship-bias-free universe, no corporate-action handling, JSON storage with no checksums or versioning. Every backtest above this layer is suspect.
- **The fill model is fiction.** Same-bar close fills with fixed-bps slippage. No effective spread from OHLC, no square-root impact, no borrow cost on shorts, no margin financing. Almgren-Chriss exists as a *planning* module but is not wired into backtest fills. Estimated optimism: **+270–1450 bps/year** depending on strategy.
- **Idempotency is optional.** `client_order_id` is generated only when callers pass an `idempotency_key`. Halt sentinel is checked on some CLI commands but **not at order submission**. Risk state (daily loss, orders/day, session-start NetLiq) is memory-only and resets on every process restart.
- **Validation framework exists but is not used and has bugs.** The CSCV PBO logit thresholding is wrong. The Deflated Sharpe in `BacktestValidator` skips skew/kurtosis. `n_trials` is hardcoded to 1 in the orchestrator. Stationary bootstrap (Politis-Romano) is not implemented. No strategy has a validation report on disk.
- **The strategies are mostly not defensible.** Lead-Lag Arbitrage hardcodes pairs from in-sample data. Volatility Maximizer levers 1.5× Kelly on noisy GARCH. HMM regime strategies have no demonstrated OOS. Options strategies use Black-Scholes with constant IV. Of 27, an estimated 5 survive a real validation run.

The next-level move is not "build more strategies." It is to **fix the foundation, run the validator, kill the dead wood, then build 3–4 real edges on top.** This document is the spec for the foundation work.

---

## 1. Phase plan

| Phase | Theme | Duration | Branch state |
|---|---|---|---|
| **1. Foundation** | PIT data, idempotency, risk persistence, realistic costs, validator wiring | 4–6 weeks | this branch (`enterprise-foundation-v1`) |
| **2. Cull** | Run validator on every existing strategy; delete failures with postmortems | 3–4 weeks | branch `phase-2-cull` |
| **3. Build edges** | XSP VRP, crypto funding carry, 8-K LLM classifier, Russell reconstitution | 4–6 months | feature branches per edge |
| **4. Process** | Research log, pre-registration, kill rules, shadow trading, factor-stripped accounting | ongoing | infrastructure branches |

This plan covers **Phase 1 only**. Phases 2–4 are scoped at the end.

---

## 2. Architecture targets

### 2.1 Bitemporal data store

Replace `quant_core/data/ibkr_data_loader.py` (JSON files, hand-picked universes) with a proper bitemporal layer.

**Stack:**
- **DuckDB** for queries (no server, queries parquet in place)
- **Parquet** for bars (partitioned by `symbol/year`)
- **SQLite + WAL** for metadata (security_master, splits, dividends, mergers, spinoffs, index_membership, ticker_history)

**Schema (SQLite):**

```sql
CREATE TABLE securities (
    internal_id INTEGER PRIMARY KEY,
    primary_ticker TEXT NOT NULL,
    cusip TEXT,
    figi TEXT,
    list_date DATE,
    delist_date DATE,
    delist_reason TEXT,           -- acquired/bankrupt/voluntary/null
    known_from TIMESTAMP NOT NULL,
    known_to TIMESTAMP NOT NULL DEFAULT '9999-12-31'
);
CREATE INDEX idx_sec_ticker ON securities(primary_ticker);

CREATE TABLE ticker_history (
    internal_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL DEFAULT '9999-12-31',
    PRIMARY KEY (internal_id, valid_from)
);

CREATE TABLE splits (
    internal_id INTEGER NOT NULL,
    ex_date DATE NOT NULL,
    ratio REAL NOT NULL,           -- new/old (e.g. 4-for-1 = 4.0)
    PRIMARY KEY (internal_id, ex_date)
);

CREATE TABLE dividends (
    internal_id INTEGER NOT NULL,
    ex_date DATE NOT NULL,
    amount REAL NOT NULL,
    div_type TEXT NOT NULL,        -- regular/special/roc/stock
    PRIMARY KEY (internal_id, ex_date, div_type)
);

CREATE TABLE mergers (
    source_id INTEGER NOT NULL,
    target_id INTEGER,             -- nullable for cash mergers
    effective_date DATE NOT NULL,
    cash_per_share REAL DEFAULT 0,
    share_ratio REAL DEFAULT 0,
    PRIMARY KEY (source_id, effective_date)
);

CREATE TABLE spinoffs (
    parent_id INTEGER NOT NULL,
    child_id INTEGER NOT NULL,
    ex_date DATE NOT NULL,
    ratio REAL NOT NULL,
    cost_basis_pct REAL NOT NULL,
    PRIMARY KEY (parent_id, child_id, ex_date)
);

CREATE TABLE index_membership (
    index_name TEXT NOT NULL,      -- SP500, R1000, R2000, R3000
    internal_id INTEGER NOT NULL,
    added_date DATE NOT NULL,
    removed_date DATE,             -- null = currently a member
    announce_date DATE,
    PRIMARY KEY (index_name, internal_id, added_date)
);
CREATE INDEX idx_im_active ON index_membership(index_name, removed_date);

CREATE TABLE risk_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    session_start_net_liq REAL,
    orders_today INTEGER NOT NULL DEFAULT 0,
    orders_today_date TEXT,
    last_updated TIMESTAMP NOT NULL
);
```

**Schema (Parquet bars):**

```
data/pit/bars/symbol={X}/year={Y}/data.parquet
columns: date, open, high, low, close, volume, vwap (optional),
         known_from, known_to
```

The `known_from`/`known_to` columns are the transaction-time axis. A query "what did our database know about AAPL on 2018-06-15" filters `known_from <= '2018-06-15' AND known_to > '2018-06-15'`. Restatements append a new row instead of overwriting; the old row gets `known_to = restatement_date`.

**Adjustment policy: store unadjusted, adjust at query time.** Every backtest builds a cumulative adjustment factor from `splits` and (optionally) `dividends` for events strictly after the simulated time T, and applies it for indicator computation only.

### 2.2 Survivorship-free universe

`UniverseResolver.get_universe(index_name, as_of_date)` returns the exact constituents of `index_name` on `as_of_date` from `index_membership`. Strategies that take a universe receive this list, not a hardcoded constants block.

`BacktestConfig` gains `universe_spec: str | list[str]` (string = index name, list = literal symbols) and `universe_source: str` (required: `pit_store` or `literal`).

### 2.3 Realistic cost stack

`backtest_v2/cost_model.py` — new module with three composable estimators.

**Effective spread (Corwin-Schultz 2012 + Abdi-Ranaldo 2017 fallback).**

For each `(symbol, date)`, daily-bar effective spread:

```
β_t = [ln(H_t/L_t)]² + [ln(H_{t+1}/L_{t+1})]²
γ_t = [ln(H_t,t+1/L_t,t+1)]²
α   = (sqrt(2β) − sqrt(β)) / (3 − 2*sqrt(2)) − sqrt(γ / (3 − 2*sqrt(2)))
S_CS = 2(e^α − 1) / (1 + e^α)
```

When `S_CS ≤ 0` (~10% of days) fall back to Abdi-Ranaldo:

```
η_t = (H_t + L_t) / 2
S_AR² = E[4 · (C_t − η_t)(C_t − η_{t+1})]
```

Cache per `(symbol, date)` to SQLite or parquet sidecar.

**Charge on fill:**
- Market orders: half-spread (effective) + impact (below).
- Limit orders that fill: marked-to-market against next-bar mid (adverse selection ≈ 1–3 bps).

**Square-root impact** (Tóth-Bouchaud 2011). For order quantity `Q` against ADV `V` (20-day):

```
participation = Q / V
if participation < 0.001:                 # < 0.1% ADV → no impact
    impact_bps = 0
else:
    impact_bps = Y * sigma_daily_bps * sqrt(participation)
```

with `Y = 0.5` (CFM consensus). Capped at participation 0.10 (above which the model itself breaks).

**Borrow cost.** `BorrowCostModel.daily_charge(symbol, date, notional)`:

```
rate = lookup_borrow_rate(symbol, date) or default_by_tier(symbol)
default tiers:
    SP500 / mega-cap   → 30 bps
    Russell 1000       → 50 bps
    Russell 2000       → 200 bps
    HTB / unknown      → SKIP (don't fake)
charge = notional * rate / 360   # daily, ACT/360
```

Recall risk: with daily probability `min(0.05, rate / 200)`, force-close at next open.

**Commission (IBKR Tiered).** Replace flat `commission_per_share` with tiered structure:

```
$0.0035/share, min $0.35, max 1% trade value, 0.5% NMS pass-through
SEC fee: 0.00278% on sells (2026 rate, recheck quarterly)
FINRA TAF: $0.000166/share on sells, max $9.27
```

### 2.4 Look-ahead protection

Add `ExecutionPolicy` enum to `backtest_v2/models.py`:

```python
class ExecutionPolicy(str, Enum):
    SAME_BAR_CLOSE = "same_bar_close"        # legacy / explicit opt-in
    NEXT_BAR_OPEN  = "next_bar_open"         # default
    NEXT_BAR_VWAP  = "next_bar_vwap"         # conservative
```

Refactor `engine.py:_process_bar` to:

1. Generate signals on bar T
2. Push to `pending_orders: dict[symbol, PendingOrder]`
3. On bar T+1 arrival, fill pending at policy-determined price
4. Apply spread + impact + commission at fill

Default policy: `NEXT_BAR_OPEN`. Strategies relying on `SAME_BAR_CLOSE` (e.g., Overnight Returns) must set it explicitly; the validator emits a warning when `SAME_BAR_CLOSE` is used.

### 2.5 OMS idempotency + halt assertion

**Mandatory `client_order_id` at intent creation.** `TradeIntent` gains `client_order_id: str` (UUID4 or BLAKE2b-derived from intent fields). `OrderManager.submit()` rejects intents with empty / null IDs.

**Halt assertion.** `oms.submit()` and `engine._place_intent()` both call `halt.assert_not_halted()` as the first action. No bypass paths.

**Two-phase persist.**

```python
def _place_intent(self, intent):
    halt.assert_not_halted()
    # 1. Persist intent + client_order_id BEFORE sending
    self._store.log_intent_pre_submit(intent)
    try:
        result = self.broker.place_order(intent.to_order_request())
    except Exception:
        self._store.log_intent_failed_submit(intent, exc)
        raise
    # 2. Update with broker order_id
    self._store.log_intent_post_submit(intent, result)
    return result
```

On startup, `OMS.reconcile_pending()` queries `intents` where `state = pre_submit` and pings IBKR by `client_order_id` (orderRef) to recover state.

### 2.6 Risk state persistence

`RiskManager.__init__(..., db_path: str | None = None)`:

- On init: `_load_state()` from `risk_state` table (single row, id=1).
- On each mutation (`_orders_today`, `_session_start_net_liq`): write back inside the same transaction as the order-log write.
- Daily reset is now date-driven, not memory-driven.

### 2.7 Validation harness v2

**Fix the bugs:**

- `pbo.py:PBOCalculator.calculate_multi_strategy()` lines 250–264: replace logit-mean with canonical `pbo = mean(oos_ranks > N/2)`.
- `backtest_validator.py:_deflated_sharpe()` lines 277–312: delegate to `DeflatedSharpe.calculate()` (the correct one with skew/kurtosis).
- `orchestrator.py:validator.validate(n_trials=1)`: replace with `n_trials = effective_param_count` from the parameter grid.

**Add stationary bootstrap.** New module `quant_core/validation/stationary_bootstrap.py`:

```python
def stationary_bootstrap(
    x: np.ndarray,
    p: float | None = None,           # None = automatic Politis-White
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> np.ndarray: ...

def bootstrap_sharpe_ci(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
) -> tuple[float, float, float]: ...
```

Use `arch.bootstrap.StationaryBootstrap` if available; fall back to native NumPy implementation.

**Report card.** New `quant_core/validation/report_card.py`:

- Inputs: returns, trial grid (N×T), hypothesis YAML, sample period.
- Computes: PBO (CSCV), DSR, CPCV path Sharpe distribution, stationary-bootstrap 95% CI, MinTRL, walk-forward 12m rolling Sharpe.
- Pass gates:

| Metric | Pass |
|---|---|
| PBO | < 0.5 |
| DSR | > 0.95 |
| Lower 95% CI on annualized SR | > 0.3 |
| CPCV mean OOS SR | > 0.5 |
| MinTRL | ≤ available track record |
| Walk-forward 12m SR > 0 | ≥ 75% of windows |

- Output: markdown report card written to `validation_reports/<strategy>_<date>.md`.
- Wire into `orchestrator.py` `BacktestEngine.run()` finale.

---

## 3. File-by-file change matrix

### 3.1 New files

| Path | Purpose |
|---|---|
| `trading_algo/data/__init__.py` | Module init |
| `trading_algo/data/pit_store.py` | Bitemporal store: SQLite metadata + DuckDB-over-parquet bars |
| `trading_algo/data/schema.sql` | Canonical DDL |
| `trading_algo/data/universe.py` | UniverseResolver |
| `trading_algo/data/corporate_actions.py` | Adjustment factor computation |
| `trading_algo/data/migration.py` | One-shot import from existing JSON cache |
| `trading_algo/backtest_v2/cost_model.py` | Corwin-Schultz, sqrt impact, borrow, IBKR commission |
| `trading_algo/backtest_v2/execution_policy.py` | ExecutionPolicy enum + pending-order machinery |
| `trading_algo/quant_core/validation/stationary_bootstrap.py` | Politis-Romano + Politis-White |
| `trading_algo/quant_core/validation/report_card.py` | Markdown report-card generator |
| `tests/test_pit_store.py` | Bitemporal store tests |
| `tests/test_universe.py` | UniverseResolver tests |
| `tests/test_cost_model.py` | Spread/impact/borrow tests vs published values |
| `tests/test_execution_policy.py` | Look-ahead protection tests |
| `tests/test_oms_idempotency.py` | Mandatory client_order_id + halt assertion tests |
| `tests/test_risk_persistence.py` | Risk state survives restart tests |
| `tests/test_pbo_correctness.py` | CSCV against published reference values |
| `tests/test_stationary_bootstrap.py` | vs `arch.bootstrap` reference |

### 3.2 Modified files

| Path | Lines | Change |
|---|---|---|
| `trading_algo/oms.py` | 94–155 | Mandatory client_order_id; halt assertion; two-phase persist |
| `trading_algo/engine.py` | 88–135 | halt assertion in `_place_intent`; persist before submit |
| `trading_algo/risk.py` | 62–185 | SQLite-backed `_load_state` / `_save_state` |
| `trading_algo/backtest_v2/engine.py` | 156–354, 376–384, 497–519 | Wire ExecutionPolicy + cost_model + borrow accrual |
| `trading_algo/backtest_v2/models.py` | 25–35, 449–460 | Add adj_close/adjustment_factor to Bar; add execution_policy to BacktestConfig |
| `trading_algo/backtest_v2/data_provider.py` | 100–260, 387–511 | Read from pit_store first; fall back to existing path |
| `trading_algo/orders.py` | TradeIntent dataclass | Add `client_order_id: str` field; default factory `uuid4().hex` |
| `trading_algo/quant_core/validation/pbo.py` | 250–264 | Canonical PBO formula |
| `trading_algo/quant_core/validation/backtest_validator.py` | 277–312 | Delegate to DeflatedSharpe |
| `trading_algo/quant_core/engine/orchestrator.py` | ~160 | n_trials from grid; emit report card |

### 3.3 Strategies — no functional changes in Phase 1

Strategies are read-only in Phase 1. They will be evaluated and culled in Phase 2 using the validator we built here.

---

## 4. Library + dependency choices

| Concern | Library | Reason |
|---|---|---|
| Columnar query | DuckDB (already in pyarrow ecosystem) | Free, fast, zero-server |
| Parquet | pyarrow ≥15 (already in requirements.txt) | Standard |
| SQLite | stdlib | Already used (persistence.py) |
| Stationary bootstrap | `arch>=6.0` (NEW DEPENDENCY) | Sheppard-grade implementation |
| CSCV PBO | mlfinlab? No — too heavy. Roll our own (~80 lines). | Reduce dep surface |
| Trading calendar | `pandas_market_calendars>=4` (NEW) | Standard for US/CME |
| Norgate connector | optional, gated behind feature flag | User subscribes separately |
| YAML for hypothesis specs | `pyyaml` (NEW) | Pre-registration files |

Add to `requirements.txt`:

```
arch>=6.0
duckdb>=0.10
pandas_market_calendars>=4.0
pyyaml>=6.0
```

---

## 5. Test strategy

**Existing tests must not regress.** Run `pytest` after each commit. The 84 existing tests pass on `main` and must still pass at every PR-ready checkpoint on this branch.

**New tests, by module:**

- **PIT store:** insert + as-of query returns expected vintage; restatement appends row not overwrites; partition pruning by symbol/year works.
- **UniverseResolver:** `get_universe('SP500', date(2008,9,15))` excludes LEH (delisted); `get_universe('SP500', date(2020,3,2))` includes XOM (still in index).
- **Corporate actions:** AAPL 4:1 split 2020-08-31, adjustment factor for query date 2019-01-01 = 4.0; for query date 2021-01-01 = 1.0.
- **Cost model:** Corwin-Schultz on 2024-01-02 SPY OHLC = published value within 0.5 bps; sqrt impact for 5% ADV at σ=20% daily-bps = 22 bps.
- **Execution policy:** signal on bar T fills at bar T+1's open; same-bar-close mode produces strictly higher Sharpe (overfitting check).
- **OMS idempotency:** intent without client_order_id raises; halt sentinel set + submit raises HaltActive.
- **Risk persistence:** set orders_today=42, restart RiskManager, get back 42 (same date); next day get 0.
- **PBO correctness:** synthetic random trial grid (true SR=0) returns PBO ≈ 0.5 ± 0.05; deterministic-best-equals-best-OOS grid returns PBO ≈ 0.0.
- **Stationary bootstrap:** Sharpe 95% CI on AR(1) returns wider than IID bootstrap; Politis-White optimal `p` matches `arch.bootstrap.optimal_block_length`.

Coverage target on new modules: ≥ 90%.

---

## 6. Acceptance criteria for PR-to-main

The PR is ready to merge when **all** of the following are true:

1. **All 84 existing tests pass.**
2. **All new tests pass.** Coverage ≥ 90% on new modules.
3. **No `import trading_algo` raises** — package imports cleanly on a fresh venv.
4. **PIT store fully migrated.** Migration script runs against the existing JSON cache, produces a populated parquet/SQLite store, validates against the original (price equality on overlapping dates).
5. **Cost stack regression.** A reference strategy (Short-Term Reversal on a fixed PIT universe, 2018-01 to 2024-12) runs end-to-end with the new cost stack. Result documented in `validation_reports/short_term_reversal_baseline.md` with full report card. Sharpe expected to fall 30–50% vs the legacy backtest. **This is success, not failure.**
6. **OMS hardening exercised.** Integration test: kill the process between `place_order` and persist, restart, verify reconcile finds and adopts the open IBKR order via `client_order_id` (orderRef).
7. **Halt assertion enforced.** Test that touches `data/HALTED` and submits an intent → raises `HaltActive`. Tested at both `oms.submit()` and `engine._place_intent()` levels.
8. **Validator wired.** Running `python -m trading_algo backtest --strategy short_term_reversal --start 2018-01-01 --end 2024-12-31 --validate` produces a report card on disk with all seven metrics.
9. **No silent fallback to forward-adjusted prices.** Code that previously assumed adjusted prices either receives `adj_close` from the new layer or fails with a clear error.
10. **Documentation:** `CLAUDE.md` updated with new commands and rules. `CHANGELOG.md` entry titled "Wave T5: enterprise foundation (PIT data, cost realism, validation)".

---

## 7. Out of scope for Phase 1

Explicitly deferred to later phases:

- Norgate / Sharadar / WRDS integration (Phase 2 — requires user subscription decisions)
- Tick-level data (not needed for current strategies)
- Order book L2/L3 simulation (overkill for current strategies)
- Agent-based fill simulation (overkill until size > 1% ADV)
- Auction matching engine for MOC (deferred until Mahimn trades MOC)
- Crypto venue adapters beyond IBKR (Phase 3 — needed for funding-carry edge)
- Strategy refactors / new strategies (Phase 3)
- Decision tree pruning / "kill the dead wood" (Phase 2 — uses validator built here)

---

## 8. Phase 2–4 outline (forward-looking, not in this PR)

**Phase 2: The cull.** Run the new validator against every existing strategy. Apply the precommit thresholds (PBO < 0.5, DSR > 0.95, walk-forward Sharpe > 0.4, cost-adjusted Sharpe > 0.3). Delete failures; write postmortems in `docs/postmortems/`. Expected survivors: Short-Term Reversal, Overnight Returns, Pure Momentum (with brutal drawdown documented), Cross-Asset Divergence, Flow Pressure (probably). Expected casualties: Lead-Lag, Volatility Maximizer, Regime Transition, Hurst-Adaptive, plus most options strategies until backtested with real chain data.

**Phase 3: Build edges.** Per the research:
1. XSP delta-hedged short strangles 30-DTE with 5-delta wings
2. Crypto BTC/ETH funding-rate carry, delta-neutral, Hyperliquid + Coinbase
3. 8-K LLM rapid classifier + execution loop on liquid mid-caps
4. Russell 2000 reconstitution May–July seasonal

Each edge gets its own feature branch, hypothesis.yaml pre-registered, full report card before deployment. Paper-trade for 60 days minimum before any real money.

**Phase 4: Process.** Research log automation, pre-registration enforcement (CI-gated), CUSUM live-Sharpe decay detection, factor-stripped accounting (Fama-French 5 + Carhart momentum residuals), shadow-trading harness comparing paper P&L vs backtest predictions on live dates.

---

## 9. Risk register for Phase 1

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bitemporal store underestimates corner cases (e.g. spin-off cost-basis) | Medium | Medium | Tests against known historical events (GE/GEHC, ABBV/ABT) |
| Migration of existing JSON cache loses fidelity | Low | High | Cross-check 100% of overlapping `(symbol, date, close)` tuples |
| Mandatory `client_order_id` breaks existing CLI tools | Medium | Low | Default factory provides UUID if caller didn't set one; just enforce non-empty |
| Risk-state persistence introduces lock contention | Low | Medium | SQLite WAL + single-row UPSERT; benchmark before merge |
| Existing tests assume fixed-bps slippage and break on new cost model | High | Low | Tests pin `BacktestConfig(execution_policy=SAME_BAR_CLOSE, cost_model=LegacyFlatCost)` for reproducibility; new tests use realistic stack |
| `arch` dependency fails on user's Python 3.11 | Low | Low | Pin to known-good version; native fallback in stationary_bootstrap |

---

## 10. Commit plan

This branch will be a sequence of focused commits, each independently reviewable:

1. `Add PLAN.md`
2. `Add data layer skeleton: pit_store + schema + universe + corporate_actions`
3. `Add bitemporal SQLite + parquet writer/reader, with tests`
4. `Add UniverseResolver with index_membership lookups, with tests`
5. `Add corporate_actions adjustment factor computation, with tests`
6. `Add cost_model: Corwin-Schultz + Abdi-Ranaldo + sqrt impact + borrow + IBKR tiers`
7. `Add ExecutionPolicy + pending-order machinery; default NEXT_BAR_OPEN`
8. `Wire cost_model + execution_policy into BacktestEngine`
9. `Mandatory client_order_id in TradeIntent + OMS rejection of empty IDs`
10. `Halt assertion at oms.submit and engine._place_intent`
11. `Two-phase persist for orders (pre-submit log → broker call → post-submit update)`
12. `Persistent RiskManager state via SQLite risk_state table`
13. `Fix CSCV PBO logit thresholding (canonical formula)`
14. `Fix DeflatedSharpe in BacktestValidator (delegate to DSR class)`
15. `Add stationary_bootstrap with Politis-Romano + Politis-White`
16. `Add report_card.py: markdown output, seven-gate decision`
17. `Wire validator into orchestrator with n_trials from parameter grid`
18. `Migration script: legacy JSON cache → pit_store, with reconciliation report`
19. `Reference baseline: Short-Term Reversal full report card`
20. `Update CLAUDE.md + CHANGELOG.md`

Each commit references this PLAN.md by section. PR description summarizes all 20 in one ToC.

---

## 11. The standard

This is the foundation of a serious quant system. We are not shipping anything that:

- Silently uses forward-adjusted prices
- Submits an order without an idempotency key
- Resets a risk counter on process restart
- Fills a backtest at the same bar's close as the signal
- Pretends shorts are free
- Reports an in-sample Sharpe without a deflated counterpart

Every change in this branch eliminates one of those failure modes. When the PR merges, trading-algo can produce a backtest result that is *defensible* — meaning if a serious quant audited it, they would not find an obvious flaw in the methodology. That is the bar.

---

*End PLAN.md.*

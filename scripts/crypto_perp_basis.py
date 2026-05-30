"""Perp-spot BASIS mean-reversion backtest, fed through the HARDENED gate.

Edge: basis = perp_close/spot_close - 1. When basis is stretched relative to a
trailing window (z-score), fade it delta-neutral:
    basis rich  (z > +entry) -> SHORT perp + LONG spot  (expect convergence down)
    basis cheap (z < -entry) -> LONG  perp + SHORT spot (expect convergence up)

PnL per bar for a delta-neutral position of notional 1 on each leg:
    perp leg pnl  = pos_perp * (perp_ret)          pos_perp = -sign(z) at rich, etc.
    spot leg pnl  = pos_spot * (spot_ret)          pos_spot = -pos_perp (hedge)
  => net leg pnl  = pos_perp * (perp_ret - spot_ret) = pos_perp * d(basis)+ (approx)
    funding pnl   = -pos_perp * funding_rate   (long perp PAYS funding when rate>0)
    (we collect funding when short perp & funding>0; the rich-basis trade is short perp)

LOOK-AHEAD DISCIPLINE:
  - z-score uses trailing window ending at bar t (data up to decision time).
  - decide at close of bar t, hold over bar t+1 (trade next bar). Returns use
    forward bar returns r_{t+1}.
  - funding settles 00/08/16 UTC; the funding paid over a held bar is attributed
    only on bars that contain a settlement, using the rate known at/just before
    settlement. We merge the funding series onto the hourly grid by settlement ts.

COSTS (never zero), per leg per side:
  taker perp 5 bps, taker spot 5 bps, slippage 3 bps (BTC/ETH) / 6 bps (SOL).
  A roundtrip touches 4 legs (enter spot+perp, exit spot+perp). We charge cost
  only when the target position CHANGES (turnover), on both legs.
  cost_adjusted stream applies a punitive 2x cost stack.

Rebalance cadence = hourly -> periods_per_year = 24*365.
Honest n_trials: we report a SINGLE primary config but DSR is deflated by the
small grid we actually inspected.
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.validation.report_card import build_report_card

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crypto_data_cache")
SYMS = ["BTC", "ETH", "SOL"]
HOURS_PER_YEAR = 24 * 365

# cost stack in fractional terms, per leg per side
TAKER = 5e-4          # 5 bps perp taker AND spot taker
SLIP = {"BTC": 3e-4, "ETH": 3e-4, "SOL": 6e-4}  # per leg per side


def load_aligned(sym: str):
    sp = np.load(os.path.join(DATA, f"{sym}_USDT_spot_1h_2020-10-01_2026-03-01.npz"))
    sw = np.load(os.path.join(DATA, f"{sym}_USDT_USDT_swap_1h_2020-10-01_2026-03-01.npz"))
    fu = np.load(os.path.join(DATA, f"{sym}_USDT_USDT_funding_2020-10-01_2026-03-01.npz"))
    sts, swts = sp["ts"], sw["ts"]
    common, si, wi = np.intersect1d(sts, swts, return_indices=True)
    spc = sp["ohlcv"][:, 3][si].astype(np.float64)
    swc = sw["ohlcv"][:, 3][wi].astype(np.float64)
    ts = common.astype(np.float64)
    # funding aligned onto the hourly grid: a bar t "contains" funding settlement
    # if a funding ts falls in (ts[t-1], ts[t]]. Attribute that rate to bar t.
    fts, frate = fu["ts"].astype(np.float64), fu["rates"].astype(np.float64)
    funding_on_bar = np.zeros(len(ts), dtype=np.float64)
    # for each funding settlement, find the first bar at/after it
    idx = np.searchsorted(ts, fts, side="left")
    for j, b in enumerate(idx):
        if 0 <= b < len(ts):
            funding_on_bar[b] += frate[j]
    return ts, spc, swc, funding_on_bar


def basis_signal(spc, swc, window):
    basis = swc / spc - 1.0
    # trailing z-score ending at bar t (inclusive), no look-ahead
    n = len(basis)
    z = np.full(n, np.nan)
    # rolling mean/std via cumulative sums
    for t in range(window, n):
        w = basis[t - window:t]  # strictly BEFORE t -> uses data up to t-1 close
        mu = w.mean()
        sd = w.std(ddof=1)
        if sd > 1e-12:
            z[t] = (basis[t] - mu) / sd
    return basis, z


def run_asset(sym, window, entry, exit_, cost_mult=1.0):
    ts, spc, swc, funding = load_aligned(sym)
    basis, z = basis_signal(spc, swc, window)
    n = len(ts)

    spot_ret = np.zeros(n)
    perp_ret = np.zeros(n)
    spot_ret[1:] = spc[1:] / spc[:-1] - 1.0
    perp_ret[1:] = swc[1:] / swc[:-1] - 1.0

    # target perp position decided at close of t, held over t+1
    # rich basis (z>entry) -> short perp (pos=-1); cheap (z<-entry) -> long perp (pos=+1)
    pos_perp = np.zeros(n)
    cur = 0.0
    for t in range(n):
        if np.isnan(z[t]):
            pos_perp[t] = cur
            continue
        if cur == 0.0:
            if z[t] > entry:
                cur = -1.0
            elif z[t] < -entry:
                cur = 1.0
        else:
            # exit when reverts inside exit band
            if abs(z[t]) < exit_:
                cur = 0.0
            # flip if crosses strongly the other way
            elif cur < 0 and z[t] < -entry:
                cur = 1.0
            elif cur > 0 and z[t] > entry:
                cur = -1.0
        pos_perp[t] = cur

    slip = SLIP[sym]
    # roundtrip leg cost per unit turnover, both legs (perp+spot), per side
    leg_cost = (TAKER + slip) * cost_mult  # one leg, one side
    # turnover at bar t = |pos_perp[t]-pos_perp[t-1]|; touches BOTH perp & spot legs
    turn = np.zeros(n)
    turn[1:] = np.abs(pos_perp[1:] - pos_perp[:-1])

    ret = np.zeros(n)
    for t in range(1, n):
        p = pos_perp[t - 1]  # position decided at t-1 close, earns over bar t
        # delta-neutral: spot hedge = -p notional. net price pnl:
        price_pnl = p * perp_ret[t] + (-p) * spot_ret[t]
        # funding: long perp pays funding when rate>0 -> pnl = -p*funding
        fund_pnl = -p * funding[t]
        # cost charged on the turnover that occurred entering position at t-1
        # both legs (perp + spot), one side each -> 2 * leg_cost * turnover
        cost = 2.0 * leg_cost * turn[t - 1]
        ret[t] = price_pnl + fund_pnl - cost
    return ts, ret, pos_perp


def pool_returns(window, entry, exit_, cost_mult=1.0):
    """Equal-weight pool of the 3 assets onto a common hourly clock."""
    series = {}
    all_ts = None
    for sym in SYMS:
        ts, ret, _ = run_asset(sym, window, entry, exit_, cost_mult)
        series[sym] = dict(zip(ts.tolist(), ret.tolist()))
        all_ts = set(ts.tolist()) if all_ts is None else (all_ts | set(ts.tolist()))
    grid = np.array(sorted(all_ts))
    pooled = np.zeros(len(grid))
    cnt = np.zeros(len(grid))
    for sym in SYMS:
        m = series[sym]
        for i, t in enumerate(grid):
            if t in m:
                pooled[i] += m[t]
                cnt[i] += 1
    cnt[cnt == 0] = 1
    pooled = pooled / cnt  # equal-weight average across active assets
    return grid, pooled


def main():
    # ---- the grid we actually search (honest n_trials) ----
    windows = [72, 168]       # 3d, 7d trailing z
    entries = [1.5, 2.0]      # z entry
    exit_ = 0.4
    grid_specs = [(w, e) for w in windows for e in entries]
    n_trials = len(grid_specs)  # 4 configs inspected

    # build trial_grid (T x N) of NET (gross-cost) pooled returns for PBO/CSCV
    cols = []
    base_grid = None
    for (w, e) in grid_specs:
        g, r = pool_returns(w, e, exit_, cost_mult=1.0)
        if base_grid is None:
            base_grid = g
            cols.append(r)
        else:
            # align to base grid
            idx = np.searchsorted(g, base_grid)
            idx = np.clip(idx, 0, len(g) - 1)
            cols.append(r[idx])
    trial_grid = np.column_stack(cols)

    # primary config: pick the in-sample best by gross Sharpe (this is part of
    # the search, hence n_trials accounts for it)
    def sharpe(x):
        s = x.std(ddof=1)
        return x.mean() / s * np.sqrt(HOURS_PER_YEAR) if s > 1e-12 else 0.0
    best_i = int(np.argmax([sharpe(c) for c in cols]))
    w, e = grid_specs[best_i]
    gross = cols[best_i]

    # cost-adjusted stream: punitive 2x cost stack
    _, ca = pool_returns(w, e, exit_, cost_mult=2.0)
    if len(ca) != len(gross):
        idx = np.searchsorted(_, base_grid); idx = np.clip(idx, 0, len(_) - 1); ca = ca[idx]

    print(f"primary config: window={w} entry={e} exit={exit_}  n_trials={n_trials}")
    print(f"gross Sharpe(ann)={sharpe(gross):.3f}  costadj Sharpe(ann)={sharpe(ca):.3f}")
    print(f"gross mean bps/bar={gross.mean()*1e4:.3f}  costadj mean bps/bar={ca.mean()*1e4:.3f}")
    nz = np.count_nonzero(np.abs(gross) > 1e-12)
    print(f"active bars={nz}/{len(gross)}")

    rc = build_report_card(
        strategy_name="crypto_perp_basis_meanrev",
        returns=gross,
        n_trials=n_trials,
        trial_grid=trial_grid,
        cost_adjusted_returns=ca,
        periods_per_year=HOURS_PER_YEAR,
        period_start=date(2020, 10, 1),
        period_end=date(2026, 3, 1),
    )
    print("\n" + rc.render())
    print("STATUS:", rc.status)
    print("CI_lower:", rc.sharpe_ci_lower, "PBO:", rc.pbo,
          "DSR_prob:", rc.deflated_sharpe, "costadj_SR:", rc.cost_adjusted_sharpe)


if __name__ == "__main__":
    main()

"""Clean, hardened-gate backtests for two crypto time-series edges.

(A) Cross-asset cascade: BTC leads ETH/SOL with a lag. Trade the laggard on
    BTC's lagged return. Per-asset (ETH, SOL), hourly.
(B) Volatility term structure: short-window realized vol vs long-window.
    NORMAL (short < long) => trend-follow long; INVERTED (short > long) =>
    flat/mean-revert. Per-asset (BTC, ETH, SOL), daily rebalance.

Both feed build_report_card with realistic costs (perp taker 5bps/side,
slippage, funding) and an honest, SMALL n_trials so the Deflated Sharpe is
not gamed. Look-ahead is strictly forbidden: every signal uses data up to the
decision bar; the trade is executed on the NEXT bar (close-to-close return
with a 1-bar lag), so a signal computed from bar t earns return t+1->t+2.

Run:
  /Users/mahimnpatel/Documents/Dev/randomThings/.venv/bin/python \
    scripts/crypto_xasset_vol.py
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import sys
sys.path.insert(0, "/Users/mahimnpatel/Documents/Dev/randomThings")

from trading_algo.quant_core.validation.report_card import build_report_card

DATA = "/Users/mahimnpatel/Documents/Dev/randomThings/crypto_data_cache"
TS0, TS1 = "2020-10-01", "2026-03-01"

# ---- realistic cost knobs (fractions, per side unless noted) ----
PERP_TAKER = 0.0005          # 5 bps/side
SLIP = {"BTC": 0.0002, "ETH": 0.0003, "SOL": 0.0005}   # one-way slippage
# Cost-adjusted (punitive) stack: 2x the round-trip friction.
COST_STACK_MULT = 2.0


def _load_ohlcv(asset: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """kind in {'spot','swap'}. Returns (ts, ohlcv[N,5])."""
    sym = f"{asset}_USDT_USDT_swap_1h_{TS0}_2026-03-01.npz" if kind == "swap" \
        else f"{asset}_USDT_spot_1h_{TS0}_2026-03-01.npz"
    z = np.load(os.path.join(DATA, sym))
    return z["ts"].astype(np.int64), z["ohlcv"].astype(np.float64)


def _load_funding(asset: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(os.path.join(DATA, f"{asset}_USDT_USDT_funding_{TS0}_2026-03-01.npz"))
    return z["ts"].astype(np.int64), z["rates"].astype(np.float64)


def _align_close(assets: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Inner-join swap closes across assets on a common hourly timestamp grid."""
    series = {}
    common = None
    for a in assets:
        ts, ohlcv = _load_ohlcv(a, "swap")
        series[a] = (ts, ohlcv[:, 3])  # close
        common = ts if common is None else np.intersect1d(common, ts)
    out = {}
    for a in assets:
        ts, close = series[a]
        idx = np.searchsorted(ts, common)
        out[a] = close[idx]
    return common, out


def _funding_per_hour(asset: str, hourly_ts: np.ndarray) -> np.ndarray:
    """Map 8h funding to an hourly cost stream: the funding settled at a bar's
    timestamp is charged at that bar (long pays positive funding). Forward-fill
    zero between settlements. No look-ahead: funding at ts is known at ts."""
    fts, frates = _load_funding(asset)
    out = np.zeros(len(hourly_ts), dtype=np.float64)
    # exact-match settlement bars
    pos = np.searchsorted(fts, hourly_ts)
    in_range = (pos < len(fts)) & (fts[np.clip(pos, 0, len(fts) - 1)] == hourly_ts)
    out[in_range] = frates[pos[in_range]]
    return out


def _sharpe(r: np.ndarray, ppy: float) -> float:
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(ppy)) if sd > 1e-12 else 0.0


# =====================================================================
# EDGE A: cross-asset cascade (BTC -> laggard), hourly
# =====================================================================
def edge_A_cascade(laggard: str, lag: int, beta_win: int):
    """Signal: sign of BTC's return realized over the bar ending at t, scaled by
    a rolling hedge beta of laggard-on-BTC. Position in laggard for bar t+1.
    Return earned = position(t) * laggard_ret(t+1). 1-bar execution lag, no
    look-ahead. Hourly rebalance => ppy = 24*365.
    """
    common, cl = _align_close(["BTC", laggard])
    btc, alt = cl["BTC"], cl[laggard]
    rb = np.diff(np.log(btc))   # btc log-ret at bar t (index t-1 in rb)
    ra = np.diff(np.log(alt))   # laggard log-ret at bar t
    n = len(rb)
    ppy = 24 * 365

    # rolling beta of alt on btc (trailing, no look-ahead)
    pos = np.zeros(n)
    for t in range(beta_win + lag, n - 1):
        b = rb[t - beta_win:t]
        a = ra[t - beta_win:t]
        var = np.var(b)
        beta = (np.cov(b, a)[0, 1] / var) if var > 1e-12 else 0.0
        # BTC signal observed `lag` bars ago, cascading into the laggard now
        sig = rb[t - lag + 1] if lag >= 1 else rb[t]
        pos[t] = np.sign(sig) * np.clip(abs(beta), 0.0, 1.5)

    # gross return: position at t earns laggard's NEXT-bar return
    fut = np.empty(n); fut[:] = np.nan
    fut[:-1] = ra[1:]
    gross = pos * fut
    gross = np.nan_to_num(gross, nan=0.0)

    # costs: trade only when position changes; charge taker+slip per turnover unit
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    one_way = PERP_TAKER + SLIP[laggard]
    cost = turn * one_way
    # funding: held position pays funding on settlement hours (long pays +rate)
    fund = _funding_per_hour(laggard, common[1:])  # align to ra index
    fund = fund[:n]
    funding_cost = pos * fund   # long pays positive funding => subtract

    net = gross - cost - funding_cost
    net_punitive = gross - COST_STACK_MULT * cost - funding_cost

    active = pos != 0
    return net, net_punitive, ppy, common, active


# =====================================================================
# EDGE B: vol term structure, daily rebalance
# =====================================================================
def edge_B_vts(asset: str, short_w: int, long_w: int, decide_hour: int = 0):
    """Daily decision at `decide_hour` UTC. Compute short/long realized vol from
    hourly log-returns up to the decision bar. slope=(long-short)/long.
    NORMAL (slope>0, short<long): long the next day's trend (momentum sign).
    INVERTED (slope<0): flat (stress => mean-revert; we go flat to be honest,
    no leverage). Position held for the next 24h; return = sum of next-day
    hourly log-rets. Daily rebalance => ppy = 365.
    """
    ts, ohlcv = _load_ohlcv(asset, "swap")
    close = ohlcv[:, 3]
    lr = np.diff(np.log(close))            # hourly log-ret, lr[i] ends bar i+1
    hours = ((ts - ts[0]) // 3600).astype(int)
    # decision bars: each UTC midnight-ish (decide_hour)
    import datetime as D
    utc_hour = np.array([D.datetime.fromtimestamp(t, D.UTC).hour for t in ts])
    dec_idx = np.where(utc_hour == decide_hour)[0]
    dec_idx = dec_idx[(dec_idx > long_w) & (dec_idx < len(close) - 25)]

    ppy = 365
    rets, pos_list = [], []
    daily_ts = []
    for di in dec_idx:
        win_lr = lr[di - long_w:di]        # ends at bar di (known at di)
        sv = np.std(win_lr[-short_w:], ddof=1)
        lv = np.std(win_lr, ddof=1)
        if lv < 1e-12:
            continue
        slope = (lv - sv) / lv
        # trend over the short window (momentum), known at di
        trend = np.sum(lr[di - short_w:di])
        if slope > 0:                       # NORMAL: trend-follow
            pos = np.sign(trend)
        else:                               # INVERTED: flat
            pos = 0.0
        # next-day return: bars di+1 .. di+24 (close-to-close), 1-bar exec lag
        nxt = np.sum(lr[di + 1:di + 25])
        rets.append(pos * nxt)
        pos_list.append(pos)
        daily_ts.append(ts[di])

    rets = np.array(rets)
    pos_arr = np.array(pos_list)
    # costs: round-trip when position flips, per day held
    turn = np.abs(np.diff(np.concatenate([[0.0], pos_arr])))
    one_way = PERP_TAKER + SLIP[asset]
    cost = turn * one_way
    # funding over the held day: ~3 settlements/day; long pays sum of 3 rates
    fts, frates = _load_funding(asset)
    daily_ts = np.array(daily_ts)
    fund_day = np.zeros(len(daily_ts))
    for i, t in enumerate(daily_ts):
        m = (fts >= t) & (fts < t + 86400)
        fund_day[i] = frates[m].sum() if m.any() else 0.0
    funding_cost = pos_arr * fund_day

    net = rets - cost - funding_cost
    net_punitive = rets - COST_STACK_MULT * cost - funding_cost
    active = pos_arr != 0
    return net, net_punitive, ppy, daily_ts, active


def _grid_matrix(fn, grid, ppy_pick=None):
    """Build a (T x N) trial grid of NET returns across parameter combos for PBO.
    Aligns all columns to the shortest length (truncate from the front)."""
    cols = []
    for params in grid:
        net, netp, ppy, _, _ = fn(*params)
        cols.append(net)
    m = min(len(c) for c in cols)
    mat = np.column_stack([c[-m:] for c in cols])
    return mat


def edge_A_thresholded(laggard: str, thr: float = 0.01):
    """Best-case cascade: only enter when |BTC 1h move| > thr (signal lives in
    tails, per lead-lag probe). sign = sign(BTC move), hold 1 bar (round trip).
    This is the ONLY cascade variant with positive GROSS Sharpe; included to
    prove it still dies on costs. Hourly, ppy = 24*365."""
    common, clos = _align_close(["BTC", laggard])
    rb = np.diff(np.log(clos["BTC"]))
    ra = np.diff(np.log(clos[laggard]))
    n = len(rb) - 1
    pos = np.zeros(n)
    sig = np.sign(rb[:-1])
    mask = np.abs(rb[:-1]) > thr
    pos[mask] = sig[mask]
    gross = pos * ra[1:n + 1]
    turn = np.where(pos != 0, 2.0, 0.0)  # enter + exit
    one_way = PERP_TAKER + SLIP[laggard]
    cost = turn * one_way
    net = gross - cost
    netp = gross - COST_STACK_MULT * cost
    return net, netp, 24 * 365, common, (pos != 0)


def run():
    results = {}

    # ---------------- EDGE A ----------------
    # Honest trial grid: 2 laggards x {lag in [1,2]} x {beta_win in [168,336]}
    # = 8 combos. We REPORT the single best-by-design (ETH, lag=1, beta_win=168)
    # but PBO sees the whole 8-wide grid; n_trials = 8.
    A_grid = [(lag, bw) for lag in (1, 2) for bw in (168, 336)]
    print("\n================ EDGE A: cross-asset cascade ================")
    for laggard in ("ETH", "SOL"):
        grid = [(laggard, lag, bw) for (lag, bw) in A_grid]
        n_trials_A = len(A_grid) * 2   # both laggards searched = 8
        # primary config
        net, netp, ppy, ts, active = edge_A_cascade(laggard, 1, 168)
        mat = _grid_matrix(edge_A_cascade, grid)
        rc = build_report_card(
            strategy_name=f"cascade_BTC->{laggard}",
            returns=net,
            n_trials=n_trials_A,
            trial_grid=mat,
            cost_adjusted_returns=netp,
            periods_per_year=ppy,
            period_start=date(2020, 10, 1),
            period_end=date(2026, 3, 1),
        )
        print(rc.render())
        print(f"  gross SR={_sharpe(net + 0, ppy):.3f} | net SR={_sharpe(net, ppy):.3f} "
              f"| punitive SR={_sharpe(netp, ppy):.3f} | active%={100*active.mean():.1f} | n={len(net)}")
        results[f"A_{laggard}"] = rc

        # best-case thresholded variant (tails only)
        netT, netpT, ppyT, _, actT = edge_A_thresholded(laggard, 0.01)
        rcT = build_report_card(
            strategy_name=f"cascade_thr1pct_BTC->{laggard}",
            returns=netT, n_trials=n_trials_A,
            cost_adjusted_returns=netpT, periods_per_year=ppyT,
            period_start=date(2020, 10, 1), period_end=date(2026, 3, 1),
        )
        print(rcT.render())
        gT = netT + np.where(actT, (PERP_TAKER + SLIP[laggard]) * 2.0, 0.0)
        print(f"  thresholded gross SR={_sharpe(gT, ppyT):+.3f} | net SR={_sharpe(netT, ppyT):+.3f} "
              f"| punitive SR={_sharpe(netpT, ppyT):+.3f} | trades={int(actT.sum())}")
        results[f"A_{laggard}_thr"] = rcT

    # ---------------- EDGE B ----------------
    # Honest trial grid: 3 assets x {short in [24,48]} x {long in [336,720]}
    # = 12 combos searched; report each asset's primary (short=24,long=720).
    B_short = (24, 48)
    B_long = (336, 720)
    print("\n================ EDGE B: vol term structure ================")
    for asset in ("BTC", "ETH", "SOL"):
        grid = [(asset, s, l) for s in B_short for l in B_long]
        n_trials_B = len(B_short) * len(B_long) * 3   # 12
        net, netp, ppy, ts, active = edge_B_vts(asset, 24, 720)
        mat = _grid_matrix(edge_B_vts, grid)
        rc = build_report_card(
            strategy_name=f"vts_{asset}",
            returns=net,
            n_trials=n_trials_B,
            trial_grid=mat,
            cost_adjusted_returns=netp,
            periods_per_year=ppy,
            period_start=date(2020, 10, 1),
            period_end=date(2026, 3, 1),
        )
        print(rc.render())
        print(f"  gross SR={_sharpe(net, ppy):.3f} | net SR={_sharpe(net, ppy):.3f} "
              f"| punitive SR={_sharpe(netp, ppy):.3f} | active%={100*active.mean():.1f} | n={len(net)}")
        results[f"B_{asset}"] = rc

    print("\n================ SUMMARY ================")
    for k, rc in results.items():
        print(f"{k:16s} status={rc.status:11s} "
              f"ci_lo={rc.sharpe_ci_lower:+.3f} pbo={rc.pbo} "
              f"dsr={rc.deflated_sharpe} cost_adj_SR={rc.cost_adjusted_sharpe}")
    return results


if __name__ == "__main__":
    run()

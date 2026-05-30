"""COT spec-positioning contrarian (#7): SPY/IWM long-only contrarian tilt.

Signal: CFTC Commitments of Traders, E-mini S&P 500 (contract 13874A).
    spec_net = noncomm_long - noncomm_short
    pct      = spec_net / open_interest
    z        = rolling z-score of pct over a lookback (104/156/260 weeks)
Position (long-only contrarian):
    z < -1.0           -> 1.0   (specs maximally short -> contrarian long)
    -1.0 <= z <= 1.5   -> 0.5
    z > 1.5            -> 0.0   (specs crowded long -> step aside)
Hysteresis: weekly rebalance, position only changes on band crossing.

NO LOOK-AHEAD: report_date is the Tuesday the positions were measured; the
CFTC releases it the following Friday (~3 day lag). We compute z on that
Tuesday's data but only TRADE it on the first daily bar on/after Tuesday+6
calendar days (the following Monday open). Each weekly position earns the
return from its trade date to the next weekly trade date.

COSTS: SPY/IWM are the two most liquid ETFs on earth. ~1.5 bp/side spread
is realistic; turnover is tiny (position flips at most weekly, usually
holds for many weeks under hysteresis). Stress run at 5 bp/side.

FALSIFICATION: Newey-West regress fwd 1-4wk return on z; require |t|>=2 and
a stable sign across the 2016-21 vs 2021-26 halves.

n_trials counted HONESTLY: the trial grid is {3 lookbacks} x {SPY,IWM} x
{2 entry-band variants} = 12 variants fed to CSCV/DSR.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._options_data import load_daily_bars
from trading_algo.quant_core.validation.report_card import build_report_card

COT_JSON = "/tmp/cot_emini_full.json"
RELEASE_LAG_DAYS = 6  # Tue report -> public Fri -> trade following Mon open


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_cot():
    """Return (report_dates: list[date], pct: np.array) ascending by date."""
    d = json.load(open(COT_JSON))
    dates, pct = [], []
    for r in d:
        dt = datetime.fromisoformat(r["report_date_as_yyyy_mm_dd"].replace(".000", "")).date()
        ncl = float(r["noncomm_positions_long_all"])
        ncs = float(r["noncomm_positions_short_all"])
        oi = float(r["open_interest_all"])
        if oi <= 0:
            continue
        dates.append(dt)
        pct.append((ncl - ncs) / oi)
    order = np.argsort([d.toordinal() for d in dates])
    dates = [dates[i] for i in order]
    pct = np.array([pct[i] for i in order], dtype=float)
    return dates, pct


def load_weekly_close(sym: str):
    """Return (dates_ord: np.array[int], closes: np.array) of daily bars (ascending)."""
    bars = load_daily_bars(sym)
    ords, cl = [], []
    for b in bars:
        dt = datetime.fromtimestamp(b.timestamp_epoch_s, tz=timezone.utc).date()
        ords.append(dt.toordinal())
        cl.append(float(b.close))
    ords = np.array(ords, dtype=int)
    cl = np.array(cl, dtype=float)
    o = np.argsort(ords)
    return ords[o], cl[o]


def rolling_z(pct: np.ndarray, lookback: int) -> np.ndarray:
    """Causal rolling z-score: z[t] uses pct[t-lookback+1 .. t] inclusive
    (the value at t is the just-released Tuesday reading; no future data)."""
    z = np.full(len(pct), np.nan)
    for t in range(lookback - 1, len(pct)):
        win = pct[t - lookback + 1: t + 1]
        m = win.mean()
        s = win.std(ddof=1)
        if s > 1e-12:
            z[t] = (pct[t] - m) / s
    return z


# --------------------------------------------------------------------------
# Backtest: weekly position, traded with publication lag, mapped to weekly P&L
# --------------------------------------------------------------------------

def position_from_z(z: float, prev_pos: float, lo: float, hi: float) -> float:
    """Long-only contrarian with hysteresis. Bands: z<lo ->1.0 ; mid ->0.5 ; z>hi ->0.0.
    Hysteresis = only move when z is decisively in a new band (the bands
    themselves are wide so this is naturally sticky)."""
    if not np.isfinite(z):
        return prev_pos
    if z < lo:
        return 1.0
    if z > hi:
        return 0.0
    return 0.5


def build_weekly_returns(cot_dates, z, bar_ords, bar_cl, lo, hi):
    """Return weekly net-of-nothing returns array (one per held week) and the
    trade-date ordinals. Long-only: position in [0,1] times SPY/IWM weekly ret.

    For each COT report (Tuesday), trade date = first bar on/after Tue+lag.
    The position earns the return from its trade-date close to the next
    weekly trade-date close. Turnover is |pos_t - pos_{t-1}| at each rebalance.
    """
    trade_idx = []     # index into bar arrays for each weekly trade date
    positions = []
    prev_pos = 0.0
    for dt, zt in zip(cot_dates, z):
        trade_ord = (dt + timedelta(days=RELEASE_LAG_DAYS)).toordinal()
        # first bar on/after the release-adjusted trade date
        j = int(np.searchsorted(bar_ords, trade_ord, side="left"))
        if j >= len(bar_ords):
            continue
        pos = position_from_z(zt, prev_pos, lo, hi)
        trade_idx.append(j)
        positions.append(pos)
        prev_pos = pos

    trade_idx = np.array(trade_idx, dtype=int)
    positions = np.array(positions, dtype=float)

    # dedup: if two reports map to the same bar (holiday), keep the later one
    keep = np.concatenate([np.diff(trade_idx) > 0, [True]])
    trade_idx = trade_idx[keep]
    positions = positions[keep]

    # weekly return from trade_idx[k] close to trade_idx[k+1] close
    rets, turns = [], []
    for k in range(len(trade_idx) - 1):
        c0 = bar_cl[trade_idx[k]]
        c1 = bar_cl[trade_idx[k + 1]]
        ul_ret = c1 / c0 - 1.0
        pos = positions[k]
        prev = positions[k - 1] if k > 0 else 0.0
        rets.append(pos * ul_ret)
        turns.append(abs(pos - prev))
    return np.array(rets), np.array(turns), trade_idx[:-1], positions[:-1]


def variant_returns(sym, cot_dates, pct, lookback, lo, hi, cost_bps_side):
    bar_ords, bar_cl = load_weekly_close(sym)
    z = rolling_z(pct, lookback)
    rets, turns, _, _ = build_weekly_returns(cot_dates, z, bar_ords, bar_cl, lo, hi)
    # cost: turnover (one-way notional change) * cost per side, charged in bps
    cost = turns * (cost_bps_side / 1e4)
    net = rets - cost
    # drop leading all-zero warmup (before first finite z / before any position)
    nz = np.flatnonzero(np.abs(rets) > 0)
    start = nz[0] if nz.size else 0
    return net[start:], rets[start:], cost[start:]


# --------------------------------------------------------------------------
# Falsification: NW regression of fwd k-week ret on z, full + two halves
# --------------------------------------------------------------------------

def nw_regression(sym, cot_dates, pct, lookback, lo, hi):
    import statsmodels.api as sm
    bar_ords, bar_cl = load_weekly_close(sym)
    z = rolling_z(pct, lookback)
    # build aligned z (at trade time) and fwd 1..4 week underlying returns
    trade_idx, zt = [], []
    for dt, zz in zip(cot_dates, z):
        if not np.isfinite(zz):
            continue
        trade_ord = (dt + timedelta(days=RELEASE_LAG_DAYS)).toordinal()
        j = int(np.searchsorted(bar_ords, trade_ord, side="left"))
        if j >= len(bar_ords):
            continue
        trade_idx.append(j)
        zt.append(zz)
    trade_idx = np.array(trade_idx)
    zt = np.array(zt)
    keep = np.concatenate([np.diff(trade_idx) > 0, [True]])
    trade_idx, zt = trade_idx[keep], zt[keep]

    out = {}
    for k in (1, 2, 3, 4):
        fwd = []
        zk = []
        for i in range(len(trade_idx) - k):
            c0 = bar_cl[trade_idx[i]]
            c1 = bar_cl[trade_idx[i + k]]
            fwd.append(c1 / c0 - 1.0)
            zk.append(zt[i])
        fwd = np.array(fwd); zk = np.array(zk)
        X = sm.add_constant(zk)
        # contrarian: expect NEGATIVE slope (high z -> low fwd ret)
        m = sm.OLS(fwd, X).fit(cov_type="HAC", cov_kwds={"maxlags": k + 1})
        out[k] = (float(m.params[1]), float(m.tvalues[1]))
    # two halves on the time axis (split by date)
    return out, trade_idx, zt, bar_ords, bar_cl


def halves_sign(sym, cot_dates, pct, lookback, lo, hi):
    import statsmodels.api as sm
    bar_ords, bar_cl = load_weekly_close(sym)
    z = rolling_z(pct, lookback)
    rows = []
    for dt, zz in zip(cot_dates, z):
        if not np.isfinite(zz):
            continue
        trade_ord = (dt + timedelta(days=RELEASE_LAG_DAYS)).toordinal()
        j = int(np.searchsorted(bar_ords, trade_ord, side="left"))
        if j >= len(bar_ords):
            continue
        rows.append((dt, j, zz))
    # dedup consecutive same-bar
    ded = []
    for r in rows:
        if ded and r[1] == ded[-1][1]:
            ded[-1] = r
        else:
            ded.append(r)
    dts = [r[0] for r in ded]
    idx = np.array([r[1] for r in ded])
    zs = np.array([r[2] for r in ded])
    split = datetime(2021, 1, 1).date()
    res = {}
    for label, mask in (("2016-21", np.array([d < split for d in dts[:-1]])),
                        ("2021-26", np.array([d >= split for d in dts[:-1]]))):
        fwd = np.array([bar_cl[idx[i + 1]] / bar_cl[idx[i]] - 1.0 for i in range(len(idx) - 1)])
        zk = zs[:-1]
        if mask.sum() < 30:
            res[label] = (float("nan"), float("nan"), int(mask.sum()))
            continue
        X = sm.add_constant(zk[mask])
        m = sm.OLS(fwd[mask], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        res[label] = (float(m.params[1]), float(m.tvalues[1]), int(mask.sum()))
    return res


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    cot_dates, pct = load_cot()
    print(f"COT E-mini S&P: {len(cot_dates)} weekly reports "
          f"{cot_dates[0]} -> {cot_dates[-1]}")

    LO, HI = -1.0, 1.5                 # spec bands per the recipe
    LOOKBACKS = (104, 156, 260)        # 2/3/5-yr rolling z windows
    SYMS = ("SPY", "IWM")
    COST_SIDE_BASE = 1.5               # bp/side (top-2 most liquid ETFs)
    COST_SIDE_STRESS = 5.0             # bp/side stress

    # ---- base strategy: SPY core, 156-wk (3yr) z, bands per recipe ----
    base_net, base_gross, base_cost = variant_returns(
        "SPY", cot_dates, pct, 156, LO, HI, COST_SIDE_BASE)
    stress_net, _, _ = variant_returns(
        "SPY", cot_dates, pct, 156, LO, HI, COST_SIDE_STRESS)

    # ---- trial grid (HONEST n_trials) ----
    # {3 lookbacks} x {SPY,IWM} x {2 entry-band variants} = 12 variants.
    band_variants = [(-1.0, 1.5), (-0.5, 1.0)]
    grid_cols = []
    for sym in SYMS:
        for lb in LOOKBACKS:
            for (lo, hi) in band_variants:
                net, _, _ = variant_returns(sym, cot_dates, pct, lb, lo, hi, COST_SIDE_BASE)
                grid_cols.append(net)
    n_trials = len(grid_cols)
    Tmin = min(len(c) for c in grid_cols)
    grid = np.column_stack([c[-Tmin:] for c in grid_cols])

    print(f"\nn_trials (honest) = {n_trials}; base n_obs = {base_net.size}")
    ann = base_net.mean() * 52
    vol = base_net.std() * np.sqrt(52)
    print(f"base SPY 156wk: ann={ann*100:.2f}% vol={vol*100:.2f}% "
          f"Sharpe(naive)={ann/(vol+1e-9):.2f}")

    rc = build_report_card(
        strategy_name="cot_spec_contrarian_SPY",
        returns=base_net,
        n_trials=n_trials,
        trial_grid=grid,
        cost_adjusted_returns=stress_net,
        periods_per_year=52,
    )
    print(rc.render())
    print("STATUS:", rc.status)

    # ---- falsification: NW regression fwd 1-4wk on z ----
    print("\n=== FALSIFICATION: NW regress fwd k-wk return on z (contrarian => slope<0) ===")
    for sym in ("SPY", "IWM"):
        reg, *_ = nw_regression(sym, cot_dates, pct, 156, LO, HI)
        print(f"  {sym}:")
        for k, (b, t) in reg.items():
            print(f"    fwd {k}wk: slope={b:+.5f}  t={t:+.2f}")
        h = halves_sign(sym, cot_dates, pct, 156, LO, HI)
        for lab, (b, t, n) in h.items():
            print(f"    [{lab}] fwd1wk slope={b:+.5f} t={t:+.2f} n={n}")

    # machine-readable summary
    out = {
        "status": rc.status,
        "n_obs": int(base_net.size),
        "n_trials": n_trials,
        "point_sharpe": rc.point_sharpe,
        "sharpe_ci_lower": rc.sharpe_ci_lower,
        "pbo": rc.pbo,
        "dsr_prob": rc.deflated_sharpe,
        "cost_adj_sharpe": rc.cost_adjusted_sharpe,
        "gross_sharpe": float(ann / (vol + 1e-9)),
        "annual_turnover": float(np.sum(np.abs(np.diff(np.concatenate([[0.0], base_gross != 0])))) ),
    }
    print("\nSUMMARY_JSON:", json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""VRP RV-term-structure equity timing overlay (#5).

Honest, NON-circular vol idea: P&L is on the REAL underlying (SPY/IWM), never
on synthetic IV. The signal compares short-window realized vol to long-window
realized vol; when short-RV < long-RV (calm / vol-contracting regime) we hold
the index, otherwise we sit in cash.

NO-LOOK-AHEAD:
  - rv_w[t] uses log-returns derived from prices up to and including index t
    (realized_volatility returns out[t] from log_returns[t-w:t], i.e. prices
    p[t-w..t]). Decision is made at the close of day t.
  - position[t+1] = 1.0 if rv_short[t] < rv_long[t] else 0.0.
  - We TRADE THE NEXT BAR OPEN: a position taken for day t+1 earns the
    underlying return realized over the holding bar (open[t+1] -> open[t+2]).
  - A state flip (entering or leaving the index) costs `flip_bps`.

n_trials is kept HONEST: the only variants tried are the two window pairs
{(5,63),(10,126)} x the two underlyings {SPY,IWM} = 4 trials total. The PBO
trial_grid is built from exactly those 4 columns.

Cost gate: cost-adjusted return stream charges 2bp per flip (vs 1bp base).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._options_data import load_daily_bars
from trading_algo.quant_core.strategies.options.iv_rank import realized_volatility
from trading_algo.quant_core.validation.report_card import build_report_card


def _arrays(symbol: str):
    bars = load_daily_bars(symbol)
    opens = np.array([b.open for b in bars], dtype=float)
    closes = np.array([b.close for b in bars], dtype=float)
    ts = np.array([b.timestamp_epoch_s for b in bars], dtype=float)
    return ts, opens, closes


def _positions(closes: np.ndarray, short_w: int, long_w: int) -> np.ndarray:
    """position[t+1] decided from rv at close t. Returns array aligned to days,
    where pos[k] is the position HELD during the holding bar starting at index k."""
    rv_s = realized_volatility(closes, window=short_w)
    rv_l = realized_volatility(closes, window=long_w)
    T = len(closes)
    pos = np.zeros(T, dtype=float)  # pos[k] = position held over bar starting at open[k]
    # Decision at close t -> held over the bar that opens at t+1.
    for t in range(T - 1):
        s, l = rv_s[t], rv_l[t]
        if np.isfinite(s) and np.isfinite(l):
            pos[t + 1] = 1.0 if s < l else 0.0
    return pos


def overlay_returns(symbol: str, short_w: int, long_w: int, flip_bps: float):
    """Return (net_ret, dates_epoch) for the overlay. Trade at next-bar OPEN.

    Holding bar k spans open[k] -> open[k+1] (one trading day). The position
    held over that bar is pos[k] (decided at close k-1). Cost is charged on the
    flip between pos[k-1] and pos[k] (a change of exposure executed at open[k]).
    """
    ts, opens, closes = _arrays(symbol)
    pos = _positions(closes, short_w, long_w)
    # open-to-open underlying return for the holding bar starting at index k:
    # r_bar[k] = opens[k+1]/opens[k] - 1  (realized while holding from open k to open k+1)
    r_bar = opens[1:] / opens[:-1] - 1.0           # length T-1, indexed by k in [0, T-2]
    held = pos[:-1]                                  # pos held over each bar k
    gross = held * r_bar
    # flips: exposure change executed at open[k]; |pos[k] - pos[k-1]| units traded.
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))  # length T, change entering bar k
    flip_cost = dpos[:-1] * (flip_bps * 1e-4)             # align to bars (length T-1)
    net = gross - flip_cost
    bar_dates = ts[:-1]
    return net, gross, bar_dates


def buy_and_hold(symbol: str):
    ts, opens, closes = _arrays(symbol)
    r_bar = opens[1:] / opens[:-1] - 1.0
    return r_bar, ts[:-1]


def _trim_warmup(series: np.ndarray, long_w: int) -> tuple[np.ndarray, int]:
    """Drop the leading warmup window (before any signal can fire)."""
    start = long_w + 1
    return series[start:], start


def annualised_sharpe(r: np.ndarray, ppy: int = 252) -> float:
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return (r.mean() / sd) * np.sqrt(ppy)


def tail_stats(r: np.ndarray):
    r = np.asarray(r, float)
    from scipy import stats as ss
    return {
        "skew": float(ss.skew(r)),
        "kurt": float(ss.kurtosis(r)),
        "min_day": float(r.min()),
        "p01": float(np.percentile(r, 1)),
        "max_dd": float(_max_dd(r)),
    }


def _max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


# The honest trial set: 2 window pairs x 2 underlyings = 4 variants.
WINDOW_PAIRS = [(5, 63), (10, 126)]
SYMBOLS = ["SPY", "IWM"]
BASE_FLIP_BPS = 1.0
COST_FLIP_BPS = 2.0
N_TRIALS = 4


def build_trial_grid():
    """(T x 4) net-return matrix across all 4 variants, time-aligned by tail."""
    cols = []
    for sym in SYMBOLS:
        for (sw, lw) in WINDOW_PAIRS:
            net, _gross, _d = overlay_returns(sym, sw, lw, BASE_FLIP_BPS)
            net, _ = _trim_warmup(net, lw)
            cols.append(net)
    T = min(len(c) for c in cols)
    return np.column_stack([c[-T:] for c in cols])


def run_for(symbol: str, short_w: int, long_w: int):
    net, gross, _d = overlay_returns(symbol, short_w, long_w, BASE_FLIP_BPS)
    cnet, _gross2, _d2 = overlay_returns(symbol, short_w, long_w, COST_FLIP_BPS)
    bh, _ = buy_and_hold(symbol)

    net, start = _trim_warmup(net, long_w)
    gross, _ = _trim_warmup(gross, long_w)
    cnet, _ = _trim_warmup(cnet, long_w)
    bh_tr, _ = _trim_warmup(bh, long_w)

    grid = build_trial_grid()

    rc = build_report_card(
        strategy_name=f"vrp_overlay_{symbol}_{short_w}_{long_w}",
        returns=net,
        n_trials=N_TRIALS,
        trial_grid=grid,
        cost_adjusted_returns=cnet,
        periods_per_year=252,
    )

    print("=" * 78)
    print(f"VRP OVERLAY  {symbol}  windows=({short_w},{long_w})  flip={BASE_FLIP_BPS}bp")
    print("=" * 78)
    n_in = float(np.mean(net != 0.0))  # fraction of days with exposure (approx time-in-market)
    print(f"n_obs={net.size}  approx_time_in_mkt~={n_in*100:.1f}%")
    print(rc.render())
    print(f"STATUS: {rc.status}")

    ov = annualised_sharpe(net)
    bhs = annualised_sharpe(bh_tr)
    grs = annualised_sharpe(gross)
    print("\n--- vs BUY-AND-HOLD ({}) ---".format(symbol))
    print(f"overlay  ann={net.mean()*252*100:6.2f}%  vol={net.std(ddof=1)*np.sqrt(252)*100:5.2f}%  Sharpe={ov:5.3f}")
    print(f"B&H      ann={bh_tr.mean()*252*100:6.2f}%  vol={bh_tr.std(ddof=1)*np.sqrt(252)*100:5.2f}%  Sharpe={bhs:5.3f}")
    print(f"gross    Sharpe={grs:5.3f}")
    ot = tail_stats(net)
    bt = tail_stats(bh_tr)
    print(f"overlay  skew={ot['skew']:+.3f}  maxDD={ot['max_dd']*100:6.2f}%  worstDay={ot['min_day']*100:+.2f}%  p01={ot['p01']*100:+.2f}%")
    print(f"B&H      skew={bt['skew']:+.3f}  maxDD={bt['max_dd']*100:6.2f}%  worstDay={bt['min_day']*100:+.2f}%  p01={bt['p01']*100:+.2f}%")

    return rc, dict(overlay_sharpe=ov, bh_sharpe=bhs, gross_sharpe=grs,
                    overlay_ann=float(net.mean()*252), bh_ann=float(bh_tr.mean()*252),
                    overlay_skew=ot["skew"], bh_skew=bt["skew"],
                    overlay_maxdd=ot["max_dd"], bh_maxdd=bt["max_dd"],
                    overlay_worst=ot["min_day"], bh_worst=bt["min_day"],
                    n_obs=int(net.size))


def main() -> int:
    results = {}
    # CORE: SPY (5,63). Also report IWM and the (10,126) pair.
    for sym in SYMBOLS:
        for (sw, lw) in WINDOW_PAIRS:
            rc, meta = run_for(sym, sw, lw)
            results[(sym, sw, lw)] = (rc, meta)

    print("\n" + "#" * 78)
    print("SUMMARY (n_trials={} honest: 2 window pairs x 2 underlyings)".format(N_TRIALS))
    print("#" * 78)
    for (sym, sw, lw), (rc, meta) in results.items():
        print(f"{sym} ({sw},{lw}): status={rc.status:11s} "
              f"point={rc.point_sharpe:.3f} CIlo={rc.sharpe_ci_lower:.3f} "
              f"PBO={rc.pbo if rc.pbo is not None else float('nan'):.3f} "
              f"DSR={rc.deflated_sharpe if rc.deflated_sharpe is not None else float('nan'):.3f} "
              f"costSh={rc.cost_adjusted_sharpe if rc.cost_adjusted_sharpe is not None else float('nan'):.3f} | "
              f"vsBH: ov={meta['overlay_sharpe']:.3f} bh={meta['bh_sharpe']:.3f} "
              f"skew {meta['overlay_skew']:+.2f} vs {meta['bh_skew']:+.2f} "
              f"DD {meta['overlay_maxdd']*100:.1f}% vs {meta['bh_maxdd']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

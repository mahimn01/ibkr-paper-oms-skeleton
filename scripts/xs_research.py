"""Look-ahead-safe cross-sectional panel backtester for the 444-symbol set.

The standard research engine for Workflow-2: load the atlas_features_v3 closes
into an aligned (T, N) panel, turn any cross-sectional signal into a long/short
or long-only portfolio with realistic turnover costs, and score the result
through the hardened build_report_card (with a param-sweep trial_grid for PBO
and a punitive-cost re-run for the cost-adjusted gate).

NO-LOOK-AHEAD INVARIANT: weights decided from data through close t earn the
t -> t+1 return. Implemented as port_ret[t] = (W.shift(1) * daily_ret)[t].

Survivorship caveat: atlas_features_v3 is a fixed symbol set (some delisted
series end early, which helps), but it is not a true point-in-time universe;
treat results as an optimistic first read, confirm later on a PIT universe.

Usage:
    python scripts/xs_research.py            # momentum + reversal first read
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_algo.quant_core.validation.report_card import build_report_card

FEATURE_DIR = "data/atlas_features_v3"


# --------------------------------------------------------------------------
# Panel loading (aligned closes)
# --------------------------------------------------------------------------

def load_panel(feature_dir: str = FEATURE_DIR, min_obs: int = 500):
    """Return (dates: int epoch-sec (T,), closes: (T, N) float with NaN, symbols: list[str])."""
    files = sorted(glob.glob(os.path.join(feature_dir, "*_features.npz")))
    if not files:
        raise FileNotFoundError(f"no npz under {feature_dir}")

    cols: dict[str, dict[int, float]] = {}
    all_ts: set[int] = set()
    for f in files:
        sym = os.path.basename(f).replace("_features.npz", "")
        z = np.load(f)
        ts = z["timestamps"].astype(np.float64)
        cl = z["closes"].astype(np.float64)
        if len(ts) < min_obs:
            continue
        keyed = {int(round(t)): float(c) for t, c in zip(ts, cl) if np.isfinite(c) and c > 0}
        if len(keyed) < min_obs:
            continue
        cols[sym] = keyed
        all_ts.update(keyed.keys())

    dates = np.array(sorted(all_ts), dtype=np.int64)
    idx = {t: i for i, t in enumerate(dates)}
    symbols = sorted(cols.keys())
    closes = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    for j, sym in enumerate(symbols):
        for t, c in cols[sym].items():
            closes[idx[t], j] = c
    return dates, closes, symbols


def load_r3000_panel(start: str = "2012-01-01", end: str = "2026-04-01",
                     min_price: float = 5.0, min_obs: int = 1500,
                     dv_lo_pct: float = 10.0, dv_hi_pct: float = 55.0,
                     data_dir: str = "data/atlas_r3000"):
    """Russell-3000 close panel restricted to a small-but-tradeable liquidity band.

    Filters: median close >= min_price, >= min_obs days in [start,end], and median
    dollar-volume within the [dv_lo_pct, dv_hi_pct] percentile band (isolates the
    small-cap segment while excluding untradeable microcaps).
    Survivorship caveat: these are a fixed (largely surviving) constituent set.
    """
    import pandas as pd
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    closes_by_sym: dict[str, "pd.Series"] = {}
    dv: dict[str, float] = {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"]).loc[start:end]
        except Exception:
            continue
        if len(df) < min_obs:
            continue
        c = df["close"].astype(float)
        if not np.isfinite(c.median()) or c.median() < min_price:
            continue
        closes_by_sym[sym] = c
        dv[sym] = float((c * df["volume"].astype(float)).median())

    syms = list(closes_by_sym)
    dvs = np.array([dv[s] for s in syms])
    lo, hi = np.nanpercentile(dvs, [dv_lo_pct, dv_hi_pct])
    keep = [s for s in syms if lo <= dv[s] <= hi]

    panel = pd.concat({s: closes_by_sym[s] for s in keep}, axis=1).sort_index().loc[start:end]
    dates = np.array([int(ts.timestamp()) for ts in panel.index], dtype=np.int64)
    return dates, panel.values.astype(np.float64), list(panel.columns)


# --------------------------------------------------------------------------
# Signals (each returns an (T, N) score; NaN where undefined)
# --------------------------------------------------------------------------

def momentum(closes: np.ndarray, lookback: int = 252, gap: int = 21) -> np.ndarray:
    """12-1 momentum: return from t-lookback to t-gap (skip the most recent month)."""
    T, N = closes.shape
    sig = np.full((T, N), np.nan)
    for t in range(lookback, T):
        past = closes[t - lookback]
        recent = closes[t - gap]
        with np.errstate(invalid="ignore", divide="ignore"):
            sig[t] = recent / past - 1.0
    return sig


def reversal(closes: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Short-term reversal: SHORT recent winners, LONG recent losers => signal = -(recent return)."""
    T, N = closes.shape
    sig = np.full((T, N), np.nan)
    for t in range(lookback, T):
        with np.errstate(invalid="ignore", divide="ignore"):
            sig[t] = -(closes[t] / closes[t - lookback] - 1.0)
    return sig


def low_vol(closes: np.ndarray, lookback: int = 63) -> np.ndarray:
    """Betting-against-vol: signal = -realized vol (long low-vol names)."""
    T, N = closes.shape
    rets = np.full((T, N), np.nan)
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    sig = np.full((T, N), np.nan)
    for t in range(lookback, T):
        window = rets[t - lookback + 1:t + 1]
        with np.errstate(invalid="ignore"):
            sig[t] = -np.nanstd(window, axis=0)
    return sig


# --------------------------------------------------------------------------
# Cross-sectional backtest (look-ahead-safe)
# --------------------------------------------------------------------------

def cross_sectional_backtest(
    closes: np.ndarray,
    signal: np.ndarray,
    *,
    top_q: float = 0.2,
    long_short: bool = True,
    rebalance: int = 5,
    cost_bps: float = 10.0,
    min_names: int = 20,
    lag: int = 1,
) -> np.ndarray:
    """Return daily net portfolio returns (T,). Weights from close t earn the
    t+lag-1 -> t+lag return. lag=1 is next-day (no look-ahead); lag=2 skips a
    day to defuse bid-ask-bounce artefacts in illiquid names."""
    T, N = closes.shape
    daily_ret = np.full((T, N), np.nan)
    daily_ret[1:] = closes[1:] / closes[:-1] - 1.0

    W = np.zeros((T, N))
    last_w = np.zeros(N)
    for t in range(T):
        if t % rebalance == 0:
            s = signal[t]
            valid = np.isfinite(s) & np.isfinite(closes[t])
            if valid.sum() >= min_names:
                sv = s[valid]
                order = np.argsort(sv)
                n = len(sv)
                k = max(1, int(np.floor(top_q * n)))
                idx_valid = np.where(valid)[0]
                w = np.zeros(N)
                longs = idx_valid[order[-k:]]
                w[longs] = 1.0 / k
                if long_short:
                    shorts = idx_valid[order[:k]]
                    w[shorts] = -1.0 / k
                last_w = w
            # else: carry previous weights
        W[t] = last_w

    # weights set at t-lag earn day t's return (lag>=1 => no look-ahead)
    w_prev = np.vstack([np.zeros((lag, N)), W[:-lag]])
    contrib = np.where(np.isfinite(daily_ret), w_prev * daily_ret, 0.0)
    gross = contrib.sum(axis=1)

    # turnover cost charged when weights change
    turnover = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, N)))).sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    return net


# --------------------------------------------------------------------------
# Score through the hardened gate
# --------------------------------------------------------------------------

def score(name: str, closes: np.ndarray, dates: np.ndarray, make_signal,
          base_params: dict, sweep: list[dict], cost_bps_base: float = 10.0,
          cost_bps_stress: float = 30.0) -> None:
    """Run base + a param-sweep trial_grid + a cost-stressed run, then print the card."""
    base = cross_sectional_backtest(closes, make_signal(closes, **base_params.get("sig", {})),
                                    cost_bps=cost_bps_base, **base_params.get("bt", {}))
    cols = []
    for p in sweep:
        r = cross_sectional_backtest(closes, make_signal(closes, **p.get("sig", {})),
                                     cost_bps=cost_bps_base, **p.get("bt", {}))
        cols.append(r)
    T = min(len(c) for c in cols)
    grid = np.column_stack([c[-T:] for c in cols])
    stress = cross_sectional_backtest(closes, make_signal(closes, **base_params.get("sig", {})),
                                      cost_bps=cost_bps_stress, **base_params.get("bt", {}))

    # warmup: drop leading zeros (pre-first-signal) for an honest return series
    nz = np.flatnonzero(base != 0.0)
    start = nz[0] if nz.size else 0
    rc = build_report_card(
        strategy_name=name,
        returns=base[start:],
        n_trials=len(sweep),
        trial_grid=grid,
        cost_adjusted_returns=stress[start:],
        periods_per_year=252,
    )
    ann = np.mean(base[start:]) * 252
    vol = np.std(base[start:]) * np.sqrt(252)
    print(f"\n### {name}  ann={ann*100:.1f}%  vol={vol*100:.1f}%  "
          f"Sharpe={ann/(vol+1e-9):.2f}  n_obs={base[start:].size}")
    print(rc.render())
    out = Path("validation_reports"); out.mkdir(exist_ok=True)
    (out / f"{name}.md").write_text(rc.render())
    print(f"STATUS: {rc.status}")


def main() -> int:
    print("loading panel ...")
    dates, closes, symbols = load_panel()
    print(f"panel: {closes.shape[0]} days x {len(symbols)} symbols "
          f"({np.isfinite(closes).sum()} valid closes)")

    # Momentum 12-1, weekly rebalance, long/short top/bottom quintile
    score("xs_momentum_12_1", closes, dates, momentum,
          base_params={"sig": {"lookback": 252, "gap": 21}, "bt": {"top_q": 0.2, "rebalance": 21}},
          sweep=[{"sig": {"lookback": lb, "gap": 21}, "bt": {"top_q": q, "rebalance": 21}}
                 for lb in (189, 252, 315) for q in (0.1, 0.2, 0.3)])

    # Short-term reversal (1-week), daily-ish rebalance
    score("xs_reversal_1w", closes, dates, reversal,
          base_params={"sig": {"lookback": 5}, "bt": {"top_q": 0.1, "rebalance": 5}},
          sweep=[{"sig": {"lookback": lb}, "bt": {"top_q": q, "rebalance": 5}}
                 for lb in (3, 5, 10) for q in (0.05, 0.1, 0.2)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

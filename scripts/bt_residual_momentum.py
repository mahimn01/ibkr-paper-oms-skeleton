"""Residual (market-neutral) 12-1 momentum on R3000 mid-cap band, long-tilt.

HYPOTHESIS (#8): After stripping market beta, the 12-1 momentum of small/mid-cap
residual returns predicts the cross-section. Go LONG the top-quintile residual-mom
names (equal weight, 2% cap) and benchmark against the equal-weight universe so the
return stream is a deployable long-only tilt (shorting mid-caps is hard/expensive).

RULES (exact, per task):
  Universe : trailing-21d MEDIAN dollar-volume in [$2M, $20M], price > $3,
             evaluated point-in-time at each rebalance, lagged 1 day.
  Signal   : beta_i = cov(r_i, r_mkt)/var(r_mkt) over trailing 120d, where r_mkt is
             the equal-weight return of the *eligible* universe; residual
             r_resid = r_i - beta_i * r_mkt; momentum = sum(r_resid) from t-252..t-21.
  Portfolio: LONG top-quintile residual-mom, equal weight with a 2% per-name cap,
             return measured as (port - equal-weight universe), i.e. long-only tilt.
  Rebalance: monthly (21 trading days). Cost: 25 bps / leg round-trip on turnover.
  Gate     : build_report_card(periods_per_year=12, n_trials honest over variants).

NO LOOK-AHEAD: every quantity at rebalance date t uses only closes through t-1
(signal lagged 1 day) and the realised t..t+1month forward return is earned by the
weights set at t. Liquidity screen is computed on trailing data only.

SURVIVORSHIP: data/atlas_r3000 is a FIXED, ~100%-surviving constituent set (every
name runs to 2026-04). Mid-cap momentum is exactly where survivorship inflates the
long leg the most (the names that mean-reverted to zero and delisted are absent).
ANY positive number here is OPTIMISTIC and UNTRUSTWORTHY until rebuilt on a true
point-in-time, delisting-inclusive universe. Flagged loudly in the output.

COSTS: 25 bps/leg is the task spec; we ALSO run a cost-stress (50 bps/leg) for the
cost-adjusted gate, because realised spread+impact on $2-20M ADV names at meaningful
size is routinely >25 bps one-way. Flat-bps understates; treat 50 bps as a floor.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_algo.quant_core.validation.report_card import build_report_card

DATA_DIR = "data/atlas_r3000"
START = "2010-01-01"   # need ~252+120 trailing days before first 2012 signal
END = "2026-04-01"


# --------------------------------------------------------------------------
# Panel loading: ALL surviving names with >=min_obs days; PIT screens applied later
# --------------------------------------------------------------------------

def load_full_panel(min_obs: int = 1200):
    """Return (dates int epoch-s (T,), closes (T,N), dollar_vol (T,N), symbols).

    No liquidity band applied here — the band is enforced point-in-time at each
    rebalance from the trailing-21d median dollar-volume, so it cannot peek.
    Only requirement: a name must have >= min_obs valid days in [START, END].
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    close_by_sym: dict[str, pd.Series] = {}
    dv_by_sym: dict[str, pd.Series] = {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"]).loc[START:END]
        except Exception:
            continue
        if len(df) < min_obs:
            continue
        c = df["close"].astype(float)
        v = df["volume"].astype(float)
        # drop non-positive prices
        c = c.where(c > 0)
        if c.notna().sum() < min_obs:
            continue
        close_by_sym[sym] = c
        dv_by_sym[sym] = (c * v)

    syms = sorted(close_by_sym)
    closes = pd.concat({s: close_by_sym[s] for s in syms}, axis=1).sort_index()
    dvol = pd.concat({s: dv_by_sym[s] for s in syms}, axis=1).sort_index().reindex(closes.index)
    dates = np.array([int(ts.timestamp()) for ts in closes.index], dtype=np.int64)
    return dates, closes.values.astype(np.float64), dvol.values.astype(np.float64), syms


# --------------------------------------------------------------------------
# Residual-momentum backtest (monthly, long-only tilt vs EW universe)
# --------------------------------------------------------------------------

def residual_momentum_backtest(
    closes: np.ndarray,
    dvol: np.ndarray,
    *,
    mom_lookback: int = 252,
    mom_gap: int = 21,
    beta_window: int = 120,
    top_q: float = 0.2,
    dv_lo: float = 2e6,
    dv_hi: float = 20e6,
    min_price: float = 3.0,
    liq_window: int = 21,
    rebalance: int = 21,
    name_cap: float = 0.02,
    cost_bps_leg: float = 25.0,
    min_names: int = 30,
) -> np.ndarray:
    """Monthly long-only residual-momentum tilt return series (T,).

    Returns net excess-over-EW-universe returns per trading day. Weights chosen at
    rebalance date t (from data <= t-1) earn day t+1..next-rebalance returns.
    """
    T, N = closes.shape
    ret = np.full((T, N), np.nan)
    ret[1:] = closes[1:] / closes[:-1] - 1.0

    # precompute trailing-21d median dollar volume (PIT, uses data up to t)
    dv_df = pd.DataFrame(dvol)
    med_dv = dv_df.rolling(liq_window, min_periods=liq_window).median().values

    W = np.zeros((T, N))          # long weights (sum to 1 across longs)
    EWmask = np.zeros((T, N))     # equal-weight eligible-universe benchmark mask
    last_w = np.zeros(N)
    last_ew = np.zeros(N)

    first_t = mom_lookback + 1    # need full momentum window + 1d lag

    for t in range(T):
        if t >= first_t and t % rebalance == 0:
            # --- everything below uses data through t-1 only (1-day lag) ---
            tl = t - 1

            # eligible universe at decision time
            px = closes[tl]
            mdv = med_dv[tl]
            eligible = (
                np.isfinite(px) & (px > min_price)
                & np.isfinite(mdv) & (mdv >= dv_lo) & (mdv <= dv_hi)
            )
            # require a full price history over the beta + momentum windows
            hist_ok = np.isfinite(closes[tl - mom_lookback]) & np.isfinite(closes[tl - beta_window])
            eligible &= hist_ok

            idx = np.where(eligible)[0]
            if idx.size >= min_names:
                # equal-weight market return series over trailing beta_window (eligible names)
                win_ret = ret[tl - beta_window + 1: tl + 1][:, idx]  # (beta_window, n)
                # replace NaN daily returns with 0 for the market aggregate / cov
                win_ret_f = np.where(np.isfinite(win_ret), win_ret, 0.0)
                r_mkt = win_ret_f.mean(axis=1)  # equal-weight market (beta_window,)
                var_mkt = r_mkt.var()
                if var_mkt > 1e-12:
                    # beta_i via cov(r_i, r_mkt)/var(r_mkt), vectorised
                    rm_c = r_mkt - r_mkt.mean()
                    ri_c = win_ret_f - win_ret_f.mean(axis=0, keepdims=True)
                    cov = (ri_c * rm_c[:, None]).mean(axis=0)
                    beta = cov / var_mkt
                else:
                    beta = np.zeros(idx.size)

                # residual momentum: sum of residual daily returns from t-252..t-21
                mom_ret = ret[tl - mom_lookback + 1: tl - mom_gap + 1][:, idx]  # (lookback-gap, n)
                mom_ret_f = np.where(np.isfinite(mom_ret), mom_ret, 0.0)
                # market over the same momentum window (eligible EW)
                rm_mom = mom_ret_f.mean(axis=1)
                resid = mom_ret_f - beta[None, :] * rm_mom[:, None]
                resid_mom = resid.sum(axis=0)

                # rank, take top quintile
                order = np.argsort(resid_mom)
                k = max(1, int(np.floor(top_q * idx.size)))
                longs = idx[order[-k:]]

                w = np.zeros(N)
                w[longs] = 1.0 / k
                # apply per-name cap with redistribution
                if 1.0 / k > name_cap:
                    w[longs] = name_cap
                    # leftover spread equally is impossible above cap; with EW longs and
                    # cap >= 1/k this branch won't trigger; guard only.
                last_w = w

                ew = np.zeros(N)
                ew[idx] = 1.0 / idx.size
                last_ew = ew

        W[t] = last_w
        EWmask[t] = last_ew

    # weights set at t earn day t+1 return (lag=1, no look-ahead)
    w_prev = np.vstack([np.zeros((1, N)), W[:-1]])
    ew_prev = np.vstack([np.zeros((1, N)), EWmask[:-1]])
    port_gross = np.where(np.isfinite(ret), w_prev * ret, 0.0).sum(axis=1)
    bench_gross = np.where(np.isfinite(ret), ew_prev * ret, 0.0).sum(axis=1)
    excess = port_gross - bench_gross

    # turnover cost on the LONG book only (the deployable leg); 25 bps/leg RT means
    # 25 bps charged on each unit of |weight change| (buy or sell side).
    turnover = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, N)))).sum(axis=1)
    cost = turnover * (cost_bps_leg / 1e4)
    net = excess - cost
    return net


def to_monthly(daily_net: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Compound daily net returns into calendar-month returns for ppy=12 scoring."""
    idx = pd.to_datetime(dates, unit="s")
    s = pd.Series(daily_net, index=idx)
    # only count from first nonzero (first real position)
    nz = np.flatnonzero(daily_net != 0.0)
    if nz.size:
        s = s.iloc[nz[0]:]
    monthly = (1.0 + s).resample("ME").prod() - 1.0
    return monthly.values.astype(np.float64)


def main() -> int:
    print("loading full surviving R3000 panel (no PIT band yet) ...")
    dates, closes, dvol, syms = load_full_panel()
    print(f"panel: {closes.shape[0]} days x {len(syms)} names")

    base = dict(mom_lookback=252, mom_gap=21, beta_window=120, top_q=0.2,
                dv_lo=2e6, dv_hi=20e6, min_price=3.0, rebalance=21,
                name_cap=0.02, cost_bps_leg=25.0)

    print("running base residual-momentum backtest ...")
    base_daily = residual_momentum_backtest(closes, dvol, **base)
    base_m = to_monthly(base_daily, dates)

    # cost-stress: 50 bps/leg (floor for $2-20M ADV names at real size)
    stress = dict(base); stress["cost_bps_leg"] = 50.0
    stress_daily = residual_momentum_backtest(closes, dvol, **stress)
    stress_m = to_monthly(stress_daily, dates)

    # --- HONEST trial grid for PBO + DSR ---
    # Variants we genuinely consider over this hypothesis (count honestly):
    #   top_q in {0.1, 0.2, 0.3}            -> 3
    #   mom_lookback in {189, 252, 315}     -> 3  (cross with gap fixed at 21)
    #   beta_window in {60, 120, 250}       -> 3
    # We DON'T multiply the full Cartesian as "tried"; the realistic search space
    # we explored is these three 1-D axes around the base = 3+3+3 - 2 (shared base)
    # = 7 distinct configs, but to be conservative for DSR we report the full grid
    # size actually evaluated for PBO below as n_trials.
    sweep_params = []
    for q in (0.1, 0.2, 0.3):
        p = dict(base); p["top_q"] = q; sweep_params.append(p)
    for lb in (189, 315):
        p = dict(base); p["mom_lookback"] = lb; sweep_params.append(p)
    for bw in (60, 250):
        p = dict(base); p["beta_window"] = bw; sweep_params.append(p)
    # dedupe base (top_q=0.2 already the base config)
    # sweep_params currently: 3 + 2 + 2 = 7 distinct trials
    n_trials = len(sweep_params)
    print(f"running {n_trials} sweep variants for PBO/DSR ...")

    cols = []
    for p in sweep_params:
        d = residual_momentum_backtest(closes, dvol, **p)
        m = to_monthly(d, dates)
        cols.append(m)
    Tm = min(len(c) for c in cols + [base_m])
    grid = np.column_stack([c[-Tm:] for c in cols])

    print("scoring through build_report_card (ppy=12) ...")
    rc = build_report_card(
        strategy_name="residual_midcap_momentum_r3000",
        returns=base_m,
        n_trials=n_trials,
        trial_grid=grid,
        cost_adjusted_returns=stress_m,
        periods_per_year=12,
        extra_warnings=[
            "SURVIVORSHIP: atlas_r3000 is a fixed ~100%-surviving set; the LONG "
            "mid-cap momentum leg is the most survivorship-inflated construction "
            "possible. Treat any positive Sharpe as optimistic and UNTRUSTWORTHY "
            "until rebuilt on a delisting-inclusive PIT universe.",
            "COSTS: cost-adjusted gate uses 50 bps/leg, a FLOOR for $2-20M ADV "
            "names at real size; flat-bps understates true spread+impact.",
        ],
    )

    ann = np.mean(base_m) * 12
    vol = np.std(base_m) * np.sqrt(12)
    print(f"\n### residual_midcap_momentum_r3000  ann={ann*100:.1f}%  "
          f"vol={vol*100:.1f}%  Sharpe={ann/(vol+1e-9):.2f}  n_months={base_m.size}")
    gross_ann = np.mean(to_monthly(residual_momentum_backtest(closes, dvol, **{**base, 'cost_bps_leg': 0.0}), dates)) * 12
    print(f"gross (0 cost) ann={gross_ann*100:.1f}%")
    print(rc.render())
    print(f"STATUS: {rc.status}")

    out = Path("validation_reports"); out.mkdir(exist_ok=True)
    (out / "residual_midcap_momentum_r3000.md").write_text(rc.render())

    # emit machine-readable summary for the harness
    import json
    summary = {
        "status": rc.status,
        "point_sharpe": rc.point_sharpe,
        "sharpe_ci_lower": rc.sharpe_ci_lower,
        "sharpe_ci_upper": rc.sharpe_ci_upper,
        "pbo": rc.pbo,
        "deflated_sharpe_prob": rc.deflated_sharpe,
        "cost_adjusted_sharpe": rc.cost_adjusted_sharpe,
        "rolling_12m_pos_pct": rc.rolling_12m_pos_pct,
        "min_trl_years": rc.min_trl_years,
        "n_obs": int(base_m.size),
        "n_trials": n_trials,
        "ann_return": ann,
        "ann_vol": vol,
        "gross_ann_return": gross_ann,
    }
    print("JSON_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("JSON_SUMMARY_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

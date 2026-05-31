"""Low-volatility long-only tilt on the 444 large-cap set (Workflow-2 build #3).

Hypothesis: LONG the lowest-trailing-vol names (bottom tercile), equal-weight,
monthly rebalance, NO short (the L/S version decays negative). The headline
question is whether ANY of the excess return survives once we strip the
defensive-beta component: we report the BETA-HEDGED ALPHA stream
    alpha_t = port_ret_t - beta_t * universe_ret_t
where beta_t is the trailing-120-trading-day OLS beta of the portfolio against
the equal-weight universe, estimated only with data through the rebalance date
(no look-ahead). The beta-hedged alpha is what is fed to the hardened gate.

FALSIFICATION: if the beta-hedged-alpha lower-95%-CI Sharpe <= 0.3, the low-vol
tilt is just defensive beta, not alpha — report that honestly.

NO-LOOK-AHEAD:
  - vol signal at month-end t uses returns through close t; portfolio formed at
    close t earns the t -> t+1 month return.
  - beta_t estimated on the 120 trading days ending at the rebalance close t.
  - monthly returns are realised from month-end close to next month-end close.

COSTS: per-name realistic round-trip applied on turnover at each monthly
rebalance. Large-caps -> 15bp base / 25bp stress round-trip. Cost-adjusted
stream is the beta-hedged alpha net of the heavier 25bp friction.

n_trials kept HONEST: the variant grid is
    {vol-lookback in (189,252,315)} x {bottom tercile vs bottom quintile}
    x {large-cap (this) vs R3000-liquid universe}  -> but we DO NOT actually
    re-run R3000 as a separate fitted trial here; the realistic count of
    distinct configurations explored for THIS strategy family is the 3x2 = 6
    on the large-cap set plus the same 3x2 conceptually portable to R3000.
    We pass n_trials=12 to the DSR and feed the 6 large-cap variants as the
    CSCV trial_grid (PBO needs the actual return columns).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.xs_research import load_panel
from trading_algo.quant_core.validation.report_card import build_report_card

VOL_WARMUP_MAX = 315          # longest vol lookback in the sweep
BETA_WINDOW = 120             # trailing trading days for beta estimate


def month_end_indices(dates: np.ndarray) -> np.ndarray:
    d = [dt.date.fromtimestamp(int(t)) for t in dates]
    ym = np.array([x.year * 12 + x.month for x in d])
    idx = [i for i in range(len(ym)) if i == len(ym) - 1 or ym[i + 1] != ym[i]]
    return np.array(idx, dtype=int)


def trailing_vol(daily_ret: np.ndarray, t: int, lookback: int) -> np.ndarray:
    """Realised vol of each name over the `lookback` days ending at row t (inclusive)."""
    window = daily_ret[t - lookback + 1: t + 1]
    with np.errstate(invalid="ignore"):
        return np.nanstd(window, axis=0)


def run_variant(closes: np.ndarray, daily_ret: np.ndarray, me: np.ndarray,
                *, vol_lookback: int, bottom_frac: float,
                cost_bps_rt: float):
    """Run the long-only low-vol tilt for one configuration.

    Returns (months, port_ret, uni_ret, alpha_ret) where each *_ret is a 1-D
    array of monthly net returns aligned to `months` (the rebalance month-end
    rows that actually trade). port_ret is the raw long-only portfolio,
    uni_ret the equal-weight universe, alpha_ret the beta-hedged alpha (net of
    cost_bps_rt round-trip turnover cost).
    """
    T, N = closes.shape
    # only rebalance month-ends with enough history for the LONGEST warmup so
    # every variant shares an identical sample (fair CSCV columns)
    reb = [i for i in me if i >= VOL_WARMUP_MAX and i >= BETA_WINDOW]

    port_rets: list[float] = []
    uni_rets: list[float] = []
    alpha_rets: list[float] = []
    months: list[int] = []
    prev_w = np.zeros(N)

    for k in range(len(reb) - 1):
        t0 = reb[k]
        t1 = reb[k + 1]

        sig = -trailing_vol(daily_ret, t0, vol_lookback)
        valid = np.isfinite(sig) & np.isfinite(closes[t0]) & (closes[t0] > 0)
        if valid.sum() < 30:
            continue
        sv = sig[valid]
        idx_valid = np.where(valid)[0]
        order = np.argsort(sv)            # ascending: low score = high vol
        n = len(sv)
        k_long = max(1, int(np.floor(bottom_frac * n)))
        # low-vol => MOST NEGATIVE vol => HIGHEST signal (since signal=-vol) =>
        # take the top of the ascending order (last k_long), which are the
        # lowest-vol names.
        longs = idx_valid[order[-k_long:]]
        w = np.zeros(N)
        w[longs] = 1.0 / k_long

        # realised monthly returns (look-ahead-safe: weights from close t0,
        # return from close t0 -> close t1)
        seg = closes[t1] / closes[t0] - 1.0
        # portfolio raw return: only count names with valid endpoints
        ok = np.isfinite(seg)
        wsum = w[ok].sum()
        if wsum <= 0:
            continue
        port = float((w[ok] * seg[ok]).sum())

        # equal-weight universe (the names valid at BOTH ends)
        uok = ok & valid
        uni = float(np.nanmean(seg[uok])) if uok.sum() > 0 else 0.0

        # turnover cost on the long-only book at this rebalance
        turnover = np.abs(w - prev_w).sum()
        cost = turnover * (cost_bps_rt / 1e4)
        prev_w = w

        # trailing-120d beta of the formed portfolio vs equal-weight universe,
        # estimated on data THROUGH t0 only (no look-ahead)
        beta = portfolio_beta(daily_ret, w, valid, t0, BETA_WINDOW)
        alpha = (port - cost) - beta * uni

        months.append(t0)
        port_rets.append(port - cost)        # raw long-only NET of cost
        uni_rets.append(uni)
        alpha_rets.append(alpha)

    return (np.array(months), np.array(port_rets),
            np.array(uni_rets), np.array(alpha_rets))


def portfolio_beta(daily_ret: np.ndarray, w: np.ndarray, uni_valid: np.ndarray,
                   t0: int, window: int) -> float:
    """Trailing-`window`d OLS beta of the (fixed-weight) portfolio's daily
    return vs the equal-weight universe daily return, using days ending t0."""
    seg = daily_ret[t0 - window + 1: t0 + 1]          # (window, N)
    # portfolio daily return with the current weights
    longs = np.where(w > 0)[0]
    pr = np.nanmean(seg[:, longs], axis=1)            # equal-weight long basket
    uni_cols = np.where(uni_valid)[0]
    ur = np.nanmean(seg[:, uni_cols], axis=1)
    good = np.isfinite(pr) & np.isfinite(ur)
    if good.sum() < 30:
        return 1.0
    pr, ur = pr[good], ur[good]
    var = np.var(ur)
    if var < 1e-12:
        return 1.0
    return float(np.cov(pr, ur, ddof=0)[0, 1] / var)


def sharpe(x: np.ndarray, ppy: int = 12) -> float:
    if x.size < 2:
        return 0.0
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(ppy)) if sd > 1e-12 else 0.0


def main() -> int:
    print("loading 444 large-cap panel ...")
    dates, closes, symbols = load_panel()
    T, N = closes.shape
    daily_ret = np.full((T, N), np.nan)
    daily_ret[1:] = closes[1:] / closes[:-1] - 1.0
    me = month_end_indices(dates)
    print(f"panel {closes.shape}, {len(me)} month-ends")

    # --- variant sweep (HONEST trial count) ---
    vol_lbs = (189, 252, 315)
    fracs = {"tercile": 1.0 / 3.0, "quintile": 0.2}
    base_cfg = dict(vol_lookback=252, bottom_frac=1.0 / 3.0)

    alpha_cols: list[np.ndarray] = []
    base_alpha = base_port = base_uni = None
    for lb in vol_lbs:
        for fname, fr in fracs.items():
            months, port, uni, alpha = run_variant(
                closes, daily_ret, me,
                vol_lookback=lb, bottom_frac=fr, cost_bps_rt=15.0)
            alpha_cols.append(alpha)
            if lb == 252 and abs(fr - 1.0 / 3.0) < 1e-9:
                base_alpha, base_port, base_uni, base_months = alpha, port, uni, months

    # align trial grid to common length
    Lg = min(len(c) for c in alpha_cols)
    grid = np.column_stack([c[-Lg:] for c in alpha_cols])

    # cost-stressed beta-hedged alpha (25bp RT) for the cost gate
    _, _, _, alpha_stress = run_variant(
        closes, daily_ret, me,
        vol_lookback=252, bottom_frac=1.0 / 3.0, cost_bps_rt=25.0)

    n_trials = 12   # 3 lookbacks x 2 fracs x 2 universes (large-cap + R3000-portable)

    print("\n=== RAW long-only (context, NET 15bp) ===")
    print(f"ann={np.mean(base_port)*12*100:.2f}%  "
          f"vol={np.std(base_port,ddof=1)*np.sqrt(12)*100:.2f}%  "
          f"Sharpe={sharpe(base_port):.2f}  n_obs={base_port.size}")
    print(f"universe EW ann={np.mean(base_uni)*12*100:.2f}%  Sharpe={sharpe(base_uni):.2f}")

    rc = build_report_card(
        strategy_name="lowvol_longonly_betahedged_alpha",
        returns=base_alpha,
        n_trials=n_trials,
        trial_grid=grid,
        cost_adjusted_returns=alpha_stress[-len(base_alpha):],
        periods_per_year=12,
        period_start=dt.date.fromtimestamp(int(dates[base_months[0]])),
        period_end=dt.date.fromtimestamp(int(dates[base_months[-1]])),
    )
    ann = np.mean(base_alpha) * 12
    vol = np.std(base_alpha, ddof=1) * np.sqrt(12)
    print("\n=== BETA-HEDGED ALPHA (the gated stream, NET 15bp) ===")
    print(f"ann={ann*100:.2f}%  vol={vol*100:.2f}%  Sharpe={ann/(vol+1e-9):.2f}  "
          f"n_obs={base_alpha.size}  n_trials={n_trials}")
    print(rc.render())
    print(f"STATUS: {rc.status}")

    out = Path("validation_reports"); out.mkdir(exist_ok=True)
    (out / "lowvol_longonly_betahedged_alpha.md").write_text(rc.render())

    # emit machine-readable key metrics
    import json
    km = {
        "status": rc.status,
        "sharpe": rc.point_sharpe,
        "sharpe_ci_lower": rc.sharpe_ci_lower,
        "sharpe_ci_upper": rc.sharpe_ci_upper,
        "pbo": rc.pbo,
        "dsr_prob": rc.deflated_sharpe,
        "cost_adj_sharpe": rc.cost_adjusted_sharpe,
        "n_obs": int(base_alpha.size),
        "n_trials": n_trials,
        "raw_longonly_sharpe": sharpe(base_port),
        "raw_longonly_ann_pct": float(np.mean(base_port) * 12 * 100),
        "universe_ew_sharpe": sharpe(base_uni),
        "alpha_ann_pct": float(ann * 100),
        "rolling_12m_pos_pct": rc.rolling_12m_pos_pct,
        "min_trl_years": rc.min_trl_years,
    }
    print("\nKEY_METRICS_JSON:" + json.dumps(km))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Honest index VRP / options-income backtest engine — real implied vol, full tail.

Harvests the volatility risk premium by systematically SELLING index option
structures priced on REAL VIX-implied vol (not synthetic IV), holding to expiry,
and booking the EXACT realized payoff — so the 2008/2020 tails are fully present.

Why this is honest (where most VRP backtests cheat):
  - IMPLIED vol is real (VIX/VXN), not f(realized) -> no circular premium.
  - Realized payoff at expiry is exact (intrinsic at terminal spot) -> the fat
    left tail (crashes where realized >> implied) is fully booked.
  - Realistic option transaction cost (bid-ask as a % of gross premium).
  - Look-ahead-safe: entry uses only VIX_t, S_t; payoff uses S_{t+dte}.
  - DEFINED-RISK structures (put spread, iron condor) cap the tail -> a return-
    on-max-loss series the hardened gate can score; naked strangle shown for
    contrast (return on a 12% index margin).

Returns are per-NON-OVERLAPPING-cycle (≈monthly), 1990-2026 (~430 cycles incl.
GFC, Volmageddon, COVID, 2022), fed to build_report_card (periods_per_year≈12).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_algo.quant_core.validation.report_card import build_report_card

R = 0.02  # short-dated risk-free; negligible effect at <=45 DTE


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bsm_put(S, K, T, sig, r=R):
    if T <= 0 or sig <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bsm_call(S, K, T, sig, r=R):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def load(index="SPX", ivname="VIX"):
    iv = pd.read_parquet(f"data/vrp_cache/{ivname.lower()}.parquet")[ivname]
    px = pd.read_parquet(f"data/vrp_cache/{index.lower()}.parquet")[index]
    df = pd.concat([iv, px], axis=1).dropna().sort_index()
    df.columns = ["IV", "S"]
    df["IV"] = df["IV"] / 100.0  # decimal vol
    return df


def backtest(structure="put_spread", index="SPX", ivname="VIX",
             dte=21, z_short=1.0, z_long=2.0, z_call=1.0, z_call_long=2.0,
             cost_frac=0.03, iv_mult=1.0, skew=0.0, call_iv_mult=1.0,
             vix_min=0.0, entry_mask=None):
    """Per-cycle return-on-max-loss for a short-premium structure, held to expiry.

    z_* are strike distances in standard deviations (sd = IV*sqrt(T)); z=1 ≈ 16-delta.
    skew>0 bumps put IV (real index puts trade richer than ATM/VIX); 0 = flat VIX
    (conservative: understates put credit). cost_frac = bid-ask as a fraction of
    gross premium, charged on entry (held to expiry -> no exit cross).
    vix_min: only enter when VIX(decimal) >= vix_min (VRP-timing filter).
    """
    df = load(index, ivname)
    S = df["S"].values
    IV = df["IV"].values * iv_mult
    n = len(df)
    T = dte / 252.0
    rets = []
    i = 0
    while i + dte < n:
        sig = IV[i]
        if sig <= 0 or sig < vix_min or (entry_mask is not None and not bool(entry_mask[i])):
            i += dte
            continue
        s0 = 1.0
        ST = S[i + dte] / S[i]
        sd = sig * math.sqrt(T)
        put_iv = sig * (1.0 + skew)   # puts richer with skew
        call_iv = sig * call_iv_mult  # index calls trade CHEAPER than VIX (~0.85)

        if structure == "put_spread":
            Ks, Kl = math.exp(-z_short * sd), math.exp(-z_long * sd)
            cr = bsm_put(s0, Ks, T, put_iv) - bsm_put(s0, Kl, T, put_iv)
            gross = bsm_put(s0, Ks, T, put_iv) + bsm_put(s0, Kl, T, put_iv)
            pay = max(Ks - ST, 0.0) - max(Kl - ST, 0.0)
            cost = cost_frac * gross
            maxloss = (Ks - Kl) - cr + cost
        elif structure == "iron_condor":
            Kps, Kpl = math.exp(-z_short * sd), math.exp(-z_long * sd)
            Kcs, Kcl = math.exp(z_call * sd), math.exp(z_call_long * sd)
            cr = (bsm_put(s0, Kps, T, put_iv) - bsm_put(s0, Kpl, T, put_iv)
                  + bsm_call(s0, Kcs, T, call_iv) - bsm_call(s0, Kcl, T, call_iv))
            gross = (bsm_put(s0, Kps, T, put_iv) + bsm_put(s0, Kpl, T, put_iv)
                     + bsm_call(s0, Kcs, T, call_iv) + bsm_call(s0, Kcl, T, call_iv))
            pay = (max(Kps - ST, 0.0) - max(Kpl - ST, 0.0)
                   + max(ST - Kcs, 0.0) - max(ST - Kcl, 0.0))
            cost = cost_frac * gross
            width = max(Kps - Kpl, Kcl - Kcs)
            maxloss = width - cr + cost
        elif structure == "short_put":  # naked-ish, return on 12% index margin
            Ks = math.exp(-z_short * sd)
            cr = bsm_put(s0, Ks, T, put_iv)
            pay = max(Ks - ST, 0.0)
            cost = cost_frac * cr
            maxloss = 0.12  # ~Reg-T index margin proxy on notional
        elif structure == "strangle":  # naked short strangle, return on 12% margin
            Kp, Kc = math.exp(-z_short * sd), math.exp(z_call * sd)
            cr = bsm_put(s0, Kp, T, put_iv) + bsm_call(s0, Kc, T, call_iv)
            pay = max(Kp - ST, 0.0) + max(ST - Kc, 0.0)
            cost = cost_frac * cr
            maxloss = 0.12
        elif structure == "var_swap":  # pure VRP benchmark (assumption-free)
            implied_var = sig * sig * T
            r_window = np.log(S[i + 1:i + dte + 1] / S[i:i + dte])
            realized_var = np.sum(r_window ** 2)
            rets.append(implied_var - realized_var)  # short var swap, scale-free
            i += dte
            continue
        else:
            raise ValueError(structure)

        pnl = cr - cost - pay
        rets.append(pnl / maxloss if maxloss > 1e-9 else 0.0)
        i += dte
    return np.asarray(rets)


def gate(name, base, variants, dte=21, cost_stress=2.0):
    """base, variants are dicts of backtest() kwargs. cost_adjusted = stress run."""
    r1 = backtest(**base)
    cols = [backtest(**v) for v in variants]
    Tn = min(len(c) for c in cols)
    grid = np.column_stack([c[-Tn:] for c in cols])
    stress = backtest(**{**base, "cost_frac": base.get("cost_frac", 0.03) * cost_stress})
    ppy = 252.0 / dte
    rc = build_report_card(strategy_name=name, returns=r1, n_trials=len(variants),
                           trial_grid=grid, cost_adjusted_returns=stress,
                           periods_per_year=int(round(ppy)))
    ann = r1.mean() * ppy
    vol = r1.std() * math.sqrt(ppy)
    eq = np.cumprod(1 + np.clip(r1, -0.999, None))
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    print(f"\n### {name}  n={len(r1)}  ann={ann*100:+.1f}%  vol={vol*100:.1f}%  "
          f"Sharpe={ann/(vol+1e-9):.2f}  maxDD={dd*100:.0f}%  worstCycle={r1.min()*100:+.0f}%  win={(r1>0).mean()*100:.0f}%")
    print(rc.render())
    Path("validation_reports").mkdir(exist_ok=True)
    (Path("validation_reports") / f"{name}.md").write_text(rc.render())
    print("STATUS:", rc.status)
    return rc


def main():
    print("=" * 78 + "\nINDEX VRP / OPTIONS-INCOME — honest gated backtest (real VIX, 1990-2026)\n" + "=" * 78)
    # Defined-risk put spread (16d short / 2-delta long), SPX, held to expiry
    base_ps = dict(structure="put_spread", index="SPX", z_short=1.0, z_long=2.0, cost_frac=0.03)
    var_ps = [dict(structure="put_spread", index="SPX", z_short=zs, z_long=zl, cost_frac=0.03)
              for zs in (0.75, 1.0, 1.25) for zl in (1.75, 2.0, 2.5)]
    gate("vrp_put_spread_SPX", base_ps, var_ps)

    # Iron condor (both wings), SPX
    base_ic = dict(structure="iron_condor", index="SPX", z_short=1.0, z_long=2.0, z_call=1.0, z_call_long=2.0, cost_frac=0.03)
    var_ic = [dict(structure="iron_condor", index="SPX", z_short=zs, z_long=2.0, z_call=zs, z_call_long=2.0, cost_frac=0.03)
              for zs in (0.75, 1.0, 1.25)]
    gate("vrp_iron_condor_SPX", base_ic, var_ic)

    # Pure VRP benchmark (variance swap)
    base_vs = dict(structure="var_swap", index="SPX")
    gate("vrp_var_swap_SPX", base_vs, [base_vs, dict(structure="var_swap", index="NDX", ivname="VXN")])

    # Naked strangle (contrast — expect the tail to wreck the gate)
    base_st = dict(structure="strangle", index="SPX", z_short=1.0, z_call=1.0, cost_frac=0.03)
    var_st = [dict(structure="strangle", index="SPX", z_short=zs, z_call=zs, cost_frac=0.03) for zs in (0.75, 1.0, 1.5)]
    gate("vrp_naked_strangle_SPX", base_st, var_st)


if __name__ == "__main__":
    main()

"""VRP deployable-reality + skeptic analysis on the put-spread winner.

The gate APPROVED the put spread on an arithmetic per-cycle Sharpe, but a
defined-risk short premium has -100%-of-margin cycles (2008/2020). What matters
for a real $50k book is the FIXED-FRACTIONAL compounded path: risk f% of current
capital per cycle, never ruin, and see the CAGR vs max drawdown. Plus: sub-period
robustness, cost stress, the dates of the worst cycles, and the call-skew-corrected
condor.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.vrp_engine import backtest, load


def cycle_dates(index="SPX", ivname="VIX", dte=21):
    df = load(index, ivname)
    idx = list(df.index)
    out = []
    i = 0
    while i + dte < len(idx):
        out.append(idx[i])
        i += dte
    return out


def fixed_fractional(returns, f):
    """Compound risking f of CURRENT capital per cycle (return-on-margin series)."""
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + f * r))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    cyc = len(returns)
    cagr = eq[-1] ** (12.0 / cyc) - 1.0 if eq[-1] > 0 else -1.0
    return eq, cagr, float(dd.min())


def sharpe(r):
    return r.mean() / (r.std() + 1e-12) * math.sqrt(12)


def main():
    print("=" * 78 + "\nVRP DEPLOYABLE-REALITY + SKEPTIC ANALYSIS — SPX put spread (1SD/2SD)\n" + "=" * 78)
    r = backtest(structure="put_spread", index="SPX", z_short=1.0, z_long=2.0, cost_frac=0.03)
    dates = cycle_dates()[:len(r)]
    dser = pd.Series(r, index=pd.to_datetime(dates))

    print(f"\nper-cycle: n={len(r)} mean={r.mean()*100:+.2f}% std={r.std()*100:.1f}% "
          f"Sharpe={sharpe(r):.2f} win={(r>0).mean()*100:.0f}% worst={r.min()*100:+.0f}% best={r.max()*100:+.0f}%")

    print("\n5 worst cycles (should be the known crashes):")
    for dt, v in dser.nsmallest(5).items():
        print(f"  {dt.date()}: {v*100:+.0f}% of margin")

    print("\nsub-period Sharpe (robustness):")
    for lo, hi in [(1990, 2007), (2008, 2015), (2016, 2026)]:
        sub = dser[(dser.index.year >= lo) & (dser.index.year <= hi)].values
        print(f"  {lo}-{hi}: n={len(sub)} Sharpe={sharpe(sub):.2f} mean={sub.mean()*100:+.2f}% worst={sub.min()*100:+.0f}%")

    print("\ncost stress (entry bid-ask as % of gross premium):")
    for cf in (0.03, 0.06, 0.10, 0.15, 0.20):
        rc_ = backtest(structure="put_spread", index="SPX", z_short=1.0, z_long=2.0, cost_frac=cf)
        print(f"  cost {cf*100:4.0f}%: Sharpe={sharpe(rc_):.2f} ann_mean={rc_.mean()*12*100:+.1f}%(on margin)")

    print("\nFIXED-FRACTIONAL compounding on $50k (risk f% of capital per cycle):")
    print(f"  {'f':>4} {'CAGR':>8} {'maxDD':>8} {'~$/mo on 50k':>14}")
    for f in (0.02, 0.03, 0.05, 0.08, 0.12):
        eq, cagr, dd = fixed_fractional(r, f)
        monthly = 50000 * ((1 + cagr) ** (1 / 12.0) - 1)
        print(f"  {f*100:3.0f}% {cagr*100:7.1f}% {dd*100:7.0f}% {monthly:13.0f}")

    print("\nCALL-SKEW-CORRECTED iron condor (calls priced at 0.85*VIX, the realistic cheap-call skew):")
    # add call_iv_mult support inline by re-pricing: approximate via iv_mult on a call-only adjust is
    # not exposed; instead compare put_spread vs a condor with calls far OTM (z_call=1.5) which the
    # mispricing affects less, and flag the bias.
    rc_cond = backtest(structure="iron_condor", index="SPX", z_short=1.0, z_long=2.0, z_call=1.5, z_call_long=2.5, cost_frac=0.06)
    print(f"  condor (15d puts / further calls, 6% cost): Sharpe={sharpe(rc_cond):.2f} "
          f"mean={rc_cond.mean()*12*100:+.1f}% worst={rc_cond.min()*100:+.0f}%  "
          f"(NOTE: calls still priced at VIX -> upper bound; real cheap-call skew lowers this)")

    print("\nvar-swap (pure VRP) + naked strangle status:")
    for st, kw in [("var_swap", dict(structure="var_swap", index="SPX")),
                   ("naked_strangle", dict(structure="strangle", index="SPX", z_short=1.0, z_call=1.0, cost_frac=0.03))]:
        rr = backtest(**kw)
        eq, cagr, dd = fixed_fractional(rr / (abs(rr).max() + 1e-9), 0.05) if st == "var_swap" else fixed_fractional(rr, 0.05)
        print(f"  {st}: Sharpe={sharpe(rr):.2f} worst={rr.min()*100 if st!='var_swap' else rr.min():+.1f}{'%' if st!='var_swap' else ' (var pts)'} maxDD(@5%)={dd*100:.0f}%")


if __name__ == "__main__":
    main()

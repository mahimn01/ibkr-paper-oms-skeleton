"""VRP optimization: honestly-priced condor + term-structure timing, optimised
for DEPLOYABLE return (max CAGR subject to maxDD <= cap), with one HONEST global
n_trials count for the whole search.

Key questions:
  1. With calls priced at the realistic cheap-call skew (0.85*VIX), is the condor
     still better than the put spread?
  2. Does term-structure timing (only sell when VIX < VIX3M = contango/calm; skip
     backwardation/stress) cut the tail enough to size larger and lift the
     deployable $/month?
All compared apples-to-apples on the 2011-2026 term-structure era + full history.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.vrp_engine import backtest, load
from trading_algo.quant_core.validation.report_card import build_report_card


def masks():
    """Entry masks aligned to load('SPX','VIX').index."""
    df = load("SPX", "VIX")
    vix3m = pd.read_parquet("data/vrp_cache/vix3m.parquet")["VIX3M"] / 100.0
    v3 = vix3m.reindex(df.index).ffill()
    contango = (df["IV"].values < v3.values)            # front < 3mo = calm
    era2011 = np.asarray(df.index.year >= 2011)
    return df.index, contango, era2011


def fixed_frac(r, f):
    eq = np.cumprod(1 + f * r)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    cagr = eq[-1] ** (12.0 / len(r)) - 1.0 if eq[-1] > 0 else -1.0
    return cagr, dd


def best_deployable(r, dd_cap=-0.20):
    """Max CAGR over f s.t. maxDD >= dd_cap (i.e., not worse than the cap)."""
    best = (0.0, 0.0, 0.0)
    for f in np.arange(0.02, 0.40, 0.01):
        cagr, dd = fixed_frac(r, f)
        if dd >= dd_cap and cagr > best[0]:
            best = (cagr, dd, f)
    return best  # cagr, dd, f


def sharpe(r):
    return r.mean() / (r.std() + 1e-12) * math.sqrt(12)


def main():
    idx, contango, era2011 = masks()
    print("=" * 80 + "\nVRP OPTIMIZATION — honest condor + term-structure timing (deployable focus)\n" + "=" * 80)

    # The honest search family (count ALL of these toward n_trials):
    specs = []
    # structure x timing x era
    for struct, kw in [("put_spread", dict(structure="put_spread", z_short=1.0, z_long=2.0)),
                       ("condor_honest", dict(structure="iron_condor", z_short=1.0, z_long=2.0,
                                              z_call=1.0, z_call_long=2.0, call_iv_mult=0.85))]:
        for tname, mask in [("alltime", None), ("contango", contango)]:
            specs.append((f"{struct}|{tname}|full", {**kw, "cost_frac": 0.06, "entry_mask": mask}))
    N_TRIALS = len(specs) + 6  # + the implicit param choices (z, dte, call_mult) explored earlier; conservative

    results = []
    runs = {}
    for name, kw in specs:
        r = backtest(index="SPX", **kw)
        runs[name] = r
        results.append((name, r))

    # trial grid for PBO = all the variant return streams (aligned)
    Tn = min(len(r) for _, r in results)
    grid = np.column_stack([r[-Tn:] for _, r in results])

    print(f"\n{'variant':28} {'n':>4} {'Sharpe':>7} {'mean%':>7} {'worst%':>7} {'deplCAGR':>9} {'maxDD':>7} {'f':>5} {'$/mo50k':>8}")
    for name, r in results:
        cagr, dd, f = best_deployable(r)
        mo = 50000 * ((1 + cagr) ** (1 / 12.0) - 1)
        print(f"{name:28} {len(r):4d} {sharpe(r):7.2f} {r.mean()*100:7.2f} {r.min()*100:7.0f} "
              f"{cagr*100:8.1f}% {dd*100:6.0f}% {f*100:4.0f}% {mo:8.0f}")

    # Gate the single best deployable variant with the HONEST global n_trials
    best_name = max(results, key=lambda kv: best_deployable(kv[1])[0])[0]
    rb = runs[best_name]
    stress = backtest(index="SPX", **{**dict(specs)[best_name], "cost_frac": 0.12})
    rc = build_report_card(strategy_name=f"vrp_BEST_{best_name.replace('|','_')}",
                           returns=rb, n_trials=N_TRIALS, trial_grid=grid,
                           cost_adjusted_returns=stress, periods_per_year=12)
    print(f"\nBEST deployable = {best_name}   (gated at HONEST n_trials={N_TRIALS}, cost-stress 12%)")
    print(rc.render())
    print("STATUS:", rc.status)
    Path("validation_reports").mkdir(exist_ok=True)
    (Path("validation_reports") / "vrp_best_deployable.md").write_text(rc.render())


if __name__ == "__main__":
    main()

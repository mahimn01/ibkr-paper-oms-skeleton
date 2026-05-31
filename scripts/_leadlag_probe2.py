"""Deeper probe on the two survivors: TLT->QQQ lead, and a long-only
credit-defensive overlay. Build actual net-of-cost return streams and check
subsample stability (the real killer for these).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._leadlag_probe import align, logret, ann_sharpe, newey_t  # noqa: E402


def subsamples(r: np.ndarray, n: int = 4):
    out = []
    k = len(r) // n
    for i in range(n):
        sl = r[i * k:(i + 1) * k] if i < n - 1 else r[i * k:]
        out.append(ann_sharpe(sl))
    return out


print("=" * 70)
print("TLT->QQQ lead: is it tradeable net of cost? (long QQQ after TLT up-day)")
print("=" * 70)
days, d = align("TLT", "QQQ")
r_tlt = logret(d["TLT"])
r_qqq = logret(d["QQQ"])
# strategy: position in QQQ tomorrow = sign-weighted by today's TLT return
# variant A: long QQQ only when TLT was up today (rates fell)
x = r_tlt[:-1]
y = r_qqq[1:]
pos_long = (x > 0).astype(float)
strat = pos_long * y
flips = np.abs(np.diff(np.concatenate([[0], pos_long])))
cost = flips * 0.00015  # 1.5bp QQQ one-way
net = strat - cost
print(f"  long-QQQ-after-TLT-up: gross SR={ann_sharpe(strat):.2f} net SR={ann_sharpe(net):.2f} "
      f"time_in={pos_long.mean()*100:.0f}%  buyhold QQQ SR={ann_sharpe(r_qqq):.2f}")
print(f"  net subsample SRs (4 blocks): {[f'{s:.2f}' for s in subsamples(net)]}")

# variant B: continuous tilt — scale QQQ exposure by yesterday's TLT z-score (capped)
z = (x - x.mean()) / x.std()
expo = np.clip(z, -1, 1)  # -1..1
strat_b = expo * y
turn = np.abs(np.diff(np.concatenate([[0], expo])))
net_b = strat_b - turn * 0.00015
print(f"  TLT-z-tilt QQQ:        gross SR={ann_sharpe(strat_b):.2f} net SR={ann_sharpe(net_b):.2f}")
print(f"  net_b subsample SRs:   {[f'{s:.2f}' for s in subsamples(net_b)]}")

# variant C: market-neutral — long QQQ / short SPY scaled by TLT signal (the lead is duration-specific)
days2, d2 = align("TLT", "QQQ", "SPY")
r_tlt2 = logret(d2["TLT"]); r_qqq2 = logret(d2["QQQ"]); r_spy2 = logret(d2["SPY"])
x2 = r_tlt2[:-1]
z2 = np.clip((x2 - x2.mean()) / x2.std(), -1, 1)
# when rates fall (TLT up), growth (QQQ) should outperform value/broad (SPY)
ls = z2 * (r_qqq2[1:] - r_spy2[1:])
turn2 = np.abs(np.diff(np.concatenate([[0], z2])))
ls_net = ls - turn2 * 0.0003  # both legs
print(f"  L/S QQQ-SPY on TLT-z:  gross SR={ann_sharpe(ls):.2f} net SR={ann_sharpe(ls_net):.2f}")
print(f"  ls_net subsample SRs:  {[f'{s:.2f}' for s in subsamples(ls_net)]}")

print()
print("=" * 70)
print("Robustness: does TLT->QQQ lead hold in BOTH halves separately?")
print("=" * 70)
n = len(x)
for label, sl in [("H1", slice(0, n // 2)), ("H2", slice(n // 2, n))]:
    b, t, r2 = newey_t(y[sl], x[sl])
    print(f"  {label}: beta={b:+.3f} NW-t={t:+.2f} R2={r2*1e4:.1f}bp")

print()
print("=" * 70)
print("Credit-defensive overlay as a RETURN STREAM for the gate (5d HYG/LQD mom)")
print("=" * 70)
days3, d3 = align("HYG", "LQD", "SPY")
ratio = d3["HYG"] / d3["LQD"]
r_spy3 = logret(d3["SPY"])
pos = np.ones(len(r_spy3))
for i in range(5, len(r_spy3)):
    mom = ratio[i] / ratio[i - 5] - 1.0
    pos[i] = 1.0 if mom > -0.005 else 0.0
strat3 = pos * r_spy3
flips3 = np.abs(np.diff(np.concatenate([[0], pos])))
net3 = strat3 - flips3 * 0.00015
print(f"  defensive net SR={ann_sharpe(net3):.2f} buyhold SR={ann_sharpe(r_spy3):.2f}")
print(f"  net3 subsample SRs: {[f'{s:.2f}' for s in subsamples(net3)]}")
# excess over buy-hold (the actual 'alpha' of the overlay)
excess = net3 - r_spy3
print(f"  EXCESS (overlay - B&H) SR={ann_sharpe(excess):.2f}  (this is what must clear the gate if framed as alpha)")

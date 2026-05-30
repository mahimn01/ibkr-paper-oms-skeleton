"""Final probe: (a) is TLT->QQQ lead concentrated in high-vol days only?
(b) gold/TLT safe-haven rotation. Establishes capacity + decay story."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._leadlag_probe import align, logret, ann_sharpe, newey_t  # noqa: E402

days, d = align("TLT", "QQQ")
r_tlt = logret(d["TLT"]); r_qqq = logret(d["QQQ"])
x = r_tlt[:-1]; y = r_qqq[1:]

# Split by magnitude of TLT move: is the lead only in big-rate-move days?
absx = np.abs(x)
thr = np.quantile(absx, 0.80)
big = absx >= thr
print("TLT->QQQ lead conditional on TLT-move size:")
for lbl, m in [("big-TLT-move (top20%)", big), ("small (bot80%)", ~big)]:
    b, t, r2 = newey_t(y[m], x[m])
    print(f"  {lbl}: n={m.sum()} beta={b:+.3f} NW-t={t:+.2f}")

# Recent-only (last 2 years ~ 504d): is there ANYTHING left?
print("\nLast ~504 trading days only (2024-2026):")
b, t, r2 = newey_t(y[-504:], x[-504:])
print(f"  beta={b:+.3f} NW-t={t:+.2f} R2={r2*1e4:.1f}bp")

# GLD/TLT defensive: does GLD lead anything risk-off?
print("\nGLD->SPY and GLD->IWM next-day lead:")
days2, d2 = align("GLD", "SPY", "IWM")
rg = logret(d2["GLD"])
for tgt in ["SPY", "IWM"]:
    rt = logret(d2[tgt])
    b, t, r2 = newey_t(rt[1:], rg[:-1])
    print(f"  GLD->{tgt}: beta={b:+.3f} NW-t={t:+.2f}")

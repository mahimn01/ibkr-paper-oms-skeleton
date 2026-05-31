"""Empirical lead-lag probes for cross-asset/macro hypotheses.

Uses only on-disk ETF daily bars (FRED MCP returns empty). Computes the raw
predictive content BEFORE any optimisation, so we know if signal exists at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._options_data import load_daily_bars  # noqa: E402


def series(sym: str) -> tuple[np.ndarray, np.ndarray]:
    bars = load_daily_bars(sym)
    ts = np.array([b.timestamp_epoch_s for b in bars])
    c = np.array([b.close for b in bars])
    return ts, c


def align(*syms: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = {s: series(s) for s in syms}
    # intersect on date (day resolution)
    daysets = []
    for s in syms:
        ts, _ = data[s]
        days = (ts // 86400).astype(int)
        daysets.append(set(days.tolist()))
    common = sorted(set.intersection(*daysets))
    common_set = set(common)
    out: dict[str, np.ndarray] = {}
    for s in syms:
        ts, c = data[s]
        days = (ts // 86400).astype(int)
        mask = np.array([d in common_set for d in days])
        # dedupe to one close per common day (last)
        dd = days[mask]
        cc = c[mask]
        m: dict[int, float] = {}
        for d, v in zip(dd, cc):
            m[d] = v
        out[s] = np.array([m[d] for d in common])
    return np.array(common), out


def logret(p: np.ndarray) -> np.ndarray:
    return np.diff(np.log(p))


def ann_sharpe(r: np.ndarray, ppy: int = 252) -> float:
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return r.mean() / sd * np.sqrt(ppy)


def newey_t(y: np.ndarray, x: np.ndarray, lags: int = 5) -> tuple[float, float, float]:
    """OLS y~x with HAC (Newey-West) t-stat on slope. Returns (beta, t, r2)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    # NW meat
    S = np.zeros((k, k))
    u = X * resid[:, None]
    for l in range(lags + 1):
        w = 1.0 - l / (lags + 1)
        for t in range(l, n):
            g_t = np.outer(u[t], u[t - l])
            if l == 0:
                S += g_t
            else:
                S += w * (g_t + g_t.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t_slope = beta[1] / se[1]
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return float(beta[1]), float(t_slope), float(r2)


print("=" * 70)
print("PROBE 1: Credit (HYG/LQD ratio) daily return -> next-day SPY/IWM return")
print("=" * 70)
days, d = align("HYG", "LQD", "SPY", "IWM", "QQQ")
print(f"common days: {len(days)}  ({len(days)/252:.1f} yrs)")
hyg, lqd = d["HYG"], d["LQD"]
ratio = hyg / lqd
r_ratio = logret(ratio)          # credit risk appetite change (HYG up vs LQD = risk-on)
for tgt in ["SPY", "IWM", "QQQ"]:
    r_tgt = logret(d[tgt])
    # predictor = today's ratio change, target = tomorrow's equity return
    x = r_ratio[:-1]
    y = r_tgt[1:]
    b, t, r2 = newey_t(y, x)
    # contemporaneous corr for context
    cc = np.corrcoef(r_ratio, r_tgt)[0, 1]
    print(f"  {tgt}: lead beta={b:+.3f} NW-t={t:+.2f} R2={r2*1e4:.1f}bp  contemp_corr={cc:+.2f}")

print()
print("=" * 70)
print("PROBE 2: Simple credit-momentum timing strategy (long SPY when credit risk-on)")
print("=" * 70)
r_spy = logret(d["SPY"])
# signal: HYG/LQD ratio above its own 20d MA -> risk-on -> hold SPY, else cash
sig_raw = ratio[:-1]  # align to decision at close t, trade t+1
for win in [10, 20, 50]:
    ma = np.array([ratio[max(0, i - win):i].mean() if i >= 1 else ratio[0]
                   for i in range(1, len(ratio))])  # ma over prior win up to t-1
    # decision at day t (index i in r_spy maps to ret from t-1->t); use ratio[i] vs ma
    # build positions aligned to r_spy (len n-1)
    pos = np.zeros(len(r_spy))
    for i in range(1, len(r_spy)):
        w0 = max(0, i - win)
        ma_i = ratio[w0:i].mean()
        pos[i] = 1.0 if ratio[i] > ma_i else 0.0
    strat = pos * r_spy
    # one-way turnover cost: 1.5bp per flip
    flips = np.abs(np.diff(np.concatenate([[0], pos])))
    cost = flips * 0.00015
    strat_net = strat - cost
    bh = ann_sharpe(r_spy)
    print(f"  win={win:>2}: strat Sharpe={ann_sharpe(strat):.2f} "
          f"net={ann_sharpe(strat_net):.2f} (buyhold={bh:.2f}) "
          f"time_in_mkt={pos.mean()*100:.0f}%")

print()
print("=" * 70)
print("PROBE 3: Credit-stress DEFENSIVE filter (avoid SPY when credit deteriorating fast)")
print("=" * 70)
# signal: 5-day HYG/LQD ratio momentum < threshold -> go to cash/TLT
days2, d2 = align("HYG", "LQD", "SPY", "TLT")
hyg2, lqd2 = d2["HYG"], d2["LQD"]
ratio2 = hyg2 / lqd2
r_spy2 = logret(d2["SPY"])
r_tlt2 = logret(d2["TLT"])
for lookback in [5, 10, 20]:
    pos = np.ones(len(r_spy2))
    for i in range(lookback, len(r_spy2)):
        mom = ratio2[i] / ratio2[i - lookback] - 1.0
        pos[i] = 1.0 if mom > -0.005 else 0.0  # exit if credit fell >0.5% over lookback
    strat = pos * r_spy2
    print(f"  lookback={lookback:>2}: defensive Sharpe={ann_sharpe(strat):.2f} "
          f"(buyhold={ann_sharpe(r_spy2):.2f}) time_in={pos.mean()*100:.0f}% "
          f"maxDD_strat={(np.minimum.accumulate(np.cumsum(strat)-np.maximum.accumulate(np.cumsum(strat)))).min():.2f} "
          f"maxDD_bh={(np.minimum.accumulate(np.cumsum(r_spy2)-np.maximum.accumulate(np.cumsum(r_spy2)))).min():.2f}")

print()
print("=" * 70)
print("PROBE 4: TLT (rates) daily move -> next-day QQQ (duration-sensitive growth)")
print("=" * 70)
days3, d3 = align("TLT", "QQQ", "IWM", "GLD")
r_tlt3 = logret(d3["TLT"])
for tgt in ["QQQ", "IWM", "GLD"]:
    r_t = logret(d3[tgt])
    x = r_tlt3[:-1]
    y = r_t[1:]
    b, t, r2 = newey_t(y, x)
    cc = np.corrcoef(r_tlt3, r_t)[0, 1]
    print(f"  TLT->{tgt}: lead beta={b:+.3f} NW-t={t:+.2f} R2={r2*1e4:.1f}bp contemp_corr={cc:+.2f}")

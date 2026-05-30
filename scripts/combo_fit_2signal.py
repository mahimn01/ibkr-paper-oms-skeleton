"""Decisive 2-signal read: insider Form-4 (sparse event) + short-interest (dense
level) on a monthly R3000 grid, fused by LightGBM under purged walk-forward.

Falsification gates (per the combination pre-registration): the combination is
only interesting if (1) combined OOS rank-IC EXCEEDS the best single signal's
OOS IC, and (2) no leave-one-out recovers >80% of the edge. Honest expectation
(from the IC-to-gate calibration): the sparse insider event signal gets diluted
across the broad cross-section, so a clear-the-gate result is unlikely — the
real read is whether combining genuinely ADDS cross-sectional information.

Look-ahead discipline: insider fires from filingDate forward (decaying); SIR is
lagged +8 cal-days from FINRA settlement then forward-filled; signals missing on
a date -> NaN (LightGBM-native), never zero. Label = forward-21d market-residual
(cross-sectional demean) return rank.
"""

from __future__ import annotations

import bisect
import glob
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from scipy.stats import spearmanr

from scripts.xs_research import load_r3000_panel
from scripts.purged_cv import purged_walk_forward
from trading_algo.quant_core.validation.report_card import build_report_card

FWD = 21
REBAL = 21
DECILE = 0.10


def _date_list(epoch):
    return [datetime.fromtimestamp(int(t), tz=timezone.utc).date() for t in epoch]


def build_insider_panel(date_list, symbols, halflife=21.0):
    sym_idx = {s: j for j, s in enumerate(symbols)}
    T, N = len(date_list), len(symbols)
    panel = np.full((T, N), np.nan)
    span = int(halflife * 4)
    for f in glob.glob("data/insider_form4_cache/*_purchases.json"):
        recs = json.load(open(f))
        if not recs:
            continue
        sym = recs[0].get("symbol")
        if sym not in sym_idx:
            continue
        events = [(datetime.fromisoformat(r["filingDate"]).date(), float(r["value"]))
                  for r in recs
                  if (r.get("is_officer") or r.get("is_director")) and not r.get("is_ten")
                  and r.get("value", 0) and float(r["value"]) > 0]
        if not events:
            continue
        j = sym_idx[sym]
        col = np.full(T, np.nan)
        for ed, val in events:
            lo = bisect.bisect_left(date_list, ed)
            w = np.log1p(val)
            for t in range(lo, min(T, lo + span)):
                ds = (date_list[t] - ed).days
                c = np.exp(-ds / halflife) * w
                col[t] = c if np.isnan(col[t]) else col[t] + c
        panel[:, j] = col
    return panel


def build_sir_panel(date_list, symbols, pub_lag_days=8):
    df = pd.read_parquet("data/short_interest_cache/finra_si.parquet")
    df["settlementDate"] = pd.to_datetime(df["settlementDate"]).dt.date
    df["pub"] = df["settlementDate"].map(lambda d: d + timedelta(days=pub_lag_days))
    sym_idx = {s: j for j, s in enumerate(symbols)}
    T, N = len(date_list), len(symbols)
    panel = np.full((T, N), np.nan)
    for sym, g in df.groupby("symbolCode"):
        if sym not in sym_idx:
            continue
        j = sym_idx[sym]
        g = g.sort_values("pub")
        pubs = list(g["pub"])
        vals = list(g["daysToCoverQuantity"].astype(float))
        pi, cur = 0, np.nan
        for t in range(T):
            while pi < len(pubs) and pubs[t if False else pi] <= date_list[t]:
                cur = vals[pi]
                pi += 1
            panel[t, j] = cur
    return panel


def _xs_z(row):
    mu = np.nanmean(row)
    sd = np.nanstd(row)
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    z = (row - mu) / sd
    return np.clip(z, -3, 3)


def build_samples(date_list, closes, signals: dict):
    T, N = closes.shape
    fwd = np.full((T, N), np.nan)
    fwd[:T - FWD] = closes[FWD:] / closes[:T - FWD] - 1.0
    names = list(signals.keys())
    rebal = list(range(252, T - FWD, REBAL))
    samples = []
    for t in rebal:
        valid = np.isfinite(closes[t]) & np.isfinite(fwd[t])
        if valid.sum() < 50:
            continue
        raw = fwd[t]
        resid = raw - np.nanmean(raw[valid])
        cols = [_xs_z(signals[n][t]) for n in names]
        X = np.column_stack(cols)
        samples.append({"valid": valid, "X": X, "resid": resid, "raw": raw})
    return samples, names


PARAMS = {"objective": "regression", "num_leaves": 15, "min_child_samples": 100,
          "learning_rate": 0.03, "lambda_l2": 5.0, "verbose": -1, "_rounds": 120}


def run_book(samples, splits, params, cols_use, cost_bps=30.0):
    monthly, ics, turn = [], [], []
    prev_w = None
    Nfull = len(samples[0]["valid"])
    for sp in splits:
        tr = [samples[i] for i in sp.train]
        Xtr = np.vstack([s["X"][s["valid"]][:, cols_use] for s in tr])
        ytr = np.concatenate([s["resid"][s["valid"]] for s in tr])
        m = lgb.train(params, lgb.Dataset(Xtr, ytr, free_raw_data=False),
                      num_boost_round=params.get("_rounds", 120))
        for i in sp.test:
            s = samples[i]; v = s["valid"]
            pred = m.predict(s["X"][v][:, cols_use])
            names_idx = np.where(v)[0]; raw = s["raw"][v]
            if len(pred) > 10 and np.isfinite(raw).all():
                ics.append(spearmanr(pred, raw).statistic)
            k = max(1, int(DECILE * len(pred)))
            order = np.argsort(pred)
            longs, shorts = names_idx[order[-k:]], names_idx[order[:k]]
            w = np.zeros(Nfull); w[longs] = 1.0 / k; w[shorts] = -1.0 / k
            ret = float(np.mean(s["raw"][longs]) - np.mean(s["raw"][shorts]))
            turn.append(float(np.abs(w - (prev_w if prev_w is not None else 0)).sum()))
            prev_w = w; monthly.append(ret)
    monthly = np.asarray(monthly); turn = np.asarray(turn)
    net = monthly - turn * (cost_bps / 1e4)
    return net, np.asarray(ics)


def single_ic(samples, col):
    acc = []
    for s in samples:
        v = s["valid"]; raw = s["raw"][v]; x = s["X"][v, col]
        m = np.isfinite(x) & np.isfinite(raw)
        if m.sum() > 10:
            acc.append(spearmanr(x[m], raw[m]).statistic)
    return float(np.nanmean(acc)) if acc else float("nan")


def main() -> int:
    print("loading broad R3000 panel (2019+, SIR-covered era) ...")
    epoch, closes, symbols = load_r3000_panel(start="2019-01-01", end="2026-04-01",
                                              min_price=3.0, min_obs=800, dv_lo_pct=2, dv_hi_pct=98)
    dl = _date_list(epoch)
    print(f"panel: {closes.shape[0]} days x {len(symbols)} names")
    ins = build_insider_panel(dl, symbols)
    sir = build_sir_panel(dl, symbols)
    print(f"insider coverage: {np.isfinite(ins).any(0).sum()} names ever-signalled; "
          f"SIR coverage: {np.isfinite(sir).any(0).sum()} names")

    samples, names = build_samples(dl, closes, {"insider": ins, "sir": sir})
    print(f"{len(samples)} monthly rebalances; signals={names}")
    splits = purged_walk_forward(len(samples), n_splits=5, label_horizon=1, embargo=1, min_train=18)

    ic_ins = single_ic(samples, 0)
    ic_sir = single_ic(samples, 1)
    print(f"standalone xs IC: insider {ic_ins:+.4f}  sir {ic_sir:+.4f}")

    # OOS IC for each single-signal fit + the combination
    _, oos_ins = run_book(samples, splits, PARAMS, [0])
    _, oos_sir = run_book(samples, splits, PARAMS, [1])
    net, oos_both = run_book(samples, splits, PARAMS, [0, 1])
    ic_b = np.nanmean(oos_both); ic_i = np.nanmean(oos_ins); ic_s = np.nanmean(oos_sir)
    best_single = max(ic_i, ic_s)
    print(f"OOS rank-IC: insider-only {ic_i:+.4f}  sir-only {ic_s:+.4f}  COMBINED {ic_b:+.4f}")
    print(f"FALSIFICATION GATE 1 (combined > best single): {'PASS' if ic_b > best_single else 'FAIL'} "
          f"(combined {ic_b:+.4f} vs best single {best_single:+.4f})")

    # gate the combination
    variants = [{**PARAMS, "num_leaves": nl, "min_child_samples": mc}
                for nl in (7, 15) for mc in (50, 100, 200)]
    cols = [run_book(samples, splits, p, [0, 1])[0] for p in variants]
    Tn = min(len(c) for c in cols); grid = np.column_stack([c[-Tn:] for c in cols])
    stress, _ = run_book(samples, splits, PARAMS, [0, 1], cost_bps=60.0)
    rc = build_report_card(strategy_name="combo_insider_sir_r3000", returns=net,
                           n_trials=len(variants) + 2, trial_grid=grid,
                           cost_adjusted_returns=stress, periods_per_year=12)
    ann = net.mean() * 12; vol = net.std() * np.sqrt(12)
    print(f"\ncombo net ann {ann*100:+.1f}%  Sharpe {ann/(vol+1e-9):.2f}  n_obs {net.size}")
    print(rc.render())
    Path("validation_reports").mkdir(exist_ok=True)
    (Path("validation_reports") / "combo_insider_sir_r3000.md").write_text(rc.render())
    print("STATUS:", rc.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

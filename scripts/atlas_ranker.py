"""ATLAS pivot prototype: cross-sectional LightGBM ranker with purged walk-forward.

Replaces the scrapped v7 RL stack with the GKX supervised paradigm:
  features (per name, per rebalance) -> LightGBM -> predicted forward residual
  return -> dollar-neutral decile L/S book -> hardened build_report_card.

This is the PRICE-ONLY CONTROL on the 444 large-cap set (less survivorship-
distorted than R3000). The research predicts it is BLOCKED net-of-cost; its job
is to validate the pipeline honestly and measure baseline rank-IC, so the
alt-data (insider Form-4) features can later be added to the SAME apparatus and
the incremental edge isolated.

Honest methodology: purged+embargoed walk-forward (scripts/purged_cv), OOS test
blocks only, monthly (periods_per_year=12), realistic costs, honest n_trials.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from scipy.stats import spearmanr

from scripts.purged_cv import purged_walk_forward
from trading_algo.quant_core.validation.report_card import build_report_card

FEATURE_DIR = "data/atlas_features_v3"
FWD = 21          # forward-return label horizon (trading days)
REBAL = 21        # non-overlapping monthly rebalance
DECILE = 0.10     # top/bottom decile L/S


def load_feature_panel(feature_dir: str = FEATURE_DIR, min_obs: int = 800):
    """Return dates(T,), closes(T,N), feats(T,N,F), symbols. Aligned on a master grid."""
    files = sorted(glob.glob(os.path.join(feature_dir, "*_features.npz")))
    per_sym = {}
    all_ts: set[int] = set()
    F = None
    for f in files:
        sym = os.path.basename(f).replace("_features.npz", "")
        z = np.load(f)
        ts = z["timestamps"].astype(np.float64)
        cl = z["closes"].astype(np.float64)
        nm = z["normed"].astype(np.float64)  # (T, F)
        if len(ts) < min_obs or len(nm) != len(ts):
            continue
        F = nm.shape[1]
        per_sym[sym] = {int(round(t)): (float(c), nm[k]) for k, (t, c) in enumerate(zip(ts, cl))
                        if np.isfinite(c) and c > 0}
        all_ts.update(per_sym[sym].keys())

    dates = np.array(sorted(all_ts), dtype=np.int64)
    idx = {t: i for i, t in enumerate(dates)}
    symbols = sorted(per_sym.keys())
    T, N = len(dates), len(symbols)
    closes = np.full((T, N), np.nan)
    feats = np.full((T, N, F), np.nan)
    for j, sym in enumerate(symbols):
        for t, (c, vec) in per_sym[sym].items():
            i = idx[t]
            closes[i, j] = c
            feats[i, j] = vec
    return dates, closes, feats, symbols


def _xs_zscore(row: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score across names (axis over valid entries), NaN-safe."""
    mu = np.nanmean(row, axis=0)
    sd = np.nanstd(row, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (row - mu) / sd


def build_dataset(dates, closes, feats):
    """Return rebalance list with per-rebalance (X, fwd_resid, fwd_raw, valid_names)."""
    T, N, F = feats.shape
    fwd = np.full((T, N), np.nan)
    fwd[:T - FWD] = closes[FWD:] / closes[:T - FWD] - 1.0  # t -> t+FWD raw return
    rebal = list(range(252, T - FWD, REBAL))  # leave warmup + room for the label
    samples = []
    for t in rebal:
        valid = np.isfinite(closes[t]) & np.isfinite(fwd[t]) & np.all(np.isfinite(feats[t]), axis=1)
        if valid.sum() < 50:
            continue
        Xt = _xs_zscore(feats[t])           # cross-sectional standardize features per date
        raw = fwd[t]
        resid = raw - np.nanmean(raw[valid])  # market-neutral (cross-sectional demean) label
        samples.append({"t": t, "valid": valid, "X": Xt, "resid": resid, "raw": raw})
    return samples


def run_book(samples, splits, params, cost_bps=10.0):
    """Train per split, predict OOS test rebalances, form L/S book, return monthly returns + IC."""
    monthly = []
    ics = []
    prev_w = None
    turn = []
    for sp in splits:
        tr = [samples[i] for i in sp.train]
        Xtr = np.vstack([s["X"][s["valid"]] for s in tr])
        ytr = np.concatenate([s["resid"][s["valid"]] for s in tr])
        m = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=params.get("_rounds", 200))
        for i in sp.test:
            s = samples[i]
            v = s["valid"]
            pred = m.predict(s["X"][v])
            names = np.where(v)[0]
            raw = s["raw"][v]
            # rank-IC diagnostic
            if len(pred) > 10 and np.isfinite(raw).all():
                ics.append(spearmanr(pred, raw).statistic)
            k = max(1, int(DECILE * len(pred)))
            order = np.argsort(pred)
            longs, shorts = names[order[-k:]], names[order[:k]]
            w = np.zeros(len(samples[0]["valid"]))
            w[longs] = 1.0 / k
            w[shorts] = -1.0 / k
            # P&L = dollar-neutral L/S spread over the holding period
            ret = float(np.mean(s["raw"][longs]) - np.mean(s["raw"][shorts]))
            if prev_w is not None:
                turn.append(float(np.abs(w - prev_w).sum()))
            else:
                turn.append(float(np.abs(w).sum()))
            prev_w = w
            monthly.append(ret)
    monthly = np.asarray(monthly)
    turn = np.asarray(turn)
    net = monthly - turn * (cost_bps / 1e4)
    return net, monthly, np.asarray(ics), turn


def main() -> int:
    print("loading feature panel ...")
    dates, closes, feats, symbols = load_feature_panel()
    print(f"panel: {closes.shape[0]} days x {len(symbols)} names x {feats.shape[2]} features")
    samples = build_dataset(dates, closes, feats)
    print(f"{len(samples)} monthly rebalances")
    splits = purged_walk_forward(len(samples), n_splits=6, label_horizon=1, embargo=1, min_train=24)
    print(f"{len(splits)} purged walk-forward folds")

    base = {"objective": "regression", "num_leaves": 15, "min_child_samples": 200,
            "learning_rate": 0.03, "lambda_l2": 5.0, "feature_fraction": 0.8,
            "verbose": -1, "_rounds": 150}
    # small honest variant grid for PBO (depth x regularization)
    variants = [{**base, "num_leaves": nl, "min_child_samples": mc}
                for nl in (7, 15, 31) for mc in (100, 200, 400)]

    cols = []
    for p in variants:
        net, _, _, _ = run_book(samples, splits, p, cost_bps=10.0)
        cols.append(net)
    Tn = min(len(c) for c in cols)
    grid = np.column_stack([c[-Tn:] for c in cols])

    net, gross, ics, turn = run_book(samples, splits, base, cost_bps=10.0)
    stress, _, _, _ = run_book(samples, splits, base, cost_bps=25.0)

    print(f"\nOOS rank-IC: mean {np.nanmean(ics):+.4f}  (t≈{np.nanmean(ics)/(np.nanstd(ics)/np.sqrt(len(ics))+1e-9):.2f})  "
          f"| monthly turnover {turn.mean()*100:.0f}%  | gross ann {gross.mean()*12*100:+.1f}%  net ann {net.mean()*12*100:+.1f}%")

    rc = build_report_card(strategy_name="atlas_ranker_priceonly_largecap",
                           returns=net, n_trials=len(variants), trial_grid=grid,
                           cost_adjusted_returns=stress, periods_per_year=12)
    print(rc.render())
    out = Path("validation_reports"); out.mkdir(exist_ok=True)
    (out / "atlas_ranker_priceonly_largecap.md").write_text(rc.render())
    print("STATUS:", rc.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Multi-signal cross-sectional LightGBM ranker — the combination apparatus.

Fuses a base price-feature panel with an arbitrary set of extra signal panels
(insider Form-4, 13F accumulation, short interest, ...) into one feature matrix,
trains a LightGBM ranker under purged/embargoed walk-forward, forms a
dollar-neutral decile L/S book, and scores it through the hardened gate.

Two built-in controls validate the apparatus is BOTH honest and powerful:
  - price-only  -> must stay BLOCKED (rejects noise; matches atlas_ranker).
  - +oracle     -> a synthetic feature = forward-residual-return + heavy noise
                   (IC~0.1). Must clear the gate, proving the machinery DETECTS
                   a real edge if one exists. (Positive control only — never a
                   real backtest.)

Per-signal IC + leave-one-out ablation quantify each signal's contribution so we
can tell a genuine multi-signal edge from re-discovered beta or one dominant feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from scipy.stats import spearmanr

from scripts.atlas_ranker import load_feature_panel, FWD, REBAL, DECILE
from scripts.purged_cv import purged_walk_forward
from trading_algo.quant_core.validation.report_card import build_report_card


def _xs_z(row: np.ndarray) -> np.ndarray:
    mu = np.nanmean(row, axis=0)
    sd = np.nanstd(row, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (row - mu) / sd


def build_samples(dates, closes, base_feats, extra: dict[str, np.ndarray] | None = None,
                  add_oracle: bool = False, oracle_noise: float = 9.0, seed: int = 0):
    """Assemble per-rebalance samples. extra: {name: (T,N)} signal panels."""
    T, N, F = base_feats.shape
    extra = dict(extra or {})
    fwd = np.full((T, N), np.nan)
    fwd[:T - FWD] = closes[FWD:] / closes[:T - FWD] - 1.0
    rng = np.random.default_rng(seed)
    feat_names = [f"f{i}" for i in range(F)] + list(extra.keys()) + (["oracle"] if add_oracle else [])

    rebal = list(range(252, T - FWD, REBAL))
    samples = []
    for t in rebal:
        valid = np.isfinite(closes[t]) & np.isfinite(fwd[t]) & np.all(np.isfinite(base_feats[t]), axis=1)
        if valid.sum() < 50:
            continue
        raw = fwd[t]
        resid = raw - np.nanmean(raw[valid])  # market-neutral label
        blocks = [_xs_z(base_feats[t])]
        for name, panel in extra.items():
            blocks.append(_xs_z(panel[t].reshape(-1, 1)))
        if add_oracle:
            oracle = resid + rng.normal(0, oracle_noise * np.nanstd(resid[valid]) + 1e-9, size=N)
            blocks.append(_xs_z(oracle.reshape(-1, 1)))
        X = np.concatenate(blocks, axis=1)
        samples.append({"t": t, "valid": valid, "X": X, "resid": resid, "raw": raw})
    return samples, feat_names


def per_signal_ic(samples, feat_names):
    """Standalone cross-sectional forward-return IC of each feature (diagnostic)."""
    acc = {n: [] for n in feat_names}
    for s in samples:
        v = s["valid"]; raw = s["raw"][v]
        if not np.isfinite(raw).all() or len(raw) < 10:
            continue
        for j, n in enumerate(feat_names):
            col = s["X"][v, j]
            if np.isfinite(col).sum() > 10:
                acc[n].append(spearmanr(col, raw, nan_policy="omit").statistic)
    return {n: float(np.nanmean(v)) if v else float("nan") for n, v in acc.items()}


def run_book(samples, splits, params, cost_bps=10.0):
    monthly, ics, turn = [], [], []
    prev_w = None
    Nfull = len(samples[0]["valid"])
    for sp in splits:
        tr = [samples[i] for i in sp.train]
        Xtr = np.vstack([s["X"][s["valid"]] for s in tr])
        ytr = np.concatenate([s["resid"][s["valid"]] for s in tr])
        m = lgb.train(params, lgb.Dataset(Xtr, ytr, free_raw_data=False),
                      num_boost_round=params.get("_rounds", 150))
        for i in sp.test:
            s = samples[i]; v = s["valid"]
            pred = m.predict(s["X"][v])
            names = np.where(v)[0]; raw = s["raw"][v]
            if len(pred) > 10 and np.isfinite(raw).all():
                ics.append(spearmanr(pred, raw).statistic)
            k = max(1, int(DECILE * len(pred)))
            order = np.argsort(pred)
            longs, shorts = names[order[-k:]], names[order[:k]]
            w = np.zeros(Nfull); w[longs] = 1.0 / k; w[shorts] = -1.0 / k
            ret = float(np.mean(s["raw"][longs]) - np.mean(s["raw"][shorts]))
            turn.append(float(np.abs(w - (prev_w if prev_w is not None else 0)).sum()))
            prev_w = w; monthly.append(ret)
    monthly = np.asarray(monthly); turn = np.asarray(turn)
    net = monthly - turn * (cost_bps / 1e4)
    return net, np.asarray(ics)


PARAMS = {"objective": "regression", "num_leaves": 15, "min_child_samples": 200,
          "learning_rate": 0.03, "lambda_l2": 5.0, "feature_fraction": 0.8,
          "verbose": -1, "_rounds": 150}


def evaluate(name, samples, feat_names, n_trials_extra=0):
    splits = purged_walk_forward(len(samples), n_splits=6, label_horizon=1, embargo=1, min_train=24)
    ic = per_signal_ic(samples, feat_names)
    variants = [{**PARAMS, "num_leaves": nl, "min_child_samples": mc}
                for nl in (7, 15, 31) for mc in (100, 200, 400)]
    cols = [run_book(samples, splits, p)[0] for p in variants]
    Tn = min(len(c) for c in cols); grid = np.column_stack([c[-Tn:] for c in cols])
    net, oos_ic = run_book(samples, splits, PARAMS)
    stress, _ = run_book(samples, splits, PARAMS, cost_bps=25.0)
    rc = build_report_card(strategy_name=name, returns=net, n_trials=len(variants) + n_trials_extra,
                           trial_grid=grid, cost_adjusted_returns=stress, periods_per_year=12)
    ann = net.mean() * 12; vol = net.std() * np.sqrt(12)
    print(f"\n### {name}  OOS rank-IC {np.nanmean(oos_ic):+.4f}  net ann {ann*100:+.1f}%  Sharpe {ann/(vol+1e-9):.2f}")
    top = sorted(ic.items(), key=lambda kv: -abs(kv[1]))[:6]
    print("  top standalone |IC|:", "  ".join(f"{n}={v:+.3f}" for n, v in top))
    print(f"  STATUS: {rc.status}  (CI_lo gate, DSR-prob, cost-adj in the card)")
    Path("validation_reports").mkdir(exist_ok=True)
    (Path("validation_reports") / f"{name}.md").write_text(rc.render())
    return rc


def main() -> int:
    print("loading feature panel ...")
    dates, closes, feats, symbols = load_feature_panel()
    print(f"panel: {closes.shape[0]} days x {len(symbols)} names x {feats.shape[2]} features")

    # Control 1 — price-only: must stay BLOCKED (rejects noise).
    s0, fn0 = build_samples(dates, closes, feats)
    evaluate("combo_control_priceonly", s0, fn0)

    # Control 2 — +oracle (synthetic predictive feature): must clear, proving the
    # apparatus DETECTS a real edge. NOT a real backtest.
    s1, fn1 = build_samples(dates, closes, feats, add_oracle=True, oracle_noise=9.0)
    evaluate("combo_control_oracle", s1, fn1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

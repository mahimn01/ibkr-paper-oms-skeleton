"""Insider Form-4 edge REFINEMENTS prototyped on the EXISTING partial cache.

DOES NOT CRAWL. Loads only data/insider_form4_cache/*_purchases.json that are
already on disk, reuses scripts/bt_insider_form4.py's parser/classifier/signal
(`classify`, `event_trades`), and layers three refinements on top:

  (1) INVERSE-VOL SIZING + VOL-TARGET. The reported run had 34% portfolio vol
      (Sharpe-killer). Size each trade by 1/trailing-vol (PIT: trailing 60d daily
      vol of the name measured strictly BEFORE entry), normalise weights within
      each monthly cohort, then scale the whole book each month so realised
      monthly portfolio vol targets ~10% annualised (lagged, walk-forward, no
      look-ahead — the scalar uses only the trailing realised vol of the sized
      stream).

  (2) HEDGE VARIANTS instead of flat beta=1 IWM:
        - flat IWM (baseline, beta=1)        [what the original ran]
        - beta-adjusted IWM (per-name trailing 60d beta to IWM, PIT)
        - size blend: 0.7*IWM + 0.3*SPY beta-adjusted (small-cap + market)
      No GICS/sector tags exist in the data tree, so a true sector-relative
      hedge is NOT possible on this panel; size-relative (IWM beta) is the
      honest substitute and is documented as such.

  (3) HOLD / CLUSTER / WINDOW sweep within an HONEST small n_trials. The DSR
      deflates for every variant tried, so the grid is kept deliberately tiny
      and n_trials is charged for EVERYTHING (sizing scheme, hedge variant,
      and the fixed-by-choice knobs), not just the swept grid.

Output: prints a comparison table (flat-equal-weight vs inverse-vol, across
hedge variants) and whether inverse-vol sizing materially cuts vol / lifts the
risk-adjusted result on the limited sample. Writes nothing into shared modules.

CAVEATS (printed loudly): atlas_r3000 is ~100% survivors -> long-leg returns
inflated; the panel is PARTIAL (subset of the eventual crawl) so every number
is provisional and the DSR is structurally underpowered at this n.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trading_algo.quant_core.validation.report_card import build_report_card

R3000_DIR = ROOT / "data" / "atlas_r3000"
CACHE = ROOT / "data" / "insider_form4_cache"

# --- reuse the existing pipeline (parser/classifier/signal) WITHOUT crawling ---
_spec = importlib.util.spec_from_file_location(
    "bt_insider_form4", str(ROOT / "scripts" / "bt_insider_form4.py")
)
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)  # safe: module only crawls inside main()

# Backtest window: classification warmup before, IWM cache ends 2026-02 so cap there.
BT_START = pd.Timestamp("2021-01-01")
BT_END_PERIOD = "2026-02"
VOL_TARGET_ANN = 0.10          # 10% annualised book vol target
VOL_LOOKBACK = 60              # trading days for trailing name vol / beta
VOL_FLOOR_ANN = 0.10           # don't divide by sub-10%-vol -> cap leverage
MAX_NAME_WEIGHT_MULT = 5.0     # cap a single name's inverse-vol weight at 5x mean
RT_BPS = 140.0
RT_BPS_STRESS = 220.0


# --------------------------------------------------------------------------
# data loaders (cache-only; NO network, NO crawl)
# --------------------------------------------------------------------------

def load_all_events() -> list[dict]:
    out: list[dict] = []
    for f in sorted(glob.glob(str(CACHE / "*_purchases.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, list):
            out.extend(d)
    return out


def load_close_panel(syms: list[str]) -> dict[str, pd.Series]:
    px: dict[str, pd.Series] = {}
    for s in syms:
        f = R3000_DIR / f"{s}.parquet"
        if f.exists():
            df = pd.read_parquet(f, columns=["close"])
            df.index = pd.to_datetime(df.index)
            px[s] = df["close"].sort_index()
    return px


def load_bench_close(loader_sym: str) -> pd.Series:
    """Daily close series for a benchmark from the 5-min cache json."""
    from scripts._options_data import load_daily_bars
    from datetime import datetime, timezone
    bars = load_daily_bars(loader_sym)
    idx = pd.to_datetime(
        [datetime.fromtimestamp(b.timestamp_epoch_s, tz=timezone.utc) for b in bars]
    ).tz_localize(None).normalize()
    return pd.Series([b.close for b in bars], index=idx).sort_index()


# --------------------------------------------------------------------------
# PIT trailing vol / beta at entry (strictly pre-entry data only)
# --------------------------------------------------------------------------

def trailing_vol_ann(close: pd.Series, entry_dt: pd.Timestamp,
                     lookback: int = VOL_LOOKBACK) -> float:
    """Annualised daily-return vol over the `lookback` closes BEFORE entry_dt."""
    pre = close.loc[close.index < entry_dt]
    if len(pre) < lookback + 1:
        return np.nan
    rets = pre.iloc[-(lookback + 1):].pct_change().dropna()
    if len(rets) < lookback // 2:
        return np.nan
    return float(rets.std() * np.sqrt(252))


def trailing_beta(close: pd.Series, bench: pd.Series, entry_dt: pd.Timestamp,
                  lookback: int = VOL_LOOKBACK) -> float:
    """OLS beta of name vs bench over `lookback` overlapping pre-entry days."""
    pre_n = close.loc[close.index < entry_dt].iloc[-(lookback + 1):]
    pre_b = bench.loc[bench.index < entry_dt]
    if len(pre_n) < lookback // 2 or len(pre_b) < lookback // 2:
        return 1.0
    rn = pre_n.pct_change().dropna()
    rb = pre_b.reindex(rn.index).pct_change().dropna()
    common = rn.index.intersection(rb.index)
    if len(common) < lookback // 2:
        return 1.0
    x = rb.loc[common].values
    y = rn.loc[common].values
    var = np.var(x)
    if var < 1e-12:
        return 1.0
    beta = float(np.cov(x, y)[0, 1] / var)
    return float(np.clip(beta, 0.0, 3.0))


# --------------------------------------------------------------------------
# rebuild per-trade hedged returns with a chosen hedge variant
# --------------------------------------------------------------------------

def rebuild_hedged(trades: pd.DataFrame, px: dict[str, pd.Series],
                   iwm: pd.Series, spy: pd.Series, hedge: str) -> pd.DataFrame:
    """Recompute the hedge leg for each trade under a hedge variant.

    `trades` already carries entry/exit/stock_ret/bench (bench is the flat-1 IWM
    span return from event_trades). We recompute bench under the variant.
      hedge='flat_iwm'  -> use trades['bench'] as-is (beta=1)
      hedge='beta_iwm'  -> beta_name * IWM_span_ret
      hedge='size_blend'-> beta_name*(0.7*IWM_span + 0.3*SPY_span)
    Also attaches trailing_vol_ann for inverse-vol sizing.
    """
    if trades.empty:
        return trades
    t = trades.copy().reset_index(drop=True)
    iwm = iwm.sort_index(); spy = spy.sort_index()

    def span_ret(series: pd.Series, e0: pd.Timestamp, e1: pd.Timestamp) -> float:
        i0 = series.index.searchsorted(e0, side="left")
        i1 = series.index.searchsorted(e1, side="left")
        if i0 >= len(series) or i1 >= len(series) or i0 == i1:
            return 0.0
        return float(series.iloc[i1] / series.iloc[i0] - 1.0)

    hedged = np.empty(len(t)); tvol = np.empty(len(t))
    for i in range(len(t)):
        sym = t.at[i, "symbol"]; e0 = t.at[i, "entry"]; e1 = t.at[i, "exit"]
        sret = t.at[i, "stock_ret"]
        close = px.get(sym)
        tvol[i] = trailing_vol_ann(close, e0) if close is not None else np.nan
        if hedge == "flat_iwm":
            bench = t.at[i, "bench"]
        elif hedge == "beta_iwm":
            beta = trailing_beta(close, iwm, e0) if close is not None else 1.0
            bench = beta * span_ret(iwm, e0, e1)
        elif hedge == "size_blend":
            beta = trailing_beta(close, iwm, e0) if close is not None else 1.0
            iwm_s = span_ret(iwm, e0, e1); spy_s = span_ret(spy, e0, e1)
            bench = beta * (0.7 * iwm_s + 0.3 * spy_s)
        else:
            raise ValueError(hedge)
        hedged[i] = sret - bench
    t["hedged_v"] = hedged
    t["tvol"] = tvol
    return t


# --------------------------------------------------------------------------
# portfolio construction: equal-weight vs inverse-vol + vol-target
# --------------------------------------------------------------------------

def monthly_equal_weight(t: pd.DataFrame, all_months: pd.PeriodIndex,
                         rt_bps: float) -> np.ndarray:
    """Original scheme: equal-weight average of trades entered each month."""
    s = pd.Series(0.0, index=all_months)
    if t.empty:
        return s.values
    tt = t.copy()
    tt["m"] = tt["entry"].dt.to_period("M")
    tt["net"] = tt["hedged_v"] - rt_bps / 1e4
    for m, g in tt.groupby("m"):
        if m in s.index:
            s.loc[m] = g["net"].mean()
    return s.values


def monthly_inverse_vol(t: pd.DataFrame, all_months: pd.PeriodIndex,
                        rt_bps: float, vol_target_ann: float) -> np.ndarray:
    """Inverse-vol weights within each monthly cohort, then a WALK-FORWARD
    vol-target scalar (uses only trailing realised vol of the sized stream).

    Step A: within month m, weight_i ∝ 1/max(tvol_i, floor); normalise to sum 1;
            cap each weight at MAX_NAME_WEIGHT_MULT * (1/n). Cohort gross
            return = Σ w_i * (hedged_i - cost).
    Step B: target. scalar_m = vol_target / trailing_realised_vol(cohort stream
            up to and INCLUDING m-1), clipped to [0.25, 3.0]; applied to month m.
            Month 0..warmup use scalar=1 (no trailing vol yet).
    """
    s_gross = pd.Series(np.nan, index=all_months)
    if not t.empty:
        tt = t.copy()
        tt["m"] = tt["entry"].dt.to_period("M")
        tt["net"] = tt["hedged_v"] - rt_bps / 1e4
        for m, g in tt.groupby("m"):
            if m not in s_gross.index:
                continue
            vol = g["tvol"].copy()
            vol = vol.where(np.isfinite(vol))
            # names without a trailing vol fall back to cohort median vol
            med = np.nanmedian(vol) if np.isfinite(vol).any() else VOL_FLOOR_ANN
            vol = vol.fillna(med).clip(lower=VOL_FLOOR_ANN)
            w = 1.0 / vol.values
            w = w / w.sum()
            cap = MAX_NAME_WEIGHT_MULT / len(w)
            w = np.minimum(w, cap)
            w = w / w.sum()
            s_gross.loc[m] = float(np.dot(w, g["net"].values))
    s_gross = s_gross.fillna(0.0)

    # Step B: walk-forward vol-target scalar
    out = np.zeros(len(s_gross))
    vals = s_gross.values
    target_per_month = vol_target_ann / np.sqrt(12)
    for i in range(len(vals)):
        if i < 6:
            scalar = 1.0
        else:
            trailing = vals[max(0, i - 12):i]
            nz = trailing[np.abs(trailing) > 1e-12]
            rv = np.std(nz) if len(nz) >= 3 else np.std(trailing)
            scalar = (target_per_month / rv) if rv > 1e-9 else 1.0
            scalar = float(np.clip(scalar, 0.25, 3.0))
        out[i] = scalar * vals[i]
    return out


def sharpe_stats(ret: np.ndarray) -> tuple[float, float, float]:
    ann = float(np.mean(ret) * 12)
    vol = float(np.std(ret) * np.sqrt(12))
    sh = ann / (vol + 1e-9)
    return ann, vol, sh


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    print("=== REFINEMENT PROTOTYPE on PARTIAL insider cache (NO CRAWL) ===")
    events = load_all_events()
    print(f"loaded {len(events)} cached purchase-events from disk")
    if not events:
        print("no events; abort")
        return 1

    df_cls = bt.classify(events)
    n_opp = int(df_cls["opportunistic"].sum()); n_rou = int(df_cls["routine"].sum())
    print(f"classified {len(df_cls)}  opportunistic={n_opp}  routine={n_rou}  "
          f"issuers={df_cls['symbol'].nunique()}")

    syms = sorted(df_cls["symbol"].unique())
    px = load_close_panel(syms)
    iwm = load_bench_close("IWM")
    spy = load_bench_close("SPY")
    print(f"price panel: {len(px)}/{len(syms)} names; IWM {len(iwm)} bars; SPY {len(spy)} bars")

    # align the script's BT window to ours
    bt.BT_START = BT_START.date()
    all_months = pd.period_range(BT_START, BT_END_PERIOD, freq="M")

    # IWM as a DataFrame for event_trades (needs open; cache gives close only ->
    # use close as a proxy for the OPEN-to-OPEN span; documented approximation
    # consistent with the original load_iwm fallback which also lacked true opens
    # only when parquet missing). We instead recompute the hedge ourselves in
    # rebuild_hedged using close spans, so event_trades' bench is recomputed.
    iwm_df = pd.DataFrame({"open": iwm, "close": iwm})

    LEG = "opportunistic"

    # ---- honest trial accounting ----
    # Swept grid (PBO + DSR deflation): holds x clusters x windows
    holds = [21, 42, 63]
    clusters = [2, 3]
    windows = [7, 14]
    sizing_schemes = ["equal", "inverse_vol"]
    hedge_variants = ["flat_iwm", "beta_iwm", "size_blend"]
    dollar_min = 50_000.0

    # Build base signal trade-sets once per (cluster,window,hold) using event_trades,
    # then re-hedge / re-size cheaply. Cache the raw stock_ret/entry/exit per config.
    raw_cache: dict[tuple, pd.DataFrame] = {}
    for c in clusters:
        for w in windows:
            for h in holds:
                tr = bt.event_trades(df_cls, LEG, {s: pd.DataFrame(
                        {"open": px[s], "close": px[s]}) for s in px},
                        iwm_df, cluster_min=c, dollar_min=dollar_min,
                        window_days=w, hold=h, beta_hedge=True)
                raw_cache[(c, w, h)] = tr

    base_key = (2, 7, 21)
    base_trades = raw_cache[base_key]
    print(f"\nbase config {base_key}: {len(base_trades)} signals "
          f"(raw hedged/trade {0 if base_trades.empty else base_trades['hedged'].mean()*100:+.2f}%)")

    # ---- headline comparison: sizing x hedge on the BASE config ----
    print("\n=== HEADLINE: sizing x hedge variant on base config (2,7,21) ===")
    print(f"{'sizing':12s} {'hedge':11s} {'ann%':>7s} {'vol%':>7s} {'Sharpe':>7s} "
          f"{'CIlo':>7s} {'DSRp':>6s} {'PBO':>5s} {'cadjSh':>7s} {'status':>9s}")
    headline = {}
    n_trials_total = (len(holds) * len(clusters) * len(windows)
                      * len(sizing_schemes) * len(hedge_variants))
    # +4 for fixed-by-choice knobs (vol-target level, vol lookback, weight cap,
    # routine-classification 3y window) per the original script's honesty rule.
    n_trials_honest = n_trials_total + 4

    for hedge in hedge_variants:
        t_h = rebuild_hedged(base_trades, px, iwm, spy, hedge)
        for sizing in sizing_schemes:
            if sizing == "equal":
                ret = monthly_equal_weight(t_h, all_months, RT_BPS)
                ret_stress = monthly_equal_weight(t_h, all_months, RT_BPS_STRESS)
            else:
                ret = monthly_inverse_vol(t_h, all_months, RT_BPS, VOL_TARGET_ANN)
                ret_stress = monthly_inverse_vol(t_h, all_months, RT_BPS_STRESS, VOL_TARGET_ANN)
            ann, vol, sh = sharpe_stats(ret)
            rc = build_report_card(
                strategy_name=f"insider_{sizing}_{hedge}",
                returns=ret, n_trials=n_trials_honest,
                cost_adjusted_returns=ret_stress,
                periods_per_year=12, seed=42,
            )
            headline[(sizing, hedge)] = dict(
                ann=ann, vol=vol, sharpe=sh, ci_lo=rc.sharpe_ci_lower,
                dsr=rc.deflated_sharpe, cadj=rc.cost_adjusted_sharpe,
                status=rc.status, ret=ret)
            print(f"{sizing:12s} {hedge:11s} {ann*100:7.1f} {vol*100:7.1f} "
                  f"{sh:7.2f} {rc.sharpe_ci_lower:7.2f} {rc.deflated_sharpe:6.2f} "
                  f"{'  n/a' if rc.pbo is None else f'{rc.pbo:5.2f}'} "
                  f"{rc.cost_adjusted_sharpe:7.2f} {rc.status:>9s}")

    # ---- PBO across the FULL swept grid for the WINNING sizing/hedge ----
    # choose best by point Sharpe among inverse-vol variants (the refinement
    # under test); build the trial grid for PBO honestly across all configs.
    print("\n=== PBO via CSCV trial grid (all holds x clusters x windows) ===")
    for sizing in sizing_schemes:
        for hedge in ["beta_iwm"]:  # one hedge to keep grid honest, not multiplied
            cols = []
            for c in clusters:
                for w in windows:
                    for h in holds:
                        t_h = rebuild_hedged(raw_cache[(c, w, h)], px, iwm, spy, hedge)
                        if sizing == "equal":
                            col = monthly_equal_weight(t_h, all_months, RT_BPS)
                        else:
                            col = monthly_inverse_vol(t_h, all_months, RT_BPS, VOL_TARGET_ANN)
                        cols.append(col)
            grid = np.column_stack(cols)
            # representative center column for the card
            center = grid[:, grid.shape[1] // 2]
            rc = build_report_card(
                strategy_name=f"insider_{sizing}_{hedge}_grid",
                returns=center, n_trials=n_trials_honest, trial_grid=grid,
                periods_per_year=12, seed=42,
            )
            print(f"{sizing:12s} {hedge:11s}  grid {grid.shape}  "
                  f"PBO={'n/a' if rc.pbo is None else f'{rc.pbo:.2f}'}  "
                  f"center_Sharpe={rc.point_sharpe:.2f}  DSRp={rc.deflated_sharpe:.2f}")

    # ---- verdict: does inverse-vol materially help? ----
    print("\n=== VERDICT: inverse-vol vs equal-weight (best hedge = beta_iwm) ===")
    eq = headline[("equal", "beta_iwm")]
    iv = headline[("inverse_vol", "beta_iwm")]
    print(f"  equal-weight : vol={eq['vol']*100:5.1f}%  Sharpe={eq['sharpe']:.2f}  "
          f"CIlo={eq['ci_lo']:.2f}  DSRp={eq['dsr']:.2f}  cadjSh={eq['cadj']:.2f}  {eq['status']}")
    print(f"  inverse-vol  : vol={iv['vol']*100:5.1f}%  Sharpe={iv['sharpe']:.2f}  "
          f"CIlo={iv['ci_lo']:.2f}  DSRp={iv['dsr']:.2f}  cadjSh={iv['cadj']:.2f}  {iv['status']}")
    vol_cut = (eq['vol'] - iv['vol']) / eq['vol'] * 100 if eq['vol'] > 0 else 0.0
    sh_lift = iv['sharpe'] - eq['sharpe']
    print(f"  --> vol cut {vol_cut:+.0f}%   Sharpe lift {sh_lift:+.2f}   n_trials_honest={n_trials_honest}")

    print("\nCAVEATS: atlas_r3000 ~100% survivors (long leg inflated); panel is "
          "PARTIAL; DSR underpowered at this n; IWM/SPY hedge spans use CLOSE "
          "(cache lacks open) -> hedge slightly approximate; no GICS tags so "
          "'size-relative' (IWM beta) substitutes for sector-relative.")

    # compact json for the harness
    summary = {
        "n_events": len(events), "n_classified": len(df_cls),
        "n_opportunistic": n_opp, "n_issuers": int(df_cls["symbol"].nunique()),
        "base_signals": int(len(base_trades)),
        "n_trials_honest": n_trials_honest,
        "headline": {f"{s}|{h}": {
            "ann_pct": v["ann"] * 100, "vol_pct": v["vol"] * 100,
            "sharpe": v["sharpe"], "ci_lo": v["ci_lo"], "dsr_prob": v["dsr"],
            "cost_adj_sharpe": v["cadj"], "status": v["status"],
        } for (s, h), v in headline.items()},
        "verdict": {
            "vol_cut_pct": vol_cut, "sharpe_lift": sh_lift,
            "eq_vol_pct": eq["vol"] * 100, "iv_vol_pct": iv["vol"] * 100,
        },
    }
    out_path = ROOT / "data" / "insider_form4_cache" / "refine_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\nwrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

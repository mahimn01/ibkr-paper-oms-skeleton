"""WorldMonitor / alt-data cross-sectional IC survey (read-only, Wave-T5).

Standalone, uniquely-named. Does NOT edit shared modules and makes NO network
calls (SEC, IBKR, or MCP) — the MCP survey is done live in the parent agent;
this script only runs the crude forward-return IC sanity checks on data that is
already on disk:

  1) insider_form4_cache  (231 files, ~6.5k opportunistic Form-4 PURCHASES,
     small-caps, all in the R3000 parquet panel)  -> event-study forward returns
  2) (placeholder) congress feed IC if a cached cross-sectional file is provided.

PIT discipline: a purchase is only ACTIONABLE on its **filingDate** (the public
disclosure), never the transactionDate. We enter at the next available close on
or after filingDate and measure forward returns. No look-ahead.

Usage:
    python scripts/altdata_ic_survey_wmt5.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

R3000_DIR = "data/atlas_r3000"
INSIDER_DIR = "data/insider_form4_cache"
HORIZONS = (5, 10, 21, 63)  # trading-day forward-return horizons


def _load_close(sym: str) -> pd.Series | None:
    p = os.path.join(R3000_DIR, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p, columns=["close"])
    except Exception:
        return None
    s = df["close"].astype(float)
    s = s[s > 0]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def load_insider_events() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(INSIDER_DIR, "*_purchases.json")):
        for r in json.load(open(f)):
            if r.get("shares", 0) and r.get("shares", 0) > 0:
                rows.append(
                    {
                        "symbol": r["symbol"],
                        "filingDate": pd.to_datetime(r["filingDate"]),
                        "value": float(r.get("value", 0.0) or 0.0),
                        "is_officer": int(r.get("is_officer", 0)),
                        "is_director": int(r.get("is_director", 0)),
                        "is_ten": int(r.get("is_ten", 0)),
                    }
                )
    df = pd.DataFrame(rows).sort_values("filingDate").reset_index(drop=True)
    return df


def event_forward_returns(events: pd.DataFrame) -> pd.DataFrame:
    """For each insider purchase, enter at the first close on/after filingDate,
    compute raw and market-relative forward returns at each horizon."""
    # market proxy: equal-weight mean daily return across the loaded panel
    closes: dict[str, pd.Series] = {}
    for sym in events["symbol"].unique():
        s = _load_close(sym)
        if s is not None and len(s) > 100:
            closes[sym] = s

    # build an equal-weight market return index from the union of loaded names
    all_idx = sorted(set().union(*[set(s.index) for s in closes.values()]))
    mkt = pd.DataFrame({sym: s.reindex(all_idx) for sym, s in closes.items()})
    mkt_ret = mkt.pct_change()
    mkt_daily = mkt_ret.mean(axis=1)  # equal-weight panel return
    mkt_cum = (1.0 + mkt_daily.fillna(0.0)).cumprod()

    out = []
    for _, e in events.iterrows():
        sym = e["symbol"]
        s = closes.get(sym)
        if s is None:
            continue
        # first trading day on/after the public filing date (PIT entry)
        pos = s.index.searchsorted(e["filingDate"], side="left")
        if pos >= len(s) - max(HORIZONS) - 1:
            continue
        entry_px = s.iloc[pos]
        entry_dt = s.index[pos]
        mpos = mkt_cum.index.searchsorted(entry_dt, side="left")
        rec = {"symbol": sym, "entry": entry_dt, "value": e["value"],
               "is_officer": e["is_officer"], "is_ten": e["is_ten"]}
        ok = True
        for h in HORIZONS:
            if pos + h >= len(s) or mpos + h >= len(mkt_cum):
                ok = False
                break
            r = s.iloc[pos + h] / entry_px - 1.0
            mr = mkt_cum.iloc[mpos + h] / mkt_cum.iloc[mpos] - 1.0
            rec[f"fwd{h}"] = r
            rec[f"abn{h}"] = r - mr  # market-relative (hedged) forward return
        if ok:
            out.append(rec)
    return pd.DataFrame(out)


def summarize(fr: pd.DataFrame) -> str:
    lines = []
    n = len(fr)
    lines.append(f"insider-purchase events with full forward window: {n}")
    if n == 0:
        return "\n".join(lines)
    for h in HORIZONS:
        raw = fr[f"fwd{h}"].dropna()
        abn = fr[f"abn{h}"].dropna()
        # t-stat of the hedged mean (are insider buys predictive of abn return?)
        t = abn.mean() / (abn.std(ddof=1) / np.sqrt(len(abn))) if len(abn) > 2 else float("nan")
        hit = (abn > 0).mean()
        lines.append(
            f"  h={h:>2}d  raw_mean={raw.mean()*100:+6.2f}%  "
            f"abn_mean={abn.mean()*100:+6.2f}%  abn_t={t:+5.2f}  "
            f"hit={hit*100:4.1f}%  n={len(abn)}"
        )
    # cross-sectional rank-IC proxy: does larger $value -> larger abn return?
    for h in HORIZONS:
        sub = fr[["value", f"abn{h}"]].dropna()
        if len(sub) > 30 and sub["value"].std() > 0:
            ic = sub["value"].rank().corr(sub[f"abn{h}"].rank())
            lines.append(f"  rank-IC(value, abn{h}) = {ic:+.4f}  (n={len(sub)})")
    return "\n".join(lines)


def main() -> int:
    print("=== INSIDER FORM-4 PURCHASE EVENT STUDY (PIT entry = filingDate) ===")
    ev = load_insider_events()
    print(f"loaded {len(ev)} purchase rows across {ev['symbol'].nunique()} symbols, "
          f"{ev['filingDate'].min().date()} -> {ev['filingDate'].max().date()}")
    fr = event_forward_returns(ev)
    print(summarize(fr))
    # officer/director-only subset (drops 10%-holder/fund noise like Paulson blocks)
    if "is_ten" in fr.columns:
        ins = fr[fr["is_ten"] == 0]
        print("\n--- officer/director-only subset (is_ten==0) ---")
        print(summarize(ins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

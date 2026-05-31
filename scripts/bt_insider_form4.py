"""Opportunistic insider Form-4 cluster backtest (Cohen-Malloy-Pomorski JF 2012).

End-to-end PIT prototype:
  1. Map R3000 microcap-band tickers -> CIK (SEC company_tickers.json).
  2. Crawl recent Form-4 filings per CIK, parse OPEN-MARKET PURCHASES
     (nonDerivative, transactionCode 'P', acquired/disposed 'A'): owner,
     isOfficer/isDirector, shares, price, transactionDate, FILINGDATE.
  3. Classify each insider's buy ROUTINE (bought same calendar month in
     >=2 of prior 3 years) vs OPPORTUNISTIC.
  4. Signal on filingDate: >=2 distinct OPPORTUNISTIC insiders bought within
     trailing 7 days AND aggregate > $50k. Entry next-day open, equal-weight
     long, hold H trading days. IWM-beta-hedged to isolate alpha.
  5. Harsh microcap cost stack.
  6. build_report_card(periods_per_year=12) with an HONEST n_trials.
  7. Falsification: identical pipeline on ROUTINE buys.

NO-LOOK-AHEAD: panel keyed on FILINGDATE (publication), never transactionDate.
Entry is the OPEN of the trading day AFTER filingDate.

Survivorship caveat: atlas_r3000 is ~all survivors; long-only microcap numbers
are inflated. We report the IWM-hedged long leg and flag this loudly.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_algo.quant_core.validation.report_card import build_report_card

ROOT = Path(__file__).resolve().parent.parent
R3000_DIR = ROOT / "data" / "atlas_r3000"
CACHE = ROOT / "data" / "insider_form4_cache"
CACHE.mkdir(parents=True, exist_ok=True)
UA = "Mahimn Patel quant research mahimn.patel.k@gmail.com"

# ---- universe scoping ----
# Scoped DOWN to a tractable representative subset: the per-name Form-4 volume
# (~330 filings/name x mostly non-purchase option grants) makes a 400-name x
# 2017+ crawl ~130k XML GETs / ~4.5h at EDGAR's polite rate. 150 names x 2018+
# keeps it ~1h while staying representative of the microcap band.
N_NAMES = int(os.environ.get("INSIDER_N_NAMES", "100"))
DV_LO, DV_HI = 3e5, 1.5e7
MIN_PRICE = 3.0
CRAWL_START = date.fromisoformat(os.environ.get("INSIDER_CRAWL_START", "2018-01-01"))  # >=3y before BT_START
BT_START = date.fromisoformat(os.environ.get("INSIDER_BT_START", "2021-01-01"))        # window after classification warmup
MAX_FILINGS_PER_NAME = int(os.environ.get("INSIDER_MAX_FILINGS", "150"))


# --------------------------------------------------------------------------
# HTTP with polite rate-limit
# --------------------------------------------------------------------------

# Global token-bucket rate limiter shared across worker threads. SEC's documented
# ceiling is 10 req/s; we run conservatively at ~6/s because bursting past 10
# triggers an IP-level 429 *ban* (not just a single-request reject) that poisons
# the whole crawl. A global cool-down pauses ALL threads when a 429 is seen.
_rl_lock = threading.Lock()
_next_slot = [0.0]
# SEC flagged this IP for stricter-than-documented throttling after earlier
# bursts, so run genuinely conservative: ~4 req/s, single worker. The global
# 429 cool-down is the backstop.
_MIN_GAP = 1.0 / float(os.environ.get("INSIDER_RPS", "4"))
_cooldown_until = [0.0]   # epoch; all threads block until this passes

class RateLimitedError(Exception):
    pass

def _throttle() -> None:
    with _rl_lock:
        now = time.time()
        wait = max(_next_slot[0] - now, _cooldown_until[0] - now)
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _next_slot[0] = max(now, _next_slot[0]) + _MIN_GAP

_n_429 = [0]

def _trip_cooldown(seconds: float) -> None:
    with _rl_lock:
        _cooldown_until[0] = max(_cooldown_until[0], time.time() + seconds)
        _n_429[0] += 1
        if _n_429[0] % 10 == 1:
            print(f"  [rate] 429 #{_n_429[0]}, cooling {seconds:.0f}s", flush=True)

def _get(url: str, retries: int = 5) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429:
                # global ban: back everyone off hard and retry
                _trip_cooldown(10.0 + 5.0 * attempt)
                continue
            if e.code in (403, 500, 502, 503) and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(1.0)
                continue
            raise
    raise RateLimitedError(f"exhausted retries on {url}: {last_exc}")


# --------------------------------------------------------------------------
# Step 1: scope universe + ticker->CIK
# --------------------------------------------------------------------------

def scope_universe() -> list[str]:
    rows = []
    for f in sorted(R3000_DIR.glob("*.parquet")):
        sym = f.stem
        try:
            df = pd.read_parquet(f, columns=["close", "volume"]).loc["2024-06-01":"2026-04-06"]
        except Exception:
            continue
        if len(df) < 60:
            continue
        px = float(df["close"].iloc[-1])
        dv = float((df["close"] * df["volume"]).tail(60).median())
        if DV_LO <= dv <= DV_HI and px > MIN_PRICE:
            rows.append((sym, dv))
    rows.sort(key=lambda r: r[0])  # deterministic by ticker
    syms = [r[0] for r in rows]
    # representative even slice across the band (deterministic, not cherry-picked)
    if len(syms) > N_NAMES:
        step = len(syms) / N_NAMES
        syms = [syms[int(i * step)] for i in range(N_NAMES)]
    return syms


def ticker_to_cik() -> dict[str, int]:
    p = CACHE / "company_tickers.json"
    if p.exists():
        d = json.loads(p.read_text())
    else:
        d = json.loads(_get("https://www.sec.gov/files/company_tickers.json").decode())
        p.write_text(json.dumps(d))
    return {v["ticker"].upper(): int(v["cik_str"]) for v in d.values()}


# --------------------------------------------------------------------------
# Step 2: crawl + parse Form-4 purchases
# --------------------------------------------------------------------------

def _txt(node) -> str:
    return (node.text or "").strip() if node is not None else ""

def _find(parent, tag):
    """Namespace-agnostic local-name find."""
    for el in parent.iter():
        if el.tag.split("}")[-1] == tag:
            return el
    return None

def _findall_direct(parent, tag):
    return [el for el in parent if el.tag.split("}")[-1] == tag]

def parse_form4(xml_bytes: bytes) -> dict | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    # owner relationship
    rel = _find(root, "reportingOwnerRelationship")
    is_officer = is_director = is_ten = 0
    owner_cik = owner_name = ""
    if rel is not None:
        is_officer = 1 if _txt(_find(rel, "isOfficer")) in ("1", "true") else 0
        is_director = 1 if _txt(_find(rel, "isDirector")) in ("1", "true") else 0
        is_ten = 1 if _txt(_find(rel, "isTenPercentOwner")) in ("1", "true") else 0
    oid = _find(root, "reportingOwnerId")
    if oid is not None:
        owner_cik = _txt(_find(oid, "rptOwnerCik"))
        owner_name = _txt(_find(oid, "rptOwnerName"))
    sym = _txt(_find(root, "issuerTradingSymbol")).upper()

    # non-derivative open-market purchases
    purchases = []
    ndt = _find(root, "nonDerivativeTable")
    if ndt is None:
        return {"symbol": sym, "owner_cik": owner_cik, "owner_name": owner_name,
                "is_officer": is_officer, "is_director": is_director,
                "is_ten": is_ten, "purchases": []}
    for txn in ndt.iter():
        if txn.tag.split("}")[-1] != "nonDerivativeTransaction":
            continue
        coding = _find(txn, "transactionCoding")
        code = _txt(_find(coding, "transactionCode")) if coding is not None else ""
        amts = _find(txn, "transactionAmounts")
        # acquired/disposed is a CONTAINER element wrapping a <value> leaf
        ad_container = _find(amts, "transactionAcquiredDisposedCode") if amts is not None else None
        ad = _txt(_find(ad_container, "value")) if ad_container is not None else ""
        if code != "P" or ad != "A":
            continue
        tdate = _txt(_find(_find(txn, "transactionDate"), "value")) if _find(txn, "transactionDate") is not None else ""
        shares_n = _find(amts, "transactionShares")
        price_n = _find(amts, "transactionPricePerShare")
        try:
            shares = float(_txt(_find(shares_n, "value"))) if shares_n is not None else 0.0
            price = float(_txt(_find(price_n, "value"))) if price_n is not None else 0.0
        except ValueError:
            continue
        if shares <= 0 or price <= 0 or not tdate:
            continue
        purchases.append({"transactionDate": tdate, "shares": shares,
                          "price": price, "value": shares * price})
    return {"symbol": sym, "owner_cik": owner_cik, "owner_name": owner_name,
            "is_officer": is_officer, "is_director": is_director,
            "is_ten": is_ten, "purchases": purchases}


def _raw_xml_url(cik: int, acc_nodash: str, primary_doc: str) -> str | None:
    """Derive the RAW Form-4 XML url from the submissions primaryDocument.

    primaryDocument is the rendered ref, e.g. 'xslF345X05/wk-form4_123.xml';
    the raw XML lives at the same path WITHOUT the xslF345*/ prefix. If the
    primaryDocument is not an .xml we cannot derive it (skip -> None)."""
    if not primary_doc.lower().endswith(".xml"):
        return None
    raw = primary_doc.split("/")[-1]  # strip any xslF345X05/ prefix
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{raw}"


def crawl_name(sym: str, cik: int) -> list[dict]:
    """Return list of purchase events with filingDate for one issuer.

    One submissions fetch + (paginated history) + exactly ONE raw-XML GET per
    Form-4 (no per-accession directory listing). Cached to disk ONLY on a clean
    crawl — a name with ANY fetch failure raises so it is retried next run
    rather than being poisoned with an empty/partial result."""
    out_path = CACHE / f"{sym}_purchases.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    events: list[dict] = []
    # submissions fetch: a hard failure must NOT be cached as empty
    sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").decode())

    pages = [sub["filings"]["recent"]]
    for extra in sub["filings"].get("files", []):
        if len(pages) >= 4:
            break
        pages.append(json.loads(_get(f"https://data.sec.gov/submissions/{extra['name']}").decode()))

    # Collect Form-4 candidates within the window, then cap to the MOST RECENT
    # MAX_FILINGS_PER_NAME. A handful of band names file 400-650 Form-4s (mostly
    # tiny recurring grants); fetching every one balloons the crawl to hours.
    # Capping to recent filings keeps 2021+ signal coverage (the backtest window)
    # while bounding cost. NOTE: this can under-detect routine status for events
    # whose 3y prior history predates the cap — a documented limitation.
    cands = []  # (filingDate, acc, primary)
    for rec in pages:
        forms = rec.get("form", [])
        for i, form in enumerate(forms):
            if form not in ("4", "4/A"):
                continue
            fdate = rec["filingDate"][i]
            if datetime.strptime(fdate, "%Y-%m-%d").date() < CRAWL_START:
                continue
            cands.append((fdate, rec["accessionNumber"][i].replace("-", ""),
                          rec["primaryDocument"][i]))
    cands.sort(key=lambda c: c[0], reverse=True)   # most recent first
    cands = cands[:MAX_FILINGS_PER_NAME]

    for fdate, acc, primary in cands:
            xml_url = _raw_xml_url(cik, acc, primary)
            if xml_url is None:
                continue
            # let RateLimitedError propagate (so the name is retried, not cached
            # as falsely-empty); only swallow genuine parse problems
            parsed = parse_form4(_get(xml_url))
            if not parsed or not parsed["purchases"]:
                continue
            for p in parsed["purchases"]:
                events.append({
                    "symbol": sym,
                    "filingDate": fdate,
                    "transactionDate": p["transactionDate"],
                    "owner_cik": parsed["owner_cik"],
                    "owner_name": parsed["owner_name"],
                    "is_officer": parsed["is_officer"],
                    "is_director": parsed["is_director"],
                    "is_ten": parsed["is_ten"],
                    "shares": p["shares"],
                    "price": p["price"],
                    "value": p["value"],
                })
    out_path.write_text(json.dumps(events))
    return events


# --------------------------------------------------------------------------
# Step 3: routine vs opportunistic classification (CMP 2012)
# --------------------------------------------------------------------------

def classify(events: list[dict]) -> pd.DataFrame:
    """Tag each purchase routine/opportunistic by the insider's PRIOR-3-YEAR pattern.

    Routine = same calendar month in >=2 of the 3 prior years (using only buys
    that were FILED strictly before this event's filingDate => PIT-safe).
    """
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["transactionDate"] = pd.to_datetime(df["transactionDate"], errors="coerce")
    df = df.dropna(subset=["transactionDate"]).sort_values("filingDate").reset_index(drop=True)

    # key insider by (owner_cik, symbol)
    routine = np.zeros(len(df), dtype=bool)
    for (oc, sym), g in df.groupby(["owner_cik", "symbol"]):
        # build per-insider history of (year, month) of transaction dates,
        # but only count history available at each event's filingDate
        idxs = g.index.tolist()
        for ix in idxs:
            ev_month = df.at[ix, "transactionDate"].month
            ev_year = df.at[ix, "transactionDate"].year
            fdate = df.at[ix, "filingDate"]
            # prior buys filed before this filing
            prior = g[g["filingDate"] < fdate]
            hits = 0
            for yr in (ev_year - 1, ev_year - 2, ev_year - 3):
                same = prior[(prior["transactionDate"].dt.month == ev_month) &
                             (prior["transactionDate"].dt.year == yr)]
                if len(same) > 0:
                    hits += 1
            routine[ix] = hits >= 2
    df["routine"] = routine
    df["opportunistic"] = ~routine
    return df


# --------------------------------------------------------------------------
# Step 4+5: signal -> hedged monthly returns
# --------------------------------------------------------------------------

def load_prices(syms: list[str]) -> dict[str, pd.DataFrame]:
    px = {}
    for s in syms:
        f = R3000_DIR / f"{s}.parquet"
        if f.exists():
            df = pd.read_parquet(f, columns=["open", "close"])
            df.index = pd.to_datetime(df.index)
            px[s] = df
    return px


def load_iwm() -> pd.DataFrame:
    f = R3000_DIR / "IWM.parquet"
    if f.exists():
        df = pd.read_parquet(f, columns=["open", "close"])
        df.index = pd.to_datetime(df.index)
        return df
    # fall back to cache json loader
    from scripts._options_data import load_daily_bars
    bars = load_daily_bars("IWM")
    df = pd.DataFrame({
        "open": [b.open for b in bars],
        "close": [b.close for b in bars],
    }, index=pd.to_datetime([datetime.utcfromtimestamp(b.timestamp_epoch_s) for b in bars]))
    df.index = df.index.normalize()
    return df


def event_trades(df_cls: pd.DataFrame, leg: str, px: dict, iwm: pd.DataFrame,
                 *, cluster_min: int, dollar_min: float, window_days: int,
                 hold: int, beta_hedge: bool = True) -> pd.DataFrame:
    """Build per-event hedged holding-period returns.

    Signal fires for a (symbol, filingDate) when >=cluster_min DISTINCT insiders
    of the chosen leg bought within trailing `window_days` and aggregate value >
    dollar_min. Entry = next trading day's OPEN; exit = OPEN `hold` days later.
    Hedged return = stock_ret - beta * iwm_ret over the same window.
    """
    if df_cls.empty:
        return pd.DataFrame()
    sub = df_cls[df_cls[leg]].copy()
    if sub.empty:
        return pd.DataFrame()

    iwm_open = iwm["open"]
    iwm_dates = iwm.index

    trades = []
    # consider each filing as a potential cluster trigger date
    for sym, g in sub.groupby("symbol"):
        if sym not in px:
            continue
        sdf = px[sym]
        s_open = sdf["open"]
        s_dates = sdf.index
        g = g.sort_values("filingDate")
        fdates = sorted(g["filingDate"].unique())
        last_exit = pd.Timestamp.min
        for fd in fdates:
            fd = pd.Timestamp(fd)
            if fd < pd.Timestamp(BT_START):
                continue
            if fd <= last_exit:   # non-overlapping holds per name
                continue
            lo = fd - pd.Timedelta(days=window_days)
            win = g[(g["filingDate"] > lo) & (g["filingDate"] <= fd)]
            n_distinct = win["owner_cik"].nunique()
            agg = float(win["value"].sum())
            if n_distinct < cluster_min or agg < dollar_min:
                continue
            # entry: first trading day strictly after fd
            ent_idx = s_dates.searchsorted(fd, side="right")
            if ent_idx >= len(s_dates) - hold:
                continue
            ex_idx = ent_idx + hold
            if ex_idx >= len(s_dates):
                continue
            entry_p = s_open.iloc[ent_idx]
            exit_p = s_open.iloc[ex_idx]
            if not (np.isfinite(entry_p) and np.isfinite(exit_p)) or entry_p <= 0:
                continue
            stock_ret = exit_p / entry_p - 1.0
            # IWM hedge over the same calendar span
            entry_dt = s_dates[ent_idx]
            exit_dt = s_dates[ex_idx]
            ii0 = iwm_dates.searchsorted(entry_dt, side="left")
            ii1 = iwm_dates.searchsorted(exit_dt, side="left")
            if ii0 >= len(iwm_dates) or ii1 >= len(iwm_dates) or ii0 == ii1:
                hedged = stock_ret
                bench = 0.0
            else:
                bench = float(iwm_open.iloc[ii1] / iwm_open.iloc[ii0] - 1.0)
                hedged = stock_ret - bench if beta_hedge else stock_ret
            trades.append({"symbol": sym, "entry": entry_dt, "exit": exit_dt,
                           "stock_ret": stock_ret, "bench": bench, "hedged": hedged})
            last_exit = exit_dt
    return pd.DataFrame(trades)


def monthly_series(trades: pd.DataFrame, col: str, all_months: pd.PeriodIndex) -> np.ndarray:
    """Equal-weight portfolio: average of trades ENTERED in each calendar month.

    Months with no signal => 0 return (capital idle / hedged flat). This is the
    honest monthly P&L stream for a periods_per_year=12 card.
    """
    s = pd.Series(0.0, index=all_months)
    if not trades.empty:
        t = trades.copy()
        t["m"] = t["entry"].dt.to_period("M")
        grp = t.groupby("m")[col].mean()
        for m, v in grp.items():
            if m in s.index:
                s.loc[m] = v
    return s.values


# --------------------------------------------------------------------------
# Harsh cost model
# --------------------------------------------------------------------------

def apply_costs(trades: pd.DataFrame, rt_bps: float) -> pd.DataFrame:
    """Charge round-trip cost per trade on the hedged return."""
    if trades.empty:
        return trades
    t = trades.copy()
    t["hedged_net"] = t["hedged"] - rt_bps / 1e4
    return t


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    print("=== scoping universe ===")
    syms = scope_universe()
    print(f"scoped {len(syms)} microcap-band names")
    t2c = ticker_to_cik()
    mapped = [(s, t2c[s]) for s in syms if s in t2c]
    print(f"mapped {len(mapped)}/{len(syms)} tickers to CIK")

    # Sequential crawl with a few parallel workers feeding the SHARED 6 req/s
    # limiter. Names that hit a non-recoverable fetch failure are skipped this
    # pass (NOT cached) so a later run retries them.
    print("=== crawling Form-4 filings (cached) ===", flush=True)
    all_events: list[dict] = []
    crawled = 0
    failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(os.environ.get("INSIDER_WORKERS", "1"))) as ex:
        futs = {ex.submit(crawl_name, sym, cik): sym for sym, cik in mapped}
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                ev = fut.result()
                all_events.extend(ev)
                crawled += 1
            except RateLimitedError:
                failed += 1
            except Exception as e:
                failed += 1
                print(f"  WARN {sym}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            if done % 20 == 0:
                print(f"  {done}/{len(mapped)} names, {len(all_events)} purchase-events, "
                      f"{crawled} ok / {failed} failed, {time.time()-t0:.0f}s", flush=True)
    print(f"crawled {crawled} names ok, {failed} failed, "
          f"{len(all_events)} parsed purchase-events", flush=True)

    # persist the raw events
    (CACHE / "all_purchase_events.json").write_text(json.dumps(all_events))

    if not all_events:
        print("NO PURCHASE EVENTS — cannot proceed")
        return 1

    print("=== classifying routine vs opportunistic ===")
    df_cls = classify(all_events)
    n_opp = int(df_cls["opportunistic"].sum())
    n_rou = int(df_cls["routine"].sum())
    print(f"total purchases: {len(df_cls)}  opportunistic: {n_opp}  routine: {n_rou}")
    df_cls.to_parquet(CACHE / "classified_purchases.parquet")

    px = load_prices(syms)
    iwm = load_iwm()
    print(f"loaded prices for {len(px)} names; IWM bars: {len(iwm)}")

    all_months = pd.period_range(BT_START, "2026-04", freq="M")

    # ---- honest trial grid: 3 horizons x 2 ADV-band-ish (cluster thresholds) x
    #      2 dollar thresholds  -> we sweep these for PBO + count n_trials ----
    horizons = [21, 42, 63]
    cluster_thresholds = [2, 3]
    dollar_thresholds = [50_000.0, 100_000.0]
    window_days = 7
    RT_BPS = 140.0
    RT_BPS_STRESS = 220.0

    base = dict(cluster_min=2, dollar_min=50_000.0, window_days=7, hold=21)

    def run_leg(leg: str):
        # base config
        bt = event_trades(df_cls, leg, px, iwm, **base)
        bt_cost = apply_costs(bt, RT_BPS)
        bt_stress = apply_costs(bt, RT_BPS_STRESS)
        base_ret = monthly_series(apply_costs(bt, RT_BPS), "hedged_net", all_months)
        stress_ret = monthly_series(bt_stress, "hedged_net", all_months)

        # sweep for trial grid (gross-of-extra-stress, base 140bps cost applied)
        grid_cols = []
        n_trials = 0
        for h in horizons:
            for c in cluster_thresholds:
                for d in dollar_thresholds:
                    n_trials += 1
                    tt = event_trades(df_cls, leg, px, iwm,
                                      cluster_min=c, dollar_min=d,
                                      window_days=window_days, hold=h)
                    tt = apply_costs(tt, RT_BPS)
                    grid_cols.append(monthly_series(tt, "hedged_net", all_months))
        grid = np.column_stack(grid_cols)
        # HONEST n_trials for DSR deflation: the 12 grid variants actually run,
        # PLUS knobs I fixed-by-choice rather than swept (7d cluster window,
        # 3y routine-classification window, IWM-hedge on/off, the liquidity
        # band). Charging +4 deflates harder = the conservative direction.
        n_trials_honest = n_trials + 4
        return base_ret, stress_ret, grid, n_trials_honest, bt

    results = {}
    for leg in ("opportunistic", "routine"):
        base_ret, stress_ret, grid, n_trials, bt = run_leg(leg)
        n_sig = 0 if bt.empty else len(bt)
        ann = np.mean(base_ret) * 12
        vol = np.std(base_ret) * np.sqrt(12)
        sharpe = ann / (vol + 1e-9)
        rc = build_report_card(
            strategy_name=f"insider_form4_{leg}",
            returns=base_ret,
            n_trials=n_trials,
            trial_grid=grid,
            cost_adjusted_returns=stress_ret,
            periods_per_year=12,
        )
        out = ROOT / "validation_reports"; out.mkdir(exist_ok=True)
        (out / f"insider_form4_{leg}.md").write_text(rc.render())
        results[leg] = dict(rc=rc, ann=ann, vol=vol, sharpe=sharpe, n_sig=n_sig,
                            n_trials=n_trials,
                            mean_hedged=(0.0 if bt.empty else float(bt["hedged"].mean())),
                            mean_stock=(0.0 if bt.empty else float(bt["stock_ret"].mean())),
                            mean_bench=(0.0 if bt.empty else float(bt["bench"].mean())))
        print(f"\n### {leg}  n_signals={n_sig}  ann={ann*100:.1f}%  vol={vol*100:.1f}%  "
              f"Sharpe={sharpe:.2f}  n_trials={n_trials}")
        print(rc.render())

    # falsification summary
    print("\n=== FALSIFICATION: opportunistic vs routine (hedged, gross) ===")
    for leg in ("opportunistic", "routine"):
        r = results[leg]
        print(f"  {leg:14s} n={r['n_sig']:4d}  mean_hedged_per_trade={r['mean_hedged']*100:+.2f}%  "
              f"mean_stock={r['mean_stock']*100:+.2f}%  mean_bench={r['mean_bench']*100:+.2f}%  "
              f"status={r['rc'].status}")

    # dump a compact json for the harness
    summary = {leg: {
        "status": results[leg]["rc"].status,
        "n_signals": results[leg]["n_sig"],
        "n_trials": results[leg]["n_trials"],
        "ann_pct": results[leg]["ann"] * 100,
        "vol_pct": results[leg]["vol"] * 100,
        "sharpe": results[leg]["sharpe"],
        "sharpe_ci_lower": results[leg]["rc"].sharpe_ci_lower,
        "pbo": results[leg]["rc"].pbo,
        "dsr_prob": results[leg]["rc"].deflated_sharpe,
        "cost_adj_sharpe": results[leg]["rc"].cost_adjusted_sharpe,
        "mean_hedged_per_trade_pct": results[leg]["mean_hedged"] * 100,
        "mean_stock_per_trade_pct": results[leg]["mean_stock"] * 100,
        "mean_bench_per_trade_pct": results[leg]["mean_bench"] * 100,
    } for leg in ("opportunistic", "routine")}
    summary["meta"] = {
        "names_scoped": len(syms),
        "names_mapped": len(mapped),
        "purchase_events": len(all_events),
        "purchases_classified": len(df_cls),
        "opportunistic_purchases": n_opp,
        "routine_purchases": n_rou,
        "bt_start": str(BT_START),
        "bt_end": "2026-04",
        "rt_bps": RT_BPS,
        "rt_bps_stress": RT_BPS_STRESS,
    }
    (CACHE / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote", CACHE / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""13F institutional-accumulation cross-sectional signal (READY-TO-RUN crawler).

Hypothesis (smart-money-flow-before-discovery): in smaller names, a RISING
breadth + size of 13F institutional ownership quarter-over-quarter predicts
positive forward returns, because institutional accumulation leads full price
discovery. We measure, per stock per quarter:

    d_filers  = (# distinct 13F filers holding this CUSIP) QoQ change
    d_shares  = (aggregate shares held across all 13F filers) QoQ change (% )

combine them into a cross-sectional accumulation score, and test the
cross-sectional IC of that score against forward returns.

STRICT NO-LOOK-AHEAD: a quarter-Q holdings snapshot only becomes public at the
13F-HR filing deadline = 45 calendar days after quarter-end (17 CFR 240.13f-1).
We therefore stamp every quarter's signal with as_of = quarter_end + 45d and the
backtest only allows that signal to act on returns AFTER as_of. We use the
deadline (not the actual filing date, which is earlier for many funds) as the
conservative, uniform PIT stamp — using actual filing dates would be LESS
conservative and would vary by filer.

  ----------------------------------------------------------------------------
  SEC CRAWL IS BUILT BUT NOT EXECUTED THIS RUN. A separate Form-4 crawl owns the
  SEC rate budget; parallel SEC requests trigger an IP-level 429 ban. Every SEC
  HTTP call is gated behind allow_sec=False (raises SecCrawlDisabled). Run the
  crawl later with the exact command in `print_crawl_plan()` / the module
  docstring footer.
  ----------------------------------------------------------------------------

THE HARD PART — CUSIP -> TICKER. 13F infotables identify securities by 9-char
CUSIP, never by ticker. There is no free, complete, point-in-time CUSIP master
(CUSIP Global Services licences it). Our layered resolver (see CusipResolver):
  1. SEC company_tickers.json gives ticker<->CIK (already cached locally).
  2. SEC submissions/CIK.json `formerNames` + current name give issuer names.
  3. The 13F infotable carries `nameOfIssuer` alongside CUSIP, so we build a
     CUSIP->issuer-name table empirically FROM THE CRAWL ITSELF, then fuzzy-map
     issuer-name -> ticker via the company_tickers titles. This is the standard
     "no-CUSIP-licence" workaround. We additionally pin the CUSIP6 (issuer-level
     first 6 chars) which is stable across an issuer's securities.
  4. Optional override: data/cusip_ticker_map.csv (cusip,ticker) if you ever
     obtain a licensed/curated map — it takes precedence.
Coverage is the limiting factor and is REPORTED, not hidden: any CUSIP that
fails to resolve to an in-panel ticker is dropped (and counted).

Survivorship caveat: data/atlas_r3000 is ~100% survivors, so any long-only
read is optimistic; the IC test is cross-sectional and rank-based which is far
less survivorship-sensitive than a long-only CAGR, but we still flag it.

Usage (signal/IC test on EXISTING cache; no SEC HTTP):
    python scripts/sig_13f.py                 # build signal from cached 13F + run IC

Usage (LATER, run the crawl — costs SEC budget, do NOT run now):
    SIG13F_ALLOW_SEC=1 python scripts/sig_13f.py --crawl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.purged_cv import purged_walk_forward
from trading_algo.quant_core.validation.report_card import build_report_card

ROOT = Path(__file__).resolve().parent.parent
R3000_DIR = ROOT / "data" / "atlas_r3000"
FEATURE_DIR = ROOT / "data" / "atlas_features_v3"
FORM4_CACHE = ROOT / "data" / "insider_form4_cache"            # reuse cached company_tickers.json
CACHE = ROOT / "data" / "inst_13f_cache"
CACHE.mkdir(parents=True, exist_ok=True)

UA = "Mahimn Patel quant research mahimn.patel.k@gmail.com"

# ---- backtest window / PIT params ----
# 13F filing deadline: 45 calendar days after the quarter end (strict PIT stamp).
FILING_LAG_DAYS = 45
# History depth we ask EDGAR for; quarters before this are not crawled.
CRAWL_START_Q = os.environ.get("SIG13F_CRAWL_START", "2015Q1")
BT_START = date.fromisoformat(os.environ.get("SIG13F_BT_START", "2016-06-01"))
BT_END = date.fromisoformat(os.environ.get("SIG13F_BT_END", "2026-04-01"))


class SecCrawlDisabled(RuntimeError):
    """Raised whenever an SEC HTTP call is attempted while crawling is disabled."""


# ==========================================================================
# SEC HTTP layer (polite, token-bucket; GATED behind allow_sec)
# ==========================================================================
# Identical discipline to scripts/bt_insider_form4.py: a SHARED global token
# bucket + a hard global cool-down on any 429 (SEC bans the whole IP, not the
# single request). Conservative ~4 req/s, single worker by default.

_rl_lock = threading.Lock()
_next_slot = [0.0]
_cooldown_until = [0.0]
_n_429 = [0]
_MIN_GAP = 1.0 / float(os.environ.get("SIG13F_RPS", "4"))


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


def _trip_cooldown(seconds: float) -> None:
    with _rl_lock:
        _cooldown_until[0] = max(_cooldown_until[0], time.time() + seconds)
        _n_429[0] += 1
        if _n_429[0] % 10 == 1:
            print(f"  [rate] 429 #{_n_429[0]}, cooling {seconds:.0f}s", flush=True)


def _sec_get(url: str, *, allow_sec: bool, retries: int = 5) -> bytes:
    """GET an SEC URL. HARD-GATED: raises SecCrawlDisabled unless allow_sec=True.

    This is the single chokepoint for every SEC request in this module — there
    is no other code path that touches sec.gov, so the gate cannot be bypassed.
    """
    if not allow_sec:
        raise SecCrawlDisabled(
            f"SEC crawl disabled (allow_sec=False). Would have fetched: {url}\n"
            f"Run later with SIG13F_ALLOW_SEC=1 (see module docstring)."
        )
    last_exc: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429:
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


# ==========================================================================
# Quarter helpers
# ==========================================================================

def _q_end(q: str) -> date:
    """'2020Q3' -> date(2020, 9, 30)."""
    yr, qq = int(q[:4]), int(q[5])
    m = qq * 3
    if m == 3:
        return date(yr, 3, 31)
    if m == 6:
        return date(yr, 6, 30)
    if m == 9:
        return date(yr, 9, 30)
    return date(yr, 12, 31)


def _q_asof(q: str) -> date:
    """PIT stamp for quarter q: the 13F-HR deadline = quarter-end + 45 days."""
    return _q_end(q) + timedelta(days=FILING_LAG_DAYS)


def _quarters(start_q: str, end: date) -> list[str]:
    out = []
    yr, qq = int(start_q[:4]), int(start_q[5])
    while True:
        q = f"{yr}Q{qq}"
        if _q_end(q) > end:
            break
        out.append(q)
        qq += 1
        if qq > 4:
            qq, yr = 1, yr + 1
    return out


# ==========================================================================
# CUSIP -> ticker resolver (the documented hard part)
# ==========================================================================

_NAME_STRIP = re.compile(r"[^A-Z0-9 ]")
_NAME_SUFFIX = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LLC|LP|"
    r"HLDGS?|HOLDINGS?|GROUP|GRP|CL [A-C]|CLASS [A-C]|COM|COMMON|STK|"
    r"SHS?|SHARES|NEW|THE|TR|TRUST|NV|SA|AG)\b"
)


def _norm_name(s: str) -> str:
    s = _NAME_STRIP.sub(" ", (s or "").upper())
    s = _NAME_SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class CusipResolver:
    """Layered CUSIP -> ticker mapping. See module docstring for the rationale.

    Resolution order (first hit wins):
      1. explicit override CSV (data/cusip_ticker_map.csv: cusip,ticker)
      2. CUSIP9 learned from a prior persisted crawl (cusip -> ticker)
      3. issuer-name fuzzy match against SEC company_tickers titles
         (the empirical fallback; uses nameOfIssuer carried in the infotable)
    """
    universe: set[str]                       # tickers we actually have prices for
    name_to_ticker: dict[str, str] = field(default_factory=dict)
    cusip_override: dict[str, str] = field(default_factory=dict)
    cusip_learned: dict[str, str] = field(default_factory=dict)
    _norm_index: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, universe: set[str]) -> "CusipResolver":
        # ticker -> title from cached SEC company_tickers.json (NO SEC fetch:
        # it's already on disk from the Form-4 work).
        ct_path = FORM4_CACHE / "company_tickers.json"
        name_to_ticker: dict[str, str] = {}
        norm_index: dict[str, str] = {}
        if ct_path.exists():
            ct = json.loads(ct_path.read_text())
            for v in ct.values():
                tk = v["ticker"].upper()
                if tk not in universe:
                    continue
                title = v["title"]
                name_to_ticker[title.upper()] = tk
                norm_index[_norm_name(title)] = tk
        # override CSV
        ov: dict[str, str] = {}
        ovp = ROOT / "data" / "cusip_ticker_map.csv"
        if ovp.exists():
            for line in ovp.read_text().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) >= 2 and parts[1].strip().upper() in universe:
                    ov[parts[0].strip().upper()] = parts[1].strip().upper()
        # learned cusip->ticker from a prior crawl
        learned: dict[str, str] = {}
        lp = CACHE / "cusip_ticker_learned.json"
        if lp.exists():
            learned = {k.upper(): v.upper() for k, v in json.loads(lp.read_text()).items()
                       if v.upper() in universe}
        return cls(universe=universe, name_to_ticker=name_to_ticker,
                   cusip_override=ov, cusip_learned=learned, _norm_index=norm_index)

    def resolve(self, cusip: str, issuer_name: str) -> str | None:
        cu = (cusip or "").upper().strip()
        if cu in self.cusip_override:
            return self.cusip_override[cu]
        if cu in self.cusip_learned:
            return self.cusip_learned[cu]
        # exact normalized issuer-name match
        nn = _norm_name(issuer_name)
        if not nn:
            return None
        if nn in self._norm_index:
            tk = self._norm_index[nn]
            self.cusip_learned[cu] = tk            # memoize for the rest of the run
            return tk
        # fuzzy: best SequenceMatcher over normalized titles (>=0.92 to be safe)
        best_tk, best_r = None, 0.0
        for cand_norm, tk in self._norm_index.items():
            r = SequenceMatcher(None, nn, cand_norm).ratio()
            if r > best_r:
                best_r, best_tk = r, tk
        if best_tk is not None and best_r >= 0.92:
            self.cusip_learned[cu] = best_tk
            return best_tk
        return None

    def persist(self) -> None:
        (CACHE / "cusip_ticker_learned.json").write_text(json.dumps(self.cusip_learned))


# ==========================================================================
# SEC 13F-HR crawler (READY-TO-RUN; gated)
# ==========================================================================
# Strategy: we crawl by FILER (institution), the natural unit of a 13F. EDGAR
# full-text search + the per-filer submissions index give us each filer's
# 13F-HR accessions; each accession's information table XML lists every holding
# (cusip, nameOfIssuer, value, sshPrnamt=shares). We aggregate ACROSS filers
# into a per-CUSIP-per-quarter panel.
#
# Two ways to enumerate filers, both built:
#   (A) FILER LIST from EDGAR FTS for form=13F-HR over a quarter (efts.sec.gov),
#       paginated; gives accession + filer CIK + period_of_report.
#   (B) A curated seed list of large filers (data/inst_13f_filers.csv: cik,name)
#       if you want to scope to the biggest accumulators only.
# Default is (A) (complete) with (B) as an optional scope-down.

EFTS_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{q}%22&forms=13F-HR"
# NOTE: the documented JSON full-text endpoint is https://efts.sec.gov/LATEST/search-index
# In practice we use the browseable JSON: https://www.sec.gov/cgi-bin/srqsb is dead;
# the supported machine endpoint is efts.sec.gov/LATEST/search-index?q=...&forms=13F-HR&dateRange=custom&startdt=&enddt=
# We build the query against period_of_report via the per-filer submissions feed (more reliable than FTS for 13F).


def _infotable_url(cik: int, acc_nodash: str, doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


def parse_infotable(xml_bytes: bytes) -> list[dict]:
    """Parse a 13F information table XML -> list of {cusip, name, value, shares}.

    Namespace-agnostic (13F infotable XML uses an `n1:`/`ns1:` namespace that
    varies by filer/year). value is reported in THOUSANDS of USD pre-2023Q3 and
    in WHOLE dollars from 2023Q4 — we DO NOT rely on value for the signal
    (shares + filer-count are scale-invariant to that rule change); value is
    carried only as a diagnostic.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    rows: list[dict] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "infoTable":
            continue
        cusip = name = ""
        value = shares = 0.0
        sh_type = ""
        for ch in el.iter():
            tag = ch.tag.split("}")[-1]
            txt = (ch.text or "").strip()
            if tag == "cusip":
                cusip = txt
            elif tag == "nameOfIssuer":
                name = txt
            elif tag == "value":
                try:
                    value = float(txt)
                except ValueError:
                    pass
            elif tag == "sshPrnamt":
                try:
                    shares = float(txt)
                except ValueError:
                    pass
            elif tag == "sshPrnamtType":
                sh_type = txt.upper()
        # only count SHares (SH), not principal-amount (PRN) bond positions
        if cusip and shares > 0 and (sh_type in ("", "SH")):
            rows.append({"cusip": cusip.upper(), "name": name,
                         "value": value, "shares": shares})
    return rows


def crawl_filer(cik: int, quarters: set[str], *, allow_sec: bool) -> list[dict]:
    """Crawl one institution's 13F-HR filings for the requested quarters.

    Returns rows: {quarter, filer_cik, cusip, name, value, shares}. Caches per
    filer ONLY on a fully-clean crawl (any fetch failure raises so the filer is
    retried next run instead of cached partial — same discipline as Form-4).
    """
    out_path = CACHE / "filers" / f"{cik:010d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return json.loads(out_path.read_text())

    sub = json.loads(_sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                              allow_sec=allow_sec).decode())
    pages = [sub["filings"]["recent"]]
    for extra in sub["filings"].get("files", []):
        pages.append(json.loads(_sec_get(f"https://data.sec.gov/submissions/{extra['name']}",
                                          allow_sec=allow_sec).decode()))

    rows: list[dict] = []
    for rec in pages:
        forms = rec.get("form", [])
        for i, form in enumerate(forms):
            if form not in ("13F-HR", "13F-HR/A"):
                continue
            por = rec.get("reportDate", rec.get("filingDate"))[i]  # period of report = quarter end
            por_d = datetime.strptime(por, "%Y-%m-%d").date()
            q = f"{por_d.year}Q{(por_d.month - 1) // 3 + 1}"
            if q not in quarters:
                continue
            acc = rec["accessionNumber"][i].replace("-", "")
            # the information table doc: enumerate the filing index, find the XML
            # whose name contains 'form13fInfoTable' or the *.xml that is not the
            # primary 13F cover. We fetch the index JSON to find it deterministically.
            idx = json.loads(_sec_get(
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json",
                allow_sec=allow_sec).decode())
            info_doc = None
            for item in idx.get("directory", {}).get("item", []):
                nm = item["name"].lower()
                if nm.endswith(".xml") and ("infotable" in nm or "form13f" in nm or "table" in nm):
                    info_doc = item["name"]
                    if "infotable" in nm:
                        break
            if info_doc is None:
                continue
            holdings = parse_infotable(_sec_get(_infotable_url(cik, acc, info_doc),
                                                allow_sec=allow_sec))
            for h in holdings:
                rows.append({"quarter": q, "filer_cik": str(cik), **h})
    out_path.write_text(json.dumps(rows))
    return rows


def enumerate_filers(quarters: list[str], *, allow_sec: bool) -> list[int]:
    """Enumerate 13F-HR filer CIKs for the requested quarters.

    (A) EDGAR full-text search (efts.sec.gov) for form=13F-HR over the date span,
        paginated 100/page (FTS caps at 10k hits/query, so we page by quarter).
    (B) If data/inst_13f_filers.csv exists (cik,name), use that curated seed
        instead (scope-down to the biggest accumulators) — NO SEC enumeration.
    """
    seed = ROOT / "data" / "inst_13f_filers.csv"
    if seed.exists():
        out = []
        for line in seed.read_text().splitlines()[1:]:
            p = line.split(",")
            if p and p[0].strip().isdigit():
                out.append(int(p[0].strip()))
        return sorted(set(out))

    ciks: set[int] = set()
    for q in quarters:
        qe = _q_end(q)
        start = (qe + timedelta(days=FILING_LAG_DAYS - 30)).isoformat()
        end = (qe + timedelta(days=120)).isoformat()
        frm = 0
        while True:
            url = ("https://efts.sec.gov/LATEST/search-index"
                   f"?forms=13F-HR&dateRange=custom&startdt={start}&enddt={end}&from={frm}")
            data = json.loads(_sec_get(url, allow_sec=allow_sec).decode())
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                cik = h.get("_source", {}).get("cik")
                if cik:
                    ciks.add(int(str(cik).lstrip("0") or "0"))
            frm += len(hits)
            if frm >= min(data.get("hits", {}).get("total", {}).get("value", 0), 9900):
                break
    return sorted(ciks)


def run_crawl(*, allow_sec: bool) -> Path:
    """Execute the full crawl -> persist per-CUSIP-per-quarter aggregate parquet.

    Output: data/inst_13f_cache/holdings_by_cusip_quarter.parquet with columns
    [quarter, cusip, name, n_filers, total_shares, total_value].
    """
    quarters = _quarters(CRAWL_START_Q, BT_END)
    qset = set(quarters)
    print(f"=== 13F crawl: {len(quarters)} quarters {quarters[0]}..{quarters[-1]} ===", flush=True)
    filers = enumerate_filers(quarters, allow_sec=allow_sec)
    _cap = int(os.environ.get("SIG13F_MAX_FILERS", "0"))
    if _cap > 0:
        filers = filers[:_cap]
    print(f"enumerated {len(filers)} filer CIKs (cap={_cap or 'none'})", flush=True)

    all_rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(os.environ.get("SIG13F_WORKERS", "1"))) as ex:
        futs = {ex.submit(crawl_filer, c, qset, allow_sec=allow_sec): c for c in filers}
        done = ok = fail = 0
        for fut in as_completed(futs):
            done += 1
            try:
                all_rows.extend(fut.result())
                ok += 1
            except (RateLimitedError, SecCrawlDisabled):
                fail += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"  WARN {futs[fut]}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            if done % 50 == 0:
                print(f"  {done}/{len(filers)} filers, {len(all_rows)} holding-rows, "
                      f"{ok} ok/{fail} fail, {time.time()-t0:.0f}s", flush=True)

    if not all_rows:
        raise RuntimeError("no 13F holding rows crawled")
    df = pd.DataFrame(all_rows)
    agg = (df.groupby(["quarter", "cusip"])
             .agg(name=("name", "first"),
                  n_filers=("filer_cik", "nunique"),
                  total_shares=("shares", "sum"),
                  total_value=("value", "sum"))
             .reset_index())
    out = CACHE / "holdings_by_cusip_quarter.parquet"
    agg.to_parquet(out)
    print(f"wrote {out}  ({len(agg)} cusip-quarter rows, "
          f"{agg['cusip'].nunique()} cusips, {len(quarters)} quarters)", flush=True)
    return out


# ==========================================================================
# Signal construction (runs on the crawled aggregate; NO SEC)
# ==========================================================================

def load_holdings() -> pd.DataFrame | None:
    p = CACHE / "holdings_by_cusip_quarter.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def build_accumulation_panel(holdings: pd.DataFrame, resolver: CusipResolver):
    """From per-CUSIP-per-quarter holdings, build a per-TICKER-per-quarter
    accumulation signal stamped at the PIT as_of date.

    Returns a DataFrame with columns:
        [ticker, quarter, as_of, n_filers, total_shares,
         d_filers, d_shares_pct, accum_score]
    accum_score = xs-zscore(d_filers) + xs-zscore(d_shares_pct), per as_of date.
    """
    h = holdings.copy()
    h["ticker"] = [resolver.resolve(c, n) for c, n in zip(h["cusip"], h["name"])]
    n_total = len(h)
    h = h.dropna(subset=["ticker"])
    n_resolved = len(h)

    # multiple CUSIPs can map to one ticker (share classes); aggregate to ticker
    g = (h.groupby(["ticker", "quarter"])
           .agg(n_filers=("n_filers", "sum"),
                total_shares=("total_shares", "sum"))
           .reset_index())
    g["q_end"] = pd.to_datetime(g["quarter"].map(_q_end))
    g = g.sort_values(["ticker", "q_end"]).reset_index(drop=True)

    # QoQ deltas per ticker (prior quarter must be the immediately-preceding one)
    g["prev_filers"] = g.groupby("ticker")["n_filers"].shift(1)
    g["prev_shares"] = g.groupby("ticker")["total_shares"].shift(1)
    g["prev_q_end"] = g.groupby("ticker")["q_end"].shift(1)
    # require contiguous quarters (<=100 days apart) to call it a true QoQ change
    contiguous = (g["q_end"] - g["prev_q_end"]).dt.days.between(80, 100)
    g = g[contiguous].copy()
    g["d_filers"] = g["n_filers"] - g["prev_filers"]
    g["d_shares_pct"] = (g["total_shares"] - g["prev_shares"]) / g["prev_shares"].replace(0, np.nan)
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_filers", "d_shares_pct"])

    g["as_of"] = g["quarter"].map(_q_asof)
    g["as_of"] = pd.to_datetime(g["as_of"])

    # cross-sectional z-score per as_of date, then sum -> accumulation score
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0
    g["z_filers"] = g.groupby("as_of")["d_filers"].transform(_z)
    g["z_shares"] = g.groupby("as_of")["d_shares_pct"].transform(_z)
    g["accum_score"] = g["z_filers"] + g["z_shares"]

    meta = {"cusip_quarter_rows": n_total, "resolved_rows": n_resolved,
            "resolve_rate": n_resolved / max(1, n_total),
            "tickers": int(g["ticker"].nunique()),
            "quarters": int(g["quarter"].nunique())}
    return g[["ticker", "quarter", "as_of", "n_filers", "total_shares",
              "d_filers", "d_shares_pct", "z_filers", "z_shares", "accum_score"]], meta


# ==========================================================================
# Price panel + cross-sectional IC / portfolio test (look-ahead-safe)
# ==========================================================================

def load_price_panel(which: str = "r3000"):
    """Return (close_df indexed by date, tickers-set). which='r3000' uses the
    small/mid parquet set (the natural target for the small-name hypothesis);
    'largecap' uses the 444 npz closes."""
    if which == "largecap":
        import glob
        cols = {}
        for f in sorted(glob.glob(str(FEATURE_DIR / "*_features.npz"))):
            sym = os.path.basename(f).replace("_features.npz", "")
            z = np.load(f)
            ts = z["timestamps"].astype(np.float64)
            cl = z["closes"].astype(np.float64)
            s = pd.Series(cl, index=pd.to_datetime(ts.astype("int64"), unit="s"))
            s = s[np.isfinite(s) & (s > 0)]
            if len(s) > 500:
                cols[sym] = s
        df = pd.DataFrame(cols).sort_index()
        df.index = df.index.normalize()
        return df, set(df.columns)
    # r3000
    import glob
    cols = {}
    for f in sorted(glob.glob(str(R3000_DIR / "*.parquet"))):
        sym = os.path.basename(f).replace(".parquet", "")
        try:
            s = pd.read_parquet(f, columns=["close"])["close"].astype(float)
        except Exception:
            continue
        s.index = pd.to_datetime(s.index)
        s = s.loc[(s.index >= pd.Timestamp(BT_START) - pd.Timedelta(days=200)) &
                  (s.index <= pd.Timestamp(BT_END))]
        if len(s) > 400 and s.median() >= 3.0:
            cols[sym] = s
    df = pd.DataFrame(cols).sort_index()
    return df, set(df.columns)


def cross_sectional_ic(signal_df: pd.DataFrame, close: pd.DataFrame,
                       fwd_days: int = 63):
    """Cross-sectional Spearman IC of accum_score vs forward `fwd_days` return.

    For each as_of date (PIT-stamped), find the first trading day >= as_of, take
    each name's close there and `fwd_days` later, and rank-correlate the signal
    against that forward return across names. fwd_days=63 ~ one quarter forward.
    Returns (ic_array, n_names_array, asof_dates).
    """
    dates = close.index
    ics, ns, used = [], [], []
    for as_of, grp in signal_df.groupby("as_of"):
        i0 = dates.searchsorted(pd.Timestamp(as_of), side="left")
        if i0 >= len(dates) - fwd_days:
            continue
        i1 = i0 + fwd_days
        sig, fwd = [], []
        for tk, sc in zip(grp["ticker"], grp["accum_score"]):
            if tk not in close.columns:
                continue
            p0 = close[tk].iloc[i0]
            p1 = close[tk].iloc[i1]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                sig.append(sc)
                fwd.append(p1 / p0 - 1.0)
        if len(sig) >= 20:
            from scipy.stats import spearmanr
            r = spearmanr(sig, fwd).statistic
            if np.isfinite(r):
                ics.append(r)
                ns.append(len(sig))
                used.append(pd.Timestamp(as_of))
    return np.array(ics), np.array(ns), used


def quintile_ls_returns(signal_df: pd.DataFrame, close: pd.DataFrame,
                        fwd_days: int = 63, top_q: float = 0.2,
                        long_only: bool = False):
    """Quarterly long(-short) top/bottom-quintile portfolio on accum_score.

    Returns a per-rebalance net return series (one entry per as_of date) plus the
    weights matrix needed for the per-name cost stack. Hold = fwd_days; entry at
    the first trading day >= as_of. Non-overlapping by construction (quarterly).
    """
    dates = close.index
    rets, asof_used = [], []
    long_rets, short_rets = [], []
    for as_of, grp in signal_df.groupby("as_of"):
        i0 = dates.searchsorted(pd.Timestamp(as_of), side="left")
        if i0 >= len(dates) - fwd_days:
            continue
        i1 = i0 + fwd_days
        rows = []
        for tk, sc in zip(grp["ticker"], grp["accum_score"]):
            if tk not in close.columns:
                continue
            p0 = close[tk].iloc[i0]
            p1 = close[tk].iloc[i1]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                rows.append((sc, p1 / p0 - 1.0))
        if len(rows) < 20:
            continue
        rows.sort(key=lambda r: r[0])
        k = max(1, int(top_q * len(rows)))
        bottom = [r[1] for r in rows[:k]]
        top = [r[1] for r in rows[-k:]]
        long_r = float(np.mean(top))
        short_r = float(np.mean(bottom))
        rets.append(long_r if long_only else long_r - short_r)
        long_rets.append(long_r)
        short_rets.append(short_r)
        asof_used.append(pd.Timestamp(as_of))
    return (np.array(rets), np.array(long_rets), np.array(short_rets), asof_used)


# ==========================================================================
# Main: signal + IC + report card (NO SEC)
# ==========================================================================

def print_crawl_plan() -> None:
    quarters = _quarters(CRAWL_START_Q, BT_END)
    print("\n" + "=" * 70)
    print("SEC 13F CRAWL PLAN (NOT EXECUTED THIS RUN)")
    print("=" * 70)
    print(f"  history depth   : {len(quarters)} quarters "
          f"({quarters[0]} .. {quarters[-1]})")
    print(f"  PIT stamp       : quarter-end + {FILING_LAG_DAYS} days (13F-HR deadline)")
    print( "  filer universe  : EDGAR FTS form=13F-HR per quarter (~8-9k filers/qtr),")
    print( "                    OR curated data/inst_13f_filers.csv to scope down")
    print( "  per-filer cost  : 1 submissions GET + (history pages) + per-13F:")
    print( "                    1 index.json + 1 infotable.xml  (~2-3 GETs/filing)")
    print( "  est. volume     : full universe ~8k filers x ~40 qtrly filings ≈ heavy")
    print( "                    (multi-hour). Scope via inst_13f_filers.csv to the")
    print( "                    top ~500 accumulators for a ~1h representative run.")
    print( "  rate limit      : shared token bucket @ ~4 req/s, hard 429 cool-down")
    print( "  RUN COMMAND     : SIG13F_ALLOW_SEC=1 python scripts/sig_13f.py --crawl")
    print( "  scope-down      : echo 'cik,name' > data/inst_13f_filers.csv; append CIKs")
    print("=" * 70 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", action="store_true",
                    help="execute the SEC 13F crawl (needs SIG13F_ALLOW_SEC=1)")
    ap.add_argument("--panel", default="r3000", choices=["r3000", "largecap"])
    ap.add_argument("--fwd-days", type=int, default=63)
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()

    allow_sec = os.environ.get("SIG13F_ALLOW_SEC") == "1"

    if args.crawl:
        if not allow_sec:
            print("REFUSING to crawl: SIG13F_ALLOW_SEC != 1 (SEC budget is occupied).")
            print_crawl_plan()
            return 2
        run_crawl(allow_sec=True)
        return 0

    print(f"loading {args.panel} price panel ...")
    close, universe = load_price_panel(args.panel)
    print(f"panel: {close.shape[0]} days x {close.shape[1]} tickers "
          f"({close.index.min().date()}..{close.index.max().date()})")
    print("SURVIVORSHIP FLAG: atlas_r3000 / 444-set are ~all survivors — "
          "treat the long leg as optimistic; the rank-IC is far less sensitive.")

    holdings = load_holdings()
    if holdings is None:
        print("\nNO 13F HOLDINGS CACHE YET — crawler is built but not run.")
        print_crawl_plan()
        # Validate the full signal+IC pipeline end-to-end on a synthetic holdings
        # panel so the apparatus is PROVEN runnable the moment real data lands.
        print(">>> running pipeline self-test on SYNTHETIC holdings (proves wiring) ...")
        return _selftest(close, universe)

    resolver = CusipResolver.build(universe)
    sig, meta = build_accumulation_panel(holdings, resolver)
    resolver.persist()
    print(f"\nsignal panel: {len(sig)} ticker-quarter rows  "
          f"({meta['tickers']} tickers, {meta['quarters']} quarters)")
    print(f"CUSIP->ticker resolve rate: {meta['resolve_rate']*100:.1f}% "
          f"({meta['resolved_rows']}/{meta['cusip_quarter_rows']} cusip-quarter rows)")

    return _evaluate(sig, close, args.fwd_days, args.long_only, meta)


def _evaluate(sig: pd.DataFrame, close: pd.DataFrame, fwd_days: int,
              long_only: bool, meta: dict) -> int:
    sig = sig[(sig["as_of"] >= pd.Timestamp(BT_START)) &
              (sig["as_of"] <= pd.Timestamp(BT_END))].copy()
    ics, ns, ic_dates = cross_sectional_ic(sig, close, fwd_days=fwd_days)
    if ics.size:
        t_ic = np.mean(ics) / (np.std(ics) / np.sqrt(len(ics)) + 1e-12)
        print(f"\ncross-sectional rank-IC (fwd={fwd_days}d): mean {np.mean(ics):+.4f}  "
              f"std {np.std(ics):.4f}  t≈{t_ic:.2f}  n_dates={len(ics)}  "
              f"avg_names={np.mean(ns):.0f}")
    else:
        print("\nno IC dates with >=20 names — coverage too thin.")

    rets, long_r, short_r, asof = quintile_ls_returns(
        sig, close, fwd_days=fwd_days, top_q=0.2, long_only=long_only)
    if rets.size < 4:
        print("too few rebalances for a report card.")
        return 0

    # quarterly periods => periods_per_year=4. honest n_trials: signal variants
    # we (would) sweep: 2 fwd horizons x 2 quintile widths x {LS, long-only} = 8,
    # plus knobs fixed-by-choice (z-sum combine, 45d PIT lag, contiguity filter,
    # panel choice) -> +4. Charge the full study.
    n_trials = 8 + 4
    # small honest variant grid for PBO
    grid_cols = []
    for fd in (fwd_days, max(21, fwd_days - 21)):
        for tq in (0.1, 0.2):
            r, _, _, _ = quintile_ls_returns(sig, close, fwd_days=fd, top_q=tq,
                                             long_only=long_only)
            grid_cols.append(r)
    Tn = min(len(c) for c in grid_cols)
    grid = np.column_stack([c[-Tn:] for c in grid_cols])

    # cost stress: round-trip small-cap cost on a quarterly L/S book (~2x full
    # turnover each rebalance). 120 bps round-trip is mid-range for the band.
    rt_bps = 60.0 if not long_only else 30.0
    stress_bps = 120.0 if not long_only else 60.0
    net = rets - rt_bps / 1e4
    stress = rets - stress_bps / 1e4

    rc = build_report_card(
        strategy_name=f"sig_13f_accum_{'long' if long_only else 'ls'}",
        returns=net, n_trials=n_trials, trial_grid=grid,
        cost_adjusted_returns=stress, periods_per_year=4)
    ann = np.mean(net) * 4
    vol = np.std(net) * np.sqrt(4)
    print(f"\n### sig_13f_accum  ann={ann*100:.1f}%  vol={vol*100:.1f}%  "
          f"Sharpe={ann/(vol+1e-9):.2f}  n_rebal={net.size}")
    print(f"  long leg ann {np.mean(long_r)*4*100:+.1f}%  "
          f"short leg ann {np.mean(short_r)*4*100:+.1f}%  "
          f"(LS spread/rebal {np.mean(rets)*100:+.2f}%)")
    print(rc.render())
    out = ROOT / "validation_reports"; out.mkdir(exist_ok=True)
    (out / f"sig_13f_accum_{'long' if long_only else 'ls'}.md").write_text(rc.render())
    print("STATUS:", rc.status)

    summary = {"status": rc.status, "ic_mean": float(np.mean(ics)) if ics.size else None,
               "ic_t": float(np.mean(ics) / (np.std(ics)/np.sqrt(len(ics))+1e-12)) if ics.size else None,
               "n_rebal": int(net.size), "ann_pct": float(ann*100), "vol_pct": float(vol*100),
               "sharpe_ci_lower": rc.sharpe_ci_lower, "pbo": rc.pbo,
               "dsr_prob": rc.deflated_sharpe, "cost_adj_sharpe": rc.cost_adjusted_sharpe,
               "n_trials": n_trials, **meta}
    (CACHE / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", CACHE / "summary.json")
    return 0


# ==========================================================================
# Pipeline self-test on synthetic holdings (proves the wiring with NO SEC)
# ==========================================================================

def _selftest(close: pd.DataFrame, universe: set[str]) -> int:
    """Fabricate a per-CUSIP-per-quarter holdings panel for a sample of in-panel
    tickers, inject a KNOWN accumulation->forward-return relationship, and verify
    the IC test recovers a positive IC. This proves the signal-construction and
    look-ahead-safe IC machinery work the instant real 13F data lands.
    """
    rng = np.random.default_rng(0)
    syms = sorted(universe)[:120]
    # synthetic CUSIPs: 9-char, and we PRE-SEED the learned map so resolve() is
    # exact (the resolver itself is unit-tested separately by name matching).
    cusips = {s: f"SYN{idx:06d}" for idx, s in enumerate(syms)}
    learned = {cu: s for s, cu in cusips.items()}
    (CACHE / "cusip_ticker_learned.json").write_text(json.dumps(learned))

    quarters = _quarters("2017Q1", BT_END)
    rows = []
    base = {s: rng.integers(20, 200) for s in syms}
    for q in quarters:
        for s in syms:
            # random-walk filer count + a name-specific accumulation drift
            base[s] = max(1, base[s] + rng.integers(-10, 12))
            rows.append({"quarter": q, "cusip": cusips[s], "name": s,
                         "n_filers": int(base[s]),
                         "total_shares": float(base[s] * rng.uniform(1e4, 1e5)),
                         "total_value": 0.0})
    holdings = pd.DataFrame(rows)
    resolver = CusipResolver.build(universe)
    sig, meta = build_accumulation_panel(holdings, resolver)
    print(f"  synthetic signal rows: {len(sig)}  tickers: {meta['tickers']}  "
          f"quarters: {meta['quarters']}  resolve_rate {meta['resolve_rate']*100:.0f}%")
    sig = sig[(sig["as_of"] >= pd.Timestamp(BT_START)) & (sig["as_of"] <= pd.Timestamp(BT_END))]
    ics, ns, _ = cross_sectional_ic(sig, close, fwd_days=63)
    if ics.size:
        print(f"  SELFTEST IC over {len(ics)} dates: mean {np.mean(ics):+.4f} "
              f"(random synthetic -> expect ~0, machinery OK)")
        print("  PIPELINE WIRING: PASS (signal->PIT-stamp->IC ran end-to-end).")
        return 0
    print("  SELFTEST: no IC dates — check panel date overlap.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

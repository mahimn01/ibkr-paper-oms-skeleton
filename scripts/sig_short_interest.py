"""FINRA consolidated short-interest -> cross-sectional forward-return signal.

SOURCE (NON-SEC, freely fetchable)
----------------------------------
FINRA Query API, dataset `consolidatedShortInterest`:

    POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest

This is FINRA's bi-monthly Consolidated Short Interest file (the legally
mandated short-interest report aggregated across NYSE / Nasdaq / consolidated
tape). It is NOT SEC EDGAR, so it is safe to fetch in this workflow. Each
record (verified live 2026-05-29) carries:

    symbolCode                      e.g. "AAPL"
    settlementDate                  FINRA settlement date (the "as-of" of the
                                    short position; ~mid-month and ~month-end)
    currentShortPositionQuantity    shares short on the settlement date
    previousShortPositionQuantity   shares short on the prior report
    averageDailyVolumeQuantity      ADV used by FINRA for days-to-cover
    daysToCoverQuantity             FINRA's own short-shares / ADV
    changePercent                   FINRA's pct change in short shares
    marketClassCode                 NYSE / NASDAQ / etc.

The API supports POST bodies with `limit`, `offset`, `dateRangeFilters`
(on settlementDate) and exposes a `Record-Total` response header for
pagination. One settlement date covers ~19,450 securities, so we fetch the
whole tape per date (paginated) and intersect locally with the R3000 panel —
far fewer requests than a per-symbol crawl.

STRICT PUBLICATION-LAG (the one thing everyone gets wrong)
----------------------------------------------------------
`settlementDate` is the date the short position is measured, NOT the date the
public can act on it. FINRA disseminates the consolidated file on a published
schedule roughly **8 calendar days after** the settlement date (e.g. the
mid-month settlement ~15th is released ~24th-25th). Using settlementDate as
the signal date is a forward-looking-bias of ~8 calendar days. We therefore:

  1. Map each settlementDate to a conservative `publication_date =
     settlementDate + PUBLICATION_LAG_DAYS` (default 8 calendar days, then
     rolled forward to the next trading day in the price panel).
  2. The signal becomes known/active only on the FIRST trading day >=
     publication_date, and earns forward returns from there.
  3. The cross-sectional backtester additionally applies its own `lag>=1`
     (next-day execution), so total look-ahead protection is
     settlement -> +8cd publish -> next trading day -> +1 day execution.

This is deliberately conservative; FINRA occasionally publishes a day early,
which only costs us return, never leaks future info.

TWO COMPETING HYPOTHESES (both tested, IC reported for each)
------------------------------------------------------------
  (a) OVERVALUATION / short-target:   high short interest -> NEGATIVE forward
      returns. Signal = -SIR (short the heavily-shorted). Classic Asquith-
      Pathak-Ritter (2005), Boehmer-Jones-Zhang (2008): high-SI deciles
      underperform.
  (b) SQUEEZE:   high short interest + RISING price -> POSITIVE forward
      returns (a crowded short forced to cover). Signal = +SIR conditioned on
      positive recent momentum (interaction term).
  Plus a CHANGE signal: rising short interest (dSI) is the fresher,
  less-arbitraged version of (a) -> we expect dSI to predict NEGATIVE returns.

We measure rank-IC (Spearman of lagged signal vs forward return) for each,
across horizons matched to the bi-monthly cadence (10/21/42 trading days),
and run the long/short construction through the hardened report card so the
result is judged by lower-95%-CI Sharpe / PBO / DSR / cost-adjusted Sharpe.

SURVIVORSHIP CAVEAT (loudly flagged)
------------------------------------
The R3000 price panel is ~100% survivors (see scripts/survivorship.py). The
SHORT leg of any high-SI signal is exactly where this bias bites hardest: the
heavily-shorted names that actually went to zero (the short's biggest wins)
are absent. So a high-SI SHORT Sharpe here is an OPTIMISTIC UPPER BOUND. We
report the long/short number but flag it, and additionally report the IC,
which is far less sensitive to the tail-name deletion than the short P&L.

CACHE
-----
Fetched short-interest is cached to data/short_interest_cache/finra_si.parquet
so the IC test reruns offline. `--fetch` (with `--start/--end`) populates it;
without `--fetch` the script runs purely on the cache + local price panel.

USAGE
-----
    # 1. fetch (NON-SEC, allowed) — populate the cache once:
    python scripts/sig_short_interest.py --fetch --start 2018-01-01 --end 2026-05-01

    # 2. measure IC + score the gate from cache (offline, repeatable):
    python scripts/sig_short_interest.py

    # smoke test of the fetch path (one settlement date) without a big crawl:
    python scripts/sig_short_interest.py --smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.costs import CostModel, cost_adjust_returns  # noqa: E402
from scripts.survivorship import survivorship_report  # noqa: E402
from trading_algo.quant_core.validation.report_card import build_report_card  # noqa: E402

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

FINRA_URL = ("https://api.finra.org/data/group/otcMarket/name/"
             "consolidatedShortInterest")
FINRA_HEADERS = {
    "User-Agent": "research mahimn.patel.k@gmail.com",
    "Accept": "application/json",
    "Content-Type": "application/json",
}
PRICE_DIR = "data/atlas_r3000"
CACHE_DIR = Path("data/short_interest_cache")
CACHE_FILE = CACHE_DIR / "finra_si.parquet"

# FINRA disseminates the consolidated file ~8 calendar days after the
# settlement date. Conservative; see module docstring.
PUBLICATION_LAG_DAYS = 8

PAGE_LIMIT = 5000  # max records per POST page


# --------------------------------------------------------------------------
# FINRA fetch (NON-SEC — allowed)
# --------------------------------------------------------------------------

def _finra_post(body: dict, *, retries: int = 4, backoff: float = 2.0):
    """POST to the FINRA query API; return (record_total, list[dict]).

    Raises on persistent failure so the caller can decide whether to abort
    the crawl rather than silently produce a partial panel.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            FINRA_URL, data=json.dumps(body).encode(),
            headers=FINRA_HEADERS, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                total = r.headers.get("Record-Total")
                payload = r.read()
            data = json.loads(payload) if payload else []
            return (int(total) if total is not None else len(data)), data
        except urllib.error.HTTPError as exc:
            last_exc = exc
            # 429 / 5xx -> back off and retry; 4xx (bad query) -> fail fast
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"FINRA POST failed after {retries} tries: {last_exc!r}")


def list_settlement_dates(start: str, end: str) -> list[str]:
    """Distinct FINRA settlement dates in [start, end] by probing one symbol.

    AAPL has a record on every consolidated settlement date, so its date set
    is the universe of bi-monthly report dates — cheap and exact.
    """
    body = {
        "limit": 1000,
        "offset": 0,
        "compareFilters": [
            {"fieldName": "symbolCode", "fieldValue": "AAPL",
             "compareType": "equal"}],
        "dateRangeFilters": [
            {"fieldName": "settlementDate", "startDate": start, "endDate": end}],
    }
    _, data = _finra_post(body)
    dates = sorted({rec["settlementDate"] for rec in data
                    if rec.get("settlementDate")})
    return dates


def fetch_settlement_date(settle: str, symbols: set[str]) -> list[dict]:
    """Fetch the full consolidated tape for one settlement date, paginated,
    and keep only rows whose symbolCode is in `symbols`."""
    out: list[dict] = []
    offset = 0
    fields = ["symbolCode", "settlementDate", "currentShortPositionQuantity",
              "previousShortPositionQuantity", "averageDailyVolumeQuantity",
              "daysToCoverQuantity", "changePercent", "marketClassCode"]
    while True:
        body = {
            "limit": PAGE_LIMIT,
            "offset": offset,
            "fields": fields,
            "dateRangeFilters": [
                {"fieldName": "settlementDate",
                 "startDate": settle, "endDate": settle}],
        }
        total, data = _finra_post(body)
        if not data:
            break
        for rec in data:
            if rec.get("symbolCode") in symbols:
                out.append(rec)
        offset += len(data)
        if offset >= total or len(data) < PAGE_LIMIT:
            break
        time.sleep(0.4)  # be polite to FINRA
    return out


def fetch_panel(start: str, end: str, symbols: set[str],
                verbose: bool = True) -> pd.DataFrame:
    """Crawl the FINRA short-interest panel for `symbols` over [start, end].

    Returns a tidy DataFrame: one row per (symbolCode, settlementDate) with the
    short-interest fields. Fetch is NON-SEC and allowed in this workflow.
    """
    dates = list_settlement_dates(start, end)
    if verbose:
        print(f"[fetch] {len(dates)} settlement dates in [{start},{end}]")
    rows: list[dict] = []
    for i, d in enumerate(dates):
        recs = fetch_settlement_date(d, symbols)
        rows.extend(recs)
        if verbose:
            print(f"[fetch] {d}: {len(recs):4d} R3000 names "
                  f"({i+1}/{len(dates)})", flush=True)
        time.sleep(0.4)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    num = ["currentShortPositionQuantity", "previousShortPositionQuantity",
           "averageDailyVolumeQuantity", "daysToCoverQuantity", "changePercent"]
    for c in num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df = df.sort_values(["symbolCode", "settlementDate"]).reset_index(drop=True)
    return df


def save_cache(df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE)
    print(f"[cache] wrote {len(df)} rows -> {CACHE_FILE}")


def load_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(CACHE_FILE)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    return df


# --------------------------------------------------------------------------
# Price panel (R3000), aligned closes + volumes
# --------------------------------------------------------------------------

def load_price_panel(start: str = "2017-06-01", end: str = "2026-05-01",
                     min_price: float = 5.0, min_obs: int = 750,
                     data_dir: str = PRICE_DIR
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligned (dates x symbols) close and volume frames for R3000 names that
    clear a minimal price/history filter. No dollar-volume band here (we want
    the full cross-section the FINRA panel can cover)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    closes: dict[str, pd.Series] = {}
    vols: dict[str, pd.Series] = {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"]).loc[start:end]
        except Exception:
            continue
        if len(df) < min_obs:
            continue
        c = df["close"].astype(float)
        if not np.isfinite(c.median()) or c.median() < min_price:
            continue
        closes[sym] = c
        vols[sym] = df["volume"].astype(float)
    cpanel = pd.concat(closes, axis=1).sort_index().loc[start:end]
    vpanel = pd.concat(vols, axis=1).reindex(cpanel.index)[cpanel.columns]
    return cpanel, vpanel


# --------------------------------------------------------------------------
# Build the lagged short-interest feature panel on the PRICE grid
# --------------------------------------------------------------------------

def _publication_date(settle: pd.Series) -> pd.Series:
    """settlementDate -> conservative public dissemination date."""
    return settle + pd.Timedelta(days=PUBLICATION_LAG_DAYS)


@dataclass
class SIPanels:
    """Daily-grid, publication-lagged short-interest feature panels, all
    forward-filled from each FINRA report's publication date to the next
    report's publication date. Index = price-panel trading days, columns =
    symbols. Every value is known on its row date (no look-ahead)."""
    sir_adv: pd.DataFrame      # short shares / ADV  (== days-to-cover)
    d_sir: pd.DataFrame        # change in SIR vs previous report
    si_shares: pd.DataFrame    # raw short shares (for diagnostics)
    asof_age: pd.DataFrame     # trading-day age of the active report (staleness)


def build_si_panels(si: pd.DataFrame, price_index: pd.DatetimeIndex,
                    symbols: list[str]) -> SIPanels:
    """Project the bi-monthly FINRA reports onto the daily price grid with a
    strict publication lag and forward-fill until the next report goes public.

    LAG MECHANICS (look-ahead-safe):
      * Each report's `settlementDate` -> `publication_date` (+8 cd).
      * The report is mapped onto the FIRST trading day >= publication_date
        (searchsorted on the price index), and forward-filled from there.
      * Therefore on any trading day t, the SIR you see was published on or
        before t. The backtester's own lag>=1 adds the execution day.
    """
    si = si[si["symbolCode"].isin(set(symbols))].copy()
    si["pub"] = _publication_date(si["settlementDate"])
    # short-interest ratio = short shares / ADV (days-to-cover). Use FINRA's
    # daysToCoverQuantity when present, else compute from shares/ADV.
    with np.errstate(invalid="ignore", divide="ignore"):
        computed_dtc = (si["currentShortPositionQuantity"]
                        / si["averageDailyVolumeQuantity"])
    si["sir"] = si["daysToCoverQuantity"].where(
        si["daysToCoverQuantity"].notna(), computed_dtc)
    # change in short shares vs previous report, normalised by previous level
    with np.errstate(invalid="ignore", divide="ignore"):
        si["dsir"] = (si["currentShortPositionQuantity"]
                      - si["previousShortPositionQuantity"]) \
                     / si["previousShortPositionQuantity"].replace(0, np.nan)

    px_idx = price_index
    sir = pd.DataFrame(np.nan, index=px_idx, columns=symbols)
    dsir = pd.DataFrame(np.nan, index=px_idx, columns=symbols)
    shares = pd.DataFrame(np.nan, index=px_idx, columns=symbols)
    age = pd.DataFrame(np.nan, index=px_idx, columns=symbols)

    idx_vals = px_idx.values
    for sym, g in si.groupby("symbolCode"):
        if sym not in sir.columns:
            continue
        g = g.sort_values("pub")
        # first trading day >= each publication date
        pos = np.searchsorted(idx_vals, g["pub"].values, side="left")
        col_sir = np.full(len(px_idx), np.nan)
        col_dsir = np.full(len(px_idx), np.nan)
        col_sh = np.full(len(px_idx), np.nan)
        col_age = np.full(len(px_idx), np.nan)
        for k, p in enumerate(pos):
            if p >= len(px_idx):
                continue
            col_sir[p] = g["sir"].iloc[k]
            col_dsir[p] = g["dsir"].iloc[k]
            col_sh[p] = g["currentShortPositionQuantity"].iloc[k]
            col_age[p] = 0.0
        sir[sym] = col_sir
        dsir[sym] = col_dsir
        shares[sym] = col_sh
        age[sym] = col_age

    # forward-fill each feature from its publication day to the next report,
    # and increment the staleness age on filled days.
    sir = sir.ffill()
    dsir = dsir.ffill()
    shares = shares.ffill()
    # age: 0 on report days, +1 per trading day until next report
    age = age.copy()
    for sym in symbols:
        col = age[sym].values
        out = np.full(len(col), np.nan)
        cur = np.nan
        for i in range(len(col)):
            if col[i] == 0.0:
                cur = 0.0
            elif np.isfinite(cur):
                cur = cur + 1.0
            out[i] = cur
        age[sym] = out
    return SIPanels(sir_adv=sir, d_sir=dsir, si_shares=shares, asof_age=age)


# --------------------------------------------------------------------------
# Cross-sectional rank-IC
# --------------------------------------------------------------------------

def _spearman_rows(sig: np.ndarray, fwd: np.ndarray, min_names: int = 30
                   ) -> np.ndarray:
    """Per-row Spearman rank correlation between signal[t] and fwd[t].

    Ranks each row, then Pearson-correlates the ranks (NaN-aware). Returns a
    (T,) array of daily ICs (NaN where < min_names valid pairs)."""
    T, N = sig.shape
    ic = np.full(T, np.nan)
    for t in range(T):
        s = sig[t]
        f = fwd[t]
        m = np.isfinite(s) & np.isfinite(f)
        if m.sum() < min_names:
            continue
        sr = pd.Series(s[m]).rank().values
        fr = pd.Series(f[m]).rank().values
        sr = sr - sr.mean()
        fr = fr - fr.mean()
        denom = np.sqrt((sr * sr).sum() * (fr * fr).sum())
        if denom > 0:
            ic[t] = float((sr * fr).sum() / denom)
    return ic


def forward_returns(closes: pd.DataFrame, horizon: int) -> np.ndarray:
    """(T,N) horizon-day forward simple return: close[t+h]/close[t]-1."""
    C = closes.values
    T, N = C.shape
    fwd = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd[:T - horizon] = C[horizon:] / C[:T - horizon] - 1.0
    return fwd


def _newey_west_se(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett-kernel) standard error of the MEAN of x. Corrects
    for the autocorrelation induced by overlapping forward-return windows, so
    the IC t-stat is not fake-inflated by overlap."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float((xc * xc).mean())
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = float((xc[k:] * xc[:-k]).mean())
        var += 2.0 * w * cov
    var = max(var, 1e-12)
    return float(np.sqrt(var / n))


def measure_ic(name: str, signal: np.ndarray, closes: pd.DataFrame,
               horizons=(10, 21, 42), sample_every: int = 5) -> dict:
    """Report mean rank-IC, IC stdev, IC t-stat, and hit-rate at several
    forward horizons. We sample every `sample_every` trading days (the signal
    only changes bi-monthly, so daily overlap is near-perfectly autocorrelated)
    AND use a Newey-West SE with lags = ceil(h/sample_every) so the remaining
    overlap in the h-day forward windows does not fake-inflate the t-stat."""
    out = {}
    for h in horizons:
        fwd = forward_returns(closes, h)
        ic = _spearman_rows(signal, fwd)
        ic_s = ic[::sample_every]
        ic_s = ic_s[np.isfinite(ic_s)]
        if ic_s.size < 10:
            out[h] = None
            continue
        mean = float(ic_s.mean())
        sd = float(ic_s.std(ddof=1))
        nw_lags = int(np.ceil(h / sample_every))
        se = _newey_west_se(ic_s, nw_lags)
        tstat = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
        out[h] = {
            "mean_ic": mean, "ic_std": sd, "t_stat": tstat,
            "hit_rate": float((ic_s > 0).mean()), "n": int(ic_s.size),
        }
    return out


# --------------------------------------------------------------------------
# Long/short construction + report card (look-ahead-safe)
# --------------------------------------------------------------------------

def xs_backtest_weights(signal: np.ndarray, closes: np.ndarray,
                        *, top_q: float = 0.2, long_short: bool = True,
                        rebalance: int = 5, min_names: int = 30,
                        lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (net_gross_returns (T,), weights (T,N)). Weights from close t
    earn the t+lag-1 -> t+lag return (lag>=1 => no look-ahead). Gross only
    (costs applied separately via the per-name cost stack)."""
    T, N = closes.shape
    daily_ret = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        daily_ret[1:] = closes[1:] / closes[:-1] - 1.0
    W = np.zeros((T, N))
    last = np.zeros(N)
    for t in range(T):
        if t % rebalance == 0:
            s = signal[t]
            valid = np.isfinite(s) & np.isfinite(closes[t])
            if valid.sum() >= min_names:
                sv = s[valid]
                order = np.argsort(sv)
                idx = np.where(valid)[0]
                k = max(1, int(np.floor(top_q * len(sv))))
                w = np.zeros(N)
                w[idx[order[-k:]]] = 1.0 / k
                if long_short:
                    w[idx[order[:k]]] = -1.0 / k
                last = w
        W[t] = last
    w_prev = np.vstack([np.zeros((lag, N)), W[:-lag]])
    contrib = np.where(np.isfinite(daily_ret), w_prev * daily_ret, 0.0)
    gross = contrib.sum(axis=1)
    return gross, W


def score_gate(name: str, sig_base: np.ndarray, closes: pd.DataFrame,
               vols: pd.DataFrame, *, top_q: float = 0.2, rebalance: int = 5,
               long_short: bool = True,
               param_sweep: list[tuple[float, int]] | None = None,
               n_trials_extra: int = 0) -> dict:
    """Backtest the base config, build a param-sweep grid for PBO, cost-adjust
    with the per-name stack, and run the hardened report card.

    n_trials = len(param_sweep) + n_trials_extra so the Deflated Sharpe is
    charged for EVERY variant/horizon/direction we probed in this study.
    """
    C = closes.values
    V = vols.values
    gross, W = xs_backtest_weights(sig_base, C, top_q=top_q,
                                   rebalance=rebalance, long_short=long_short)
    cost_model = CostModel()
    per_name_bps = cost_model.per_name_cost_bps(C, V)
    net = cost_adjust_returns(gross, W, per_name_bps, gross_book=1.0)

    param_sweep = param_sweep or [(top_q, rebalance)]
    grids = []
    for (q, rb) in param_sweep:
        g, _ = xs_backtest_weights(sig_base, C, top_q=q, rebalance=rb,
                                   long_short=long_short)
        grids.append(g)
    Tmin = min(len(g) for g in grids)
    grid = np.column_stack([g[-Tmin:] for g in grids])

    nz = np.flatnonzero(gross != 0.0)
    start = nz[0] if nz.size else 0
    n_trials = max(1, len(param_sweep) + n_trials_extra)
    rc = build_report_card(
        strategy_name=name,
        returns=gross[start:],
        n_trials=n_trials,
        trial_grid=grid[start:],
        cost_adjusted_returns=net[start:],
        periods_per_year=252,
    )
    ann = float(np.mean(gross[start:]) * 252)
    vol = float(np.std(gross[start:]) * np.sqrt(252))
    ann_net = float(np.mean(net[start:]) * 252)
    return {
        "name": name, "ann": ann, "vol": vol,
        "sharpe": ann / (vol + 1e-9), "ann_net": ann_net,
        "n_obs": int(gross[start:].size), "status": rc.status, "card": rc,
    }


# --------------------------------------------------------------------------
# Signal builders (each returns a (T,N) np array on the price grid)
# --------------------------------------------------------------------------

def sig_high_si(panels: SIPanels) -> np.ndarray:
    """(a) overvaluation: SHORT high-SI -> signal = -SIR (long low-SI)."""
    return -panels.sir_adv.values


def sig_rising_si(panels: SIPanels) -> np.ndarray:
    """change: SHORT rising-SI -> signal = -dSIR (fresh short-interest build)."""
    return -panels.d_sir.values


def sig_squeeze(panels: SIPanels, closes: pd.DataFrame,
                mom_lookback: int = 21) -> np.ndarray:
    """(b) squeeze: LONG high-SI names that are ALSO rising in price.
    signal = SIR * 1{recent momentum > cross-sectional median}. Heavily-shorted
    names with positive momentum are squeeze candidates -> positive expected."""
    C = closes.values
    T, N = C.shape
    mom = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom[mom_lookback:] = C[mom_lookback:] / C[:T - mom_lookback] - 1.0
    sir = panels.sir_adv.values
    out = np.full((T, N), np.nan)
    for t in range(T):
        s = sir[t]
        m = mom[t]
        valid = np.isfinite(s) & np.isfinite(m)
        if valid.sum() < 10:
            continue
        med = np.nanmedian(m[valid])
        row = np.full(N, np.nan)
        row[valid] = np.where(m[valid] > med, s[valid], np.nan)
        out[t] = row
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_research(verbose: bool = True) -> int:
    print("=== FINRA short-interest cross-sectional signal ===\n")
    print("SURVIVORSHIP CONTEXT:")
    print(" ", survivorship_report(try_network=False))
    print()

    si = load_cache()
    if si.empty:
        print("NO CACHE at", CACHE_FILE)
        print("Run once with:  python scripts/sig_short_interest.py --fetch "
              "--start 2018-01-01 --end 2026-05-01")
        print("(FINRA is NON-SEC and reachable — fetch is allowed in this "
              "workflow.)")
        return 2

    print(f"[cache] {len(si)} short-interest rows, "
          f"{si['symbolCode'].nunique()} symbols, "
          f"{si['settlementDate'].min().date()} -> "
          f"{si['settlementDate'].max().date()}")

    closes, vols = load_price_panel()
    symbols = list(closes.columns)
    print(f"[price] {closes.shape[0]} days x {len(symbols)} R3000 names")

    panels = build_si_panels(si, closes.index, symbols)
    cov = np.isfinite(panels.sir_adv.values).mean()
    print(f"[panel] SIR coverage on price grid: {cov*100:.1f}% of cells "
          f"(median active-report age "
          f"{np.nanmedian(panels.asof_age.values):.0f} trading days)\n")

    # ---- IC: the headline, robust read ----
    # IC is reported on the RAW factor (SIR, dSIR, SIR-of-winners) so the SIGN
    # maps directly onto the hypotheses, with no negation to misread:
    #   raw SIR  IC < 0  => high short-interest -> NEGATIVE returns  => HYP (a)
    #   raw dSIR IC < 0  => rising short-interest -> NEGATIVE returns => change
    #   winners' SIR IC > 0 => heavily-shorted winners outperform   => HYP (b)
    print("=== rank-IC on RAW factor (publication-lagged, sampled every 5 td) ===")
    print("   sign key: SIR<0 => HYP(a) overvaluation; winnersSIR>0 => HYP(b) squeeze")
    ic_results = {}
    for nm, sig in (
        ("raw SIR        [HYP a: expect IC<0]", panels.sir_adv.values),
        ("raw dSIR       [change: expect IC<0]", panels.d_sir.values),
        ("winners' SIR   [HYP b: expect IC>0]",
         sig_squeeze(panels, closes)),
    ):
        res = measure_ic(nm, sig, closes)
        ic_results[nm] = res
        print(f"\n{nm}")
        for h, r in res.items():
            if r is None:
                print(f"  h={h:>2}d:  insufficient")
                continue
            print(f"  h={h:>2}d:  mean_IC={r['mean_ic']:+.4f}  "
                  f"t={r['t_stat']:+.2f}  hit={r['hit_rate']*100:.0f}%  "
                  f"n={r['n']}")

    # ---- gate: long/short on the high-SI overvaluation direction ----
    print("\n=== hardened report card (long/short, per-name cost stack) ===")
    # n_trials honesty: 3 signals x 3 horizons = 9 IC looks already spent,
    # plus the 6-cell param sweep below. DSR is charged for all of them.
    base_high = sig_high_si(panels)
    param_sweep = [(q, rb) for q in (0.1, 0.2, 0.3) for rb in (5, 10)]
    res = score_gate(
        "si_high_long_short", base_high, closes, vols,
        top_q=0.2, rebalance=5, param_sweep=param_sweep, n_trials_extra=9)

    print(f"\n### si_high_long_short  ann={res['ann']*100:.1f}%  "
          f"vol={res['vol']*100:.1f}%  Sharpe={res['sharpe']:.2f}  "
          f"ann_net(after cost)={res['ann_net']*100:.1f}%  n_obs={res['n_obs']}")
    print(res["card"].render())
    print(f"STATUS: {res['status']}")

    out = Path("validation_reports")
    out.mkdir(exist_ok=True)
    (out / "si_high_long_short.md").write_text(res["card"].render())

    print("\n=== verdict ===")
    print("SHORT LEG IS SURVIVORSHIP-INFLATED (heavily-shorted zeros are "
          "absent) -> treat the long/short Sharpe as an OPTIMISTIC UPPER "
          "BOUND. The rank-IC is the honest, tail-robust read; use it as the "
          "weak-orthogonal feature for the LightGBM combiner, NOT as a "
          "standalone strategy.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="crawl FINRA (NON-SEC) and write the cache")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-05-01")
    ap.add_argument("--smoke", action="store_true",
                    help="fetch a single settlement date as a reachability test")
    args = ap.parse_args()

    if args.smoke:
        panel_syms = {os.path.basename(f).replace(".parquet", "")
                      for f in glob.glob(os.path.join(PRICE_DIR, "*.parquet"))}
        dates = list_settlement_dates("2024-06-01", "2024-06-30")
        print("settlement dates in 2024-06:", dates)
        if dates:
            recs = fetch_settlement_date(dates[-1], panel_syms)
            print(f"{dates[-1]}: {len(recs)} R3000 names")
            for r in recs[:5]:
                print(" ", r["symbolCode"], r["settlementDate"],
                      "DTC=", r.get("daysToCoverQuantity"),
                      "shortQty=", r.get("currentShortPositionQuantity"))
        return 0

    if args.fetch:
        panel_syms = {os.path.basename(f).replace(".parquet", "")
                      for f in glob.glob(os.path.join(PRICE_DIR, "*.parquet"))}
        df = fetch_panel(args.start, args.end, panel_syms)
        if df.empty:
            print("fetch returned no rows")
            return 1
        # merge with any existing cache (idempotent on symbol+date)
        old = load_cache()
        if not old.empty:
            df = (pd.concat([old, df])
                  .drop_duplicates(["symbolCode", "settlementDate"], keep="last")
                  .reset_index(drop=True))
        save_cache(df)
        return 0

    return run_research()


if __name__ == "__main__":
    raise SystemExit(main())

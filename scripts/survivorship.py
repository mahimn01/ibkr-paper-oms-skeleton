"""Quantify and bound the survivorship bias in data/atlas_r3000.

THE PROBLEM (measured, not assumed)
-----------------------------------
Every one of the ~2060 names in data/atlas_r3000 has its last bar on exactly
2026-04-06. That is not a coincidence: it is a *current-membership snapshot* of
the Russell 3000 with the full history of each surviving name backfilled, and
ZERO retention of names that delisted, went bankrupt, were acquired at a loss,
or fell out of the index before 2026. `share_survivors()` confirms this is
~100%.

Why this inflates any long/short small-cap number:
  * The losers that would have dragged the LONG leg down (names that went to
    zero, got delisted for price, or were taken under) are simply absent. The
    surviving small-caps you DO see are conditioned on having survived 10-14
    years — a powerful positive selection.
  * The SHORT leg is the opposite and is the more dangerous illusion: the
    biggest short winners (names that actually went to zero) are missing, so a
    backtested short leg looks far safer and more profitable than it was. A
    short-small-cap Sharpe on this panel is essentially uninterpretable.

MAGNITUDE (from the literature)
-------------------------------
There is no free CRSP-quality delisting-return feed reachable here, so we bound
the bias with documented estimates rather than pretend to fix it:

  * Shumway (1997, J. Finance): the average CRSP delisting return for
    performance-related (bankruptcy/price) delistings is about -30%, and
    omitting it materially biases small-stock returns upward.
  * Shumway & Warther (1999): NASDAQ delisting bias is concentrated in the
    smallest names and can overstate small-cap returns by ~1-3%/yr.
  * Survivorship in fixed back-filled universes (Brown-Goetzmann-Ibbotson-Ross
    1992; Elton-Gruber-Blake 1996 on funds) commonly inflates annual returns
    by ~1-4%/yr for small / high-attrition segments.

We therefore expose a CONSERVATIVE haircut that subtracts a per-year drag from
the LONG-leg returns (default 2%/yr, configurable to the 1-3%/yr band) and,
for long/short, refuses to certify the short leg — bounding it instead.

WHAT A REAL FIX REQUIRES (stated honestly)
------------------------------------------
A correct fix needs (a) point-in-time index membership (who was actually in the
Russell 3000 on each rebalance date) and (b) CRSP delisting returns appended to
each name's terminal bar (the realized return through the delist, often -100%
for bankruptcies). Neither is in this repo or freely reachable; SEC
company_tickers is itself a current-registrant list (survivors only) and cannot
reconstruct the historical membership of names that have since deregistered.
Until that data exists, treat every small-cap long/short figure on this panel
as an OPTIMISTIC UPPER BOUND, and report the haircut-applied long-leg number as
the honest read.

Public API
----------
    last_dates(data_dir)                 -> {symbol: pd.Timestamp}
    share_survivors(...)                 -> (share, n_survivors, n_total)
    try_source_delisted(...)             -> (list[str] | None, provenance str)
    survivorship_report(...)             -> human-readable string summary
    apply_long_leg_haircut(returns, ...) -> haircut-adjusted long-leg returns
    SurvivorshipBound                    -> dataclass bundling the verdict
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np


DATA_DIR = "data/atlas_r3000"

# Documented haircut band (annualised, fraction). See module docstring refs.
HAIRCUT_LO = 0.01
HAIRCUT_DEFAULT = 0.02
HAIRCUT_HI = 0.03


# --------------------------------------------------------------------------
# Measure the bias
# --------------------------------------------------------------------------

def last_dates(data_dir: str = DATA_DIR) -> dict[str, "object"]:
    """Map each symbol to its last bar date (pandas Timestamp)."""
    import pandas as pd  # local import; pandas is heavy
    out: dict[str, object] = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
        sym = os.path.basename(f).replace(".parquet", "")
        try:
            idx = pd.read_parquet(f, columns=["close"]).index
        except Exception:
            continue
        if len(idx):
            out[sym] = idx[-1]
    return out


def share_survivors(data_dir: str = DATA_DIR,
                    survivor_year: int = 2026) -> tuple[float, int, int]:
    """Return (share, n_survivors, n_total) where a survivor is a name whose
    last bar falls in `survivor_year` (the panel's final year)."""
    ld = last_dates(data_dir)
    if not ld:
        return (float("nan"), 0, 0)
    n_total = len(ld)
    n_surv = sum(1 for d in ld.values() if getattr(d, "year", None) == survivor_year)
    return (n_surv / n_total, n_surv, n_total)


def attrition_implied(years: float, annual_delist_rate: float = 0.06) -> float:
    """Fraction of an initial cohort that would be EXPECTED to leave the index
    over `years` at a typical small-cap annual delisting/attrition rate.

    Russell reconstitution + M&A + bankruptcy churn runs ~5-8%/yr for the
    small-cap segment; at 6%/yr over ~14 years that is 1-(1-0.06)^14 ~= 58% of
    an initial cohort gone. A panel that retains 0% of those names is missing a
    majority of the names that ever existed in the segment — this function makes
    that magnitude explicit."""
    return 1.0 - (1.0 - annual_delist_rate) ** years


# --------------------------------------------------------------------------
# Try to source a delisted/bankrupt list (best-effort, may be unreachable)
# --------------------------------------------------------------------------

def try_source_delisted(timeout: float = 15.0) -> tuple[list[str] | None, str]:
    """Attempt to build a list of tickers that are in our panel but are NOT in
    SEC's current company_tickers (a weak proxy for 'no longer a live US
    registrant'). Returns (tickers_or_None, provenance).

    HONEST CAVEAT: SEC company_tickers is a SURVIVOR list (current registrants
    only). Names IN our panel that are MISSING from it are plausibly delisted/
    renamed/acquired — but our panel is itself all-survivors, so this set is
    expected to be tiny and is NOT a substitute for CRSP delisting data. We use
    it only to demonstrate the *direction* of the gap, never to repair returns.
    """
    import json
    import urllib.request

    url = "https://www.sec.gov/files/company_tickers.json"
    req = urllib.request.Request(
        url, headers={"User-Agent": "research mahimn.patel.k@gmail.com"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception as exc:  # network blocked / rate-limited
        return None, f"SEC unreachable ({exc!r}); fell back to literature haircut"

    live = {v["ticker"].upper() for v in data.values()}
    panel = {os.path.basename(f).replace(".parquet", "").upper()
             for f in glob.glob(os.path.join(DATA_DIR, "*.parquet"))}
    missing = sorted(panel - live)
    prov = (f"SEC company_tickers reachable: {len(live)} live registrants; "
            f"{len(missing)}/{len(panel)} panel names absent from the live list "
            f"(weak delisting proxy — panel is all-survivors so this is small "
            f"and is NOT a CRSP substitute)")
    return missing, prov


# --------------------------------------------------------------------------
# Apply a conservative haircut to a backtest's long-leg returns
# --------------------------------------------------------------------------

def apply_long_leg_haircut(returns: np.ndarray,
                           annual_haircut: float = HAIRCUT_DEFAULT,
                           periods_per_year: int = 252) -> np.ndarray:
    """Subtract a constant per-period drag from a LONG-leg return stream to
    bound survivorship inflation.

    drag_per_period = annual_haircut / periods_per_year, applied additively to
    every period (a flat, always-against-you haircut — deliberately blunt and
    conservative). Use this on the LONG-ONLY leg; do NOT use it to rescue a
    long/short number whose short leg is the real problem.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    drag = annual_haircut / periods_per_year
    return r - drag


@dataclass
class SurvivorshipBound:
    share_survivors: float
    n_survivors: int
    n_total: int
    years_span: float
    implied_attrition: float
    haircut_lo: float
    haircut_default: float
    haircut_hi: float
    delisted_proxy_n: int | None
    provenance: str

    def verdict(self) -> str:
        return (
            f"R3000 panel is {self.share_survivors*100:.1f}% survivors "
            f"({self.n_survivors}/{self.n_total} end in the final year). "
            f"Over ~{self.years_span:.0f}y a typical 6%/yr small-cap attrition "
            f"implies ~{self.implied_attrition*100:.0f}% of an initial cohort "
            f"should have exited — this panel retains ~0% of them. "
            f"=> ANY small-cap long/short Sharpe here is an OPTIMISTIC UPPER "
            f"BOUND. Honest read: take the LONG leg, subtract a "
            f"{self.haircut_lo*100:.0f}-{self.haircut_hi*100:.0f}%/yr haircut "
            f"(default {self.haircut_default*100:.0f}%/yr), and BOUND (do not "
            f"certify) the short leg. Real fix needs PIT membership + CRSP "
            f"delisting returns. Delisting proxy: {self.provenance}"
        )


def build_bound(data_dir: str = DATA_DIR, years_span: float = 14.0,
                try_network: bool = True) -> SurvivorshipBound:
    share, n_surv, n_total = share_survivors(data_dir)
    proxy_n: int | None = None
    prov = "network skipped"
    if try_network:
        missing, prov = try_source_delisted()
        proxy_n = None if missing is None else len(missing)
    return SurvivorshipBound(
        share_survivors=share,
        n_survivors=n_surv,
        n_total=n_total,
        years_span=years_span,
        implied_attrition=attrition_implied(years_span),
        haircut_lo=HAIRCUT_LO,
        haircut_default=HAIRCUT_DEFAULT,
        haircut_hi=HAIRCUT_HI,
        delisted_proxy_n=proxy_n,
        provenance=prov,
    )


def survivorship_report(try_network: bool = True) -> str:
    b = build_bound(try_network=try_network)
    return b.verdict()


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _selftest() -> int:
    print("=== survivorship.py self-test ===")
    share, n_surv, n_total = share_survivors()
    print(f"survivors: {n_surv}/{n_total} = {share*100:.2f}%")
    assert n_total > 1500, "expected ~2060 R3000 names"
    assert share > 0.98, "expected ~100% survivors (current-membership snapshot)"

    att = attrition_implied(14.0)
    print(f"implied 14y attrition @6%/yr: {att*100:.1f}% of an initial cohort")
    assert 0.4 < att < 0.7

    # haircut sanity: 2%/yr should drop a 10%/yr long leg to ~8%/yr
    rng = np.random.default_rng(0)
    r = rng.normal(0.10 / 252, 0.01, size=252 * 5)
    hc = apply_long_leg_haircut(r, 0.02)
    drop = (r.mean() - hc.mean()) * 252
    print(f"haircut check: long-leg annual return reduced by {drop*100:.2f}%/yr "
          f"(target 2.00%)")
    assert abs(drop - 0.02) < 1e-6

    print("\n--- network delisting-proxy attempt ---")
    missing, prov = try_source_delisted()
    print(prov)
    if missing is not None:
        print(f"sample panel-names absent from SEC live list: {missing[:15]}")

    print("\n--- verdict ---")
    print(survivorship_report(try_network=False))
    print("\nSELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())

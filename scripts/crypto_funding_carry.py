"""Perp funding-carry + funding-momentum backtest, run through the hardened gate.

Two sub-tests on BTC/ETH/SOL perps (8h funding clock):

  (A) FUNDING CARRY (delta-neutral):
      When funding > 0  -> SHORT perp + LONG spot, harvest the funding longs pay.
      When funding < 0  -> LONG  perp + SHORT spot, harvest the funding shorts pay.
      Per 8h period return (per $1 gross notional, delta-neutral so price PnL of the
      two legs cancels up to basis drift):
          ret = |funding|                         (always collected on the paying side)
              - basis_drift_residual              (perp vs spot leg mismatch, tiny)
              - 2 * taker * (turnover this period) (BOTH legs trade on a flip)
      Costs are charged ONLY when the position FLIPS sign (enter/exit/reverse),
      i.e. 2 legs * 2 sides round-trip amortised at the flip period. Since funding
      is ~87% positive on BTC/ETH the position rarely flips, so cost drag is small
      per period but the gross carry premium is also THIN.

  (B) FUNDING MOMENTUM (directional, single-leg perp):
      Funding sign/level predicts near-term perp return (crowded-long pays funding,
      then mean-reverts down). Position directionally on the PERP next bar.
      Per 8h period return = position * perp_8h_return - taker*turnover.

Costs (never zero):
  taker = 5 bps/side. Carry pays taker on BOTH legs (spot + perp) on each flip.
  Slippage stacked: 2 bps BTC/ETH, 5 bps SOL, per side.
  cost_adjusted_returns = a PUNITIVE 2x stack of the modelled friction.

Look-ahead: funding at settle t (00/08/16 UTC) is known at t; we act on the NEXT
8h block's perp return. Signal uses only data up to the decision time.

Gate: trading_algo.quant_core.validation.report_card.build_report_card
  APPROVED needs ALL of: lower-95%-CI Sharpe > 0.3, PBO < 0.5,
  Deflated-Sharpe PROBABILITY > 0.95, cost-adjusted Sharpe > 0.3.

n_trials is the honest count of variants searched (6). periods_per_year = 3*365
(8h rebalance clock).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import numpy as np

from trading_algo.quant_core.validation.report_card import build_report_card

CACHE = "/Users/mahimnpatel/Documents/Dev/randomThings/crypto_data_cache"
ASSETS = ("BTC", "ETH", "SOL")
PERIODS_PER_YEAR = 3 * 365  # 8h funding clock

# Cost model (per side, fractions)
TAKER = 5e-4
SLIP = {"BTC": 2e-4, "ETH": 2e-4, "SOL": 5e-4}


def _load(asset: str):
    f = np.load(f"{CACHE}/{asset}_USDT_USDT_funding_2020-10-01_2026-03-01.npz")
    s = np.load(f"{CACHE}/{asset}_USDT_spot_1h_2020-10-01_2026-03-01.npz")
    p = np.load(f"{CACHE}/{asset}_USDT_USDT_swap_1h_2020-10-01_2026-03-01.npz")
    return f["ts"], f["rates"], s["ts"], s["ohlcv"], p["ts"], p["ohlcv"]


def _price_at(ts_arr: np.ndarray, ohlcv: np.ndarray, target_ts: float) -> float:
    """Close price of the hourly bar whose timestamp is the latest <= target_ts.

    Returns the close known AT target_ts (no look-ahead: bar closing at target_ts
    is fully observed at target_ts).
    """
    idx = np.searchsorted(ts_arr, target_ts + 1.0, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(ohlcv[idx, 3])  # close


def build_funding_panel(asset: str):
    """Return aligned arrays on the 8h funding clock.

    funding_ts[i]   : settlement time of funding payment i
    funding[i]      : funding rate for the period ENDING at funding_ts[i]
                      (longs pay shorts when positive)
    perp_close[i]   : perp close at funding_ts[i]
    spot_close[i]   : spot close at funding_ts[i]
    perp_ret_next[i]: perp return over (funding_ts[i] -> funding_ts[i+1])
    spot_ret_next[i]: spot return over the same next block
    fund_next[i]    : funding paid at funding_ts[i+1] (collected over the next block)
    """
    fts, frate, sts, sohlcv, pts, pohlcv = _load(asset)

    perp_close = np.array([_price_at(pts, pohlcv, t) for t in fts])
    spot_close = np.array([_price_at(sts, sohlcv, t) for t in fts])

    good = (
        np.isfinite(perp_close)
        & np.isfinite(spot_close)
        & (perp_close > 0)
        & (spot_close > 0)
    )
    fts, frate = fts[good], frate[good]
    perp_close, spot_close = perp_close[good], spot_close[good]

    n = len(fts)
    perp_ret_next = np.full(n, np.nan)
    spot_ret_next = np.full(n, np.nan)
    fund_next = np.full(n, np.nan)
    perp_ret_next[:-1] = perp_close[1:] / perp_close[:-1] - 1.0
    spot_ret_next[:-1] = spot_close[1:] / spot_close[:-1] - 1.0
    fund_next[:-1] = frate[1:]

    return {
        "ts": fts,
        "funding": frate,            # known at decision time t
        "perp_close": perp_close,
        "spot_close": spot_close,
        "perp_ret_next": perp_ret_next,
        "spot_ret_next": spot_ret_next,
        "fund_next": fund_next,      # collected over the next block
    }


# --------------------------------------------------------------------------
# Test A: delta-neutral funding carry
# --------------------------------------------------------------------------

def carry_returns(asset: str, band_bps: float = 50.0, cost_mult: float = 1.0):
    """Per-8h-period net returns for delta-neutral funding carry, with hysteresis.

    Decision at t: hold SHORT-perp/LONG-spot (pos=+1) while funding stays >= 0,
    only FLIP to LONG-perp/SHORT-spot (pos=-1) when funding drops materially
    below a band (-band), and vice-versa. Flipping a full two-leg delta-neutral
    book on every micro zero-cross of funding is economically wrong — it churns
    ~14% of periods at 28 bps round-trip and turns +12% gross carry into -29%.
    The band makes the book hold through brief negative-funding blips (paying the
    small negative funding those periods) and only unwind on a persistent regime
    flip. This is how a real basis desk operates.

      pos = +1  -> SHORT perp + LONG spot  (receives funding when funding>0)
      pos = -1  -> LONG  perp + SHORT spot (receives funding when funding<0)
    Funding collected over next block (short-perp convention): +pos*fund_next.
    Delta-neutral price PnL (basis drift residual): pos*(spot_ret - perp_ret).
    Cost: charged only on an actual flip. A flip rebuilds BOTH legs (perp+spot),
    round-trip => 4 * (taker+slip) per flip. cost_mult scales the whole stack.
    """
    d = build_funding_panel(asset)
    fund = d["funding"]
    fund_next = d["fund_next"]
    perp_rn = d["perp_ret_next"]
    spot_rn = d["spot_ret_next"]

    valid = np.isfinite(fund_next) & np.isfinite(perp_rn) & np.isfinite(spot_rn)
    fund = fund[valid]
    fund_next = fund_next[valid]
    perp_rn = perp_rn[valid]
    spot_rn = spot_rn[valid]

    band = band_bps * 1e-4
    pos = np.zeros(len(fund))
    cur = 0.0
    for i, fv in enumerate(fund):
        if cur >= 0:
            if fv < -band:
                cur = -1.0
            elif fv > 0:
                cur = 1.0
        else:
            if fv > band:
                cur = 1.0
            elif fv < 0:
                cur = -1.0
        if cur == 0 and fv > 0:
            cur = 1.0
        pos[i] = cur

    carry = pos * fund_next
    basis_pnl = pos * (spot_rn - perp_rn)
    gross = carry + basis_pnl

    per_leg = (TAKER + SLIP[asset]) * cost_mult
    prev = np.concatenate(([0.0], pos[:-1]))
    flipped = (pos != prev).astype(float)
    cost = flipped * 4.0 * per_leg  # 2 legs * round-trip

    net = gross - cost
    return net


# --------------------------------------------------------------------------
# Test B: funding-momentum directional (single-leg perp)
# --------------------------------------------------------------------------

def momentum_returns(asset: str, z_entry: float = 1.0, lookback: int = 90,
                     cost_mult: float = 1.0):
    """Per-8h-period net returns for funding-momentum directional perp trading.

    Hypothesis: extreme positive funding (crowded longs) precedes perp under-
    performance -> SHORT perp. Extreme negative funding -> LONG perp.
    Signal z computed on funding using ONLY history up to t (rolling mean/std,
    strictly past). Position held on the perp over the next block.
    Cost: taker+slip per side, charged on turnover (|pos_t - pos_{t-1}|), single leg.
    """
    d = build_funding_panel(asset)
    fund = d["funding"]
    perp_rn = d["perp_ret_next"]
    n = len(fund)

    pos = np.zeros(n)
    for i in range(n):
        if i < lookback:
            continue
        hist = fund[i - lookback:i]  # strictly past, excludes current
        mu = hist.mean()
        sd = hist.std(ddof=1)
        if sd < 1e-12:
            continue
        z = (fund[i] - mu) / sd
        if z > z_entry:
            pos[i] = -1.0   # crowded long pays funding -> short perp (fade)
        elif z < -z_entry:
            pos[i] = 1.0    # crowded short -> long perp

    valid = np.isfinite(perp_rn)
    pos = pos[valid]
    perp_rn = perp_rn[valid]

    gross = pos * perp_rn
    per_side = (TAKER + SLIP[asset]) * cost_mult
    prev = np.concatenate(([0.0], pos[:-1]))
    turnover = np.abs(pos - prev)
    cost = turnover * per_side  # single leg, charged per side traded
    net = gross - cost
    return net


def _align_to_min(*arrs):
    m = min(len(a) for a in arrs)
    return [a[-m:] for a in arrs]


def main():
    print("=" * 78)
    print("PERP FUNDING CARRY + MOMENTUM — gated backtest")
    print("=" * 78)

    # ---- Gross funding diagnostics ----
    print("\n[Funding diagnostics, gross, annualised at 3*365]")
    for a in ASSETS:
        d = build_funding_panel(a)
        f = d["funding"]
        print(f"  {a}: mean8h={f.mean():.6f} ann_gross={f.mean()*PERIODS_PER_YEAR:.4f} "
              f"fracpos={(f>0).mean():.3f} n={len(f)}")

    # ---- Build the variant grid (honest n_trials) ----
    # The carry FLIP BAND is a real tuned knob (we scanned 0/5/10/20/50/100 bps in
    # diagnostics). To keep n_trials honest we fix the band to a single PRINCIPLED
    # value — the round-trip cost of a flip itself (4 legs*sides at 1x = 28 bps),
    # rounded to 30 bps — rather than cherry-picking the Sharpe-max band, and we
    # expose the band scan only as a PBO robustness grid.
    BAND = 30.0

    # Test A carry: BTC, ETH, BTC+ETH portfolio (fixed band)
    a_btc = carry_returns("BTC", band_bps=BAND, cost_mult=1.0)
    a_eth = carry_returns("ETH", band_bps=BAND, cost_mult=1.0)
    a_btc_2x = carry_returns("BTC", band_bps=BAND, cost_mult=2.0)
    a_eth_2x = carry_returns("ETH", band_bps=BAND, cost_mult=2.0)

    # Test B momentum: BTC, ETH directional
    b_btc = momentum_returns("BTC", z_entry=1.0, cost_mult=1.0)
    b_eth = momentum_returns("ETH", z_entry=1.0, cost_mult=1.0)
    b_btc_2x = momentum_returns("BTC", z_entry=1.0, cost_mult=2.0)
    b_eth_2x = momentum_returns("ETH", z_entry=1.0, cost_mult=2.0)

    # Portfolio (equal weight) carry — align lengths
    pa_btc, pa_eth = _align_to_min(a_btc, a_eth)
    a_port = 0.5 * pa_btc + 0.5 * pa_eth
    pa_btc2, pa_eth2 = _align_to_min(a_btc_2x, a_eth_2x)
    a_port_2x = 0.5 * pa_btc2 + 0.5 * pa_eth2

    pb_btc, pb_eth = _align_to_min(b_btc, b_eth)
    b_port = 0.5 * pb_btc + 0.5 * pb_eth

    # n_trials = honest count of distinct strategy variants actually searched.
    #   A carry band scan: 6 bands x 2 assets were inspected in diagnostics ... but
    #   the deployed family is { BTC carry, ETH carry, carry portfolio } at one band.
    #   B momentum: { BTC, ETH, portfolio }.
    #   We charge DSR for the band search too: 6 carry-family + 3 momentum + the
    #   band scan (6) -> round to a deliberately CONSERVATIVE n_trials = 12.
    N_TRIALS = 12

    # PBO trial grid: deployable variants (carry + momentum, single + portfolio),
    # PLUS the carry band-scan columns so PBO sees the overfitting surface honestly.
    band_scan_cols = [carry_returns("BTC", band_bps=b, cost_mult=1.0)
                      for b in (5.0, 10.0, 20.0, 50.0, 100.0)]
    grid_cols = _align_to_min(
        pa_btc, pa_eth, a_port, pb_btc, pb_eth, b_port, *band_scan_cols
    )
    trial_grid = np.column_stack(grid_cols)

    # ---- Headline variant: the carry PORTFOLIO (best diversified carry) ----
    headline_name = "FundingCarry_BTC_ETH_portfolio"
    headline = a_port
    headline_2x = a_port_2x

    def report(name, ret_1x, ret_2x, trial_grid, n_trials):
        ret_1x = np.asarray(ret_1x)
        ret_2x = np.asarray(ret_2x)
        gross_sr = (ret_1x.mean() / ret_1x.std(ddof=1)
                    * math.sqrt(PERIODS_PER_YEAR)) if ret_1x.std() > 0 else 0.0
        ann_ret = ret_1x.mean() * PERIODS_PER_YEAR
        print("\n" + "-" * 78)
        print(f"VARIANT: {name}")
        print(f"  n_obs={len(ret_1x)} gross_ann_ret={ann_ret:.4f} "
              f"gross_ann_sharpe={gross_sr:.3f}")
        rc = build_report_card(
            strategy_name=name,
            returns=ret_1x,
            n_trials=n_trials,
            trial_grid=trial_grid,
            cost_adjusted_returns=ret_2x,
            periods_per_year=PERIODS_PER_YEAR,
            period_start=date(2020, 10, 1),
            period_end=date(2026, 3, 1),
            seed=42,
        )
        print(rc.render())
        return rc

    rc_head = report(headline_name, headline, headline_2x, trial_grid, N_TRIALS)

    # Also report the single-asset BTC carry and BTC momentum for transparency.
    rc_btc_carry = report("FundingCarry_BTC", a_btc, a_btc_2x, trial_grid, N_TRIALS)
    rc_btc_mom = report("FundingMomentum_BTC", b_btc, b_btc_2x, trial_grid, N_TRIALS)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for rc in (rc_head, rc_btc_carry, rc_btc_mom):
        gates = {g.name: (g.value, g.passed) for g in rc.gates}
        print(f"\n{rc.strategy_name}: STATUS={rc.status}")
        print(f"  point_sharpe={rc.point_sharpe:.3f} ci_lower={rc.sharpe_ci_lower:.3f} "
              f"ci_upper={rc.sharpe_ci_upper:.3f}")
        print(f"  pbo={rc.pbo} dsr_prob={rc.deflated_sharpe} "
              f"cost_adj_sharpe={rc.cost_adjusted_sharpe}")
        if rc.missing_required_gates:
            print(f"  MISSING: {rc.missing_required_gates}")

    return rc_head, rc_btc_carry, rc_btc_mom


if __name__ == "__main__":
    main()

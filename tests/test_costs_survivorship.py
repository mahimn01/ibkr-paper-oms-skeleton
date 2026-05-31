"""Tests for the realistic cost stack (scripts/costs.py) and the R3000
survivorship bound (scripts/survivorship.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import costs, survivorship  # noqa: E402


# --------------------------------------------------------------------------
# costs.py
# --------------------------------------------------------------------------

def _toy_panel(seed: int = 0):
    """Two synthetic names: one liquid (huge $vol), one thin (tiny $vol)."""
    rng = np.random.default_rng(seed)
    T = 400
    r_liq = rng.normal(0, 0.01, T)
    r_thin = rng.normal(0, 0.03, T)
    c_liq = 100 * np.cumprod(1 + r_liq)
    c_thin = 20 * np.cumprod(1 + r_thin)
    closes = np.column_stack([c_liq, c_thin])
    # liquid: 5M shares/day; thin: 5k shares/day
    vols = np.column_stack([np.full(T, 5_000_000.0), np.full(T, 5_000.0)])
    return closes, vols


def test_amihud_monotone_liquid_vs_thin():
    closes, vols = _toy_panel()
    ill = costs.amihud_illiquidity(closes, vols, lookback=63)
    med = np.nanmedian(ill[100:], axis=0)
    # thin name must have STRICTLY larger Amihud (more illiquid)
    assert med[1] > med[0] * 10


def test_no_lookahead_in_trailing():
    # trailing windows must end at t-1: row 0 and the warmup region are NaN,
    # and the value at t must not depend on data at t.
    closes, vols = _toy_panel()
    ill = costs.amihud_illiquidity(closes, vols, lookback=20)
    assert np.all(np.isnan(ill[0]))
    # perturb the LAST row; trailing values up to T-1 must be unchanged
    c2 = closes.copy()
    c2[-1] *= 1.5
    ill2 = costs.amihud_illiquidity(c2, vols, lookback=20)
    np.testing.assert_allclose(ill[:-1], ill2[:-1], equal_nan=True)


def test_per_name_cost_calibration_band():
    # Build a wider cross-section so the percentile mapping has room to work.
    rng = np.random.default_rng(1)
    T, N = 300, 40
    closes = np.empty((T, N))
    vols = np.empty((T, N))
    for j in range(N):
        r = rng.normal(0, 0.01 + 0.001 * j, T)
        closes[:, j] = 50 * np.cumprod(1 + r)
        # dollar-volume spans 3 orders of magnitude across names
        vols[:, j] = np.full(T, 10_000.0 * (10 ** (3 * j / N)))
    model = costs.CostModel()
    cost = model.per_name_cost_bps(closes, vols)
    med = np.nanmedian(cost[150:], axis=0)
    med = med[np.isfinite(med)]
    # liquid end near the floor band, thin end near the ceiling band
    assert med.min() <= 18.0, f"liquid names should be cheap, got {med.min():.1f}"
    assert med.max() >= 25.0, f"thin names should be pricey, got {med.max():.1f}"
    assert med.max() <= 120.0


def test_cost_adjust_charges_only_on_turnover():
    T, N = 50, 3
    gross = np.full(T, 0.001)
    W = np.zeros((T, N))
    W[:] = np.array([0.5, 0.5, 0.0])  # constant weights => zero turnover after t0
    bps = np.full((T, N), 20.0)
    net = costs.cost_adjust_returns(gross, W, bps)
    # only the first day (0 -> initial weights) is charged
    assert net[0] < gross[0]
    np.testing.assert_allclose(net[1:], gross[1:], atol=1e-12)


def test_cost_adjust_nan_bps_never_free():
    T, N = 10, 2
    gross = np.zeros(T)
    W = np.zeros((T, N))
    W[1, 0] = 1.0  # a trade on day 1
    bps = np.full((T, N), np.nan)  # no cost info at all
    net = costs.cost_adjust_returns(gross, W, bps)
    # NaN bps must fall back to a positive cost, never zero
    assert net[1] < 0.0


# --------------------------------------------------------------------------
# survivorship.py
# --------------------------------------------------------------------------

def test_panel_is_all_survivors():
    share, n_surv, n_total = survivorship.share_survivors()
    assert n_total > 1500
    assert share > 0.98  # current-membership snapshot => ~100% survivors


def test_implied_attrition_reasonable():
    att = survivorship.attrition_implied(14.0, 0.06)
    assert 0.45 < att < 0.65


def test_haircut_is_exact_annual_drag():
    rng = np.random.default_rng(0)
    r = rng.normal(0.10 / 252, 0.01, 252 * 5)
    hc = survivorship.apply_long_leg_haircut(r, 0.02)
    drop = (r.mean() - hc.mean()) * 252
    assert abs(drop - 0.02) < 1e-9


def test_bound_verdict_mentions_upper_bound():
    b = survivorship.build_bound(try_network=False)
    v = b.verdict()
    assert "UPPER BOUND" in v
    assert b.share_survivors > 0.98


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

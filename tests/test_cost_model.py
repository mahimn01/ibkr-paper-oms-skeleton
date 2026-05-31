"""Tests for the realistic transaction cost model."""

from __future__ import annotations

import math
import pytest

from trading_algo.backtest_v2.cost_model import (
    BorrowTier,
    CostModelConfig,
    FillCostResult,
    IBKRTieredFees,
    abdi_ranaldo_spread,
    adjust_fill_price,
    borrow_charge,
    compute_fill_cost,
    corwin_schultz_spread,
    daily_effective_spread,
    ibkr_tiered_commission,
    recall_probability,
    sqrt_impact_bps,
)

# ----------------------------------------------------------------- Corwin-Schultz


def test_corwin_schultz_returns_positive_for_normal_bars() -> None:
    # Two adjacent bars with similar intraday ranges and modest overlap.
    # CS requires beta (sum of individual ranges) > gamma (joint range).
    s = corwin_schultz_spread(100.5, 99.5, 100.6, 99.4)
    assert s > 0
    # Ballpark: a sub-1% range pair typically yields tens of bps.
    assert s < 0.05  # < 500 bps sanity ceiling


def test_corwin_schultz_zero_when_inputs_invalid() -> None:
    # Negative or zero prices.
    assert corwin_schultz_spread(0, 0, 0, 0) == 0.0
    # Inverted high-low.
    assert corwin_schultz_spread(99.0, 100.0, 99.0, 100.0) == 0.0


def test_corwin_schultz_zero_when_high_equals_low() -> None:
    # Zero range -> zero variance -> formula returns 0.
    s = corwin_schultz_spread(100.0, 100.0, 100.0, 100.0)
    assert s == 0.0


# ----------------------------------------------------------------- Abdi-Ranaldo


def test_abdi_ranaldo_round_trip() -> None:
    closes = [100.0, 100.5, 99.8, 100.2, 100.1]
    highs  = [101.0, 101.0, 100.5, 100.8, 100.7]
    lows   = [ 99.5,  99.8,  99.5, 100.0, 100.0]
    s = abdi_ranaldo_spread(closes, highs, lows)
    assert s >= 0.0
    assert s < 0.05


def test_abdi_ranaldo_short_returns_zero() -> None:
    assert abdi_ranaldo_spread([100.0], [101.0], [99.0]) == 0.0
    assert abdi_ranaldo_spread([], [], []) == 0.0


# ----------------------------------------------------------------- daily_effective_spread


def test_daily_effective_spread_uses_cs_when_positive() -> None:
    # Two bars with similar ranges, modest overlap -> CS should yield positive.
    bars = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.6, 99.4, 100.0),
    ]
    s = daily_effective_spread(bars)
    assert s > 0


def test_daily_effective_spread_falls_back_to_ar_when_cs_fails() -> None:
    # Flat bars where high == low -> CS returns 0 -> AR fallback over the window.
    bars = [(100.0, 100.0, 100.0, 100.0)] * 25
    s = daily_effective_spread(bars)
    assert s == 0.0  # also AR collapses on flat data; both safely return 0


# ----------------------------------------------------------------- sqrt impact


def test_sqrt_impact_zero_below_min_participation() -> None:
    # 0.05% of ADV -> below 0.1% threshold.
    assert sqrt_impact_bps(quantity=500, adv=1_000_000, daily_vol_bps=200) == 0.0


def test_sqrt_impact_published_value() -> None:
    # 5% ADV at 200 bps daily vol, Y=0.5:
    # 0.5 * 200 * sqrt(0.05) = 100 * 0.2236 = 22.36 bps
    bps = sqrt_impact_bps(
        quantity=50_000, adv=1_000_000, daily_vol_bps=200.0,
    )
    assert bps == pytest.approx(0.5 * 200 * math.sqrt(0.05), rel=1e-6)


def test_sqrt_impact_caps_at_max_participation() -> None:
    # 50% ADV — capped to 10%.
    bps = sqrt_impact_bps(quantity=500_000, adv=1_000_000, daily_vol_bps=200.0)
    assert bps == pytest.approx(0.5 * 200 * math.sqrt(0.10), rel=1e-6)


# ----------------------------------------------------------------- borrow


def test_borrow_charge_act_360() -> None:
    # $100k notional at 50 bps annual for 1 day: 100000 * 0.005 / 360 ≈ $1.39
    c = borrow_charge(100_000, 50.0, days=1)
    assert c == pytest.approx(100_000 * 0.005 / 360.0, rel=1e-6)


def test_borrow_charge_zero_when_inputs_invalid() -> None:
    assert borrow_charge(0, 50, days=1) == 0.0
    assert borrow_charge(100_000, 0, days=1) == 0.0
    assert borrow_charge(100_000, 50, days=0) == 0.0


def test_recall_probability_increasing_with_rate() -> None:
    assert recall_probability(50) < recall_probability(500)
    assert recall_probability(50_000) == pytest.approx(0.05)  # capped


# ----------------------------------------------------------------- IBKR commission


def test_ibkr_tiered_floor_applies_for_tiny_orders() -> None:
    # 10 shares @ $100 -> base = 0.0035*10 = $0.035 -> floor $0.35.
    c = ibkr_tiered_commission("BUY", 10, 100.0)
    # Should hit the $0.35 minimum + small pass-through.
    assert c >= 0.35
    assert c < 1.0


def test_ibkr_tiered_per_share_for_larger_orders() -> None:
    # 1000 shares @ $50 -> base = 0.0035*1000 = $3.50.
    c = ibkr_tiered_commission("BUY", 1000, 50.0)
    # Should be roughly $3.50 + small pass-through.
    assert c == pytest.approx(3.50 + 1000 * 50.0 * 0.000175, rel=0.01)


def test_ibkr_tiered_sells_pay_sec_fee_and_taf() -> None:
    buy = ibkr_tiered_commission("BUY",  1000, 100.0)
    sell = ibkr_tiered_commission("SELL", 1000, 100.0)
    assert sell > buy   # SELLs pay SEC fee + TAF on top.


def test_ibkr_tiered_unknown_side_raises() -> None:
    with pytest.raises(ValueError):
        ibkr_tiered_commission("HOLD", 100, 100.0)


# ----------------------------------------------------------------- composite


def test_compute_fill_cost_decomposes_components() -> None:
    cost = compute_fill_cost(
        side="BUY",
        quantity=500,            # 0.05% of ADV — below sqrt-impact threshold
        fill_price=100.0,
        spread_fraction=0.0010,  # 10 bps
        adv=1_000_000,
        daily_vol_bps=200.0,
    )
    # Half-spread on $50k notional = 5 bps * $50k = $25
    assert cost.spread_cost == pytest.approx(25.0)
    # Impact at <0.1% participation -> below threshold -> 0
    assert cost.impact_cost == 0.0
    # Commission > 0
    assert cost.commission > 0
    assert cost.total > 0


def test_compute_fill_cost_disables_layers() -> None:
    cfg = CostModelConfig(
        enable_spread=False, enable_impact=False, enable_commission=False,
    )
    cost = compute_fill_cost(
        side="BUY", quantity=100, fill_price=50.0,
        spread_fraction=0.001, adv=1e6, daily_vol_bps=200.0, config=cfg,
    )
    assert cost.spread_cost == 0
    assert cost.impact_cost == 0
    assert cost.commission == 0


def test_adjust_fill_price_buys_pay_slippage() -> None:
    cost = FillCostResult(spread_cost=50.0, impact_cost=0.0, commission=10.0)
    fill = adjust_fill_price("BUY", paper_price=100.0, cost=cost, quantity=1000)
    # Per-share spread+impact = 0.05; BUY pays -> fill = 100.05
    assert fill == pytest.approx(100.05)


def test_adjust_fill_price_sells_receive_less() -> None:
    cost = FillCostResult(spread_cost=50.0, impact_cost=0.0, commission=10.0)
    fill = adjust_fill_price("SELL", paper_price=100.0, cost=cost, quantity=1000)
    assert fill == pytest.approx(99.95)


def test_borrow_tier_enum_round_trip() -> None:
    # Smoke test: enum lookups work.
    assert BorrowTier("SP500_MEGACAP") is BorrowTier.SP500_MEGACAP


def test_default_fees_initialise() -> None:
    # IBKRTieredFees default-constructs cleanly and has expected fields.
    fees = IBKRTieredFees()
    assert fees.per_share == pytest.approx(0.0035)
    assert fees.min_per_order == pytest.approx(0.35)

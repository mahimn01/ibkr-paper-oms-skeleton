"""Tests for the pre-transmit constitution gate (trading_algo/constitution.py)."""

from __future__ import annotations

import pytest

from trading_algo.constitution import (
    EvalContext,
    PositionView,
    ProposedTrade,
    account_type_for,
    evaluate,
    factor_for,
    intent_hash,
)

NL = 50_000.0


@pytest.fixture(autouse=True)
def _tfsa_env(monkeypatch):
    # Real account IDs live only in the gitignored .env; tests use a fake.
    monkeypatch.setenv("TRADING_TFSA_ACCOUNTS", "UTESTTFSA1")


def _trade(**kw) -> ProposedTrade:
    base = dict(
        symbol="IWM", kind="STK", right=None, side="BUY", quantity=10,
        structure=None, factor="index", account="U1234567", account_type="MARGIN",
        written_exit="20%-off-close trailing stop",
    )
    base.update(kw)
    return ProposedTrade(**base)


def _ctx(trade: ProposedTrade, positions=None, net_liq: float = NL) -> EvalContext:
    return EvalContext(net_liq=net_liq, trade=trade, positions=positions or [])


def _result(verdict, rule_id):
    return next(c for c in verdict.checks if c.rule_id == rule_id)


# ---------------------------------------------------------------- helpers

def test_factor_and_account_maps():
    assert factor_for("GLD") == "real-assets"
    assert factor_for("URA") == "real-assets"
    assert factor_for("XLE") == "energy"
    assert factor_for("ZZZZ") == "other"
    assert account_type_for("UTESTTFSA1") == "TFSA"
    assert account_type_for("U9999999") == "MARGIN"


def test_clean_stock_buy_passes():
    v = evaluate(_ctx(_trade(symbol="IWM")))
    assert v.decision == "PASS"
    assert not v.blocks


def test_intent_hash_stable_and_sensitive():
    a = intent_hash(_trade(symbol="IWM", quantity=10))
    b = intent_hash(_trade(symbol="IWM", quantity=10))
    c = intent_hash(_trade(symbol="IWM", quantity=11))
    assert a == b and a != c


# ---------------------------------------------------------------- BLOCK rules

def test_c1_factor_cap_blocks_when_over_50pct():
    # existing real-assets sleeve = 24k, proposed adds 5k -> 29k/50k = 58%
    positions = [PositionView("IAU", "STK", None, 600, "real-assets", 24_000.0, None, None)]
    t = _trade(symbol="URA", factor="real-assets", delta_notional=5_000.0)
    v = evaluate(_ctx(t, positions))
    r = _result(v, "C1_factor_cap")
    assert r.status == "FAIL" and v.decision == "BLOCK"


def test_c1_factor_cap_passes_under_cap():
    positions = [PositionView("IAU", "STK", None, 200, "real-assets", 8_000.0, None, None)]
    t = _trade(symbol="URA", factor="real-assets", delta_notional=5_000.0)
    assert _result(evaluate(_ctx(t, positions)), "C1_factor_cap").status == "PASS"


def test_c2_per_underlying_assignment_blocks():
    t = _trade(symbol="MP", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="real-assets", assignment_cash=9_000.0, delta=-0.30, dte=30,
               extrinsic_pct_of_credit=0.95)
    v = evaluate(_ctx(t))  # 9k/50k = 18% > 15%
    assert _result(v, "C2_per_underlying_assignment").status == "FAIL"
    assert v.decision == "BLOCK"


def test_c3_aggregate_assignment_blocks():
    positions = [PositionView("IAU", "OPT", "P", -2, "real-assets", None, 12_000.0, 30)]
    t = _trade(symbol="MP", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="real-assets", assignment_cash=5_000.0, delta=-0.30, dte=30,
               extrinsic_pct_of_credit=0.95)
    v = evaluate(_ctx(t, positions))  # 17k/50k = 34% > 30%
    assert _result(v, "C3_aggregate_assignment").status == "FAIL"


def test_c4_no_short_over_60dte_blocks():
    t = _trade(symbol="SPY", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="index", dte=75, delta=-0.30, extrinsic_pct_of_credit=0.95,
               assignment_cash=1_000.0)
    assert _result(evaluate(_ctx(t)), "C4_no_short_over_60dte").status == "FAIL"


def test_c5_gld_long_call_blocks():
    t = _trade(symbol="GLD", kind="OPT", right="C", side="BUY", structure="leap-long",
               factor="real-assets", delta=0.85, extrinsic_pct=0.10, size_notional=3_000.0,
               iv_rank=30)
    v = evaluate(_ctx(t))
    assert _result(v, "C5_no_gld_long_call").status == "FAIL"
    assert v.decision == "BLOCK"


def test_c6_tfsa_short_put_blocks():
    t = _trade(symbol="SPY", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="index", account="UTESTTFSA1", account_type="TFSA",
               delta=-0.30, dte=30, extrinsic_pct_of_credit=0.95, assignment_cash=1_000.0)
    assert _result(evaluate(_ctx(t)), "C6_tfsa_no_short_puts").status == "FAIL"


def test_c7a_leap_gate_blocks_low_delta():
    t = _trade(symbol="AMZN", kind="OPT", right="C", side="BUY", structure="leap-long",
               factor="tech", delta=0.55, extrinsic_pct=0.10, size_notional=3_000.0, iv_rank=30)
    assert _result(evaluate(_ctx(t)), "C7a_leap_gate").status == "FAIL"


def test_c7a_leap_gate_passes_clean():
    t = _trade(symbol="AMZN", kind="OPT", right="C", side="BUY", structure="leap-long",
               factor="tech", delta=0.85, extrinsic_pct=0.15, size_notional=4_000.0, iv_rank=30)
    assert _result(evaluate(_ctx(t)), "C7a_leap_gate").status == "PASS"


def test_c7b_iv_rank_warns_not_blocks():
    t = _trade(symbol="AMZN", kind="OPT", right="C", side="BUY", structure="leap-long",
               factor="tech", delta=0.85, extrinsic_pct=0.15, size_notional=4_000.0, iv_rank=70)
    v = evaluate(_ctx(t))
    assert _result(v, "C7b_leap_iv_rank").status == "FAIL"
    assert v.decision == "WARN"  # WARN does not block


def test_c8_short_put_gate_blocks_bad_delta():
    t = _trade(symbol="SPY", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="index", delta=-0.50, dte=30, extrinsic_pct_of_credit=0.95,
               assignment_cash=1_000.0)
    assert _result(evaluate(_ctx(t)), "C8_short_put_gate").status == "FAIL"


def test_c8_short_put_gate_passes_in_band():
    t = _trade(symbol="SPY", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="index", delta=-0.28, dte=35, extrinsic_pct_of_credit=0.95,
               assignment_cash=1_000.0)
    assert _result(evaluate(_ctx(t)), "C8_short_put_gate").status == "PASS"


def test_c9_instrument_gate_blocks_wide_spread():
    t = _trade(symbol="SLV", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="real-assets", is_new_program=True, spread_pct=0.05,
               strike_width_pct=0.01, weeklies_listed=True, near_money_oi=2000,
               mark_below_intrinsic=False, delta=-0.30, dte=30, extrinsic_pct_of_credit=0.95,
               assignment_cash=1_000.0)
    assert _result(evaluate(_ctx(t)), "C9_instrument_gate").status == "FAIL"


def test_w2_martingale_blocks():
    t = _trade(symbol="MP", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="real-assets", delta=-0.30, dte=30, extrinsic_pct_of_credit=0.95,
               assignment_cash=1_000.0, losing_put_close_same_underlying_30d=True)
    v = evaluate(_ctx(t))
    assert _result(v, "W2_martingale").status == "FAIL"
    assert v.decision == "BLOCK"


def test_w5_missing_written_exit_blocks():
    t = _trade(symbol="IWM", written_exit=None)
    v = evaluate(_ctx(t))
    assert _result(v, "W5_written_exit").status == "FAIL"
    assert v.decision == "BLOCK"


def test_w5_roll_skips_exit_requirement():
    t = _trade(symbol="IWM", kind="OPT", right="C", side="BUY", is_roll=True,
               structure="roll", written_exit=None)
    assert _result(evaluate(_ctx(t)), "W5_written_exit").status == "SKIP"


# ---------------------------------------------------------------- WARN rules

def test_w1_opportunistic_roll_warns():
    t = _trade(symbol="IAU", kind="OPT", right="C", side="BUY", is_roll=True,
               structure="roll", rolled_short_is_working=True, written_exit=None)
    v = evaluate(_ctx(t))
    assert _result(v, "W1_opportunistic_roll").status == "FAIL"
    assert v.decision == "WARN"


def test_w4_hedge_floor_warns_on_big_unhedged_sleeve():
    positions = [PositionView("MP", "STK", None, 500, "real-assets", 20_000.0, None, None)]
    t = _trade(symbol="IWM")
    v = evaluate(_ctx(t, positions))  # real-assets 20k/50k = 40% > 25%
    assert _result(v, "W4_hedge_floor").status == "FAIL"


# ---------------------------------------------------------------- data honesty

def test_missing_greeks_skips_not_blocks():
    # short put with no greeks at all -> C8 must SKIP, never FAIL on missing data
    t = _trade(symbol="SPY", kind="OPT", right="P", side="SELL", structure="short-put",
               factor="index", delta=None, dte=None, extrinsic_pct_of_credit=None,
               assignment_cash=None)
    v = evaluate(_ctx(t))
    assert _result(v, "C8_short_put_gate").status == "SKIP"
    # with only WARN/SKIP and no BLOCK FAIL, this short put does not get BLOCKed by missing data
    assert v.decision in {"PASS", "WARN"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

"""Tests for the constitution enrichment adapter, persistence, and config wiring."""

from __future__ import annotations

import time
from datetime import date, timedelta

import pytest

from trading_algo.broker.base import Position
from trading_algo.broker.sim import SimBroker
from trading_algo.config import EnvParseError, TradingConfig
from trading_algo.constitution import ConstitutionViolation, order_key
from trading_algo.constitution_adapter import (
    OptionGreeks,
    ProposedOrderInput,
    build_eval_context,
    gate,
    parse_dte,
)
from trading_algo.instruments import InstrumentSpec
from trading_algo.persistence import SqliteStore


@pytest.fixture(autouse=True)
def _tfsa_env(monkeypatch):
    # Real account IDs live only in the gitignored .env; tests use a fake.
    monkeypatch.setenv("TRADING_TFSA_ACCOUNTS", "UTESTTFSA1")


def _broker(net_liq: float = 50_000.0, positions: list[Position] | None = None) -> SimBroker:
    b = SimBroker()
    b.connect()
    b.set_account_values({"NetLiquidation": net_liq})
    b.set_positions(positions or [])
    return b


def _pos(symbol: str, qty: float, *, kind="STK", right=None, strike=None, expiry=None,
         avg_cost=None, account="U1234567") -> Position:
    return Position(
        account=account,
        instrument=InstrumentSpec(kind=kind, symbol=symbol, right=right, strike=strike, expiry=expiry,
                                  multiplier="100" if kind == "OPT" else None),
        quantity=qty, avg_cost=avg_cost, timestamp_epoch_s=time.time(),
    )


def _future_expiry(days: int = 35) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y%m%d")


def _stock_buy(**kw) -> ProposedOrderInput:
    base = dict(symbol="IWM", kind="STK", side="BUY", quantity=10, account="U1234567",
                written_exit="20%-off-close trailing stop")
    base.update(kw)
    return ProposedOrderInput(**base)


# ---------------------------------------------------------------- helpers

def test_parse_dte():
    from datetime import date
    assert parse_dte("20260116", today=date(2026, 1, 1)) == 15
    assert parse_dte(None) is None
    assert parse_dte("garbage") is None


def test_order_key_excludes_structure_includes_strike():
    # the keyed hash must distinguish strikes but not depend on the derived structure tag
    a = ProposedOrderInput(symbol="MP", kind="OPT", side="SELL", quantity=1, account="U1",
                           right="P", strike=80, expiry="20260116", structure="short-put")
    b = ProposedOrderInput(symbol="MP", kind="OPT", side="SELL", quantity=1, account="U1",
                           right="P", strike=80, expiry="20260116", structure="something-else")
    c = ProposedOrderInput(symbol="MP", kind="OPT", side="SELL", quantity=1, account="U1",
                           right="P", strike=85, expiry="20260116", structure="short-put")
    assert a.to_key() == b.to_key()      # structure excluded
    assert a.to_key() != c.to_key()      # strike included


# ---------------------------------------------------------------- config flags

def test_config_constitution_required_is_opt_in(monkeypatch):
    # opt-in: unset -> False even when live (the writer must be wired first)
    monkeypatch.delenv("TRADING_CONSTITUTION_REQUIRED", raising=False)
    monkeypatch.setenv("TRADING_LIVE_ENABLED", "true")
    assert TradingConfig.from_env().constitution_required is False
    monkeypatch.setenv("TRADING_LIVE_ENABLED", "false")
    assert TradingConfig.from_env().constitution_required is False


def test_config_constitution_required_explicit_override(monkeypatch):
    monkeypatch.setenv("TRADING_LIVE_ENABLED", "false")
    monkeypatch.setenv("TRADING_CONSTITUTION_REQUIRED", "true")
    assert TradingConfig.from_env().constitution_required is True


def test_config_constitution_required_typo_raises(monkeypatch):
    monkeypatch.setenv("TRADING_CONSTITUTION_REQUIRED", "tru")
    with pytest.raises(EnvParseError):
        TradingConfig.from_env()


def test_config_max_age_default_and_parse(monkeypatch):
    monkeypatch.delenv("TRADING_CONSTITUTION_MAX_AGE_S", raising=False)
    assert TradingConfig.from_env().constitution_max_age_s == 120
    monkeypatch.setenv("TRADING_CONSTITUTION_MAX_AGE_S", "60")
    assert TradingConfig.from_env().constitution_max_age_s == 60


# ---------------------------------------------------------------- persistence (cross-process)

def test_verdict_roundtrip_cross_store(tmp_path):
    # two SqliteStore instances on the same file == the two-process scenario
    db = str(tmp_path / "trading.sqlite3")
    writer = SqliteStore(db)
    writer.log_constitution_verdict(
        order_key="abc123", decision="WARN", complete=True,
        checks=[{"rule_id": "W1_opportunistic_roll", "severity": "WARN", "status": "FAIL"}],
        symbol="IAU", account="U1", ts_epoch_s=time.time(),
    )
    reader = SqliteStore(db)
    rec = reader.latest_constitution_verdict("abc123")
    assert rec is not None and rec["decision"] == "WARN" and rec["complete"] == 1
    assert reader.latest_constitution_verdict("nope") is None
    writer.close()
    reader.close()


def test_verdict_staleness(tmp_path):
    db = str(tmp_path / "trading.sqlite3")
    store = SqliteStore(db)
    t0 = 1_000_000.0
    store.log_constitution_verdict(order_key="k", decision="PASS", complete=True, checks=[], ts_epoch_s=t0)
    # fresh within window
    assert store.latest_constitution_verdict("k", max_age_s=30, now=t0 + 10) is not None
    # stale beyond window -> treated as absent (cannot authorize a transmit)
    assert store.latest_constitution_verdict("k", max_age_s=30, now=t0 + 60) is None
    store.close()


# ---------------------------------------------------------------- adapter / gate

def test_gate_blocks_gld_long_call_and_raises():
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="GLD", kind="OPT", side="BUY", quantity=1, account="U1234567",
        right="C", strike=300, expiry="20270115", structure="leap-long",
    )
    greeks = lambda spec: OptionGreeks(delta=0.85, iv=0.2, opt_price=40.0, und_price=320.0)
    with pytest.raises(ConstitutionViolation, match="C5_no_gld_long_call"):
        gate(broker, proposed, greeks_provider=greeks)


def test_gate_clean_stock_buy_passes_and_records(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    store = SqliteStore(db)
    broker = _broker(positions=[_pos("IAU", 200, avg_cost=40.0)])
    proposed = _stock_buy(symbol="IWM")
    spot = lambda sym: {"IWM": 220.0, "IAU": 45.0}.get(sym)
    v = gate(broker, proposed, store=store, spot_provider=spot)
    assert v.decision == "PASS"
    rec = store.latest_constitution_verdict(proposed.to_key())
    assert rec is not None and rec["decision"] == "PASS"
    store.close()


def test_factor_cap_blocks_with_real_enrichment():
    # existing IAU sleeve 24k (real-assets) + proposed URA 5k stock -> 58% of 50k NetLiq
    broker = _broker(positions=[_pos("IAU", 600, avg_cost=40.0)])  # 600*40 = 24k
    proposed = ProposedOrderInput(symbol="URA", kind="STK", side="BUY", quantity=200,
                                  account="U1234567", written_exit="exit at -8%")
    spot = lambda sym: {"IAU": 40.0, "URA": 25.0}.get(sym)  # URA 200*25 = 5k
    with pytest.raises(ConstitutionViolation, match="C1_factor_cap"):
        gate(broker, proposed, spot_provider=spot)


def test_short_put_gate_enriched_from_greeks():
    # small strike (assignment ~2% of NetLiq) + in-band DTE so the ONLY failing
    # check is the delta band -> isolates C8's delta logic.
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="U1234567",
        right="P", strike=10, expiry=_future_expiry(35), structure="short-put",
        credit=6.0, written_exit="close at 50% / roll if breached",
    )
    # delta -0.50 is OUTSIDE the 0.20-0.35 gate -> C8 BLOCK
    greeks = lambda spec: OptionGreeks(delta=-0.50, iv=0.18, opt_price=6.0, und_price=11.0,
                                       bid=5.9, ask=6.1)
    with pytest.raises(ConstitutionViolation, match="C8_short_put_gate"):
        gate(broker, proposed, greeks_provider=greeks)


def test_missing_greeks_skips_delta_and_gate_fails_closed():
    # in-band DTE + small assignment, but NO greeks -> the delta/extrinsic sub-checks
    # SKIP (rules never false-BLOCK on missing data), BUT the gate itself refuses
    # to CLEAR an under-enriched order (fail-closed).
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="U1234567",
        right="P", strike=10, expiry=_future_expiry(35), structure="short-put",
        written_exit="close at 50%",
    )
    # advisory mode: inspect the verdict without the completeness rail
    v = gate(broker, proposed, greeks_provider=None, require_complete=False)
    assert v.decision in {"PASS", "WARN"}  # rules did not false-BLOCK
    c8 = next(c for c in v.checks if c.rule_id == "C8_short_put_gate")
    assert c8.status == "SKIP"
    # enrichment is flagged incomplete...
    _, meta = build_eval_context(broker, proposed)
    assert meta.complete is False and "trade_greeks" in meta.missing
    # ...and the DEFAULT gate refuses to clear it
    with pytest.raises(ConstitutionViolation, match="INCOMPLETE"):
        gate(broker, proposed, greeks_provider=None)


def test_tfsa_short_put_blocks():
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="UTESTTFSA1",  # TFSA
        right="P", strike=10, expiry=_future_expiry(35), structure="short-put", credit=6.0,
        written_exit="x",
    )
    greeks = lambda spec: OptionGreeks(delta=-0.28, iv=0.18, opt_price=6.0, und_price=11.0)
    with pytest.raises(ConstitutionViolation, match="C6_tfsa_no_short_puts"):
        gate(broker, proposed, greeks_provider=greeks)


# ---------------------------------------------------------------- review fixes

def test_structure_inferred_tfsa_short_put_blocks_without_tag():
    # S5-4: omitting structure must NOT disable the TFSA short-put ban.
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="UTESTTFSA1",  # TFSA
        right="P", strike=10, expiry=_future_expiry(35), structure=None,  # NO tag
        credit=6.0, written_exit="x",
    )
    greeks = lambda spec: OptionGreeks(delta=-0.28, iv=0.18, opt_price=6.0, und_price=11.0)
    with pytest.raises(ConstitutionViolation, match="C6_tfsa_no_short_puts"):
        gate(broker, proposed, greeks_provider=greeks)


def test_lowercase_right_still_blocks_gld_call():
    # F3: lowercase right must not slip past C5 (the GLD long-call ban).
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="GLD", kind="OPT", side="BUY", quantity=1, account="U1234567",
        right="c", strike=300, expiry="20270115", structure="leap-long",  # lowercase 'c'
    )
    greeks = lambda spec: OptionGreeks(delta=0.85, iv=0.2, opt_price=40.0, und_price=320.0)
    with pytest.raises(ConstitutionViolation, match="C5_no_gld_long_call"):
        gate(broker, proposed, greeks_provider=greeks)


def test_new_program_marks_incomplete():
    # F4: a new option program with no instrument metrics is incomplete (fail-closed).
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="SLV", kind="OPT", side="SELL", quantity=1, account="U1234567",
        right="P", strike=10, expiry=_future_expiry(35), structure="short-put",
        credit=2.0, written_exit="x", is_new_program=True,
    )
    greeks = lambda spec: OptionGreeks(delta=-0.28, iv=0.3, opt_price=2.0, und_price=11.0)
    _, meta = build_eval_context(broker, proposed, greeks_provider=greeks)
    assert meta.complete is False and "trade_instrument_metrics" in meta.missing


def test_market_short_put_no_credit_marks_incomplete():
    # F6: a short put with no credit/limit can't evaluate the extrinsic floor -> incomplete.
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="U1234567",
        right="P", strike=10, expiry=_future_expiry(35), structure="short-put",
        written_exit="x",  # no credit, no limit_price
    )
    greeks = lambda spec: OptionGreeks(delta=-0.28, iv=0.3, opt_price=2.0, und_price=11.0)
    _, meta = build_eval_context(broker, proposed, greeks_provider=greeks)
    assert meta.complete is False and "trade_credit" in meta.missing


# ------------------------------------------------- SELL != open-short family

def test_trim_of_over_cap_factor_is_allowed():
    # C1: the factor cap must never block the trim it demands.
    # real-assets = 30k of 50k NetLiq (60%, breached); selling 250 IAU reduces it.
    broker = _broker(positions=[_pos("IAU", 750, avg_cost=40.0)])  # 750*40 = 30k
    proposed = ProposedOrderInput(symbol="IAU", kind="STK", side="SELL", quantity=250,
                                  account="U1234567")  # no written_exit needed for a close
    spot = lambda sym: 40.0
    v = gate(broker, proposed, spot_provider=spot)  # must NOT raise
    c1 = next(c for c in v.checks if c.rule_id == "C1_factor_cap")
    assert c1.status == "PASS" and "not increasing" in c1.observed


def test_index_core_exempt_from_factor_cap():
    # Broad-index (VFV/SPY core) is the sanctioned diversified base, not a
    # macro-factor bet — C1 SKIPs it (mirrors W4's index exemption).
    from trading_algo.constitution import EvalContext, PositionView, ProposedTrade, evaluate
    positions = [PositionView("VFV", "STK", None, 700, "index", 70_000.0, None, None)]
    t = ProposedTrade(symbol="SPY", kind="STK", right=None, side="BUY", quantity=10,
                      structure=None, factor="index", account="U1", account_type="MARGIN",
                      delta_notional=5_000.0, written_exit="x")
    v = evaluate(EvalContext(net_liq=50_000.0, trade=t, positions=positions))
    c1 = next(c for c in v.checks if c.rule_id == "C1_factor_cap")
    assert c1.status == "SKIP" and "exempt" in c1.observed


def test_sign_flip_is_not_a_reduction():
    # Flipping a +30k long sleeve to -28k short in one trade is NOT a trim —
    # it must face the cap on the far side (56% of 50k NetLiq -> BLOCK).
    from trading_algo.constitution import EvalContext, PositionView, ProposedTrade, evaluate
    positions = [PositionView("IAU", "STK", None, 750, "real-assets", 30_000.0, None, None)]
    flip = ProposedTrade(symbol="IAU", kind="STK", right=None, side="SELL", quantity=1450,
                         structure="close", factor="real-assets", account="U1",
                         account_type="MARGIN", delta_notional=-58_000.0, written_exit=None)
    v = evaluate(EvalContext(net_liq=50_000.0, trade=flip, positions=positions))
    c1 = next(c for c in v.checks if c.rule_id == "C1_factor_cap")
    assert c1.status == "FAIL"


def test_buy_into_over_cap_factor_still_blocks():
    broker = _broker(positions=[_pos("IAU", 750, avg_cost=40.0)])  # 30k = 60%, breached
    proposed = ProposedOrderInput(symbol="URA", kind="STK", side="BUY", quantity=100,
                                  account="U1234567", written_exit="exit -8%")
    spot = lambda sym: {"IAU": 40.0, "URA": 25.0}.get(sym)
    with pytest.raises(ConstitutionViolation, match="C1_factor_cap"):
        gate(broker, proposed, spot_provider=spot)


def test_sell_to_close_long_leap_over_60dte_allowed():
    # C4: closing a long 200-DTE call is risk-reducing, not a short open.
    expiry = _future_expiry(200)
    broker = _broker(positions=[
        _pos("NVDA", 1, kind="OPT", right="C", strike=900.0, expiry=expiry)])
    proposed = ProposedOrderInput(symbol="NVDA", kind="OPT", side="SELL", quantity=1,
                                  account="U1234567", right="C", strike=900.0,
                                  expiry=expiry, limit_price=50.0)
    greeks = lambda spec: OptionGreeks(delta=0.85, iv=0.3, opt_price=50.0, und_price=950.0)
    v = gate(broker, proposed, greeks_provider=greeks)  # must NOT raise
    c4 = next(c for c in v.checks if c.rule_id == "C4_no_short_over_60dte")
    assert c4.status == "SKIP"


def test_covered_call_over_60dte_still_blocked():
    # Selling a call against SHARES (no long option) IS a short-option open -> C4 fires.
    expiry = _future_expiry(90)
    broker = _broker(positions=[_pos("IAU", 100, avg_cost=60.0)])  # shares, not a long call
    proposed = ProposedOrderInput(symbol="IAU", kind="OPT", side="SELL", quantity=1,
                                  account="U1234567", right="C", strike=90.0,
                                  expiry=expiry, limit_price=1.0,
                                  structure="covered-call", written_exit="called away OK")
    greeks = lambda spec: OptionGreeks(delta=0.25, iv=0.2, opt_price=1.0, und_price=62.0)
    with pytest.raises(ConstitutionViolation, match="C4_no_short_over_60dte"):
        gate(broker, proposed, greeks_provider=greeks)


def test_sell_to_close_long_put_in_tfsa_allowed():
    # C6: closing a (permitted) long put in the TFSA must not be tagged short-put.
    # The covering long must be in the SAME account (account-scoped matching).
    expiry = _future_expiry(30)
    broker = _broker(positions=[
        _pos("VFV", 2, kind="OPT", right="P", strike=140.0, expiry=expiry,
             account="UTESTTFSA1")])
    proposed = ProposedOrderInput(symbol="VFV", kind="OPT", side="SELL", quantity=2,
                                  account="UTESTTFSA1", right="P", strike=140.0,
                                  expiry=expiry, limit_price=2.0)
    greeks = lambda spec: OptionGreeks(delta=-0.30, iv=0.2, opt_price=2.0, und_price=145.0)
    v = gate(broker, proposed, greeks_provider=greeks)  # must NOT raise
    c6 = next(c for c in v.checks if c.rule_id == "C6_tfsa_no_short_puts")
    assert c6.status == "PASS"


def test_partial_close_still_treated_as_short_open():
    # Selling 2 when long only 1 = partially opening a short -> conservative, gate applies.
    expiry = _future_expiry(90)
    broker = _broker(positions=[
        _pos("NVDA", 1, kind="OPT", right="C", strike=900.0, expiry=expiry)])
    proposed = ProposedOrderInput(symbol="NVDA", kind="OPT", side="SELL", quantity=2,
                                  account="U1234567", right="C", strike=900.0,
                                  expiry=expiry, limit_price=50.0, written_exit="x")
    greeks = lambda spec: OptionGreeks(delta=0.85, iv=0.3, opt_price=50.0, und_price=950.0)
    with pytest.raises(ConstitutionViolation, match="C4_no_short_over_60dte"):
        gate(broker, proposed, greeks_provider=greeks)


def test_share_trim_needs_no_written_exit():
    # W5: selling shares you own is not a new position; no written exit required.
    broker = _broker(positions=[_pos("MP", 100, avg_cost=60.0)])
    proposed = ProposedOrderInput(symbol="MP", kind="STK", side="SELL", quantity=50,
                                  account="U1234567")  # no written_exit
    v = gate(broker, proposed, spot_provider=lambda s: 60.0)  # must NOT raise
    w5 = next(c for c in v.checks if c.rule_id == "W5_written_exit")
    assert w5.status == "SKIP"


# ----------------------------------------- final-review regressions (tag/account)

def test_cross_account_long_does_not_cover():
    # CG-1: a long put in a SIBLING account can never cover a short put here.
    expiry = _future_expiry(30)
    broker = _broker(positions=[
        _pos("VFV", 2, kind="OPT", right="P", strike=140.0, expiry=expiry,
             account="U9999999")])  # different account
    proposed = ProposedOrderInput(symbol="VFV", kind="OPT", side="SELL", quantity=2,
                                  account="UTESTTFSA1", right="P", strike=140.0,
                                  expiry=expiry, limit_price=2.0)
    greeks = lambda spec: OptionGreeks(delta=-0.30, iv=0.2, opt_price=2.0, und_price=145.0)
    with pytest.raises(ConstitutionViolation, match="C6_tfsa_no_short_puts"):
        gate(broker, proposed, greeks_provider=greeks)


def test_roll_tag_cannot_narrow_the_gate():
    # F1: structure='roll' + is_roll=True must NOT bypass C6/C8 — the sell-leg
    # of a put roll IS a short-put open through the same gate.
    expiry = _future_expiry(35)
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="UTESTTFSA1",  # TFSA
        right="P", strike=10, expiry=expiry, structure="roll", is_roll=True,
        credit=0.6, limit_price=0.6,
    )
    greeks = lambda spec: OptionGreeks(delta=-0.28, iv=0.2, opt_price=0.6, und_price=11.0)
    with pytest.raises(ConstitutionViolation, match="C6_tfsa_no_short_puts"):
        gate(broker, proposed, greeks_provider=greeks)


def test_roll_tagged_short_put_faces_c8_in_margin():
    expiry = _future_expiry(35)
    broker = _broker()
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="U1234567",
        right="P", strike=10, expiry=expiry, structure="roll", is_roll=True,
        credit=0.6, limit_price=0.6,
    )
    greeks = lambda spec: OptionGreeks(delta=-0.55, iv=0.2, opt_price=0.6, und_price=10.2)
    with pytest.raises(ConstitutionViolation, match="C8_short_put_gate"):
        gate(broker, proposed, greeks_provider=greeks)


def test_gld_is_roll_claim_does_not_bypass_c5():
    # F2: is_roll is caller-asserted; with NO existing GLD short call the BUY
    # cannot be a close -> the absolute GLD ban fires.
    broker = _broker()  # zero GLD positions
    proposed = ProposedOrderInput(
        symbol="GLD", kind="OPT", side="BUY", quantity=1, account="U1234567",
        right="C", strike=300, expiry=_future_expiry(200), structure="roll", is_roll=True,
    )
    greeks = lambda spec: OptionGreeks(delta=0.85, iv=0.2, opt_price=40.0, und_price=320.0)
    with pytest.raises(ConstitutionViolation, match="C5_no_gld_long_call"):
        gate(broker, proposed, greeks_provider=greeks)


def test_gld_buy_to_close_existing_short_call_allowed():
    # The ONLY legitimate GLD call BUY: position-verified buy-to-close.
    expiry = _future_expiry(30)
    broker = _broker(positions=[
        _pos("GLD", -1, kind="OPT", right="C", strike=300.0, expiry=expiry)])
    proposed = ProposedOrderInput(
        symbol="GLD", kind="OPT", side="BUY", quantity=1, account="U1234567",
        right="C", strike=300.0, expiry=expiry, limit_price=2.0,
    )
    greeks = lambda spec: OptionGreeks(delta=0.30, iv=0.2, opt_price=2.0, und_price=295.0)
    v = gate(broker, proposed, greeks_provider=greeks)  # must NOT raise
    c5 = next(c for c in v.checks if c.rule_id == "C5_no_gld_long_call")
    assert c5.status == "PASS"


def test_overlay_cannot_mask_long_side_cap():
    # CG-2: a short-call overlay must not manufacture headroom for MORE longs.
    from trading_algo.constitution import EvalContext, PositionView, ProposedTrade, evaluate
    positions = [
        PositionView("IAU", "STK", None, 750, "real-assets", 30_000.0, None, None),
        PositionView("IAU", "OPT", "C", -5, "real-assets", -28_000.0, None, 30),  # overlay
    ]
    buy = ProposedTrade(symbol="URA", kind="STK", right=None, side="BUY", quantity=800,
                        structure=None, factor="real-assets", account="U1",
                        account_type="MARGIN", delta_notional=22_000.0, written_exit="x")
    v = evaluate(EvalContext(net_liq=50_000.0, trade=buy, positions=positions))
    c1 = next(c for c in v.checks if c.rule_id == "C1_factor_cap")
    assert c1.status == "FAIL"  # long side 52k = 104%, net-masking refused


def test_c1_partial_unknown_never_false_blocks():
    # CG-3(low): a would-FAIL on PARTIAL sums must SKIP (missing deltas are
    # usually offsetting hedges); transmit still fails closed via incomplete.
    from trading_algo.constitution import EvalContext, PositionView, ProposedTrade, evaluate
    positions = [
        PositionView("IAU", "STK", None, 500, "real-assets", 20_000.0, None, None),
        PositionView("IAU", "OPT", "C", -5, "real-assets", None, None, 30),  # unknown delta
    ]
    buy = ProposedTrade(symbol="URA", kind="STK", right=None, side="BUY", quantity=300,
                        structure=None, factor="real-assets", account="U1",
                        account_type="MARGIN", delta_notional=8_000.0, written_exit="x")
    v = evaluate(EvalContext(net_liq=50_000.0, trade=buy, positions=positions))
    c1 = next(c for c in v.checks if c.rule_id == "C1_factor_cap")
    assert c1.status == "SKIP" and c1.confidence == "LOW"


def test_fut_sell_never_closes_long_and_flagged_unsupported():
    # CG-4(low): FUT cannot be positively matched (no expiry on the view) ->
    # conservative + the kind is flagged incomplete.
    broker = _broker(positions=[_pos("MES", 1, kind="FUT", expiry="20261218")])
    proposed = ProposedOrderInput(symbol="MES", kind="FUT", side="SELL", quantity=1,
                                  account="U1234567", expiry="20260918")
    ctx, meta = build_eval_context(broker, proposed, spot_provider=lambda s: 5000.0)
    assert ctx.trade.closes_long is False
    assert "trade_kind_unsupported" in meta.missing


def test_nl_missing_skips_sizing_rules():
    # CG-3(med): NetLiq absent -> C1/C2/C3 SKIP (never PASS at "0%"), and the
    # gate refuses to clear (incomplete).
    broker = SimBroker(); broker.connect()
    broker.set_account_values({})  # no NetLiquidation
    proposed = ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="U1234567",
        right="P", strike=10, expiry=_future_expiry(35), structure="short-put",
        credit=0.6, limit_price=0.6, written_exit="x",
    )
    greeks = lambda spec: OptionGreeks(delta=-0.28, iv=0.2, opt_price=0.6, und_price=11.0)
    v = gate(broker, proposed, greeks_provider=greeks, require_complete=False)
    for rid in ("C1_factor_cap", "C2_per_underlying_assignment", "C3_aggregate_assignment"):
        assert next(c for c in v.checks if c.rule_id == rid).status == "SKIP"
    with pytest.raises(ConstitutionViolation, match="INCOMPLETE"):
        gate(broker, proposed, greeks_provider=greeks)


def test_order_key_binds_price():
    # F2(price): a re-priced order must produce a different key (forces re-check).
    a = ProposedOrderInput(symbol="SPY", kind="OPT", side="SELL", quantity=1, account="U1",
                           right="P", strike=500, expiry="20260801", limit_price=6.0)
    b = ProposedOrderInput(symbol="SPY", kind="OPT", side="SELL", quantity=1, account="U1",
                           right="P", strike=500, expiry="20260801", limit_price=5.0)
    assert a.to_key() != b.to_key()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

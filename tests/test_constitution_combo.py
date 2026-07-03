"""Combo (BAG) clearance: canonical leg-set key + verify/claim semantics."""

from __future__ import annotations

import time

import pytest

from trading_algo.constitution import ConstitutionViolation, combo_key
from trading_algo.constitution_clearance import verify_combo_clearance
from trading_algo.persistence import SqliteStore

LEGS = [("BUY", 111, 1), ("SELL", 222, 1)]


def test_combo_key_order_insensitive_and_content_bound():
    a = combo_key(legs=[("BUY", 111, 1), ("SELL", 222, 1)], side="BUY", quantity=2,
                  symbol="XSP", account="U1", limit_price=-1.10, order_type="LMT")
    b = combo_key(legs=[("SELL", 222, 1), ("BUY", 111, 1)], side="BUY", quantity=2,
                  symbol="XSP", account="U1", limit_price=-1.10, order_type="LMT")
    assert a == b  # leg ORDER never changes the key
    # every content dimension binds
    assert a != combo_key(legs=LEGS, side="BUY", quantity=3, symbol="XSP",
                          account="U1", limit_price=-1.10, order_type="LMT")
    assert a != combo_key(legs=[("BUY", 111, 1), ("SELL", 333, 1)], side="BUY",
                          quantity=2, symbol="XSP", account="U1",
                          limit_price=-1.10, order_type="LMT")
    assert a != combo_key(legs=LEGS, side="BUY", quantity=2, symbol="XSP",
                          account="U1", limit_price=-1.05, order_type="LMT")
    assert a.startswith("BAG")  # namespaced away from single-leg keys


def _write(db, key, decision="PASS", complete=True, age_s=0.0):
    s = SqliteStore(db)
    s.log_constitution_verdict(order_key=key, decision=decision, complete=complete,
                               checks=[], symbol="XSP", ts_epoch_s=time.time() - age_s)
    s.close()


def _verify(db, *, order_ref=None, **kw):
    s = SqliteStore(db)
    base = dict(legs=LEGS, side="BUY", quantity=2, symbol="XSP", account="U1",
                limit_price=-1.10, order_type="LMT", required=True, max_age_s=120)
    base.update(kw)
    try:
        verify_combo_clearance(s, order_ref=order_ref, **base)
    finally:
        s.close()


def _key(**kw):
    base = dict(legs=LEGS, side="BUY", quantity=2, symbol="XSP", account="U1",
                limit_price=-1.10, order_type="LMT")
    base.update(kw)
    return combo_key(**base)


def test_combo_clearance_roundtrip_and_claim(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _write(db, _key())
    _verify(db, order_ref="TAcombo1")          # clears + claims
    _verify(db, order_ref="TAcombo1")          # same ref re-check OK
    with pytest.raises(ConstitutionViolation, match="already used"):
        _verify(db, order_ref="TAcombo2")      # different ref refused


def test_combo_clearance_refusals(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    with pytest.raises(ConstitutionViolation, match="no fresh constitution clearance"):
        _verify(db)  # nothing on file
    _write(db, _key(), decision="BLOCK")
    with pytest.raises(ConstitutionViolation, match="BLOCK is on file"):
        _verify(db)
    db2 = str(tmp_path / "t2.sqlite3")
    _write(db2, _key(), complete=False)
    with pytest.raises(ConstitutionViolation, match="incomplete"):
        _verify(db2)
    db3 = str(tmp_path / "t3.sqlite3")
    _write(db3, _key(), age_s=999.0)
    with pytest.raises(ConstitutionViolation, match="no fresh constitution clearance"):
        _verify(db3)
    db4 = str(tmp_path / "t4.sqlite3")
    _write(db4, _key())
    with pytest.raises(ConstitutionViolation, match="no fresh"):
        _verify(db4, limit_price=-1.05)  # re-priced package -> different key


def test_effective_leg_action_inversion():
    from trading_algo.ibkr_tool import _effective_leg_action
    assert _effective_leg_action("BUY", "BUY") == "BUY"
    assert _effective_leg_action("BUY", "SELL") == "SELL"
    assert _effective_leg_action("SELL", "BUY") == "SELL"   # selling a BAG flips legs
    assert _effective_leg_action("SELL", "SELL") == "BUY"


def test_parse_legs():
    from trading_algo.ibkr_tool import _parse_legs
    assert _parse_legs("BUY:1:1,SELL:2:1") == [("BUY", 1, 1), ("SELL", 2, 1)]
    with pytest.raises(SystemExit):
        _parse_legs("BUY:1")  # malformed
    with pytest.raises(SystemExit):
        _parse_legs("HOLD:1:1")  # bad action


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

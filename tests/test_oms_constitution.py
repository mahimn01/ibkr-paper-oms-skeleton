"""OMS constitution-clearance backstop (Family A chokepoint), runnable with SimBroker."""

from __future__ import annotations

import time

import pytest

from trading_algo.broker.base import OrderRequest
from trading_algo.broker.sim import SimBroker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.constitution import ConstitutionViolation, order_key
from trading_algo.instruments import InstrumentSpec
from trading_algo.oms import OrderManager
from trading_algo.persistence import SqliteStore

REQ = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY",
                   quantity=1, order_type="MKT")
KEY = order_key(symbol="AAPL", kind="STK", side="BUY", quantity=1, order_type="MKT")


def _cfg(db_path, *, required=True, max_age=30):
    return TradingConfig(broker="ibkr", live_enabled=True, dry_run=False, order_token="T",
                         confirm_token_required=False, db_path=db_path,
                         constitution_required=required, constitution_max_age_s=max_age,
                         ibkr=IBKRConfig())


def _write_verdict(db_path, decision, *, age_s=0.0, complete=True):
    s = SqliteStore(db_path)
    s.log_constitution_verdict(order_key=KEY, decision=decision, complete=complete, checks=[],
                               symbol="AAPL", ts_epoch_s=time.time() - age_s)
    s.close()


def _oms(broker, cfg):
    return OrderManager(broker, cfg, confirm_token="T")


def test_blocks_without_any_verdict(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(db))
    try:
        with pytest.raises(ConstitutionViolation, match="no fresh constitution clearance"):
            oms.submit(REQ)
    finally:
        oms.close()


def test_passes_with_fresh_pass_verdict(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(db))
    _write_verdict(db, "PASS")  # written by a separate store == cross-process
    try:
        res = oms.submit(REQ)
        assert res.status  # SimBroker accepted the order
    finally:
        oms.close()


def test_blocks_on_block_verdict(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(db))
    _write_verdict(db, "BLOCK")
    try:
        with pytest.raises(ConstitutionViolation, match="BLOCK is on file"):
            oms.submit(REQ)
    finally:
        oms.close()


def test_blocks_on_incomplete_verdict(tmp_path):
    # F1: an under-enriched (complete=False) PASS verdict must NOT clear a transmit.
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(db))
    _write_verdict(db, "PASS", complete=False)
    try:
        with pytest.raises(ConstitutionViolation, match="incomplete"):
            oms.submit(REQ)
    finally:
        oms.close()


def test_blocks_on_stale_verdict(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(db, max_age=30))
    _write_verdict(db, "PASS", age_s=100.0)  # older than max_age -> stale -> absent
    try:
        with pytest.raises(ConstitutionViolation, match="no fresh constitution clearance"):
            oms.submit(REQ)
    finally:
        oms.close()


def test_disabled_flag_passes_without_verdict(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(db, required=False))  # backward-compat: gate off
    try:
        res = oms.submit(REQ)
        assert res.status
    finally:
        oms.close()


def test_modify_requires_clearance_too(tmp_path):
    # CG-5: a modify routes NEW exposure — same clearance as submit.
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    # seed an order while the gate is OFF
    oms_off = _oms(broker, _cfg(db, required=False))
    res = oms_off.submit(REQ)
    oms_off.close()
    oms_on = _oms(broker, _cfg(db))
    try:
        with pytest.raises(ConstitutionViolation, match="no fresh constitution clearance"):
            oms_on.modify(res.order_id, REQ)
    finally:
        oms_on.close()


def test_clearance_is_single_use(tmp_path):
    # CG-4: one PASS verdict cannot authorize two transmits (fresh order_refs).
    db = str(tmp_path / "t.sqlite3")
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(db, max_age=3600))
    _write_verdict(db, "PASS")
    try:
        assert oms.submit(REQ).status  # first transmit claims the verdict
        with pytest.raises(ConstitutionViolation, match="already used"):
            oms.submit(REQ)  # normalized() mints a fresh order_ref -> refused
    finally:
        oms.close()


def test_required_but_no_store_fails_closed():
    broker = SimBroker(); broker.connect()
    oms = _oms(broker, _cfg(None, required=True))  # no db_path -> no store
    try:
        with pytest.raises(ConstitutionViolation, match="no TRADING_DB_PATH"):
            oms.submit(REQ)
    finally:
        oms.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

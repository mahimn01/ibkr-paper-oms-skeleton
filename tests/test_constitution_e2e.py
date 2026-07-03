"""End-to-end contract: the constitution-check WRITE site (adapter.gate via
ProposedOrderInput.to_key) and the OMS transmit READ site (key rebuilt from an
OrderRequest) must produce the SAME key for the same order — and different
orders must NOT match. Two separate SqliteStore instances on one file mirror
the real two-process flow."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trading_algo.broker.base import OrderRequest
from trading_algo.broker.sim import SimBroker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.constitution import ConstitutionViolation
from trading_algo.constitution_adapter import OptionGreeks, ProposedOrderInput, gate
from trading_algo.instruments import InstrumentSpec
from trading_algo.oms import OrderManager
from trading_algo.persistence import SqliteStore


def _future_expiry(days: int = 35) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y%m%d")


def _write_via_gate(db: str, proposed: ProposedOrderInput, **gate_kw) -> None:
    broker = SimBroker(); broker.connect()
    store = SqliteStore(db)
    try:
        gate(broker, proposed, store=store, **gate_kw)
    finally:
        store.close()


def _oms(db: str) -> OrderManager:
    broker = SimBroker(); broker.connect()
    cfg = TradingConfig(broker="ibkr", live_enabled=True, dry_run=False, order_token="T",
                        db_path=db, constitution_required=True, constitution_max_age_s=30,
                        ibkr=IBKRConfig())
    return OrderManager(broker, cfg, confirm_token="T")


def test_stk_clearance_roundtrip(tmp_path):
    db = str(tmp_path / "trading.sqlite3")
    _write_via_gate(db, ProposedOrderInput(
        symbol="IWM", kind="STK", side="BUY", quantity=10, account="U1234567",
        order_type="MKT", written_exit="20% trail"), spot_provider=lambda s: 220.0)
    oms = _oms(db)
    try:
        req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="IWM"), side="BUY",
                           quantity=10, order_type="MKT", account="U1234567")
        assert oms.submit(req).status  # cleared
        # a DIFFERENT order (qty) must not reuse the clearance
        req2 = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="IWM"), side="BUY",
                            quantity=11, order_type="MKT", account="U1234567")
        with pytest.raises(ConstitutionViolation):
            oms.submit(req2)
    finally:
        oms.close()


def test_opt_limit_clearance_roundtrip_and_reprice_refused(tmp_path):
    db = str(tmp_path / "trading.sqlite3")
    expiry = _future_expiry(35)
    greeks = lambda spec: OptionGreeks(delta=-0.28, iv=0.2, opt_price=0.60, und_price=11.0,
                                       bid=0.55, ask=0.65)
    _write_via_gate(db, ProposedOrderInput(
        symbol="F", kind="OPT", side="SELL", quantity=1, account="U1234567",
        right="P", strike=10, expiry=expiry, order_type="LMT", limit_price=0.60,
        credit=0.60, structure="short-put", written_exit="close at 50%"),
        greeks_provider=greeks)
    oms = _oms(db)
    inst = InstrumentSpec(kind="OPT", symbol="F", right="P", strike=10.0,
                          expiry=expiry, multiplier="100")
    try:
        req = OrderRequest(instrument=inst, side="SELL", quantity=1, order_type="LMT",
                           limit_price=0.60, account="U1234567")
        assert oms.submit(req).status  # same price -> cleared
        # re-priced order must force a fresh check (price is bound into the key)
        req2 = OrderRequest(instrument=inst, side="SELL", quantity=1, order_type="LMT",
                            limit_price=0.50, account="U1234567")
        with pytest.raises(ConstitutionViolation):
            oms.submit(req2)
    finally:
        oms.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

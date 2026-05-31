"""Tests that mandatory client_order_id reaches the broker on every send.

Closes the gap where a strategy crash mid-place_order could leave the
process unable to recover the in-flight order: the OrderRef on file
(derived from intent or auto-generated) is the venue-side handle the
idempotency layer queries to resume.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading_algo.broker.base import OrderRequest, OrderResult, OrderStatus
from trading_algo.config import TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.oms import OrderManager
from trading_algo.orders import TradeIntent


def _intent() -> TradeIntent:
    return TradeIntent(
        instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART"),
        side="BUY",
        quantity=10,
        order_type="MKT",
    )


def _config(tmp_path: Path) -> TradingConfig:
    return TradingConfig(
        broker="ibkr",
        live_enabled=True,
        order_token=None,
        confirm_token_required=False,
        dry_run=False,
        db_path=str(tmp_path / "oms.sqlite"),
    )


# ---------------------------------------------------------------- intent layer


def test_trade_intent_default_factory_unique() -> None:
    a = _intent()
    b = _intent()
    assert a.client_order_id != b.client_order_id
    assert len(a.client_order_id) == 32  # UUID4 hex


def test_trade_intent_to_order_request_propagates_id() -> None:
    intent = _intent()
    req = intent.to_order_request()
    assert req.order_ref == intent.client_order_id


def test_trade_intent_explicit_id_preserved() -> None:
    intent = TradeIntent(
        instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART"),
        side="BUY",
        quantity=10,
        order_type="MKT",
        client_order_id="my-deterministic-key-abc",
    )
    assert intent.to_order_request().order_ref == "my-deterministic-key-abc"


def test_trade_intent_empty_id_rejected() -> None:
    intent = TradeIntent(
        instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART"),
        side="BUY",
        quantity=10,
        order_type="MKT",
        client_order_id="",
    )
    with pytest.raises(ValueError):
        intent.to_order_request()


def test_trade_intent_whitespace_id_rejected() -> None:
    intent = TradeIntent(
        instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART"),
        side="BUY",
        quantity=10,
        order_type="MKT",
        client_order_id="   ",
    )
    with pytest.raises(ValueError):
        intent.to_order_request()


# ---------------------------------------------------------------- OMS layer


def test_oms_autogenerates_order_ref_when_missing(tmp_path: Path) -> None:
    """A bare OrderRequest with no order_ref still ends up at the broker
    *with* a UUID4 — OrderRequest.normalized() auto-fills it.
    Defense-in-depth contract: every order that hits the broker carries
    a venue-side idempotency key, even if the caller forgot."""
    broker = MagicMock()
    broker.place_order.return_value = OrderResult(order_id="42", status="Submitted")
    broker.get_order_status.return_value = OrderStatus(
        order_id="42", status="Submitted", filled=0.0, remaining=10.0, avg_fill_price=None,
    )
    cfg = _config(tmp_path)
    oms = OrderManager(broker, cfg)
    try:
        bare = OrderRequest(
            instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART"),
            side="BUY",
            quantity=10,
            order_type="MKT",
            order_ref=None,
        )
        oms.submit(bare)
        sent = broker.place_order.call_args[0][0]
        assert sent.order_ref is not None
        assert len(sent.order_ref) == 32  # UUID4 hex
    finally:
        oms.close()


def test_oms_accepts_request_with_explicit_order_ref(tmp_path: Path) -> None:
    broker = MagicMock()
    broker.place_order.return_value = OrderResult(order_id="42", status="Submitted")
    broker.get_order_status.return_value = OrderStatus(
        order_id="42", status="Submitted", filled=0.0, remaining=10.0, avg_fill_price=None,
    )
    cfg = _config(tmp_path)
    oms = OrderManager(broker, cfg)
    try:
        req = OrderRequest(
            instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART"),
            side="BUY",
            quantity=10,
            order_type="MKT",
            order_ref="explicit-ref-001",
        )
        result = oms.submit(req)
        assert result.order_id == "42"
        # Broker received the request with the explicit ref preserved.
        sent = broker.place_order.call_args[0][0]
        assert sent.order_ref == "explicit-ref-001"
    finally:
        oms.close()


def test_oms_intent_path_always_sets_order_ref(tmp_path: Path) -> None:
    """End-to-end: TradeIntent -> to_order_request() -> oms.submit() must
    reach the broker with order_ref populated."""
    broker = MagicMock()
    broker.place_order.return_value = OrderResult(order_id="42", status="Submitted")
    broker.get_order_status.return_value = OrderStatus(
        order_id="42", status="Submitted", filled=0.0, remaining=10.0, avg_fill_price=None,
    )
    cfg = _config(tmp_path)
    oms = OrderManager(broker, cfg)
    try:
        intent = _intent()
        oms.submit(intent.to_order_request())
        sent = broker.place_order.call_args[0][0]
        assert sent.order_ref == intent.client_order_id
        assert len(sent.order_ref) == 32
    finally:
        oms.close()

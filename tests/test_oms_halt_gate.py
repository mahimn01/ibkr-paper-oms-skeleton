"""Test that OMS.submit/modify is gated by the halt sentinel."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading_algo.broker.base import OrderRequest, OrderResult
from trading_algo.config import TradingConfig
from trading_algo.halt import HaltActive, write_halt
from trading_algo.oms import OrderManager


def _config(tmp_path: Path) -> TradingConfig:
    """A minimal config with live trading disabled (dry_run=True)."""
    return TradingConfig(
        broker="ibkr",
        live_enabled=False,
        order_token=None,
        confirm_token_required=False,
        dry_run=True,
        db_path=str(tmp_path / "oms.sqlite"),
    )


def _request() -> OrderRequest:
    """A minimal OrderRequest the broker would accept."""
    from trading_algo.instruments import InstrumentSpec
    return OrderRequest(
        instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART"),
        side="BUY",
        quantity=10,
        order_type="MKT",
    )


def test_submit_raises_when_halted(tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    halt_path = tmp_path / "HALTED"
    monkeypatch.setenv("TRADING_HALT_PATH", str(halt_path))

    write_halt(reason="test", by="pytest", path=halt_path)

    broker = MagicMock()
    cfg = _config(tmp_path)
    oms = OrderManager(broker, cfg)
    try:
        with pytest.raises(HaltActive):
            oms.submit(_request())
        # Critical: even dry-run submits are blocked while halted.
        broker.place_order.assert_not_called()
    finally:
        oms.close()


def test_submit_succeeds_when_not_halted(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    halt_path = tmp_path / "HALTED"
    monkeypatch.setenv("TRADING_HALT_PATH", str(halt_path))

    broker = MagicMock()
    broker.place_order.return_value = OrderResult(order_id="100", status="Submitted")
    # live_enabled + dry_run so we exercise the halt-then-dry-run short-circuit
    cfg = TradingConfig(
        broker="ibkr",
        live_enabled=True,
        order_token=None,
        confirm_token_required=False,
        dry_run=True,
        db_path=str(tmp_path / "oms.sqlite"),
    )
    oms = OrderManager(broker, cfg)
    try:
        # Dry run path: returns "dry-run" without touching broker.
        result = oms.submit(_request())
        assert result.status == "DryRun"
        broker.place_order.assert_not_called()
    finally:
        oms.close()


def test_modify_raises_when_halted(tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    halt_path = tmp_path / "HALTED"
    monkeypatch.setenv("TRADING_HALT_PATH", str(halt_path))
    write_halt(reason="test", by="pytest", path=halt_path)

    broker = MagicMock()
    cfg = _config(tmp_path)
    oms = OrderManager(broker, cfg)
    try:
        with pytest.raises(HaltActive):
            oms.modify("100", _request())
        broker.modify_order.assert_not_called()
    finally:
        oms.close()


def test_cancel_NOT_blocked_when_halted(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancels reduce exposure — by design, the halt sentinel does not
    block them. An operator who halted often wants to flatten the book.

    We assert specifically that HaltActive is NOT raised; other gates
    (authorize_send) are orthogonal.
    """
    halt_path = tmp_path / "HALTED"
    monkeypatch.setenv("TRADING_HALT_PATH", str(halt_path))
    write_halt(reason="test", by="pytest", path=halt_path)

    broker = MagicMock()
    # Live-enabled so authorize_send doesn't reject before we reach the
    # halt check (which we're proving doesn't gate cancels).
    cfg = TradingConfig(
        broker="ibkr",
        live_enabled=True,
        order_token=None,
        confirm_token_required=False,
        dry_run=False,
        db_path=str(tmp_path / "oms.sqlite"),
    )
    oms = OrderManager(broker, cfg)
    try:
        # Must not raise HaltActive. Other unrelated errors are fine.
        try:
            oms.cancel("100")
        except HaltActive:
            pytest.fail("cancel was unexpectedly halt-gated")
        except Exception:
            pass  # tolerate other errors from the mock chain
        broker.cancel_order.assert_called_once()
    finally:
        oms.close()

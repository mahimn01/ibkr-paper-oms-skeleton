from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from trading_algo.broker.base import Broker, OrderRequest
from trading_algo.instruments import InstrumentSpec, validate_instrument
from trading_algo.market_data import MarketDataClient, MarketDataConfig
from trading_algo.oms import OrderManager


class ToolError(RuntimeError):
    pass


def list_tools() -> list[dict[str, Any]]:
    """
    For display only (Gemini function calling is not used; we use a custom JSON protocol).
    """
    return [
        {"name": "get_snapshot", "args": {"kind": "STK|FUT|FX", "symbol": "str", "exchange": "str?", "currency": "str?", "expiry": "str?"}},
        {"name": "get_positions", "args": {}},
        {"name": "get_account", "args": {}},
        {"name": "list_open_orders", "args": {}},
        {"name": "place_order", "args": {"order": {"instrument": {"kind": "STK|FUT|FX", "symbol": "str"}, "side": "BUY|SELL", "qty": "float", "type": "MKT|LMT|STP|STPLMT"}}},
        {"name": "modify_order", "args": {"order_id": "str", "order": {"...": "same as place_order"}}},
        {"name": "cancel_order", "args": {"order_id": "str"}},
        {"name": "oms_reconcile", "args": {}},
        {"name": "oms_track", "args": {"poll_seconds": "float?", "timeout_seconds": "float?"}},
    ]


def dispatch_tool(
    *,
    call_name: str,
    call_args: dict[str, Any],
    broker: Broker,
    oms: OrderManager,
    allowed_kinds: set[str],
    allowed_symbols: set[str],
) -> Any:
    name = str(call_name).strip()
    args = dict(call_args or {})
    if name == "get_snapshot":
        inst = _parse_instrument(args)
        _enforce_allowlist(inst, allowed_kinds, allowed_symbols)
        md = MarketDataClient(broker, MarketDataConfig())
        snap = md.get_snapshot(inst)
        return asdict(snap)

    if name == "get_positions":
        positions = broker.get_positions()
        return [asdict(p) for p in positions]

    if name == "get_account":
        acct = broker.get_account_snapshot()
        return asdict(acct)

    if name == "list_open_orders":
        st = broker.list_open_order_statuses()
        return [asdict(s) for s in st]

    if name == "place_order":
        req = _parse_order_request(args.get("order"))
        _enforce_allowlist(req.instrument, allowed_kinds, allowed_symbols)
        res = oms.submit(req)
        return {"order_id": res.order_id, "status": res.status}

    if name == "modify_order":
        order_id = str(args.get("order_id", "")).strip()
        if not order_id:
            raise ToolError("modify_order requires order_id")
        req = _parse_order_request(args.get("order"))
        _enforce_allowlist(req.instrument, allowed_kinds, allowed_symbols)
        res = oms.modify(order_id, req)
        return {"order_id": res.order_id, "status": res.status}

    if name == "cancel_order":
        order_id = str(args.get("order_id", "")).strip()
        if not order_id:
            raise ToolError("cancel_order requires order_id")
        oms.cancel(order_id)
        return {"order_id": order_id, "status": "CancelRequested", "ts": time.time()}

    if name == "oms_reconcile":
        return oms.reconcile()

    if name == "oms_track":
        poll_seconds = float(args.get("poll_seconds", 1.0))
        timeout = args.get("timeout_seconds")
        timeout_seconds = float(timeout) if timeout is not None else None
        oms.track_open_orders(poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
        return {"ok": True}

    raise ToolError(f"Unknown tool: {name}")


def _parse_instrument(obj: dict[str, Any]) -> InstrumentSpec:
    kind = str(obj.get("kind", "STK")).strip().upper()
    symbol = str(obj.get("symbol", "")).strip().upper()
    if not symbol:
        raise ToolError("instrument.symbol is required")
    inst = InstrumentSpec(
        kind=kind,
        symbol=symbol,
        exchange=(str(obj["exchange"]).strip() if obj.get("exchange") else None),
        currency=(str(obj["currency"]).strip().upper() if obj.get("currency") else None),
        expiry=(str(obj["expiry"]).strip() if obj.get("expiry") else None),
    )
    return validate_instrument(inst)


def _parse_order_request(order_obj: Any) -> OrderRequest:
    if not isinstance(order_obj, dict):
        raise ToolError("order must be an object")
    inst_obj = order_obj.get("instrument")
    if not isinstance(inst_obj, dict):
        raise ToolError("order.instrument must be an object")
    inst = _parse_instrument(inst_obj)

    req = OrderRequest(
        instrument=inst,
        side=str(order_obj.get("side", "BUY")).strip().upper(),
        quantity=float(order_obj.get("qty", 0.0)),
        order_type=str(order_obj.get("type", "MKT")).strip().upper(),
        limit_price=(float(order_obj["limit_price"]) if order_obj.get("limit_price") is not None else None),
        stop_price=(float(order_obj["stop_price"]) if order_obj.get("stop_price") is not None else None),
        tif=str(order_obj.get("tif", "DAY")).strip().upper(),
        outside_rth=bool(order_obj.get("outside_rth", False)),
        good_till_date=(str(order_obj.get("good_till_date")).strip() if order_obj.get("good_till_date") else None),
        account=(str(order_obj.get("account")).strip() if order_obj.get("account") else None),
        order_ref=(str(order_obj.get("order_ref")).strip() if order_obj.get("order_ref") else None),
        oca_group=(str(order_obj.get("oca_group")).strip() if order_obj.get("oca_group") else None),
        transmit=bool(order_obj.get("transmit", True)),
    )
    return req.normalized()


def _enforce_allowlist(inst: InstrumentSpec, allowed_kinds: set[str], allowed_symbols: set[str]) -> None:
    if inst.kind.upper() not in {k.upper() for k in allowed_kinds}:
        raise ToolError(f"Instrument kind not allowed: {inst.kind}")
    if allowed_symbols and inst.symbol.upper() not in {s.upper() for s in allowed_symbols}:
        raise ToolError(f"Symbol not allowed: {inst.symbol}")


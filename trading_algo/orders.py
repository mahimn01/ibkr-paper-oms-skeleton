from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from trading_algo.broker.base import OrderRequest
from trading_algo.instruments import InstrumentSpec, validate_instrument


def _new_client_order_id() -> str:
    """Default factory for TradeIntent.client_order_id.

    A 32-char hex UUID4. Routed to IBKR via OrderRequest.order_ref, where
    it serves as the venue-side idempotency key: if a place_order retry
    happens after a connection drop, the broker recognises the same
    orderRef and returns the existing order rather than creating a duplicate.
    """
    return uuid.uuid4().hex


@dataclass(frozen=True)
class TradeIntent:
    instrument: InstrumentSpec
    side: str  # BUY|SELL
    quantity: float
    order_type: str = "MKT"
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "DAY"
    client_order_id: str = field(default_factory=_new_client_order_id)
    """Mandatory idempotency key. Defaults to a fresh UUID4. Pass an
    explicit value to make a retry deterministic (e.g. derive_order_ref
    from idempotency.py for content-addressed orders)."""

    def to_order_request(self) -> OrderRequest:
        if not self.client_order_id or not self.client_order_id.strip():
            raise ValueError(
                "TradeIntent.client_order_id is required (non-empty). "
                "Construct with the default factory or supply an explicit "
                "idempotency key."
            )
        return OrderRequest(
            instrument=validate_instrument(self.instrument),
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            tif=self.tif,
            order_ref=self.client_order_id,
        )

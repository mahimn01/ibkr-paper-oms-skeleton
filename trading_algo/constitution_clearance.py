"""Shared fail-closed clearance check used at every transmit chokepoint
(OMS.submit, IBKRBroker.place_order/modify/bracket, the CLI families).

Keeping it in one tiny, dependency-light module (no broker/ib_async import) means
the same audited logic guards every path and is unit-testable without a gateway.
"""

from __future__ import annotations

from trading_algo.constitution import ConstitutionViolation, combo_key, order_key
from trading_algo.persistence import SqliteStore


def _verify_by_key(
    store: SqliteStore | None,
    key: str,
    describe: str,
    *,
    required: bool,
    max_age_s: float,
    order_ref: str | None = None,
    now: float | None = None,
) -> None:
    """Key-agnostic core of the clearance check (single-leg and combo share it).
    Fail-CLOSED on: no store, no verdict, stale, incomplete, BLOCK, or claimed
    by a different order."""
    if not required:
        return
    if store is None:
        raise ConstitutionViolation(
            "constitution_required is set but there is no TRADING_DB_PATH store to "
            "read a cleared verdict from — refusing to transmit."
        )
    rec = store.latest_constitution_verdict(key, max_age_s=max_age_s, now=now)
    if rec is None:
        raise ConstitutionViolation(
            f"no fresh constitution clearance for {describe} "
            f"(run a constitution-check within {float(max_age_s):g}s of transmit)."
        )
    if not rec.get("complete"):
        raise ConstitutionViolation(
            f"constitution verdict for {describe} is incomplete — its BLOCK-relevant "
            "analytics were unavailable at check time. Refusing to transmit."
        )
    if str(rec.get("decision")) == "BLOCK":
        raise ConstitutionViolation(
            f"a constitution BLOCK is on file for {describe} — refusing to transmit."
        )
    if order_ref:
        if not store.claim_constitution_verdict(int(rec["id"]), order_ref):
            raise ConstitutionViolation(
                f"the constitution clearance for {describe} was already used by a "
                "different order — run a fresh constitution-check for this transmit."
            )


def verify_combo_clearance(
    store: SqliteStore | None,
    *,
    legs: list[tuple[str, int, int]],
    side: str,
    quantity: float,
    symbol: str | None = None,
    account: str | None = None,
    limit_price: float | None = None,
    order_type: str | None = None,
    required: bool,
    max_age_s: float,
    order_ref: str | None = None,
    now: float | None = None,
) -> None:
    """Clearance check for a multi-leg BAG order, keyed by the canonical leg set."""
    key = combo_key(legs=legs, side=side, quantity=quantity, symbol=symbol,
                    account=account, limit_price=limit_price, order_type=order_type)
    _verify_by_key(
        store, key, f"combo {symbol or ''} {side.upper()} {float(quantity):g} ({len(legs)} legs)",
        required=required, max_age_s=max_age_s, order_ref=order_ref, now=now,
    )


def verify_clearance(
    store: SqliteStore | None,
    *,
    symbol: str,
    kind: str,
    side: str,
    quantity: float,
    right: str | None = None,
    strike: float | None = None,
    expiry: str | None = None,
    account: str | None = None,
    limit_price: float | None = None,
    order_type: str | None = None,
    required: bool,
    max_age_s: float,
    order_ref: str | None = None,
    now: float | None = None,
) -> None:
    """Raise ConstitutionViolation unless a fresh, COMPLETE, non-BLOCK verdict is
    on file for this order. Fail-CLOSED in every degraded case (no store, no
    verdict, stale, under-enriched, BLOCK, or already claimed by another order).

    When order_ref is supplied the clearance is SINGLE-USE: it atomically binds
    to that ref, so the same physical transmit passes both chokepoints (OMS +
    broker see one normalized ref) but a retry/duplicate with a fresh ref is
    refused — one clearance cannot authorize two orders."""
    key = order_key(
        symbol=symbol, kind=kind, side=side, quantity=quantity, right=right,
        strike=strike, expiry=expiry, account=account, limit_price=limit_price,
        order_type=order_type,
    )
    _verify_by_key(
        store, key, f"{symbol} {side} {float(quantity):g}",
        required=required, max_age_s=max_age_s, order_ref=order_ref, now=now,
    )

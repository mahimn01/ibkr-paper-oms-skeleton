"""Enrichment adapter for the pre-transmit constitution gate.

Turns a live broker (account + positions) plus a proposed order and injectable
option-analytics into a constitution.EvalContext, then evaluates, persists the
verdict, and (optionally) raises on BLOCK.

constitution.py stays pure (no IBKR, no clock); this is the one place that
touches market data. Greeks/spot come through injected providers so the whole
thing unit-tests with a SimBroker and fakes — no live gateway required.

Design rules (enterprise):
  * Enrichment is FAIL-SOFT: a provider that returns None or raises leaves the
    dependent ProposedTrade field None, so its rule SKIPs (never a false BLOCK).
  * The gate is FAIL-CLOSED: `EnrichmentMeta.complete` records whether the
    BLOCK-relevant analytics were actually available, so the transmit site can
    refuse to clear an under-enriched live order.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from trading_algo.broker.base import AccountSnapshot, Broker, Position
from trading_algo.constitution import (
    ConstitutionVerdict,
    ConstitutionViolation,
    EvalContext,
    PositionView,
    ProposedTrade,
    account_type_for,
    evaluate,
    factor_for,
    order_key,
)
from trading_algo.instruments import InstrumentSpec
from trading_algo.persistence import SqliteStore

OPT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class OptionGreeks:
    delta: float | None = None
    iv: float | None = None
    opt_price: float | None = None  # model/theoretical premium per share
    und_price: float | None = None  # underlying spot
    bid: float | None = None
    ask: float | None = None


# Injected so the adapter is testable without IBKR. Return None on any failure.
GreeksProvider = Callable[[InstrumentSpec], OptionGreeks | None]
SpotProvider = Callable[[str], float | None]


@dataclass(frozen=True)
class ProposedOrderInput:
    symbol: str
    kind: str  # STK | OPT
    side: str  # BUY | SELL
    quantity: float
    account: str
    right: str | None = None  # C | P
    strike: float | None = None
    expiry: str | None = None  # YYYYMMDD
    order_type: str = "MKT"
    limit_price: float | None = None
    structure: str | None = None  # short-put | leap-long | pmcc-long | covered-call | roll | close
    credit: float | None = None  # actual fill credit per share (C8 denominator); falls back to limit_price
    is_new_program: bool = False
    written_exit: str | None = None
    is_roll: bool = False
    rolled_short_is_working: bool | None = None
    rolled_short_breached_within_10d: bool | None = None
    losing_put_close_same_underlying_30d: bool | None = None

    def to_key(self) -> str:
        return order_key(
            symbol=self.symbol, kind=self.kind, side=self.side, quantity=self.quantity,
            right=self.right, strike=self.strike, expiry=self.expiry, account=self.account,
            limit_price=self.limit_price, order_type=self.order_type,
        )


@dataclass(frozen=True)
class EnrichmentMeta:
    produced_at: float
    complete: bool
    missing: list[str] = field(default_factory=list)


def parse_dte(expiry: str | None, *, today: date | None = None) -> int | None:
    if not expiry:
        return None
    try:
        exp = datetime.strptime(expiry.strip(), "%Y%m%d").date()
    except (ValueError, AttributeError):
        return None
    return (exp - (today or date.today())).days


def intrinsic_per_share(right: str | None, und: float, strike: float) -> float:
    if right == "C":
        return max(0.0, und - strike)
    if right == "P":
        return max(0.0, strike - und)
    return 0.0


def _signed_contracts(side: str, quantity: float) -> float:
    return float(quantity) if side.upper() == "BUY" else -float(quantity)


def _safe_call(provider, *args):
    if provider is None:
        return None
    try:
        return provider(*args)
    except Exception:
        return None


def position_views(
    positions: list[Position],
    *,
    greeks_provider: GreeksProvider | None = None,
    spot_provider: SpotProvider | None = None,
) -> list[PositionView]:
    views: list[PositionView] = []
    for p in positions:
        inst = p.instrument
        sym = inst.symbol.upper()
        qty = float(p.quantity)
        if inst.kind == "OPT":
            assignment = None
            if inst.right == "P" and qty < 0 and inst.strike is not None:
                assignment = float(inst.strike) * OPT_MULTIPLIER * abs(qty)
            delta_notional = None
            g = _safe_call(greeks_provider, inst)
            if g is not None and g.delta is not None and g.und_price is not None:
                delta_notional = g.delta * g.und_price * OPT_MULTIPLIER * qty
            views.append(PositionView(
                symbol=sym, kind="OPT", right=(inst.right or "").upper() or None, quantity=qty,
                factor=factor_for(sym), delta_notional=delta_notional,
                assignment_cash=assignment, dte=parse_dte(inst.expiry),
                strike=float(inst.strike) if inst.strike is not None else None,
                account=(p.account or "").upper() or None,
                avg_cost=float(p.avg_cost) if p.avg_cost is not None else None,
            ))
        else:
            spot = _safe_call(spot_provider, sym)
            if spot is None and p.avg_cost is not None:
                spot = float(p.avg_cost)
            delta_notional = qty * float(spot) if spot is not None else None
            views.append(PositionView(
                symbol=sym, kind=inst.kind, right=None, quantity=qty,
                factor=factor_for(sym), delta_notional=delta_notional,
                assignment_cash=None, dte=None,
                account=(p.account or "").upper() or None,
                avg_cost=float(p.avg_cost) if p.avg_cost is not None else None,
            ))
    return views


def _net_liq(account: AccountSnapshot) -> float | None:
    nl = account.values.get("NetLiquidation")
    return float(nl) if nl is not None and nl > 0 else None


def _covering_views(
    views: list[PositionView], *, symbol: str, kind: str, right: str | None,
    strike: float | None, dte: int | None, account: str | None,
) -> list[PositionView]:
    """Positions in the SAME instrument and the SAME account. Account-scoped:
    a long in a sibling account (e.g. the TFSA) can NEVER cover a trade in
    margin — treating it as cover would let a naked short masquerade as a
    close. Views with no account never cover anything (fail-closed)."""
    acct = (account or "").strip().upper()
    if not acct:
        return []
    same_acct = [v for v in views if v.account is not None and v.account == acct]
    if kind == "OPT":
        if right is None or strike is None or dte is None:
            return []
        return [
            v for v in same_acct
            if v.kind == "OPT" and v.symbol == symbol and v.right == right
            and v.strike is not None and abs(v.strike - float(strike)) < 1e-9
            and v.dte is not None and v.dte == dte
        ]
    if kind == "STK":
        return [v for v in same_acct if v.kind == "STK" and v.symbol == symbol]
    return []  # FUT/FX etc: never positively identified (conservative)


def _detect_closes_long(
    views: list[PositionView], *, symbol: str, kind: str, right: str | None,
    strike: float | None, dte: int | None, side: str, quantity: float,
    account: str | None,
) -> bool:
    """SELL != open-short: True when this SELL is fully covered by an existing
    LONG in the SAME instrument+account. Partial covers stay False
    (conservative: treated as opening)."""
    if side != "SELL":
        return False
    cover = _covering_views(views, symbol=symbol, kind=kind, right=right,
                            strike=strike, dte=dte, account=account)
    long_qty = sum(v.quantity for v in cover if v.quantity > 0)
    return long_qty >= float(quantity) - 1e-9


def _detect_closes_short(
    views: list[PositionView], *, symbol: str, kind: str, right: str | None,
    strike: float | None, dte: int | None, side: str, quantity: float,
    account: str | None,
) -> bool:
    """BUY != open-long: True when this BUY is fully covered by an existing
    SHORT in the SAME instrument+account (buy-to-close). Position-verified —
    never caller-asserted (this exempts C5's GLD ban, so it must be strict)."""
    if side != "BUY":
        return False
    cover = _covering_views(views, symbol=symbol, kind=kind, right=right,
                            strike=strike, dte=dte, account=account)
    short_qty = sum(-v.quantity for v in cover if v.quantity < 0)
    return short_qty >= float(quantity) - 1e-9


def _detect_profitable_close(
    views: list[PositionView], *, symbol: str, kind: str, right: str | None,
    strike: float | None, dte: int | None, account: str | None,
    mark: float | None,
) -> bool | None:
    """For a position-verified close of a LONG: is it green (mark > avg_cost)?
    Only derived for STK where units are unambiguous (IBKR option avg_cost
    carries multiplier ambiguity); None = unknown -> W6 passes conservatively."""
    if kind != "STK" or mark is None:
        return None
    cover = [v for v in _covering_views(views, symbol=symbol, kind=kind, right=right,
                                        strike=strike, dte=dte, account=account)
             if v.quantity > 0 and v.avg_cost is not None]
    if not cover:
        return None
    total_qty = sum(v.quantity for v in cover)
    if total_qty <= 0:
        return None
    wavg_cost = sum(v.quantity * v.avg_cost for v in cover) / total_qty
    return float(mark) > wavg_cost


def build_eval_context(
    broker: Broker,
    proposed: ProposedOrderInput,
    *,
    greeks_provider: GreeksProvider | None = None,
    spot_provider: SpotProvider | None = None,
    now: float | None = None,
) -> tuple[EvalContext, EnrichmentMeta]:
    produced_at = now if now is not None else time.time()
    missing: list[str] = []

    account = broker.get_account_snapshot()
    positions = broker.get_positions()
    net_liq = _net_liq(account)
    if net_liq is None:
        missing.append("net_liq")
        net_liq = 0.0

    views = position_views(positions, greeks_provider=greeks_provider, spot_provider=spot_provider)
    if any(v.delta_notional is None for v in views):
        missing.append("position_delta_notional")

    trade = _build_proposed_trade(
        proposed, views=views, greeks_provider=greeks_provider,
        spot_provider=spot_provider, missing=missing,
    )
    complete = not missing
    ctx = EvalContext(net_liq=net_liq, trade=trade, positions=views)
    return ctx, EnrichmentMeta(produced_at=produced_at, complete=complete, missing=missing)


def _build_proposed_trade(
    proposed: ProposedOrderInput,
    *,
    views: list[PositionView],
    greeks_provider: GreeksProvider | None,
    spot_provider: SpotProvider | None,
    missing: list[str],
) -> ProposedTrade:
    sym = proposed.symbol.upper()
    side = proposed.side.upper()
    kind = proposed.kind.upper()
    right = (proposed.right or "").upper() or None  # normalize: rules compare C/P case-sensitively
    qty = float(proposed.quantity)
    signed = _signed_contracts(side, qty)
    dte = parse_dte(proposed.expiry) if kind == "OPT" else None
    # SELL != open-short / BUY != open-long: position-verified, ACCOUNT-SCOPED.
    detect_kw = dict(views=views, symbol=sym, kind=kind, right=right,
                     strike=proposed.strike, dte=dte, account=proposed.account)
    closes_long = _detect_closes_long(side=side, quantity=qty, **detect_kw)
    closes_short = _detect_closes_short(side=side, quantity=qty, **detect_kw)
    # NORMALIZE structure from primitives — the tag can widen but never narrow
    # the gate: ANY sell-to-open of a put is a short-put (a roll's sell-leg
    # included: "a roll is a NEW trade through the same gate"), regardless of
    # what the caller labeled it. Only the position-verified close is exempt,
    # and put-credit-spread keeps its own (equally gated) tag.
    structure = proposed.structure
    if (kind == "OPT" and side == "SELL" and right == "P" and not closes_long
            and structure != "put-credit-spread"):
        structure = "short-put"
    common = dict(
        symbol=sym, kind=kind, right=right, side=side,
        quantity=qty, structure=structure, factor=factor_for(sym),
        account=proposed.account, account_type=account_type_for(proposed.account),
        is_new_program=proposed.is_new_program, written_exit=proposed.written_exit,
        is_roll=proposed.is_roll, rolled_short_is_working=proposed.rolled_short_is_working,
        rolled_short_breached_within_10d=proposed.rolled_short_breached_within_10d,
        losing_put_close_same_underlying_30d=proposed.losing_put_close_same_underlying_30d,
        closes_long=closes_long, closes_short=closes_short,
    )

    # C9 instrument-gate metrics (OI / weeklies / strike-width / mark-vs-intrinsic)
    # are not available through this adapter; flag incomplete so a new option
    # program fails closed rather than passing on a single spread check.
    if proposed.is_new_program:
        missing.append("trade_instrument_metrics")
    # Kinds this adapter cannot enrich correctly (FUT multipliers, FX) must not
    # silently produce a clean-looking verdict.
    if kind not in {"STK", "OPT"}:
        missing.append("trade_kind_unsupported")

    if kind != "OPT":
        spot = _safe_call(spot_provider, sym)
        delta_notional = signed * float(spot) if spot is not None else None
        if delta_notional is None:
            missing.append("trade_spot")
        profitable = _detect_profitable_close(**detect_kw, mark=spot) if closes_long else None
        return ProposedTrade(**common, delta_notional=delta_notional,
                             closing_profitable_position=profitable)

    g = _safe_call(greeks_provider, _opt_spec(proposed))
    delta = g.delta if g else None
    und = g.und_price if g else None
    opt_price = g.opt_price if g else None
    if g is None or delta is None or und is None or opt_price is None:
        missing.append("trade_greeks")

    extrinsic_pct = None
    extrinsic_pct_of_credit = None
    delta_notional = None
    size_notional = None
    assignment_cash = None
    spread_pct = None
    credit = proposed.credit if proposed.credit is not None else proposed.limit_price
    if und is not None and opt_price is not None and proposed.strike is not None:
        extr = opt_price - intrinsic_per_share(right, und, float(proposed.strike))
        if opt_price > 0:
            extrinsic_pct = extr / opt_price
        if credit is not None and credit > 0:
            extrinsic_pct_of_credit = extr / float(credit)
        size_notional = opt_price * OPT_MULTIPLIER * qty
    if delta is not None and und is not None:
        delta_notional = delta * signed * und * OPT_MULTIPLIER
    if right == "P" and side == "SELL" and proposed.strike is not None and not closes_long:
        assignment_cash = float(proposed.strike) * OPT_MULTIPLIER * qty
    if g is not None and g.bid is not None and g.ask is not None and g.ask > 0 and g.bid > 0:
        mid = (g.bid + g.ask) / 2.0
        if mid > 0:
            spread_pct = (g.ask - g.bid) / mid
    # A short put with no credit/limit has an unknowable extrinsic-of-credit floor
    # (C8) — flag incomplete so it fails closed at the transmit site.
    if structure == "short-put" and credit is None:
        missing.append("trade_credit")

    return ProposedTrade(
        **common, dte=dte, delta=delta, extrinsic_pct=extrinsic_pct,
        extrinsic_pct_of_credit=extrinsic_pct_of_credit, iv_rank=None,
        delta_notional=delta_notional, assignment_cash=assignment_cash,
        premium=opt_price, size_notional=size_notional, spread_pct=spread_pct,
    )


def _opt_spec(proposed: ProposedOrderInput) -> InstrumentSpec:
    return InstrumentSpec(
        kind="OPT", symbol=proposed.symbol, expiry=proposed.expiry,
        right=(proposed.right or "").upper() or None, strike=proposed.strike, multiplier="100",
    )


def record_verdict(
    store: SqliteStore,
    verdict: ConstitutionVerdict,
    meta: EnrichmentMeta,
    proposed: ProposedOrderInput,
) -> int:
    return store.log_constitution_verdict(
        order_key=proposed.to_key(),
        decision=verdict.decision,
        complete=meta.complete,
        checks=[asdict(c) for c in verdict.checks],
        symbol=proposed.symbol.upper(),
        account=proposed.account,
        context={"missing": meta.missing, "produced_at": meta.produced_at},
        ts_epoch_s=meta.produced_at,
    )


def enforce_or_raise(verdict: ConstitutionVerdict) -> None:
    if verdict.blocked:
        reasons = "; ".join(f"{c.rule_id}: {c.message}" for c in verdict.blocks)
        raise ConstitutionViolation(f"Constitution BLOCK — {reasons}")


def gate(
    broker: Broker,
    proposed: ProposedOrderInput,
    *,
    store: SqliteStore | None = None,
    greeks_provider: GreeksProvider | None = None,
    spot_provider: SpotProvider | None = None,
    raise_on_block: bool = True,
    require_complete: bool = True,
    now: float | None = None,
) -> ConstitutionVerdict:
    """Full pre-transmit path: enrich -> evaluate -> persist -> (raise on BLOCK
    or on incomplete enrichment). Fail-CLOSED by default: an under-enriched
    verdict (missing BLOCK-relevant analytics) raises rather than clearing —
    mirroring verify_clearance at the transmit chokepoints. Set
    require_complete=False only for advisory/inspection use."""
    ctx, meta = build_eval_context(
        broker, proposed, greeks_provider=greeks_provider, spot_provider=spot_provider, now=now,
    )
    verdict = evaluate(ctx)
    if store is not None:
        record_verdict(store, verdict, meta, proposed)
    if raise_on_block:
        enforce_or_raise(verdict)
        if require_complete and not meta.complete:
            raise ConstitutionViolation(
                f"constitution check for {proposed.symbol.upper()} {proposed.side.upper()} is "
                f"INCOMPLETE (missing: {', '.join(meta.missing)}) — refusing to clear."
            )
    return verdict

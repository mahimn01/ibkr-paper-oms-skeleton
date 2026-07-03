"""Pre-transmit constitution gate.

Encodes Mahimn's written trading rules (CLAUDE.md "Trading & Options
Management" + "Risk & Sizing Rules") as deterministic, portfolio-aware
preconditions evaluated against the LIVE account + the proposed order.

This module is PURE: it imports nothing from the broker / order path and
takes a fully-populated EvalContext, so it is unit-testable with synthetic
state. The IBKR-touching enrichment (pulling greeks / marks / DTE to build
the context) lives in a thin adapter, not here.

Rule severities (per the locked design):
  BLOCK (non-bypassable): C1 C2 C3 C4 C5 C6 C7a C8 C9 W2 W5
  WARN  (human decides):  C7b W1 W3 W4 W6
Missing data NEVER produces a false BLOCK — a rule that lacks its inputs
returns status=SKIP (confidence=LOW), it does not FAIL.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["BLOCK", "WARN", "INFO"]
Status = Literal["PASS", "FAIL", "SKIP"]
Decision = Literal["PASS", "WARN", "BLOCK"]


class ConstitutionViolation(ValueError):
    """Raised when a proposed order FAILs a BLOCK-severity constitution rule.

    Classified as VALIDATION (exit code 2) by exit_codes, mirroring
    RiskViolation, so agents can distinguish a deterministic constitution
    refusal from other ValueErrors.
    """

# Macro factor buckets for the C1 factor cap. real-assets = gold + silver +
# uranium + rare-earth (CLAUDE.md). Extend as the book grows.
FACTOR_MAP: dict[str, str] = {
    # gold
    "GLD": "real-assets", "IAU": "real-assets", "GDX": "real-assets", "GDXJ": "real-assets",
    "NEM": "real-assets", "AEM": "real-assets",
    # silver
    "SLV": "real-assets", "SIVR": "real-assets", "SIL": "real-assets", "PSLV": "real-assets",
    # uranium
    "URA": "real-assets", "URNM": "real-assets", "UUUU": "real-assets", "CCJ": "real-assets",
    "DNN": "real-assets",
    # rare-earth
    "REMX": "real-assets", "MP": "real-assets",
    # energy (separate factor)
    "XLE": "energy", "USO": "energy", "CVX": "energy", "XOM": "energy",
    # broad index
    "SPY": "index", "VFV": "index", "VOO": "index", "QQQ": "index", "IWM": "index",
    "XSP": "index", "SPX": "index",
    # tech
    "NVDA": "tech", "AMZN": "tech", "MSFT": "tech", "AAPL": "tech", "SMCI": "tech",
}

def factor_for(symbol: str) -> str:
    return FACTOR_MAP.get(symbol.upper(), "other")


def tfsa_accounts() -> frozenset[str]:
    """Accounts where short puts / put spreads are forbidden (TFSA: covered-calls
    + long options only). Sourced from TRADING_TFSA_ACCOUNTS (comma-separated,
    set in the gitignored .env) — real account IDs never live in this repo.
    NOTE: C6 depends on this being configured; an empty set means no account is
    treated as a TFSA."""
    raw = os.getenv("TRADING_TFSA_ACCOUNTS", "")
    return frozenset(a.strip().upper() for a in raw.split(",") if a.strip())


def account_type_for(account: str) -> str:
    return "TFSA" if account.upper() in tfsa_accounts() else "MARGIN"


@dataclass(frozen=True)
class PositionView:
    symbol: str
    kind: str  # STK | OPT | ...
    right: str | None  # C | P | None
    quantity: float  # signed; negative = short
    factor: str
    delta_notional: float | None  # signed $ delta exposure; None = unknown
    assignment_cash: float | None  # short puts only: strike * 100 * |qty|
    dte: int | None
    strike: float | None = None  # OPT only; used to match a SELL against a covering long
    account: str | None = None  # closes_long/closes_short matching is account-scoped
    avg_cost: float | None = None  # entry cost; used to tag profitable closes (W6)


@dataclass(frozen=True)
class ProposedTrade:
    symbol: str
    kind: str  # STK | OPT
    right: str | None  # C | P | None
    side: str  # BUY | SELL
    quantity: float
    structure: str | None  # short-put | leap-long | pmcc-long | covered-call | roll | ...
    factor: str
    account: str
    account_type: str  # TFSA | MARGIN
    # option analytics (None when greeks unavailable -> dependent rules SKIP)
    dte: int | None = None
    delta: float | None = None  # signed option delta
    extrinsic_pct: float | None = None  # extrinsic / premium  (C7a)
    extrinsic_pct_of_credit: float | None = None  # extrinsic / credit (C8, >= 0.90)
    iv_rank: float | None = None  # 0..100 (C7b)
    delta_notional: float | None = None  # signed $; for the C1 aggregate
    assignment_cash: float | None = None  # short puts
    premium: float | None = None
    size_notional: float | None = None  # ticket cost, for the <=10% NetLiq LEAP cap
    # instrument-gate inputs (C9), only checked when is_new_program
    is_new_program: bool = False
    spread_pct: float | None = None  # quoted spread / premium
    strike_width_pct: float | None = None  # width / spot
    weeklies_listed: bool | None = None
    near_money_oi: int | None = None
    mark_below_intrinsic: bool | None = None
    # journal-derived
    written_exit: str | None = None  # W5
    losing_put_close_same_underlying_30d: bool | None = None  # W2
    # roll context (W1/W3)
    is_roll: bool = False
    rolled_short_is_working: bool | None = None  # OTM + profitable + 0.20-0.35 delta
    rolled_short_breached_within_10d: bool | None = None
    # SELL != open-short: True when this SELL closes/reduces an existing LONG in
    # the same instrument AND ACCOUNT (detected from positions by the adapter).
    # Closing a long is risk-REDUCING — the short-opening rules (C4/C6/C8,
    # structure inference, W5) must not fire on it. Default False = conservative
    # (treated as opening) for callers that don't supply position context.
    closes_long: bool = False
    # BUY != open-long: True when this BUY closes an existing SHORT option in
    # the same instrument+account (position-verified). Exempts C5's GLD long-call
    # ban for the buy-to-close leg of an existing GLD short call — the only
    # legitimate GLD call BUY. NEVER caller-asserted.
    closes_short: bool = False
    # Set by the adapter when a close realizes a GAIN (mark > avg_cost of the
    # covered long). Drives W6: closing a green position on a feeling, not a
    # written trigger, is the revenge/cost-recovery reflex. None = unknown.
    closing_profitable_position: bool | None = None


@dataclass(frozen=True)
class EvalContext:
    net_liq: float
    trade: ProposedTrade
    positions: list[PositionView] = field(default_factory=list)


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    severity: Severity
    status: Status
    observed: str
    limit: str
    message: str
    confidence: str = "HIGH"  # HIGH | LOW


@dataclass(frozen=True)
class ConstitutionVerdict:
    decision: Decision
    checks: list[RuleResult]
    intent_hash: str

    @property
    def blocked(self) -> bool:
        return self.decision == "BLOCK"

    @property
    def warnings(self) -> list[RuleResult]:
        return [c for c in self.checks if c.severity == "WARN" and c.status == "FAIL"]

    @property
    def blocks(self) -> list[RuleResult]:
        return [c for c in self.checks if c.severity == "BLOCK" and c.status == "FAIL"]


def _is_short_option_open(t: ProposedTrade) -> bool:
    # A roll's sell-leg is still a short-option open — the >60 DTE cap (C4)
    # applies to it ("a roll is a NEW trade through the same gate"). Two exempt
    # cases: the buy-to-close leg (side=BUY), and a SELL that closes an existing
    # LONG option (closes_long) — that is risk-reducing, not a short open.
    return t.kind == "OPT" and t.side == "SELL" and not t.closes_long


# ---------------------------------------------------------------- BLOCK rules

def _c1_factor_cap(ctx: EvalContext) -> RuleResult:
    nl = ctx.net_liq
    t = ctx.trade
    tgt = t.factor
    # Broad-index is the sanctioned diversified CORE (TFSA holds VFV etc. by
    # design), not a macro-factor bet — the 50% cap targets concentrated macro
    # sleeves ("e.g. real-assets"). Same exemption W4 already applies.
    if tgt == "index":
        return RuleResult("C1_factor_cap", "BLOCK", "SKIP", "broad-index core (exempt)",
                          "<= 50% NetLiq", "The factor cap targets macro sleeves, not the index core.")
    if nl <= 0:
        return RuleResult("C1_factor_cap", "BLOCK", "SKIP", "NetLiq unknown",
                          "<= 50% NetLiq", "Cannot size against NetLiq.", "LOW")
    if t.delta_notional is None:
        return RuleResult("C1_factor_cap", "BLOCK", "SKIP",
                          f"{tgt} trade delta-notional unknown", "<= 50% NetLiq",
                          "Could not size the proposed trade's factor exposure.", "LOW")
    # SIDE-AWARE sleeve arithmetic. Exposure = max(long side, |short side|):
    # a short overlay (covered call) does not manufacture headroom for MORE
    # longs, and a sleeve can't be ratcheted up via sign-flip round trips —
    # each side faces the cap independently. Net is tracked for the flip guard.
    pre_long = 0.0
    pre_short = 0.0
    unknown = False
    for p in ctx.positions:
        if p.factor != tgt:
            continue
        if p.delta_notional is None:
            unknown = True
            continue
        if p.delta_notional >= 0:
            pre_long += p.delta_notional
        else:
            pre_short += p.delta_notional
    # Side assignment respects position-verified closes: selling shares you own
    # SHRINKS the long side (it does not create a short); buying back a short
    # SHRINKS the short side (it does not create a long).
    d = t.delta_notional
    post_long, post_short = pre_long, pre_short
    if d >= 0:
        if t.closes_short:
            post_short += d
        else:
            post_long += d
    else:
        if t.closes_long:
            post_long += d
        else:
            post_short += d
    pre_e = max(pre_long, abs(pre_short))
    post_e = max(post_long, abs(post_short))
    pre_net, post_net = pre_long + pre_short, post_long + post_short
    pct = post_e / nl
    # A trade that does not GROW either side's exposure can never be blocked —
    # blocking the trim/hedge the cap demands would invert the rule. A NET SIGN
    # FLIP is not a reduction: it passes through zero into a new exposure.
    same_side = pre_net == 0.0 or post_net == 0.0 or (pre_net > 0) == (post_net > 0)
    if post_e <= pre_e and same_side:
        return RuleResult(
            "C1_factor_cap", "BLOCK", "PASS",
            f"{tgt} {pre_e / nl:.0%} -> {pct:.0%} of NetLiq (not increasing)",
            "<= 50% NetLiq", "Trade does not grow factor exposure — cap cannot block a trim/hedge.",
            "LOW" if unknown else "HIGH",
        )
    if pct > 0.50:
        if unknown:
            # Partial sums are systematically biased (missing deltas are usually
            # offsetting hedges): missing data must never produce a false BLOCK.
            # The verdict is separately marked incomplete, so transmit still
            # fails closed.
            return RuleResult("C1_factor_cap", "BLOCK", "SKIP",
                              f"{tgt} >= {pct:.0%} on PARTIAL data (some deltas unknown)",
                              "<= 50% NetLiq", "Cannot size the sleeve reliably.", "LOW")
        return RuleResult(
            "C1_factor_cap", "BLOCK", "FAIL",
            f"{tgt} = {pct:.0%} of NetLiq",
            "<= 50% NetLiq",
            f"Single-factor cap: {tgt} would be {pct:.0%} of NetLiq delta-notional.",
        )
    return RuleResult(
        "C1_factor_cap", "BLOCK", "PASS",
        f"{tgt} = {pct:.0%} of NetLiq" + (" (partial: some deltas unknown)" if unknown else ""),
        "<= 50% NetLiq",
        f"Single-factor cap: {tgt} would be {pct:.0%} of NetLiq delta-notional.",
        "LOW" if unknown else "HIGH",
    )


def _c2_per_underlying_assignment(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if t.structure != "short-put":
        return RuleResult("C2_per_underlying_assignment", "BLOCK", "SKIP",
                          "n/a (not a short put)", "<= 15% NetLiq", "Only applies to short puts.")
    if t.assignment_cash is None:
        return RuleResult("C2_per_underlying_assignment", "BLOCK", "SKIP",
                          "assignment cash unknown", "<= 15% NetLiq",
                          "Missing strike/qty to size assignment.", "LOW")
    if ctx.net_liq <= 0:
        return RuleResult("C2_per_underlying_assignment", "BLOCK", "SKIP",
                          "NetLiq unknown", "<= 15% NetLiq",
                          "Cannot size against NetLiq.", "LOW")
    existing = sum(
        p.assignment_cash or 0.0 for p in ctx.positions
        if p.symbol == t.symbol and p.right == "P" and p.quantity < 0
    )
    total = existing + t.assignment_cash
    pct = total / ctx.net_liq if ctx.net_liq > 0 else 0.0
    status = "FAIL" if pct > 0.15 else "PASS"
    return RuleResult("C2_per_underlying_assignment", "BLOCK", status,
                      f"{t.symbol} assignment = {pct:.0%} of NetLiq", "<= 15% NetLiq",
                      f"Per-underlying assignment obligation for {t.symbol} would be {pct:.0%}.")


def _c3_aggregate_assignment(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if t.structure != "short-put":
        return RuleResult("C3_aggregate_assignment", "BLOCK", "SKIP",
                          "n/a (not a short put)", "<= 30% NetLiq", "Only applies to short puts.")
    if t.assignment_cash is None:
        return RuleResult("C3_aggregate_assignment", "BLOCK", "SKIP",
                          "assignment cash unknown", "<= 30% NetLiq", "Missing data.", "LOW")
    if ctx.net_liq <= 0:
        return RuleResult("C3_aggregate_assignment", "BLOCK", "SKIP",
                          "NetLiq unknown", "<= 30% NetLiq",
                          "Cannot size against NetLiq.", "LOW")
    existing = sum(
        p.assignment_cash or 0.0 for p in ctx.positions
        if p.right == "P" and p.quantity < 0
    )
    total = existing + t.assignment_cash
    pct = total / ctx.net_liq if ctx.net_liq > 0 else 0.0
    status = "FAIL" if pct > 0.30 else "PASS"
    return RuleResult("C3_aggregate_assignment", "BLOCK", status,
                      f"aggregate short-put cash = {pct:.0%} of NetLiq", "<= 30% NetLiq",
                      f"Aggregate short-put assignment cash would be {pct:.0%}.")


def _c4_no_short_over_60dte(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if not _is_short_option_open(t):
        return RuleResult("C4_no_short_over_60dte", "BLOCK", "SKIP",
                          "n/a (not a short-option open)", "<= 60 DTE", "Only applies to opening shorts.")
    if t.dte is None:
        return RuleResult("C4_no_short_over_60dte", "BLOCK", "SKIP",
                          "DTE unknown", "<= 60 DTE", "Missing expiry.", "LOW")
    status = "FAIL" if t.dte > 60 else "PASS"
    return RuleResult("C4_no_short_over_60dte", "BLOCK", status,
                      f"{t.dte} DTE", "<= 60 DTE", f"No short option > 60 DTE (this is {t.dte}).")


def _c5_no_gld_long_call(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    # The ban is ABSOLUTE for entries. The ONLY exempt GLD call BUY is the
    # position-verified buy-to-close of an existing GLD short call
    # (closes_short, detected from live positions — never a caller tag).
    is_gld_long_call = (
        t.symbol == "GLD" and t.kind == "OPT" and t.right == "C"
        and t.side == "BUY" and not t.closes_short
    )
    status = "FAIL" if is_gld_long_call else "PASS"
    return RuleResult("C5_no_gld_long_call", "BLOCK", status,
                      "GLD long call" if is_gld_long_call else "n/a", "forbidden",
                      "NEVER a long call (LEAP/PMCC) on GLD — standing rule after the 2026 GLD LEAP loss.")


def _c6_tfsa_no_short_puts(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    # Primitives-based, NOT tag-dependent: a TFSA short put must be caught even
    # if the operator omits/mis-tags `structure` or claims a roll (a put roll in
    # a TFSA is never legitimate — the account cannot hold the short being
    # rolled). The ONLY exemption is the position-verified sell-to-close of a
    # long put (closes_long).
    is_short_put_like = (
        t.structure in {"short-put", "put-credit-spread"}
        or (t.kind == "OPT" and t.side == "SELL" and t.right == "P" and not t.closes_long)
    )
    if t.account_type == "TFSA" and is_short_put_like:
        return RuleResult("C6_tfsa_no_short_puts", "BLOCK", "FAIL",
                          f"short put in TFSA ({t.account})", "covered-calls + long only",
                          "TFSA at IBKR Canada blocks short puts/spreads.")
    return RuleResult("C6_tfsa_no_short_puts", "BLOCK", "PASS",
                      "n/a", "covered-calls + long only", "OK.")


def _c7a_leap_hard_gate(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if t.structure not in {"leap-long", "pmcc-long"}:
        return RuleResult("C7a_leap_gate", "BLOCK", "SKIP", "n/a (not a long-premium leg)",
                          "delta>=0.80, extrinsic<=20%, <=10% NetLiq", "Only long LEAP/PMCC legs.")
    fails: list[str] = []
    low_conf = False
    if t.delta is not None:
        if abs(t.delta) < 0.80:
            fails.append(f"delta {abs(t.delta):.2f} < 0.80")
    else:
        low_conf = True
    if t.extrinsic_pct is not None:
        if t.extrinsic_pct > 0.20:
            fails.append(f"extrinsic {t.extrinsic_pct:.0%} > 20%")
    else:
        low_conf = True
    if t.size_notional is not None and ctx.net_liq > 0:
        sz = t.size_notional / ctx.net_liq
        if sz > 0.10:
            fails.append(f"size {sz:.0%} > 10% NetLiq")
    else:
        low_conf = True
    if fails:
        return RuleResult("C7a_leap_gate", "BLOCK", "FAIL", "; ".join(fails),
                          "delta>=0.80, extrinsic<=20%, <=10% NetLiq",
                          "LEAP/long-premium hard gate breached: " + "; ".join(fails))
    if low_conf:
        return RuleResult("C7a_leap_gate", "BLOCK", "SKIP", "greeks/size partially unknown",
                          "delta>=0.80, extrinsic<=20%, <=10% NetLiq", "Insufficient data to gate.", "LOW")
    return RuleResult("C7a_leap_gate", "BLOCK", "PASS", "delta/extrinsic/size within limits",
                      "delta>=0.80, extrinsic<=20%, <=10% NetLiq", "OK.")


def _c7b_leap_iv_rank(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if t.structure not in {"leap-long", "pmcc-long"}:
        return RuleResult("C7b_leap_iv_rank", "WARN", "SKIP", "n/a", "IV-rank < 50", "Only long legs.")
    if t.iv_rank is None:
        return RuleResult("C7b_leap_iv_rank", "WARN", "SKIP", "IV-rank unknown", "IV-rank < 50",
                          "No IV history — buying premium into rich IV is the risk; consider a debit spread.", "LOW")
    status = "FAIL" if t.iv_rank >= 50 else "PASS"
    return RuleResult("C7b_leap_iv_rank", "WARN", status, f"IV-rank {t.iv_rank:.0f}", "IV-rank < 50",
                      "IV-rank rich — prefer a debit spread over a long call." if status == "FAIL" else "OK.")


def _c8_short_put_entry_gate(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if t.structure != "short-put":
        return RuleResult("C8_short_put_gate", "BLOCK", "SKIP", "n/a",
                          "|delta| 0.20-0.35, 21-45 DTE, extrinsic >= 90% credit", "Only short puts.")
    fails: list[str] = []
    low_conf = False
    if t.delta is not None:
        ad = abs(t.delta)
        if not (0.20 <= ad <= 0.35):
            fails.append(f"delta {ad:.2f} outside 0.20-0.35")
    else:
        low_conf = True
    if t.dte is not None:
        if not (21 <= t.dte <= 45):
            fails.append(f"DTE {t.dte} outside 21-45")
    else:
        low_conf = True
    if t.extrinsic_pct_of_credit is not None:
        if t.extrinsic_pct_of_credit < 0.90:
            fails.append(f"extrinsic {t.extrinsic_pct_of_credit:.0%} < 90% of credit")
    else:
        low_conf = True
    if fails:
        return RuleResult("C8_short_put_gate", "BLOCK", "FAIL", "; ".join(fails),
                          "|delta| 0.20-0.35, 21-45 DTE, extrinsic >= 90% credit",
                          "Short-put entry gate breached: " + "; ".join(fails))
    if low_conf:
        return RuleResult("C8_short_put_gate", "BLOCK", "SKIP", "greeks/DTE partially unknown",
                          "|delta| 0.20-0.35, 21-45 DTE, extrinsic >= 90% credit",
                          "Insufficient data to gate.", "LOW")
    return RuleResult("C8_short_put_gate", "BLOCK", "PASS", "delta/DTE/extrinsic within gate",
                      "|delta| 0.20-0.35, 21-45 DTE, extrinsic >= 90% credit", "OK.")


def _c9_instrument_gate(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if not t.is_new_program:
        return RuleResult("C9_instrument_gate", "BLOCK", "SKIP", "n/a (not a new program)",
                          "spread<2%, width<=2%, weeklies, OI>1000, mark>=intrinsic",
                          "Only enforced when opening a new option program.")
    fails: list[str] = []
    low_conf = False
    checks: list[tuple[str, object, object]] = [
        ("spread", t.spread_pct, lambda v: v >= 0.02),
        ("strike-width", t.strike_width_pct, lambda v: v > 0.02),
        ("near-money OI", t.near_money_oi, lambda v: v <= 1000),
    ]
    for name, val, bad in checks:
        if val is None:
            low_conf = True
        elif bad(val):  # type: ignore[operator]
            fails.append(f"{name} {val}")
    if t.weeklies_listed is False:
        fails.append("no weeklies")
    elif t.weeklies_listed is None:
        low_conf = True
    if t.mark_below_intrinsic is True:
        fails.append("mark below intrinsic")
    elif t.mark_below_intrinsic is None:
        low_conf = True
    if fails:
        return RuleResult("C9_instrument_gate", "BLOCK", "FAIL", "; ".join(fails),
                          "spread<2%, width<=2%, weeklies, OI>1000, mark>=intrinsic",
                          "Instrument gate failed for a new option program: " + "; ".join(fails))
    if low_conf:
        return RuleResult("C9_instrument_gate", "BLOCK", "SKIP", "instrument metrics partially unknown",
                          "spread<2%, width<=2%, weeklies, OI>1000, mark>=intrinsic", "Missing data.", "LOW")
    return RuleResult("C9_instrument_gate", "BLOCK", "PASS", "all instrument checks pass",
                      "spread<2%, width<=2%, weeklies, OI>1000, mark>=intrinsic", "OK.")


def _w2_martingale(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if t.structure != "short-put":
        return RuleResult("W2_martingale", "BLOCK", "SKIP", "n/a", "no new short put <=30d after a loss",
                          "Only short puts.")
    if t.losing_put_close_same_underlying_30d is None:
        return RuleResult("W2_martingale", "BLOCK", "SKIP", "journal history unknown",
                          "no new short put <=30d after a loss", "Journal not available.", "LOW")
    if t.losing_put_close_same_underlying_30d:
        return RuleResult("W2_martingale", "BLOCK", "FAIL", f"losing {t.symbol} put closed <=30d ago",
                          "no new short put <=30d after a loss",
                          "Martingale reflex: never answer a losing put closure with a new put in the same underlying within 30 days.")
    return RuleResult("W2_martingale", "BLOCK", "PASS", "no recent losing close", "OK", "OK.")


def _w5_written_exit(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if (t.is_roll or t.closes_long or t.closes_short
            or t.structure in {"close", "roll-close"}):
        return RuleResult("W5_written_exit", "BLOCK", "SKIP", "n/a (not a new position)",
                          "written exit required", "Closes/rolls don't need a new exit.")
    has_exit = bool(t.written_exit and t.written_exit.strip())
    status = "PASS" if has_exit else "FAIL"
    return RuleResult("W5_written_exit", "BLOCK", status,
                      "exit recorded" if has_exit else "NO written exit",
                      "written exit required day-one",
                      "Every new position needs a written exit (trim ladder or trailing stop) before transmit.")


# ----------------------------------------------------------------- WARN rules

def _w1_opportunistic_roll(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if not (t.is_roll and t.rolled_short_is_working):
        return RuleResult("W1_opportunistic_roll", "WARN", "PASS", "n/a", "hold working shorts", "OK.")
    return RuleResult("W1_opportunistic_roll", "WARN", "FAIL", "rolling a working (OTM, profitable, 0.20-0.35) short",
                      "hold to expiration unless defensive",
                      "Opportunistic roll: you give up 70%+ of remaining extrinsic. Show the cycle math before transmitting.")


def _w3_anti_whipsaw(ctx: EvalContext) -> RuleResult:
    t = ctx.trade
    if not (t.is_roll and t.rolled_short_breached_within_10d):
        return RuleResult("W3_anti_whipsaw", "WARN", "PASS", "n/a", "close once or hold; never roll", "OK.")
    return RuleResult("W3_anti_whipsaw", "WARN", "FAIL", "rolling a short breached within 10d of entry",
                      "close once or hold; never roll",
                      "Anti-whipsaw: a short breached within 10 days of entry gets closed once or held — not rolled.")


def _w4_hedge_floor(ctx: EvalContext) -> RuleResult:
    nl = ctx.net_liq
    if nl <= 0:
        return RuleResult("W4_hedge_floor", "WARN", "SKIP", "NetLiq unknown",
                          "hedge sleeves > 25%", "Cannot size sleeves against NetLiq.", "LOW")
    by_factor: dict[str, float] = {}
    for p in ctx.positions:
        if p.delta_notional is not None:
            by_factor[p.factor] = by_factor.get(p.factor, 0.0) + abs(p.delta_notional)
    big = {f: v for f, v in by_factor.items() if v / nl > 0.25 and f not in {"index"}}
    if not big:
        return RuleResult("W4_hedge_floor", "WARN", "PASS", "no sleeve > 25% NetLiq", "hedge sleeves > 25%", "OK.")
    names = ", ".join(f"{f} {v/nl:.0%}" for f, v in big.items())
    return RuleResult("W4_hedge_floor", "WARN", "FAIL", f"unhedged sleeve(s): {names}",
                      "rolling put spreads >= 25% of sleeve delta",
                      "Standing hedge floor: any sleeve > 25% of NetLiq carries put-spread protection. Confirm the hedge exists.")


def _w6_revenge(ctx: EvalContext) -> RuleResult:
    # Fires when closing a GREEN position (mark > entry, adapter-derived from
    # the covered position's avg_cost) — the "cost recovery is NOT a sell
    # signal" / green-screen reflex. None = unknown -> pass conservatively.
    t = ctx.trade
    is_close = t.closes_long or t.structure in {"close", "roll-close"}
    if is_close and t.closing_profitable_position is True:
        return RuleResult("W6_revenge", "WARN", "FAIL", "closing a GREEN position",
                          "trim on SIZE/trigger, not on a green screen",
                          "Cost recovery is not a sell signal. Is this a written risk-control trigger or a feeling?")
    return RuleResult("W6_revenge", "WARN", "PASS", "n/a", "trim on size/trigger", "OK.")


_RULES = [
    _c1_factor_cap, _c2_per_underlying_assignment, _c3_aggregate_assignment,
    _c4_no_short_over_60dte, _c5_no_gld_long_call, _c6_tfsa_no_short_puts,
    _c7a_leap_hard_gate, _c7b_leap_iv_rank, _c8_short_put_entry_gate, _c9_instrument_gate,
    _w2_martingale, _w5_written_exit,
    _w1_opportunistic_roll, _w3_anti_whipsaw, _w4_hedge_floor, _w6_revenge,
]


def intent_hash(t: ProposedTrade) -> str:
    core = {
        "symbol": t.symbol, "kind": t.kind, "right": t.right, "side": t.side,
        "quantity": t.quantity, "structure": t.structure, "account": t.account,
    }
    blob = json.dumps(core, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def order_key(
    *, symbol: str, kind: str, side: str, quantity: float,
    right: str | None = None, strike: float | None = None,
    expiry: str | None = None, account: str | None = None,
    limit_price: float | None = None, order_type: str | None = None,
) -> str:
    """Stable content key over the order PRIMITIVES that exist at BOTH the
    standalone-check site and the transmit site (an OrderRequest carries these;
    `structure` is enrichment-derived and is deliberately excluded). Price IS
    included: the C8 'extrinsic >= 90% of credit' floor is price-dependent, so a
    re-priced order must force a fresh check rather than reuse a stale clearance.
    """
    core = {
        "symbol": symbol.upper(), "kind": kind.upper(),
        "right": (right or "").upper() or None,
        "strike": float(strike) if strike is not None else None,
        "expiry": expiry or None, "side": side.upper(),
        "quantity": float(quantity),
        "account": (account or "").upper() or None,
        "limit_price": round(float(limit_price), 4) if limit_price is not None else None,
        "order_type": (order_type or "").upper() or None,
    }
    blob = json.dumps(core, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def combo_key(
    *, legs: list[tuple[str, int, int]], side: str, quantity: float,
    account: str | None = None, limit_price: float | None = None,
    order_type: str | None = None, symbol: str | None = None,
) -> str:
    """Content key for a multi-leg BAG order. Legs are (action, conId, ratio)
    tuples, canonicalized (sorted, normalized) so leg ORDER in the CLI string
    never changes the key. Everything price/size-relevant is bound, exactly
    like order_key for single legs."""
    canon = sorted((a.upper(), int(c), int(r)) for a, c, r in legs)
    core = {
        "legs": canon,
        "symbol": (symbol or "").upper() or None,
        "side": side.upper(),
        "quantity": float(quantity),
        "account": (account or "").upper() or None,
        "limit_price": round(float(limit_price), 4) if limit_price is not None else None,
        "order_type": (order_type or "").upper() or None,
    }
    blob = json.dumps(core, sort_keys=True).encode()
    return "BAG" + hashlib.sha256(blob).hexdigest()[:16]


def evaluate(ctx: EvalContext) -> ConstitutionVerdict:
    checks = [rule(ctx) for rule in _RULES]
    decision: Decision = "PASS"
    if any(c.severity == "BLOCK" and c.status == "FAIL" for c in checks):
        decision = "BLOCK"
    elif any(c.severity == "WARN" and c.status == "FAIL" for c in checks):
        decision = "WARN"
    return ConstitutionVerdict(decision=decision, checks=checks, intent_hash=intent_hash(ctx.trade))

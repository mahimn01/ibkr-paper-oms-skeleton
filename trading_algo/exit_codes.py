"""Enumerated exit codes for agent-driven control flow.

A CLI consumer — especially an AI agent — needs to branch on *why* a command
failed before it reads any output. Collapsing every failure into `1` forces
the agent to string-match stderr, which is slow, error-prone, and gives the
model yet another free-text parsing problem.

Conventions drawn from sysexits.h (BSD), kubectl, and git. The IBKR
classifier maps ib_async error codes + our own exception hierarchy into
this table.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Codes
# ---------------------------------------------------------------------------

OK = 0
"""Success. Side effects (if any) completed as described."""

GENERIC = 1
"""Non-specific failure. Prefer a specific code below when possible.

Also used for partial successes (e.g. `cancel-all` where some cancels
succeeded and others failed)."""

USAGE = 2
"""Bad invocation: missing `--yes`, unknown flag, wrong enum value.
argparse already exits 2 for unknown flags — we align."""

VALIDATION = 3
"""Pre-flight client-side validation rejected the request. No API call
was made. Agent should inspect `error.field_errors` and retry with fixed
params. Examples: negative quantity, LMT without limit_price, STP
without stop_price."""

HARD_REJECT = 4
"""IBKR OMS rejected the request. Never retry. Examples: order quantity
too large, outside-RTH on an instrument that doesn't allow it, contract
not found, insufficient buying power. Maps to `IBKRDependencyError`
surfaced with an ib_async errorCode in the 200-range."""

AUTH = 5
"""Gateway not connected / auth invalid. Corresponds to
`IBKRConnectionError` on initial handshake + errorCode 1100 (connection
lost) + errorCode 502/504 (not connected). Agent must restart
TWS/Gateway session."""

PERMISSION = 6
"""Feature not available to this account. Examples: requesting market
data without a subscription, news providers not entitled. Corresponds
to ib_async errorCode 354 (requested market data is not subscribed)."""

LEASE = 10
"""Another agent/process holds the trading lease. Retry after backoff.
Reserved for future multi-agent coordination primitives."""

HALTED = 11
"""Trading is administratively halted via the `halt` command. Resume
with `trading-algo resume --confirm-resume`. All write commands refuse
while halted."""

OUT_OF_WINDOW = 12
"""Order attempted outside the configured live-trade window. Reserved
for when T3 adds market-hours risk checks."""

MARKET_CLOSED = 13
"""Regular session is closed for the target exchange. Retry at next
market open."""

UNAVAILABLE = 69
"""Upstream service unavailable. Corresponds to ib_async errorCode 1100
(connectivity lost), 1102 (restored), `IBKRConnectionError`,
`IBKRCircuitOpenError`. Retryable with backoff."""

INTERNAL = 70
"""Uncaught exception inside trading-algo (our bug). Also used for
ib_async errors we haven't classified yet."""

TRANSIENT = 75
"""Transient failure we've classified: `IBKRRateLimitError`, ib_async
errorCode 162 (historical data pacing violation), 165 (historical
pacing), network timeouts. Retryable with backoff."""

TIMEOUT = 124
"""A wait-for-state call hit its deadline. Matches coreutils
`timeout(1)`. The operation may still be in progress — poll separately."""

SIGINT = 130
"""User interrupt (Ctrl+C). Standard Unix 128 + SIGINT(2)."""


ALL_CODES = frozenset({
    OK, GENERIC, USAGE, VALIDATION, HARD_REJECT, AUTH, PERMISSION,
    LEASE, HALTED, OUT_OF_WINDOW, MARKET_CLOSED,
    UNAVAILABLE, INTERNAL, TRANSIENT, TIMEOUT, SIGINT,
})


@dataclass(frozen=True)
class ClassifiedError:
    """Result of classifying an exception for agent consumption."""
    exit_code: int
    error_code: str
    retryable: bool


# ---------------------------------------------------------------------------
# Exception-class-name mapping
# ---------------------------------------------------------------------------

# Matching on class name (not isinstance) lets us handle both our own
# exceptions and IBKR/ib_async exceptions without a hard import dependency
# on ib_insync/ib_async at module-load time.
_EXCEPTION_MAP: dict[str, ClassifiedError] = {
    # Our own IBKR wrappers.
    "IBKRConnectionError": ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),
    "IBKRCircuitOpenError": ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),
    "IBKRRateLimitError": ClassifiedError(TRANSIENT, "TRANSIENT", retryable=True),
    "IBKRDependencyError": ClassifiedError(INTERNAL, "INTERNAL", retryable=False),
    "IBKROrderbookLookupError": ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),
    # ib_async / ib_insync exception classes — names we might see.
    "ConnectionError": ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),
    "TimeoutError": ClassifiedError(TRANSIENT, "TRANSIENT", retryable=True),
    # Our own lifecycle + config.
    "HaltActive": ClassifiedError(HALTED, "HALTED", retryable=False),
    "EnvParseError": ClassifiedError(USAGE, "USAGE", retryable=False),
    "ModificationLimitExceeded": ClassifiedError(HARD_REJECT, "HARD_REJECT", retryable=False),
    "RiskViolation": ClassifiedError(VALIDATION, "VALIDATION", retryable=False),
    "ConstitutionViolation": ClassifiedError(VALIDATION, "VALIDATION", retryable=False),
}


# ib_async / ib_insync surfaces errors via `error` events on the IB object,
# with a numeric errorCode. If any code reraises them as a Python exception,
# the exception message often includes "errorCode=NNN" — so we also pattern-
# match when needed. This mapping covers the well-known ones.
_IBKR_ERROR_CODE_MAP: dict[int, ClassifiedError] = {
    # Connectivity.
    1100: ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),   # connectivity lost
    1101: ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),   # connectivity restored, data lost
    1102: ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),   # connectivity restored
    1300: ClassifiedError(UNAVAILABLE, "UNAVAILABLE", retryable=True),   # socket port reset
    502:  ClassifiedError(AUTH, "AUTH", retryable=False),                # not connected
    504:  ClassifiedError(AUTH, "AUTH", retryable=False),                # not connected
    # Contract / validation.
    200: ClassifiedError(VALIDATION, "VALIDATION", retryable=False),     # no security definition
    201: ClassifiedError(HARD_REJECT, "HARD_REJECT", retryable=False),   # order rejected
    202: ClassifiedError(OK, "OK", retryable=False),                     # order cancelled (informational)
    203: ClassifiedError(HARD_REJECT, "HARD_REJECT", retryable=False),   # security not available/allowed
    103: ClassifiedError(HARD_REJECT, "HARD_REJECT", retryable=False),   # duplicate order id
    # Historical pacing.
    162: ClassifiedError(TRANSIENT, "TRANSIENT", retryable=True),
    165: ClassifiedError(TRANSIENT, "TRANSIENT", retryable=True),
    420: ClassifiedError(TRANSIENT, "TRANSIENT", retryable=True),        # pacing
    # Permissions / subscriptions.
    354: ClassifiedError(PERMISSION, "PERMISSION", retryable=False),     # market data not subscribed
    430: ClassifiedError(PERMISSION, "PERMISSION", retryable=False),
    10197: ClassifiedError(PERMISSION, "PERMISSION", retryable=False),   # news not subscribed
}


def _extract_ib_error_code(exc: BaseException) -> int | None:
    """Best-effort extraction of `errorCode` from an exception.

    ib_async attaches `errorCode` to many of its error objects; other
    wrappers stringify as ``error 1100, reqId 0: Connectivity between
    IB and TWS has been lost.`` We try attribute first, then string scan.
    """
    code = getattr(exc, "errorCode", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    # String fallback: look for "error NNN" or "errorCode=NNN" shapes.
    msg = str(exc)
    import re
    match = re.search(r"\berror(?:Code)?[\s=:]+(\d{2,5})\b", msg, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_exception(exc: BaseException) -> ClassifiedError:
    """Return the exit-code + error-code + retryable classification.

    Priority:
    1. Non-Exception `BaseException` (KeyboardInterrupt, SystemExit).
    2. Class-name lookup in `_EXCEPTION_MAP` — our own classes.
    3. ib_async `errorCode` attribute/string scan → `_IBKR_ERROR_CODE_MAP`.
    4. ValueError / TypeError → VALIDATION.
    5. Transient message markers (timeout, 5xx, rate limit).
    6. Fallback: INTERNAL (safe — do not retry).
    """
    if isinstance(exc, KeyboardInterrupt):
        return ClassifiedError(SIGINT, "SIGINT", retryable=False)
    if isinstance(exc, SystemExit):
        code = int(exc.code) if isinstance(exc.code, int) else GENERIC
        if isinstance(exc.code, str):
            return ClassifiedError(USAGE, "USAGE", retryable=False)
        return ClassifiedError(code, "GENERIC", retryable=False)

    name = type(exc).__name__
    if name in _EXCEPTION_MAP:
        return _EXCEPTION_MAP[name]

    ib_code = _extract_ib_error_code(exc)
    if ib_code is not None and ib_code in _IBKR_ERROR_CODE_MAP:
        return _IBKR_ERROR_CODE_MAP[ib_code]

    if isinstance(exc, (ValueError, TypeError)):
        return ClassifiedError(VALIDATION, "VALIDATION", retryable=False)

    msg = str(exc).lower()
    transient_markers = (
        "timeout", "timed out", "connection reset",
        "500", "502", "503", "504",
        "rate limit", "too many requests", "pacing",
    )
    if any(m in msg for m in transient_markers):
        return ClassifiedError(TRANSIENT, "TRANSIENT", retryable=True)

    return ClassifiedError(INTERNAL, "INTERNAL", retryable=False)


def exit_code_name(code: int) -> str:
    """Reverse lookup: int → constant name. For suggested_action output."""
    for name, value in globals().items():
        if name.isupper() and value == code and name != "ALL_CODES":
            return name
    return f"UNKNOWN_{code}"

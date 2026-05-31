"""Execution policy + pending-order machinery for the backtest engine.

Look-ahead protection (PLAN.md §2.4):

    The legacy BacktestEngine generates a signal on bar T's close and fills
    at bar T's close — same-bar look-ahead. Default policy here is
    NEXT_BAR_OPEN: signals are *queued* on bar T and *filled* at bar T+1's
    open. This is the standard institutional default.

    SAME_BAR_CLOSE remains available as an explicit opt-in for strategies
    that genuinely fill on the close (e.g. MOC strategies). Validator emits
    a warning when this policy is selected, since it inflates Sharpe by
    ~10-30% on most strategies.

    NEXT_BAR_VWAP fills at bar T+1's VWAP for a more conservative estimate
    of average entry price during high-volatility opens.

Design:
    PendingOrder is a tiny container: side, qty, intent metadata, queued
    at bar T. The engine drains the queue at the start of bar T+1 and
    converts each pending into a fill priced according to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ExecutionPolicy(str, Enum):
    """How a signal generated on bar T is converted to a fill price."""
    SAME_BAR_CLOSE = "same_bar_close"   # legacy / opt-in
    NEXT_BAR_OPEN  = "next_bar_open"    # default
    NEXT_BAR_VWAP  = "next_bar_vwap"    # conservative


@dataclass
class PendingOrder:
    """A signal queued on bar T awaiting fill on bar T+1.

    Carries enough metadata to recreate the trade at fill time without
    holding a reference to the original signal object.
    """
    symbol: str
    side: str              # "BUY" / "SELL"
    quantity: float
    queued_at: datetime
    direction: int         # +1 for long, -1 for short
    metadata: dict[str, Any]
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None


def fill_price_for_policy(
    policy: ExecutionPolicy,
    *,
    bar_open:  float,
    bar_close: float,
    bar_vwap:  float | None = None,
) -> float:
    """Compute the paper fill price for the bar at which a pending order
    is being executed under `policy`.

    Falls back to bar_open if VWAP is requested but unavailable.
    """
    if policy is ExecutionPolicy.SAME_BAR_CLOSE:
        return bar_close
    if policy is ExecutionPolicy.NEXT_BAR_VWAP:
        return bar_vwap if bar_vwap and bar_vwap > 0 else bar_open
    # NEXT_BAR_OPEN
    return bar_open


def policy_introduces_lookahead(policy: ExecutionPolicy) -> bool:
    """Lookahead-safe policies fill at bar T+1; SAME_BAR_CLOSE does not."""
    return policy is ExecutionPolicy.SAME_BAR_CLOSE

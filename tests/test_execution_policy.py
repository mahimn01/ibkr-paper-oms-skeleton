"""Tests for ExecutionPolicy + pending-order machinery."""

from __future__ import annotations

from datetime import datetime, timezone

from trading_algo.backtest_v2.execution_policy import (
    ExecutionPolicy,
    PendingOrder,
    fill_price_for_policy,
    policy_introduces_lookahead,
)


def test_same_bar_close_uses_close() -> None:
    p = fill_price_for_policy(
        ExecutionPolicy.SAME_BAR_CLOSE,
        bar_open=100.0, bar_close=101.0, bar_vwap=100.5,
    )
    assert p == 101.0


def test_next_bar_open_uses_open() -> None:
    p = fill_price_for_policy(
        ExecutionPolicy.NEXT_BAR_OPEN,
        bar_open=100.0, bar_close=101.0, bar_vwap=100.5,
    )
    assert p == 100.0


def test_next_bar_vwap_prefers_vwap() -> None:
    p = fill_price_for_policy(
        ExecutionPolicy.NEXT_BAR_VWAP,
        bar_open=100.0, bar_close=101.0, bar_vwap=100.5,
    )
    assert p == 100.5


def test_next_bar_vwap_falls_back_to_open() -> None:
    p = fill_price_for_policy(
        ExecutionPolicy.NEXT_BAR_VWAP,
        bar_open=100.0, bar_close=101.0, bar_vwap=None,
    )
    assert p == 100.0
    p2 = fill_price_for_policy(
        ExecutionPolicy.NEXT_BAR_VWAP,
        bar_open=100.0, bar_close=101.0, bar_vwap=0.0,
    )
    assert p2 == 100.0


def test_only_same_bar_close_introduces_lookahead() -> None:
    assert policy_introduces_lookahead(ExecutionPolicy.SAME_BAR_CLOSE) is True
    assert policy_introduces_lookahead(ExecutionPolicy.NEXT_BAR_OPEN) is False
    assert policy_introduces_lookahead(ExecutionPolicy.NEXT_BAR_VWAP) is False


def test_pending_order_carries_metadata() -> None:
    pending = PendingOrder(
        symbol="AAPL",
        side="BUY",
        quantity=100,
        queued_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        direction=+1,
        metadata={"strategy": "short_term_reversal", "reason": "z<-2"},
        stop_loss=95.0,
        take_profit=110.0,
    )
    assert pending.metadata["strategy"] == "short_term_reversal"
    assert pending.stop_loss == 95.0
    assert pending.take_profit == 110.0


def test_execution_policy_string_values() -> None:
    # Round-trip stability for serialization to BacktestConfig.
    assert ExecutionPolicy("next_bar_open") is ExecutionPolicy.NEXT_BAR_OPEN
    assert ExecutionPolicy("same_bar_close") is ExecutionPolicy.SAME_BAR_CLOSE
    assert ExecutionPolicy("next_bar_vwap") is ExecutionPolicy.NEXT_BAR_VWAP

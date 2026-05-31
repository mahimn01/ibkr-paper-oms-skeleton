"""Tests that RiskManager state survives process restarts."""

from __future__ import annotations

import time
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading_algo.broker.base import AccountSnapshot
from trading_algo.risk import RiskLimits, RiskManager


def _account(net_liq: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        account="DU000001",
        values={
            "NetLiquidation":     net_liq,
            "GrossPositionValue": 0.0,
            "MaintMarginReq":     0.0,
        },
        timestamp_epoch_s=time.time(),
    )


# ----------------------------------------------------------------- session NL


def test_session_start_persisted_across_restart(tmp_path: Path) -> None:
    db = tmp_path / "risk.sqlite"
    rm = RiskManager(RiskLimits(max_daily_loss=1000), db_path=db)
    # Initial save via _update_session_start (the public path is validate()).
    rm._update_session_start(_account(105_000.0))
    assert rm._session_start_net_liq == pytest.approx(105_000.0)

    # Simulate process restart.
    rm2 = RiskManager(RiskLimits(max_daily_loss=1000), db_path=db)
    assert rm2._session_start_net_liq == pytest.approx(105_000.0)


def test_session_start_resets_on_new_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "risk.sqlite"
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, t=None: "2026-05-01" if "%Y-%m-%d" == fmt else "x")
    rm = RiskManager(RiskLimits(max_daily_loss=1000), db_path=db)
    rm._update_session_start(_account(105_000.0))
    assert rm._session_start_net_liq == pytest.approx(105_000.0)

    # Date rolls forward.
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, t=None: "2026-05-02" if "%Y-%m-%d" == fmt else "x")
    rm2 = RiskManager(RiskLimits(max_daily_loss=1000), db_path=db)
    # Stale session NL should not carry over.
    assert rm2._session_start_net_liq is None


# ----------------------------------------------------------------- orders/day


def test_orders_today_persisted_across_restart(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "risk.sqlite"
    fixed = "2026-05-01"
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, t=None: fixed if "%Y-%m-%d" == fmt else "x")

    limits = RiskLimits(max_orders_per_day=10)
    rm = RiskManager(limits, db_path=db)
    # Bump the counter three times.
    for _ in range(3):
        rm._check_and_count_daily_orders()
    assert rm.orders_today_count == 3

    # Restart.
    rm2 = RiskManager(limits, db_path=db)
    assert rm2.orders_today_count == 3

    # One more bump persists.
    rm2._check_and_count_daily_orders()
    rm3 = RiskManager(limits, db_path=db)
    assert rm3.orders_today_count == 4


def test_orders_today_resets_on_new_day(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "risk.sqlite"
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, t=None: "2026-05-01" if "%Y-%m-%d" == fmt else "x")
    limits = RiskLimits(max_orders_per_day=10)
    rm = RiskManager(limits, db_path=db)
    rm._check_and_count_daily_orders()
    rm._check_and_count_daily_orders()
    assert rm.orders_today_count == 2

    # Date rolls.
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, t=None: "2026-05-02" if "%Y-%m-%d" == fmt else "x")
    rm2 = RiskManager(limits, db_path=db)
    assert rm2.orders_today_count == 0  # fresh day


def test_orders_per_day_cap_persists_across_restart(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "risk.sqlite"
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, t=None: "2026-05-01" if "%Y-%m-%d" == fmt else "x")
    limits = RiskLimits(max_orders_per_day=3)
    rm = RiskManager(limits, db_path=db)
    rm._check_and_count_daily_orders()
    rm._check_and_count_daily_orders()
    rm._check_and_count_daily_orders()

    rm2 = RiskManager(limits, db_path=db)
    # Cap should be enforced even after restart — counter is at 3, max is 3.
    from trading_algo.risk import RiskViolation
    with pytest.raises(RiskViolation):
        rm2._check_and_count_daily_orders()


# ----------------------------------------------------------------- without DB

def test_risk_manager_works_without_db_path() -> None:
    """Backwards compatibility: db_path=None preserves the legacy behavior."""
    rm = RiskManager(RiskLimits(max_orders_per_day=10))
    rm._check_and_count_daily_orders()
    assert rm.orders_today_count == 1


def test_risk_state_table_created_when_init_schema_true(tmp_path: Path) -> None:
    db = tmp_path / "risk.sqlite"
    RiskManager(RiskLimits(), db_path=db, init_schema=True)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_state'"
        ).fetchall()
    assert rows

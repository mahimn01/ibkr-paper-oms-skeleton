"""Tests for AdjustmentEngine."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_algo.data.corporate_actions import AdjustmentEngine
from trading_algo.data.pit_store import Bar, PITStore


@pytest.fixture
def store(tmp_path: Path) -> PITStore:
    s = PITStore(tmp_path / "pit")
    sid = s.upsert_security("AAPL", list_date=date(1980, 12, 12))
    # AAPL splits.
    s.add_split(sid, date(2014, 6, 9), 7.0)
    s.add_split(sid, date(2020, 8, 31), 4.0)
    return s


def test_factor_unity_when_no_split_in_window(store: PITStore) -> None:
    eng = AdjustmentEngine(store)
    sid = store.resolve_ticker("AAPL", date(2024, 1, 1))
    assert sid is not None
    # No splits between 2021-01-01 and 2022-01-01 -> factor 1.0
    assert eng.factor(sid, date(2021, 1, 1), date(2022, 1, 1)) == pytest.approx(1.0)


def test_factor_for_pre_split_bar_after_4_for_1(store: PITStore) -> None:
    """A 2019-01-01 bar viewed as of 2021-01-01 has the 2020 4:1 split
    in between, so factor = 1/4."""
    eng = AdjustmentEngine(store)
    sid = store.resolve_ticker("AAPL", date(2024, 1, 1))
    f = eng.factor(sid, date(2019, 1, 1), date(2021, 1, 1))
    assert f == pytest.approx(0.25)


def test_factor_for_pre_2014_bar_after_both_splits(store: PITStore) -> None:
    """A 2010 bar viewed as of 2024 has both 7:1 (2014) and 4:1 (2020)
    splits in between -> factor = 1/(7*4) = 1/28."""
    eng = AdjustmentEngine(store)
    sid = store.resolve_ticker("AAPL", date(2024, 1, 1))
    f = eng.factor(sid, date(2010, 1, 1), date(2024, 1, 1))
    assert f == pytest.approx(1.0 / 28.0)


def test_adjust_series_adjusts_pre_split_close(store: PITStore) -> None:
    eng = AdjustmentEngine(store)
    sid = store.resolve_ticker("AAPL", date(2024, 1, 1))
    bars = [
        Bar(symbol="AAPL", date=date(2010, 1, 4),
            open=210, high=215, low=208, close=214, volume=1_000_000),
        Bar(symbol="AAPL", date=date(2024, 1, 4),
            open=180, high=182, low=178, close=181, volume=1_000_000),
    ]
    adjusted = eng.adjust_series(sid, bars, as_of=date(2024, 1, 5))
    # 2010 bar: split-adjusted close = 214 / 28 = 7.6428...
    assert adjusted[0].adj_close == pytest.approx(214.0 / 28.0)
    assert adjusted[0].adjustment_factor == pytest.approx(1.0 / 28.0)
    # 2024 bar: no future splits, factor = 1, adj_close == close.
    assert adjusted[1].adj_close == pytest.approx(181.0)
    assert adjusted[1].adjustment_factor == pytest.approx(1.0)


def test_factor_zero_when_as_of_before_bar(store: PITStore) -> None:
    eng = AdjustmentEngine(store)
    sid = store.resolve_ticker("AAPL", date(2024, 1, 1))
    # as_of < bar_date is a degenerate query — engine returns 1.0 (no-op).
    assert eng.factor(sid, date(2020, 1, 1), date(2019, 1, 1)) == pytest.approx(1.0)

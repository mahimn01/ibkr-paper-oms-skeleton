"""Tests for the bitemporal point-in-time store."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from trading_algo.data.pit_store import Bar, PITStore


@pytest.fixture
def store(tmp_path: Path) -> PITStore:
    return PITStore(tmp_path / "pit")


# --------------------------------------------------------------------- securities

def test_upsert_security_returns_internal_id(store: PITStore) -> None:
    sid = store.upsert_security("AAPL", cusip="037833100")
    assert sid > 0
    # Idempotent on (ticker, currently active) pair.
    sid2 = store.upsert_security("AAPL", cusip="037833100")
    assert sid == sid2


def test_resolve_ticker_returns_internal_id_at_date(store: PITStore) -> None:
    sid = store.upsert_security("AAPL", list_date=date(2000, 1, 1))
    assert store.resolve_ticker("AAPL", date(2020, 1, 1)) == sid


def test_resolve_ticker_returns_none_before_listing(store: PITStore) -> None:
    store.upsert_security("AAPL", list_date=date(2010, 1, 1))
    assert store.resolve_ticker("AAPL", date(2005, 1, 1)) is None


def test_record_ticker_change_preserves_history(store: PITStore) -> None:
    sid = store.upsert_security("FB", list_date=date(2012, 5, 18))
    store.record_ticker_change(sid, "META", date(2022, 6, 9))
    # Pre-change: FB resolves; META does not.
    assert store.resolve_ticker("FB", date(2020, 1, 1)) == sid
    assert store.resolve_ticker("META", date(2020, 1, 1)) is None
    # Post-change: META resolves; FB does not.
    assert store.resolve_ticker("META", date(2024, 1, 1)) == sid
    assert store.resolve_ticker("FB", date(2024, 1, 1)) is None


# --------------------------------------------------------------------- splits / divs

def test_split_round_trip(store: PITStore) -> None:
    sid = store.upsert_security("AAPL")
    store.add_split(sid, date(2020, 8, 31), 4.0)
    store.add_split(sid, date(2014, 6, 9), 7.0)
    splits = store.get_splits(sid)
    assert splits == [(date(2014, 6, 9), 7.0), (date(2020, 8, 31), 4.0)]


def test_split_rejects_zero_ratio(store: PITStore) -> None:
    sid = store.upsert_security("AAPL")
    with pytest.raises(ValueError):
        store.add_split(sid, date(2020, 8, 31), 0.0)


def test_dividend_round_trip(store: PITStore) -> None:
    sid = store.upsert_security("AAPL")
    store.add_dividend(sid, date(2024, 11, 8), 0.25, "regular")
    store.add_dividend(sid, date(2025, 2, 7), 0.26, "regular")
    divs = store.get_dividends(sid, types=("regular",))
    assert len(divs) == 2
    assert divs[0][1] == pytest.approx(0.25)


def test_dividend_rejects_unknown_type(store: PITStore) -> None:
    sid = store.upsert_security("AAPL")
    with pytest.raises(ValueError):
        store.add_dividend(sid, date(2024, 11, 8), 0.25, "weird")


# --------------------------------------------------------------------- bars

def _bar(year: int, month: int, day: int, close: float = 100.0) -> Bar:
    return Bar(
        symbol="AAPL",
        date=date(year, month, day),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1_000_000,
    )


def test_write_and_read_bars(store: PITStore) -> None:
    bars = [_bar(2024, 1, d) for d in range(2, 8)]
    n = store.write_bars("AAPL", bars)
    assert n == 6
    out = store.read_bars("AAPL", date(2024, 1, 2), date(2024, 1, 7))
    assert len(out) == 6
    assert all(b.symbol == "AAPL" for b in out)
    assert [b.date for b in out] == [b.date for b in bars]


def test_write_bars_skips_duplicates_silently(store: PITStore) -> None:
    bars = [_bar(2024, 1, 2), _bar(2024, 1, 3)]
    assert store.write_bars("AAPL", bars) == 2
    # Second write of the same dates returns 0 (skipped).
    assert store.write_bars("AAPL", bars) == 0


def test_read_bars_filters_by_known_at(store: PITStore) -> None:
    """Bars written with a future `known_from` should not appear in past
    as-of queries."""
    far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)
    future_bar = Bar(
        symbol="AAPL",
        date=date(2024, 1, 2),
        open=100, high=101, low=99, close=100, volume=1,
        known_from=far_future,
        known_to=datetime(9999, 12, 31),
    )
    store.write_bars("AAPL", [future_bar])
    # Query at "today" — bar isn't visible yet because we recorded it
    # as known_from=2030.
    out = store.read_bars(
        "AAPL", date(2024, 1, 2), date(2024, 1, 2),
        as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert out == []


def test_restate_bar_appends_new_vintage(store: PITStore) -> None:
    """Restating a bar should preserve the original vintage."""
    original_known_from = datetime(2024, 1, 3, tzinfo=timezone.utc)
    bar = Bar(
        symbol="AAPL", date=date(2024, 1, 2),
        open=100, high=101, low=99, close=100.0, volume=1_000_000,
        known_from=original_known_from,
    )
    store.write_bars("AAPL", [bar])

    revised = Bar(
        symbol="AAPL", date=date(2024, 1, 2),
        open=100.5, high=101.5, low=99.5, close=100.5, volume=1_100_000,
    )
    restatement_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
    store.restate_bar("AAPL", revised, restatement_time=restatement_time)

    # Query as of restatement-1: original vintage.
    pre = store.read_bars(
        "AAPL", date(2024, 1, 2), date(2024, 1, 2),
        as_of=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )
    assert len(pre) == 1
    assert pre[0].close == pytest.approx(100.0)

    # Query as of restatement+1: revised vintage.
    post = store.read_bars(
        "AAPL", date(2024, 1, 2), date(2024, 1, 2),
        as_of=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )
    assert len(post) == 1
    assert post[0].close == pytest.approx(100.5)


# --------------------------------------------------------------------- index

def test_index_membership_lookup(store: PITStore) -> None:
    aapl = store.upsert_security("AAPL")
    leh = store.upsert_security("LEH", delist_date=date(2008, 9, 15),
                                delist_reason="bankrupt")
    store.add_index_membership("SP500", aapl, date(2000, 1, 1))
    store.add_index_membership("SP500", leh, date(1990, 1, 1),
                               removed_date=date(2008, 9, 15))

    with store._conn() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM index_membership WHERE index_name = 'SP500'"
        ).fetchone()
        assert row["n"] == 2

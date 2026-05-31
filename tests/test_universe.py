"""Tests for survivorship-bias-free UniverseResolver."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_algo.data.pit_store import PITStore
from trading_algo.data.universe import SurvivorshipBiasError, UniverseResolver


@pytest.fixture
def populated_store(tmp_path: Path) -> PITStore:
    store = PITStore(tmp_path / "pit")
    aapl = store.upsert_security("AAPL", list_date=date(1980, 12, 12))
    msft = store.upsert_security("MSFT", list_date=date(1986, 3, 13))
    leh  = store.upsert_security("LEH",
                                 list_date=date(1994, 5, 31),
                                 delist_date=date(2008, 9, 15),
                                 delist_reason="bankrupt")
    fb   = store.upsert_security("FB", list_date=date(2012, 5, 18))
    # Ticker change FB -> META
    store.record_ticker_change(fb, "META", date(2022, 6, 9))
    # Build SP500 membership timeline.
    store.add_index_membership("SP500", aapl, date(1982, 11, 30))
    store.add_index_membership("SP500", msft, date(1994, 6, 1))
    store.add_index_membership("SP500", leh,  date(1994, 9, 1),
                               removed_date=date(2008, 9, 15))
    store.add_index_membership("SP500", fb,   date(2013, 12, 23))
    return store


def test_literal_universe_returns_as_is() -> None:
    r = UniverseResolver()
    assert r.get_universe(["aapl", "msft"]) == ["AAPL", "MSFT"]


def test_dev_universe_requires_explicit_optin() -> None:
    r = UniverseResolver()
    with pytest.raises(SurvivorshipBiasError):
        r.get_universe("DEV_MEGACAP")
    out = r.get_universe("DEV_MEGACAP", allow_dev=True)
    assert "AAPL" in out


def test_index_universe_requires_as_of_date(populated_store: PITStore) -> None:
    r = UniverseResolver(store=populated_store)
    with pytest.raises(ValueError):
        r.get_universe("SP500")


def test_sp500_includes_lehman_before_bankruptcy(populated_store: PITStore) -> None:
    r = UniverseResolver(store=populated_store)
    members = r.get_universe("SP500", date(2007, 1, 1))
    assert "LEH" in members
    assert "AAPL" in members


def test_sp500_excludes_lehman_after_bankruptcy(populated_store: PITStore) -> None:
    r = UniverseResolver(store=populated_store)
    members = r.get_universe("SP500", date(2009, 1, 1))
    assert "LEH" not in members


def test_sp500_resolves_fb_meta_ticker_change(populated_store: PITStore) -> None:
    r = UniverseResolver(store=populated_store)
    pre = r.get_universe("SP500", date(2020, 1, 1))
    post = r.get_universe("SP500", date(2024, 1, 1))
    assert "FB" in pre and "META" not in pre
    assert "META" in post and "FB" not in post


def test_index_universe_without_store_raises(populated_store: PITStore) -> None:
    r = UniverseResolver()
    with pytest.raises(SurvivorshipBiasError):
        r.get_universe("SP500", date(2020, 1, 1))


def test_universe_timeline(populated_store: PITStore) -> None:
    r = UniverseResolver(store=populated_store)
    rows = r.get_universe_timeline("SP500", date(2007, 1, 1), date(2010, 1, 1))
    tickers = {t for _, _, t in rows}
    assert {"AAPL", "MSFT", "LEH"}.issubset(tickers)
    leh_row = [r for r in rows if r[2] == "LEH"][0]
    assert leh_row[1] == date(2008, 9, 15)

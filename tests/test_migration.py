"""Tests for the legacy JSON cache → PIT store migration."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from trading_algo.data.migration import (
    import_directory,
    import_file,
    reconcile,
    _parse_legacy_filename,
    _parse_bar_entry,
)
from trading_algo.data.pit_store import PITStore


def _write_legacy_file(
    cache: Path, symbol: str, bar_size: str,
    start: date, end: date,
    bars: list[dict],
) -> Path:
    name = f"{symbol}_{bar_size}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
    path = cache / name
    path.write_text(json.dumps({
        "symbol": symbol,
        "bar_size": bar_size,
        "bars": bars,
    }))
    return path


def _bars(start: date, n: int) -> list[dict]:
    return [
        {
            "timestamp": (start.replace(day=min(28, start.day + i))).isoformat(),
            "open": 100 + i,
            "high": 101 + i,
            "low":   99 + i,
            "close": 100 + i,
            "volume": 1_000_000,
        }
        for i in range(n)
    ]


# ----------------------------------------------------------------- filename parser


def test_parse_legacy_filename_compact_dates() -> None:
    sym, bs, s, e = _parse_legacy_filename("AAPL_5min_20240101_20240301.json")
    assert sym == "AAPL"
    assert bs == "5min"
    assert s == date(2024, 1, 1)
    assert e == date(2024, 3, 1)


def test_parse_legacy_filename_iso_dates() -> None:
    out = _parse_legacy_filename("MSFT_1day_2024-01-01_2024-03-01.json")
    assert out is not None
    sym, _, s, e = out
    assert sym == "MSFT"
    assert s == date(2024, 1, 1)
    assert e == date(2024, 3, 1)


def test_parse_legacy_filename_garbage_returns_none() -> None:
    assert _parse_legacy_filename("nope.json") is None
    assert _parse_legacy_filename("AAPL_5min.json") is None


# ----------------------------------------------------------------- bar parser


def test_parse_bar_entry_iso_timestamp() -> None:
    bar = _parse_bar_entry("AAPL", {
        "timestamp": "2024-01-02",
        "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100,
    })
    assert bar is not None
    assert bar.date == date(2024, 1, 2)
    assert bar.symbol == "AAPL"


def test_parse_bar_entry_epoch_timestamp() -> None:
    bar = _parse_bar_entry("AAPL", {
        "timestamp": 1704153600,    # 2024-01-02 some moment UTC
        "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100,
    })
    assert bar is not None


def test_parse_bar_entry_bad_returns_none() -> None:
    assert _parse_bar_entry("AAPL", {"open": 1}) is None  # no timestamp


# ----------------------------------------------------------------- end-to-end


def test_import_file_writes_bars(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    pit_root = tmp_path / "pit"
    store = PITStore(pit_root)

    p = _write_legacy_file(cache, "AAPL", "1day",
                           date(2024, 1, 1), date(2024, 1, 10),
                           _bars(date(2024, 1, 1), 5))
    result = import_file(store, p)
    assert result.error is None
    assert result.bars_in == 5
    assert result.bars_written == 5

    # Reading back produces 5 bars.
    out = store.read_bars("AAPL", date(2024, 1, 1), date(2024, 12, 31))
    assert len(out) == 5


def test_import_directory_aggregates(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    store = PITStore(tmp_path / "pit")
    _write_legacy_file(cache, "AAPL", "1day", date(2024, 1, 1), date(2024, 1, 10),
                       _bars(date(2024, 1, 1), 3))
    _write_legacy_file(cache, "MSFT", "1day", date(2024, 2, 1), date(2024, 2, 10),
                       _bars(date(2024, 2, 1), 4))
    report = import_directory(store, cache)
    assert report.files_processed == 2
    assert report.files_failed == 0
    assert report.bars_in == 7
    assert report.bars_written == 7
    md = report.render()
    assert "Files in:   2" in md


def test_import_garbage_file_fails_gracefully(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "garbage.json").write_text("{not json}")
    store = PITStore(tmp_path / "pit")
    report = import_directory(store, cache)
    assert report.files_failed >= 1


def test_reconcile_zero_mismatches_after_clean_import(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    store = PITStore(tmp_path / "pit")
    _write_legacy_file(cache, "AAPL", "1day", date(2024, 1, 1), date(2024, 1, 10),
                       _bars(date(2024, 1, 1), 7))
    import_directory(store, cache)
    recon = reconcile(store, cache, sample_per_file=3)
    assert recon.mismatched_pairs == 0
    assert recon.reconciled_pairs > 0


def test_migration_log_recorded(tmp_path: Path) -> None:
    """Each migration leaves an audit row in migration_log."""
    cache = tmp_path / "cache"
    cache.mkdir()
    store = PITStore(tmp_path / "pit")
    _write_legacy_file(cache, "AAPL", "1day", date(2024, 1, 1), date(2024, 1, 10),
                       _bars(date(2024, 1, 1), 3))
    import_directory(store, cache)
    with store._conn() as conn:
        rows = conn.execute("SELECT * FROM migration_log").fetchall()
    assert len(rows) == 1
    assert rows[0]["source"] == "legacy_json_cache"

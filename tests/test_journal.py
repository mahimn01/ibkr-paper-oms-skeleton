"""Tests for the trade journal: persistence, the W2/W5 bridges, the brief, and the CLI."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from trading_algo import cli
from trading_algo.journal_cli import cmd_journal, render_brief
from trading_algo.persistence import SqliteStore


def _store(tmp_path):
    return SqliteStore(str(tmp_path / "trading.sqlite3"))


# ---------------------------------------------------------------- persistence

def test_journal_roundtrip_and_open(tmp_path):
    s = _store(tmp_path)
    eid = s.log_journal_entry(
        underlying="IAU", thesis="gold rent via shares", written_exit="called-away OK @90",
        structure="covered-call", side="SELL", quantity=1, account="U1",
    )
    assert eid > 0
    opens = s.open_journal_entries()
    assert len(opens) == 1 and opens[0]["underlying"] == "IAU"
    assert opens[0]["defensive_state"] == "none" and opens[0]["status"] == "open"
    s.close()


def test_missing_written_exit_detected(tmp_path):
    s = _store(tmp_path)
    # written_exit is NOT NULL in schema, but an empty string must be flagged (W5 spirit)
    s.log_journal_entry(underlying="MP", thesis="rare earth", written_exit="  ")
    s.log_journal_entry(underlying="URA", thesis="uranium", written_exit="20% trail")
    missing = s.positions_missing_written_exit()
    assert len(missing) == 1 and missing[0]["underlying"] == "MP"
    s.close()


def test_recent_losing_put_close_w2_bridge(tmp_path):
    s = _store(tmp_path)
    eid = s.log_journal_entry(underlying="MP", thesis="wheel", written_exit="x",
                              structure="short-put", side="SELL", quantity=1)
    # not closed yet -> no recent loss
    assert s.recent_losing_put_close("MP", within_s=30 * 86400) is False
    # close it at a loss at t=1000
    s.update_journal_status(eid, status="closed", realized_pnl=-120.0, closed_ts_epoch_s=1000.0)
    # within 30d window of the close
    assert s.recent_losing_put_close("MP", within_s=30 * 86400, now=1000.0 + 5 * 86400) is True
    # outside the window
    assert s.recent_losing_put_close("MP", within_s=30 * 86400, now=1000.0 + 40 * 86400) is False
    # a winning close does NOT trigger the martingale flag
    s.update_journal_status(eid, status="closed", realized_pnl=+50.0, closed_ts_epoch_s=2000.0)
    assert s.recent_losing_put_close("MP", within_s=30 * 86400, now=2000.0 + 1 * 86400) is False
    s.close()


def test_retag_preserves_recorded_loss(tmp_path):
    # F3: re-tagging closed -> rolled without --realized-pnl must NOT null the
    # loss (the W2 martingale bridge depends on it).
    s = _store(tmp_path)
    eid = s.log_journal_entry(underlying="MP", thesis="wheel", written_exit="x",
                              structure="short-put")
    s.update_journal_status(eid, status="closed", realized_pnl=-120.0, closed_ts_epoch_s=1000.0)
    s.update_journal_status(eid, status="rolled")  # no pnl passed
    assert s.recent_losing_put_close("MP", within_s=30 * 86400, now=1000.0 + 86400) is True
    rows = s.journal_entries_by_underlying("MP")
    assert rows[0]["realized_pnl"] == -120.0 and rows[0]["closed_ts_epoch_s"] == 1000.0
    s.close()


def test_journal_update_missing_id_raises(tmp_path):
    # F8: updating a nonexistent entry must be loud, not a silent no-op.
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        s.update_journal_status(999, status="closed")
    with pytest.raises(ValueError, match="does not exist"):
        s.set_journal_defensive_state(999, "watch")
    s.close()


def test_defensive_state_update(tmp_path):
    s = _store(tmp_path)
    eid = s.log_journal_entry(underlying="MP", thesis="t", written_exit="x", structure="short-call")
    s.set_journal_defensive_state(eid, "watch:delta>0.40")
    assert s.open_journal_entries()[0]["defensive_state"] == "watch:delta>0.40"
    s.close()


# ---------------------------------------------------------------- brief renderer

def test_render_brief_empty():
    assert "no open positions" in render_brief([])


def test_render_brief_counts_defensive_and_missing():
    entries = [
        {"underlying": "IAU", "structure": "covered-call", "side": "SELL", "quantity": 1,
         "thesis": "gold rent", "written_exit": "@90", "defensive_state": "none"},
        {"underlying": "MP", "structure": "short-call", "side": "SELL", "quantity": 1,
         "thesis": "wheel", "written_exit": "roll@0.40", "defensive_state": "watch:delta>0.40"},
    ]
    out = render_brief(entries, missing_exit=2)
    assert "2 open position(s)" in out
    assert "UNRESOLVED DEFENSIVE TRIGGERS: 1" in out
    assert "POSITIONS MISSING WRITTEN EXIT: 2" in out
    assert "⚠ DEFENSIVE: watch:delta>0.40" in out


# ---------------------------------------------------------------- CLI end-to-end

def _run(argv: list[str]) -> tuple[int, str]:
    args = cli.build_parser().parse_args(argv)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = args.func(args)
    return rc, buf.getvalue()


def test_cli_journal_add_then_brief(tmp_path):
    db = str(tmp_path / "trading.sqlite3")
    rc, out = _run(["journal", "add", "--db-path", db, "--underlying", "IAU",
                    "--thesis", "gold rent", "--exit", "called-away @90",
                    "--structure", "covered-call"])
    assert rc == 0 and "recorded" in out
    rc, out = _run(["journal", "brief", "--db-path", db])
    assert rc == 0
    assert "1 open position(s)" in out and "IAU" in out and "gold rent" in out


def test_cli_journal_brief_no_db_is_failopen(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADING_DB_PATH", raising=False)
    rc, out = _run(["journal", "brief"])  # no --db-path, no env
    assert rc == 0  # must never break SessionStart
    assert "no trading_db_path" in out.lower()


def test_cli_journal_add_requires_fields(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    with pytest.raises(SystemExit):
        _run(["journal", "add", "--db-path", db, "--underlying", "IAU"])  # missing thesis/exit


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

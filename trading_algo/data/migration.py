"""Migrate the legacy JSON cache to the bitemporal PIT store.

Source schema (legacy `quant_core/data/ibkr_data_loader.py`):
    Files at <cache>/<symbol>_<bar_size>_<start>_<end>.json
    Body: {"symbol": "...", "bar_size": "...",
           "bars": [{"timestamp": "...", "open": ..., "high": ..., ...}, ...]}

Destination (PLAN.md §2.1):
    <pit_root>/bars/symbol={X}/year={Y}/data.parquet  (pyarrow + zstd)
    <pit_root>/meta.sqlite                            (securities, ticker_history)

Reconciliation:
    For each (symbol, date) tuple in the source, the migration writes one
    Bar to the PIT store. After ingest the script reads back a sample and
    confirms close-price equality (within float tolerance) on every
    overlapping (symbol, date). Any mismatch is reported.

CLI entry point: scripts/migrate_to_pit.py.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from trading_algo.data.pit_store import Bar, PITStore

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class FileImportResult:
    path: Path
    symbol: str
    bars_in: int
    bars_written: int
    error: Optional[str] = None


@dataclass
class MigrationReport:
    started_at: datetime
    finished_at: Optional[datetime]
    files_processed: int = 0
    files_failed: int = 0
    bars_in: int = 0
    bars_written: int = 0
    reconciled_pairs: int = 0
    mismatched_pairs: int = 0
    per_file: list[FileImportResult] = None

    def __post_init__(self) -> None:
        if self.per_file is None:
            self.per_file = []

    def render(self) -> str:
        finished = self.finished_at.isoformat() if self.finished_at else "running"
        lines = [
            "Legacy JSON cache → PIT store migration",
            "=" * 56,
            f"Started:    {self.started_at.isoformat()}",
            f"Finished:   {finished}",
            f"Files in:   {self.files_processed} (failed: {self.files_failed})",
            f"Bars in:    {self.bars_in}",
            f"Bars wrote: {self.bars_written}",
            f"Reconciled: {self.reconciled_pairs} pairs "
            f"(mismatched: {self.mismatched_pairs})",
        ]
        if self.files_failed:
            lines.append("")
            lines.append("Failures:")
            for r in self.per_file:
                if r.error:
                    lines.append(f"  - {r.path.name}: {r.error}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _parse_legacy_filename(name: str) -> tuple[str, str, Optional[date], Optional[date]] | None:
    """Best-effort filename parse. Returns (symbol, bar_size, start, end)
    or None if the format is unrecognised.

    Legacy formats vary; we accept:
        AAPL_5min_20240101_20240301.json
        AAPL_1day_20240101_20240301.json
        AAPL_daily_2024-01-01_2024-03-01.json
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    symbol = parts[0]
    bar_size = parts[1]
    try:
        start_raw, end_raw = parts[-2], parts[-1]
        start = _parse_date_loose(start_raw)
        end = _parse_date_loose(end_raw)
    except Exception:
        return None
    return symbol, bar_size, start, end


def _parse_date_loose(s: str) -> date:
    s = s.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {s!r}")


def _parse_bar_entry(symbol: str, raw: dict) -> Optional[Bar]:
    ts_raw = raw.get("timestamp") or raw.get("date") or raw.get("ts")
    if ts_raw is None:
        return None
    if isinstance(ts_raw, (int, float)):
        try:
            d = datetime.fromtimestamp(float(ts_raw)).date()
        except (OSError, ValueError, OverflowError):
            return None
    else:
        try:
            d = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).date()
        except ValueError:
            try:
                d = datetime.strptime(str(ts_raw)[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
    try:
        return Bar(
            symbol=symbol,
            date=d,
            open=float(raw.get("open", 0.0)),
            high=float(raw.get("high", 0.0)),
            low=float(raw.get("low", 0.0)),
            close=float(raw.get("close", 0.0)),
            volume=int(raw.get("volume", 0) or 0),
            vwap=float(raw["vwap"]) if raw.get("vwap") not in (None, "") else None,
        )
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------


def import_file(store: PITStore, path: Path) -> FileImportResult:
    """Import a single legacy JSON file into `store`. Returns a per-file result."""
    parsed = _parse_legacy_filename(path.name)
    if parsed is None:
        return FileImportResult(
            path=path, symbol="", bars_in=0, bars_written=0,
            error=f"unrecognised filename layout: {path.name}",
        )
    symbol, _bar_size, _start, _end = parsed

    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return FileImportResult(
            path=path, symbol=symbol, bars_in=0, bars_written=0,
            error=f"read/parse error: {exc}",
        )

    raw_bars = body.get("bars") if isinstance(body, dict) else body
    if not isinstance(raw_bars, list):
        return FileImportResult(
            path=path, symbol=symbol, bars_in=0, bars_written=0,
            error="no 'bars' list in payload",
        )

    bars: list[Bar] = []
    for r in raw_bars:
        if not isinstance(r, dict):
            continue
        b = _parse_bar_entry(symbol, r)
        if b is not None:
            bars.append(b)

    # Ensure security row exists.
    store.upsert_security(symbol)
    written = store.write_bars(symbol, bars)
    return FileImportResult(
        path=path, symbol=symbol, bars_in=len(bars), bars_written=written,
    )


def import_directory(
    store: PITStore,
    cache_dir: Path,
    *,
    glob: str = "*.json",
) -> MigrationReport:
    """Walk `cache_dir` for JSON cache files and import each."""
    started = datetime.now()
    report = MigrationReport(started_at=started, finished_at=None)
    files = sorted(cache_dir.rglob(glob))
    for fp in files:
        result = import_file(store, fp)
        report.per_file.append(result)
        report.files_processed += 1
        if result.error:
            report.files_failed += 1
        else:
            report.bars_in += result.bars_in
            report.bars_written += result.bars_written
    report.finished_at = datetime.now()
    _log_migration(store, started, report)
    return report


def reconcile(
    store: PITStore,
    cache_dir: Path,
    *,
    sample_per_file: int = 5,
    tolerance: float = 1e-6,
) -> MigrationReport:
    """Spot-check reconciliation.

    For each legacy JSON file, sample up to `sample_per_file` bars and
    compare close prices against what's in the PIT store. Returns a
    report with reconciled_pairs / mismatched_pairs populated.
    """
    started = datetime.now()
    report = MigrationReport(started_at=started, finished_at=None)
    files = sorted(cache_dir.rglob("*.json"))
    for fp in files:
        parsed = _parse_legacy_filename(fp.name)
        if parsed is None:
            continue
        symbol = parsed[0]
        try:
            body = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        raw_bars = body.get("bars") if isinstance(body, dict) else body
        if not isinstance(raw_bars, list):
            continue
        # Sample uniformly.
        n = len(raw_bars)
        if n == 0:
            continue
        step = max(1, n // sample_per_file)
        for i in range(0, n, step):
            r = raw_bars[i]
            if not isinstance(r, dict):
                continue
            b = _parse_bar_entry(symbol, r)
            if b is None:
                continue
            recovered = store.read_bars(symbol, b.date, b.date)
            if not recovered:
                report.mismatched_pairs += 1
                continue
            if abs(recovered[0].close - b.close) > tolerance:
                report.mismatched_pairs += 1
            else:
                report.reconciled_pairs += 1
    report.finished_at = datetime.now()
    return report


def _log_migration(
    store: PITStore,
    started: datetime,
    report: MigrationReport,
) -> None:
    """Insert a row into migration_log for audit."""
    try:
        with store._conn() as conn:    # noqa: SLF001
            conn.execute(
                """
                INSERT INTO migration_log
                  (started_at, finished_at, source, rows_in, rows_written, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    started.isoformat(),
                    (report.finished_at or datetime.now()).isoformat(),
                    "legacy_json_cache",
                    report.bars_in,
                    report.bars_written,
                    f"files={report.files_processed} failed={report.files_failed}",
                ),
            )
    except Exception as exc:
        log.warning("migration_log insert failed: %s", exc)

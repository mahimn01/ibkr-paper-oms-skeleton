"""Bitemporal point-in-time data store.

Architecture (PLAN.md §2.1):

    SQLite + WAL  -> metadata: securities, ticker_history, splits, dividends,
                     mergers, spinoffs, index_membership, risk_state,
                     spread_cache, borrow_rates, migration_log.
    Parquet       -> bars, partitioned by symbol/year. Stored unadjusted.
    DuckDB        -> query layer over the parquet shards (zero-server).

Bitemporal columns on bars:
    known_from  -> when our system learned this bar (transaction-time start)
    known_to    -> when we superseded it (default '9999-12-31')

Restatements append a new row with `known_from = today`, set the prior row's
`known_to = today`. Old backtests, queried as-of their original run date,
return the original prices; queried today, return the corrected prices.
Both are auditable.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_FAR_FUTURE = "9999-12-31"


# --------------------------------------------------------------------------
# DTOs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Security:
    """Permanent surrogate identity for a tradable instrument."""
    internal_id: int
    primary_ticker: str
    cusip: str | None = None
    figi: str | None = None
    list_date: date | None = None
    delist_date: date | None = None
    delist_reason: str | None = None


@dataclass(frozen=True)
class Bar:
    """Unadjusted OHLCV bar with bitemporal stamps.

    Adjustment factors are applied at query time by AdjustmentEngine.
    """
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    known_from: datetime | None = None
    known_to: datetime | None = None


# Parquet schema for bars. Held constant so partitions are append-compatible.
# Microsecond resolution so we can represent the far-future sentinel (9999-12-31)
# which falls outside nanosecond range (~year 2262).
_BAR_SCHEMA = pa.schema([
    ("date", pa.timestamp("us")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("vwap", pa.float64()),
    ("known_from", pa.timestamp("us")),
    ("known_to", pa.timestamp("us")),
])


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

class PITStore:
    """Bitemporal store. SQLite for metadata, parquet for bars.

    Construct with a single root path; layout is:
        <root>/meta.sqlite
        <root>/bars/symbol={X}/year={Y}/data.parquet

    Concurrency: SQLite WAL handles concurrent readers + one writer. Parquet
    writes are atomic (rename-to-final). Avoid concurrent writers for the
    same (symbol, year) shard.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bars_root = self.root / "bars"
        self.bars_root.mkdir(exist_ok=True)
        self.db_path = self.root / "meta.sqlite"
        self._init_schema()

    # ------------------------------------------------------------------ schema

    def _init_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text()
        with self._conn() as conn:
            conn.executescript(sql)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,            # explicit BEGIN/COMMIT
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ securities

    def upsert_security(
        self,
        primary_ticker: str,
        *,
        cusip: str | None = None,
        figi: str | None = None,
        list_date: date | None = None,
        delist_date: date | None = None,
        delist_reason: str | None = None,
        as_of: datetime | None = None,
    ) -> int:
        """Insert or update a security; returns its `internal_id`.

        Lookup is by `primary_ticker` + currently-active row (`known_to = far future`).
        If a row exists with matching ticker, returns that id (idempotent).
        Otherwise creates a fresh row.

        For a true ticker change, call `record_ticker_change` separately so the
        history is preserved in `ticker_history`.
        """
        as_of = (as_of or datetime.now(timezone.utc)).replace(microsecond=0)
        with self._conn() as conn:
            conn.execute("BEGIN")
            row = conn.execute(
                """
                SELECT internal_id FROM securities
                WHERE primary_ticker = ? AND known_to = ?
                """,
                (primary_ticker, _FAR_FUTURE),
            ).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                return int(row["internal_id"])

            cur = conn.execute(
                """
                INSERT INTO securities
                    (primary_ticker, cusip, figi, list_date, delist_date,
                     delist_reason, known_from, known_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    primary_ticker,
                    cusip,
                    figi,
                    list_date.isoformat() if list_date else None,
                    delist_date.isoformat() if delist_date else None,
                    delist_reason,
                    as_of.isoformat(),
                    _FAR_FUTURE,
                ),
            )
            internal_id = int(cur.lastrowid or 0)
            # Always record the initial ticker into ticker_history so as-of
            # ticker resolution works without special cases.
            conn.execute(
                """
                INSERT OR IGNORE INTO ticker_history
                    (internal_id, ticker, valid_from, valid_to)
                VALUES (?, ?, ?, ?)
                """,
                (
                    internal_id,
                    primary_ticker,
                    (list_date or as_of.date()).isoformat(),
                    _FAR_FUTURE,
                ),
            )
            conn.execute("COMMIT")
            return internal_id

    def record_ticker_change(
        self,
        internal_id: int,
        new_ticker: str,
        change_date: date,
    ) -> None:
        """Record a ticker change; e.g. FB -> META on 2022-06-09."""
        with self._conn() as conn:
            conn.execute("BEGIN")
            # Close out the current row.
            conn.execute(
                """
                UPDATE ticker_history SET valid_to = ?
                WHERE internal_id = ? AND valid_to = ?
                """,
                (change_date.isoformat(), internal_id, _FAR_FUTURE),
            )
            # Open the new row.
            conn.execute(
                """
                INSERT INTO ticker_history
                    (internal_id, ticker, valid_from, valid_to)
                VALUES (?, ?, ?, ?)
                """,
                (internal_id, new_ticker, change_date.isoformat(), _FAR_FUTURE),
            )
            # Update the canonical primary_ticker on the security row.
            conn.execute(
                "UPDATE securities SET primary_ticker = ? WHERE internal_id = ?",
                (new_ticker, internal_id),
            )
            conn.execute("COMMIT")

    def resolve_ticker(self, ticker: str, as_of: date) -> int | None:
        """Resolve a ticker as of a given date to its `internal_id`.

        Returns None if no security used that ticker on that date.
        """
        d = as_of.isoformat()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT internal_id FROM ticker_history
                WHERE ticker = ? AND valid_from <= ? AND valid_to > ?
                LIMIT 1
                """,
                (ticker, d, d),
            ).fetchone()
            return int(row["internal_id"]) if row else None

    # ------------------------------------------------------------------ corp actions

    def add_split(self, internal_id: int, ex_date: date, ratio: float) -> None:
        if ratio <= 0:
            raise ValueError(f"split ratio must be > 0, got {ratio}")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO splits (internal_id, ex_date, ratio)
                VALUES (?, ?, ?)
                """,
                (internal_id, ex_date.isoformat(), ratio),
            )

    def add_dividend(
        self,
        internal_id: int,
        ex_date: date,
        amount: float,
        div_type: str = "regular",
    ) -> None:
        if div_type not in {"regular", "special", "roc", "stock"}:
            raise ValueError(f"unknown div_type: {div_type}")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dividends
                    (internal_id, ex_date, amount, div_type)
                VALUES (?, ?, ?, ?)
                """,
                (internal_id, ex_date.isoformat(), amount, div_type),
            )

    def get_splits(self, internal_id: int) -> list[tuple[date, float]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ex_date, ratio FROM splits WHERE internal_id = ? ORDER BY ex_date",
                (internal_id,),
            ).fetchall()
            return [(date.fromisoformat(r["ex_date"]), float(r["ratio"])) for r in rows]

    def get_dividends(
        self,
        internal_id: int,
        types: Sequence[str] = ("regular",),
    ) -> list[tuple[date, float, str]]:
        placeholders = ",".join("?" for _ in types)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT ex_date, amount, div_type FROM dividends
                WHERE internal_id = ? AND div_type IN ({placeholders})
                ORDER BY ex_date
                """,
                (internal_id, *types),
            ).fetchall()
            return [
                (date.fromisoformat(r["ex_date"]), float(r["amount"]), r["div_type"])
                for r in rows
            ]

    # ------------------------------------------------------------------ index membership

    def add_index_membership(
        self,
        index_name: str,
        internal_id: int,
        added_date: date,
        removed_date: date | None = None,
        announce_date: date | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO index_membership
                    (index_name, internal_id, added_date, removed_date, announce_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    index_name,
                    internal_id,
                    added_date.isoformat(),
                    removed_date.isoformat() if removed_date else None,
                    announce_date.isoformat() if announce_date else None,
                ),
            )

    # ------------------------------------------------------------------ bars

    def write_bars(self, symbol: str, bars: Iterable[Bar]) -> int:
        """Append bars for a single symbol to its parquet partitions.

        Caller must not write the same `(symbol, date)` twice without first
        retiring the prior row (call `restate_bar` for a true revision). Bars
        already on disk for the same date are detected and skipped with a warning.
        """
        bars_list = sorted(bars, key=lambda b: b.date)
        if not bars_list:
            return 0
        # Group by year for partitioned writes.
        now = datetime.now(timezone.utc).replace(microsecond=0)
        written = 0
        by_year: dict[int, list[Bar]] = {}
        for b in bars_list:
            by_year.setdefault(b.date.year, []).append(b)
        for year, year_bars in by_year.items():
            written += self._append_year_partition(symbol, year, year_bars, default_known_from=now)
        return written

    def _partition_path(self, symbol: str, year: int) -> Path:
        # Sanitize symbol for filesystem: uppercase, replace forbidden chars.
        safe = symbol.upper().replace("/", "_").replace("\\", "_")
        return self.bars_root / f"symbol={safe}" / f"year={year}" / "data.parquet"

    @staticmethod
    def _strip_tz(dt: datetime) -> datetime:
        """Strip tzinfo so all timestamps share a (naive) representation
        compatible with pyarrow's timestamp('us') schema."""
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

    def _append_year_partition(
        self,
        symbol: str,
        year: int,
        bars: list[Bar],
        default_known_from: datetime,
    ) -> int:
        path = self._partition_path(symbol, year)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_dates: set[date] = set()
        if path.exists():
            existing = pq.read_table(path, columns=["date", "known_to"])
            for d, kt in zip(existing.column("date").to_pylist(),
                             existing.column("known_to").to_pylist()):
                # Only count rows that are still currently known (active).
                if kt is not None and kt.year == 9999:
                    existing_dates.add(d.date() if hasattr(d, "date") else d)

        rows = []
        skipped = 0
        default_known_from = self._strip_tz(default_known_from)
        for b in bars:
            if b.date in existing_dates:
                skipped += 1
                continue
            rows.append({
                "date": datetime(b.date.year, b.date.month, b.date.day),
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
                "vwap": float(b.vwap) if b.vwap is not None else float("nan"),
                "known_from": self._strip_tz(b.known_from) if b.known_from else default_known_from,
                "known_to": self._strip_tz(b.known_to) if b.known_to else datetime(9999, 12, 31),
            })

        if skipped:
            log.warning(
                "PITStore.write_bars: %d duplicate dates skipped for %s/%d "
                "(use restate_bar for revisions)",
                skipped, symbol, year,
            )
        if not rows:
            return 0

        new_table = pa.Table.from_pylist(rows, schema=_BAR_SCHEMA)
        if path.exists():
            old = pq.read_table(path, schema=_BAR_SCHEMA)
            combined = pa.concat_tables([old, new_table]).sort_by([("date", "ascending")])
        else:
            combined = new_table.sort_by([("date", "ascending")])

        # Atomic write: write to .tmp then rename.
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(combined, tmp, compression="zstd")
        tmp.replace(path)
        return len(rows)

    def read_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        as_of: datetime | None = None,
    ) -> list[Bar]:
        """Read bars for `symbol` over [start, end] visible at `as_of`.

        `as_of` defaults to "now" — gives the latest known data. Set it to a
        past timestamp to recover the data your system would have seen at
        that historical moment (e.g. for re-running an old backtest exactly).
        """
        years = range(start.year, end.year + 1)
        out: list[Bar] = []
        cutoff = self._strip_tz(as_of or datetime.now(timezone.utc))
        for year in years:
            path = self._partition_path(symbol, year)
            if not path.exists():
                continue
            table = pq.read_table(path, schema=_BAR_SCHEMA)
            for row in table.to_pylist():
                d = row["date"].date()
                if d < start or d > end:
                    continue
                kf = row["known_from"]
                kt = row["known_to"]
                if kf is not None and kf > cutoff:
                    continue
                if kt is not None and kt <= cutoff:
                    continue
                out.append(Bar(
                    symbol=symbol,
                    date=d,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    vwap=row["vwap"] if row["vwap"] == row["vwap"] else None,  # NaN check
                    known_from=kf,
                    known_to=kt,
                ))
        out.sort(key=lambda b: b.date)
        return out

    def restate_bar(
        self,
        symbol: str,
        bar: Bar,
        *,
        restatement_time: datetime | None = None,
    ) -> None:
        """Record a vendor restatement for a single bar.

        Closes out the prior currently-known row (sets `known_to = restatement_time`)
        and inserts the new row with `known_from = restatement_time`.
        """
        rt = self._strip_tz(restatement_time or datetime.now(timezone.utc))
        path = self._partition_path(symbol, bar.date.year)
        if not path.exists():
            # No prior bar; just write the new one.
            self.write_bars(symbol, [bar])
            return

        old = pq.read_table(path, schema=_BAR_SCHEMA).to_pylist()
        # Close out the active row for that date.
        target_date = datetime(bar.date.year, bar.date.month, bar.date.day)
        far_future = datetime(9999, 12, 31)
        revised = []
        replaced = False
        for r in old:
            if r["date"] == target_date and r["known_to"] == far_future:
                r = dict(r)
                r["known_to"] = rt
                replaced = True
            revised.append(r)
        # Append the new vintage.
        revised.append({
            "date": target_date,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "vwap": float(bar.vwap) if bar.vwap is not None else float("nan"),
            "known_from": rt,
            "known_to": far_future,
        })
        if not replaced:
            log.warning(
                "PITStore.restate_bar: no active prior row for %s %s; treating as fresh insert",
                symbol, bar.date,
            )
        new_table = pa.Table.from_pylist(revised, schema=_BAR_SCHEMA).sort_by(
            [("date", "ascending"), ("known_from", "ascending")]
        )
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(new_table, tmp, compression="zstd")
        tmp.replace(path)

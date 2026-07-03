from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from typing import Any

from trading_algo.broker.base import OrderRequest, OrderStatus
from trading_algo.config import TradingConfig
from trading_algo.orders import TradeIntent


class SqliteStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        # timeout + busy_timeout are required for cross-process safety: the
        # constitution-check CLI (one process) writes a verdict that a later
        # place/OMS process reads from the same WAL file. Without these the
        # second writer raises "database is locked" instead of waiting.
        # Mirrors idempotency.py's connection setup.
        self._conn = sqlite3.connect(db_path, timeout=30.0)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=30000;")
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def start_run(self, cfg: TradingConfig) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO runs(started_epoch_s, config_json) VALUES(?, ?)",
            (time.time(), json.dumps(asdict(cfg), sort_keys=True)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def end_run(self, run_id: int) -> None:
        self._conn.execute("UPDATE runs SET ended_epoch_s=? WHERE id=?", (time.time(), int(run_id)))
        self._conn.commit()

    def log_decision(
        self,
        run_id: int,
        *,
        strategy: str,
        intent: TradeIntent,
        accepted: bool,
        reason: str | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO decisions(run_id, ts_epoch_s, strategy, intent_json, accepted, reason) VALUES(?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(strategy),
                json.dumps(_to_jsonable(asdict(intent)), sort_keys=True),
                1 if accepted else 0,
                reason,
            ),
        )
        self._conn.commit()

    def log_order(
        self,
        run_id: int,
        *,
        broker: str,
        order_id: str,
        request: OrderRequest,
        status: str,
        perm_id: str | None = None,
        order_ref: str | None = None,
        account: str | None = None,
        strategy_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        req_n = request.normalized()
        inst = req_n.instrument
        self._conn.execute(
            "INSERT INTO orders("
            "run_id, ts_epoch_s, broker, order_id, instrument_kind, instrument_symbol, "
            "side, quantity, order_type, request_json, status, "
            "perm_id, order_ref, account, strategy_id, agent_id, group_id, idempotency_key"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(broker),
                str(order_id),
                str(inst.kind),
                str(inst.symbol),
                str(req_n.side),
                float(req_n.quantity),
                str(req_n.order_type),
                json.dumps(_to_jsonable(asdict(req_n)), sort_keys=True),
                str(status),
                perm_id,
                order_ref,
                account,
                strategy_id,
                agent_id,
                group_id,
                idempotency_key,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # T4.1 accessors for richer order lookups
    # ------------------------------------------------------------------

    def orders_by_group(self, group_id: str) -> list[dict]:
        """Return all order rows in a group ordered by ts ascending."""
        cur = self._conn.execute(
            "SELECT id, run_id, ts_epoch_s, broker, order_id, "
            "instrument_kind, instrument_symbol, side, quantity, order_type, "
            "status, perm_id, order_ref, account, strategy_id, agent_id, "
            "group_id, idempotency_key "
            "FROM orders WHERE group_id=? ORDER BY ts_epoch_s ASC, id ASC",
            (str(group_id),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def list_groups(self) -> list[dict]:
        """Aggregate summary of every distinct group_id seen."""
        cur = self._conn.execute(
            "SELECT group_id, COUNT(*) AS n, "
            "MIN(ts_epoch_s) AS first_ts, MAX(ts_epoch_s) AS last_ts, "
            "GROUP_CONCAT(DISTINCT status) AS statuses "
            "FROM orders WHERE group_id IS NOT NULL GROUP BY group_id "
            "ORDER BY MAX(ts_epoch_s) DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def order_by_idempotency_key(self, key: str) -> dict | None:
        """Return the most recent order row matching `idempotency_key`."""
        cur = self._conn.execute(
            "SELECT id, order_id, status, ts_epoch_s FROM orders "
            "WHERE idempotency_key=? ORDER BY ts_epoch_s DESC, id DESC LIMIT 1",
            (str(key),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def update_order_status(self, order_id: str, status: str) -> None:
        self._conn.execute("UPDATE orders SET status=? WHERE order_id=?", (str(status), str(order_id)))
        self._conn.commit()

    def list_non_terminal_order_ids(self) -> list[str]:
        cur = self._conn.execute("SELECT DISTINCT order_id, status FROM orders")
        order_ids: list[str] = []
        for oid, st in cur.fetchall():
            if oid and not _is_terminal_status(str(st)):
                order_ids.append(str(oid))
        return order_ids

    def get_latest_status(self, order_id: str) -> str | None:
        cur = self._conn.execute(
            "SELECT status FROM orders WHERE order_id=? ORDER BY ts_epoch_s DESC, id DESC LIMIT 1",
            (str(order_id),),
        )
        row = cur.fetchone()
        return str(row[0]) if row else None

    def log_order_status_event(self, run_id: int, broker: str, st: OrderStatus) -> None:
        self._conn.execute(
            "INSERT INTO order_status_events(run_id, ts_epoch_s, broker, order_id, status, filled, remaining, avg_fill_price) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(broker),
                str(st.order_id),
                str(st.status),
                st.filled,
                st.remaining,
                st.avg_fill_price,
            ),
        )
        self._conn.commit()

    def log_error(self, run_id: int, *, where: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO errors(run_id, ts_epoch_s, where_text, message) VALUES(?, ?, ?, ?)",
            (int(run_id), time.time(), str(where), str(message)),
        )
        self._conn.commit()

    def log_action(
        self,
        run_id: int,
        *,
        actor: str,
        payload: dict[str, Any],
        accepted: bool,
        reason: str | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO actions(run_id, ts_epoch_s, actor, payload_json, accepted, reason) VALUES(?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(actor),
                json.dumps(_to_jsonable(payload), sort_keys=True),
                1 if accepted else 0,
                reason,
            ),
        )
        self._conn.commit()

    def log_constitution_verdict(
        self,
        *,
        order_key: str,
        decision: str,
        complete: bool,
        checks: list[dict[str, Any]],
        symbol: str | None = None,
        account: str | None = None,
        context: dict[str, Any] | None = None,
        ts_epoch_s: float | None = None,
    ) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO constitution_verdicts"
            "(ts_epoch_s, order_key, decision, complete, symbol, account, checks_json, context_json) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                float(ts_epoch_s) if ts_epoch_s is not None else time.time(),
                str(order_key),
                str(decision),
                1 if complete else 0,
                symbol,
                account,
                json.dumps(_to_jsonable(checks), sort_keys=True),
                json.dumps(_to_jsonable(context), sort_keys=True) if context is not None else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def latest_constitution_verdict(
        self, order_key: str, *, max_age_s: float | None = None, now: float | None = None
    ) -> dict | None:
        """Most recent verdict for an order_key, or None. When max_age_s is set,
        a verdict older than that is treated as absent (returns None) so a stale
        clearance cannot authorize a transmit."""
        cur = self._conn.execute(
            "SELECT * FROM constitution_verdicts WHERE order_key=? "
            "ORDER BY ts_epoch_s DESC, id DESC LIMIT 1",
            (str(order_key),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        rec = dict(zip([d[0] for d in cur.description], row))
        if max_age_s is not None:
            ref = now if now is not None else time.time()
            if ref - float(rec["ts_epoch_s"]) > float(max_age_s):
                return None
        return rec

    def claim_constitution_verdict(self, verdict_id: int, order_ref: str) -> bool:
        """Atomically bind a verdict to ONE order_ref (single-use clearance).
        The same ref may claim repeatedly (the OMS and broker chokepoints both
        check one transmit); a DIFFERENT ref finds it taken and fails closed —
        so one clearance cannot authorize two orders."""
        cur = self._conn.execute(
            "UPDATE constitution_verdicts SET claimed_order_ref=?, claimed_ts_epoch_s=? "
            "WHERE id=? AND (claimed_order_ref IS NULL OR claimed_order_ref=?)",
            (str(order_ref), time.time(), int(verdict_id), str(order_ref)),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ----------------------------------------------------------- trade journal

    def log_journal_entry(
        self,
        *,
        underlying: str,
        thesis: str,
        written_exit: str,
        structure: str | None = None,
        side: str | None = None,
        quantity: float | None = None,
        account: str | None = None,
        order_key: str | None = None,
        order_ref: str | None = None,
        constitution_decision: str | None = None,
        context: dict[str, Any] | None = None,
        defensive_state: str = "none",
        status: str = "open",
        notes: str | None = None,
        ts_epoch_s: float | None = None,
    ) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO trade_journal"
            "(ts_epoch_s, order_key, order_ref, account, underlying, structure, side, quantity, "
            " thesis, written_exit, constitution_decision, context_json, defensive_state, status, notes) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                float(ts_epoch_s) if ts_epoch_s is not None else time.time(),
                order_key, order_ref, account, str(underlying).upper(), structure, side,
                float(quantity) if quantity is not None else None,
                str(thesis), str(written_exit), constitution_decision,
                json.dumps(_to_jsonable(context), sort_keys=True) if context is not None else None,
                str(defensive_state), str(status), notes,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def open_journal_entries(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM trade_journal WHERE status='open' ORDER BY underlying ASC, ts_epoch_s ASC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def journal_entries_by_underlying(self, underlying: str, *, status: str | None = None) -> list[dict]:
        if status is None:
            cur = self._conn.execute(
                "SELECT * FROM trade_journal WHERE underlying=? ORDER BY ts_epoch_s DESC",
                (underlying.upper(),),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM trade_journal WHERE underlying=? AND status=? ORDER BY ts_epoch_s DESC",
                (underlying.upper(), status),
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update_journal_status(
        self, entry_id: int, *, status: str, realized_pnl: float | None = None,
        closed_ts_epoch_s: float | None = None,
    ) -> None:
        """COALESCE preserves an existing realized_pnl / closed_ts when the
        caller omits them (e.g. re-tagging closed -> rolled) — a recorded LOSS
        must never be silently NULLed, or the W2 martingale bridge goes blind."""
        ts = closed_ts_epoch_s
        if ts is None and status in {"closed", "rolled"}:
            ts = time.time()
        cur = self._conn.execute(
            "UPDATE trade_journal SET status=?, "
            "realized_pnl=COALESCE(?, realized_pnl), "
            "closed_ts_epoch_s=COALESCE(closed_ts_epoch_s, ?) WHERE id=?",
            (
                str(status),
                float(realized_pnl) if realized_pnl is not None else None,
                float(ts) if ts is not None else None,
                int(entry_id),
            ),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"trade_journal entry {entry_id} does not exist")

    def set_journal_defensive_state(self, entry_id: int, state: str) -> None:
        cur = self._conn.execute(
            "UPDATE trade_journal SET defensive_state=? WHERE id=?", (str(state), int(entry_id))
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"trade_journal entry {entry_id} does not exist")

    def recent_losing_put_close(
        self, underlying: str, *, within_s: float, now: float | None = None
    ) -> bool:
        """W2 martingale bridge: was a short put on this underlying closed at a
        loss within the window? Feeds ProposedOrderInput.losing_put_close_*."""
        ref = now if now is not None else time.time()
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM trade_journal WHERE underlying=? AND structure='short-put' "
            "AND status IN ('closed','rolled') AND realized_pnl IS NOT NULL AND realized_pnl < 0 "
            "AND closed_ts_epoch_s IS NOT NULL AND closed_ts_epoch_s >= ?",
            (underlying.upper(), float(ref) - float(within_s)),
        )
        return int(cur.fetchone()[0]) > 0

    def positions_missing_written_exit(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM trade_journal WHERE status='open' "
            "AND (written_exit IS NULL OR TRIM(written_exit)='')"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version(
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_epoch_s REAL NOT NULL,
                ended_epoch_s REAL,
                config_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                strategy TEXT NOT NULL,
                intent_json TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS orders(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                broker TEXT NOT NULL,
                order_id TEXT NOT NULL,
                instrument_kind TEXT,
                instrument_symbol TEXT,
                side TEXT,
                quantity REAL,
                order_type TEXT,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                perm_id TEXT,
                order_ref TEXT,
                account TEXT,
                strategy_id TEXT,
                agent_id TEXT,
                group_id TEXT,
                idempotency_key TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(instrument_symbol);

            CREATE TABLE IF NOT EXISTS order_status_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                broker TEXT NOT NULL,
                order_id TEXT NOT NULL,
                status TEXT NOT NULL,
                filled REAL,
                remaining REAL,
                avg_fill_price REAL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_order_status_events_order_id ON order_status_events(order_id);

            CREATE TABLE IF NOT EXISTS errors(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                where_text TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS actions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                actor TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            -- Pre-transmit constitution gate. Deliberately NO run_id FK: the
            -- constitution-check CLI has no run, and foreign_keys=ON would
            -- reject a write keyed to a non-existent run. order_key is the
            -- content key (constitution.order_key) used to match a verdict back
            -- at transmit time.
            CREATE TABLE IF NOT EXISTS constitution_verdicts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_epoch_s REAL NOT NULL,
                order_key TEXT NOT NULL,
                decision TEXT NOT NULL,
                complete INTEGER NOT NULL,
                symbol TEXT,
                account TEXT,
                checks_json TEXT NOT NULL,
                context_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_constitution_verdicts_key
                ON constitution_verdicts(order_key, ts_epoch_s);

            -- Trade journal: the durable "why" behind each live position, read
            -- back at session start to close the cross-session memory gap. NO
            -- run_id FK (a journal entry outlives any single run/process).
            CREATE TABLE IF NOT EXISTS trade_journal(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_epoch_s REAL NOT NULL,
                order_key TEXT,
                order_ref TEXT,
                account TEXT,
                underlying TEXT NOT NULL,
                structure TEXT,
                side TEXT,
                quantity REAL,
                thesis TEXT NOT NULL,
                written_exit TEXT NOT NULL,
                constitution_decision TEXT,
                context_json TEXT,
                defensive_state TEXT NOT NULL DEFAULT 'none',
                status TEXT NOT NULL DEFAULT 'open',
                realized_pnl REAL,
                closed_ts_epoch_s REAL,
                notes TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trade_journal_underlying
                ON trade_journal(underlying, status);
            CREATE INDEX IF NOT EXISTS idx_trade_journal_status
                ON trade_journal(status);
            """
        )
        # T4.1 — idempotent migration: older DBs won't have these columns yet.
        self._add_column_if_missing("orders", "perm_id", "TEXT")
        self._add_column_if_missing("orders", "order_ref", "TEXT")
        self._add_column_if_missing("orders", "account", "TEXT")
        self._add_column_if_missing("orders", "strategy_id", "TEXT")
        self._add_column_if_missing("orders", "agent_id", "TEXT")
        self._add_column_if_missing("orders", "group_id", "TEXT")
        self._add_column_if_missing("orders", "idempotency_key", "TEXT")
        # Single-use clearance claim (constitution gate): a verdict binds to the
        # first order_ref that transmits against it.
        self._add_column_if_missing("constitution_verdicts", "claimed_order_ref", "TEXT")
        self._add_column_if_missing("constitution_verdicts", "claimed_ts_epoch_s", "REAL")
        # Indexes (safe once columns exist).
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_group_id ON orders(group_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_idempotency_key ON orders(idempotency_key)"
        )

        cur = self._conn.execute("SELECT COUNT(*) FROM schema_version")
        if int(cur.fetchone()[0]) == 0:
            self._conn.execute("INSERT INTO schema_version(version) VALUES(2)")
        self._conn.commit()

    def _add_column_if_missing(self, table: str, col: str, col_type: str) -> None:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        if col not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float)) or obj is None:
        return obj
    return str(obj)


def _is_terminal_status(status: str) -> bool:
    s = str(status).strip()
    return s in {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}

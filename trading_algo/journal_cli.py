"""Trade-journal CLI: record the WHY behind each live position and replay it at
session start. Deliberately free of any broker/ib_async import so `journal brief`
runs at SessionStart even with no gateway up.
"""

from __future__ import annotations

import argparse
import sys

from trading_algo.config import TradingConfig
from trading_algo.persistence import SqliteStore


def render_brief(open_entries: list[dict], *, missing_exit: int = 0) -> str:
    if not open_entries:
        return "TRADE JOURNAL: no open positions recorded."
    lines = [f"TRADE JOURNAL BRIEF — {len(open_entries)} open position(s)"]
    defensive = 0
    for e in open_entries:
        u = str(e.get("underlying") or "?")
        st = str(e.get("structure") or "")
        side = str(e.get("side") or "")
        qty = e.get("quantity")
        thesis = (str(e.get("thesis") or "")).strip()
        exit_ = (str(e.get("written_exit") or "")).strip()
        dstate = str(e.get("defensive_state") or "none")
        marker = ""
        if dstate and dstate != "none":
            defensive += 1
            marker = f"   ⚠ DEFENSIVE: {dstate}"
        qtystr = f" x{qty:g}" if isinstance(qty, (int, float)) else ""
        lines.append(
            f"  {u:<6} {st:<14} {side}{qtystr}".rstrip()
            + f"  thesis: {thesis}  exit: {exit_}{marker}"
        )
    lines.append(f"UNRESOLVED DEFENSIVE TRIGGERS: {defensive}")
    lines.append(f"POSITIONS MISSING WRITTEN EXIT: {missing_exit}")
    return "\n".join(lines)


def _resolve_db_path(args: argparse.Namespace) -> str | None:
    explicit = getattr(args, "db_path", None)
    if explicit:
        return str(explicit)
    return TradingConfig.from_env().db_path


def cmd_journal(args: argparse.Namespace) -> int:
    db = _resolve_db_path(args)
    if not db:
        # Fail-open: a missing DB must never break a SessionStart brief.
        if args.action == "brief":
            print("TRADE JOURNAL: no TRADING_DB_PATH configured.")
            return 0
        print("No TRADING_DB_PATH configured (set TRADING_DB_PATH or pass --db-path).",
              file=sys.stderr)
        return 1  # a failed write/query is observable to scripts; only `brief` fails open
    store = SqliteStore(db)
    try:
        if args.action == "brief":
            entries = store.open_journal_entries()
            missing = len(store.positions_missing_written_exit())
            print(render_brief(entries, missing_exit=missing))
            return 0
        if args.action == "list":
            entries = (store.journal_entries_by_underlying(args.underlying)
                       if args.underlying else store.open_journal_entries())
            for e in entries:
                print(f"[{e['id']}] {e['status']:<7} {e['underlying']:<6} "
                      f"{e.get('structure') or '':<14} exit: {e.get('written_exit') or ''}")
            return 0
        if args.action == "add":
            if not (args.underlying and args.thesis and args.exit):
                raise SystemExit("journal add requires --underlying, --thesis, and --exit")
            eid = store.log_journal_entry(
                underlying=args.underlying, thesis=args.thesis, written_exit=args.exit,
                structure=args.structure, side=args.side, quantity=args.quantity,
                account=args.account, order_ref=args.order_ref,
                constitution_decision=args.constitution_decision,
                defensive_state=args.defensive_state or "none",
            )
            print(f"journal entry {eid} recorded")
            return 0
        if args.action == "defensive":
            if args.id is None or args.state is None:
                raise SystemExit("journal defensive requires --id and --state")
            store.set_journal_defensive_state(args.id, args.state)
            print(f"journal entry {args.id} defensive_state={args.state}")
            return 0
        if args.action == "close":
            if args.id is None:
                raise SystemExit("journal close requires --id")
            store.update_journal_status(args.id, status=args.status or "closed",
                                        realized_pnl=args.realized_pnl)
            print(f"journal entry {args.id} -> {args.status or 'closed'}")
            return 0
        raise SystemExit(f"unknown journal action: {args.action}")
    finally:
        store.close()


def add_journal_subparser(sub: argparse._SubParsersAction) -> None:
    jp = sub.add_parser("journal", help="Record / replay the trade journal (thesis, exit, defensive state)")
    jp.add_argument("action", choices=["brief", "list", "add", "defensive", "close"])
    jp.add_argument("--db-path", default=None, help="override TRADING_DB_PATH")
    jp.add_argument("--underlying", default=None)
    jp.add_argument("--thesis", default=None)
    jp.add_argument("--exit", default=None, help="written exit (trim ladder / trailing stop)")
    jp.add_argument("--structure", default=None)
    jp.add_argument("--side", default=None)
    jp.add_argument("--quantity", type=float, default=None)
    jp.add_argument("--account", default=None)
    jp.add_argument("--order-ref", dest="order_ref", default=None)
    jp.add_argument("--constitution-decision", dest="constitution_decision", default=None)
    jp.add_argument("--defensive-state", dest="defensive_state", default=None)
    jp.add_argument("--state", default=None, help="defensive state for `journal defensive`")
    jp.add_argument("--status", default=None, help="closed|rolled for `journal close`")
    jp.add_argument("--realized-pnl", dest="realized_pnl", type=float, default=None)
    jp.add_argument("--id", type=int, default=None)
    jp.set_defaults(func=cmd_journal)

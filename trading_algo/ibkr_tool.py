"""
Comprehensive IBKR data + operations CLI.

Covers: accounts, positions, PnL streaming, quotes, chains, greeks, depth,
real-time bars, tick-by-tick, historical bars/ticks, fundamentals, news,
scanner, contract search, executions, open/completed orders, what-if preview,
combo orders, global cancel, WSH events, FX, time, market rules.

Usage: python -m trading_algo.ibkr_tool <command> [args]
Defaults to IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID from .env.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

try:
    from ib_async import (
        IB,
        Bag,
        ComboLeg,
        Contract,
        ExecutionFilter,
        Forex,
        Future,
        Index,
        Option,
        Order,
        ScannerSubscription,
        StartupFetch,
        StartupFetchNONE,
        Stock,
        TagValue,
        Ticker,
        util,
    )
except Exception as exc:  # pragma: no cover
    print(f"ERROR: ib_async not installed: {exc}", file=sys.stderr)
    print("Install: .venv/bin/pip install ib_async", file=sys.stderr)
    sys.exit(2)


# ============================================================
# .env loader + connection helpers
# ============================================================

def _load_dotenv() -> None:
    path = ".env"
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if os.getenv(k) in (None, ""):
                os.environ[k] = v


_load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


DEFAULT_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
DEFAULT_PORT = _env_int("IBKR_PORT", 4001)
# Tool uses a distinct client id so it doesn't collide with the trading engine (which uses IBKR_CLIENT_ID from .env).
DEFAULT_CLIENT_ID = _env_int("IBKR_TOOL_CLIENT_ID", 177)


def _connect(
    args: argparse.Namespace,
    *,
    readonly: bool = False,
    account: str = "",
    raise_sync_errors: bool = False,
    fetch_fields: Any | None = None,
) -> IB:
    """Connect with explicit synchronization controls.

    ``ib_async`` 2.1 synchronizes account/portfolio state during connect when
    an account is supplied. Read handlers use this instead of manually calling
    the legacy ``reqAccountUpdates(subscribe, account)`` overload, which no
    longer exists. ``fetch_fields`` lets discovery connections avoid unrelated
    order/execution synchronization.
    """
    ib = IB()
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT
    client_id = args.client_id if args.client_id is not None else DEFAULT_CLIENT_ID
    connect_kwargs: dict[str, Any] = {
        "clientId": client_id,
        "timeout": args.timeout,
        "readonly": readonly,
        "account": account,
        "raiseSyncErrors": raise_sync_errors,
    }
    if fetch_fields is not None:
        connect_kwargs["fetchFields"] = fetch_fields
    ib.connect(host, port, **connect_kwargs)
    market_data_type = getattr(args, "market_data_type", None)
    if market_data_type:
        ib.reqMarketDataType(market_data_type)
    return ib


@contextmanager
def _ib_session(
    args: argparse.Namespace,
    *,
    readonly: bool = False,
    account: str = "",
    raise_sync_errors: bool = False,
    fetch_fields: Any | None = None,
) -> Iterator[IB]:
    """Yield an IB session and never leak its client ID on failure."""
    ib = _connect(
        args,
        readonly=readonly,
        account=account,
        raise_sync_errors=raise_sync_errors,
        fetch_fields=fetch_fields,
    )
    try:
        yield ib
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            # Preserve the command's original exception. If cleanup itself is
            # the only failure, surface it to the shared error classifier.
            if not active_error:
                raise


def _account_targets(ib: IB, requested: str | None) -> list[str]:
    """Resolve and validate account scope deterministically."""
    accounts = sorted({str(a) for a in (ib.managedAccounts() or []) if a})
    if not accounts:
        raise RuntimeError("IBKR returned no managed accounts for this session")
    if requested:
        if requested not in accounts:
            raise ValueError(
                f"Unknown IBKR account {requested!r}; managed accounts: {accounts}"
            )
        return [requested]
    return accounts


def _discover_account_targets(args: argparse.Namespace) -> list[str]:
    """Discover account scope without syncing orders or executions."""
    with _ib_session(
        args,
        readonly=True,
        fetch_fields=StartupFetchNONE,
    ) as ib:
        return _account_targets(ib, getattr(args, "account", None))


def _finite_ib_float(value: Any) -> float | None:
    """Normalize IBKR numeric sentinels and non-finite values to ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number == util.UNSET_DOUBLE:
        return None
    return number


def _wait_for_pnl_updates(
    ib: IB,
    objects: list[Any],
    *,
    timeout: float,
    single: bool = False,
) -> None:
    """Wait until every PnL subscription has received an actual update.

    PnL dataclasses start with NaN fields (and ``PnLSingle.position`` starts at
    zero), so emitting immediately can misreport a real position as flat. A
    timeout is safer than returning those constructor defaults as live data.
    """
    fields = (
        ("dailyPnL", "unrealizedPnL", "realizedPnL", "value")
        if single
        else ("dailyPnL", "unrealizedPnL", "realizedPnL")
    )
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        pending = [
            obj
            for obj in objects
            if not any(
                _finite_ib_float(getattr(obj, field, None)) is not None
                for field in fields
            )
        ]
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            kind = "position PnL" if single else "account PnL"
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for initial {kind} update"
            )
        ib.sleep(min(0.1, remaining))


# ============================================================
# Output helpers
# ============================================================

def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (
            not math.isfinite(obj) or obj == util.UNSET_DOUBLE
        ):
            return None
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _emit(data: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(_to_jsonable(data), indent=2, default=str))
        return
    if fmt == "csv":
        rows = data if isinstance(data, list) else [data]
        rows = [_to_jsonable(r) for r in rows if r is not None]
        rows = [r if isinstance(r, dict) else {"value": r} for r in rows]
        if not rows:
            return
        keys = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(sys.stdout, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
        return
    # table
    rows = data if isinstance(data, list) else [data]
    rows = [_to_jsonable(r) for r in rows if r is not None]
    if not rows:
        print("(no rows)")
        return
    if isinstance(rows[0], dict):
        keys = sorted({k for r in rows for k in r.keys()})
        widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
        header = "  ".join(k.ljust(widths[k]) for k in keys)
        print(header)
        print("  ".join("-" * widths[k] for k in keys))
        for r in rows:
            print("  ".join(str(r.get(k, "") if r.get(k) is not None else "").ljust(widths[k]) for k in keys))
    else:
        for r in rows:
            print(r)


# ============================================================
# Contract builders
# ============================================================

def _build_contract(args: argparse.Namespace) -> Contract:
    kind = args.kind.upper()
    sym = args.symbol
    exch = args.exchange or ("SMART" if kind in ("STK", "OPT") else "")
    ccy = args.currency or "USD"

    if kind == "STK":
        return Stock(sym, exch, ccy, primaryExchange=args.primary or "")
    if kind == "OPT":
        if not (args.expiry and args.right and args.strike):
            raise SystemExit("OPT requires --expiry YYYYMMDD --right C|P --strike N")
        return Option(sym, args.expiry, float(args.strike), args.right, exch or "SMART", multiplier=args.multiplier or "100", currency=ccy)
    if kind == "FUT":
        if not args.expiry:
            raise SystemExit("FUT requires --expiry YYYYMM or YYYYMMDD")
        return Future(sym, args.expiry, exch or "CME", currency=ccy, multiplier=args.multiplier or "")
    if kind == "FX":
        # symbol like "USDJPY" or pair
        return Forex(sym)
    if kind == "IND":
        return Index(sym, exch or "CBOE", ccy)
    raise SystemExit(f"Unknown kind: {kind}")


def _add_contract_args(p: argparse.ArgumentParser, default_kind: str = "STK") -> None:
    p.add_argument("--kind", choices=["STK", "OPT", "FUT", "FX", "IND"], default=default_kind)
    p.add_argument("--symbol", required=True)
    p.add_argument("--exchange", default=None)
    p.add_argument("--primary", default=None, help="STK primaryExchange (e.g. NASDAQ, NYSE)")
    p.add_argument("--currency", default=None)
    p.add_argument("--expiry", default=None, help="OPT: YYYYMMDD; FUT: YYYYMM or YYYYMMDD")
    p.add_argument("--right", choices=["C", "P"], default=None)
    p.add_argument("--strike", default=None)
    p.add_argument("--multiplier", default=None)


# ============================================================
# Commands — connection / meta
# ============================================================

def cmd_connect(args: argparse.Namespace) -> int:
    ib = _connect(args)
    info = {
        "connected": ib.isConnected(),
        "client_id": ib.client.clientId,
        "server_version": ib.client.serverVersion(),
        "server_time": str(ib.reqCurrentTime()),
        "managed_accounts": ib.managedAccounts(),
    }
    _emit(info, args.format)
    ib.disconnect()
    return 0


def cmd_time(args: argparse.Namespace) -> int:
    ib = _connect(args)
    t = ib.reqCurrentTime()
    _emit({"server_time": str(t), "local_time": datetime.now(timezone.utc).isoformat()}, args.format)
    ib.disconnect()
    return 0


def cmd_accounts(args: argparse.Namespace) -> int:
    ib = _connect(args)
    _emit(list(ib.managedAccounts()), args.format)
    ib.disconnect()
    return 0


def cmd_user_info(args: argparse.Namespace) -> int:
    ib = _connect(args)
    try:
        info = ib.reqUserInfo()
    except Exception as exc:
        info = {"error": str(exc)}
    _emit(info, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Account — summary, values, positions, PnL
# ============================================================

def cmd_summary(args: argparse.Namespace) -> int:
    tags = args.tags or "NetLiquidation,TotalCashValue,SettledCash,AccruedCash,BuyingPower,EquityWithLoanValue,GrossPositionValue,InitMarginReq,MaintMarginReq,AvailableFunds,ExcessLiquidity,Cushion,FullInitMarginReq,FullMaintMarginReq,FullAvailableFunds,FullExcessLiquidity,LookAheadNextChange,LookAheadInitMarginReq,LookAheadMaintMarginReq,LookAheadAvailableFunds,LookAheadExcessLiquidity,HighestSeverity,DayTradesRemaining,Leverage,$LEDGER:ALL"
    wanted = {tag.strip() for tag in tags.split(",") if tag.strip()}
    with _ib_session(
        args,
        readonly=True,
        fetch_fields=StartupFetchNONE,
    ) as ib:
        _account_targets(ib, args.account)
        rows = ib.accountSummary(args.account or "")
        out = []
        for item in rows:
            if args.account and item.account != args.account:
                continue
            if (
                "$LEDGER:ALL" not in wanted
                and item.tag not in wanted
                and item.tag != "$LEDGER"
            ):
                continue
            out.append({
                "account": item.account,
                "tag": item.tag,
                "value": item.value,
                "currency": item.currency,
            })
        out.sort(key=lambda row: (row["account"], row["tag"], row["currency"]))
        _emit(out, args.format)
    return 0


def cmd_values(args: argparse.Namespace) -> int:
    values = []
    for account in _discover_account_targets(args):
        # ACCOUNT_UPDATES is the only startup subscription that populates both
        # accountValues and portfolio. connect() bounds the initial download by
        # --timeout and disconnect() reliably unsubscribes it.
        with _ib_session(
            args,
            readonly=True,
            account=account,
            raise_sync_errors=True,
            fetch_fields=StartupFetch.ACCOUNT_UPDATES,
        ) as ib:
            for item in ib.accountValues():
                if item.account != account:
                    continue
                values.append({
                    "account": item.account,
                    "tag": item.tag,
                    "value": item.value,
                    "currency": item.currency,
                    "modelCode": item.modelCode,
                })
    if args.tag:
        needle = args.tag.casefold()
        values = [v for v in values if needle in v["tag"].casefold()]
    values.sort(
        key=lambda row: (
            row["account"], row["tag"], row["currency"], row["modelCode"]
        )
    )
    _emit(values, args.format)
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    with _ib_session(
        args,
        readonly=True,
        fetch_fields=StartupFetchNONE,
    ) as ib:
        accounts = set(_account_targets(ib, args.account))
        out = []
        for position in ib.positions():
            if position.account not in accounts:
                continue
            contract = position.contract
            quantity = _finite_ib_float(position.position)
            avg_cost = _finite_ib_float(position.avgCost)
            cost_basis = (
                quantity * avg_cost
                if quantity is not None and avg_cost is not None
                else None
            )
            out.append({
                "account": position.account,
                "conId": contract.conId,
                "secType": contract.secType,
                "symbol": contract.symbol,
                "localSymbol": contract.localSymbol,
                "currency": contract.currency,
                "exchange": contract.exchange,
                "expiry": (
                    getattr(contract, "lastTradeDateOrContractMonth", "") or ""
                ),
                "right": getattr(contract, "right", "") or "",
                "strike": getattr(contract, "strike", 0.0) or 0.0,
                "multiplier": getattr(contract, "multiplier", "") or "",
                "position": quantity,
                "avgCost": avg_cost,
                # positions() has no live mark. The old implementation called
                # this marketValue, but quantity * average cost is cost basis.
                "costBasis": cost_basis,
            })
        if args.symbol:
            symbol = args.symbol.casefold()
            out = [r for r in out if r["symbol"].casefold() == symbol]
        out.sort(
            key=lambda row: (
                row["account"], row["symbol"], row["secType"], row["localSymbol"]
            )
        )
        _emit(out, args.format)
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    out = []
    for account in _discover_account_targets(args):
        with _ib_session(
            args,
            readonly=True,
            account=account,
            raise_sync_errors=True,
            fetch_fields=StartupFetch.ACCOUNT_UPDATES,
        ) as ib:
            # Filter ourselves instead of relying on portfolio(account), which
            # was only added in ib_async 2.1 and is easy to misuse across
            # multi-account sessions.
            for item in ib.portfolio():
                if item.account != account:
                    continue
                contract = item.contract
                out.append({
                    "account": item.account,
                    "conId": contract.conId,
                    "secType": contract.secType,
                    "symbol": contract.symbol,
                    "localSymbol": contract.localSymbol,
                    "currency": contract.currency,
                    "exchange": contract.exchange,
                    "expiry": (
                        getattr(contract, "lastTradeDateOrContractMonth", "")
                        or ""
                    ),
                    "right": getattr(contract, "right", "") or "",
                    "strike": getattr(contract, "strike", 0.0) or 0.0,
                    "multiplier": getattr(contract, "multiplier", "") or "",
                    "position": item.position,
                    "marketPrice": item.marketPrice,
                    "marketValue": item.marketValue,
                    "avgCost": item.averageCost,
                    "unrealizedPNL": item.unrealizedPNL,
                    "realizedPNL": item.realizedPNL,
                })
    out.sort(
        key=lambda row: (
            row["account"], row["symbol"], row["secType"], row["localSymbol"]
        )
    )
    _emit(out, args.format)
    return 0


def cmd_pnl(args: argparse.Namespace) -> int:
    with _ib_session(
        args,
        readonly=True,
        fetch_fields=StartupFetchNONE,
    ) as ib:
        accounts = _account_targets(ib, args.account)
        subscriptions: list[tuple[str, Any]] = []
        try:
            for account in accounts:
                subscriptions.append((account, ib.reqPnL(account, "")))
            _wait_for_pnl_updates(
                ib,
                [pnl for _account, pnl in subscriptions],
                timeout=args.wait,
            )
            results = [
                {
                    "account": account,
                    "dailyPnL": pnl.dailyPnL,
                    "unrealizedPnL": pnl.unrealizedPnL,
                    "realizedPnL": pnl.realizedPnL,
                }
                for account, pnl in subscriptions
            ]
            _emit(results, args.format)
        finally:
            for account, _pnl in subscriptions:
                try:
                    ib.cancelPnL(account, "")
                except Exception:
                    pass
    return 0


def cmd_pnl_single(args: argparse.Namespace) -> int:
    with _ib_session(
        args,
        readonly=True,
        fetch_fields=StartupFetchNONE,
    ) as ib:
        _account_targets(ib, args.account)
        pnl = ib.reqPnLSingle(args.account, "", args.con_id)
        try:
            _wait_for_pnl_updates(
                ib,
                [pnl],
                timeout=args.wait,
                single=True,
            )
            result = {
                "account": args.account,
                "conId": args.con_id,
                "position": pnl.position,
                "dailyPnL": pnl.dailyPnL,
                "unrealizedPnL": pnl.unrealizedPnL,
                "realizedPnL": pnl.realizedPnL,
                "value": pnl.value,
            }
            _emit(result, args.format)
        finally:
            try:
                ib.cancelPnLSingle(args.account, "", args.con_id)
            except Exception:
                pass
    return 0


# ============================================================
# Quotes / market data
# ============================================================

def _ticker_to_dict(t: Ticker) -> dict:
    g = t.modelGreeks or t.lastGreeks or t.askGreeks or t.bidGreeks
    return {
        "symbol": t.contract.symbol,
        "localSymbol": t.contract.localSymbol or "",
        "secType": t.contract.secType,
        "bid": t.bid, "bidSize": t.bidSize,
        "ask": t.ask, "askSize": t.askSize,
        "last": t.last, "lastSize": t.lastSize,
        "close": t.close, "open": t.open,
        "high": t.high, "low": t.low,
        "volume": t.volume, "vwap": t.vwap,
        "halted": t.halted,
        "delta": getattr(g, "delta", None) if g else None,
        "gamma": getattr(g, "gamma", None) if g else None,
        "vega": getattr(g, "vega", None) if g else None,
        "theta": getattr(g, "theta", None) if g else None,
        "impliedVol": getattr(g, "impliedVol", None) if g else None,
        "undPrice": getattr(g, "undPrice", None) if g else None,
    }


def cmd_quote(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    t = ib.reqMktData(c, "", False, False)
    deadline = time.time() + args.wait
    while time.time() < deadline:
        ib.sleep(0.2)
        if (t.bid and t.ask) or t.last:
            if args.kind == "OPT" and not t.modelGreeks and time.time() < deadline - 0.5:
                continue
            break
    _emit(_ticker_to_dict(t), args.format)
    ib.cancelMktData(c)
    ib.disconnect()
    return 0


def cmd_quotes(args: argparse.Namespace) -> int:
    """Batch snapshot for multiple symbols using reqTickers (one-shot)."""
    ib = _connect(args)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    contracts = [Stock(s, "SMART", args.currency or "USD") for s in symbols]
    ib.qualifyContracts(*contracts)
    tickers = ib.reqTickers(*contracts, regulatorySnapshot=False)
    _emit([_ticker_to_dict(t) for t in tickers], args.format)
    ib.disconnect()
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    t = ib.reqMktData(c, "", False, False)
    end = time.time() + args.duration
    print(f"streaming {c.symbol} for {args.duration}s (Ctrl+C to stop)", file=sys.stderr)
    try:
        while time.time() < end:
            ib.sleep(max(0.1, args.interval))
            row = _ticker_to_dict(t)
            row["ts"] = datetime.now(timezone.utc).isoformat()
            print(json.dumps(_to_jsonable(row), default=str))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    ib.cancelMktData(c)
    ib.disconnect()
    return 0


def cmd_depth(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    t = ib.reqMktDepth(c, numRows=args.rows, isSmartDepth=args.smart)
    ib.sleep(args.wait)
    bids = [{"side": "BID", "pos": b.position, "price": b.price, "size": b.size, "mm": b.marketMaker, "exch": b.exchange} for b in t.domBids]
    asks = [{"side": "ASK", "pos": b.position, "price": b.price, "size": b.size, "mm": b.marketMaker, "exch": b.exchange} for b in t.domAsks]
    _emit(bids + asks, args.format)
    ib.cancelMktDepth(c, isSmartDepth=args.smart)
    ib.disconnect()
    return 0


def cmd_depth_exchanges(args: argparse.Namespace) -> int:
    ib = _connect(args)
    out = [
        {"exchange": d.exchange, "secType": d.secType, "listingExch": d.listingExch, "serviceDataType": d.serviceDataType, "aggGroup": d.aggGroup}
        for d in ib.reqMktDepthExchanges()
    ]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_realtime_bars(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    bars = ib.reqRealTimeBars(c, barSize=5, whatToShow=args.what_to_show, useRTH=args.rth)
    end = time.time() + args.duration
    print(f"realtime bars for {c.symbol} ({args.duration}s)...", file=sys.stderr)
    seen = 0
    while time.time() < end:
        ib.sleep(1.0)
        while seen < len(bars):
            b = bars[seen]
            seen += 1
            row = {"time": str(b.time), "open": b.open_, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume, "wap": b.wap, "count": b.count}
            print(json.dumps(_to_jsonable(row), default=str))
            sys.stdout.flush()
    ib.cancelRealTimeBars(bars)
    ib.disconnect()
    return 0


def cmd_ticks(args: argparse.Namespace) -> int:
    """Tick-by-tick: Last, AllLast, BidAsk, MidPoint. Reads ticker.tickByTicks."""
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    tt = ib.reqTickByTickData(c, tickType=args.tick_type, numberOfTicks=0, ignoreSize=False)
    end = time.time() + args.duration
    print(f"tick-by-tick {args.tick_type} on {c.symbol} ({args.duration}s)...", file=sys.stderr)
    seen = 0
    while time.time() < end:
        ib.sleep(0.25)
        ticks = tt.tickByTicks
        while seen < len(ticks):
            x = ticks[seen]
            seen += 1
            fields = getattr(x, "_fields", None)
            row: dict
            if fields:
                row = {k: getattr(x, k) for k in fields if k != "tickAttribBidAsk" and k != "tickAttribLast"}
            else:
                row = {"tick": repr(x)}
            print(json.dumps(_to_jsonable(row), default=str))
            sys.stdout.flush()
    ib.cancelTickByTickData(c, args.tick_type)
    ib.disconnect()
    return 0


# ============================================================
# Historical
# ============================================================

def cmd_history(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    bars = ib.reqHistoricalData(
        c,
        endDateTime=args.end or "",
        durationStr=args.duration,
        barSizeSetting=args.bar_size,
        whatToShow=args.what_to_show,
        useRTH=args.rth,
        formatDate=1,
        keepUpToDate=False,
    )
    out = [{"time": str(b.date), "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume, "wap": b.average, "count": b.barCount} for b in bars]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_history_ticks(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    ticks = ib.reqHistoricalTicks(
        c,
        startDateTime=args.start or "",
        endDateTime=args.end or "",
        numberOfTicks=args.count,
        whatToShow=args.what_to_show,
        useRth=args.rth,
    )
    out = []
    for x in ticks:
        row = {"time": str(x.time)}
        for k in ("price", "size", "priceBid", "sizeBid", "priceAsk", "sizeAsk", "exchange"):
            if hasattr(x, k):
                row[k] = getattr(x, k)
        out.append(row)
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_head_timestamp(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    ts = ib.reqHeadTimeStamp(c, whatToShow=args.what_to_show, useRTH=args.rth, formatDate=1)
    _emit({"symbol": c.symbol, "head_timestamp": str(ts)}, args.format)
    ib.disconnect()
    return 0


def cmd_histogram(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    hist = ib.reqHistogramData(c, useRTH=args.rth, period=args.period)
    _emit([{"price": h.price, "count": h.count} for h in hist], args.format)
    ib.disconnect()
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    sched = ib.reqHistoricalSchedule(c, endDateTime=args.end or "", durationStr=args.duration, useRTH=args.rth)
    out = [{"start": str(s.startDateTime), "end": str(s.endDateTime), "refDate": str(s.refDate), "timezone": s.timeZone} for s in sched.sessions]
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Options
# ============================================================

def cmd_chain(args: argparse.Namespace) -> int:
    ib = _connect(args)
    underlying = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    [u] = ib.qualifyContracts(underlying)
    params = ib.reqSecDefOptParams(u.symbol, "", u.secType, u.conId)
    out = []
    for p in params:
        out.append({
            "exchange": p.exchange,
            "underlyingConId": p.underlyingConId,
            "tradingClass": p.tradingClass,
            "multiplier": p.multiplier,
            "expirations": sorted(p.expirations),
            "strikes": sorted(p.strikes),
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_chain_quote(args: argparse.Namespace) -> int:
    """Snapshot all strikes for a given expiry, both rights."""
    ib = _connect(args)
    underlying = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    [u] = ib.qualifyContracts(underlying)
    params = ib.reqSecDefOptParams(u.symbol, "", u.secType, u.conId)
    if not params:
        _emit({"error": "no chain params"}, args.format)
        ib.disconnect()
        return 1
    # pick SMART exchange param if possible
    pp = next((p for p in params if p.exchange == "SMART"), params[0])
    strikes = sorted(pp.strikes)
    if args.min_strike is not None:
        strikes = [s for s in strikes if s >= args.min_strike]
    if args.max_strike is not None:
        strikes = [s for s in strikes if s <= args.max_strike]
    if args.expiry not in pp.expirations:
        print(f"WARN: expiry {args.expiry} not in chain {sorted(pp.expirations)[:5]}...", file=sys.stderr)

    rights = ["C", "P"] if args.rights == "both" else [args.rights]
    opts = []
    for s in strikes:
        for r in rights:
            opts.append(Option(args.symbol, args.expiry, s, r, "SMART", tradingClass=pp.tradingClass, multiplier=pp.multiplier, currency=args.currency or "USD"))
    ib.qualifyContracts(*opts)
    # Request market data for all
    tickers = [ib.reqMktData(o, "", False, False) for o in opts if o.conId]
    ib.sleep(args.wait)
    out = []
    for t in tickers:
        d = _ticker_to_dict(t)
        d["strike"] = t.contract.strike
        d["right"] = t.contract.right
        d["expiry"] = t.contract.lastTradeDateOrContractMonth
        out.append(d)
    for o in opts:
        if o.conId:
            ib.cancelMktData(o)
    out.sort(key=lambda r: (r.get("right", ""), r.get("strike", 0)))
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_calc_iv(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    iv = ib.calculateImpliedVolatility(c, optionPrice=args.option_price, underPrice=args.under_price)
    _emit(_ticker_to_dict(iv) if iv else None, args.format)
    ib.disconnect()
    return 0


def cmd_calc_price(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    p = ib.calculateOptionPrice(c, volatility=args.vol, underPrice=args.under_price)
    _emit(_ticker_to_dict(p) if p else None, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Discovery / contract details
# ============================================================

def cmd_search(args: argparse.Namespace) -> int:
    ib = _connect(args)
    matches = ib.reqMatchingSymbols(args.query)
    out = []
    for m in matches:
        c = m.contract
        out.append({
            "conId": c.conId,
            "symbol": c.symbol,
            "secType": c.secType,
            "primaryExchange": c.primaryExchange,
            "currency": c.currency,
            "description": getattr(m, "description", ""),
            "derivativeSecTypes": ",".join(m.derivativeSecTypes or []),
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_contract(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    details = ib.reqContractDetails(c)
    out = []
    for d in details:
        out.append({
            "conId": d.contract.conId,
            "symbol": d.contract.symbol,
            "localSymbol": d.contract.localSymbol,
            "secType": d.contract.secType,
            "exchange": d.contract.exchange,
            "primaryExchange": d.contract.primaryExchange,
            "currency": d.contract.currency,
            "longName": d.longName,
            "industry": d.industry,
            "category": d.category,
            "subcategory": d.subcategory,
            "timeZoneId": d.timeZoneId,
            "tradingHours": d.tradingHours,
            "liquidHours": d.liquidHours,
            "minTick": d.minTick,
            "orderTypes": d.orderTypes,
            "validExchanges": d.validExchanges,
            "priceMagnifier": d.priceMagnifier,
            "underConId": d.underConId,
            "underSymbol": d.underSymbol,
            "underSecType": d.underSecType,
            "marketRuleIds": d.marketRuleIds,
            "secIdList": [(t.tag, t.value) for t in (d.secIdList or [])],
            "stockType": d.stockType,
            "contractMonth": d.contractMonth,
            "lastTradeTime": d.lastTradeTime,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_smart_components(args: argparse.Namespace) -> int:
    ib = _connect(args)
    comps = ib.reqSmartComponents(args.bbo_exchange)
    out = [{"bitNumber": k, "exchange": v.exchange, "exchangeLetter": v.exchangeLetter} for k, v in comps.items()]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_market_rule(args: argparse.Namespace) -> int:
    ib = _connect(args)
    rule = ib.reqMarketRule(args.rule_id)
    out = [{"lowEdge": inc.lowEdge, "increment": inc.increment} for inc in (rule or [])]
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Fundamentals
# ============================================================

def cmd_fundamentals(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    ib.qualifyContracts(c)
    data = ib.reqFundamentalData(c, args.report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(data or "")
        print(f"wrote {args.out} ({len(data or '')} bytes)")
    else:
        print(data or "")
    ib.disconnect()
    return 0


# ============================================================
# News
# ============================================================

def cmd_news_providers(args: argparse.Namespace) -> int:
    ib = _connect(args)
    provs = ib.reqNewsProviders()
    _emit([{"code": p.code, "name": p.name} for p in provs], args.format)
    ib.disconnect()
    return 0


def cmd_news(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    [c] = ib.qualifyContracts(c)
    start = args.start or (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S.0")
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.0")
    providers = args.providers or "BRFG+BRFUPDN+DJNL+DJ-RT"
    items = ib.reqHistoricalNews(c.conId, providers, start, end, args.count)
    out = []
    for h in items:
        out.append({
            "time": str(h.time),
            "providerCode": h.providerCode,
            "articleId": h.articleId,
            "headline": h.headline,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_article(args: argparse.Namespace) -> int:
    ib = _connect(args)
    art = ib.reqNewsArticle(args.provider, args.article_id)
    out = {"articleType": art.articleType, "articleText": art.articleText}
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_news_bulletins(args: argparse.Namespace) -> int:
    ib = _connect(args)
    ib.reqNewsBulletins(allMessages=args.all_messages)
    end = time.time() + args.duration
    print(f"news bulletins for {args.duration}s...", file=sys.stderr)
    seen = 0
    while time.time() < end:
        ib.sleep(1.0)
        bulletins = ib.newsBulletins()
        while seen < len(bulletins):
            b = bulletins[seen]
            seen += 1
            print(json.dumps(_to_jsonable({
                "msgId": b.msgId, "msgType": b.msgType, "message": b.message, "origExchange": b.origExchange,
            }), default=str))
            sys.stdout.flush()
    ib.cancelNewsBulletins()
    ib.disconnect()
    return 0


# ============================================================
# Scanner
# ============================================================

def cmd_scanner_params(args: argparse.Namespace) -> int:
    ib = _connect(args)
    xml = ib.reqScannerParameters()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(xml)
        print(f"wrote {args.out} ({len(xml)} bytes)")
    else:
        print(xml)
    ib.disconnect()
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    ib = _connect(args)
    sub = ScannerSubscription(
        instrument=args.instrument,
        locationCode=args.location,
        scanCode=args.scan_code,
        numberOfRows=args.count,
    )
    if args.above_price is not None:
        sub.abovePrice = args.above_price
    if args.below_price is not None:
        sub.belowPrice = args.below_price
    if args.above_volume is not None:
        sub.aboveVolume = args.above_volume
    if args.market_cap_above is not None:
        sub.marketCapAbove = args.market_cap_above
    if args.market_cap_below is not None:
        sub.marketCapBelow = args.market_cap_below
    filters = []
    if args.filters:
        for kv in args.filters.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                filters.append(TagValue(k.strip(), v.strip()))
    results = ib.reqScannerData(sub, [], filters)
    out = []
    for r in results:
        c = r.contractDetails.contract
        out.append({
            "rank": r.rank,
            "symbol": c.symbol,
            "conId": c.conId,
            "secType": c.secType,
            "primaryExchange": c.primaryExchange,
            "currency": c.currency,
            "distance": r.distance,
            "benchmark": r.benchmark,
            "projection": r.projection,
            "legsStr": r.legsStr,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Orders — list / what-if / place / cancel
# ============================================================

def _order_dict(trade: Any) -> dict:
    o = trade.order
    c = trade.contract
    s = trade.orderStatus
    return {
        "orderId": o.orderId,
        "permId": o.permId,
        "account": o.account,
        "action": o.action,
        "totalQuantity": o.totalQuantity,
        "orderType": o.orderType,
        "lmtPrice": o.lmtPrice,
        "auxPrice": o.auxPrice,
        "tif": o.tif,
        "symbol": c.symbol,
        "secType": c.secType,
        "localSymbol": c.localSymbol,
        "expiry": getattr(c, "lastTradeDateOrContractMonth", ""),
        "right": getattr(c, "right", ""),
        "strike": getattr(c, "strike", 0.0),
        "status": s.status,
        "filled": s.filled,
        "remaining": s.remaining,
        "avgFillPrice": s.avgFillPrice,
        "whyHeld": s.whyHeld,
    }


def cmd_open_orders(args: argparse.Namespace) -> int:
    ib = _connect(args)
    if args.all:
        trades = ib.reqAllOpenOrders()
    else:
        trades = ib.reqOpenOrders()
    ib.sleep(1.0)
    out = [_order_dict(t) for t in trades]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_completed_orders(args: argparse.Namespace) -> int:
    ib = _connect(args)
    trades = ib.reqCompletedOrders(apiOnly=args.api_only)
    ib.sleep(1.0)
    out = [_order_dict(t) for t in trades]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_executions(args: argparse.Namespace) -> int:
    ib = _connect(args)
    f = ExecutionFilter()
    if args.account:
        f.acctCode = args.account
    if args.client_id_filter is not None:
        f.clientId = args.client_id_filter
    if args.symbol:
        f.symbol = args.symbol
    if args.sec_type:
        f.secType = args.sec_type
    if args.exchange_filter:
        f.exchange = args.exchange_filter
    if args.side:
        f.side = args.side
    if args.time:
        f.time = args.time
    fills = ib.reqExecutions(f)
    out = []
    for x in fills:
        c = x.contract
        e = x.execution
        cr = x.commissionReport
        out.append({
            "time": str(e.time),
            "account": e.acctNumber,
            "symbol": c.symbol,
            "secType": c.secType,
            "localSymbol": c.localSymbol,
            "side": e.side,
            "shares": e.shares,
            "price": e.price,
            "exchange": e.exchange,
            "orderId": e.orderId,
            "permId": e.permId,
            "execId": e.execId,
            "liquidation": e.liquidation,
            "cumQty": e.cumQty,
            "avgPrice": e.avgPrice,
            "commission": cr.commission if cr else None,
            "commissionCurrency": cr.currency if cr else None,
            "realizedPNL": cr.realizedPNL if cr else None,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def _build_order(args: argparse.Namespace) -> Order:
    o = Order()
    o.action = args.side
    o.totalQuantity = float(args.qty)
    o.orderType = args.type
    if args.limit_price is not None:
        o.lmtPrice = float(args.limit_price)
    if args.stop_price is not None:
        o.auxPrice = float(args.stop_price)
    o.tif = args.tif
    if args.account:
        o.account = args.account
    if args.order_ref:
        o.orderRef = args.order_ref
    if args.oca_group:
        o.ocaGroup = args.oca_group
    if args.outside_rth:
        o.outsideRth = True
    o.transmit = not args.no_transmit
    return o


def cmd_whatif(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    o = _build_order(args)
    if not o.account:
        accts = ib.managedAccounts()
        if not accts:
            raise SystemExit("No account specified and no managed accounts visible.")
        o.account = accts[0]
    st = ib.whatIfOrder(c, o)
    out = {
        "account": o.account,
        "status": st.status,
        "initMarginBefore": st.initMarginBefore,
        "initMarginChange": st.initMarginChange,
        "initMarginAfter": st.initMarginAfter,
        "maintMarginBefore": st.maintMarginBefore,
        "maintMarginChange": st.maintMarginChange,
        "maintMarginAfter": st.maintMarginAfter,
        "equityWithLoanBefore": st.equityWithLoanBefore,
        "equityWithLoanChange": st.equityWithLoanChange,
        "equityWithLoanAfter": st.equityWithLoanAfter,
        "commission": st.commission,
        "minCommission": st.minCommission,
        "maxCommission": st.maxCommission,
        "commissionCurrency": st.commissionCurrency,
        "warningText": st.warningText,
    }
    _emit(out, args.format)
    ib.disconnect()
    return 0


def _resolve_account(ib: IB, explicit: str | None) -> str:
    """Pick the order account: --account > IBKR_ACCOUNT env > sole managed
    account. Fail loudly if ambiguous (multiple accounts linked to login).

    IBKR Error 435 ("You must specify an account") fires when an order
    omits the account field on a multi-account login.
    """
    if explicit:
        return explicit
    env_acct = os.getenv("IBKR_ACCOUNT")
    if env_acct:
        return env_acct
    managed = list(ib.managedAccounts() or [])
    if len(managed) == 1:
        return managed[0]
    if not managed:
        raise SystemExit("No managed accounts visible via managedAccounts().")
    raise SystemExit(
        f"Multiple accounts linked ({managed}). Specify --account <ID> "
        f"or set IBKR_ACCOUNT env. IBKR rejects order submissions without "
        f"an explicit account on multi-account logins (Error 435)."
    )


def _wait_for_order_ack(ib: IB, trade: Any, timeout: float = 15.0) -> bool:
    """Block until the order reaches a stable state or times out.

    Returns True if the order is live (Submitted/PreSubmitted/Filled) or
    False if it was Cancelled/Inactive. Surfaces errors from trade.log.

    Replaces the naive `ib.sleep(2.0)` which disconnected before IBKR
    could deliver rejection notifications (Error 435, risk rejects, etc.).
    """
    terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
    stable_live = {"Submitted", "PreSubmitted"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        ib.waitOnUpdate(timeout=0.5)
        status = trade.orderStatus.status
        if status in terminal or status in stable_live:
            break
    # Surface any errors recorded on the trade log.
    errors = [e for e in trade.log if e.errorCode]
    for e in errors:
        print(f"IBKR Error {e.errorCode}: {e.message}", file=sys.stderr)
    return trade.orderStatus.status not in ("Cancelled", "ApiCancelled", "Inactive")


# ============================================================
# Constitution gate — the verdict WRITER (read-only) + Family B clearance
# ============================================================

def _connect_readonly(args: argparse.Namespace) -> IB:
    """Data-only connection for the constitution check: readonly=True means the
    API session CANNOT transmit orders even by bug."""
    ib = IB()
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT
    client_id = args.client_id if args.client_id is not None else DEFAULT_CLIENT_ID
    ib.connect(host, port, clientId=client_id, timeout=args.timeout, readonly=True)
    ib.reqMarketDataType(args.market_data_type or 1)
    return ib


def _make_usd_rate(ib: IB, *, wait: float = 4.0) -> Callable[[str | None], float | None]:
    """ccy -> USD conversion via IDEALPRO marks (delayed is fine for ratios).
    Everything the gate compares against NetLiq is normalized to USD."""
    cache: dict[str, float | None] = {"USD": 1.0}

    def _pair_mark(pair: str) -> float | None:
        try:
            f = Forex(pair)
            qualified = ib.qualifyContracts(f)
            if not qualified:
                return None
            t = ib.reqMktData(qualified[0], "", False, False)
            deadline = time.time() + wait
            px = None
            while time.time() < deadline:
                ib.sleep(0.25)
                px = _safe_num(t.marketPrice()) or _safe_num(t.close)
                if px is not None:
                    break
            ib.cancelMktData(qualified[0])
            return px
        except Exception:
            return None

    def rate(ccy: str | None) -> float | None:
        c = (ccy or "USD").upper()
        if c in cache:
            return cache[c]
        r = _pair_mark(c + "USD")  # direct (e.g. EURUSD)
        if r is None:
            inv = _pair_mark("USD" + c)  # inverse (e.g. USDCAD)
            r = (1.0 / inv) if inv else None
        cache[c] = r
        return r

    return rate


def _combined_net_liq_usd(ib: IB, usd_rate) -> float | None:
    """Combined NetLiquidation across ALL managed accounts, in USD — the
    constitution's caps are written against the combined family book."""
    total = 0.0
    seen = False
    try:
        for row in ib.accountSummary():
            if row.tag != "NetLiquidation":
                continue
            try:
                v = float(row.value)
            except (TypeError, ValueError):
                continue
            r = usd_rate(getattr(row, "currency", "USD"))
            if r is None:
                return None  # can't normalize -> unknown (rules SKIP, verdict incomplete)
            total += v * r
            seen = True
    except Exception:
        return None
    return total if seen and total > 0 else None


def _positions_as_base(ib: IB) -> tuple[list, dict]:
    """ib.positions() (all accounts) -> (trading_algo Positions, contract map).
    The map keys each position's OWN qualified-able contract + currency so
    quoting uses the real listing (VFV=TSE/CAD etc.), never a USD guess."""
    from trading_algo.broker.base import Position as BasePosition
    from trading_algo.instruments import InstrumentSpec

    kind_map = {"STK": "STK", "OPT": "OPT", "FUT": "FUT", "CASH": "FX"}
    out: list = []
    contract_map: dict = {}
    for p in ib.positions():
        c = p.contract
        kind = kind_map.get(getattr(c, "secType", ""), getattr(c, "secType", "STK"))
        ccy = (getattr(c, "currency", "") or "USD").upper()
        sym = c.symbol.upper()
        expiry = getattr(c, "lastTradeDateOrContractMonth", "") or None
        right = (getattr(c, "right", "") or "").upper() or None
        strike = float(c.strike) if getattr(c, "strike", 0) else None
        if kind == "STK":
            contract_map[("STK", sym)] = (c, ccy)
        elif kind == "OPT":
            contract_map[("OPT", sym, expiry, right, strike)] = (c, ccy)
        out.append(BasePosition(
            account=p.account,
            instrument=InstrumentSpec(
                kind=kind, symbol=sym, currency=ccy,
                expiry=expiry, right=right, strike=strike,
                multiplier=(str(c.multiplier) if getattr(c, "multiplier", "") else None),
            ),
            quantity=float(p.position),
            avg_cost=(float(p.avgCost) if p.avgCost is not None else None),
            timestamp_epoch_s=time.time(),
        ))
    return out, contract_map


def _safe_num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f) or f == -1.0:
        return None
    return f


def _make_gateway_providers(ib: IB, *, wait: float, contract_map: dict, usd_rate):
    """greeks/spot providers over the live gateway, USD-normalized. All position
    contracts are BATCH-prefetched (one subscription sweep, one shared wait) —
    sequential subscribe/cancel cycles were shown to drop greeks under pacing.
    The trade's own contract is fetched on demand. Any persistent miss returns
    None (the adapter marks the verdict incomplete — fail-closed)."""
    from trading_algo.constitution_adapter import OptionGreeks

    greeks_cache: dict[tuple, OptionGreeks | None] = {}
    spot_cache: dict[str, float | None] = {}

    def _qualified(c) -> Any | None:
        try:
            q = ib.qualifyContracts(c)
            return q[0] if q else None
        except Exception:
            return None

    def _opt_ready(t) -> bool:
        return bool(t.modelGreeks or t.lastGreeks)

    def _stk_px(t) -> float | None:
        return _safe_num(t.last) or _safe_num(t.marketPrice()) or _safe_num(t.close)

    def _harvest_opt(t, ccy) -> OptionGreeks | None:
        g = t.modelGreeks or t.lastGreeks or t.askGreeks or t.bidGreeks
        r = usd_rate(ccy)
        if g is None or r is None:
            return None
        opt_px = _safe_num(g.optPrice)
        und_px = _safe_num(g.undPrice)
        return OptionGreeks(
            delta=_safe_num(g.delta), iv=_safe_num(g.impliedVol),
            opt_price=opt_px * r if opt_px is not None else None,
            und_price=und_px * r if und_px is not None else None,
            bid=_safe_num(t.bid), ask=_safe_num(t.ask),
        )

    # ---- batch prefetch of every position contract ----
    subs: list[tuple[tuple, Any, str, Any]] = []  # (map_key, ticker, ccy, qualified)
    try:
        for key, (c, ccy) in contract_map.items():
            q = _qualified(c)
            if q is not None:
                subs.append((key, ib.reqMktData(q, "", False, False), ccy, q))
        deadline = time.time() + max(wait, 10.0)
        while time.time() < deadline:
            ib.sleep(0.3)
            pending = [
                s for s in subs
                if (s[0][0] == "OPT" and not _opt_ready(s[1]))
                or (s[0][0] == "STK" and _stk_px(s[1]) is None)
            ]
            if not pending:
                break
        for key, t, ccy, q in subs:
            if key[0] == "OPT":
                greeks_cache[key[1:]] = _harvest_opt(t, ccy)
            else:
                px = _stk_px(t)
                r = usd_rate(ccy)
                spot_cache[key[1]] = px * r if (px is not None and r is not None) else None
            try:
                ib.cancelMktData(q)
            except Exception:
                pass
    except Exception:
        pass  # missing entries stay unfetched -> providers fall back / stay None

    def greeks_provider(spec) -> OptionGreeks | None:
        key = (spec.symbol.upper(), spec.expiry, (spec.right or "").upper(),
               float(spec.strike) if spec.strike is not None else None)
        if key in greeks_cache:
            return greeks_cache[key]
        result = None
        ccy = ((spec.currency or "USD").upper()
               if getattr(spec, "currency", None) else "USD")
        q = _qualified(Option(key[0], spec.expiry, key[3], key[2], "SMART", currency=ccy))
        if q is not None:
            try:
                t = ib.reqMktData(q, "", False, False)
                deadline = time.time() + wait
                while time.time() < deadline:
                    ib.sleep(0.25)
                    if _opt_ready(t):
                        break
                result = _harvest_opt(t, ccy)
                ib.cancelMktData(q)
            except Exception:
                result = None
        greeks_cache[key] = result
        return result

    def spot_provider(symbol: str) -> float | None:
        sym = symbol.upper()
        if sym in spot_cache:
            return spot_cache[sym]
        px = None
        q = _qualified(Stock(sym, "SMART", "USD"))
        if q is not None:
            try:
                t = ib.reqMktData(q, "", False, False)
                deadline = time.time() + min(wait, 4.0)
                while time.time() < deadline:
                    ib.sleep(0.25)
                    px = _stk_px(t)
                    if px is not None:
                        break
                ib.cancelMktData(q)
            except Exception:
                px = None
        spot_cache[sym] = px
        return px

    return greeks_provider, spot_provider


class _GatewaySnapshotBroker:
    """Minimal duck-typed broker for build_eval_context: a frozen snapshot of
    the gateway's account + positions (read-only, no order capability)."""

    def __init__(self, account: str, net_liq: float | None, positions: list) -> None:
        from trading_algo.broker.base import AccountSnapshot
        values = {"NetLiquidation": net_liq} if net_liq is not None else {}
        self._snap = AccountSnapshot(account=account, values=values,
                                     timestamp_epoch_s=time.time())
        self._positions = positions

    def get_account_snapshot(self):
        return self._snap

    def get_positions(self):
        return self._positions


def _parse_legs(spec: str) -> list[tuple[str, int, int]]:
    """'BUY:conId:ratio,SELL:conId:ratio' -> [(action, conId, ratio), ...]"""
    legs: list[tuple[str, int, int]] = []
    for leg_spec in spec.split(","):
        parts = leg_spec.strip().split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad leg spec '{leg_spec}' (want ACTION:conId:ratio)")
        action, con_id, ratio = parts
        action = action.upper()
        if action not in {"BUY", "SELL"}:
            raise SystemExit(f"bad leg action '{action}' (want BUY|SELL)")
        legs.append((action, int(con_id), int(ratio)))
    return legs


def _effective_leg_action(order_side: str, leg_action: str) -> str:
    """IBKR semantics: SELLING a BAG reverses every leg's action."""
    if order_side.upper() == "BUY":
        return leg_action
    return "SELL" if leg_action == "BUY" else "BUY"


def _resolve_leg_contracts(ib: IB, legs: list[tuple[str, int, int]]) -> list[dict]:
    """conId -> full contract fields, via the read-only session."""
    out = []
    for action, con_id, ratio in legs:
        q = ib.qualifyContracts(Contract(conId=con_id))
        if not q:
            raise SystemExit(f"cannot resolve combo leg conId {con_id}")
        c = q[0]
        out.append({
            "action": action, "con_id": con_id, "ratio": ratio,
            "kind": getattr(c, "secType", "OPT"), "symbol": c.symbol.upper(),
            "expiry": getattr(c, "lastTradeDateOrContractMonth", "") or None,
            "right": (getattr(c, "right", "") or "").upper() or None,
            "strike": float(c.strike) if getattr(c, "strike", 0) else None,
        })
    return out


def cmd_constitution_check(args: argparse.Namespace) -> int:
    """Evaluate the constitution against LIVE account state and persist the
    verdict (the clearance a later `place`/`combo` transmits against). READ-ONLY."""
    from trading_algo.config import TradingConfig
    from trading_algo.persistence import SqliteStore

    ib = _connect_readonly(args)
    try:
        account = _resolve_account(ib, args.account)
        usd_rate = _make_usd_rate(ib)
        net_liq = _combined_net_liq_usd(ib, usd_rate)
        positions, contract_map = _positions_as_base(ib)
        greeks_provider, spot_provider = _make_gateway_providers(
            ib, wait=float(args.greeks_wait), contract_map=contract_map,
            usd_rate=usd_rate)

        db_path = args.db_path or TradingConfig.from_env().db_path
        store = SqliteStore(db_path) if db_path else None
        shim = _GatewaySnapshotBroker(account, net_liq, positions)
        try:
            if getattr(args, "legs", None):
                decision, complete, missing, key, checks_out, extras = _check_combo(
                    ib, args, account=account, store=store, shim=shim,
                    greeks_provider=greeks_provider, spot_provider=spot_provider,
                )
            else:
                decision, complete, missing, key, checks_out, extras = _check_single(
                    args, account=account, store=store, shim=shim,
                    greeks_provider=greeks_provider, spot_provider=spot_provider,
                )
        finally:
            if store is not None:
                store.close()

        out = {
            "decision": decision,
            "complete": complete,
            "missing": missing,
            "order_key": key,
            "account": account,
            "combined_net_liq": net_liq,
            "persisted": db_path is not None,
            "valid_for_s": TradingConfig.from_env().constitution_max_age_s,
            **extras,
            "checks": checks_out,
        }
        _emit(out, args.format)
        if decision == "BLOCK":
            print("BLOCKED — this order violates the constitution.", file=sys.stderr)
            return 2
        if not complete:
            print(f"INCOMPLETE (missing: {', '.join(missing)}) — "
                  "a transmit against this verdict will be refused.", file=sys.stderr)
            return 2
        return 0
    finally:
        ib.disconnect()


def _rules_out(checks) -> list[dict]:
    return [
        {"rule": c.rule_id, "severity": c.severity, "status": c.status,
         "observed": c.observed, "message": c.message}
        for c in checks if c.status != "SKIP" or c.confidence == "LOW"
    ]


def _check_single(args, *, account, store, shim, greeks_provider, spot_provider):
    from trading_algo.constitution import evaluate
    from trading_algo.constitution_adapter import (
        ProposedOrderInput, build_eval_context, record_verdict,
    )
    losing_30d = None
    if store is not None:
        losing_30d = store.recent_losing_put_close(args.symbol.upper(), within_s=30 * 86400)
    proposed = ProposedOrderInput(
        symbol=args.symbol, kind=args.kind, side=args.side,
        quantity=float(args.qty), account=account,
        right=args.right,
        strike=float(args.strike) if args.strike is not None else None,
        expiry=args.expiry, order_type=args.type,
        limit_price=args.limit_price,
        structure=args.structure, credit=args.credit,
        is_new_program=bool(args.new_program),
        written_exit=args.written_exit,
        is_roll=bool(args.is_roll),
        losing_put_close_same_underlying_30d=losing_30d,
    )
    ctx, meta = build_eval_context(shim, proposed, greeks_provider=greeks_provider,
                                   spot_provider=spot_provider)
    verdict = evaluate(ctx)
    if store is not None:
        record_verdict(store, verdict, meta, proposed)
    extras = {"closes_long": ctx.trade.closes_long, "closes_short": ctx.trade.closes_short,
              "structure": ctx.trade.structure}
    return (verdict.decision, meta.complete, meta.missing, proposed.to_key(),
            _rules_out(verdict.checks), extras)


def _check_combo(ib, args, *, account, store, shim, greeks_provider, spot_provider):
    """Per-leg constitution evaluation for a BAG order; ONE verdict persisted
    under the canonical leg-set key. Verdict = worst leg; complete = all legs."""
    from trading_algo.constitution import combo_key, evaluate
    from trading_algo.constitution_adapter import ProposedOrderInput, build_eval_context

    legs = _parse_legs(args.legs)
    resolved = _resolve_leg_contracts(ib, legs)
    for r in resolved:
        r["eff_action"] = _effective_leg_action(args.side, r["action"])
    has_long_put = any(r["eff_action"] == "BUY" and r["right"] == "P" for r in resolved)
    has_long_call = any(r["eff_action"] == "BUY" and r["right"] == "C" for r in resolved)

    rank = {"PASS": 0, "WARN": 1, "BLOCK": 2}
    decision = "PASS"
    complete = True
    missing: list[str] = []
    checks_out: list[dict] = []
    leg_summary: list[dict] = []
    produced_at = None
    for i, r in enumerate(resolved):
        structure = None
        # A short leg PAIRED with a long same-right leg is a defined-risk
        # spread: the naked-put rules (C8 entry gate, assignment caps) don't
        # apply, but TFSA (C6) and the DTE cap (C4) still do.
        if r["eff_action"] == "SELL" and r["right"] == "P" and has_long_put:
            structure = "put-credit-spread"
        elif r["eff_action"] == "SELL" and r["right"] == "C" and has_long_call:
            structure = "call-credit-spread"
        losing_30d = None
        if store is not None:
            losing_30d = store.recent_losing_put_close(r["symbol"], within_s=30 * 86400)
        lp = ProposedOrderInput(
            symbol=r["symbol"], kind=r["kind"], side=r["eff_action"],
            quantity=float(args.qty) * r["ratio"], account=account,
            right=r["right"], strike=r["strike"], expiry=r["expiry"],
            order_type=args.type, structure=structure,
            written_exit=args.written_exit, is_roll=bool(args.is_roll),
            losing_put_close_same_underlying_30d=losing_30d,
        )
        ctx, meta = build_eval_context(shim, lp, greeks_provider=greeks_provider,
                                       spot_provider=spot_provider)
        v = evaluate(ctx)
        if produced_at is None:
            produced_at = meta.produced_at
        if rank[v.decision] > rank[decision]:
            decision = v.decision
        complete = complete and meta.complete
        missing.extend(f"leg{i}:{m}" for m in meta.missing)
        for c in _rules_out(v.checks):
            checks_out.append({**c, "leg": i, "leg_desc":
                               f"{r['eff_action']} {r['symbol']} {r['right'] or ''}{r['strike'] or ''} x{r['ratio']}"})
        leg_summary.append({"leg": i, "conId": r["con_id"], "desc":
                            f"{r['eff_action']} {r['symbol']} {r['right'] or ''}{r['strike'] or ''} "
                            f"{r['expiry'] or ''} x{float(args.qty) * r['ratio']:g}",
                            "structure": ctx.trade.structure,
                            "closes_long": ctx.trade.closes_long,
                            "closes_short": ctx.trade.closes_short,
                            "decision": v.decision, "complete": meta.complete})

    key = combo_key(legs=legs, side=args.side, quantity=float(args.qty),
                    symbol=args.symbol, account=account,
                    limit_price=args.limit_price, order_type=args.type)
    if store is not None:
        store.log_constitution_verdict(
            order_key=key, decision=decision, complete=complete, checks=checks_out,
            symbol=args.symbol.upper(), account=account,
            context={"missing": missing, "legs": leg_summary},
            ts_epoch_s=produced_at,
        )
    return decision, complete, missing, key, checks_out, {"legs": leg_summary}


def _require_family_b_clearance(
    *, symbol: str, kind: str, side: str, qty: float, right: str | None,
    strike: float | None, expiry: str | None, account: str,
    limit_price: float | None, order_type: str, order_ref: str,
) -> None:
    """Family B chokepoint (raw ib.placeOrder path): same fail-closed clearance
    as the OMS/broker sites. No-op unless TRADING_CONSTITUTION_REQUIRED=true."""
    from trading_algo.config import TradingConfig
    from trading_algo.constitution_clearance import verify_clearance
    from trading_algo.persistence import SqliteStore

    cfg = TradingConfig.from_env()
    if not cfg.constitution_required:
        return
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    try:
        verify_clearance(
            store, symbol=symbol, kind=kind, side=side, quantity=qty,
            right=right, strike=strike, expiry=expiry, account=account,
            limit_price=limit_price, order_type=order_type,
            required=True, max_age_s=cfg.constitution_max_age_s,
            order_ref=order_ref,
        )
    finally:
        if store is not None:
            store.close()


def _require_combo_clearance(
    *, legs: list[tuple[str, int, int]], side: str, qty: float, symbol: str,
    account: str, limit_price: float | None, order_type: str, order_ref: str,
) -> None:
    """Family B chokepoint for BAG orders, keyed by the canonical leg set."""
    from trading_algo.config import TradingConfig
    from trading_algo.constitution_clearance import verify_combo_clearance
    from trading_algo.persistence import SqliteStore

    cfg = TradingConfig.from_env()
    if not cfg.constitution_required:
        return
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    try:
        verify_combo_clearance(
            store, legs=legs, side=side, quantity=qty, symbol=symbol,
            account=account, limit_price=limit_price, order_type=order_type,
            required=True, max_age_s=cfg.constitution_max_age_s,
            order_ref=order_ref,
        )
    finally:
        if store is not None:
            store.close()


def cmd_place(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    if not args.yes:
        raise SystemExit("Refusing to place live order without --yes confirmation flag.")
    ib = _connect(args)
    try:
        c = _build_contract(args)
        ib.qualifyContracts(c)
        o = _build_order(args)
        o.account = _resolve_account(ib, o.account or None)
        # Constitution clearance (fail-closed when TRADING_CONSTITUTION_REQUIRED=true).
        # The claim binds to orderRef, so ensure one exists (visible in the IBKR log).
        if not o.orderRef:
            import uuid as _uuid
            o.orderRef = f"TA{_uuid.uuid4().hex[:18]}"
        _require_family_b_clearance(
            symbol=args.symbol.upper(), kind=args.kind, side=args.side,
            qty=float(args.qty), right=args.right,
            strike=float(args.strike) if args.strike is not None else None,
            expiry=args.expiry, account=o.account,
            limit_price=args.limit_price, order_type=args.type, order_ref=o.orderRef,
        )
        trade = ib.placeOrder(c, o)
        ok = _wait_for_order_ack(ib, trade, timeout=float(args.wait_timeout))
        out = _order_dict(trade)
        out["account"] = o.account
        _emit(out, args.format)
        return 0 if ok else 1
    finally:
        # A gate refusal must not leak the API connection — a stuck clientId
        # would block every subsequent tool invocation.
        if ib.isConnected():
            ib.disconnect()


def cmd_combo(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    """Place a multi-leg BAG order. Legs: --legs 'BUY:conId:ratio,SELL:conId:ratio,...'"""
    if not args.yes:
        raise SystemExit("Refusing to place live combo order without --yes confirmation flag.")
    parsed_legs = _parse_legs(args.legs)
    ib = _connect(args)
    try:
        legs = [
            ComboLeg(conId=con_id, ratio=ratio, action=action,
                     exchange=args.exchange or "SMART")
            for action, con_id, ratio in parsed_legs
        ]
        bag = Bag(symbol=args.symbol, currency=args.currency or "USD", exchange=args.exchange or "SMART")
        bag.comboLegs = legs
        o = Order()
        o.action = args.side
        o.totalQuantity = float(args.qty)
        o.orderType = args.type
        if args.limit_price is not None:
            o.lmtPrice = float(args.limit_price)
        o.tif = args.tif
        o.account = _resolve_account(ib, args.account)
        if args.order_ref:
            o.orderRef = args.order_ref
        else:
            import uuid as _uuid
            o.orderRef = f"TA{_uuid.uuid4().hex[:18]}"
        o.transmit = not args.no_transmit
        # Constitution clearance under the canonical leg-set key (fail-closed
        # when TRADING_CONSTITUTION_REQUIRED=true). Run `constitution-check
        # --legs ...` with the SAME legs/side/qty/price/account first.
        _require_combo_clearance(
            legs=parsed_legs, side=args.side, qty=float(args.qty),
            symbol=args.symbol, account=o.account,
            limit_price=args.limit_price, order_type=args.type, order_ref=o.orderRef,
        )
        trade = ib.placeOrder(bag, o)
        ok = _wait_for_order_ack(ib, trade, timeout=float(args.wait_timeout))
        out = _order_dict(trade)
        out["legs"] = [(leg.action, leg.conId, leg.ratio) for leg in legs]
        out["account"] = o.account
        _emit(out, args.format)
        return 0 if ok else 1
    finally:
        # A gate refusal must not leak the API connection.
        if ib.isConnected():
            ib.disconnect()


def cmd_cancel(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    ib = _connect(args)
    trades = ib.reqOpenOrders()
    ib.sleep(0.5)
    target = None
    for t in trades:
        if t.order.orderId == args.order_id:
            target = t
            break
    if not target:
        _emit({"error": f"order {args.order_id} not found"}, args.format)
        ib.disconnect()
        return 1
    ib.cancelOrder(target.order)
    ib.sleep(1.0)
    _emit(_order_dict(target), args.format)
    ib.disconnect()
    return 0


def cmd_cancel_all(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    if not args.yes:
        raise SystemExit("Refusing to global-cancel without --yes")
    # Panic-level gate: global cancel kills every working order across every
    # client on the account. Intentionally a DIFFERENT token from --yes so
    # an agent replaying a prior `--yes` batch cannot accidentally fire
    # reqGlobalCancel.
    if not getattr(args, "confirm_panic", False):
        raise SystemExit(
            "Refusing global-cancel without --confirm-panic. This flag is "
            "intentionally distinct from --yes to prevent accidental global "
            "cancellation via retried commands."
        )
    ib = _connect(args)
    ib.reqGlobalCancel()
    ib.sleep(1.5)
    _emit({"global_cancel": "sent"}, args.format)
    ib.disconnect()
    return 0


# ============================================================
# WSH / FX / misc
# ============================================================

def cmd_wsh_meta(args: argparse.Namespace) -> int:
    ib = _connect(args)
    try:
        meta = ib.reqWshMetaData()
        ib.sleep(2.0)
    except Exception as exc:
        meta = f"error: {exc}"
    _emit({"meta": str(meta)[:10000]}, args.format)
    ib.disconnect()
    return 0


def cmd_fx(args: argparse.Namespace) -> int:
    ib = _connect(args)
    pair = args.pair.upper().replace("/", "").replace(".", "")
    c = Forex(pair)
    ib.qualifyContracts(c)
    t = ib.reqMktData(c, "", False, False)
    ib.sleep(args.wait)
    out = _ticker_to_dict(t)
    out["pair"] = pair
    if args.amount and out.get("bid") and out.get("ask"):
        mid = (out["bid"] + out["ask"]) / 2
        out["converted"] = args.amount * mid
    ib.cancelMktData(c)
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Argparse wiring
# ============================================================

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--client-id", type=int, default=None)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--market-data-type", type=int, default=None, help="1=Live 2=Frozen 3=Delayed 4=DelayedFrozen")
    p.add_argument("--format", choices=["json", "csv", "table"], default="table")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trading_algo.ibkr_tool",
        description="Comprehensive IBKR data + operations CLI (ib_async based).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], int], help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        _add_common(sp)
        sp.set_defaults(func=fn)
        return sp

    # --- Meta ---
    add("connect", cmd_connect, "Test connection and show server info")
    add("time", cmd_time, "Get IBKR server time")
    add("accounts", cmd_accounts, "List managed account codes")
    add("user-info", cmd_user_info, "Get user info (white-brand)")

    # --- Account ---
    s = add("summary", cmd_summary, "Account summary tags")
    s.add_argument("--account", default=None)
    s.add_argument("--tags", default=None, help="Comma list or leave blank for defaults")

    s = add("values", cmd_values, "Full synchronized account values")
    s.add_argument("--account", default=None)
    s.add_argument("--tag", default=None, help="Substring filter on tag name")

    s = add("positions", cmd_positions, "List all positions")
    s.add_argument("--account", default=None)
    s.add_argument("--symbol", default=None)

    s = add("portfolio", cmd_portfolio, "Synchronized portfolio items with MTM")
    s.add_argument("--account", default=None)

    s = add("pnl", cmd_pnl, "Account-level daily/realized/unrealized PnL")
    s.add_argument("--account", default=None)
    s.add_argument("--wait", type=float, default=5.0, help="Maximum seconds to wait for the initial PnL update")

    s = add("pnl-single", cmd_pnl_single, "Per-position PnL")
    s.add_argument("--account", required=True)
    s.add_argument("--con-id", type=int, required=True)
    s.add_argument("--wait", type=float, default=5.0, help="Maximum seconds to wait for the initial PnL update")

    # --- Quotes / live data ---
    s = add("quote", cmd_quote, "Snap quote for one contract (includes greeks for OPT)")
    _add_contract_args(s)
    s.add_argument("--wait", type=float, default=3.0)

    s = add("quotes", cmd_quotes, "Batch snap for multiple STK symbols via reqTickers")
    s.add_argument("--symbols", required=True, help="Comma list")
    s.add_argument("--currency", default=None)

    s = add("stream", cmd_stream, "Stream ticks for a contract")
    _add_contract_args(s)
    s.add_argument("--duration", type=float, default=30.0)
    s.add_argument("--interval", type=float, default=1.0)

    s = add("depth", cmd_depth, "Market depth (DOM) ladder")
    _add_contract_args(s)
    s.add_argument("--rows", type=int, default=10)
    s.add_argument("--smart", action="store_true")
    s.add_argument("--wait", type=float, default=2.0)

    s = add("depth-exchanges", cmd_depth_exchanges, "List market depth exchanges")

    s = add("realtime-bars", cmd_realtime_bars, "5-sec realtime bars stream")
    _add_contract_args(s)
    s.add_argument("--what-to-show", default="TRADES")
    s.add_argument("--rth", action="store_true")
    s.add_argument("--duration", type=float, default=30.0)

    s = add("ticks", cmd_ticks, "Tick-by-tick data stream")
    _add_contract_args(s)
    s.add_argument("--tick-type", choices=["Last", "AllLast", "BidAsk", "MidPoint"], default="Last")
    s.add_argument("--duration", type=float, default=30.0)

    # --- Historical ---
    s = add("history", cmd_history, "Historical bars (reqHistoricalData)")
    _add_contract_args(s)
    s.add_argument("--duration", default="1 D")
    s.add_argument("--bar-size", default="5 mins")
    s.add_argument("--what-to-show", default="TRADES")
    s.add_argument("--rth", action="store_true")
    s.add_argument("--end", default=None)

    s = add("history-ticks", cmd_history_ticks, "Historical ticks")
    _add_contract_args(s)
    s.add_argument("--start", default=None)
    s.add_argument("--end", default=None)
    s.add_argument("--count", type=int, default=1000)
    s.add_argument("--what-to-show", choices=["TRADES", "BID_ASK", "MIDPOINT"], default="TRADES")
    s.add_argument("--rth", action="store_true")

    s = add("head-timestamp", cmd_head_timestamp, "Earliest available data timestamp")
    _add_contract_args(s)
    s.add_argument("--what-to-show", default="TRADES")
    s.add_argument("--rth", action="store_true")

    s = add("histogram", cmd_histogram, "Price histogram (reqHistogramData)")
    _add_contract_args(s)
    s.add_argument("--rth", action="store_true")
    s.add_argument("--period", default="20 days")

    s = add("schedule", cmd_schedule, "Historical trading schedule/sessions")
    _add_contract_args(s)
    s.add_argument("--duration", default="1 M")
    s.add_argument("--end", default=None)
    s.add_argument("--rth", action="store_true")

    # --- Options ---
    s = add("chain", cmd_chain, "Option chain metadata (reqSecDefOptParams)")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)

    s = add("chain-quote", cmd_chain_quote, "Snap all strikes at one expiry (both rights)")
    s.add_argument("--symbol", required=True)
    s.add_argument("--expiry", required=True)
    s.add_argument("--rights", choices=["C", "P", "both"], default="both")
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--min-strike", type=float, default=None)
    s.add_argument("--max-strike", type=float, default=None)
    s.add_argument("--wait", type=float, default=4.0)

    s = add("calc-iv", cmd_calc_iv, "Calculate implied volatility for an option")
    _add_contract_args(s, default_kind="OPT")
    s.add_argument("--option-price", type=float, required=True)
    s.add_argument("--under-price", type=float, required=True)

    s = add("calc-price", cmd_calc_price, "Calculate theoretical option price")
    _add_contract_args(s, default_kind="OPT")
    s.add_argument("--vol", type=float, required=True)
    s.add_argument("--under-price", type=float, required=True)

    # --- Discovery ---
    s = add("search", cmd_search, "Search matching symbols (reqMatchingSymbols)")
    s.add_argument("--query", required=True)

    s = add("contract", cmd_contract, "Full contract details")
    _add_contract_args(s)

    s = add("smart-components", cmd_smart_components, "SMART routing components")
    s.add_argument("--bbo-exchange", required=True)

    s = add("market-rule", cmd_market_rule, "Price increment rules")
    s.add_argument("--rule-id", type=int, required=True)

    # --- Fundamentals ---
    s = add("fundamentals", cmd_fundamentals, "Fundamental data reports")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--report", choices=[
        "ReportsFinSummary", "ReportSnapshot", "ReportRatios",
        "ReportsFinStatements", "ReportsOwnership", "RESC", "CalendarReport",
    ], default="ReportSnapshot")
    s.add_argument("--out", default=None)

    # --- News ---
    add("news-providers", cmd_news_providers, "List news providers")

    s = add("news", cmd_news, "Historical news headlines")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--providers", default=None, help="Default: BRFG+BRFUPDN+DJNL+DJ-RT")
    s.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM:SS.0")
    s.add_argument("--end", default=None)
    s.add_argument("--count", type=int, default=20)

    s = add("article", cmd_article, "Fetch news article body")
    s.add_argument("--provider", required=True)
    s.add_argument("--article-id", required=True)

    s = add("news-bulletins", cmd_news_bulletins, "Stream TWS news bulletins")
    s.add_argument("--duration", type=float, default=30.0)
    s.add_argument("--all-messages", action="store_true")

    # --- Scanner ---
    s = add("scanner-params", cmd_scanner_params, "Dump scanner parameters XML")
    s.add_argument("--out", default=None)

    s = add("scan", cmd_scan, "Run a scanner subscription")
    s.add_argument("--scan-code", default="TOP_PERC_GAIN")
    s.add_argument("--instrument", default="STK")
    s.add_argument("--location", default="STK.US.MAJOR")
    s.add_argument("--count", type=int, default=25)
    s.add_argument("--above-price", type=float, default=None)
    s.add_argument("--below-price", type=float, default=None)
    s.add_argument("--above-volume", type=int, default=None)
    s.add_argument("--market-cap-above", type=float, default=None)
    s.add_argument("--market-cap-below", type=float, default=None)
    s.add_argument("--filters", default=None, help="Comma list k=v")

    # --- Orders ---
    s = add("open-orders", cmd_open_orders, "List open orders")
    s.add_argument("--all", action="store_true", help="All clients (reqAllOpenOrders)")

    s = add("completed-orders", cmd_completed_orders, "List completed orders")
    s.add_argument("--api-only", action="store_true")

    s = add("executions", cmd_executions, "List executions (fills) with commission reports")
    s.add_argument("--account", default=None)
    s.add_argument("--client-id-filter", type=int, default=None)
    s.add_argument("--symbol", default=None)
    s.add_argument("--sec-type", default=None)
    s.add_argument("--exchange-filter", default=None)
    s.add_argument("--side", default=None)
    s.add_argument("--time", default=None, help="yyyymmdd-hh:mm:ss UTC")

    def _add_order_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--side", choices=["BUY", "SELL"], required=True)
        sp.add_argument("--qty", required=True)
        sp.add_argument("--type", choices=["MKT", "LMT", "STP", "STPLMT", "TRAIL", "MOC", "LOC"], default="LMT")
        sp.add_argument("--limit-price", type=float, default=None)
        sp.add_argument("--stop-price", type=float, default=None)
        sp.add_argument("--tif", default="DAY")
        sp.add_argument("--account", default=None)
        sp.add_argument("--order-ref", default=None)
        sp.add_argument("--oca-group", default=None)
        sp.add_argument("--outside-rth", action="store_true")
        sp.add_argument("--no-transmit", action="store_true")

    s = add("whatif", cmd_whatif, "Preview order (whatIf) — margin impact, commission")
    _add_contract_args(s)
    _add_order_args(s)

    s = add("place", cmd_place, "Place a single-leg order (requires --yes)")
    _add_contract_args(s)
    _add_order_args(s)
    s.add_argument("--yes", action="store_true", help="Required live confirmation flag")
    s.add_argument("--wait-timeout", type=float, default=15.0, help="Seconds to wait for order ack (default 15)")

    s = add("constitution-check", cmd_constitution_check,
            "Evaluate the constitution gate against live account state (READ-ONLY; writes the clearance verdict)")
    _add_contract_args(s)
    _add_order_args(s)
    s.add_argument("--legs", default=None,
                   help="combo mode: 'BUY:conId:ratio,SELL:conId:ratio,...' — per-leg evaluation, one verdict under the leg-set key (contract args ignored; --symbol is the BAG symbol)")
    s.add_argument("--structure", default=None,
                   help="short-put | leap-long | pmcc-long | covered-call | roll | close (inferred from primitives when omitted)")
    s.add_argument("--credit", type=float, default=None, help="expected fill credit/share (C8 denominator; defaults to --limit-price)")
    s.add_argument("--written-exit", dest="written_exit", default=None, help="W5: the written exit plan (required for new positions)")
    s.add_argument("--is-roll", dest="is_roll", action="store_true")
    s.add_argument("--new-program", dest="new_program", action="store_true", help="C9: first trade of a new option program")
    s.add_argument("--db-path", dest="db_path", default=None, help="override TRADING_DB_PATH for the verdict store")
    s.add_argument("--greeks-wait", dest="greeks_wait", type=float, default=8.0, help="seconds to wait for modelGreeks per contract")

    s = add("combo", cmd_combo, "Place multi-leg BAG combo order")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--legs", required=True, help="ACTION:conId:ratio,ACTION:conId:ratio,...")
    _add_order_args(s)
    s.add_argument("--yes", action="store_true")
    s.add_argument("--wait-timeout", type=float, default=15.0, help="Seconds to wait for order ack (default 15)")

    s = add("cancel", cmd_cancel, "Cancel one order by orderId")
    s.add_argument("--order-id", type=int, required=True)

    s = add("cancel-all", cmd_cancel_all, "Global cancel (all orders, all clients)")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--confirm-panic", action="store_true", dest="confirm_panic",
                   help="Required alongside --yes — distinct token so a replayed --yes "
                        "batch cannot accidentally fire a global cancel.")

    # --- WSH / FX ---
    add("wsh-meta", cmd_wsh_meta, "Wall Street Horizon metadata")

    s = add("fx", cmd_fx, "Forex quote + optional conversion")
    s.add_argument("--pair", required=True, help="e.g. USDCAD, EURUSD")
    s.add_argument("--amount", type=float, default=None)
    s.add_argument("--wait", type=float, default=2.0)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Route through the shared runner: audit every invocation, emit
    # structured JSON on exception, classify exit codes via the IBKR-aware
    # classifier. Agents consuming this CLI never have to regex stderr.
    from trading_algo.cli_runner import run_command
    return run_command(args, default_cmd_name="ibkr-tool")


if __name__ == "__main__":
    sys.exit(main())

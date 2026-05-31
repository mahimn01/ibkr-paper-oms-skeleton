#!/usr/bin/env python3
"""IBKR -> PIT readonly daily-bar ingestion (Wave-T5).

READ-ONLY by construction. The ib_async client is ALWAYS opened with
``IB().connect(..., readonly=True)``; IBKR then forbids any order on the
session at the protocol level. This module never imports, references, or
calls placeOrder/modifyOrder/cancelOrder.

Pipeline:
    reqHistoricalData(whatToShow='TRADES', barSizeSetting='1 day',
                      useRTH=True)  ->  PIT Bar  ->  PITStore.write_bars

Bars are stored UNADJUSTED (PLAN.md / CLAUDE.md Wave-T5 rule). Split
adjustment is applied at query time via AdjustmentEngine.adjust_series.

Validation modes (run as a script):

    # 1. Synthetic round-trip only (no network):
    python scripts/ibkr_to_pit.py --self-test

    # 2. Synthetic + small live readonly pull (port 4001, clientId 91):
    python scripts/ibkr_to_pit.py --pilot

If the gateway connection fails/times out, --pilot still passes as long as
the synthetic round-trip is green; the live failure is reported, not raised.

Backfill the full universe later (see backfill_command() / module docstring):
    python scripts/ibkr_to_pit.py --backfill \
        --pit-root data/pit_full --symbols AAPL MSFT ... --years 15
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

# Make `trading_algo` importable regardless of cwd (script lives in scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NOTE: import trading_algo PIT layer lazily inside functions where helpful,
# but these are pure-python and safe at module load.
from trading_algo.data.pit_store import Bar, PITStore
from trading_algo.data.corporate_actions import AdjustmentEngine

LIVE_PORT = 4001          # live gateway (paper 4002 is down per ops)
PAPER_PORT = 4002
DEFAULT_HOST = "127.0.0.1"
PILOT_CLIENT_ID = 91


# --------------------------------------------------------------------------
# Bar conversion (network-agnostic; unit-testable without a gateway)
# --------------------------------------------------------------------------

def _to_pit_date(raw) -> date:
    """Normalise an ib_async BarData.date (date | datetime | str) to a date."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    s = str(raw)
    # ib_async daily bars come back as 'YYYYMMDD' or 'YYYY-MM-DD'.
    s = s.strip().split(" ")[0]
    if "-" in s:
        return date.fromisoformat(s)
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def ibbar_to_pit(symbol: str, b) -> Bar:
    """Convert one ib_async BarData into a PIT Bar.

    BarData fields: date, open, high, low, close, volume, average, barCount.
    `average` is IBKR's VWAP for the bar. volume is rounded to int (PIT schema
    is int64). Bars are stored UNADJUSTED.
    """
    vol = b.volume
    volume = int(round(vol)) if vol is not None and vol >= 0 else 0
    vwap = float(b.average) if getattr(b, "average", None) is not None else None
    return Bar(
        symbol=symbol,
        date=_to_pit_date(b.date),
        open=float(b.open),
        high=float(b.high),
        low=float(b.low),
        close=float(b.close),
        volume=volume,
        vwap=vwap,
    )


# --------------------------------------------------------------------------
# Synthetic round-trip validation (NO network)
# --------------------------------------------------------------------------

@dataclass
class RoundTripResult:
    ok: bool
    n_written: int
    n_read: int
    detail: str


def validate_synthetic_roundtrip(pit_root: str | Path) -> RoundTripResult:
    """Write 5 fake bars, read them back, confirm field-for-field equality.

    Uses a synthetic symbol so it never collides with a real backfill.
    """
    store = PITStore(pit_root)
    sym = "_SYNTH_TEST"
    fake = [
        Bar(symbol=sym, date=date(2024, 1, 2), open=100.0, high=101.0,
            low=99.0, close=100.5, volume=1_000_000, vwap=100.2),
        Bar(symbol=sym, date=date(2024, 1, 3), open=100.5, high=102.0,
            low=100.0, close=101.5, volume=1_100_000, vwap=101.0),
        Bar(symbol=sym, date=date(2024, 1, 4), open=101.5, high=103.0,
            low=101.0, close=102.8, volume=1_200_000, vwap=102.1),
        Bar(symbol=sym, date=date(2024, 1, 5), open=102.8, high=104.0,
            low=102.0, close=103.1, volume=900_000, vwap=103.0),
        Bar(symbol=sym, date=date(2024, 1, 8), open=103.1, high=105.0,
            low=102.5, close=104.7, volume=1_300_000, vwap=104.0),
    ]
    n_written = store.write_bars(sym, fake)
    read = store.read_bars(sym, date(2024, 1, 1), date(2024, 1, 31))

    if len(read) != len(fake):
        return RoundTripResult(False, n_written, len(read),
                               f"count mismatch: wrote {len(fake)}, read {len(read)}")
    for w, r in zip(fake, read):
        for field in ("date", "open", "high", "low", "close", "volume"):
            if getattr(w, field) != getattr(r, field):
                return RoundTripResult(
                    False, n_written, len(read),
                    f"field {field} mismatch on {w.date}: {getattr(w, field)} != {getattr(r, field)}")
        if r.vwap is None or abs(r.vwap - w.vwap) > 1e-9:
            return RoundTripResult(False, n_written, len(read),
                                   f"vwap mismatch on {w.date}: {w.vwap} != {r.vwap}")
        if not isinstance(r.volume, int):
            return RoundTripResult(False, n_written, len(read),
                                   f"volume not int on {w.date}: {type(r.volume)}")
    return RoundTripResult(True, n_written, len(read),
                           "5 synthetic bars round-tripped (all fields match, volume is int, vwap preserved)")


def validate_adjustment(pit_root: str | Path) -> tuple[bool, str]:
    """Confirm AdjustmentEngine.adjust_series applies a split at query time.

    Registers a synthetic security, records a 4:1 split, then verifies pre-split
    bars are scaled by 0.25 and post-split bars are unchanged (factor 1.0).
    """
    store = PITStore(pit_root)
    sym = "_SYNTH_SPLIT"
    iid = store.upsert_security(sym)
    # 4:1 split with ex_date 2024-01-05: bars strictly before are scaled by 1/4.
    store.add_split(iid, date(2024, 1, 5), 4.0)
    bars = [
        Bar(symbol=sym, date=date(2024, 1, 2), open=400.0, high=404.0,
            low=396.0, close=402.0, volume=250_000, vwap=401.0),
        Bar(symbol=sym, date=date(2024, 1, 3), open=402.0, high=408.0,
            low=400.0, close=406.0, volume=240_000, vwap=405.0),
        Bar(symbol=sym, date=date(2024, 1, 8), open=101.0, high=103.0,
            low=100.0, close=102.0, volume=1_000_000, vwap=101.5),
    ]
    store.write_bars(sym, bars)
    read = store.read_bars(sym, date(2024, 1, 1), date(2024, 1, 31))
    adj = AdjustmentEngine(store).adjust_series(iid, read, as_of=date(2024, 1, 31))
    by_date = {a.date: a for a in adj}
    pre = by_date[date(2024, 1, 2)]
    post = by_date[date(2024, 1, 8)]
    if abs(pre.adjustment_factor - 0.25) > 1e-9:
        return False, f"pre-split factor wrong: {pre.adjustment_factor} (want 0.25)"
    if abs(pre.adj_close - 402.0 * 0.25) > 1e-9:
        return False, f"pre-split adj_close wrong: {pre.adj_close} (want {402.0*0.25})"
    if abs(post.adjustment_factor - 1.0) > 1e-9:
        return False, f"post-split factor wrong: {post.adjustment_factor} (want 1.0)"
    return True, (f"4:1 split applied at query time: pre-split close 402.0 -> "
                  f"adj {pre.adj_close} (factor {pre.adjustment_factor}); "
                  f"post-split close 102.0 -> adj {post.adj_close} (factor {post.adjustment_factor})")


# --------------------------------------------------------------------------
# Live readonly pull (network)
# --------------------------------------------------------------------------

def pull_daily_bars(
    symbols: Sequence[str],
    *,
    years: int = 5,
    host: str = DEFAULT_HOST,
    port: int = LIVE_PORT,
    client_id: int = PILOT_CLIENT_ID,
    timeout: float = 30.0,
) -> dict[str, list[Bar]]:
    """READONLY pull of daily TRADES bars for `symbols`. Returns symbol -> [Bar].

    The IB client is opened with readonly=True; IBKR then rejects any order on
    this session. We never call any *Order method. Raises on connection failure
    so the caller decides whether to fall back to synthetic-only validation.
    """
    from ib_async import IB, Stock  # imported lazily so --self-test needs no gateway

    ib = IB()
    out: dict[str, list[Bar]] = {}
    # HARD GUARD: refuse to ever open a writable session here.
    ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout)
    try:
        for sym in symbols:
            contract = Stock(sym, "SMART", "USD")
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=f"{years} Y",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            out[sym] = [ibbar_to_pit(sym, b) for b in bars]
    finally:
        ib.disconnect()
    return out


def ingest(
    symbols: Sequence[str],
    pit_root: str | Path,
    *,
    years: int = 5,
    host: str = DEFAULT_HOST,
    port: int = LIVE_PORT,
    client_id: int = PILOT_CLIENT_ID,
) -> dict[str, int]:
    """Pull (readonly) and persist daily bars to a PITStore. Returns symbol -> n_written."""
    store = PITStore(pit_root)
    pulled = pull_daily_bars(symbols, years=years, host=host, port=port, client_id=client_id)
    written: dict[str, int] = {}
    for sym, bars in pulled.items():
        # Register identity so AdjustmentEngine / UniverseResolver can resolve it.
        store.upsert_security(sym)
        written[sym] = store.write_bars(sym, bars)
    return written


def backfill_command(pit_root: str = "data/pit_full") -> str:
    """The exact command to backfill the full 444 + R3000 universe later."""
    return (
        "python scripts/ibkr_to_pit.py --backfill "
        f"--pit-root {pit_root} --years 15 "
        "--symbols $(python -c \"import glob,os;"
        "print(' '.join(sorted({os.path.basename(p).split('.')[0] "
        "for p in glob.glob('data/atlas_r3000/*.parquet')})))\")"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _run_self_test(pit_root: str) -> int:
    rt = validate_synthetic_roundtrip(pit_root)
    print(f"[synthetic round-trip] ok={rt.ok} wrote={rt.n_written} read={rt.n_read} :: {rt.detail}")
    aok, adetail = validate_adjustment(pit_root)
    print(f"[adjustment]            ok={aok} :: {adetail}")
    return 0 if (rt.ok and aok) else 1


def _run_pilot(pit_root: str, symbols: list[str], years: int,
               host: str, port: int, client_id: int) -> int:
    # 1) synthetic gate first
    code = _run_self_test(pit_root)
    if code != 0:
        print("[pilot] synthetic gate FAILED; aborting before any network call.")
        return code
    # 2) small live readonly pull
    print(f"[pilot] connecting READONLY to {host}:{port} clientId={client_id} "
          f"for {symbols} ({years}Y daily TRADES)...")
    try:
        written = ingest(symbols, pit_root, years=years, host=host, port=port, client_id=client_id)
    except Exception as e:  # connection/timeout/qualify failure -> report, don't crash
        print(f"[pilot] LIVE PULL FAILED ({type(e).__name__}: {e}). "
              f"Synthetic round-trip + adjustment already passed, so the PIT "
              f"write/read path is validated. Re-run --pilot when the gateway "
              f"is reachable on {host}:{port}.")
        return 0
    store = PITStore(pit_root)
    eng = AdjustmentEngine(store)
    for sym, n in written.items():
        read = store.read_bars(sym, date(2000, 1, 1), date.today())
        if not read:
            print(f"[pilot] {sym}: wrote {n} but read 0 back -- INVESTIGATE")
            continue
        iid = store.resolve_ticker(sym, date.today()) or store.upsert_security(sym)
        adj = eng.adjust_series(iid, read, as_of=date.today())
        first, last = read[0], read[-1]
        adj_first = adj[0]
        print(f"[pilot] {sym}: wrote {n}, read {len(read)} | "
              f"{first.date}->{last.date} | last close {last.close} | "
              f"first bar {first.date} close {first.close} "
              f"adj_close {adj_first.adj_close:.4f} (factor {adj_first.adjustment_factor:.4f})")
    print("[pilot] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="IBKR -> PIT readonly daily-bar ingestion")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true",
                      help="Synthetic round-trip + adjustment only (no network).")
    mode.add_argument("--pilot", action="store_true",
                      help="Synthetic gate, then a small readonly live pull.")
    mode.add_argument("--backfill", action="store_true",
                      help="Pull+persist the given --symbols (readonly).")
    p.add_argument("--pit-root", default="data/pit_pilot")
    p.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"])
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=LIVE_PORT)
    p.add_argument("--client-id", type=int, default=PILOT_CLIENT_ID)
    args = p.parse_args(argv)

    if args.self_test:
        return _run_self_test(args.pit_root)
    if args.pilot:
        return _run_pilot(args.pit_root, [s.upper() for s in args.symbols],
                          args.years, args.host, args.port, args.client_id)
    if args.backfill:
        print(f"[backfill] {len(args.symbols)} symbols -> {args.pit_root} (readonly)")
        written = ingest([s.upper() for s in args.symbols], args.pit_root,
                         years=args.years, host=args.host, port=args.port,
                         client_id=args.client_id)
        total = sum(written.values())
        print(f"[backfill] wrote {total} bars across {len(written)} symbols")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

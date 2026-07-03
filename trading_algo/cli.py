from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Literal

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.broker.sim import SimBroker
from trading_algo.config import TradingConfig
from trading_algo.engine import Engine, default_risk_manager
from trading_algo.broker.base import OrderRequest
from trading_algo.instruments import InstrumentSpec, validate_instrument
from trading_algo.logging_setup import configure_logging
from trading_algo.orders import TradeIntent
from trading_algo.persistence import SqliteStore
from trading_algo.journal_cli import add_journal_subparser
from trading_algo.strategy.example import ExampleStrategy
from trading_algo.oms import OrderManager
from trading_algo.backtest.data import load_bars_csv
from trading_algo.backtest.runner import BacktestConfig, run_backtest
from trading_algo.backtest.export import ExportConfig, export_historical_bars
from trading_algo.backtest.validate import validate_bars


def _load_dotenv_if_present() -> None:
    # Minimal .env loader to avoid extra dependencies.
    if not os.path.exists(".env"):
        return
    with open(".env", "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Set only if missing/empty so shell overrides still work, but blanks get filled.
            if os.getenv(k) is None or os.getenv(k) == "":
                os.environ[k] = v


def _live_confirm_prompt(message: str) -> bool:
    """Interactive confirmation prompt for live account operations."""
    try:
        response = input(message).strip()
        return response == "YES"
    except (EOFError, KeyboardInterrupt):
        return False


def _make_broker(kind: Literal["ibkr", "sim"], cfg: TradingConfig):
    if kind == "sim":
        return SimBroker()
    if kind == "ibkr":
        return IBKRBroker(
            cfg.ibkr,
            require_paper=cfg.require_paper,
            allow_live=cfg.allow_live,
            live_confirm_callback=_live_confirm_prompt if cfg.allow_live else None,
            db_path=cfg.db_path,
            constitution_required=cfg.constitution_required,
            constitution_max_age_s=cfg.constitution_max_age_s,
        )
    raise ValueError(f"Unsupported broker: {kind}")


def _apply_cli_overrides(cfg: TradingConfig, args: argparse.Namespace) -> TradingConfig:
    from .config import IBKR_PORT_GW_PAPER, IBKR_PORT_TWS_LIVE

    ibkr = cfg.ibkr

    # Resolve port: explicit --ibkr-port > --paper/--live shortcuts > env/default
    resolved_port = None
    if args.ibkr_port is not None:
        resolved_port = int(args.ibkr_port)
    elif getattr(args, "paper", False):
        resolved_port = IBKR_PORT_GW_PAPER
    elif getattr(args, "live", False):
        resolved_port = IBKR_PORT_TWS_LIVE

    if resolved_port is not None or args.ibkr_host is not None or args.ibkr_client_id is not None:
        ibkr = type(cfg.ibkr)(
            host=args.ibkr_host or cfg.ibkr.host,
            port=resolved_port or cfg.ibkr.port,
            client_id=int(args.ibkr_client_id or cfg.ibkr.client_id),
        )

    dry_run = cfg.dry_run
    if getattr(args, "dry_run", False):
        dry_run = True
    if getattr(args, "no_dry_run", False):
        dry_run = False

    allow_live = cfg.allow_live or getattr(args, "allow_live", False) or getattr(args, "live", False)

    return TradingConfig(
        broker=cfg.broker,
        live_enabled=cfg.live_enabled,
        require_paper=not allow_live,
        allow_live=allow_live,
        dry_run=dry_run,
        order_token=cfg.order_token,
        confirm_token_required=cfg.confirm_token_required,
        db_path=cfg.db_path,
        poll_seconds=cfg.poll_seconds,
        ibkr=ibkr,
    )


def _assert_ibkr_order_authorized(cfg: TradingConfig, confirm_token: str | None) -> None:
    """
    Second safety gate for any IBKR order submission.
    """
    if cfg.dry_run:
        return
    if not cfg.live_enabled:
        raise SystemExit("Refusing to place IBKR orders with TRADING_LIVE_ENABLED=false (set it true explicitly).")
    if cfg.confirm_token_required:
        if not cfg.order_token:
            raise SystemExit("Refusing to place IBKR orders without TRADING_ORDER_TOKEN set (second confirmation gate).")
        if confirm_token != cfg.order_token:
            raise SystemExit("Refusing to place IBKR orders: --confirm-token does not match TRADING_ORDER_TOKEN.")


def _cmd_place_order(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker == "ibkr":
        _assert_ibkr_order_authorized(cfg, args.confirm_token)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        intent = TradeIntent(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type=args.type,
            limit_price=float(args.limit_price) if args.limit_price is not None else None,
            stop_price=float(args.stop_price) if args.stop_price is not None else None,
            tif=args.tif,
        )
        if cfg.dry_run:
            print(f"DRY RUN: would place {intent}")
            if store and run_id is not None:
                store.log_decision(run_id, strategy="cli.place-order", intent=intent, accepted=False, reason="dry_run")
            return 0
        # This command is intentionally "direct": it does not use Engine risk gating.
        req = OrderRequest(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type=args.type,
            limit_price=float(args.limit_price) if args.limit_price is not None else None,
            stop_price=float(args.stop_price) if args.stop_price is not None else None,
            tif=args.tif,
            outside_rth=bool(args.outside_rth),
            good_till_date=args.good_till_date,
            account=args.account,
            order_ref=args.order_ref,
            oca_group=args.oca_group,
            transmit=not bool(args.no_transmit),
        )

        # Idempotency path. If --idempotency-key is supplied, check a
        # SQLite cache for a prior completed attempt with this key: if
        # present, replay the stored result (never re-transmit). If the
        # cache shows an in-flight attempt, fall through to the
        # IdempotentOrderPlacer which will orderbook-check before
        # retransmitting.
        idem_key = getattr(args, "idempotency_key", None)
        result = None
        idem_store = None
        if idem_key:
            from trading_algo.idempotency import IdempotencyStore, derive_order_ref
            idem_store = IdempotencyStore()
            existing = idem_store.lookup(idem_key)
            if existing is not None and existing.completed:
                # Short-circuit replay — no broker call.
                replayed = existing.result or {}
                if isinstance(replayed, dict):
                    print(
                        f"orderId={replayed.get('order_id', existing.ib_order_id or '?')} "
                        f"status={replayed.get('status', '?')} replayed=true"
                    )
                    return int(existing.exit_code or 0)
            # Derive a deterministic orderRef from the key so cross-process
            # retries produce the same orderRef and the orderbook check
            # can find the in-flight order.
            derived_ref = derive_order_ref(idem_key)
            from dataclasses import replace
            req = replace(req, order_ref=args.order_ref or derived_ref)
            idem_store.record_attempt(
                key=idem_key, cmd="place-order",
                request={
                    "symbol": args.symbol, "side": args.side,
                    "qty": args.qty, "type": args.type,
                    "limit_price": args.limit_price, "stop_price": args.stop_price,
                },
                order_ref=req.order_ref,
            )

        if idem_key and args.broker == "ibkr":
            # Route through the idempotent placer so a crashed retry
            # doesn't double-fill.
            from trading_algo.broker.idempotent_placer import IdempotentOrderPlacer
            placer = IdempotentOrderPlacer(broker)
            result = placer.place(req, idempotency_key=idem_key)
        else:
            result = broker.place_order(req)

        if idem_store and idem_key:
            try:
                idem_store.record_completion(
                    key=idem_key,
                    result={"order_id": result.order_id, "status": result.status},
                    exit_code=0,
                    ib_order_id=int(result.order_id) if str(result.order_id).isdigit() else None,
                )
            except Exception:
                pass  # Audit should never fail the command.
        if store and run_id is not None:
            store.log_order(run_id, broker=args.broker, order_id=result.order_id, request=req, status=result.status)
            try:
                st = broker.get_order_status(result.order_id)
                store.log_order_status_event(run_id, args.broker, st)
            except Exception as exc:
                store.log_error(run_id, where="cli.place-order.status", message=str(exc))
        print(f"orderId={result.order_id} status={result.status}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_snapshot(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        snap = broker.get_market_data_snapshot(instrument)
        print(
            f"{snap.instrument.kind} {snap.instrument.symbol} bid={snap.bid} ask={snap.ask} last={snap.last} "
            f"close={snap.close} volume={snap.volume} ts={snap.timestamp_epoch_s}"
        )
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_history(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        bars = broker.get_historical_bars(
            instrument,
            duration=args.duration,
            bar_size=args.bar_size,
            what_to_show=args.what_to_show,
            use_rth=bool(args.use_rth),
        )
        print(f"bars={len(bars)}")
        for b in bars[: min(len(bars), 5)]:
            print(f"ts={b.timestamp_epoch_s} o={b.open} h={b.high} l={b.low} c={b.close} v={b.volume}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_run(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    cfg = TradingConfig(
        broker=args.broker,
        live_enabled=cfg.live_enabled,
        require_paper=True,
        dry_run=cfg.dry_run,
        order_token=cfg.order_token,
        db_path=cfg.db_path,
        poll_seconds=args.poll_seconds or cfg.poll_seconds,
        ibkr=cfg.ibkr,
    )
    broker = _make_broker(args.broker, cfg)
    strategy = ExampleStrategy(symbol=args.symbol)
    engine = Engine(
        broker=broker,
        config=cfg,
        strategy=strategy,
        risk=default_risk_manager(),
        confirm_token=args.confirm_token,
    )

    if args.once:
        engine.run_once()
    else:
        engine.run_forever()
    return 0


def _cmd_order_status(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        st = broker.get_order_status(args.order_id)
        if store and run_id is not None:
            store.log_order_status_event(run_id, args.broker, st)
        print(f"orderId={st.order_id} status={st.status} filled={st.filled} remaining={st.remaining} avgFill={st.avg_fill_price}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_cancel_order(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        broker.cancel_order(args.order_id)
        if store and run_id is not None:
            try:
                st = broker.get_order_status(args.order_id)
                store.log_order_status_event(run_id, args.broker, st)
            except Exception as exc:
                store.log_error(run_id, where="cli.cancel-order.status", message=str(exc))
        print(f"cancelled orderId={args.order_id}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_modify_order(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker == "ibkr":
        _assert_ibkr_order_authorized(cfg, args.confirm_token)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        req = OrderRequest(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type=args.type,
            limit_price=float(args.limit_price) if args.limit_price is not None else None,
            stop_price=float(args.stop_price) if args.stop_price is not None else None,
            tif=args.tif,
            outside_rth=bool(args.outside_rth),
            good_till_date=args.good_till_date,
            account=args.account,
            order_ref=args.order_ref,
            oca_group=args.oca_group,
            transmit=not bool(args.no_transmit),
        )
        if cfg.dry_run:
            print(f"DRY RUN: would modify orderId={args.order_id} -> {req}")
            return 0

        res = broker.modify_order(args.order_id, req)
        if store and run_id is not None:
            store.log_order(run_id, broker=args.broker, order_id=res.order_id, request=req, status=res.status)
            try:
                st = broker.get_order_status(res.order_id)
                store.log_order_status_event(run_id, args.broker, st)
            except Exception as exc:
                store.log_error(run_id, where="cli.modify-order.status", message=str(exc))
        print(f"orderId={res.order_id} status={res.status}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_place_bracket(args: argparse.Namespace) -> int:
    # Halt sentinel check — refuses writes while data/HALTED exists.
    from trading_algo.halt import assert_not_halted
    assert_not_halted()
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker == "ibkr":
        _assert_ibkr_order_authorized(cfg, args.confirm_token)

    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        if cfg.dry_run:
            print(
                "DRY RUN: would place bracket "
                f"{instrument.kind} {instrument.symbol} side={args.side} qty={args.qty} "
                f"entry={args.entry_limit} tp={args.take_profit} sl={args.stop_loss}"
            )
            if store and run_id is not None:
                store.log_error(run_id, where="cli.place-bracket", message="dry_run")
            return 0

        from trading_algo.broker.base import BracketOrderRequest

        req = BracketOrderRequest(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            entry_limit_price=float(args.entry_limit),
            take_profit_limit_price=float(args.take_profit),
            stop_loss_stop_price=float(args.stop_loss),
            tif=args.tif,
        )
        res = broker.place_bracket_order(req)
        if store and run_id is not None:
            store.log_error(run_id, where="cli.place-bracket", message=f"placed parent={res.parent_order_id}")
        print(
            f"parent={res.parent_order_id} takeProfit={res.take_profit_order_id} stopLoss={res.stop_loss_order_id}"
        )
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_paper_smoke(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker != "ibkr":
        raise SystemExit("paper-smoke is only supported with --broker ibkr")

    broker = _make_broker("ibkr", cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        snap = broker.get_market_data_snapshot(instrument)
        print(
            f"OK: connected paper account, snapshot {snap.instrument.kind} {snap.instrument.symbol} "
            f"bid={snap.bid} ask={snap.ask} last={snap.last} ts={snap.timestamp_epoch_s}"
        )

        if not args.order_test:
            return 0

        _assert_ibkr_order_authorized(cfg, args.confirm_token)

        px = snap.last or snap.close
        if px is None or px <= 0:
            raise SystemExit("Cannot run order-test without a usable last/close price from snapshot")

        # Place a limit order far from market and cancel shortly after.
        if args.side == "BUY":
            limit_price = max(0.01, float(px) * 0.5)
        else:
            limit_price = float(px) * 1.5

        intent = TradeIntent(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type="LMT",
            limit_price=limit_price,
        )
        if cfg.dry_run:
            print(f"DRY RUN: would place+cancel smoke-test order {intent}")
            return 0

        res = broker.place_order(intent.to_order_request())
        print(f"Placed smoke-test order orderId={res.order_id} status={res.status}; cancelling...")
        broker.cancel_order(res.order_id)
        st = broker.get_order_status(res.order_id)
        print(f"After cancel: orderId={st.order_id} status={st.status}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_oms_reconcile(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if not cfg.db_path:
        raise SystemExit("oms-reconcile requires TRADING_DB_PATH to be set")
    broker = _make_broker(args.broker, cfg)
    broker.connect()
    try:
        oms = OrderManager(broker, cfg, confirm_token=args.confirm_token)
        try:
            res = oms.reconcile()
            print(f"reconciled={len(res)}")
            for oid, st in res.items():
                print(f"orderId={oid} status={st}")
        finally:
            oms.close()
        return 0
    finally:
        broker.disconnect()


def _cmd_oms_track(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if not cfg.db_path:
        raise SystemExit("oms-track requires TRADING_DB_PATH to be set")
    broker = _make_broker(args.broker, cfg)
    broker.connect()
    try:
        oms = OrderManager(broker, cfg, confirm_token=args.confirm_token)
        try:
            oms.reconcile()
            oms.track_open_orders(poll_seconds=float(args.poll_seconds), timeout_seconds=float(args.timeout_seconds) if args.timeout_seconds else None)
            print("ok")
        finally:
            oms.close()
        return 0
    finally:
        broker.disconnect()


def _cmd_backtest(args: argparse.Namespace) -> int:
    instrument = validate_instrument(
        InstrumentSpec(
            kind=args.kind,
            symbol=args.symbol,
            exchange=args.exchange,
            currency=args.currency,
            expiry=args.expiry,
            right=getattr(args, "right", None),
            strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
            multiplier=getattr(args, "multiplier", None),
        )
    )
    series = load_bars_csv(args.csv, instrument)
    cfg = BacktestConfig(
        initial_cash=float(args.initial_cash),
        commission_per_order=float(args.commission_per_order),
        slippage_bps=float(args.slippage_bps),
        spread=float(args.spread),
        db_path=args.db_path,
    )
    res = run_backtest(ExampleStrategy(symbol=instrument.symbol), instrument, series.bars, cfg)
    print(f"start={res.start_equity} end={res.end_equity} returnPct={res.return_pct}")
    return 0


def _cmd_export_history(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker != "ibkr":
        raise SystemExit("export-history currently supports only --broker ibkr")
    import os

    if os.path.exists(args.out_csv) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.out_csv} (use --overwrite)")
    broker = _make_broker("ibkr", cfg)
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        export_cfg = ExportConfig(
            duration_per_call=args.duration_per_call,
            bar_size=args.bar_size,
            what_to_show=args.what_to_show,
            use_rth=bool(args.use_rth),
            pacing_sleep_seconds=float(args.pacing_sleep_seconds),
            max_calls=int(args.max_calls),
        )
        bars = export_historical_bars(
            broker,
            instrument,
            out_csv_path=args.out_csv,
            cfg=export_cfg,
            end_datetime=args.end_datetime,
        )
        if args.validate:
            issues = validate_bars(bars)
            errors = [i for i in issues if i.level == "error"]
            for i in issues:
                print(f"{i.level}: {i.message}")
            if errors:
                raise SystemExit("bar validation failed")
        print(f"wrote={args.out_csv} bars={len(bars)}")
        return 0
    finally:
        broker.disconnect()


def _cmd_backtest_options(args: argparse.Namespace) -> int:
    """Backtest Wheel or PMCC strategy using IBKR historical bars."""
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)

    from trading_algo.quant_core.strategies.options.options_backtester import (
        run_options_backtest,
        print_report,
    )
    from trading_algo.quant_core.strategies.options.wheel import WheelStrategy, WheelConfig
    from trading_algo.quant_core.strategies.options.pmcc import PMCCStrategy, PMCCConfig

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    capital = float(args.capital)
    strategy_name = args.strategy.lower()

    broker = _make_broker("ibkr", cfg)
    broker.connect()

    try:
        for symbol in symbols:
            print(f"\n--- Pulling {args.duration} of daily bars for {symbol} ---")
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            bars = broker.get_historical_bars(
                instrument,
                duration=args.duration,
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
            )
            print(f"  Got {len(bars)} bars")
            if len(bars) < 100:
                print(f"  SKIP: need >= 100 bars, got {len(bars)}")
                continue

            if strategy_name == "wheel":
                strat = WheelStrategy(WheelConfig(
                    initial_capital=capital,
                    put_delta=float(args.put_delta),
                    call_delta=float(args.call_delta),
                    target_dte=int(args.dte),
                    profit_target=float(args.profit_target),
                ))
            elif strategy_name == "pmcc":
                strat = PMCCStrategy(PMCCConfig(
                    initial_capital=capital,
                    leaps_delta=float(args.leaps_delta),
                    short_delta=float(args.short_delta),
                    short_dte=int(args.dte),
                    short_profit_target=float(args.profit_target),
                ))
            else:
                print(f"  Unknown strategy: {strategy_name}")
                return 1

            report = run_options_backtest(
                strategy=strat,
                bars=bars,
                symbol=symbol,
                iv_premium_factor=float(args.iv_premium),
            )
            print(print_report(report))
    finally:
        broker.disconnect()

    return 0


def _cmd_llm_run(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)

    from trading_algo.llm.config import LLMConfig
    from trading_algo.llm.gemini import GeminiClient
    from trading_algo.llm.trader import LLMTrader
    from trading_algo.risk import RiskLimits, RiskManager

    llm_cfg = LLMConfig.from_env()
    if llm_cfg.provider != "gemini":
        raise SystemExit("LLM_PROVIDER must be 'gemini' for llm-run")
    if not llm_cfg.enabled:
        raise SystemExit("LLM_ENABLED must be true for llm-run")
    if not llm_cfg.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY must be set for llm-run")
    if not str(llm_cfg.gemini_model).startswith("gemini-3"):
        raise SystemExit(
            f"Refusing to run with GEMINI_MODEL={llm_cfg.gemini_model!r}; set GEMINI_MODEL to a Gemini 3 model id "
            "(e.g. gemini-3-pro-preview or gemini-3-flash-preview)."
        )
    if not llm_cfg.allowed_symbols():
        raise SystemExit("LLM_ALLOWED_SYMBOLS must be set (comma-separated)")

    if args.broker == "ibkr":
        # Orders go through OMS gates too, but keep explicit CLI auth for clarity.
        _assert_ibkr_order_authorized(cfg, args.confirm_token)

    broker = _make_broker(args.broker, cfg)
    trader = LLMTrader(
        broker=broker,
        trading=cfg,
        llm=llm_cfg,
        client=GeminiClient(api_key=llm_cfg.gemini_api_key, model=llm_cfg.normalized_gemini_model()),
        risk=RiskManager(RiskLimits()),
        confirm_token=args.confirm_token,
        sleep_seconds=float(args.sleep_seconds),
        max_ticks=(int(args.max_ticks) if args.max_ticks is not None else None),
    )
    if args.once:
        trader.run_once()
    else:
        trader.run()
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    broker.connect()
    try:
        results = broker.scan_market(
            args.scan_type,
            instrument_type=args.instrument_type,
            location=args.location,
            num_rows=int(args.max_results),
            above_price=float(args.min_price) if args.min_price is not None else None,
            below_price=float(args.max_price) if args.max_price is not None else None,
            above_volume=int(args.min_volume) if args.min_volume is not None else None,
            market_cap_above=float(args.min_market_cap) if args.min_market_cap is not None else None,
            market_cap_below=float(args.max_market_cap) if args.max_market_cap is not None else None,
        )
        print(f"results={len(results)} scan={args.scan_type}")
        for r in results:
            extra_str = ""
            if r.extra:
                name = r.extra.get("longName", "")
                industry = r.extra.get("industry", "")
                if name:
                    extra_str += f" name={name}"
                if industry:
                    extra_str += f" industry={industry}"
            print(f"  #{r.rank}: {r.instrument.symbol} ({r.instrument.kind}){extra_str}")
        return 0
    finally:
        broker.disconnect()


def _cmd_wheel_live(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)

    from trading_algo.quant_core.strategies.options.wheel import WheelConfig
    from trading_algo.quant_core.strategies.options.live_wheel_adapter import (
        LiveWheelAdapter,
        LiveWheelConfig,
    )

    if args.broker == "ibkr":
        _assert_ibkr_order_authorized(cfg, args.confirm_token)

    wheel_cfg = WheelConfig(
        initial_capital=float(args.capital),
        put_delta=float(args.put_delta),
        call_delta=float(args.call_delta),
        target_dte=int(args.dte),
        profit_target=float(args.profit_target),
        min_iv_rank=float(args.min_iv_rank),
    )
    live_cfg = LiveWheelConfig(
        symbol=args.symbol.upper(),
        wheel_config=wheel_cfg,
        use_lmt_orders=not args.use_mkt,
        lmt_offset_pct=float(args.lmt_offset_pct),
        price_history_bars=int(args.history_bars),
        iv_rv_window=int(args.iv_window),
    )

    broker = _make_broker(args.broker, cfg)
    strategy = LiveWheelAdapter(live_cfg)
    cfg = TradingConfig(
        broker=args.broker,
        live_enabled=cfg.live_enabled,
        require_paper=cfg.require_paper,
        allow_live=cfg.allow_live,
        dry_run=cfg.dry_run,
        order_token=cfg.order_token,
        confirm_token_required=cfg.confirm_token_required,
        db_path=cfg.db_path,
        poll_seconds=int(args.poll_seconds) if args.poll_seconds else cfg.poll_seconds,
        ibkr=cfg.ibkr,
    )
    engine = Engine(
        broker=broker,
        config=cfg,
        strategy=strategy,
        risk=default_risk_manager(),
        confirm_token=args.confirm_token,
    )

    logging.getLogger(__name__).info(
        "Starting wheel-live: symbol=%s broker=%s dry_run=%s poll=%ss",
        live_cfg.symbol, args.broker, cfg.dry_run, cfg.poll_seconds,
    )

    if args.once:
        engine.run_once()
    else:
        engine.run_forever()
    return 0


# ---------------------------------------------------------------------------
# T2.5 — watch / status / time (agent-ergonomics surface)
# ---------------------------------------------------------------------------

def _emit_t2_json(data: dict, cmd: str) -> None:
    """Emit a single JSON blob. No envelope here — the cli_runner wraps
    structured errors, but these commands already print JSON directly.
    """
    print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def _cmd_watch(args: argparse.Namespace) -> int:
    """Poll a named resource every N seconds; exit 0 with the snapshot when
    `--until EXPR` evaluates True; exit 124 if the deadline elapses first.

    Resources:
      quote   --symbol AAPL         → bid/ask/last/close/volume
      order   --order-id 12345      → latest order status
      position --symbol AAPL        → latest broker position snapshot

    EXPR is a restricted Python expression evaluated against the snapshot
    dict (see watch_expr.py). Examples:
      --until "last > 150"
      --until "status == 'Filled'"
      --until "filled_quantity >= 100"
    """
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    import time as _time
    from trading_algo.exit_codes import TIMEOUT
    from trading_algo.watch_expr import UnsafeExpression, evaluate

    resource: str = args.resource
    interval = max(0.2, float(args.every))
    timeout = float(args.timeout)
    deadline = _time.monotonic() + timeout if timeout > 0 else float("inf")

    # Pre-validate --until so a typo doesn't waste polls.
    try:
        evaluate(args.until, {})
    except UnsafeExpression as exc:
        print(f"ERROR: unsafe --until expression: {exc}", file=sys.stderr)
        return 2
    except SyntaxError as exc:
        print(f"ERROR: --until does not parse: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # Tolerated — may only succeed once a field is bound.
        pass

    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    broker.connect()

    def _fetch_quote() -> dict:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(args.strike) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        snap = broker.get_market_data_snapshot(instrument)
        return {
            "symbol": snap.instrument.symbol,
            "bid": snap.bid,
            "ask": snap.ask,
            "last": snap.last,
            "close": snap.close,
            "volume": snap.volume,
            "timestamp_epoch_s": snap.timestamp_epoch_s,
        }

    def _fetch_order() -> dict:
        oid = int(args.order_id)
        statuses = broker.get_order_statuses() if hasattr(broker, "get_order_statuses") else {}
        row = statuses.get(oid) or statuses.get(str(oid)) or {}
        return {
            "order_id": oid,
            "status": row.get("status"),
            "filled": row.get("filled"),
            "remaining": row.get("remaining"),
            "avg_fill_price": row.get("avg_fill_price") or row.get("avgFillPrice"),
        }

    def _fetch_position() -> dict:
        positions = broker.get_positions() if hasattr(broker, "get_positions") else []
        for p in positions or []:
            sym = p.get("symbol") if isinstance(p, dict) else getattr(p, "symbol", None)
            if sym == args.symbol:
                if isinstance(p, dict):
                    return {"symbol": sym, **p}
                return {
                    "symbol": sym,
                    "position": getattr(p, "position", None),
                    "avg_cost": getattr(p, "avg_cost", None),
                    "market_value": getattr(p, "market_value", None),
                    "unrealized_pnl": getattr(p, "unrealized_pnl", None),
                }
        return {"symbol": args.symbol, "position": 0}

    fetchers = {"quote": _fetch_quote, "order": _fetch_order, "position": _fetch_position}
    fetch = fetchers.get(resource)
    if fetch is None:
        broker.disconnect()
        print(f"ERROR: unknown watch resource: {resource!r}", file=sys.stderr)
        return 2

    last_snapshot: dict = {}
    polls = 0
    try:
        while _time.monotonic() < deadline:
            try:
                last_snapshot = fetch()
            except Exception as exc:
                logging.getLogger(__name__).warning("watch poll failed: %s", exc)
            polls += 1
            try:
                if evaluate(args.until, last_snapshot):
                    _emit_t2_json(
                        {
                            "matched": True,
                            "polls": polls,
                            "snapshot": last_snapshot,
                            "expression": args.until,
                        },
                        cmd="watch",
                    )
                    return 0
            except Exception as exc:
                logging.getLogger(__name__).debug("watch eval failed: %s", exc)
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break
            _time.sleep(min(interval, remaining))
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass

    _emit_t2_json(
        {
            "matched": False,
            "polls": polls,
            "snapshot": last_snapshot,
            "expression": args.until,
            "reason": "timeout",
        },
        cmd="watch",
    )
    return TIMEOUT


def _cmd_status(args: argparse.Namespace) -> int:
    """One JSON blob summarising the state of the world for an agent loop:
    broker connectivity, config (paper/live/dry-run), halt state, market
    hours, open-orders/positions counts. Each section independent — a
    missing section becomes null instead of failing the whole call.
    """
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    from trading_algo.market_rules import market_state

    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)

    # Broker section ------------------------------------------------------
    broker_block: dict = {
        "kind": args.broker,
        "host": cfg.ibkr.host,
        "port": cfg.ibkr.port,
        "client_id": cfg.ibkr.client_id,
        "connected": None,
        "paper": cfg.ibkr.port == 4002,
    }
    account_block: dict = {
        "net_liquidation": None,
        "buying_power": None,
        "open_orders": None,
        "open_positions": None,
    }
    if args.broker == "ibkr" and not getattr(args, "skip_broker", False):
        try:
            broker = _make_broker(args.broker, cfg)
            broker.connect()
            broker_block["connected"] = True
            try:
                acct = broker.get_account_summary() if hasattr(broker, "get_account_summary") else {}
                account_block["net_liquidation"] = acct.get("NetLiquidation")
                account_block["buying_power"] = acct.get("BuyingPower")
            except Exception:
                pass
            try:
                orders = broker.get_open_orders() if hasattr(broker, "get_open_orders") else []
                account_block["open_orders"] = len(orders or [])
            except Exception:
                pass
            try:
                positions = broker.get_positions() if hasattr(broker, "get_positions") else []
                account_block["open_positions"] = sum(
                    1 for p in (positions or [])
                    if (p.get("position") if isinstance(p, dict) else getattr(p, "position", 0)) not in (0, None)
                )
            except Exception:
                pass
            broker.disconnect()
        except Exception as exc:
            broker_block["connected"] = False
            broker_block["error"] = f"{type(exc).__name__}: {exc}"

    # Market section ------------------------------------------------------
    market_block = market_state()

    # Config section ------------------------------------------------------
    config_block = {
        "dry_run": cfg.dry_run,
        "live_enabled": cfg.live_enabled,
        "allow_live": cfg.allow_live,
        "require_paper": cfg.require_paper,
        "confirm_token_required": cfg.confirm_token_required,
        "poll_seconds": cfg.poll_seconds,
        "db_path": cfg.db_path or None,
    }

    # Halt section --------------------------------------------------------
    from trading_algo.halt import read_halt
    halt_state = read_halt()
    halt_block: dict = {"is_halted": halt_state is not None}
    if halt_state is not None:
        halt_block.update(halt_state.to_dict())

    out = {
        "broker": broker_block,
        "account": account_block,
        "market": market_block,
        "config": config_block,
        "halt": halt_block,
    }
    _emit_t2_json(out, cmd="status")
    return 0


def _cmd_time(args: argparse.Namespace) -> int:
    """Emit all the clocks an agent needs to plan actions. No broker call."""
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    from trading_algo.market_rules import market_state
    _emit_t2_json(market_state(), cmd="time")
    return 0


# ---------------------------------------------------------------------------
# T2.6 — events + reconcile (agent-first JSON surface)
# ---------------------------------------------------------------------------

def _cmd_events(args: argparse.Namespace) -> int:
    """Read the SEC-17a-4 NDJSON audit log under data/audit/*.jsonl.

    Agents reconstruct their own history after a crash, verify a request_id
    actually ran, or diff their assumptions against what really happened.
    No broker call — purely local file read.

    Filters:
      --since YYYY-MM-DD    inclusive lower date bound (local)
      --until YYYY-MM-DD    inclusive upper date bound (local)
      --cmd NAME            only entries for this subcommand
      --outcome ok|error    exit_code==0 vs non-zero
      --tail N              last N matching entries only
    """
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    from datetime import date
    from trading_algo.audit import iter_entries, tail as audit_tail

    def _parse_day(raw: str | None) -> date | None:
        if not raw:
            return None
        try:
            return date.fromisoformat(raw)
        except ValueError as exc:
            raise SystemExit(f"--since/--until must be YYYY-MM-DD: {raw!r} ({exc})")

    since = _parse_day(getattr(args, "since", None))
    until = _parse_day(getattr(args, "until", None))
    cmd_filter = getattr(args, "cmd_filter", None)
    outcome = getattr(args, "outcome", None)
    tail_n = getattr(args, "tail", None)

    if tail_n is not None and int(tail_n) > 0:
        entries = audit_tail(int(tail_n), cmd=cmd_filter, outcome=outcome)
        if since or until:
            entries = [
                e for e in entries
                if (not since or (e.get("ts") and date.fromisoformat(e["ts"][:10]) >= since))
                and (not until or (e.get("ts") and date.fromisoformat(e["ts"][:10]) <= until))
            ]
    else:
        entries = list(iter_entries(
            since=since, until=until, cmd=cmd_filter, outcome=outcome,
        ))

    # --fields / --summary projection (T2.7).
    projected = _maybe_project_and_summarize(entries, args, summarizer=None)
    if getattr(args, "summary", False):
        # Minimal rollup: count + distinct cmds + outcome breakdown.
        from collections import Counter
        cmds = Counter(e.get("cmd") for e in entries)
        outcomes = {
            "ok": sum(1 for e in entries if e.get("exit_code") == 0),
            "error": sum(1 for e in entries if e.get("exit_code") not in (0, None)),
        }
        projected = {
            "count": len(entries),
            "by_cmd": dict(cmds),
            "outcome": outcomes,
        }
        _emit_t2_json(projected, cmd="events")
        return 0
    _emit_t2_json({"count": len(entries), "entries": projected}, cmd="events")
    return 0


# ---------------------------------------------------------------------------
# T2.7 — --explain / tools-describe / --fields / --summary
# ---------------------------------------------------------------------------

def _maybe_handle_explain(args: argparse.Namespace) -> int | None:
    """If `--explain` was passed, emit explain(cmd) JSON and short-circuit.

    Returns an exit code to return from the handler, or None if normal
    execution should proceed.
    """
    if not getattr(args, "explain", False):
        return None
    from trading_algo.explain import explain
    cmd_name = getattr(args, "cmd", None) or "unknown"
    _emit_t2_json({"cmd": cmd_name, "explanation": explain(cmd_name)}, cmd="explain")
    return 0


def _cmd_tools_describe(args: argparse.Namespace) -> int:
    """Emit the JSONSchema array for every subcommand — agents use this
    to auto-generate tool/function definitions for LLM tool calling.
    """
    from trading_algo.tool_schema import describe_tools
    tools = describe_tools(build_parser())
    _emit_t2_json({"count": len(tools), "tools": tools}, cmd="tools-describe")
    return 0


def _maybe_project_and_summarize(
    rows: list[dict],
    args: argparse.Namespace,
    summarizer=None,
) -> list[dict] | dict:
    """Apply --fields / --summary to list-of-dict output.

    Precedence: --summary wins over --fields if both are set.
    """
    from trading_algo.projection import parse_fields, project_rows

    if getattr(args, "summary", False) and summarizer is not None:
        return summarizer(rows)
    fields = parse_fields(getattr(args, "fields", None))
    if fields:
        return project_rows(rows, fields)
    return rows


def _cmd_stream(args: argparse.Namespace) -> int:
    """Stream market-data snapshots as NDJSON to stdout (and optionally
    a buffer file) until `--duration` expires or SIGINT.

    This is a poll-based stream — each tick is one broker snapshot,
    rate-limited by `--every` seconds and by the IBKR rate-limit layer.
    For turn-based agents it's cheaper than an open WebSocket:

      - Fire `stream --buffer-to ticks.ndjson --duration 60` in a
        subprocess.
      - When the turn ends, use `tail-ticks --file ticks.ndjson
        --from-seq N` to consume only the newly-appended ticks.

    Each emitted line carries `_seq` (monotonic) and `_ts_epoch_ms` so
    downstream consumers can dedupe / resume cleanly.
    """
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    import time as _time
    from trading_algo.exit_codes import TIMEOUT

    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    broker.connect()

    interval = max(0.2, float(args.every))
    duration = float(args.duration)
    deadline = _time.monotonic() + duration if duration > 0 else float("inf")

    instrument = validate_instrument(
        InstrumentSpec(
            kind=args.kind, symbol=args.symbol,
            exchange=args.exchange, currency=args.currency,
            expiry=args.expiry, right=getattr(args, "right", None),
            strike=(float(args.strike) if getattr(args, "strike", None) is not None else None),
            multiplier=getattr(args, "multiplier", None),
        )
    )

    buf_fh = None
    if getattr(args, "buffer_to", None):
        from pathlib import Path
        p = Path(args.buffer_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        buf_fh = open(p, "a", encoding="utf-8", buffering=1)  # line-buffered

    seq = 0
    try:
        while _time.monotonic() < deadline:
            try:
                snap = broker.get_market_data_snapshot(instrument)
                seq += 1
                payload = {
                    "_seq": seq,
                    "_ts_epoch_ms": int(_time.time() * 1000),
                    "symbol": snap.instrument.symbol,
                    "bid": snap.bid, "ask": snap.ask, "last": snap.last,
                    "close": snap.close, "volume": snap.volume,
                    "timestamp_epoch_s": snap.timestamp_epoch_s,
                }
                line = json.dumps(payload, default=str, ensure_ascii=False)
                print(line, flush=True)
                if buf_fh is not None:
                    buf_fh.write(line + "\n")
            except Exception as exc:
                logging.getLogger(__name__).warning("stream tick failed: %s", exc)
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break
            _time.sleep(min(interval, remaining))
        return 0
    finally:
        if buf_fh is not None:
            try:
                buf_fh.close()
            except Exception:
                pass
        try:
            broker.disconnect()
        except Exception:
            pass


def _cmd_tail_ticks(args: argparse.Namespace) -> int:
    """Read ticks from a `stream --buffer-to` file, optionally starting
    at `--from-seq` and limiting to `--max` lines.

    No broker call. Safe to run concurrently with the producer — the
    buffer file is line-buffered.
    """
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    from pathlib import Path

    path = Path(args.file)
    if not path.exists():
        _emit_t2_json({"count": 0, "ticks": []}, cmd="tail-ticks")
        return 0
    from_seq = int(getattr(args, "from_seq", 0) or 0)
    max_n = getattr(args, "max", None)

    matching: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = entry.get("_seq", 0)
            if seq > from_seq:
                matching.append(entry)
    if max_n is not None:
        matching = matching[-int(max_n):]
    last_seq = matching[-1].get("_seq") if matching else from_seq
    _emit_t2_json(
        {"count": len(matching), "last_seq": last_seq, "ticks": matching},
        cmd="tail-ticks",
    )
    return 0


def _cmd_groups_list(args: argparse.Namespace) -> int:
    """List distinct groups observed in the OMS DB."""
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if not cfg.db_path:
        raise SystemExit("groups-list requires TRADING_DB_PATH to be set")
    store = SqliteStore(cfg.db_path)
    try:
        groups = store.list_groups()
    finally:
        store.close()
    _emit_t2_json({"count": len(groups), "groups": groups}, cmd="groups-list")
    return 0


def _cmd_groups_show(args: argparse.Namespace) -> int:
    """Show every order in a given group_id."""
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if not cfg.db_path:
        raise SystemExit("groups-show requires TRADING_DB_PATH to be set")
    store = SqliteStore(cfg.db_path)
    try:
        orders = store.orders_by_group(args.group_id)
    finally:
        store.close()
    _emit_t2_json(
        {"group_id": args.group_id, "count": len(orders), "orders": orders},
        cmd="groups-show",
    )
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    """Reconcile OMS's view of open orders with broker's live openTrades().

    A structured, agent-consumable version of `oms-reconcile`. Same
    underlying engine logic (OrderManager.reconcile) but emits JSON.

    Requires TRADING_DB_PATH to be set — reconcile only makes sense when
    the OMS has persistent state to reconcile against.
    """
    rc = _maybe_handle_explain(args)
    if rc is not None:
        return rc
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if not cfg.db_path:
        raise SystemExit("reconcile requires TRADING_DB_PATH to be set")
    broker = _make_broker(args.broker, cfg)
    broker.connect()
    try:
        oms = OrderManager(broker, cfg, confirm_token=args.confirm_token)
        try:
            res = oms.reconcile()
        finally:
            oms.close()
        entries = []
        for oid, st in (res or {}).items():
            entries.append({"order_id": oid, "status": st})
        out = {
            "reconciled_count": len(entries),
            "orders": entries,
        }
        _emit_t2_json(out, cmd="reconcile")
        return 0
    finally:
        broker.disconnect()


def _cmd_halt(args: argparse.Namespace) -> int:
    """Write the HALTED sentinel. Every subsequent write command refuses
    until the sentinel is removed. Safe to call while already halted (the
    new reason / expires-in overwrite the existing sentinel).
    """
    from trading_algo.halt import parse_duration, write_halt

    expires_seconds: float | None = None
    if getattr(args, "expires_in", None):
        try:
            expires_seconds = parse_duration(args.expires_in)
        except ValueError as exc:
            print(f"ERROR: --expires-in: {exc}", file=sys.stderr)
            return 2

    state = write_halt(
        reason=args.reason,
        by=(args.by or os.getenv("TRADING_OPERATOR", "operator")),
        expires_in_seconds=expires_seconds,
    )
    out = state.to_dict()
    out["halted"] = True
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    """Clear the HALTED sentinel. Requires --confirm-resume — distinct
    from --yes so a replayed `halt --yes` can't accidentally lift the halt.
    """
    from trading_algo.halt import clear_halt

    if not getattr(args, "confirm_resume", False):
        print(
            "ERROR: `resume` requires --confirm-resume. This is "
            "intentionally a different token from --yes to prevent an "
            "accidental lift of a halt.",
            file=sys.stderr,
        )
        return 2
    cleared = clear_halt()
    print(json.dumps({"resumed": cleared}, indent=2))
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from trading_algo.llm.chat import main as chat_main

    argv: list[str] = []
    if args.broker is not None:
        argv += ["--broker", str(args.broker)]
    if args.confirm_token is not None:
        argv += ["--confirm-token", str(args.confirm_token)]
    if args.ibkr_host is not None:
        argv += ["--ibkr-host", str(args.ibkr_host)]
    if args.ibkr_port is not None:
        argv += ["--ibkr-port", str(args.ibkr_port)]
    if args.ibkr_client_id is not None:
        argv += ["--ibkr-client-id", str(args.ibkr_client_id)]
    if bool(args.no_stream):
        argv += ["--no-stream"]
    if bool(args.show_raw):
        argv += ["--show-raw"]
    if bool(args.no_color):
        argv += ["--no-color"]
    if bool(getattr(args, "quiet_ibkr_logs", False)):
        argv += ["--quiet-ibkr-logs"]
    if getattr(args, "ui", None):
        argv += ["--ui", str(args.ui)]
    return int(chat_main(argv))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading-algo", description="IBKR paper trading algo skeleton")
    p.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    p.add_argument("--ibkr-host", default=None, help="Override IBKR host (default from env/.env)")
    p.add_argument("--ibkr-port", default=None, help="Override IBKR port (default from env/.env)")
    p.add_argument("--ibkr-client-id", default=None, help="Override IBKR clientId (default from env/.env)")
    conn_group = p.add_mutually_exclusive_group()
    conn_group.add_argument(
        "--paper", action="store_true",
        help="Connect to IB Gateway paper trading (port 4002).",
    )
    conn_group.add_argument(
        "--live", action="store_true",
        help="Connect to TWS live account (port 7496). Implies --allow-live.",
    )
    p.add_argument(
        "--confirm-token",
        default=None,
        help="Must match TRADING_ORDER_TOKEN if TRADING_CONFIRM_TOKEN_REQUIRED=true",
    )
    p.add_argument("--dry-run", action="store_true", help="Stage orders only (no sends), overrides TRADING_DRY_RUN")
    p.add_argument("--no-dry-run", action="store_true", help="Allow sending orders, overrides TRADING_DRY_RUN")
    p.add_argument(
        "--allow-live", action="store_true",
        help="Allow connecting to LIVE (non-paper) IBKR accounts. "
             "All order operations will require interactive YES confirmation.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    place = sub.add_parser("place-order", help="Place a single test order")
    place.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    place.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    place.add_argument("--symbol", required=True)
    place.add_argument("--exchange", default=None)
    place.add_argument("--currency", default=None)
    place.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    place.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    place.add_argument("--strike", default=None, help="OPT only: strike price")
    place.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    place.add_argument("--side", choices=["BUY", "SELL"], required=True)
    place.add_argument("--qty", required=True)
    place.add_argument("--type", choices=["MKT", "LMT", "STP", "STPLMT"], default="MKT")
    place.add_argument("--limit-price", default=None)
    place.add_argument("--stop-price", default=None)
    place.add_argument("--tif", default="DAY", help="DAY|GTC|GTD (if GTD, set --good-till-date)")
    place.add_argument("--good-till-date", default=None, help="IBKR GTD time string (e.g. 20260116 09:30:00)")
    place.add_argument("--outside-rth", action="store_true", help="Allow execution outside regular trading hours")
    place.add_argument("--account", default=None)
    place.add_argument("--order-ref", default=None)
    place.add_argument("--oca-group", default=None)
    place.add_argument("--no-transmit", action="store_true", help="Create order with transmit=false (advanced)")
    place.add_argument(
        "--idempotency-key", default=None, metavar="KEY",
        help="Durable idempotency key. On retry with the same key, replay the "
             "prior result from data/idempotency.sqlite (never re-transmit). "
             "BLAKE2b-derived into an IBKR orderRef so orderbook-based dedup "
             "works across process restarts. Required for crash-safe retries.",
    )
    place.set_defaults(func=_cmd_place_order)

    snap = sub.add_parser("snapshot", help="Fetch a market data snapshot")
    snap.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    snap.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    snap.add_argument("--symbol", required=True)
    snap.add_argument("--exchange", default=None)
    snap.add_argument("--currency", default=None)
    snap.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    snap.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    snap.add_argument("--strike", default=None, help="OPT only: strike price")
    snap.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    snap.set_defaults(func=_cmd_snapshot)

    hist = sub.add_parser("history", help="Fetch historical bars (IBKR reqHistoricalData)")
    hist.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    hist.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    hist.add_argument("--symbol", required=True)
    hist.add_argument("--exchange", default=None)
    hist.add_argument("--currency", default=None)
    hist.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    hist.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    hist.add_argument("--strike", default=None, help="OPT only: strike price")
    hist.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    hist.add_argument("--duration", default="1 D", help="IBKR durationStr (e.g. '1 D', '2 W')")
    hist.add_argument("--bar-size", default="5 mins", help="IBKR barSizeSetting (e.g. '1 min', '5 mins')")
    hist.add_argument("--what-to-show", default="TRADES")
    hist.add_argument("--use-rth", action="store_true")
    hist.set_defaults(func=_cmd_history)

    run = sub.add_parser("run", help="Run example strategy loop")
    run.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    run.add_argument("--symbol", default="AAPL")
    run.add_argument("--poll-seconds", type=int, default=None)
    run.add_argument("--once", action="store_true")
    run.set_defaults(func=_cmd_run)

    status = sub.add_parser("order-status", help="Get order status by orderId")
    status.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    status.add_argument("--order-id", required=True)
    status.set_defaults(func=_cmd_order_status)

    cancel = sub.add_parser("cancel-order", help="Cancel order by orderId")
    cancel.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    cancel.add_argument("--order-id", required=True)
    cancel.set_defaults(func=_cmd_cancel_order)

    mod = sub.add_parser("modify-order", help="Modify an existing order by orderId")
    mod.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    mod.add_argument("--order-id", required=True)
    mod.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    mod.add_argument("--symbol", required=True)
    mod.add_argument("--exchange", default=None)
    mod.add_argument("--currency", default=None)
    mod.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    mod.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    mod.add_argument("--strike", default=None, help="OPT only: strike price")
    mod.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    mod.add_argument("--side", choices=["BUY", "SELL"], required=True)
    mod.add_argument("--qty", required=True)
    mod.add_argument("--type", choices=["MKT", "LMT", "STP", "STPLMT"], default="LMT")
    mod.add_argument("--limit-price", default=None)
    mod.add_argument("--stop-price", default=None)
    mod.add_argument("--tif", default="DAY")
    mod.add_argument("--good-till-date", default=None)
    mod.add_argument("--outside-rth", action="store_true")
    mod.add_argument("--account", default=None)
    mod.add_argument("--order-ref", default=None)
    mod.add_argument("--oca-group", default=None)
    mod.add_argument("--no-transmit", action="store_true")
    mod.set_defaults(func=_cmd_modify_order)

    bracket = sub.add_parser("place-bracket", help="Place a bracket order (LMT entry + TP LMT + SL STP)")
    bracket.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    bracket.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    bracket.add_argument("--symbol", required=True)
    bracket.add_argument("--exchange", default=None)
    bracket.add_argument("--currency", default=None)
    bracket.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    bracket.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    bracket.add_argument("--strike", default=None, help="OPT only: strike price")
    bracket.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    bracket.add_argument("--side", choices=["BUY", "SELL"], required=True)
    bracket.add_argument("--qty", required=True)
    bracket.add_argument("--entry-limit", required=True)
    bracket.add_argument("--take-profit", required=True)
    bracket.add_argument("--stop-loss", required=True)
    bracket.add_argument("--tif", default="DAY")
    bracket.set_defaults(func=_cmd_place_bracket)

    smoke = sub.add_parser("paper-smoke", help="Paper connectivity smoke test (connect + verify paper + snapshot; optional place+cancel)")
    smoke.add_argument("--broker", choices=["ibkr"], default="ibkr")
    smoke.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    smoke.add_argument("--symbol", default="AAPL")
    smoke.add_argument("--exchange", default=None)
    smoke.add_argument("--currency", default=None)
    smoke.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    smoke.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    smoke.add_argument("--strike", default=None, help="OPT only: strike price")
    smoke.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    smoke.add_argument("--order-test", action="store_true", help="Place a tiny LMT order and cancel it (requires TRADING_LIVE_ENABLED + token)")
    smoke.add_argument("--side", choices=["BUY", "SELL"], default="BUY")
    smoke.add_argument("--qty", default="1")
    smoke.set_defaults(func=_cmd_paper_smoke)

    rec = sub.add_parser("oms-reconcile", help="Reconcile open orders from TRADING_DB_PATH with broker open orders")
    rec.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    rec.set_defaults(func=_cmd_oms_reconcile)

    track = sub.add_parser("oms-track", help="Poll and persist order status transitions until terminal/timeout")
    track.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    track.add_argument("--poll-seconds", default="1.0")
    track.add_argument("--timeout-seconds", default=None)
    track.set_defaults(func=_cmd_oms_track)

    bt = sub.add_parser("backtest", help="Run a deterministic historical backtest from a CSV file")
    bt.add_argument("--csv", required=True, help="CSV with columns: timestamp,open,high,low,close[,volume]")
    bt.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    bt.add_argument("--symbol", required=True)
    bt.add_argument("--exchange", default=None)
    bt.add_argument("--currency", default=None)
    bt.add_argument("--expiry", default=None)
    bt.add_argument("--right", choices=["C", "P"], default=None)
    bt.add_argument("--strike", default=None)
    bt.add_argument("--multiplier", default=None)
    bt.add_argument("--initial-cash", type=float, default=100000.0)
    bt.add_argument("--commission-per-order", type=float, default=0.0)
    bt.add_argument("--slippage-bps", type=float, default=0.0)
    bt.add_argument("--spread", type=float, default=0.0)
    bt.add_argument("--db-path", default=None)
    bt.set_defaults(func=_cmd_backtest)

    # --- T2.5: watch / status / time ---
    watch_p = sub.add_parser(
        "watch",
        help="Poll a resource until a restricted expression is True (exit 124 on timeout)",
    )
    watch_p.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    watch_p.add_argument(
        "--resource", choices=["quote", "order", "position"], required=True,
        help="What to poll: quote (market data), order (by id), position (by symbol)",
    )
    watch_p.add_argument("--symbol", default=None, help="Symbol for quote/position resources")
    watch_p.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    watch_p.add_argument("--exchange", default=None)
    watch_p.add_argument("--currency", default=None)
    watch_p.add_argument("--expiry", default=None)
    watch_p.add_argument("--right", choices=["C", "P"], default=None)
    watch_p.add_argument("--strike", default=None)
    watch_p.add_argument("--multiplier", default=None)
    watch_p.add_argument("--order-id", default=None, help="Order id (order resource)")
    watch_p.add_argument(
        "--until", required=True,
        help="Restricted Python expression over the snapshot (e.g. 'last > 150', "
             "'status == \"Filled\"'). Only compare/logic/arith ops are allowed.",
    )
    watch_p.add_argument("--every", type=float, default=2.0,
                         help="Poll interval seconds (min 0.2)")
    watch_p.add_argument("--timeout", type=float, default=60.0,
                         help="Max seconds to poll before giving up (exit 124)")
    watch_p.add_argument("--explain", action="store_true",
                         help="Print the command's explanation and exit 0 without running it")
    watch_p.set_defaults(func=_cmd_watch)

    status_p = sub.add_parser(
        "status",
        help="One JSON blob: broker connectivity, config, halt state, market hours",
    )
    status_p.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    status_p.add_argument("--skip-broker", action="store_true",
                          help="Skip broker connect (config + market + halt only)")
    status_p.add_argument("--explain", action="store_true",
                          help="Print the command's explanation and exit 0 without running it")
    status_p.set_defaults(func=_cmd_status)

    time_p = sub.add_parser(
        "time",
        help="Emit clocks — UTC, ET, market open/close, weekday (no broker call)",
    )
    time_p.add_argument("--explain", action="store_true",
                        help="Print the command's explanation and exit 0 without running it")
    time_p.set_defaults(func=_cmd_time)

    # --- T2.6: events + reconcile ---
    events_p = sub.add_parser(
        "events",
        help="Read the local NDJSON audit log (data/audit/*.jsonl). No broker call.",
    )
    events_p.add_argument("--since", default=None, help="Inclusive lower date bound (YYYY-MM-DD)")
    events_p.add_argument("--until", default=None, help="Inclusive upper date bound (YYYY-MM-DD)")
    events_p.add_argument("--cmd-filter", dest="cmd_filter", default=None,
                          help="Only entries where cmd==NAME")
    events_p.add_argument("--outcome", choices=["ok", "error"], default=None,
                          help="ok = exit_code 0; error = non-zero")
    events_p.add_argument("--tail", type=int, default=None,
                          help="Only the most recent N matching entries")
    events_p.add_argument("--fields", default=None,
                          help="Comma-separated keys to keep per entry (e.g. 'ts,cmd,exit_code')")
    events_p.add_argument("--summary", action="store_true",
                          help="Emit roll-up (count + by-cmd + outcome) instead of entries")
    events_p.add_argument("--explain", action="store_true",
                          help="Print the command's explanation and exit 0 without running it")
    events_p.set_defaults(func=_cmd_events)

    rec2_p = sub.add_parser(
        "reconcile",
        help="Reconcile OMS DB with broker openTrades (JSON output, agent-friendly)",
    )
    rec2_p.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    rec2_p.add_argument("--explain", action="store_true",
                        help="Print the command's explanation and exit 0 without running it")
    rec2_p.set_defaults(func=_cmd_reconcile)

    td_p = sub.add_parser(
        "tools-describe",
        help="Emit JSONSchema for every subcommand — agents build tool-call specs from this",
    )
    td_p.set_defaults(func=_cmd_tools_describe)

    # --- T4.2: order groups ---
    gl_p = sub.add_parser(
        "groups-list",
        help="List distinct group_ids in the OMS DB (baskets, bracket legs, etc.)",
    )
    gl_p.add_argument("--explain", action="store_true",
                      help="Print explanation and exit 0 without running")
    gl_p.set_defaults(func=_cmd_groups_list)

    gs_p = sub.add_parser(
        "groups-show",
        help="Show every order in a given group_id",
    )
    gs_p.add_argument("--group-id", required=True, dest="group_id")
    gs_p.add_argument("--explain", action="store_true",
                      help="Print explanation and exit 0 without running")
    gs_p.set_defaults(func=_cmd_groups_show)

    # --- T4.3: stream / tail-ticks ---
    stream_p = sub.add_parser(
        "stream",
        help="Poll market-data snapshots and emit NDJSON ticks (optionally to a buffer file)",
    )
    stream_p.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    stream_p.add_argument("--symbol", required=True)
    stream_p.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    stream_p.add_argument("--exchange", default=None)
    stream_p.add_argument("--currency", default=None)
    stream_p.add_argument("--expiry", default=None)
    stream_p.add_argument("--right", choices=["C", "P"], default=None)
    stream_p.add_argument("--strike", default=None)
    stream_p.add_argument("--multiplier", default=None)
    stream_p.add_argument("--every", type=float, default=1.0,
                          help="Poll interval seconds (min 0.2)")
    stream_p.add_argument("--duration", type=float, default=60.0,
                          help="Stream duration seconds; <=0 means forever (SIGINT to stop)")
    stream_p.add_argument("--buffer-to", dest="buffer_to", default=None,
                          help="Append NDJSON ticks to this file (created if missing)")
    stream_p.add_argument("--explain", action="store_true",
                          help="Print explanation and exit 0 without running")
    stream_p.set_defaults(func=_cmd_stream)

    tt_p = sub.add_parser(
        "tail-ticks",
        help="Read a stream --buffer-to file; emit only ticks after --from-seq (no broker call)",
    )
    tt_p.add_argument("--file", required=True, help="Path to an NDJSON buffer produced by `stream --buffer-to`")
    tt_p.add_argument("--from-seq", dest="from_seq", type=int, default=0,
                      help="Only include ticks with _seq > this value")
    tt_p.add_argument("--max", type=int, default=None, help="Cap at the last N matching ticks")
    tt_p.add_argument("--explain", action="store_true",
                      help="Print explanation and exit 0 without running")
    tt_p.set_defaults(func=_cmd_tail_ticks)

    # --- kill switch (halt / resume) ---
    halt_p = sub.add_parser(
        "halt",
        help="Write the HALTED sentinel — refuses all write commands until cleared",
    )
    halt_p.add_argument("--reason", required=True,
                        help="Short description of why trading is halted (stored in sentinel)")
    halt_p.add_argument("--by", default=None,
                        help="Operator / agent identifier. Defaults to $TRADING_OPERATOR or 'operator'.")
    halt_p.add_argument("--expires-in", default=None, metavar="DURATION",
                        help="Auto-clear after duration (e.g. 30s, 5m, 1h, 2d). "
                             "Without this flag the halt persists until `resume`.")
    halt_p.set_defaults(func=_cmd_halt)

    resume_p = sub.add_parser(
        "resume",
        help="Clear the HALTED sentinel (requires --confirm-resume)",
    )
    resume_p.add_argument("--confirm-resume", action="store_true",
                          help="Required. Distinct from --yes to prevent accidental resume.")
    resume_p.set_defaults(func=_cmd_resume)

    exp = sub.add_parser("export-history", help="Export IBKR historical bars to a backtest CSV")
    exp.add_argument("--broker", choices=["ibkr"], default="ibkr")
    exp.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    exp.add_argument("--symbol", required=True)
    exp.add_argument("--exchange", default=None)
    exp.add_argument("--currency", default=None)
    exp.add_argument("--expiry", default=None)
    exp.add_argument("--right", choices=["C", "P"], default=None)
    exp.add_argument("--strike", default=None)
    exp.add_argument("--multiplier", default=None)
    exp.add_argument("--out-csv", required=True)
    exp.add_argument("--overwrite", action="store_true")
    exp.add_argument("--bar-size", default="5 mins")
    exp.add_argument("--duration-per-call", default="30 D")
    exp.add_argument("--what-to-show", default="TRADES")
    exp.add_argument("--use-rth", action="store_true")
    exp.add_argument("--end-datetime", default=None, help="IBKR endDateTime; empty means now. Epoch/ISO are accepted.")
    exp.add_argument("--pacing-sleep-seconds", default="0.25")
    exp.add_argument("--max-calls", default="500")
    exp.add_argument("--validate", action="store_true")
    exp.set_defaults(func=_cmd_export_history)

    obt = sub.add_parser("backtest-options", help="Backtest Wheel or PMCC strategy on IBKR historical data")
    obt.add_argument("--strategy", choices=["wheel", "pmcc"], default="wheel", help="Strategy to backtest")
    obt.add_argument("--symbols", default="SOFI,F,PLTR", help="Comma-separated symbols")
    obt.add_argument("--capital", default="10000", help="Initial capital")
    obt.add_argument("--duration", default="2 Y", help="IBKR duration string (e.g. '1 Y', '2 Y')")
    obt.add_argument("--dte", default="35", help="Target DTE for short options")
    obt.add_argument("--put-delta", default="0.30", help="Target put delta (Wheel)")
    obt.add_argument("--call-delta", default="0.30", help="Target call delta (Wheel/PMCC)")
    obt.add_argument("--leaps-delta", default="0.80", help="Target LEAPS delta (PMCC)")
    obt.add_argument("--short-delta", default="0.25", help="Target short call delta (PMCC)")
    obt.add_argument("--profit-target", default="0.50", help="Close at N%% profit (0.50 = 50%%)")
    obt.add_argument("--iv-premium", default="1.20", help="IV/RV premium factor for simulation")
    obt.set_defaults(func=_cmd_backtest_options)

    llm_run = sub.add_parser("llm-run", help="Run the LLM trader loop (paper-only enforced)")
    llm_run.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    llm_run.add_argument("--sleep-seconds", type=float, default=5.0)
    llm_run.add_argument("--max-ticks", type=int, default=None)
    llm_run.add_argument("--once", action="store_true", help="Run exactly one LLM tick")
    llm_run.set_defaults(func=_cmd_llm_run)

    scan = sub.add_parser("scan", help="Run IBKR market-wide scanner (e.g. top gainers, most active, high options volume)")
    scan.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    scan.add_argument("--scan-type", default="MOST_ACTIVE",
        help="IBKR scan code: MOST_ACTIVE, TOP_PERC_GAIN, TOP_PERC_LOSE, "
             "HOT_BY_VOLUME, HIGH_OPT_IMP_VOLAT, HOT_BY_OPT_VOLUME, "
             "HIGH_DIVIDEND_YIELD_IB, TOP_OPEN_PERC_GAIN, TOP_OPEN_PERC_LOSE, "
             "HIGH_VS_13W_HL, LOW_VS_13W_HL, MOST_ACTIVE_USD, HIGH_SYNTH_BID_REV_NAT")
    scan.add_argument("--instrument-type", default="STK", help="STK, FUT, IND, etc.")
    scan.add_argument("--location", default="STK.US.MAJOR",
        help="Scanner location: STK.US.MAJOR, STK.US, STK.NASDAQ, STK.NYSE, STK.AMEX")
    scan.add_argument("--max-results", type=int, default=25, help="Max results 1-50")
    scan.add_argument("--min-price", type=float, default=None, help="Min stock price filter")
    scan.add_argument("--max-price", type=float, default=None, help="Max stock price filter")
    scan.add_argument("--min-volume", type=int, default=None, help="Min volume filter")
    scan.add_argument("--min-market-cap", type=float, default=None, help="Min market cap in USD (e.g. 1000000000 for $1B)")
    scan.add_argument("--max-market-cap", type=float, default=None, help="Max market cap in USD")
    scan.set_defaults(func=_cmd_scan)

    wl = sub.add_parser("wheel-live", help="Run Wheel options strategy live via Engine polling loop")
    wl.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    wl.add_argument("--symbol", required=True, help="Underlying symbol (e.g. SOFI)")
    wl.add_argument("--capital", default="10000", help="Initial capital for sizing")
    wl.add_argument("--put-delta", default="0.30", help="Target put delta")
    wl.add_argument("--call-delta", default="0.30", help="Target call delta")
    wl.add_argument("--dte", default="45", help="Target DTE for short options")
    wl.add_argument("--profit-target", default="0.50", help="Close at N%% profit")
    wl.add_argument("--min-iv-rank", default="25", help="Minimum IV rank to open")
    wl.add_argument("--use-mkt", action="store_true", help="Use MKT orders instead of LMT")
    wl.add_argument("--lmt-offset-pct", default="0.02", help="LMT price offset from theoretical (0.02 = 2%%)")
    wl.add_argument("--history-bars", default="300", help="Price history bars for IV calc")
    wl.add_argument("--iv-window", default="30", help="Rolling window for realized vol")
    wl.add_argument("--poll-seconds", default=None, help="Override poll interval (seconds)")
    wl.add_argument("--once", action="store_true", help="Run a single tick then exit")
    wl.set_defaults(func=_cmd_wheel_live)

    chat = sub.add_parser("chat", help="Interactive terminal chat (Gemini + OMS tools)")
    chat.add_argument("--broker", choices=["ibkr", "sim"], default=None)
    chat.add_argument("--no-stream", action="store_true")
    chat.add_argument("--show-raw", action="store_true")
    chat.add_argument("--no-color", action="store_true")
    chat.add_argument("--quiet-ibkr-logs", action="store_true", dest="quiet_ibkr_logs")
    chat.add_argument("--ui", choices=["auto", "plain", "rich", "tui"], default="auto")
    chat.set_defaults(func=_cmd_chat)

    add_journal_subparser(sub)

    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_if_present()
    cfg = TradingConfig.from_env()

    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)

    # prompt_toolkit runs in full-screen mode for `chat --ui tui`; any stdout/stderr writes from
    # logging will corrupt the terminal UI. Route logs to a file instead.
    tui_mode = bool(getattr(args, "ui", None) == "tui" and getattr(args, "func", None) == _cmd_chat)
    if tui_mode:
        log_path = os.getenv("TUI_LOG_PATH", "logs/tui.log")
        configure_logging(level=log_level, log_file=log_path, console=False)
    else:
        configure_logging(level=log_level)
    logging.getLogger(__name__).debug("Loaded config: %s", cfg)

    # TUI mode owns the terminal — bypass the structured runner (stderr
    # writes would corrupt the full-screen UI).
    if tui_mode:
        return int(args.func(args))

    # Every other invocation: audit + structured errors + exit-code
    # classification via the shared runner.
    from trading_algo.cli_runner import run_command
    return run_command(args, default_cmd_name="cli")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

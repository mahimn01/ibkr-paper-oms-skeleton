from __future__ import annotations

import json
import time
from dataclasses import dataclass

from trading_algo.broker.base import Broker, OrderRequest, OrderResult, OrderStatus
from trading_algo.config import TradingConfig
from trading_algo.halt import assert_not_halted
from trading_algo.orders import TradeIntent
from trading_algo.persistence import SqliteStore


@dataclass(frozen=True)
class OMSResult:
    order_id: str
    status: str


class OrderManager:
    """
    Minimal OMS:
    - submit/modify/cancel through Broker
    - persist run + orders + status events to SQLite (optional)
    - keep strict safety gates for IBKR sends (dry-run/live/token)
    """

    def __init__(self, broker: Broker, cfg: TradingConfig, *, confirm_token: str | None = None) -> None:
        self._broker = broker
        self._cfg = cfg
        self._confirm_token = confirm_token
        self._store: SqliteStore | None = SqliteStore(cfg.db_path) if cfg.db_path else None
        self._run_id: int | None = self._store.start_run(cfg) if self._store else None
        self._last_status: dict[str, str] = {}

    def close(self) -> None:
        if self._store is not None and self._run_id is not None:
            self._store.end_run(self._run_id)
            self._store.close()
        self._store = None
        self._run_id = None

    def reconcile(self) -> dict[str, str]:
        """
        Reconcile orders persisted in SQLite with the broker's current open orders/trades.

        Returns a mapping of order_id -> status for reconciled orders.
        """
        if self._store is None:
            return {}
        tracked = self._store.list_non_terminal_order_ids()
        broker_open = {st.order_id: st for st in self._broker.list_open_order_statuses()}

        results: dict[str, str] = {}
        for oid in tracked:
            st = broker_open.get(str(oid))
            if st is None:
                try:
                    st = self._broker.get_order_status(str(oid))
                except Exception as exc:
                    self._log_error("oms.reconcile", f"order_id={oid} err={exc}")
                    continue
            self._record_status(st)
            results[str(oid)] = st.status
        return results

    def track_open_orders(self, *, poll_seconds: float = 1.0, timeout_seconds: float | None = None) -> None:
        """
        Poll broker open orders and record status transitions until all become terminal or timeout.
        """
        start = time.time()
        while True:
            open_statuses = list(self._broker.list_open_order_statuses())
            for st in open_statuses:
                self._record_status(st)

            # also advance any tracked orders that might not appear in openTrades but still have statuses
            if self._store is not None:
                for oid in self._store.list_non_terminal_order_ids():
                    if any(s.order_id == oid for s in open_statuses):
                        continue
                    try:
                        st = self._broker.get_order_status(oid)
                    except Exception:
                        continue
                    self._record_status(st)

            if not open_statuses and (self._store is None or not self._store.list_non_terminal_order_ids()):
                return
            if timeout_seconds is not None and (time.time() - start) > float(timeout_seconds):
                self._log_error("oms.track", f"timeout_seconds={timeout_seconds} open={len(open_statuses)}")
                return
            time.sleep(float(poll_seconds))

    def submit(self, req: OrderRequest) -> OMSResult:
        # Halt sentinel is the first gate, *before* normalisation, dry-run,
        # or any broker call. Even dry-runs are blocked while halted to
        # match operator intent ("stop everything until I clear it").
        assert_not_halted()
        req = req.normalized()
        self._require_idempotency_key(req)
        self._authorize_send()
        if self._cfg.dry_run:
            self._log_error("oms.submit", "dry_run")
            return OMSResult(order_id="dry-run", status="DryRun")
        res = self._broker.place_order(req)
        self._log_order(req, res)
        self._log_status(res.order_id)
        return OMSResult(order_id=res.order_id, status=res.status)

    @staticmethod
    def _require_idempotency_key(req: OrderRequest) -> None:
        """Defense-in-depth assertion that every send carries an idempotency key.

        OrderRequest.normalized() auto-fills order_ref with a UUID4 when
        missing, so this should fire only if a caller bypassed normalisation
        or supplied an explicit empty string. Either case is a programmer
        error worth raising on.
        """
        if not req.order_ref or not req.order_ref.strip():
            raise ValueError(
                "OrderRequest.order_ref must be set on submit. "
                "Did you bypass OrderRequest.normalized()?"
            )

    def log_decision(self, strategy: str, intent: TradeIntent, *, accepted: bool, reason: str | None) -> None:
        if self._store is None or self._run_id is None:
            return
        self._store.log_decision(self._run_id, strategy=str(strategy), intent=intent, accepted=accepted, reason=reason)

    def log_action(self, actor: str, *, payload: dict[str, object], accepted: bool, reason: str | None) -> None:
        if self._store is None or self._run_id is None:
            return
        # For non-TradeIntent actions (LLM modify/cancel, etc).
        self._store.log_action(
            self._run_id,
            actor=str(actor),
            payload=dict(payload) if isinstance(payload, dict) else {"payload": str(payload)},
            accepted=accepted,
            reason=reason,
        )

    def modify(self, order_id: str, new_req: OrderRequest) -> OMSResult:
        # Modify routes new exposure to the venue; treat as "send" for halt purposes.
        assert_not_halted()
        new_req = new_req.normalized()
        self._authorize_send()
        if self._cfg.dry_run:
            self._log_error("oms.modify", "dry_run")
            return OMSResult(order_id=str(order_id), status="DryRun")
        res = self._broker.modify_order(str(order_id), new_req)
        self._log_order(new_req, res)
        self._log_status(str(order_id))
        return OMSResult(order_id=str(order_id), status=res.status)

    def cancel(self, order_id: str) -> None:
        # Cancel REDUCES exposure. We deliberately do NOT halt-gate cancels —
        # an operator who has halted often wants to flatten via cancel_all.
        # If you need to block all broker traffic, use a separate kill-switch.
        self._authorize_send()
        if self._cfg.dry_run:
            self._log_error("oms.cancel", "dry_run")
            return
        self._broker.cancel_order(str(order_id))
        self._log_status(str(order_id))

    def status(self, order_id: str) -> OrderStatus:
        st = self._broker.get_order_status(str(order_id))
        self._record_status(st)
        return st

    def _authorize_send(self) -> None:
        # Paper-only is enforced at connect-time in IBKRBroker; keep additional send gates here.
        if self._cfg.broker == "ibkr":
            if not self._cfg.live_enabled:
                raise RuntimeError("IBKR sending blocked: TRADING_LIVE_ENABLED=false")
            if self._cfg.confirm_token_required:
                if not self._cfg.order_token:
                    raise RuntimeError("IBKR sending blocked: TRADING_ORDER_TOKEN missing")
                if self._confirm_token != self._cfg.order_token:
                    raise RuntimeError("IBKR sending blocked: confirm token mismatch")

    def _log_order(self, req: OrderRequest, res: OrderResult) -> None:
        if self._store is None or self._run_id is None:
            return
        self._store.log_order(
            self._run_id,
            broker=self._cfg.broker,
            order_id=res.order_id,
            request=req,
            status=res.status,
        )

    def _log_status(self, order_id: str) -> None:
        if self._store is None or self._run_id is None:
            return
        try:
            st = self._broker.get_order_status(str(order_id))
        except Exception as exc:
            self._store.log_error(self._run_id, where="oms.status", message=str(exc))
            return
        self._record_status(st)

    def _record_status(self, st: OrderStatus) -> None:
        if self._store is None or self._run_id is None:
            return
        prev = self._last_status.get(str(st.order_id))
        if prev == str(st.status):
            return
        self._last_status[str(st.order_id)] = str(st.status)
        self._store.log_order_status_event(self._run_id, self._cfg.broker, st)
        # Keep "orders.status" reasonably up-to-date for recovery queries.
        try:
            self._store.update_order_status(str(st.order_id), str(st.status))
        except Exception:
            pass

    def _log_error(self, where: str, msg: str) -> None:
        if self._store is None or self._run_id is None:
            return
        self._store.log_error(self._run_id, where=where, message=json.dumps({"msg": msg, "ts": time.time()}))

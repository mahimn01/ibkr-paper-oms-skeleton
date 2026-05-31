from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from typing import Callable

from trading_algo.broker.base import AccountSnapshot, Broker, MarketDataSnapshot, Position
from trading_algo.instruments import InstrumentSpec, validate_instrument
from trading_algo.orders import TradeIntent

log = logging.getLogger(__name__)


class RiskViolation(ValueError):
    """Raised by RiskManager.validate when an intent violates a risk rule.

    Classified as VALIDATION (exit code 2) by the error classifier so
    agents can distinguish deterministic risk rejections from other
    ValueErrors.
    """


@dataclass(frozen=True)
class RiskLimits:
    max_order_quantity: float = 100.0
    max_abs_position_per_symbol: float = 500.0
    max_notional_exposure: float = 250_000.0
    max_leverage: float = 2.0
    max_margin_utilization: float = 0.5  # MaintMarginReq / NetLiquidation
    max_daily_loss: float = 1_000.0  # NetLiquidation drawdown from session start
    allow_short: bool = False
    # T3.5 additions:
    max_loss_per_trade: float | None = None
    """Max $ loss a single order may commit — (entry_price - stop_loss_price)
    * quantity * sign. Only enforced when stop_price is present on the
    intent; MKT/LMT orders without a stop bypass this check."""
    max_orders_per_day: int | None = None
    """Hard cap on order submissions per calendar day. Resets at local
    midnight. None → unlimited."""
    symbol_deny_list: frozenset[str] = frozenset()
    """Symbols the RiskManager must reject outright, e.g. halted tickers,
    OFAC-restricted issuers, or a per-strategy exclusion list."""


def _price_from_snapshot(snap: MarketDataSnapshot) -> float:
    candidates = [snap.last]
    if snap.bid is not None and snap.ask is not None and snap.bid > 0 and snap.ask > 0:
        candidates.append((snap.bid + snap.ask) / 2.0)
    candidates.append(snap.close)
    for c in candidates:
        if c is not None and c > 0:
            return float(c)
    raise ValueError(f"No usable price in market data snapshot for {snap.instrument}")


def _instrument_key(inst: InstrumentSpec) -> tuple[str, str]:
    inst = validate_instrument(inst)
    # For risk, key by kind+symbol; exchange differences are ignored at this layer.
    return (inst.kind, inst.symbol)


class RiskManager:
    """Pre-trade risk gate.

    Persistence (PLAN.md §2.6):
        Pass `db_path` (a SQLite file path) and the manager will
        load/save session_start_net_liq, orders_today, and
        orders_today_date across process restarts. Without it the
        counters reset on every start and the daily-loss circuit
        breaker never trips after a crash.

        The DB schema lives in trading_algo/data/schema.sql
        (table risk_state, single row id=1). The init_schema bool
        controls whether to create the table inline (handy for tests
        that don't go through the full PIT bootstrap).
    """

    def __init__(
        self,
        limits: RiskLimits,
        *,
        db_path: str | Path | None = None,
        init_schema: bool = True,
    ) -> None:
        self._limits = limits
        self._session_start_net_liq: float | None = None
        self._orders_today: int = 0
        self._orders_today_date: str | None = None
        self._lock = threading.Lock()
        self._db_path: str | None = str(db_path) if db_path else None
        if self._db_path:
            if init_schema:
                self._init_schema()
            self._load_state()

    def validate(self, intent: TradeIntent, broker: Broker, get_snapshot: Callable[[InstrumentSpec], MarketDataSnapshot]) -> None:
        intent_inst = validate_instrument(intent.instrument)

        if intent.quantity <= 0:
            raise RiskViolation("quantity must be positive")
        if intent.quantity > self._limits.max_order_quantity:
            raise RiskViolation(
                f"order quantity {intent.quantity} exceeds max_order_quantity "
                f"{self._limits.max_order_quantity}"
            )
        if intent_inst.symbol in self._limits.symbol_deny_list:
            raise RiskViolation(
                f"symbol {intent_inst.symbol!r} is in the risk deny-list"
            )

        positions = broker.get_positions()
        account = broker.get_account_snapshot()

        self._update_session_start(account)

        current_qty = _position_for(positions, intent_inst)
        delta = intent.quantity if intent.side.upper() == "BUY" else -intent.quantity
        resulting_qty = current_qty + delta

        if not self._limits.allow_short and resulting_qty < 0:
            raise RiskViolation("order would create a short position (allow_short=false)")

        if abs(resulting_qty) > self._limits.max_abs_position_per_symbol:
            raise RiskViolation(
                f"resulting position {resulting_qty} exceeds "
                f"max_abs_position_per_symbol {self._limits.max_abs_position_per_symbol}"
            )

        snap = get_snapshot(intent_inst)
        px = _price_from_snapshot(snap)
        trade_notional = abs(delta) * px

        gross_position_value = _gross_position_value(account, positions)
        if gross_position_value is not None:
            if gross_position_value + trade_notional > self._limits.max_notional_exposure:
                raise RiskViolation(
                    f"max_notional_exposure {self._limits.max_notional_exposure} "
                    f"would be exceeded ({gross_position_value} + {trade_notional})"
                )

        _check_leverage_and_margin(self._limits, account)
        _check_daily_loss(self._limits, self._session_start_net_liq, account)
        self._check_max_loss_per_trade(intent, px, delta)
        self._check_and_count_daily_orders()

    def _update_session_start(self, account: AccountSnapshot) -> None:
        if self._session_start_net_liq is not None:
            return
        net_liq = account.values.get("NetLiquidation")
        if net_liq is not None and net_liq > 0:
            self._session_start_net_liq = float(net_liq)
            self._save_state()

    # ------------------------------------------------------------------ persistence

    def _init_schema(self) -> None:
        """Create the risk_state table if it doesn't exist.

        Idempotent. The data layer's schema.sql also creates this table
        — having it here lets RiskManager work standalone in tests.
        """
        with self._open_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_state (
                    id                       INTEGER PRIMARY KEY CHECK (id = 1),
                    session_start_net_liq    REAL,
                    orders_today             INTEGER NOT NULL DEFAULT 0,
                    orders_today_date        TEXT,
                    last_updated             TEXT    NOT NULL
                )
                """
            )

    def _open_db(self) -> sqlite3.Connection:
        assert self._db_path is not None
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _load_state(self) -> None:
        """Load persisted state. Resets orders_today if the date has rolled over."""
        if self._db_path is None:
            return
        try:
            with self._open_db() as conn:
                row = conn.execute(
                    """
                    SELECT session_start_net_liq, orders_today, orders_today_date,
                           last_updated
                    FROM risk_state WHERE id = 1
                    """
                ).fetchone()
            if row is None:
                return
            ssnl, ot, otd, last_updated = row
            today = time.strftime("%Y-%m-%d", time.localtime())
            # last_updated is "YYYY-MM-DDTHH:MM:SS"; first 10 chars = date.
            last_date = (last_updated or "")[:10]

            # Daily-orders counter: only carry if otd matches today.
            if otd == today:
                self._orders_today = int(ot or 0)
                self._orders_today_date = otd
            else:
                self._orders_today = 0
                self._orders_today_date = None

            # Session NL: only carry if it was last updated today.
            if last_date == today and ssnl is not None:
                self._session_start_net_liq = float(ssnl)
        except sqlite3.DatabaseError as exc:
            log.warning("RiskManager._load_state failed: %s", exc)

    def _save_state(self) -> None:
        """Persist current state to SQLite. Called inside the threading lock."""
        if self._db_path is None:
            return
        try:
            with self._open_db() as conn:
                conn.execute(
                    """
                    INSERT INTO risk_state
                      (id, session_start_net_liq, orders_today, orders_today_date, last_updated)
                    VALUES (1, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      session_start_net_liq = excluded.session_start_net_liq,
                      orders_today          = excluded.orders_today,
                      orders_today_date     = excluded.orders_today_date,
                      last_updated          = excluded.last_updated
                    """,
                    (
                        self._session_start_net_liq,
                        self._orders_today,
                        self._orders_today_date,
                        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    ),
                )
        except sqlite3.DatabaseError as exc:
            log.warning("RiskManager._save_state failed: %s", exc)

    def _check_max_loss_per_trade(
        self, intent: TradeIntent, entry_price: float, delta: float
    ) -> None:
        """Enforce max_loss_per_trade when the intent carries a stop.

        Worst-case loss =
          BUY: (entry_price - stop_price) * qty    # stop is below entry
          SELL: (stop_price - entry_price) * qty   # stop is above entry
        We require a positive loss figure — a "stop" that wouldn't reduce
        loss (wrong side) is itself a RiskViolation.
        """
        if self._limits.max_loss_per_trade is None:
            return
        stop = getattr(intent, "stop_price", None)
        if stop is None:
            return
        qty = abs(delta)
        side = intent.side.upper()
        if side == "BUY":
            loss = (entry_price - float(stop)) * qty
        else:
            loss = (float(stop) - entry_price) * qty
        if loss <= 0:
            raise RiskViolation(
                f"stop_price {stop} is on the wrong side of entry {entry_price} for "
                f"side {side} — a working stop must reduce loss"
            )
        if loss > self._limits.max_loss_per_trade:
            raise RiskViolation(
                f"worst-case loss {loss:.2f} exceeds max_loss_per_trade "
                f"{self._limits.max_loss_per_trade:.2f}"
            )

    def _check_and_count_daily_orders(self) -> None:
        """Increment the daily-order counter (and enforce cap) atomically.

        Counter resets at local midnight — we compare the stored ISO date
        against today's. This runs AFTER all other validations so rejected
        orders don't consume a daily slot.
        """
        if self._limits.max_orders_per_day is None:
            return
        with self._lock:
            today = time.strftime("%Y-%m-%d", time.localtime())
            if self._orders_today_date != today:
                self._orders_today_date = today
                self._orders_today = 0
            if self._orders_today >= int(self._limits.max_orders_per_day):
                raise RiskViolation(
                    f"max_orders_per_day {self._limits.max_orders_per_day} reached "
                    f"for {today}"
                )
            self._orders_today += 1
            self._save_state()

    @property
    def orders_today_count(self) -> int:
        """Test helper — current day's accepted-order count."""
        return self._orders_today


def _position_for(positions: list[Position], instrument: InstrumentSpec) -> float:
    key = _instrument_key(instrument)
    total = 0.0
    for pos in positions:
        try:
            if _instrument_key(pos.instrument) == key:
                total += float(pos.quantity)
        except Exception:
            continue
    return total


def _gross_position_value(account: AccountSnapshot, _positions: list[Position]) -> float | None:
    gpv = account.values.get("GrossPositionValue")
    if gpv is None:
        return None
    try:
        return float(gpv)
    except Exception:
        return None


def _check_leverage_and_margin(limits: RiskLimits, account: AccountSnapshot) -> None:
    net_liq = account.values.get("NetLiquidation")
    gross = account.values.get("GrossPositionValue")
    maint = account.values.get("MaintMarginReq")

    if net_liq is not None and gross is not None:
        if net_liq <= 0:
            raise RiskViolation("Invalid NetLiquidation")
        leverage = float(gross) / float(net_liq)
        if leverage > limits.max_leverage:
            raise RiskViolation(
                f"leverage {leverage:.2f} exceeds max_leverage {limits.max_leverage:.2f}"
            )

    if net_liq is not None and maint is not None:
        if net_liq <= 0:
            raise RiskViolation("Invalid NetLiquidation")
        util = float(maint) / float(net_liq)
        if util > limits.max_margin_utilization:
            raise RiskViolation(
                f"margin utilisation {util:.3f} exceeds max_margin_utilization "
                f"{limits.max_margin_utilization:.3f}"
            )


def _check_daily_loss(limits: RiskLimits, start_net_liq: float | None, account: AccountSnapshot) -> None:
    if start_net_liq is None:
        return
    now = account.values.get("NetLiquidation")
    if now is None:
        return
    drawdown = float(start_net_liq) - float(now)
    if drawdown > limits.max_daily_loss:
        raise RiskViolation(
            f"drawdown {drawdown:.2f} exceeds max_daily_loss "
            f"{limits.max_daily_loss:.2f} (circuit breaker)"
        )

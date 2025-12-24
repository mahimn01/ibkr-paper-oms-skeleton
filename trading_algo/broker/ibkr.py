from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from trading_algo.broker.base import (
    AccountSnapshot,
    Bar,
    BracketOrderRequest,
    BracketOrderResult,
    MarketDataSnapshot,
    OrderRequest,
    OrderResult,
    Position,
    validate_order_request,
)
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec, validate_instrument

log = logging.getLogger(__name__)


class IBKRDependencyError(RuntimeError):
    pass


@dataclass
class _Factories:
    IB: Any
    Stock: Any
    Future: Any
    Forex: Any
    MarketOrder: Any
    LimitOrder: Any
    StopOrder: Any
    StopLimitOrder: Any


def _load_ib_insync_factories() -> _Factories:
    # Python 3.12+ tightened asyncio event loop behavior; some deps (eventkit) expect
    # a current loop to exist during import. Ensure one exists for the main thread.
    import asyncio
    import warnings

    # Suppress third-party deprecations on newer Python versions.
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*get_event_loop_policy.*")

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    try:
        from ib_insync import IB, Forex, Future, LimitOrder, MarketOrder, Stock, StopLimitOrder, StopOrder  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise IBKRDependencyError(
            "Failed to import 'ib_insync' (check your environment and installed dependencies)."
        ) from exc
    return _Factories(
        IB=IB,
        Stock=Stock,
        Future=Future,
        Forex=Forex,
        MarketOrder=MarketOrder,
        LimitOrder=LimitOrder,
        StopOrder=StopOrder,
        StopLimitOrder=StopLimitOrder,
    )


@dataclass
class IBKRBroker:
    """
    IBKR broker adapter for TWS / IB Gateway (paper or live).

    For precision and safety:
    - Contracts are qualified before use.
    - Orders are validated before submission.
    """

    config: IBKRConfig
    require_paper: bool = True
    ib_factory: Callable[[], Any] | None = None
    _factories: _Factories | None = field(default=None, init=False, repr=False)
    _ib: Any | None = field(default=None, init=False, repr=False)
    _trades: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _contracts: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _ensure_factories(self) -> _Factories:
        if self._factories is None:
            self._factories = _load_ib_insync_factories()
        return self._factories

    def connect(self) -> None:
        factories = self._ensure_factories()
        self._ib = (self.ib_factory or factories.IB)()
        log.info("Connecting to IBKR %s:%s clientId=%s", self.config.host, self.config.port, self.config.client_id)
        # Preflight only for real socket connections; unit tests inject `ib_factory`.
        if self.ib_factory is None:
            _preflight_check_socket(self.config.host, self.config.port)
        try:
            self._ib.connect(self.config.host, self.config.port, clientId=self.config.client_id)
        except Exception as exc:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
            raise RuntimeError(
                "Failed to connect to IBKR TWS/IB Gateway. Ensure TWS/IBG is running, you are logged in "
                "to Paper Trading, API access is enabled, and IBKR_PORT matches the configured API port."
            ) from exc
        else:
            log.info("Connected")
            if self.require_paper:
                self._assert_paper_trading()

    def disconnect(self) -> None:
        if self._ib is None:
            return
        self._ib.disconnect()
        self._ib = None
        self._trades.clear()
        self._contracts.clear()
        log.info("Disconnected")

    def _to_contract(self, instrument: InstrumentSpec) -> Any:
        factories = self._ensure_factories()
        spec = validate_instrument(instrument)

        if spec.kind == "STK":
            return factories.Stock(spec.symbol, spec.exchange, spec.currency)
        if spec.kind == "FUT":
            return factories.Future(spec.symbol, spec.expiry, spec.exchange, currency=spec.currency)
        if spec.kind == "FX":
            # ib_insync parses 'EURUSD' into base/quote automatically.
            return factories.Forex(spec.symbol)

        raise ValueError(f"Unsupported instrument kind: {spec.kind}")

    def _qualify(self, contract: Any) -> Any:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError("Failed to qualify contract with IBKR")
        return qualified[0]

    def place_order(self, req: OrderRequest) -> OrderResult:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")

        factories = self._ensure_factories()
        req = validate_order_request(req)

        contract = self._qualify(self._to_contract(req.instrument))
        order = self._build_order(req, order_id=None)
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.25)
        status = getattr(trade.orderStatus, "status", "Submitted")
        order_id = str(getattr(trade.order, "orderId", "unknown"))
        self._trades[order_id] = trade
        self._contracts[order_id] = contract
        log.info(
            "Order placed kind=%s symbol=%s side=%s qty=%s type=%s tif=%s status=%s orderId=%s",
            req.instrument.kind,
            req.instrument.symbol,
            req.side,
            req.quantity,
            req.order_type,
            req.tif,
            status,
            order_id,
        )
        return OrderResult(order_id=order_id, status=status)

    def modify_order(self, order_id: str, new_req: OrderRequest) -> OrderResult:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        new_req = validate_order_request(new_req)
        contract = self._contracts.get(str(order_id))
        if contract is None:
            trade = self._trades.get(str(order_id)) or _find_trade(self._ib, str(order_id))
            contract = getattr(trade, "contract", None) if trade is not None else None
        if contract is None:
            raise KeyError(f"Unknown order_id (no contract cached): {order_id}")

        # In IBKR API, modifying is done by re-sending placeOrder with the same orderId.
        try:
            oid_int = int(str(order_id))
        except Exception as exc:
            raise ValueError(f"order_id must be numeric for IBKR modification: {order_id}") from exc

        order = self._build_order(new_req, order_id=oid_int)
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.25)
        status = getattr(trade.orderStatus, "status", "Submitted")
        self._trades[str(order_id)] = trade
        self._contracts[str(order_id)] = contract
        return OrderResult(order_id=str(order_id), status=str(status))

    def cancel_order(self, order_id: str) -> None:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        trade = self._trades.get(str(order_id)) or _find_trade(self._ib, str(order_id))
        if trade is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        self._ib.cancelOrder(trade.order)
        self._ib.sleep(0.1)

    def get_order_status(self, order_id: str):
        from trading_algo.broker.base import OrderStatus

        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        trade = self._trades.get(str(order_id)) or _find_trade(self._ib, str(order_id))
        if trade is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        os = getattr(trade, "orderStatus", None)
        status = str(getattr(os, "status", "Unknown"))
        filled = getattr(os, "filled", None)
        remaining = getattr(os, "remaining", None)
        avg = getattr(os, "avgFillPrice", None)
        return OrderStatus(
            order_id=str(order_id),
            status=status,
            filled=float(filled) if filled is not None else None,
            remaining=float(remaining) if remaining is not None else None,
            avg_fill_price=float(avg) if avg is not None else None,
        )

    def list_open_order_statuses(self) -> list["OrderStatus"]:
        from trading_algo.broker.base import OrderStatus

        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        out: list[OrderStatus] = []
        try:
            trades = list(self._ib.openTrades())
        except Exception:
            trades = []
        for t in trades:
            oid = str(getattr(getattr(t, "order", None), "orderId", ""))
            if not oid:
                continue
            self._trades[oid] = t
            try:
                self._contracts[oid] = getattr(t, "contract", None)
            except Exception:
                pass
            os = getattr(t, "orderStatus", None)
            status = str(getattr(os, "status", "Unknown"))
            filled = getattr(os, "filled", None)
            remaining = getattr(os, "remaining", None)
            avg = getattr(os, "avgFillPrice", None)
            out.append(
                OrderStatus(
                    order_id=oid,
                    status=status,
                    filled=float(filled) if filled is not None else None,
                    remaining=float(remaining) if remaining is not None else None,
                    avg_fill_price=float(avg) if avg is not None else None,
                )
            )
        return out

    def place_bracket_order(self, req: BracketOrderRequest) -> BracketOrderResult:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        req_inst = validate_instrument(req.instrument)
        side = req.side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if req.quantity <= 0:
            raise ValueError("quantity must be positive")
        if req.entry_limit_price <= 0 or req.take_profit_limit_price <= 0 or req.stop_loss_stop_price <= 0:
            raise ValueError("Bracket prices must be positive")

        contract = self._qualify(self._to_contract(req_inst))
        orders = self._ib.bracketOrder(
            side,
            float(req.quantity),
            float(req.entry_limit_price),
            float(req.take_profit_limit_price),
            float(req.stop_loss_stop_price),
        )
        for o in orders:
            o.tif = req.tif

        trades = [self._ib.placeOrder(contract, o) for o in orders]
        self._ib.sleep(0.25)
        ids = [str(getattr(t.order, "orderId", "unknown")) for t in trades]
        for oid, t in zip(ids, trades, strict=False):
            self._trades[oid] = t
            self._contracts[oid] = contract

        if len(ids) != 3:
            raise RuntimeError(f"Unexpected bracket order result: {ids}")
        return BracketOrderResult(parent_order_id=ids[0], take_profit_order_id=ids[1], stop_loss_order_id=ids[2])

    def _build_order(self, req: OrderRequest, order_id: int | None) -> Any:
        factories = self._ensure_factories()
        if req.order_type == "MKT":
            order = factories.MarketOrder(req.side, req.quantity, tif=req.tif)
        elif req.order_type == "LMT":
            order = factories.LimitOrder(req.side, req.quantity, req.limit_price, tif=req.tif)
        elif req.order_type == "STP":
            order = factories.StopOrder(req.side, req.quantity, req.stop_price, tif=req.tif)
        elif req.order_type == "STPLMT":
            order = factories.StopLimitOrder(req.side, req.quantity, req.limit_price, req.stop_price, tif=req.tif)
        else:
            raise ValueError(f"Unsupported order_type: {req.order_type}")

        if order_id is not None:
            order.orderId = int(order_id)
        order.outsideRth = bool(req.outside_rth)
        order.transmit = bool(req.transmit)
        if req.good_till_date:
            order.goodTillDate = str(req.good_till_date)
        if req.account:
            order.account = str(req.account)
        if req.order_ref:
            order.orderRef = str(req.order_ref)
        if req.oca_group:
            order.ocaGroup = str(req.oca_group)
        return order

    def get_market_data_snapshot(self, instrument: InstrumentSpec) -> MarketDataSnapshot:
        """
        Snapshot market data (best-effort).

        Notes:
        - IBKR market data may be delayed or unavailable without subscriptions.
        - Fields may be None if not available.
        """
        if self._ib is None:
            raise RuntimeError("Broker is not connected")

        instrument = validate_instrument(instrument)
        contract = self._qualify(self._to_contract(instrument))
        # Prefer delayed snapshot data to avoid requiring real-time market data subscriptions.
        # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
        try:
            self._ib.reqMarketDataType(3)
        except Exception:
            pass

        ticker = self._ib.reqMktData(contract, "", True, False)
        self._ib.sleep(0.8)

        def _f(value: Any) -> float | None:
            if value is None:
                return None
            try:
                if value != value:  # NaN
                    return None
            except Exception:
                pass
            try:
                return float(value)
            except Exception:
                return None

        snap = MarketDataSnapshot(
            instrument=instrument,
            bid=_f(getattr(ticker, "bid", None)),
            ask=_f(getattr(ticker, "ask", None)),
            last=_f(getattr(ticker, "last", None)),
            close=_f(getattr(ticker, "close", None)),
            volume=_f(getattr(ticker, "volume", None)),
            timestamp_epoch_s=time.time(),
        )
        # Snapshot requests may still fail without subscriptions; fall back to last historical close.
        if (snap.bid is None) and (snap.ask is None) and (snap.last is None) and (snap.close is None):
            try:
                bars = self.get_historical_bars(
                    instrument,
                    duration="1 D",
                    bar_size="1 day",
                    what_to_show="TRADES",
                    use_rth=False,
                )
                if bars:
                    last_close = bars[-1].close
                    snap = MarketDataSnapshot(
                        instrument=instrument,
                        bid=None,
                        ask=None,
                        last=float(last_close),
                        close=float(last_close),
                        volume=bars[-1].volume,
                        timestamp_epoch_s=time.time(),
                    )
            except Exception:
                pass
        return snap

    def get_historical_bars(
        self,
        instrument: InstrumentSpec,
        *,
        end_datetime: str | None = None,
        duration: str,
        bar_size: str,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list[Bar]:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        instrument = validate_instrument(instrument)
        contract = self._qualify(self._to_contract(instrument))
        end_dt = _parse_ibkr_end_datetime(end_datetime)
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=str(duration),
            barSizeSetting=str(bar_size),
            whatToShow=str(what_to_show),
            useRTH=1 if use_rth else 0,
            # Use epoch timestamps for robust parsing.
            formatDate=2,
        )
        out: list[Bar] = []
        for b in list(bars or []):
            ts = getattr(b, "date", None)
            ts_epoch = time.time()
            try:
                if hasattr(ts, "timestamp"):
                    ts_epoch = float(ts.timestamp())
                else:
                    ts_epoch = float(ts)
            except Exception:
                pass
            out.append(
                Bar(
                    timestamp_epoch_s=ts_epoch,
                    open=float(getattr(b, "open", 0.0)),
                    high=float(getattr(b, "high", 0.0)),
                    low=float(getattr(b, "low", 0.0)),
                    close=float(getattr(b, "close", 0.0)),
                    volume=float(getattr(b, "volume", 0.0)) if getattr(b, "volume", None) is not None else None,
                )
            )
        return out

    def get_positions(self) -> list[Position]:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")
        positions = []
        for pos in list(self._ib.positions()):
            try:
                instrument = _contract_to_instrument(pos.contract)
            except Exception:
                continue
            avg_cost = getattr(pos, "avgCost", None)
            try:
                avg_cost_f = float(avg_cost) if avg_cost is not None else None
            except Exception:
                avg_cost_f = None
            positions.append(
                Position(
                    account=str(getattr(pos, "account", "")),
                    instrument=instrument,
                    quantity=float(getattr(pos, "position", 0.0)),
                    avg_cost=avg_cost_f,
                    timestamp_epoch_s=time.time(),
                )
            )
        return positions

    def get_account_snapshot(self) -> AccountSnapshot:
        if self._ib is None:
            raise RuntimeError("Broker is not connected")

        accounts = list(self._ib.managedAccounts() or [])
        account = accounts[0] if accounts else ""
        values: dict[str, float] = {}

        try:
            summary_items = list(self._ib.accountSummary(account))
        except Exception:
            summary_items = []

        for item in summary_items:
            tag = str(getattr(item, "tag", "")).strip()
            value_raw = getattr(item, "value", None)
            try:
                value = float(value_raw)
            except Exception:
                continue
            if tag:
                values[tag] = value

        return AccountSnapshot(account=str(account), values=values, timestamp_epoch_s=time.time())

    def _assert_paper_trading(self) -> None:
        """
        Guard rail: refuse to operate unless the connected session appears to be Paper Trading.

        IB paper accounts are typically prefixed with "DU". If any managed account does not match,
        we treat it as unsafe and abort.
        """
        if self._ib is None:
            raise RuntimeError("Broker is not connected")

        accounts: list[str] | None = None
        for _ in range(10):
            try:
                accounts = list(self._ib.managedAccounts())
            except Exception:
                accounts = None
            if accounts:
                break
            self._ib.sleep(0.2)

        if not accounts:
            raise RuntimeError(
                "Connected to IBKR, but could not read managed accounts to verify paper trading. "
                "This is unsafe; refusing to continue."
            )

        non_paper = [a for a in accounts if not str(a).startswith("DU")]
        if non_paper:
            raise RuntimeError(
                "Refusing to run because this does not look like Paper Trading. "
                f"Managed accounts: {accounts}. "
                "Paper accounts usually start with 'DU'. "
                "Fix by logging into Paper Trading and using the paper API port (TWS paper typically 7497)."
            )


def _find_trade(ib: Any, order_id: str) -> Any | None:
    try:
        for t in list(ib.trades()):
            oid = str(getattr(getattr(t, "order", None), "orderId", ""))
            if oid == str(order_id):
                return t
    except Exception:
        return None
    return None


def _parse_ibkr_end_datetime(value: str | None):
    """
    ib_insync accepts endDateTime as '' (now), a datetime, or an IB-formatted string.
    We also accept epoch seconds or ISO-8601 strings for convenience.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    # Epoch seconds
    try:
        epoch = float(s)
    except Exception:
        epoch = None
    if epoch is not None:
        import datetime as dt

        return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
    # ISO-8601
    try:
        import datetime as dt

        iso = s
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        return dt.datetime.fromisoformat(iso)
    except Exception:
        return s


def _preflight_check_socket(host: str, port: int) -> None:
    """
    Fast, explicit TCP check so "connection refused" becomes a clearer action item.
    """
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            return
    except ConnectionRefusedError as exc:
        raise RuntimeError(
            f"IBKR API port is not accepting connections at {host}:{port} (connection refused). "
            "Start TWS/IB Gateway, enable API access, and confirm the configured API port. "
            "Common ports: TWS paper=7497 live=7496, Gateway paper=4002 live=4001. "
            "You can override with CLI flags: --ibkr-port / --ibkr-host."
        ) from exc
    except socket.timeout as exc:
        raise RuntimeError(
            f"IBKR API port check timed out connecting to {host}:{port}. "
            "Verify host/port, firewall settings, and that TWS/IBG is listening. "
            "You can override with CLI flags: --ibkr-port / --ibkr-host."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"IBKR API port check failed for {host}:{port}: {exc}. "
            "Verify host/port and that TWS/IBG is running with API enabled."
        ) from exc


def _contract_to_instrument(contract: Any) -> InstrumentSpec:
    sec_type = str(getattr(contract, "secType", "")).upper()
    symbol = str(getattr(contract, "symbol", "")).upper()
    exchange = (str(getattr(contract, "exchange", "")).upper() or None)
    currency = (str(getattr(contract, "currency", "")).upper() or None)

    if sec_type == "STK":
        return validate_instrument(InstrumentSpec(kind="STK", symbol=symbol, exchange=exchange or "SMART", currency=currency or "USD"))
    if sec_type == "FUT":
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "")).strip()
        return validate_instrument(InstrumentSpec(kind="FUT", symbol=symbol, exchange=exchange or "", currency=currency or "USD", expiry=expiry))
    if sec_type == "CASH":
        # FX uses base in `symbol` and quote in `currency`.
        pair = f"{symbol}{currency or ''}".upper()
        return validate_instrument(InstrumentSpec(kind="FX", symbol=pair, exchange=exchange or "IDEALPRO"))

    raise ValueError(f"Unsupported contract secType: {sec_type}")

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trading_algo.broker.base import (
    AccountSnapshot,
    Bar,
    BracketOrderRequest,
    BracketOrderResult,
    Broker,
    MarketDataSnapshot,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    validate_order_request,
)
from trading_algo.instruments import InstrumentSpec, validate_instrument


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class FillModel:
    """
    Deterministic fill model:
    - MKT fills at next bar open
    - LMT fills if next bar trades through limit (BUY: low<=limit, SELL: high>=limit)
    - STP triggers if next bar trades through stop (BUY: high>=stop, SELL: low<=stop), fills at stop price
    - STPLMT triggers like STP then must satisfy limit like LMT within same bar
    """

    commission_per_order: float = 0.0
    slippage_bps: float = 0.0  # applied to fill price

    def apply_slippage(self, price: float, side: str) -> float:
        bps = float(self.slippage_bps)
        if bps <= 0:
            return price
        sign = 1.0 if side.upper() == "BUY" else -1.0
        return price * (1.0 + sign * (bps / 10_000.0))


@dataclass
class BacktestBroker(Broker):
    instrument: InstrumentSpec
    bars: list[Bar]
    initial_cash: float = 100_000.0
    fill_model: FillModel = field(default_factory=FillModel)
    spread: float = 0.0

    connected: bool = False
    _i: int = 0
    _cash: float = 0.0
    _qty: float = 0.0
    _avg_cost: float | None = None
    _orders: dict[str, OrderRequest] = field(default_factory=dict, repr=False)
    _statuses: dict[str, OrderStatus] = field(default_factory=dict, repr=False)
    _open: set[str] = field(default_factory=set, repr=False)

    def connect(self) -> None:
        self.connected = True
        self.instrument = validate_instrument(self.instrument)
        self._cash = float(self.initial_cash)
        self._qty = 0.0
        self._avg_cost = None
        self._i = 0

    def disconnect(self) -> None:
        self.connected = False

    def step(self) -> bool:
        """
        Advance the simulation by one bar.
        Returns False when no more bars exist.
        """
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        if self._i >= len(self.bars):
            return False
        # Evaluate open orders against *this* bar.
        bar = self.bars[self._i]
        self._fill_open_orders(bar)
        self._i += 1
        return self._i < len(self.bars)

    def current_bar(self) -> Bar:
        if self._i == 0:
            return self.bars[0]
        return self.bars[min(self._i - 1, len(self.bars) - 1)]

    def get_market_data_snapshot(self, instrument: InstrumentSpec) -> MarketDataSnapshot:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        instrument = validate_instrument(instrument)
        if instrument != self.instrument:
            raise KeyError(f"BacktestBroker only supports {self.instrument}")
        bar = self.current_bar()
        mid = float(bar.close)
        bid = mid - self.spread / 2.0 if self.spread else None
        ask = mid + self.spread / 2.0 if self.spread else None
        return MarketDataSnapshot(
            instrument=instrument,
            bid=bid,
            ask=ask,
            last=float(bar.close),
            close=float(bar.close),
            volume=bar.volume,
            timestamp_epoch_s=float(bar.timestamp_epoch_s),
        )

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
        instrument = validate_instrument(instrument)
        if instrument != self.instrument:
            raise KeyError(f"BacktestBroker only supports {self.instrument}")
        return list(self.bars[: self._i])

    def get_positions(self) -> list[Position]:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return [
            Position(
                account="BACKTEST",
                instrument=self.instrument,
                quantity=float(self._qty),
                avg_cost=self._avg_cost,
                timestamp_epoch_s=float(self.current_bar().timestamp_epoch_s),
            )
        ]

    def get_account_snapshot(self) -> AccountSnapshot:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        bar = self.current_bar()
        last = float(bar.close)
        gpv = abs(self._qty) * last
        net_liq = float(self._cash) + (self._qty * last)
        return AccountSnapshot(
            account="BACKTEST",
            values={
                "NetLiquidation": net_liq,
                "GrossPositionValue": gpv,
                "AvailableFunds": net_liq,
                "MaintMarginReq": 0.0,
            },
            timestamp_epoch_s=float(bar.timestamp_epoch_s),
        )

    def place_order(self, req: OrderRequest) -> OrderResult:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        req = validate_order_request(req)
        if validate_instrument(req.instrument) != self.instrument:
            raise ValueError("BacktestBroker only supports one instrument per run")

        order_id = f"bt-{uuid.uuid4()}"
        self._orders[order_id] = req
        self._open.add(order_id)
        st = OrderStatus(order_id=order_id, status="Submitted", filled=0.0, remaining=req.quantity, avg_fill_price=None)
        self._statuses[order_id] = st
        return OrderResult(order_id=order_id, status=st.status)

    def modify_order(self, order_id: str, new_req: OrderRequest) -> OrderResult:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        if order_id not in self._orders:
            raise KeyError(f"Unknown order_id: {order_id}")
        new_req = validate_order_request(new_req)
        self._orders[order_id] = new_req
        if order_id in self._open:
            self._statuses[order_id] = OrderStatus(order_id, "Submitted", 0.0, new_req.quantity, None)
        return OrderResult(order_id=order_id, status=self._statuses[order_id].status)

    def cancel_order(self, order_id: str) -> None:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        if order_id not in self._orders:
            raise KeyError(f"Unknown order_id: {order_id}")
        self._open.discard(order_id)
        st = self._statuses.get(order_id)
        self._statuses[order_id] = OrderStatus(order_id, "Cancelled", st.filled if st else 0.0, st.remaining if st else 0.0, st.avg_fill_price if st else None)

    def get_order_status(self, order_id: str) -> OrderStatus:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        if order_id not in self._statuses:
            raise KeyError(f"Unknown order_id: {order_id}")
        return self._statuses[order_id]

    def list_open_order_statuses(self) -> list[OrderStatus]:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return [self._statuses[oid] for oid in list(self._open)]

    def place_bracket_order(self, req: BracketOrderRequest) -> BracketOrderResult:
        # Minimal: create three linked orders; advanced parent/child semantics are out of scope for backtests.
        parent = self.place_order(
            OrderRequest(
                instrument=req.instrument,
                side=req.side,
                quantity=req.quantity,
                order_type="LMT",
                limit_price=req.entry_limit_price,
                tif=req.tif,
            )
        ).order_id
        tp = self.place_order(
            OrderRequest(
                instrument=req.instrument,
                side="SELL" if req.side.upper() == "BUY" else "BUY",
                quantity=req.quantity,
                order_type="LMT",
                limit_price=req.take_profit_limit_price,
                tif=req.tif,
            )
        ).order_id
        sl = self.place_order(
            OrderRequest(
                instrument=req.instrument,
                side="SELL" if req.side.upper() == "BUY" else "BUY",
                quantity=req.quantity,
                order_type="STP",
                stop_price=req.stop_loss_stop_price,
                tif=req.tif,
            )
        ).order_id
        return BracketOrderResult(parent_order_id=parent, take_profit_order_id=tp, stop_loss_order_id=sl)

    def _fill_open_orders(self, bar: Bar) -> None:
        for oid in list(self._open):
            req = self._orders[oid]
            fill = self._try_fill(req, bar)
            if fill is None:
                continue
            fill_price = fill
            fill_price = self.fill_model.apply_slippage(fill_price, req.side)
            commission = float(self.fill_model.commission_per_order)
            self._apply_fill(req.side, req.quantity, fill_price, commission)
            self._open.discard(oid)
            self._statuses[oid] = OrderStatus(
                order_id=oid,
                status="Filled",
                filled=req.quantity,
                remaining=0.0,
                avg_fill_price=fill_price,
            )

    @staticmethod
    def _try_fill(req: OrderRequest, bar: Bar) -> float | None:
        side = req.side.upper()
        if req.order_type == "MKT":
            return float(bar.open)
        if req.order_type == "LMT":
            lp = float(req.limit_price)
            if side == "BUY" and bar.low <= lp:
                return lp
            if side == "SELL" and bar.high >= lp:
                return lp
            return None
        if req.order_type == "STP":
            sp = float(req.stop_price)
            if side == "BUY" and bar.high >= sp:
                return sp
            if side == "SELL" and bar.low <= sp:
                return sp
            return None
        if req.order_type == "STPLMT":
            sp = float(req.stop_price)
            lp = float(req.limit_price)
            triggered = (bar.high >= sp) if side == "BUY" else (bar.low <= sp)
            if not triggered:
                return None
            if side == "BUY" and bar.low <= lp:
                return lp
            if side == "SELL" and bar.high >= lp:
                return lp
            return None
        return None

    def _apply_fill(self, side: str, qty: float, price: float, commission: float) -> None:
        sign = 1.0 if side.upper() == "BUY" else -1.0
        delta_qty = sign * float(qty)
        cost = float(price) * float(qty) * sign
        self._cash -= cost
        self._cash -= commission

        new_qty = self._qty + delta_qty
        if self._qty == 0:
            self._avg_cost = float(price) if new_qty != 0 else None
        elif (self._qty > 0 and delta_qty > 0) or (self._qty < 0 and delta_qty < 0):
            # increasing same-side position: weighted avg
            prev_abs = abs(self._qty)
            new_abs = abs(new_qty)
            if new_abs > 0:
                self._avg_cost = ((self._avg_cost or 0.0) * prev_abs + float(price) * abs(delta_qty)) / new_abs
        else:
            # reducing/closing; keep avg_cost if still open, else clear
            if new_qty == 0:
                self._avg_cost = None
        self._qty = new_qty

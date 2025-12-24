import logging
import unittest

from trading_algo.backtest.broker import BacktestBroker, FillModel
from trading_algo.broker.base import Bar, OrderRequest
from trading_algo.instruments import InstrumentSpec

logging.disable(logging.CRITICAL)


class TestBacktestBroker(unittest.TestCase):
    def _bars(self):
        return [
            Bar(timestamp_epoch_s=1, open=100, high=105, low=95, close=102, volume=1000),
            Bar(timestamp_epoch_s=2, open=102, high=106, low=101, close=104, volume=1000),
            Bar(timestamp_epoch_s=3, open=104, high=110, low=103, close=109, volume=1000),
        ]

    def test_market_order_fills_next_bar_open(self):
        inst = InstrumentSpec(kind="STK", symbol="AAPL")
        b = BacktestBroker(inst, self._bars(), initial_cash=1000, fill_model=FillModel(), spread=0)
        b.connect()
        try:
            res = b.place_order(OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="MKT"))
            st0 = b.get_order_status(res.order_id)
            self.assertEqual(st0.status, "Submitted")
            b.step()  # evaluate on bar[0]
            st1 = b.get_order_status(res.order_id)
            # MKT fills at bar[0].open since we evaluate at current bar in step
            self.assertEqual(st1.status, "Filled")
            self.assertEqual(st1.avg_fill_price, 100.0)
        finally:
            b.disconnect()

    def test_limit_order_fill(self):
        inst = InstrumentSpec(kind="STK", symbol="AAPL")
        b = BacktestBroker(inst, self._bars(), initial_cash=1000, fill_model=FillModel(), spread=0)
        b.connect()
        try:
            res = b.place_order(OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="LMT", limit_price=99))
            b.step()
            st = b.get_order_status(res.order_id)
            self.assertEqual(st.status, "Filled")
            self.assertEqual(st.avg_fill_price, 99.0)
        finally:
            b.disconnect()

    def test_modify_and_cancel(self):
        inst = InstrumentSpec(kind="STK", symbol="AAPL")
        b = BacktestBroker(inst, self._bars(), initial_cash=1000, fill_model=FillModel(), spread=0)
        b.connect()
        try:
            res = b.place_order(OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="LMT", limit_price=1))
            # won't fill
            b.modify_order(res.order_id, OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="LMT", limit_price=99))
            b.step()
            self.assertEqual(b.get_order_status(res.order_id).status, "Filled")

            res2 = b.place_order(OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="LMT", limit_price=1))
            b.cancel_order(res2.order_id)
            self.assertEqual(b.get_order_status(res2.order_id).status, "Cancelled")
        finally:
            b.disconnect()


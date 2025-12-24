import logging
import unittest

from trading_algo.broker.sim import SimBroker
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent

logging.disable(logging.CRITICAL)


class TestSimBroker(unittest.TestCase):
    def test_order_fill(self):
        broker = SimBroker()
        broker.connect()
        try:
            intent = TradeIntent(
                instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                side="BUY",
                quantity=2,
            )
            result = broker.place_order(intent.to_order_request())
            self.assertTrue(result.order_id.startswith("sim-"))
            self.assertEqual(result.status, "Filled")
            self.assertEqual(len(broker.orders), 1)
            self.assertEqual(broker.orders[0].instrument.symbol, "AAPL")
        finally:
            broker.disconnect()

    def test_snapshot_requires_data(self):
        broker = SimBroker()
        broker.connect()
        try:
            with self.assertRaises(KeyError):
                broker.get_market_data_snapshot(InstrumentSpec(kind="STK", symbol="AAPL"))
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), bid=99, ask=101, last=100)
            snap = broker.get_market_data_snapshot(InstrumentSpec(kind="STK", symbol="AAPL"))
            self.assertEqual(snap.bid, 99)
            self.assertEqual(snap.ask, 101)
            self.assertEqual(snap.last, 100)
        finally:
            broker.disconnect()

    def test_order_status_and_cancel(self):
        broker = SimBroker()
        broker.connect()
        try:
            intent = TradeIntent(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1)
            res = broker.place_order(intent.to_order_request())
            st = broker.get_order_status(res.order_id)
            self.assertEqual(st.status, "Filled")
            broker.cancel_order(res.order_id)  # no-op for filled
            st2 = broker.get_order_status(res.order_id)
            self.assertEqual(st2.status, "Filled")
        finally:
            broker.disconnect()

    def test_bracket_returns_three_ids(self):
        from trading_algo.broker.base import BracketOrderRequest

        broker = SimBroker()
        broker.connect()
        try:
            req = BracketOrderRequest(
                instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                side="BUY",
                quantity=1,
                entry_limit_price=100,
                take_profit_limit_price=110,
                stop_loss_stop_price=95,
            )
            res = broker.place_bracket_order(req)
            self.assertTrue(res.parent_order_id.startswith("sim-"))
            self.assertTrue(res.take_profit_order_id.startswith("sim-"))
            self.assertTrue(res.stop_loss_order_id.startswith("sim-"))
        finally:
            broker.disconnect()

    def test_modify_order(self):
        broker = SimBroker()
        broker.connect()
        try:
            res = broker.place_order(TradeIntent(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1).to_order_request())
            req2 = TradeIntent(
                instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                side="BUY",
                quantity=1,
                order_type="LMT",
                limit_price=100,
            ).to_order_request()
            mod_res = broker.modify_order(res.order_id, req2)
            self.assertEqual(mod_res.order_id, res.order_id)
        finally:
            broker.disconnect()

    def test_historical_bars(self):
        from trading_algo.broker.base import Bar

        broker = SimBroker()
        broker.connect()
        try:
            inst = InstrumentSpec(kind="STK", symbol="AAPL")
            broker.set_historical_bars(
                inst,
                [Bar(timestamp_epoch_s=1, open=1, high=2, low=0.5, close=1.5, volume=10)],
            )
            bars = broker.get_historical_bars(inst, end_datetime=None, duration="1 D", bar_size="5 mins")
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0].close, 1.5)
        finally:
            broker.disconnect()

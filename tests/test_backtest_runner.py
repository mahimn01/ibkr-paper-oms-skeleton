import logging
import unittest

from trading_algo.backtest.runner import BacktestConfig, run_backtest
from trading_algo.broker.base import Bar
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent

logging.disable(logging.CRITICAL)


class _BuyOnce:
    name = "buy-once"

    def __init__(self):
        self._did = False

    def on_tick(self, ctx):
        if self._did:
            return []
        self._did = True
        return [TradeIntent(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="MKT")]


class TestBacktestRunner(unittest.TestCase):
    def test_backtest_runs_and_returns_result(self):
        inst = InstrumentSpec(kind="STK", symbol="AAPL")
        bars = [
            Bar(timestamp_epoch_s=1, open=100, high=105, low=95, close=102, volume=1000),
            Bar(timestamp_epoch_s=2, open=102, high=106, low=101, close=104, volume=1000),
        ]
        res = run_backtest(_BuyOnce(), inst, bars, BacktestConfig(initial_cash=1000))
        self.assertIsNotNone(res.start_equity)
        self.assertIsNotNone(res.end_equity)


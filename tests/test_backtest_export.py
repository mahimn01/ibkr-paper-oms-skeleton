import logging
import os
import tempfile
import unittest

from trading_algo.backtest.export import ExportConfig, export_historical_bars
from trading_algo.broker.base import Bar
from trading_algo.instruments import InstrumentSpec

logging.disable(logging.CRITICAL)


class _FakeHistoryBroker:
    def __init__(self, bars):
        self._bars = list(bars)

    def get_historical_bars(self, instrument, *, end_datetime=None, duration, bar_size, what_to_show="TRADES", use_rth=False):
        # Simulate pagination: if end_datetime is None, return last 2 bars; else return the prior 2 bars.
        if end_datetime is None:
            return self._bars[-2:]
        end_ts = float(end_datetime)
        eligible = [b for b in self._bars if b.timestamp_epoch_s <= end_ts]
        return eligible[-2:]


class TestBacktestExport(unittest.TestCase):
    def test_export_paginates_and_writes_csv(self):
        bars = [
            Bar(timestamp_epoch_s=1, open=1, high=1, low=1, close=1, volume=1),
            Bar(timestamp_epoch_s=2, open=2, high=2, low=2, close=2, volume=2),
            Bar(timestamp_epoch_s=3, open=3, high=3, low=3, close=3, volume=3),
            Bar(timestamp_epoch_s=4, open=4, high=4, low=4, close=4, volume=4),
        ]
        broker = _FakeHistoryBroker(bars)
        inst = InstrumentSpec(kind="STK", symbol="AAPL")

        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            out = export_historical_bars(
                broker,  # type: ignore[arg-type]
                inst,
                out_csv_path=path,
                cfg=ExportConfig(duration_per_call="1 D", bar_size="1 min", pacing_sleep_seconds=0.0, max_calls=10),
            )
            self.assertGreaterEqual(len(out), 4)
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            self.assertEqual(header, "timestamp,open,high,low,close,volume")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


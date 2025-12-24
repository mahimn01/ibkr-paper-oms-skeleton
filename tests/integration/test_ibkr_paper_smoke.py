import logging
import os
import time
import unittest
import warnings

# Silence third-party deprecation warnings/logging during integration runs.
warnings.simplefilter("ignore", DeprecationWarning)
logging.disable(logging.CRITICAL)

from trading_algo.broker.base import OrderRequest
from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


@unittest.skipUnless(os.getenv("RUN_IBKR_INTEGRATION") == "1", "set RUN_IBKR_INTEGRATION=1 to run paper integration tests")
class TestIBKRPaperIntegration(unittest.TestCase):
    def setUp(self):
        self.cfg = IBKRConfig(
            host=_env("IBKR_HOST", "127.0.0.1"),
            port=int(_env("IBKR_PORT", "7497")),
            client_id=int(_env("IBKR_CLIENT_ID", "77")),
        )
        self.broker = IBKRBroker(self.cfg, require_paper=True)
        try:
            self.broker.connect()
        except Exception as exc:
            raise unittest.SkipTest(f"IBKR connection not available ({exc}); start TWS paper and enable API port") from exc

    def tearDown(self):
        try:
            self.broker.disconnect()
        except Exception:
            pass

    def test_snapshot(self):
        snap = self.broker.get_market_data_snapshot(InstrumentSpec(kind="STK", symbol="AAPL"))
        self.assertEqual(snap.instrument.symbol, "AAPL")

    def test_place_modify_cancel_limit_order(self):
        inst = InstrumentSpec(kind="STK", symbol="AAPL")
        # Far away BUY so it should not fill; safe for paper.
        req = OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="LMT", limit_price=0.5, tif="DAY")
        res = self.broker.place_order(req)
        self.assertTrue(res.order_id.isdigit())

        # Modify to another far away limit.
        req2 = OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="LMT", limit_price=0.6, tif="DAY")
        res2 = self.broker.modify_order(res.order_id, req2)
        self.assertEqual(res2.order_id, res.order_id)

        time.sleep(0.5)
        self.broker.cancel_order(res.order_id)
        st = self.broker.get_order_status(res.order_id)
        self.assertIn(st.status, {"Cancelled", "PendingCancel", "ApiCancelled", "Inactive", "Submitted", "PreSubmitted"})

    def test_reconcile_and_track_with_sqlite(self):
        import tempfile

        from trading_algo.config import TradingConfig
        from trading_algo.oms import OrderManager
        from trading_algo.persistence import SqliteStore

        # Place far-away limit and persist it, then create a new OMS and reconcile it.
        inst = InstrumentSpec(kind="STK", symbol="AAPL")
        req = OrderRequest(instrument=inst, side="BUY", quantity=1, order_type="LMT", limit_price=0.4, tif="DAY")
        res = self.broker.place_order(req)

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as f:
            cfg = TradingConfig(
                broker="ibkr",
                live_enabled=False,
                dry_run=True,
                order_token=None,
                db_path=f.name,
                ibkr=self.cfg,
            )
            store = SqliteStore(cfg.db_path)
            run_id = store.start_run(cfg)
            store.log_order(run_id, broker="ibkr", order_id=res.order_id, request=req, status=res.status)
            store.end_run(run_id)
            store.close()

            oms = OrderManager(self.broker, cfg, confirm_token=None)
            try:
                out = oms.reconcile()
                self.assertIn(res.order_id, out)
                oms.track_open_orders(poll_seconds=0.2, timeout_seconds=1.0)
            finally:
                oms.close()

    def test_export_history_and_backtest_roundtrip(self):
        import tempfile

        from trading_algo.backtest.export import ExportConfig, export_historical_bars
        from trading_algo.backtest.runner import BacktestConfig, run_backtest
        from trading_algo.strategy.example import ExampleStrategy

        inst = InstrumentSpec(kind="STK", symbol="AAPL")
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            bars = export_historical_bars(
                self.broker,
                inst,
                out_csv_path=f.name,
                cfg=ExportConfig(duration_per_call="2 D", bar_size="5 mins", pacing_sleep_seconds=0.25, max_calls=5),
            )
            self.assertGreater(len(bars), 0)
            res = run_backtest(ExampleStrategy(symbol="AAPL"), inst, bars, BacktestConfig(initial_cash=100000))
            self.assertIsNotNone(res.end_equity)

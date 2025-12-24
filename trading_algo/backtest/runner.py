from __future__ import annotations

from dataclasses import dataclass

from trading_algo.backtest.broker import BacktestBroker, FillModel
from trading_algo.broker.base import Bar
from trading_algo.instruments import InstrumentSpec
from trading_algo.oms import OrderManager
from trading_algo.risk import RiskLimits, RiskManager
from trading_algo.strategy.base import Strategy, StrategyContext


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_per_order: float = 0.0
    slippage_bps: float = 0.0
    spread: float = 0.0
    db_path: str | None = None


@dataclass(frozen=True)
class BacktestResult:
    start_equity: float
    end_equity: float
    return_pct: float


def run_backtest(strategy: Strategy, instrument: InstrumentSpec, bars: list[Bar], cfg: BacktestConfig) -> BacktestResult:
    broker = BacktestBroker(
        instrument=instrument,
        bars=bars,
        initial_cash=cfg.initial_cash,
        fill_model=FillModel(commission_per_order=cfg.commission_per_order, slippage_bps=cfg.slippage_bps),
        spread=cfg.spread,
    )
    broker.connect()
    try:
        # For backtests, treat as a "sim-like" broker with no token gating.
        from trading_algo.config import TradingConfig, IBKRConfig

        tcfg = TradingConfig(
            broker="sim",
            live_enabled=True,
            require_paper=True,
            dry_run=False,
            order_token=None,
            db_path=cfg.db_path,
            ibkr=IBKRConfig(),
        )
        oms = OrderManager(broker, tcfg, confirm_token=None)
        risk = RiskManager(RiskLimits(allow_short=True))
        try:
            start_equity = broker.get_account_snapshot().values["NetLiquidation"]
            while True:
                bar = broker.current_bar()
                ctx = StrategyContext(now_epoch_s=bar.timestamp_epoch_s, get_snapshot=broker.get_market_data_snapshot)
                intents = list(strategy.on_tick(ctx))
                for intent in intents:
                    try:
                        risk.validate(intent, broker, ctx.get_snapshot)
                        oms.log_decision(getattr(strategy, "name", "strategy"), intent, accepted=True, reason="backtest")
                        oms.submit(intent.to_order_request())
                    except Exception as exc:
                        oms.log_decision(getattr(strategy, "name", "strategy"), intent, accepted=False, reason=str(exc))
                if not broker.step():
                    break
            end_equity = broker.get_account_snapshot().values["NetLiquidation"]
            return_pct = (end_equity - start_equity) / start_equity * 100.0 if start_equity else 0.0
            return BacktestResult(start_equity=start_equity, end_equity=end_equity, return_pct=return_pct)
        finally:
            oms.close()
    finally:
        broker.disconnect()


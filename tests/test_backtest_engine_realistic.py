"""End-to-end tests for the realistic cost model + NEXT_BAR_OPEN policy
wired into BacktestEngine.

The legacy path (cost_model_config=None, execution_policy='same_bar_close')
is exercised by the pre-existing test_backtest*.py suite. These tests
prove the new path:
  * realistic fills cost more than legacy flat-bps in normal conditions
  * NEXT_BAR_OPEN fills at the next bar's open, not the signal bar's close
  * borrow accrual reduces cash on shorts held overnight
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from trading_algo.backtest_v2.cost_model import CostModelConfig
from trading_algo.backtest_v2.engine import BacktestEngine
from trading_algo.backtest_v2.models import Bar, BacktestConfig


# ----------------------------------------------------------------- test fixtures


@dataclass
class _Sig:
    action: str
    confidence: float = 0.5
    reason: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None
    edge_votes: dict = None
    market_regime: str = "UNKNOWN"

    def __post_init__(self):
        if self.edge_votes is None:
            self.edge_votes = {}


class _ScriptedStrategy:
    """A pre-scripted strategy used for deterministic engine assertions.

    The `script` is a list of (timestamp_index, action) tuples. When
    generate_signal is asked at the matching timestamp, it returns the
    action; otherwise None.
    """

    def __init__(self, script: List[tuple]):
        self.script = list(script)
        self.positions: Dict[str, Any] = {}
        self.asset_states: Dict[str, Any] = {}
        self._idx = 0
        self._ts_to_idx: Dict[datetime, int] = {}

    def update_asset(self, symbol, timestamp, open_price, high, low, close, volume):
        if timestamp not in self._ts_to_idx:
            self._ts_to_idx[timestamp] = len(self._ts_to_idx)

    def generate_signal(self, symbol, timestamp):
        i = self._ts_to_idx.get(timestamp, -1)
        for (idx, action) in self.script:
            if idx == i:
                return _Sig(action=action, reason=f"scripted:{action}")
        return None

    def clear_positions(self) -> None:
        self.positions = {}


def _make_bars(n: int, start: float = 100.0, step: float = 0.0,
               volume: int = 1_000_000) -> List[Bar]:
    """Daily bars with a fixed close path. step=0 gives a flat market."""
    out: List[Bar] = []
    base = datetime(2024, 1, 2, 9, 30)
    for i in range(n):
        c = start + step * i
        out.append(Bar(
            timestamp=base + timedelta(days=i),
            open=c, high=c + 0.20, low=c - 0.20, close=c,
            volume=volume, vwap=c,
        ))
    return out


# ----------------------------------------------------------------- realistic costs

def test_legacy_engine_runs_with_unchanged_defaults() -> None:
    """Sanity: BacktestConfig defaults still execute the legacy path
    (no exception, costs computed via flat slippage_pct)."""
    cfg = BacktestConfig(strategy_name="legacy", symbols=["AAPL"],
                         bar_size="1 day", initial_capital=100_000)
    engine = BacktestEngine(cfg)
    # 100 flat bars, single buy -> hold -> sell.
    bars = _make_bars(100)
    strat = _ScriptedStrategy(script=[(60, "buy"), (80, "sell")])
    results = engine.run(strat, {"AAPL": bars})
    # Should produce one trade.
    assert len(results.trades) == 1


def test_realistic_cost_model_runs_and_charges_more_than_legacy() -> None:
    """With identical paper fills, realistic cost > flat 5bps slippage on
    a small order. Half-spread alone is typically 5-50 bps for small caps;
    the realistic stack should dominate."""
    bars = _make_bars(120, start=10.0, step=0.0, volume=1_000_000)

    # Legacy run.
    legacy_cfg = BacktestConfig(
        strategy_name="legacy", symbols=["XYZ"], bar_size="1 day",
        initial_capital=100_000,
        slippage_pct=0.0005,         # 5 bps flat
    )
    legacy_engine = BacktestEngine(legacy_cfg)
    legacy_results = legacy_engine.run(
        _ScriptedStrategy(script=[(60, "buy"), (80, "sell")]),
        {"XYZ": bars},
    )

    # Realistic run.
    realistic_cfg = BacktestConfig(
        strategy_name="realistic", symbols=["XYZ"], bar_size="1 day",
        initial_capital=100_000,
        cost_model_config=CostModelConfig(
            enable_spread=True,
            enable_impact=True,
            enable_commission=True,
            fallback_spread_bps=20.0,    # used when CS+AR fail (flat market)
        ),
    )
    realistic_engine = BacktestEngine(realistic_cfg)
    realistic_results = realistic_engine.run(
        _ScriptedStrategy(script=[(60, "buy"), (80, "sell")]),
        {"XYZ": bars},
    )

    # Both should produce one trade.
    assert len(legacy_results.trades) == 1
    assert len(realistic_results.trades) == 1

    legacy_pnl = legacy_results.trades[0].net_pnl
    realistic_pnl = realistic_results.trades[0].net_pnl
    # Realistic path charges more (lower or equal P&L on flat market).
    assert realistic_pnl <= legacy_pnl + 1e-6


# ----------------------------------------------------------------- NEXT_BAR_OPEN

def test_next_bar_open_fills_at_next_bar_not_signal_bar() -> None:
    """A signal generated on bar T fills at bar T+1's open under
    NEXT_BAR_OPEN policy. We construct bars where signal-bar close =
    100 but next-bar open = 105, then verify the trade entry > 100."""
    base = datetime(2024, 1, 2, 9, 30)
    bars: List[Bar] = []
    for i in range(120):
        if i == 60:
            # Signal bar: close = 100.
            bars.append(Bar(timestamp=base + timedelta(days=i),
                            open=99, high=101, low=98, close=100.0,
                            volume=1_000_000, vwap=100.0))
        elif i == 61:
            # Fill bar: open = 105 (gap up).
            bars.append(Bar(timestamp=base + timedelta(days=i),
                            open=105.0, high=106, low=104, close=105.5,
                            volume=1_000_000, vwap=105.0))
        else:
            bars.append(Bar(timestamp=base + timedelta(days=i),
                            open=100, high=101, low=99, close=100,
                            volume=1_000_000, vwap=100.0))

    cfg = BacktestConfig(
        strategy_name="lookahead-test", symbols=["XYZ"], bar_size="1 day",
        initial_capital=1_000_000,
        execution_policy="next_bar_open",
        # Use legacy cost path for clean, predictable fills.
        slippage_pct=0.0,
    )
    engine = BacktestEngine(cfg)
    strat = _ScriptedStrategy(script=[(60, "buy"), (90, "sell")])
    results = engine.run(strat, {"XYZ": bars})

    assert len(results.trades) == 1
    entry = results.trades[0].entry_price
    # Under NEXT_BAR_OPEN, entry should be ~ bar 61's open (105), NOT bar 60's close (100).
    assert entry == pytest.approx(105.0, abs=0.01)


def test_same_bar_close_fills_at_signal_bar() -> None:
    """Counterfactual: under SAME_BAR_CLOSE, the same signal fills at
    the signal bar's close (100), demonstrating the look-ahead window."""
    base = datetime(2024, 1, 2, 9, 30)
    bars: List[Bar] = []
    for i in range(120):
        if i == 60:
            bars.append(Bar(timestamp=base + timedelta(days=i),
                            open=99, high=101, low=98, close=100.0,
                            volume=1_000_000, vwap=100.0))
        elif i == 61:
            bars.append(Bar(timestamp=base + timedelta(days=i),
                            open=105.0, high=106, low=104, close=105.5,
                            volume=1_000_000, vwap=105.0))
        else:
            bars.append(Bar(timestamp=base + timedelta(days=i),
                            open=100, high=101, low=99, close=100,
                            volume=1_000_000, vwap=100.0))

    cfg = BacktestConfig(
        strategy_name="legacy-look", symbols=["XYZ"], bar_size="1 day",
        initial_capital=1_000_000,
        execution_policy="same_bar_close",   # legacy
        slippage_pct=0.0,
    )
    engine = BacktestEngine(cfg)
    strat = _ScriptedStrategy(script=[(60, "buy"), (90, "sell")])
    results = engine.run(strat, {"XYZ": bars})

    assert len(results.trades) == 1
    entry = results.trades[0].entry_price
    # Under legacy SAME_BAR_CLOSE, fill is at signal bar's close (100).
    assert entry == pytest.approx(100.0, abs=0.01)
    # Validator-relevant warning surfaces.
    assert any("look-ahead" in w.lower() for w in results.warnings)


# ----------------------------------------------------------------- borrow accrual

def test_borrow_accrual_reduces_equity_on_shorts() -> None:
    """A short position held over many bars with default_borrow_rate_bps>0
    should cumulatively reduce equity vs. the same setup without borrow."""
    bars = _make_bars(120, start=100.0, step=0.0, volume=1_000_000)

    # Run identical scripts; only borrow rate differs.
    def _run(borrow_bps: float | None):
        cfg = BacktestConfig(
            strategy_name="short-test", symbols=["XYZ"], bar_size="1 day",
            initial_capital=100_000,
            allow_shorting=True,
            default_borrow_rate_bps=borrow_bps,
            slippage_pct=0.0,
            execution_policy="same_bar_close",  # avoid deferral noise
        )
        engine = BacktestEngine(cfg)
        # Open a short on bar 60, hold until bar 100.
        strat = _ScriptedStrategy(script=[(60, "short"), (100, "cover")])
        return engine.run(strat, {"XYZ": bars})

    no_borrow = _run(None)
    with_borrow = _run(500.0)   # 5% annualized

    assert len(no_borrow.trades) == 1
    assert len(with_borrow.trades) == 1
    # Cash trajectory: with-borrow strictly lower than no-borrow at end.
    assert with_borrow.metrics.final_capital < no_borrow.metrics.final_capital

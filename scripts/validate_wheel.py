"""Route the Wheel options strategy through the realistic options backtest and
the 7-gate validator, producing a real APPROVED/BLOCKED report card on disk.

This is the first end-to-end use of ``build_report_card`` on a strategy. To make
the card honest (not a silently-shrunk subset of gates) we populate BOTH the
PBO gate (via a real parameter-sweep ``trial_grid``) and the cost-adjusted
Sharpe gate (via a punitive-friction re-run).

Note: options strategies do NOT run through ``backtest_v2`` (an equity-only
engine). ``run_options_backtest`` is the purpose-built options harness and
carries its own realistic friction model (bid/ask slippage + per-contract
commission + BSM pricing with dynamic IV and skew), so the
``cost_model_config``/``next_bar_open`` mandate that applies to equity backtests
is satisfied here by the harness + the cost-adjusted gate below.

Usage:
    python scripts/validate_wheel.py --symbol AAPL
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_algo.quant_core.strategies.options.wheel import WheelStrategy, WheelConfig
from trading_algo.quant_core.strategies.options.options_backtester import run_options_backtest
from trading_algo.quant_core.validation.report_card import build_report_card
from scripts._options_data import load_daily_bars


def _daily_returns(equity_curve: list[tuple[datetime, float]]) -> np.ndarray:
    eq = np.asarray([e for _, e in equity_curve], dtype=np.float64)
    if eq.size < 2:
        return np.array([], dtype=np.float64)
    rets = np.diff(eq) / eq[:-1]
    return rets[np.isfinite(rets)]


def _run(cfg: WheelConfig, bars, symbol: str) -> np.ndarray:
    report = run_options_backtest(WheelStrategy(cfg), bars, symbol)
    return _daily_returns(report.equity_curve)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    args = ap.parse_args()
    symbol = args.symbol.upper()

    bars = load_daily_bars(symbol)
    print(f"{symbol}: {len(bars)} daily bars loaded")

    # --- Base variant (the configuration we are actually validating) ---
    base_cfg = WheelConfig(
        put_delta=0.30,
        call_delta=0.30,
        target_dte=45,
        profit_target=0.50,
        trend_sma_period=50,
        risk_free_rate=0.045,
        skew_slope=0.8,
        commission_per_contract=0.90,
        bid_ask_slip_per_share=0.05,
    )
    base_report = run_options_backtest(WheelStrategy(base_cfg), bars, symbol)
    base_rets = _daily_returns(base_report.equity_curve)
    dates = [d for d, _ in base_report.equity_curve]
    s = base_report.summary
    print(
        f"base run: total_return={s['total_return_pct']:.1f}%  "
        f"sharpe={s['sharpe_ratio']:.2f}  trades={s['total_trades']}  "
        f"maxDD={s['max_drawdown_pct']:.1f}%"
    )

    # --- Parameter sweep -> trial_grid (T, N) for the PBO/CSCV gate ---
    put_deltas = [0.20, 0.25, 0.30, 0.35]
    target_dtes = [30, 45, 60]
    profit_targets = [0.50, 0.75]
    grid_combos = list(itertools.product(put_deltas, target_dtes, profit_targets))  # 24
    print(f"running {len(grid_combos)}-variant sweep for trial_grid ...")

    columns: list[np.ndarray] = []
    for pd_, dte_, pt_ in grid_combos:
        cfg = replace(base_cfg, put_delta=pd_, call_delta=pd_, target_dte=dte_, profit_target=pt_)
        columns.append(_run(cfg, bars, symbol))

    # Align columns to a common length (front-truncate so rows are time-aligned).
    common_t = min(len(c) for c in columns)
    grid = np.column_stack([c[-common_t:] for c in columns])  # (T, N)
    print(f"trial_grid shape: {grid.shape}")

    # --- Cost-adjusted re-run (punitive friction) for the cost-adjusted gate ---
    hc_cfg = replace(
        base_cfg,
        bid_ask_slip_per_share=0.15,
        commission_per_contract=1.30,
        commission_per_share=0.01,
    )
    cost_adj_rets = _run(hc_cfg, bars, symbol)

    # --- Build the report card (all gates populated) ---
    rc = build_report_card(
        strategy_name=f"wheel_{symbol}",
        returns=base_rets,
        n_trials=len(grid_combos),
        trial_grid=grid,
        cost_adjusted_returns=cost_adj_rets,
        periods_per_year=252,
        period_start=dates[0].date(),
        period_end=dates[-1].date(),
    )

    out_dir = Path("validation_reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"wheel_{symbol}.md"
    out_path.write_text(rc.render())

    print("\n" + rc.render())
    print(f"\nwrote {out_path}")
    print(f"STATUS: {rc.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

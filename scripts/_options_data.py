"""Hermetic local daily-bar loader for options backtests (no network).

Resamples the cached intraday bars under ``data/cache/<SYM>_5mins_*.json`` to
daily OHLCV ``broker.base.Bar`` objects — the format ``run_options_backtest``
expects. Replaces the yfinance fetch in the older wheel scripts so validation
runs are reproducible.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_algo.broker.base import Bar


def load_daily_bars(symbol: str, cache_dir: str = "data/cache") -> list[Bar]:
    """Load intraday cache for ``symbol`` and resample to daily Bars."""
    pattern = os.path.join(cache_dir, f"{symbol}_5mins_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No cache file matching {pattern}")
    # Prefer the longest-history file (largest on disk).
    path = max(files, key=os.path.getsize)

    with open(path) as f:
        records = json.load(f)

    # Records are chronological; group by calendar date preserving first-seen order.
    daily: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for r in records:
        key = datetime.fromisoformat(r["timestamp"]).date().isoformat()
        o, h, lo, c = r["open"], r["high"], r["low"], r["close"]
        v = r.get("volume") or 0.0
        agg = daily.get(key)
        if agg is None:
            daily[key] = {"open": o, "high": h, "low": lo, "close": c, "volume": v}
            order.append(key)
        else:
            agg["high"] = max(agg["high"], h)
            agg["low"] = min(agg["low"], lo)
            agg["close"] = c  # last intraday close of the day
            agg["volume"] += v

    bars: list[Bar] = []
    for key in order:
        agg = daily[key]
        dt = datetime.combine(datetime.fromisoformat(key).date(), time(16, 0))
        bars.append(
            Bar(
                timestamp_epoch_s=dt.timestamp(),
                open=agg["open"],
                high=agg["high"],
                low=agg["low"],
                close=agg["close"],
                volume=agg["volume"],
            )
        )
    return bars


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    bars = load_daily_bars(sym)
    d0 = datetime.fromtimestamp(bars[0].timestamp_epoch_s).date()
    d1 = datetime.fromtimestamp(bars[-1].timestamp_epoch_s).date()
    print(f"{sym}: {len(bars)} daily bars {d0} -> {d1}")

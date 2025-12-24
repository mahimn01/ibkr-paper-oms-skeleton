from __future__ import annotations

import csv
import time
from dataclasses import dataclass

from trading_algo.broker.base import Bar, Broker
from trading_algo.instruments import InstrumentSpec, validate_instrument


@dataclass(frozen=True)
class ExportConfig:
    duration_per_call: str = "30 D"
    bar_size: str = "5 mins"
    what_to_show: str = "TRADES"
    use_rth: bool = False
    pacing_sleep_seconds: float = 0.25
    max_calls: int = 500


def export_historical_bars(
    broker: Broker,
    instrument: InstrumentSpec,
    *,
    out_csv_path: str,
    cfg: ExportConfig,
    end_datetime: str | None = None,
) -> list[Bar]:
    """
    Export historical bars to CSV in backtest-compatible format.

    Uses IBKR-style pagination by repeatedly calling `get_historical_bars` with a moving `end_datetime`.
    """
    instrument = validate_instrument(instrument)
    end_dt = end_datetime
    all_bars: list[Bar] = []

    for _ in range(int(cfg.max_calls)):
        chunk = broker.get_historical_bars(
            instrument,
            end_datetime=end_dt,
            duration=cfg.duration_per_call,
            bar_size=cfg.bar_size,
            what_to_show=cfg.what_to_show,
            use_rth=cfg.use_rth,
        )
        if not chunk:
            break
        # Deduplicate by timestamp
        existing = {b.timestamp_epoch_s for b in all_bars}
        for b in chunk:
            if b.timestamp_epoch_s not in existing:
                all_bars.append(b)
                existing.add(b.timestamp_epoch_s)

        all_bars.sort(key=lambda b: b.timestamp_epoch_s)
        earliest = all_bars[0].timestamp_epoch_s
        # Move end backward (IB expects local/UTC-ish string; keep simple epoch string)
        end_dt = str(int(earliest) - 1)
        time.sleep(float(cfg.pacing_sleep_seconds))

    # Write CSV
    all_bars.sort(key=lambda b: b.timestamp_epoch_s)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        w.writeheader()
        for b in all_bars:
            w.writerow(
                {
                    "timestamp": int(b.timestamp_epoch_s),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": "" if b.volume is None else b.volume,
                }
            )
    return all_bars


from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass

from trading_algo.broker.base import Bar
from trading_algo.instruments import InstrumentSpec, validate_instrument


@dataclass(frozen=True)
class BarSeries:
    instrument: InstrumentSpec
    bars: list[Bar]


def _parse_timestamp(value: str) -> float:
    """
    Accept epoch seconds or ISO-like datetimes.
    """
    value = value.strip()
    if not value:
        raise ValueError("timestamp is required")
    try:
        return float(value)
    except ValueError:
        pass
    # Try ISO-8601
    try:
        # Support 'Z'
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return dt.datetime.fromisoformat(value).timestamp()
    except Exception as exc:
        raise ValueError(f"Unsupported timestamp format: {value}") from exc


def load_bars_csv(path: str, instrument: InstrumentSpec) -> BarSeries:
    """
    Load bars from CSV.

    Required columns: timestamp, open, high, low, close
    Optional: volume
    """
    instrument = validate_instrument(instrument)
    bars: list[Bar] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "open", "high", "low", "close"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required columns {sorted(required)}; got {reader.fieldnames}")

        for row in reader:
            ts = _parse_timestamp(str(row["timestamp"]))
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            v_raw = row.get("volume", None)
            v = float(v_raw) if v_raw not in (None, "", "null", "None") else None
            bars.append(Bar(timestamp_epoch_s=ts, open=o, high=h, low=l, close=c, volume=v))

    bars.sort(key=lambda b: b.timestamp_epoch_s)
    return BarSeries(instrument=instrument, bars=bars)


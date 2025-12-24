from __future__ import annotations

from dataclasses import dataclass

from trading_algo.broker.base import Bar


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error" | "warn"
    message: str


def validate_bars(bars: list[Bar]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not bars:
        issues.append(ValidationIssue("error", "no bars"))
        return issues

    last_ts = None
    for i, b in enumerate(bars):
        if b.timestamp_epoch_s is None:
            issues.append(ValidationIssue("error", f"bar[{i}] missing timestamp"))
            continue
        if last_ts is not None and b.timestamp_epoch_s < last_ts:
            issues.append(ValidationIssue("error", f"timestamps not sorted at bar[{i}]"))
        last_ts = b.timestamp_epoch_s

        for name, v in [("open", b.open), ("high", b.high), ("low", b.low), ("close", b.close)]:
            if v is None:
                issues.append(ValidationIssue("error", f"bar[{i}] missing {name}"))
            elif v <= 0:
                issues.append(ValidationIssue("error", f"bar[{i}] non-positive {name}={v}"))

        if b.low > b.high:
            issues.append(ValidationIssue("error", f"bar[{i}] low > high"))
        if not (b.low <= b.open <= b.high):
            issues.append(ValidationIssue("warn", f"bar[{i}] open outside [low,high]"))
        if not (b.low <= b.close <= b.high):
            issues.append(ValidationIssue("warn", f"bar[{i}] close outside [low,high]"))

    # Duplicate timestamps
    seen: set[float] = set()
    for i, b in enumerate(bars):
        if b.timestamp_epoch_s in seen:
            issues.append(ValidationIssue("warn", f"duplicate timestamp at bar[{i}]"))
        seen.add(b.timestamp_epoch_s)
    return issues


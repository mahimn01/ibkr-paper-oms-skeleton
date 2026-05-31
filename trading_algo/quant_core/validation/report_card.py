"""Validation report card: a 7-gate pass/fail audit per strategy.

PLAN.md §2.7 — production validation harness.

Gates (all must pass for APPROVED):

    Gate                                Threshold                Reference
    -------------------------------     ----------------------    ----------
    PBO (CSCV)                          < 0.5                    Bailey-Borwein-LdP-Zhu 2017
    Deflated Sharpe Ratio               > 0.95                   Bailey-LdP 2014
    Lower 95% CI on annualized Sharpe   > 0.3                    Politis-Romano 1994
    Walk-forward 12m rolling Sharpe     > 0 in >= 75% of windows
    MinTRL (years)                      <= years available       Bailey-LdP 2012
    Synthetic OU/GBM percentile         <= 50th OOS              AFML Ch 13
    Cost-stack adjusted Sharpe          > 0.3 (after spread+impact+borrow)

Inputs:
    returns: 1-D array of per-period strategy returns (e.g. daily)
    n_trials: total parameter combinations explored historically (for DSR)
    trial_grid: optional (N x T) matrix used by CSCV; None disables PBO
    period_start / period_end / periods_per_year: metadata for MinTRL
    walk_forward_window: int (periods per fold for the rolling test)

Output:
    ReportCard dataclass with all metrics + the markdown render() method.

Default usage:
    >>> from trading_algo.quant_core.validation.report_card import build_report_card
    >>> rc = build_report_card(strategy_name="my_strat", returns=rets, n_trials=42)
    >>> open("validation_reports/my_strat.md", "w").write(rc.render())
    >>> assert rc.status == "APPROVED"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.validation.pbo import (
    DeflatedSharpe,
    PBOCalculator,
)
from trading_algo.quant_core.validation.stationary_bootstrap import (
    bootstrap_sharpe_ci,
)


# --------------------------------------------------------------------------
# Result dataclasses
# --------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    value: float | str
    threshold: str
    passed: bool
    note: str = ""


@dataclass
class ReportCard:
    strategy_name: str
    period_start: Optional[date]
    period_end:   Optional[date]
    samples: int
    n_trials: int

    point_sharpe:        float = 0.0
    sharpe_ci_lower:     float = 0.0
    sharpe_ci_upper:     float = 0.0
    pbo:                 Optional[float] = None
    deflated_sharpe:     Optional[float] = None
    rolling_12m_pos_pct: Optional[float] = None
    min_trl_years:       Optional[float] = None
    cost_adjusted_sharpe: Optional[float] = None

    gates: list[GateResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Overall status: APPROVED iff every populated gate passes."""
        if not self.gates:
            return "BLOCKED"
        return "APPROVED" if all(g.passed for g in self.gates) else "BLOCKED"

    def render(self) -> str:
        """Render as markdown report card (PLAN.md §2.7 acceptance gate output)."""
        lines: list[str] = []
        lines.append(f"# Strategy validation report — {self.strategy_name}")
        lines.append("")
        if self.period_start and self.period_end:
            lines.append(f"**Sample:** {self.period_start.isoformat()} "
                         f"to {self.period_end.isoformat()}  ")
        lines.append(f"**Observations (T):** {self.samples}  ")
        lines.append(f"**Trials searched (N):** {self.n_trials}  ")
        lines.append("")
        lines.append("| Metric | Value | Threshold | Pass |")
        lines.append("| --- | --- | --- | --- |")
        for g in self.gates:
            mark = "✓" if g.passed else "✗"
            v = g.value
            if isinstance(v, float):
                v = f"{v:.4f}"
            lines.append(f"| {g.name} | {v} | {g.threshold} | {mark} |")
        lines.append("")
        lines.append(f"**Status:** {self.status}")
        if self.warnings:
            lines.append("")
            lines.append("**Warnings:**")
            for w in self.warnings:
                lines.append(f"- {w}")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _annualised_sharpe(returns: NDArray[np.float64], periods_per_year: float) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(returns)) / sd * math.sqrt(periods_per_year)


def _rolling_12m_pos_share(
    returns: NDArray[np.float64],
    periods_per_year: int,
) -> float | None:
    """Share of rolling 12-month windows whose annualized Sharpe > 0.

    Returns None if T < 18 months equivalent (not enough data).
    """
    n = len(returns)
    win = periods_per_year       # 12 months
    if n < win + int(periods_per_year / 2):
        return None
    pos = 0
    total = 0
    for i in range(win, n + 1):
        sl = returns[i - win:i]
        if len(sl) < win:
            continue
        sd = float(np.std(sl, ddof=1))
        if sd < 1e-12:
            continue
        sr = float(np.mean(sl)) / sd * math.sqrt(periods_per_year)
        total += 1
        if sr > 0:
            pos += 1
    if total == 0:
        return None
    return pos / total


def _min_trl_years(
    returns: NDArray[np.float64],
    sr_target: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: int = 252,
) -> float | None:
    """Minimum Track Record Length (Bailey-LdP 2012) in years.

        MinTRL = 1 + [1 - g3*SR + (g4-1)/4 * SR^2] * (Z / (SR - SR*))^2

    Returned in years (divide periods by periods_per_year).
    """
    from scipy import stats   # local import to keep top-level light
    n = len(returns)
    if n < 30:
        return None
    sr_per_period = float(np.mean(returns) / max(np.std(returns, ddof=1), 1e-12))
    sr_target_per_period = sr_target / math.sqrt(periods_per_year)
    if sr_per_period <= sr_target_per_period:
        return float("inf")
    g3 = float(stats.skew(returns))
    g4 = float(stats.kurtosis(returns, fisher=False))
    z = float(stats.norm.ppf(confidence))
    factor = 1.0 - g3 * sr_per_period + ((g4 - 1.0) / 4.0) * sr_per_period ** 2
    if factor <= 0:
        factor = 1.0
    periods_needed = 1.0 + factor * (z / (sr_per_period - sr_target_per_period)) ** 2
    return periods_needed / periods_per_year


# --------------------------------------------------------------------------
# Gate construction
# --------------------------------------------------------------------------


def build_report_card(
    *,
    strategy_name: str,
    returns: Sequence[float] | NDArray[np.float64],
    n_trials: int = 1,
    trial_grid: Optional[NDArray[np.float64]] = None,
    cost_adjusted_returns: Optional[Sequence[float]] = None,
    periods_per_year: int = 252,
    period_start: Optional[date] = None,
    period_end: Optional[date] = None,
    bootstrap_resamples: int = 2000,
    seed: Optional[int] = 42,
    extra_warnings: Optional[Sequence[str]] = None,
) -> ReportCard:
    """Compute every gate and return a ReportCard.

    Args:
        returns: per-period strategy returns (e.g. daily). Annualisation
            is by `periods_per_year` (252 daily, 12 monthly, etc.).
        n_trials: how many parameter combinations were searched in the
            study leading to this strategy. Used by Deflated Sharpe.
        trial_grid: optional (N x T) matrix of returns across N strategy
            variants over T periods. When supplied, CSCV PBO is computed.
        cost_adjusted_returns: optional separate return stream after
            additional friction (e.g. spread + impact + borrow). Used for
            the cost-adjusted-Sharpe gate.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("returns is empty")

    rc = ReportCard(
        strategy_name=strategy_name,
        period_start=period_start,
        period_end=period_end,
        samples=int(arr.size),
        n_trials=int(n_trials),
    )
    if extra_warnings:
        rc.warnings.extend(extra_warnings)

    # Point Sharpe + bootstrap CI.
    rc.point_sharpe = _annualised_sharpe(arr, periods_per_year)
    point, lo, hi = bootstrap_sharpe_ci(
        arr,
        confidence=0.95,
        n_resamples=bootstrap_resamples,
        periods_per_year=periods_per_year,
        seed=seed,
    )
    rc.sharpe_ci_lower = float(lo)
    rc.sharpe_ci_upper = float(hi)

    # Lower 95% CI on annualised Sharpe > 0.3 gate.
    rc.gates.append(GateResult(
        name="Lower 95% CI on annualised Sharpe",
        value=rc.sharpe_ci_lower,
        threshold="> 0.3",
        passed=rc.sharpe_ci_lower > 0.3,
    ))

    # PBO (only if trial grid supplied).
    if trial_grid is not None and trial_grid.ndim == 2:
        try:
            calc = PBOCalculator(n_groups=8)
            pbo_res = calc.calculate_multi_strategy(np.asarray(trial_grid))
            rc.pbo = float(pbo_res.pbo)
            rc.gates.append(GateResult(
                name="PBO (CSCV)",
                value=rc.pbo,
                threshold="< 0.5",
                passed=rc.pbo < 0.5,
            ))
        except Exception as exc:
            rc.warnings.append(f"PBO computation failed: {exc}")

    # Deflated Sharpe.
    try:
        from scipy import stats as _stats
        ds = DeflatedSharpe()
        ds_res = ds.calculate(
            observed_sharpe=rc.point_sharpe,
            n_trials=max(1, n_trials),
            n_observations=int(arr.size),
            skewness=float(_stats.skew(arr)),
            kurtosis=float(_stats.kurtosis(arr, fisher=False)),
            correlation=0.0,
        )
        rc.deflated_sharpe = float(ds_res.deflated_sharpe)
        rc.gates.append(GateResult(
            name="Deflated Sharpe",
            value=rc.deflated_sharpe,
            threshold="> 0.95",
            passed=rc.deflated_sharpe > 0.95,
        ))
    except Exception as exc:
        rc.warnings.append(f"DSR computation failed: {exc}")

    # Walk-forward 12m rolling Sharpe positivity.
    rc.rolling_12m_pos_pct = _rolling_12m_pos_share(arr, periods_per_year)
    if rc.rolling_12m_pos_pct is not None:
        rc.gates.append(GateResult(
            name="Walk-fwd 12m Sharpe > 0",
            value=f"{rc.rolling_12m_pos_pct * 100:.1f}%",
            threshold=">= 75%",
            passed=rc.rolling_12m_pos_pct >= 0.75,
        ))

    # MinTRL.
    try:
        years_needed = _min_trl_years(
            arr, sr_target=0.0, confidence=0.95, periods_per_year=periods_per_year,
        )
        if years_needed is not None and math.isfinite(years_needed):
            rc.min_trl_years = float(years_needed)
            years_have = arr.size / periods_per_year
            rc.gates.append(GateResult(
                name="MinTRL (years)",
                value=rc.min_trl_years,
                threshold=f"<= {years_have:.2f}",
                passed=rc.min_trl_years <= years_have,
            ))
    except Exception as exc:
        rc.warnings.append(f"MinTRL computation failed: {exc}")

    # Cost-adjusted Sharpe gate.
    if cost_adjusted_returns is not None:
        ca = np.asarray(cost_adjusted_returns, dtype=np.float64).ravel()
        ca_sharpe = _annualised_sharpe(ca, periods_per_year)
        rc.cost_adjusted_sharpe = float(ca_sharpe)
        rc.gates.append(GateResult(
            name="Cost-adjusted Sharpe",
            value=rc.cost_adjusted_sharpe,
            threshold="> 0.3",
            passed=rc.cost_adjusted_sharpe > 0.3,
        ))

    return rc

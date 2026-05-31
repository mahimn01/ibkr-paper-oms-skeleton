"""Tests for the validation report card."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trading_algo.quant_core.validation.report_card import (
    GateResult,
    ReportCard,
    build_report_card,
)


def test_report_card_renders_markdown_table() -> None:
    rng = np.random.default_rng(0)
    # Returns with healthy positive Sharpe.
    rets = rng.normal(loc=0.001, scale=0.01, size=300)
    rc = build_report_card(
        strategy_name="unit_test",
        returns=rets,
        n_trials=1,
        period_start=date(2023, 1, 1),
        period_end=date(2024, 3, 1),
        bootstrap_resamples=200,
        seed=11,
    )
    assert rc.samples == 300
    md = rc.render()
    assert "# Strategy validation report — unit_test" in md
    assert "| Metric | Value | Threshold | Pass |" in md
    assert "Status:" in md


def test_report_card_blocks_on_negative_sharpe() -> None:
    """A losing return stream should fail the lower-CI gate."""
    rng = np.random.default_rng(1)
    rets = rng.normal(loc=-0.0005, scale=0.01, size=500)  # Sharpe < 0
    rc = build_report_card(
        strategy_name="loser",
        returns=rets,
        n_trials=10,
        bootstrap_resamples=200,
        seed=7,
    )
    assert rc.status == "BLOCKED"
    # The CI lower bound gate should fail.
    ci_gate = next(g for g in rc.gates if g.name.startswith("Lower 95% CI"))
    assert ci_gate.passed is False


def test_report_card_with_pbo_grid() -> None:
    """Trial grid -> PBO gate populated."""
    rng = np.random.default_rng(3)
    grid = rng.normal(loc=0.0001, scale=0.01, size=(252, 30))   # T x N
    rets = grid[:, 0]
    rc = build_report_card(
        strategy_name="with_pbo",
        returns=rets,
        n_trials=30,
        trial_grid=grid,
        bootstrap_resamples=200,
        seed=5,
    )
    pbo_gates = [g for g in rc.gates if "PBO" in g.name]
    assert pbo_gates
    assert isinstance(pbo_gates[0].value, float)


def test_report_card_with_cost_adjusted_returns_blocks_when_unprofitable() -> None:
    rng = np.random.default_rng(4)
    raw = rng.normal(loc=0.001, scale=0.01, size=400)            # Sharpe ~ 1.6
    cost_adj = raw - 0.0015                                       # net negative
    rc = build_report_card(
        strategy_name="costed",
        returns=raw,
        n_trials=1,
        cost_adjusted_returns=cost_adj,
        bootstrap_resamples=200,
        seed=1,
    )
    ca_gate = next(g for g in rc.gates if g.name.startswith("Cost-adjusted"))
    assert ca_gate.passed is False
    assert rc.status == "BLOCKED"


def test_report_card_status_approved_when_all_gates_pass() -> None:
    """A strong strategy with EVERY required gate present -> APPROVED.

    Must supply trial_grid (PBO) and cost_adjusted_returns (cost gate) or the
    card is INCOMPLETE, not APPROVED.
    """
    rng = np.random.default_rng(8)
    T, N = 2_500, 20
    grid = rng.normal(loc=0.0002, scale=0.008, size=(T, N))  # T x N
    grid[:, 0] += 0.0010  # the chosen variant carries a persistent edge -> low PBO
    rets = grid[:, 0]
    cost_adj = rets - 0.0002  # mild costs; still a strong positive Sharpe
    rc = build_report_card(
        strategy_name="winner",
        returns=rets,
        n_trials=N,
        trial_grid=grid,
        cost_adjusted_returns=cost_adj,
        bootstrap_resamples=300,
        seed=42,
    )
    assert rc.status == "APPROVED", rc.render()


def test_dsr_gate_is_a_probability_not_a_zscore() -> None:
    """The Deflated Sharpe gate value must be a probability in [0, 1].

    A marginal edge searched over many trials should NOT clear DSR>0.95 (the
    old z-score gate passed trivially for any positive Sharpe).
    """
    rng = np.random.default_rng(21)
    rets = rng.normal(loc=0.00035, scale=0.012, size=900)  # ~0.46 annualised Sharpe
    rc = build_report_card(
        strategy_name="marginal",
        returns=rets,
        n_trials=200,  # heavy multiple testing -> strong deflation
        bootstrap_resamples=200,
        seed=3,
    )
    dsr_gate = next(g for g in rc.gates if g.name == "Deflated Sharpe")
    assert 0.0 <= dsr_gate.value <= 1.0, "DSR gate must be a probability"
    assert dsr_gate.passed is False, "marginal edge over 200 trials must fail DSR"


def test_missing_required_gate_is_incomplete_not_approved() -> None:
    """A strong stream with no trial_grid / cost_adjusted cannot be APPROVED."""
    rng = np.random.default_rng(8)
    rets = rng.normal(loc=0.0010, scale=0.008, size=2_000)
    rc = build_report_card(
        strategy_name="no_overfit_gate",
        returns=rets,
        n_trials=1,
        bootstrap_resamples=200,
        seed=42,
    )
    assert rc.status == "INCOMPLETE", rc.render()
    assert "PBO (CSCV)" in rc.missing_required_gates
    assert "Cost-adjusted Sharpe" in rc.missing_required_gates


def test_empty_returns_raises() -> None:
    with pytest.raises(ValueError):
        build_report_card(strategy_name="empty", returns=[], n_trials=1)

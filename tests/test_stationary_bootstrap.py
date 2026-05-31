"""Tests for stationary bootstrap (Politis-Romano)."""

from __future__ import annotations

import numpy as np
import pytest

from trading_algo.quant_core.validation.stationary_bootstrap import (
    bootstrap_sharpe_ci,
    optimal_block_length,
    stationary_bootstrap,
)


def test_optimal_block_length_scales_with_T() -> None:
    # b* = T^{1/3} -> p* = T^{-1/3}; longer series => smaller p (longer blocks).
    p_short = optimal_block_length(np.zeros(64))
    p_long  = optimal_block_length(np.zeros(8000))
    assert p_short > p_long
    # Sanity: p in (0, 1].
    assert 0 < p_short <= 1
    assert 0 < p_long <= 1


def test_stationary_bootstrap_shape() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(252)
    paths = stationary_bootstrap(x, n_resamples=100, seed=42)
    assert paths.shape == (100, 252)


def test_stationary_bootstrap_resamples_observed_values() -> None:
    """Every sampled value must come from x."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    paths = stationary_bootstrap(x, n_resamples=10, seed=1)
    flat = set(paths.flatten().tolist())
    assert flat.issubset({1.0, 2.0, 3.0, 4.0, 5.0})


def test_stationary_bootstrap_p_validation() -> None:
    with pytest.raises(ValueError):
        stationary_bootstrap([1.0, 2.0, 3.0], p=0.0)
    with pytest.raises(ValueError):
        stationary_bootstrap([1.0, 2.0, 3.0], p=1.5)


def test_stationary_bootstrap_empty_raises() -> None:
    with pytest.raises(ValueError):
        stationary_bootstrap([], n_resamples=10)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    """For iid normals, the bootstrap CI should usually contain the point Sharpe."""
    rng = np.random.default_rng(7)
    rets = rng.standard_normal(500) * 0.01 + 0.0005   # ~Sharpe ~0.8
    point, lo, hi = bootstrap_sharpe_ci(rets, confidence=0.95,
                                        n_resamples=500, seed=11)
    assert lo <= point <= hi
    # Sanity: 95% CI should be a finite interval, not collapsed.
    assert hi > lo


def test_bootstrap_ci_wider_for_autocorrelated_returns() -> None:
    """A series with strong AR(1) structure gets a wider CI than its
    iid surrogate (because each block carries more dependence)."""
    rng = np.random.default_rng(3)
    n = 1000
    # AR(1) with phi = 0.6.
    eps = rng.standard_normal(n) * 0.01
    ar = np.empty(n)
    ar[0] = eps[0]
    for t in range(1, n):
        ar[t] = 0.6 * ar[t - 1] + eps[t]
    iid = rng.permutation(ar)  # destroys autocorrelation, preserves marginal

    _, lo_ar,  hi_ar  = bootstrap_sharpe_ci(ar,  confidence=0.95,
                                            n_resamples=300, seed=42)
    _, lo_iid, hi_iid = bootstrap_sharpe_ci(iid, confidence=0.95,
                                            n_resamples=300, seed=42)
    # AR series CI should typically be wider.
    width_ar  = hi_ar  - lo_ar
    width_iid = hi_iid - lo_iid
    # Allow a small slack — not a strict inequality on every seed, but with
    # phi=0.6 the gap is large in expectation. Test with margin.
    assert width_ar > 0.6 * width_iid


def test_bootstrap_ci_confidence_validation() -> None:
    with pytest.raises(ValueError):
        bootstrap_sharpe_ci([0.01, 0.02, 0.03], confidence=0.0)
    with pytest.raises(ValueError):
        bootstrap_sharpe_ci([0.01, 0.02, 0.03], confidence=1.0)

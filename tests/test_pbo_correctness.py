"""Tests for the canonical PBO formula (CSCV).

Previously the implementation reported the *complement* of PBO due to an
inverted comparison. This test pins the canonical Bailey-Borwein-LdP-Zhu
(2017) definition:

    PBO = P( rank_OOS(best_IS) > N/2 )

where N is the number of strategies (trials).
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_algo.quant_core.validation.pbo import PBOCalculator


# ----------------------------------------------------------------- adversarial null

def test_pbo_high_for_pure_noise_with_many_trials() -> None:
    """If all strategies are statistical clones (pure noise) and we
    selected the IS winner from a large pool, regression-to-the-mean
    pushes the OOS rank below the IS rank -> PBO well above 0.5.

    The prior buggy implementation returned the *complement* (~0.13);
    this test pins that PBO is now correctly elevated. With N=50
    trials and T=252 observations, empirical PBO is typically
    0.7-0.95 across seeds.
    """
    rng = np.random.default_rng(7)
    n_obs, n_strats = 252, 50
    returns = rng.standard_normal((n_obs, n_strats)) * 0.01
    calc = PBOCalculator(n_groups=8)
    result = calc.calculate_multi_strategy(returns)
    assert result.pbo > 0.5
    assert result.pbo < 1.0


# ----------------------------------------------------------------- positive edge

def test_pbo_low_when_one_strategy_strictly_dominates() -> None:
    """Construct a setup where strategy 0 has uniformly higher returns —
    its IS rank == OOS rank == top, so PBO -> 0."""
    rng = np.random.default_rng(11)
    n_obs, n_strats = 252, 20
    base = rng.standard_normal((n_obs, n_strats)) * 0.001  # tiny noise
    # Strategy 0 has a constant +20 bps per period edge over the others.
    base[:, 0] += 0.0020
    calc = PBOCalculator(n_groups=8)
    result = calc.calculate_multi_strategy(base)
    # Best IS strategy is consistently strategy 0; it ranks at the top OOS too.
    # PBO should be very low.
    assert result.pbo < 0.10


# ----------------------------------------------------------------- inverted edge

def test_pbo_high_when_is_winner_is_oos_loser() -> None:
    """Construct a pathological case: IS-best strategy is OOS-worst across
    every CSCV combination -> PBO -> 1."""
    rng = np.random.default_rng(13)
    n_obs, n_strats = 100, 8
    # Each strategy has alternating positive/negative chunks; the one that
    # happens to be best in the first half is worst in the second half by
    # construction.
    returns = np.zeros((n_obs, n_strats))
    for j in range(n_strats):
        first_half_mean = 0.001 * (j + 1)
        second_half_mean = -first_half_mean
        first = rng.normal(first_half_mean, 0.0001, n_obs // 2)
        second = rng.normal(second_half_mean, 0.0001, n_obs - n_obs // 2)
        returns[:, j] = np.concatenate([first, second])
    calc = PBOCalculator(n_groups=4)
    result = calc.calculate_multi_strategy(returns)
    # The flip-flop construction makes the IS-best the OOS-worst when the
    # split aligns by halves -> high PBO.
    assert result.pbo > 0.50


def test_pbo_returns_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((128, 16)) * 0.01
    calc = PBOCalculator(n_groups=8)
    result = calc.calculate_multi_strategy(returns)
    assert 0.0 <= result.pbo <= 1.0
    assert 0.0 <= result.pbo_std
    # Logits finite.
    assert np.all(np.isfinite(result.logits))

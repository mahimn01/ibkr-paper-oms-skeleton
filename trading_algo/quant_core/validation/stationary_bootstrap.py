"""Stationary bootstrap (Politis-Romano 1994) and Politis-White (2004)
optimal block-length selection.

Why stationary bootstrap (PLAN.md §2.7, validation appendix):
    Trading returns are autocorrelated (momentum, mean reversion, vol
    clustering). IID bootstrap of returns destroys that structure and
    underestimates the variance of derived statistics like Sharpe.
    Stationary bootstrap resamples *blocks* of geometric length so the
    autocorrelation structure is preserved up to the typical block size.

References:
    Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
        J. Amer. Statist. Assoc. 89(428), 1303-1313.
    Politis, D. N., & White, H. (2004). "Automatic Block-Length Selection
        for the Dependent Bootstrap." Econometric Reviews 23(1), 53-70.
        (Corrigendum in Patton, Politis, White 2009.)

Public API:
    stationary_bootstrap(x, p, n_resamples) -> (B, T) ndarray
    optimal_block_length(x) -> p (float)
    bootstrap_sharpe_ci(returns, confidence) -> (sharpe, lo, hi)

Implementation note:
    For full Sheppard-grade Politis-White block selection, use
    `arch.bootstrap.optimal_block_length`. The native fallback below
    uses the simpler rule b ~ T^{1/3} which is good enough for
    confidence intervals at retail scale. The library version is
    preferred when arch is installed.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# --------------------------------------------------------------------------
# Block-length selection
# --------------------------------------------------------------------------


def optimal_block_length(x: Sequence[float] | NDArray[np.float64]) -> float:
    """Return geometric-distribution parameter p for stationary bootstrap.

    Mean block length E[L] = 1/p. We use the simple sample-size rule

        b* = T^{1/3}    ->    p* = 1 / b*

    which is asymptotically valid for weakly-dependent series and avoids
    the optional `arch` dependency. For Politis-White 2004 (which is
    sharper for highly autocorrelated series) call the version inside
    `arch.bootstrap.optimal_block_length` when arch is available.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n < 4:
        # Degenerate; default to no resampling structure.
        return 1.0
    b = max(1.0, n ** (1.0 / 3.0))
    return 1.0 / b


# --------------------------------------------------------------------------
# Stationary bootstrap
# --------------------------------------------------------------------------


def stationary_bootstrap(
    x: Sequence[float] | NDArray[np.float64],
    p: float | None = None,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate `n_resamples` resampled paths from `x` of length T.

    Algorithm (Politis-Romano 1994):
        1. Choose a uniform start index in [0, T).
        2. With probability (1 - p), advance index by 1 (wrap modulo T).
        3. With probability p, jump to a fresh uniform index.
        4. Continue until T observations are collected.

    Returns an array of shape (n_resamples, T).
    """
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("cannot bootstrap an empty series")
    if p is None:
        p = optimal_block_length(arr)
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, n), dtype=np.float64)
    for b in range(n_resamples):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        jumps = rng.random(n) < p
        for t in range(1, n):
            if jumps[t]:
                idx[t] = rng.integers(0, n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        out[b] = arr[idx]
    return out


# --------------------------------------------------------------------------
# Sharpe ratio CI
# --------------------------------------------------------------------------


def _sharpe(returns: NDArray[np.float64], periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe of `returns`. Returns 0 when std is degenerate."""
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(returns)) / sd * math.sqrt(periods_per_year)


def bootstrap_sharpe_ci(
    returns: Sequence[float] | NDArray[np.float64],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    p: float | None = None,
    periods_per_year: float = 252.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return (point_sharpe, lower, upper) where (lower, upper) is the
    `confidence`-level percentile interval from a stationary bootstrap.

    For autocorrelated returns this CI is wider — i.e. more honest —
    than the IID-bootstrap analogue.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    arr = np.asarray(returns, dtype=np.float64)
    point = _sharpe(arr, periods_per_year)
    paths = stationary_bootstrap(arr, p=p, n_resamples=n_resamples, seed=seed)
    sharpes = np.array([_sharpe(p_, periods_per_year) for p_ in paths])
    alpha = 1.0 - confidence
    lo = float(np.quantile(sharpes, alpha / 2.0))
    hi = float(np.quantile(sharpes, 1.0 - alpha / 2.0))
    return point, lo, hi

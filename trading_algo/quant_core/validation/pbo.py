"""
Probability of Backtest Overfitting (PBO)

Implements the PBO methodology from:
    Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. (2014).
    "The Probability of Backtest Overfitting"
    Journal of Computational Finance, 17(4), 47-79.

Also includes:
    - Deflated Sharpe Ratio (DSR)
    - Multiple Testing Corrections
    - Minimum Backtest Length (MinBL)

Key Insights:
    - More parameter combinations tested → higher false discovery rate
    - Short backtests → unreliable results
    - Selection bias from choosing best strategy

References:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
    - Harvey, Liu & Zhu (2016): "...and the Cross-Section of Expected Returns"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import comb
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from itertools import combinations


@dataclass
class PBOResult:
    """Result of PBO calculation."""
    pbo: float                      # Probability of Backtest Overfitting
    pbo_std: float                  # Standard error of PBO estimate
    logits: NDArray[np.float64]     # Distribution of logits
    is_performance: NDArray[np.float64]   # In-sample performance
    oos_performance: NDArray[np.float64]  # Out-of-sample performance
    rank_correlation: float         # Spearman correlation


class PBOCalculator:
    """
    Calculate Probability of Backtest Overfitting.

    PBO measures the probability that a strategy selected based on
    in-sample performance will have negative out-of-sample performance.

    Algorithm:
        1. Partition data into N non-overlapping groups
        2. For each combination C(N, N/2):
           - Use half as "in-sample", half as "out-of-sample"
           - Find best strategy in-sample
           - Record its out-of-sample rank
        3. PBO = P(OOS rank <= N/2) across all combinations

    Usage:
        calculator = PBOCalculator(n_groups=16)
        result = calculator.calculate(strategy_returns_matrix)
        print(f"PBO: {result.pbo:.2%}")
    """

    def __init__(
        self,
        n_groups: int = 16,
        metric: str = "sharpe",
    ):
        """
        Initialize PBO calculator.

        Args:
            n_groups: Number of groups to partition data into
                (should be even, typically 10-20)
            metric: Performance metric ('sharpe', 'return', 'sortino')
        """
        if n_groups % 2 != 0:
            n_groups += 1  # Make even
        self.n_groups = n_groups
        self.metric = metric

    def calculate(
        self,
        returns: NDArray[np.float64],
    ) -> PBOResult:
        """
        Calculate PBO for a single strategy.

        For a single strategy, we use the approach from CSCV:
        partition the returns and calculate performance across
        different train/test combinations.

        Args:
            returns: Strategy returns (T,)

        Returns:
            PBOResult with PBO estimate
        """
        n_samples = len(returns)

        # Partition into groups
        group_size = n_samples // self.n_groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = start + group_size if i < self.n_groups - 1 else n_samples
            groups.append(returns[start:end])

        # Generate all combinations of IS/OOS splits
        n_is = self.n_groups // 2
        n_combinations = int(comb(self.n_groups, n_is))

        is_performances = []
        oos_performances = []
        logits = []

        # Sample combinations if too many
        max_combinations = 1000
        if n_combinations > max_combinations:
            # Random sample of combinations
            all_indices = list(range(self.n_groups))
            sampled_combos = []
            for _ in range(max_combinations):
                np.random.shuffle(all_indices)
                sampled_combos.append(tuple(sorted(all_indices[:n_is])))
            is_indices_list = list(set(sampled_combos))
        else:
            is_indices_list = list(combinations(range(self.n_groups), n_is))

        for is_indices in is_indices_list:
            oos_indices = [i for i in range(self.n_groups) if i not in is_indices]

            # Combine groups
            is_returns = np.concatenate([groups[i] for i in is_indices])
            oos_returns = np.concatenate([groups[i] for i in oos_indices])

            # Calculate performance
            is_perf = self._calculate_metric(is_returns)
            oos_perf = self._calculate_metric(oos_returns)

            is_performances.append(is_perf)
            oos_performances.append(oos_perf)

            # Logit: log(rank_IS / rank_OOS)
            # For single strategy, use sign of OOS performance
            if oos_perf > 0:
                logits.append(1.0)  # Good: IS positive, OOS positive
            else:
                logits.append(-1.0)  # Bad: OOS negative

        is_performances = np.array(is_performances)
        oos_performances = np.array(oos_performances)
        logits = np.array(logits)

        # PBO = fraction of negative OOS performance
        pbo = np.mean(oos_performances <= 0)
        pbo_std = np.std(oos_performances <= 0) / np.sqrt(len(oos_performances))

        # Rank correlation
        if len(is_performances) >= 3:
            rank_corr, _ = stats.spearmanr(is_performances, oos_performances)
        else:
            rank_corr = 0.0

        return PBOResult(
            pbo=float(pbo),
            pbo_std=float(pbo_std),
            logits=logits,
            is_performance=is_performances,
            oos_performance=oos_performances,
            rank_correlation=float(rank_corr) if not np.isnan(rank_corr) else 0.0,
        )

    def calculate_multi_strategy(
        self,
        returns_matrix: NDArray[np.float64],
    ) -> PBOResult:
        """
        Calculate PBO for multiple strategies.

        This is the full PBO methodology: given multiple strategies,
        calculate the probability that the best in-sample strategy
        has negative out-of-sample performance.

        Args:
            returns_matrix: Strategy returns (T, N) where N is number of strategies

        Returns:
            PBOResult
        """
        n_samples, n_strategies = returns_matrix.shape

        if n_strategies < 2:
            # Single strategy, use simpler method
            return self.calculate(returns_matrix[:, 0])

        # Partition into groups
        group_size = n_samples // self.n_groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = start + group_size if i < self.n_groups - 1 else n_samples
            groups.append(returns_matrix[start:end, :])

        # Generate combinations
        n_is = self.n_groups // 2
        n_combinations = int(comb(self.n_groups, n_is))

        logits = []
        is_performances = []
        oos_performances = []

        # Sample if too many combinations
        max_combinations = 1000
        if n_combinations > max_combinations:
            all_indices = list(range(self.n_groups))
            sampled_combos = set()
            while len(sampled_combos) < max_combinations:
                np.random.shuffle(all_indices)
                sampled_combos.add(tuple(sorted(all_indices[:n_is])))
            is_indices_list = list(sampled_combos)
        else:
            is_indices_list = list(combinations(range(self.n_groups), n_is))

        for is_indices in is_indices_list:
            oos_indices = [i for i in range(self.n_groups) if i not in is_indices]

            # Combine groups
            is_data = np.vstack([groups[i] for i in is_indices])
            oos_data = np.vstack([groups[i] for i in oos_indices])

            # Calculate performance for each strategy
            is_perf = np.array([self._calculate_metric(is_data[:, j])
                                for j in range(n_strategies)])
            oos_perf = np.array([self._calculate_metric(oos_data[:, j])
                                 for j in range(n_strategies)])

            # Find best IS strategy
            best_is_idx = np.argmax(is_perf)
            best_is_perf = is_perf[best_is_idx]
            best_oos_perf = oos_perf[best_is_idx]

            # OOS rank of the IS-winner among the N strategies. Convention
            # (Bailey, Borwein, Lopez de Prado, Zhu 2017):
            #     omega = rank_OOS(best_IS) / (N + 1)        in (0, 1)
            #     logit = log(omega / (1 - omega))
            #     PBO  = P(rank_OOS(best_IS) > N/2)
            #          = P(omega > 0.5)
            #          = P(logit > 0)
            # Earlier code returned `mean(logit < 0)` (the inverse) which
            # systematically *under*-reported PBO. Fixed below.
            #
            # `argsort(argsort(x))[i]` returns the 0-indexed rank of x[i]
            # in ascending order. We want descending order (best=rank 1),
            # so subtract from N.
            oos_rank_desc = n_strategies - np.argsort(np.argsort(oos_perf))[best_is_idx]
            omega = oos_rank_desc / (n_strategies + 1)
            # Clip omega to (eps, 1-eps) so logit is finite.
            eps = 1e-10
            omega = float(min(max(omega, eps), 1.0 - eps))
            logit = float(np.log(omega / (1.0 - omega)))

            logits.append(logit)
            is_performances.append(best_is_perf)
            oos_performances.append(best_oos_perf)

        logits = np.array(logits)
        is_performances = np.array(is_performances)
        oos_performances = np.array(oos_performances)

        # Canonical PBO: probability the IS winner is in the *bottom half*
        # of OOS ranks. omega > 0.5  <=>  logit > 0.
        pbo = float(np.mean(logits > 0))
        pbo_std = float(np.std(logits > 0)) / float(np.sqrt(len(logits)))

        # Rank correlation
        rank_corr, _ = stats.spearmanr(is_performances, oos_performances)

        return PBOResult(
            pbo=float(pbo),
            pbo_std=float(pbo_std),
            logits=logits,
            is_performance=is_performances,
            oos_performance=oos_performances,
            rank_correlation=float(rank_corr) if not np.isnan(rank_corr) else 0.0,
        )

    def _calculate_metric(self, returns: NDArray[np.float64]) -> float:
        """Calculate performance metric."""
        if len(returns) < 2:
            return 0.0

        if self.metric == "sharpe":
            std = np.std(returns, ddof=1)
            if std < 1e-10:
                return 0.0
            return float(np.mean(returns) / std * np.sqrt(252))

        elif self.metric == "return":
            return float(np.mean(returns) * 252)

        elif self.metric == "sortino":
            downside = returns[returns < 0]
            if len(downside) < 2:
                return float(np.mean(returns) * 252 * 100)  # High if no downside
            downside_std = np.std(downside, ddof=1)
            if downside_std < 1e-10:
                return float(np.mean(returns) * 252 * 100)
            return float(np.mean(returns) / downside_std * np.sqrt(252))

        return 0.0


@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio calculation."""
    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    p_value: float
    is_significant: bool
    haircut: float  # Percentage reduction from observed


class DeflatedSharpe:
    """
    Deflated Sharpe Ratio calculator.

    Adjusts Sharpe ratio for multiple testing, accounting for:
    1. Number of strategies/parameters tested
    2. Length of backtest
    3. Correlation among trials

    From: Bailey & López de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias"

    Usage:
        dsr = DeflatedSharpe()
        result = dsr.calculate(
            observed_sharpe=1.5,
            n_trials=100,
            n_observations=252*5,
        )
    """

    def __init__(
        self,
        significance_level: float = 0.05,
    ):
        """
        Initialize calculator.

        Args:
            significance_level: For significance testing
        """
        self.significance_level = significance_level

    def calculate(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        correlation: float = 0.0,
    ) -> DeflatedSharpeResult:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            observed_sharpe: Observed annualized Sharpe ratio
            n_trials: Number of strategy variations tested
            n_observations: Number of return observations
            skewness: Return skewness (0 for normal)
            kurtosis: Return kurtosis (3 for normal)
            correlation: Average correlation among trials

        Returns:
            DeflatedSharpeResult
        """
        # Expected maximum Sharpe under null hypothesis
        # E[max(Z_1,...,Z_n)] ≈ sqrt(2*log(n)) * (1 - γ/(2*log(n)))
        # where γ ≈ 0.5772 (Euler-Mascheroni constant)
        euler = 0.5772

        if n_trials <= 1:
            expected_max = 0.0
        else:
            # Adjust for correlation
            effective_trials = n_trials * (1 - correlation) + correlation
            effective_trials = max(1, effective_trials)

            log_n = np.log(effective_trials)
            expected_max = np.sqrt(2 * log_n) * (1 - euler / (2 * log_n))

        # Variance of Sharpe ratio estimator
        # Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / T
        var_sr = (
            1 + 0.5 * observed_sharpe**2
            - skewness * observed_sharpe
            + (kurtosis - 3) / 4 * observed_sharpe**2
        ) / n_observations

        std_sr = np.sqrt(var_sr)

        # Deflated Sharpe Ratio
        if std_sr > 1e-10:
            dsr = (observed_sharpe - expected_max * std_sr) / std_sr
        else:
            dsr = observed_sharpe

        # p-value (probability of observing SR >= observed under null)
        p_value = 1 - stats.norm.cdf(dsr)

        # Haircut (percentage reduction)
        if abs(observed_sharpe) > 1e-10:
            haircut = 1 - dsr / observed_sharpe if dsr > 0 else 1.0
        else:
            haircut = 1.0

        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=dsr,
            expected_max_sharpe=expected_max * std_sr,
            p_value=p_value,
            is_significant=p_value < self.significance_level,
            haircut=haircut,
        )

    def minimum_backtest_length(
        self,
        target_sharpe: float,
        n_trials: int,
        significance_level: Optional[float] = None,
    ) -> int:
        """
        Calculate minimum backtest length for reliable results.

        How many observations needed to detect a true Sharpe ratio
        with statistical significance?

        Args:
            target_sharpe: Expected true Sharpe ratio
            n_trials: Number of strategy variations
            significance_level: Override default significance

        Returns:
            Minimum number of observations (days)
        """
        if significance_level is None:
            significance_level = self.significance_level

        # z-score for significance
        z = stats.norm.ppf(1 - significance_level)

        # Expected max Sharpe under null
        expected_max = np.sqrt(2 * np.log(max(1, n_trials)))

        # MinBL ≈ (z + expected_max)^2 / SR^2
        if abs(target_sharpe) < 1e-10:
            return int(1e6)  # Very long if SR is zero

        min_bl = ((z + expected_max) / target_sharpe) ** 2

        return int(np.ceil(min_bl))


class MultipleTestingCorrection:
    """
    Multiple testing corrections for strategy selection.

    Implements:
    - Bonferroni: Conservative, controls family-wise error rate (FWER)
    - Holm-Bonferroni: Step-down Bonferroni
    - Benjamini-Hochberg: Controls false discovery rate (FDR)
    - Storey's q-value: Positive FDR control
    """

    @staticmethod
    def bonferroni(
        p_values: NDArray[np.float64],
        alpha: float = 0.05,
    ) -> NDArray[np.bool_]:
        """
        Bonferroni correction.

        Most conservative. Reject if p < α/n.
        """
        n = len(p_values)
        threshold = alpha / n
        return p_values < threshold

    @staticmethod
    def holm(
        p_values: NDArray[np.float64],
        alpha: float = 0.05,
    ) -> NDArray[np.bool_]:
        """
        Holm-Bonferroni step-down procedure.

        Less conservative than Bonferroni.
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        reject = np.zeros(n, dtype=bool)

        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            threshold = alpha / (n - i)
            if p < threshold:
                reject[idx] = True
            else:
                break  # Stop at first non-rejection

        return reject

    @staticmethod
    def benjamini_hochberg(
        p_values: NDArray[np.float64],
        alpha: float = 0.05,
    ) -> NDArray[np.bool_]:
        """
        Benjamini-Hochberg procedure for FDR control.

        Controls expected proportion of false discoveries.
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Find largest k where p_(k) <= k/n * α
        thresholds = (np.arange(1, n + 1) / n) * alpha
        below_threshold = sorted_p <= thresholds

        if not np.any(below_threshold):
            return np.zeros(n, dtype=bool)

        k = np.max(np.where(below_threshold)[0]) + 1

        reject = np.zeros(n, dtype=bool)
        reject[sorted_indices[:k]] = True

        return reject

    @staticmethod
    def adjusted_p_values(
        p_values: NDArray[np.float64],
        method: str = "bh",
    ) -> NDArray[np.float64]:
        """
        Calculate adjusted p-values.

        Args:
            p_values: Original p-values
            method: 'bonferroni', 'holm', or 'bh' (Benjamini-Hochberg)

        Returns:
            Adjusted p-values
        """
        n = len(p_values)

        if method == "bonferroni":
            return np.minimum(p_values * n, 1.0)

        elif method == "holm":
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            adjusted = np.zeros(n)
            for i in range(n):
                adjusted[i] = (n - i) * sorted_p[i]

            # Enforce monotonicity
            for i in range(1, n):
                adjusted[i] = max(adjusted[i], adjusted[i-1])

            # Restore original order
            result = np.zeros(n)
            result[sorted_indices] = np.minimum(adjusted, 1.0)
            return result

        elif method == "bh":
            sorted_indices = np.argsort(p_values)[::-1]  # Descending
            sorted_p = p_values[sorted_indices]

            adjusted = np.zeros(n)
            adjusted[0] = sorted_p[0]

            for i in range(1, n):
                adjusted[i] = min(adjusted[i-1], sorted_p[i] * n / (n - i))

            # Restore original order
            result = np.zeros(n)
            result[sorted_indices] = np.minimum(adjusted, 1.0)
            return result

        return p_values

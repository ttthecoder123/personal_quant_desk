"""
Monte Carlo Validation Module

Implements Monte Carlo-based validation methods including:
- Permutation tests for strategy significance
- Bootstrap confidence intervals
- Randomization tests
- Block bootstrap for time series
- Strategy comparison via Monte Carlo
- P-value calculation
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo validation results."""

    test_name: str
    observed_statistic: float
    null_distribution: np.ndarray
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    details: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'observed_statistic': self.observed_statistic,
            'p_value': self.p_value,
            'confidence_interval': self.confidence_interval,
            'is_significant': self.is_significant,
            'null_distribution_mean': float(np.mean(self.null_distribution)),
            'null_distribution_std': float(np.std(self.null_distribution)),
            'details': self.details
        }


class MonteCarloValidator:
    """
    Monte Carlo-based validation for backtesting strategies.

    Implements resampling and randomization methods to assess
    statistical significance and robustness of strategy performance.
    """

    def __init__(self, n_simulations: int = 10000, confidence_level: float = 0.95, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo validator.

        Args:
            n_simulations: Number of Monte Carlo simulations (default 10000)
            confidence_level: Confidence level for intervals (default 0.95)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"Initialized MonteCarloValidator with {n_simulations} simulations, "
                   f"{confidence_level:.1%} confidence")

    def permutation_test(
        self,
        returns: np.ndarray,
        statistic_func: Callable[[np.ndarray], float],
        alternative: str = 'greater'
    ) -> MonteCarloResult:
        """
        Permutation test for strategy significance.

        Tests whether the observed statistic is significantly different from
        what would be expected by random chance by randomly permuting returns.

        Args:
            returns: Array of returns
            statistic_func: Function to calculate test statistic
            alternative: 'greater', 'less', or 'two-sided'

        Returns:
            MonteCarloResult with test results
        """
        logger.debug(f"Running permutation test ({self.n_simulations} permutations)")

        # Calculate observed statistic
        observed_stat = statistic_func(returns)

        # Generate null distribution by permuting returns
        null_distribution = []
        for _ in range(self.n_simulations):
            permuted_returns = np.random.permutation(returns)
            null_stat = statistic_func(permuted_returns)
            null_distribution.append(null_stat)

        null_distribution = np.array(null_distribution)

        # Calculate p-value
        if alternative == 'greater':
            p_value = np.mean(null_distribution >= observed_stat)
        elif alternative == 'less':
            p_value = np.mean(null_distribution <= observed_stat)
        else:  # two-sided
            p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))

        # Confidence interval from null distribution
        ci_lower = np.percentile(null_distribution, self.alpha / 2 * 100)
        ci_upper = np.percentile(null_distribution, (1 - self.alpha / 2) * 100)

        is_significant = p_value < self.alpha

        conclusion = (
            f"Observed statistic {observed_stat:.4f} is "
            f"{'significantly different' if is_significant else 'not significantly different'} "
            f"from random (p={p_value:.4f})"
        )

        return MonteCarloResult(
            test_name="Permutation Test",
            observed_statistic=observed_stat,
            null_distribution=null_distribution,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            details={
                'alternative': alternative,
                'n_simulations': self.n_simulations,
                'conclusion': conclusion
            }
        )

    def bootstrap_confidence_interval(
        self,
        returns: np.ndarray,
        statistic_func: Callable[[np.ndarray], float],
        method: str = 'percentile'
    ) -> MonteCarloResult:
        """
        Bootstrap confidence interval for a statistic.

        Args:
            returns: Array of returns
            statistic_func: Function to calculate statistic
            method: 'percentile', 'bca' (bias-corrected accelerated), or 'basic'

        Returns:
            MonteCarloResult with bootstrap results
        """
        logger.debug(f"Calculating bootstrap CI ({self.n_simulations} samples)")

        n = len(returns)

        # Calculate observed statistic
        observed_stat = statistic_func(returns)

        # Generate bootstrap distribution
        bootstrap_stats = []
        for _ in range(self.n_simulations):
            # Resample with replacement
            bootstrap_sample = np.random.choice(returns, size=n, replace=True)
            boot_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(boot_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        if method == 'percentile':
            # Percentile method
            ci_lower = np.percentile(bootstrap_stats, self.alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)

        elif method == 'basic':
            # Basic bootstrap method
            ci_lower = 2 * observed_stat - np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)
            ci_upper = 2 * observed_stat - np.percentile(bootstrap_stats, self.alpha / 2 * 100)

        elif method == 'bca':
            # Bias-corrected and accelerated (BCa) method
            # Calculate bias correction
            z0 = stats.norm.ppf(np.mean(bootstrap_stats < observed_stat))

            # Calculate acceleration (using jackknife)
            jackknife_stats = []
            for i in range(n):
                jackknife_sample = np.delete(returns, i)
                jack_stat = statistic_func(jackknife_sample)
                jackknife_stats.append(jack_stat)

            jackknife_mean = np.mean(jackknife_stats)
            numerator = np.sum((jackknife_mean - np.array(jackknife_stats)) ** 3)
            denominator = 6 * (np.sum((jackknife_mean - np.array(jackknife_stats)) ** 2) ** 1.5)
            acceleration = numerator / denominator if denominator != 0 else 0

            # Adjusted percentiles
            z_lower = stats.norm.ppf(self.alpha / 2)
            z_upper = stats.norm.ppf(1 - self.alpha / 2)

            p_lower = stats.norm.cdf(z0 + (z0 + z_lower) / (1 - acceleration * (z0 + z_lower)))
            p_upper = stats.norm.cdf(z0 + (z0 + z_upper) / (1 - acceleration * (z0 + z_upper)))

            ci_lower = np.percentile(bootstrap_stats, p_lower * 100)
            ci_upper = np.percentile(bootstrap_stats, p_upper * 100)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Standard error
        bootstrap_se = np.std(bootstrap_stats, ddof=1)

        conclusion = (
            f"Bootstrap {self.confidence_level:.0%} CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"SE: {bootstrap_se:.4f}"
        )

        return MonteCarloResult(
            test_name=f"Bootstrap CI ({method})",
            observed_statistic=observed_stat,
            null_distribution=bootstrap_stats,
            p_value=0.0,  # Not applicable for CI
            confidence_interval=(ci_lower, ci_upper),
            is_significant=False,  # Not applicable for CI
            details={
                'method': method,
                'standard_error': bootstrap_se,
                'n_simulations': self.n_simulations,
                'conclusion': conclusion
            }
        )

    def randomization_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        statistic_func: Callable[[np.ndarray, np.ndarray], float]
    ) -> MonteCarloResult:
        """
        Randomization test for comparing two groups.

        Tests whether the difference between two groups is significant
        by randomly reassigning observations to groups.

        Args:
            group1: First group of observations
            group2: Second group of observations
            statistic_func: Function that takes two groups and returns statistic

        Returns:
            MonteCarloResult with test results
        """
        logger.debug(f"Running randomization test ({self.n_simulations} randomizations)")

        # Calculate observed statistic
        observed_stat = statistic_func(group1, group2)

        # Combine groups
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        n_total = len(combined)

        # Generate null distribution by random assignment
        null_distribution = []
        for _ in range(self.n_simulations):
            # Randomly shuffle and split
            shuffled = np.random.permutation(combined)
            random_group1 = shuffled[:n1]
            random_group2 = shuffled[n1:]

            null_stat = statistic_func(random_group1, random_group2)
            null_distribution.append(null_stat)

        null_distribution = np.array(null_distribution)

        # Calculate p-value (two-sided)
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))

        # Confidence interval
        ci_lower = np.percentile(null_distribution, self.alpha / 2 * 100)
        ci_upper = np.percentile(null_distribution, (1 - self.alpha / 2) * 100)

        is_significant = p_value < self.alpha

        conclusion = (
            f"Observed difference {observed_stat:.4f} is "
            f"{'statistically significant' if is_significant else 'not statistically significant'} "
            f"(p={p_value:.4f})"
        )

        return MonteCarloResult(
            test_name="Randomization Test",
            observed_statistic=observed_stat,
            null_distribution=null_distribution,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            details={
                'n_group1': len(group1),
                'n_group2': len(group2),
                'n_simulations': self.n_simulations,
                'conclusion': conclusion
            }
        )

    def block_bootstrap(
        self,
        returns: np.ndarray,
        statistic_func: Callable[[np.ndarray], float],
        block_length: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Block bootstrap for time series data.

        Preserves temporal dependence structure by resampling blocks
        instead of individual observations.

        Args:
            returns: Array of returns (time series)
            statistic_func: Function to calculate statistic
            block_length: Length of blocks (default: sqrt(n))

        Returns:
            MonteCarloResult with block bootstrap results
        """
        logger.debug("Running block bootstrap for time series")

        n = len(returns)

        # Default block length: sqrt(n)
        if block_length is None:
            block_length = int(np.sqrt(n))

        # Calculate observed statistic
        observed_stat = statistic_func(returns)

        # Generate bootstrap distribution
        bootstrap_stats = []

        for _ in range(self.n_simulations):
            # Moving block bootstrap
            n_blocks = int(np.ceil(n / block_length))
            bootstrap_sample = []

            for _ in range(n_blocks):
                # Random starting point
                start = np.random.randint(0, n - block_length + 1)
                block = returns[start:start + block_length]
                bootstrap_sample.extend(block)

            # Trim to original length
            bootstrap_sample = np.array(bootstrap_sample[:n])

            boot_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(boot_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Confidence interval
        ci_lower = np.percentile(bootstrap_stats, self.alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)

        # Standard error
        bootstrap_se = np.std(bootstrap_stats, ddof=1)

        conclusion = (
            f"Block bootstrap {self.confidence_level:.0%} CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"SE: {bootstrap_se:.4f} (block length: {block_length})"
        )

        return MonteCarloResult(
            test_name="Block Bootstrap",
            observed_statistic=observed_stat,
            null_distribution=bootstrap_stats,
            p_value=0.0,  # Not applicable for CI
            confidence_interval=(ci_lower, ci_upper),
            is_significant=False,  # Not applicable for CI
            details={
                'block_length': block_length,
                'standard_error': bootstrap_se,
                'n_simulations': self.n_simulations,
                'conclusion': conclusion
            }
        )

    def strategy_comparison_mc(
        self,
        strategy1_returns: np.ndarray,
        strategy2_returns: np.ndarray,
        metric_func: Callable[[np.ndarray], float],
        paired: bool = True
    ) -> MonteCarloResult:
        """
        Compare two strategies using Monte Carlo methods.

        Args:
            strategy1_returns: First strategy returns
            strategy2_returns: Second strategy returns
            metric_func: Function to calculate performance metric
            paired: Whether returns are paired (same time periods)

        Returns:
            MonteCarloResult with comparison results
        """
        logger.debug("Comparing strategies via Monte Carlo")

        # Calculate observed metrics
        metric1 = metric_func(strategy1_returns)
        metric2 = metric_func(strategy2_returns)
        observed_diff = metric1 - metric2

        if paired:
            if len(strategy1_returns) != len(strategy2_returns):
                raise ValueError("Paired comparison requires equal length arrays")

            # Bootstrap paired differences
            n = len(strategy1_returns)
            bootstrap_diffs = []

            for _ in range(self.n_simulations):
                # Resample pairs
                indices = np.random.choice(n, size=n, replace=True)
                boot_returns1 = strategy1_returns[indices]
                boot_returns2 = strategy2_returns[indices]

                boot_metric1 = metric_func(boot_returns1)
                boot_metric2 = metric_func(boot_returns2)
                boot_diff = boot_metric1 - boot_metric2

                bootstrap_diffs.append(boot_diff)

        else:
            # Independent bootstrap
            bootstrap_diffs = []

            for _ in range(self.n_simulations):
                # Resample each strategy independently
                boot_returns1 = np.random.choice(strategy1_returns, size=len(strategy1_returns), replace=True)
                boot_returns2 = np.random.choice(strategy2_returns, size=len(strategy2_returns), replace=True)

                boot_metric1 = metric_func(boot_returns1)
                boot_metric2 = metric_func(boot_returns2)
                boot_diff = boot_metric1 - boot_metric2

                bootstrap_diffs.append(boot_diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value: proportion of bootstrap samples where strategy2 is better
        p_value = np.mean(bootstrap_diffs <= 0)

        # Confidence interval for difference
        ci_lower = np.percentile(bootstrap_diffs, self.alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha / 2) * 100)

        is_significant = ci_lower > 0 or ci_upper < 0

        conclusion = (
            f"Strategy 1 metric: {metric1:.4f}, Strategy 2 metric: {metric2:.4f}, "
            f"Difference: {observed_diff:.4f} "
            f"({'statistically significant' if is_significant else 'not statistically significant'})"
        )

        return MonteCarloResult(
            test_name=f"Strategy Comparison ({'paired' if paired else 'independent'})",
            observed_statistic=observed_diff,
            null_distribution=bootstrap_diffs,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            details={
                'strategy1_metric': metric1,
                'strategy2_metric': metric2,
                'paired': paired,
                'n_simulations': self.n_simulations,
                'conclusion': conclusion
            }
        )

    def calculate_p_value(
        self,
        observed_statistic: float,
        null_distribution: np.ndarray,
        alternative: str = 'two-sided'
    ) -> float:
        """
        Calculate p-value from null distribution.

        Args:
            observed_statistic: Observed test statistic
            null_distribution: Null distribution from Monte Carlo
            alternative: 'greater', 'less', or 'two-sided'

        Returns:
            P-value
        """
        if alternative == 'greater':
            p_value = np.mean(null_distribution >= observed_statistic)
        elif alternative == 'less':
            p_value = np.mean(null_distribution <= observed_statistic)
        else:  # two-sided
            p_value = np.mean(np.abs(null_distribution - np.mean(null_distribution)) >=
                            np.abs(observed_statistic - np.mean(null_distribution)))

        return p_value

    def sharpe_ratio_monte_carlo(
        self,
        returns: np.ndarray,
        n_years: float = 1.0
    ) -> MonteCarloResult:
        """
        Monte Carlo test for Sharpe ratio significance.

        Tests whether observed Sharpe ratio is significantly different from zero.

        Args:
            returns: Array of returns
            n_years: Number of years of data (for annualization)

        Returns:
            MonteCarloResult with Sharpe ratio test results
        """
        logger.debug("Running Sharpe ratio Monte Carlo test")

        # Calculate observed Sharpe ratio
        periods_per_year = len(returns) / n_years
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        observed_sharpe = mean_return / std_return * np.sqrt(periods_per_year) if std_return > 0 else 0

        # Permutation test
        def sharpe_func(rets):
            m = np.mean(rets)
            s = np.std(rets, ddof=1)
            return m / s * np.sqrt(periods_per_year) if s > 0 else 0

        result = self.permutation_test(returns, sharpe_func, alternative='greater')

        result.test_name = "Sharpe Ratio Monte Carlo Test"
        result.details['annualized_sharpe'] = observed_sharpe
        result.details['n_years'] = n_years

        return result

    def maximum_drawdown_monte_carlo(
        self,
        returns: np.ndarray
    ) -> MonteCarloResult:
        """
        Monte Carlo test for maximum drawdown.

        Tests whether observed MDD is consistent with random returns.

        Args:
            returns: Array of returns

        Returns:
            MonteCarloResult with MDD test results
        """
        logger.debug("Running maximum drawdown Monte Carlo test")

        # Calculate observed MDD
        def mdd_func(rets):
            cum_returns = np.cumprod(1 + rets)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            return np.min(drawdown)

        observed_mdd = mdd_func(returns)

        # Permutation test (worse MDD = more negative)
        result = self.permutation_test(returns, mdd_func, alternative='greater')

        result.test_name = "Maximum Drawdown Monte Carlo Test"
        result.details['observed_mdd'] = observed_mdd

        return result

    def run_monte_carlo_suite(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, MonteCarloResult]:
        """
        Run comprehensive Monte Carlo validation suite.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary of Monte Carlo test results
        """
        logger.info("Running comprehensive Monte Carlo validation suite")

        results = {}

        # Sharpe ratio test
        try:
            results['sharpe_ratio'] = self.sharpe_ratio_monte_carlo(returns)
            logger.info("Completed Sharpe ratio Monte Carlo test")
        except Exception as e:
            logger.error(f"Sharpe ratio test failed: {e}")

        # Maximum drawdown test
        try:
            results['maximum_drawdown'] = self.maximum_drawdown_monte_carlo(returns)
            logger.info("Completed maximum drawdown Monte Carlo test")
        except Exception as e:
            logger.error(f"Maximum drawdown test failed: {e}")

        # Bootstrap CI for mean return
        try:
            results['mean_return_ci'] = self.bootstrap_confidence_interval(
                returns,
                lambda x: np.mean(x) * 252  # Annualized
            )
            logger.info("Completed mean return bootstrap CI")
        except Exception as e:
            logger.error(f"Mean return CI failed: {e}")

        # Block bootstrap for Sharpe ratio (accounting for autocorrelation)
        try:
            periods_per_year = 252
            results['sharpe_block_bootstrap'] = self.block_bootstrap(
                returns,
                lambda x: np.mean(x) / np.std(x, ddof=1) * np.sqrt(periods_per_year) if np.std(x, ddof=1) > 0 else 0
            )
            logger.info("Completed Sharpe ratio block bootstrap")
        except Exception as e:
            logger.error(f"Sharpe block bootstrap failed: {e}")

        # Strategy comparison (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            try:
                results['vs_benchmark'] = self.strategy_comparison_mc(
                    returns,
                    benchmark_returns,
                    lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0,
                    paired=True
                )
                logger.info("Completed strategy comparison")
            except Exception as e:
                logger.error(f"Strategy comparison failed: {e}")

        logger.info(f"Completed {len(results)} Monte Carlo tests")
        return results

"""
Overfitting Detection Module

Implements López de Prado's comprehensive overfitting detection methods including:
- Bailey's Sharpe ratio deflation
- Combinatorial Purged Cross-Validation (CPCV)
- Multiple testing corrections
- Strategy complexity penalties
- Probability of Backtest Overfitting (PBO)
- False Discovery Rate control
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm


@dataclass
class OverfittingResult:
    """Container for overfitting detection results."""

    test_name: str
    metric_value: float
    threshold: float
    is_overfit: bool
    confidence_level: float
    details: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'is_overfit': self.is_overfit,
            'confidence_level': self.confidence_level,
            'details': self.details
        }


class OverfittingDetector:
    """
    Comprehensive overfitting detection using López de Prado methods.

    Implements state-of-the-art techniques from "Advances in Financial Machine Learning"
    to detect and quantify overfitting in backtested strategies.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize overfitting detector.

        Args:
            confidence_level: Confidence level for tests (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        logger.info(f"Initialized OverfittingDetector with {confidence_level:.1%} confidence")

    def deflated_sharpe_ratio(
        self,
        sharpe_ratio: float,
        n_observations: int,
        n_trials: int,
        skewness: float = 0,
        kurtosis: float = 3,
        benchmark_sharpe: float = 0
    ) -> OverfittingResult:
        """
        Calculate deflated Sharpe ratio (Bailey & López de Prado).

        Adjusts Sharpe ratio for multiple testing and selection bias.

        Args:
            sharpe_ratio: Observed Sharpe ratio
            n_observations: Number of observations
            n_trials: Number of strategies tested
            skewness: Return skewness
            kurtosis: Return kurtosis
            benchmark_sharpe: Benchmark Sharpe ratio to test against

        Returns:
            OverfittingResult with deflated Sharpe ratio
        """
        logger.debug(f"Calculating deflated Sharpe ratio for {n_trials} trials")

        # Variance of Sharpe ratio estimator
        variance = (
            1 +
            sharpe_ratio**2 / 2 -
            skewness * sharpe_ratio +
            (kurtosis - 1) * sharpe_ratio**2 / 4
        ) / n_observations

        # Expected maximum Sharpe ratio under null (all strategies have SR=0)
        # Using extreme value theory
        gamma = 0.5772156649  # Euler-Mascheroni constant
        expected_max_sr = (1 - gamma) * norm.ppf(1 - 1/n_trials) + gamma * norm.ppf(1 - 1/(n_trials * np.e))

        # Adjusted variance accounting for selection bias
        variance_max = variance * (1 + variance * expected_max_sr**2)

        # Deflated Sharpe ratio
        deflated_sr = (sharpe_ratio - expected_max_sr * np.sqrt(variance)) / np.sqrt(variance_max)

        # p-value: probability that deflated SR > benchmark
        p_value = norm.cdf(deflated_sr) if deflated_sr < 0 else 1 - norm.cdf(deflated_sr)

        is_overfit = deflated_sr < benchmark_sharpe

        conclusion = (
            f"Deflated SR of {deflated_sr:.3f} "
            f"{'suggests overfitting' if is_overfit else 'does not suggest overfitting'} "
            f"(original SR: {sharpe_ratio:.3f})"
        )

        return OverfittingResult(
            test_name="Deflated Sharpe Ratio",
            metric_value=deflated_sr,
            threshold=benchmark_sharpe,
            is_overfit=is_overfit,
            confidence_level=self.confidence_level,
            details={
                'original_sharpe': sharpe_ratio,
                'deflated_sharpe': deflated_sr,
                'expected_max_sharpe': expected_max_sr,
                'p_value': p_value,
                'n_trials': n_trials,
                'n_observations': n_observations,
                'conclusion': conclusion
            }
        )

    def effective_number_of_tests(
        self,
        correlation_matrix: np.ndarray,
        eigenvalue_threshold: float = 0.95
    ) -> Tuple[float, Dict]:
        """
        Calculate effective number of independent tests (Bailey-López de Prado).

        Accounts for correlation between strategies when calculating
        multiple testing adjustments.

        Args:
            correlation_matrix: Correlation matrix of strategy returns
            eigenvalue_threshold: Threshold for eigenvalue contribution

        Returns:
            Effective number of tests and details
        """
        logger.debug("Calculating effective number of tests")

        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

        # Cumulative variance explained
        total_variance = np.sum(eigenvalues)
        cumulative_variance = np.cumsum(eigenvalues) / total_variance

        # Number of eigenvalues needed to explain threshold variance
        n_effective = np.searchsorted(cumulative_variance, eigenvalue_threshold) + 1

        # Alternative: Bailey's method using all eigenvalues
        # M* = sum(eigenvalues)^2 / sum(eigenvalues^2)
        bailey_effective = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)

        details = {
            'n_strategies': len(correlation_matrix),
            'n_effective_variance': n_effective,
            'n_effective_bailey': bailey_effective,
            'variance_explained': cumulative_variance[n_effective-1] if n_effective > 0 else 0,
            'eigenvalues': eigenvalues.tolist(),
            'cumulative_variance': cumulative_variance.tolist()
        }

        logger.info(f"Effective tests: {bailey_effective:.1f} (from {len(correlation_matrix)} strategies)")

        return bailey_effective, details

    def combinatorial_purged_cv(
        self,
        returns: pd.Series,
        strategy_func: Callable,
        n_splits: int = 10,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01
    ) -> OverfittingResult:
        """
        Combinatorial Purged Cross-Validation (CPCV).

        Implements López de Prado's CPCV to estimate out-of-sample performance
        while accounting for overfitting.

        Args:
            returns: Time series of returns
            strategy_func: Function that takes returns and returns performance metric
            n_splits: Number of splits for cross-validation
            n_test_splits: Number of splits to use as test set
            embargo_pct: Percentage of data to embargo after test set

        Returns:
            OverfittingResult with CPCV metrics
        """
        logger.debug(f"Running CPCV with {n_splits} splits, {n_test_splits} test splits")

        n = len(returns)
        split_size = n // n_splits
        indices = np.arange(n)

        # Create split indices
        splits = []
        for i in range(n_splits):
            start = i * split_size
            end = start + split_size if i < n_splits - 1 else n
            splits.append(indices[start:end])

        # Generate all combinations of test sets
        test_combinations = list(combinations(range(n_splits), n_test_splits))

        in_sample_performances = []
        out_sample_performances = []

        for test_indices in test_combinations:
            # Create test set
            test_mask = np.zeros(n, dtype=bool)
            for idx in test_indices:
                test_mask[splits[idx]] = True

            # Apply embargo: remove data immediately after test set
            embargo_size = int(n * embargo_pct)
            for idx in test_indices:
                if idx < n_splits - 1:
                    end_idx = splits[idx][-1]
                    embargo_end = min(end_idx + embargo_size, n)
                    test_mask[end_idx:embargo_end] = True

            # Training set is everything not in test
            train_mask = ~test_mask

            # Calculate performance on both sets
            train_returns = returns[train_mask]
            test_returns = returns[test_mask]

            if len(train_returns) > 0 and len(test_returns) > 0:
                is_perf = strategy_func(train_returns)
                oos_perf = strategy_func(test_returns)

                in_sample_performances.append(is_perf)
                out_sample_performances.append(oos_perf)

        is_performances = np.array(in_sample_performances)
        oos_performances = np.array(out_sample_performances)

        # Calculate degradation
        mean_is = np.mean(is_performances)
        mean_oos = np.mean(oos_performances)
        degradation = (mean_is - mean_oos) / abs(mean_is) if mean_is != 0 else 0

        # Statistical test: is OOS significantly worse than IS?
        if len(is_performances) > 1:
            t_stat, p_value = stats.ttest_rel(is_performances, oos_performances)
        else:
            t_stat, p_value = 0, 1.0

        # Consider overfit if degradation > 20% or p_value < 0.05
        is_overfit = degradation > 0.2 or (p_value < 0.05 and mean_oos < mean_is)

        conclusion = (
            f"Performance degradation of {degradation:.1%} from IS to OOS "
            f"{'suggests overfitting' if is_overfit else 'acceptable'}"
        )

        return OverfittingResult(
            test_name="Combinatorial Purged Cross-Validation",
            metric_value=degradation,
            threshold=0.2,
            is_overfit=is_overfit,
            confidence_level=self.confidence_level,
            details={
                'mean_in_sample': mean_is,
                'mean_out_sample': mean_oos,
                'degradation': degradation,
                'n_combinations': len(test_combinations),
                'p_value': p_value,
                't_statistic': t_stat,
                'conclusion': conclusion
            }
        )

    def probability_of_backtest_overfitting(
        self,
        is_performances: np.ndarray,
        oos_performances: np.ndarray,
        n_bootstrap: int = 10000
    ) -> OverfittingResult:
        """
        Probability of Backtest Overfitting (PBO) - López de Prado.

        Calculates the probability that the best in-sample strategy
        performs worse than median out-of-sample.

        Args:
            is_performances: In-sample performances for each configuration
            oos_performances: Out-of-sample performances for each configuration
            n_bootstrap: Number of bootstrap samples

        Returns:
            OverfittingResult with PBO metric
        """
        logger.debug(f"Calculating PBO with {n_bootstrap} bootstrap samples")

        n_configs = len(is_performances)

        if n_configs != len(oos_performances):
            raise ValueError("IS and OOS performances must have same length")

        pbo_count = 0

        for _ in range(n_bootstrap):
            # Bootstrap sample configurations
            sample_idx = np.random.choice(n_configs, size=n_configs, replace=True)

            is_sample = is_performances[sample_idx]
            oos_sample = oos_performances[sample_idx]

            # Find best in-sample configuration
            best_is_idx = np.argmax(is_sample)

            # Get its OOS performance
            best_is_oos_perf = oos_sample[best_is_idx]

            # Compare to median OOS
            median_oos = np.median(oos_sample)

            if best_is_oos_perf < median_oos:
                pbo_count += 1

        pbo = pbo_count / n_bootstrap

        # PBO > 0.5 suggests overfitting
        is_overfit = pbo > 0.5

        conclusion = (
            f"PBO of {pbo:.2%} "
            f"{'suggests significant overfitting' if is_overfit else 'acceptable overfitting risk'}"
        )

        return OverfittingResult(
            test_name="Probability of Backtest Overfitting",
            metric_value=pbo,
            threshold=0.5,
            is_overfit=is_overfit,
            confidence_level=self.confidence_level,
            details={
                'pbo': pbo,
                'n_configurations': n_configs,
                'n_bootstrap': n_bootstrap,
                'best_is_performance': np.max(is_performances),
                'median_oos_performance': np.median(oos_performances),
                'conclusion': conclusion
            }
        )

    def multiple_testing_correction(
        self,
        p_values: np.ndarray,
        method: str = 'bonferroni'
    ) -> Dict[str, np.ndarray]:
        """
        Apply multiple testing corrections.

        Args:
            p_values: Array of p-values
            method: Correction method ('bonferroni', 'holm', 'bh')

        Returns:
            Dictionary with corrected p-values and significance
        """
        logger.debug(f"Applying {method} correction to {len(p_values)} p-values")

        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = p_values[sorted_indices]

        if method == 'bonferroni':
            # Bonferroni correction: multiply by number of tests
            corrected = np.minimum(sorted_pvals * n, 1.0)

        elif method == 'holm':
            # Holm-Bonferroni: step-down method
            corrected = np.zeros(n)
            for i in range(n):
                corrected[i] = min(sorted_pvals[i] * (n - i), 1.0)
                if i > 0:
                    corrected[i] = max(corrected[i], corrected[i-1])

        elif method == 'bh':
            # Benjamini-Hochberg (FDR control)
            corrected = np.zeros(n)
            for i in range(n-1, -1, -1):
                corrected[i] = min(sorted_pvals[i] * n / (i + 1), 1.0)
                if i < n - 1:
                    corrected[i] = min(corrected[i], corrected[i+1])

        else:
            raise ValueError(f"Unknown method: {method}")

        # Reorder to original indices
        result_corrected = np.zeros(n)
        result_corrected[sorted_indices] = corrected

        # Determine significance at various levels
        significant_05 = result_corrected < 0.05
        significant_01 = result_corrected < 0.01

        return {
            'original_p_values': p_values,
            'corrected_p_values': result_corrected,
            'significant_0.05': significant_05,
            'significant_0.01': significant_01,
            'method': method,
            'n_significant_0.05': np.sum(significant_05),
            'n_significant_0.01': np.sum(significant_01)
        }

    def false_discovery_rate(
        self,
        p_values: np.ndarray,
        fdr_level: float = 0.05
    ) -> Dict:
        """
        Control False Discovery Rate using Benjamini-Hochberg procedure.

        Args:
            p_values: Array of p-values
            fdr_level: Desired FDR level

        Returns:
            Dictionary with FDR-controlled results
        """
        logger.debug(f"Controlling FDR at {fdr_level:.1%}")

        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = p_values[sorted_indices]

        # Find largest i where p(i) <= (i/n) * q
        threshold_idx = -1
        for i in range(n):
            if sorted_pvals[i] <= ((i + 1) / n) * fdr_level:
                threshold_idx = i

        # Reject all hypotheses up to threshold_idx
        if threshold_idx >= 0:
            rejected = np.zeros(n, dtype=bool)
            rejected[sorted_indices[:threshold_idx+1]] = True
            threshold_pvalue = sorted_pvals[threshold_idx]
            n_rejected = threshold_idx + 1
        else:
            rejected = np.zeros(n, dtype=bool)
            threshold_pvalue = 0
            n_rejected = 0

        return {
            'rejected': rejected,
            'n_rejected': n_rejected,
            'threshold_p_value': threshold_pvalue,
            'fdr_level': fdr_level,
            'proportion_rejected': n_rejected / n if n > 0 else 0
        }

    def strategy_complexity_penalty(
        self,
        n_parameters: int,
        n_observations: int,
        method: str = 'aic'
    ) -> float:
        """
        Calculate complexity penalty for strategy.

        Args:
            n_parameters: Number of strategy parameters
            n_observations: Number of observations
            method: Penalty method ('aic', 'bic', 'hqic')

        Returns:
            Penalty value
        """
        logger.debug(f"Calculating {method} penalty")

        if method == 'aic':
            # Akaike Information Criterion
            penalty = 2 * n_parameters

        elif method == 'bic':
            # Bayesian Information Criterion
            penalty = n_parameters * np.log(n_observations)

        elif method == 'hqic':
            # Hannan-Quinn Information Criterion
            penalty = 2 * n_parameters * np.log(np.log(n_observations))

        else:
            raise ValueError(f"Unknown method: {method}")

        return penalty

    def adjusted_performance(
        self,
        log_likelihood: float,
        n_parameters: int,
        n_observations: int,
        method: str = 'aic'
    ) -> float:
        """
        Calculate complexity-adjusted performance metric.

        Args:
            log_likelihood: Log-likelihood of strategy returns
            n_parameters: Number of strategy parameters
            n_observations: Number of observations
            method: Adjustment method ('aic', 'bic', 'hqic')

        Returns:
            Adjusted performance (lower is better for IC, higher for log-likelihood)
        """
        penalty = self.strategy_complexity_penalty(n_parameters, n_observations, method)

        if method == 'aic':
            adjusted = -2 * log_likelihood + penalty
        elif method == 'bic':
            adjusted = -2 * log_likelihood + penalty
        elif method == 'hqic':
            adjusted = -2 * log_likelihood + penalty
        else:
            adjusted = log_likelihood - penalty

        return adjusted

    def run_overfitting_analysis(
        self,
        returns: pd.Series,
        sharpe_ratio: float,
        n_trials: int,
        is_performances: Optional[np.ndarray] = None,
        oos_performances: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, OverfittingResult]:
        """
        Run comprehensive overfitting analysis.

        Args:
            returns: Strategy returns
            sharpe_ratio: Observed Sharpe ratio
            n_trials: Number of strategies tested
            is_performances: Optional in-sample performances
            oos_performances: Optional out-of-sample performances
            correlation_matrix: Optional correlation matrix of strategies

        Returns:
            Dictionary of overfitting test results
        """
        logger.info("Running comprehensive overfitting analysis")

        results = {}

        # Deflated Sharpe ratio
        n_obs = len(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=False)
        results['deflated_sharpe'] = self.deflated_sharpe_ratio(
            sharpe_ratio, n_obs, n_trials, skew, kurt
        )

        # Effective number of tests (if correlation matrix provided)
        if correlation_matrix is not None:
            n_eff, details = self.effective_number_of_tests(correlation_matrix)
            results['effective_tests'] = OverfittingResult(
                test_name="Effective Number of Tests",
                metric_value=n_eff,
                threshold=n_trials,
                is_overfit=n_eff < n_trials * 0.5,
                confidence_level=self.confidence_level,
                details=details
            )

        # PBO (if IS/OOS performances provided)
        if is_performances is not None and oos_performances is not None:
            results['pbo'] = self.probability_of_backtest_overfitting(
                is_performances, oos_performances
            )

        logger.info(f"Completed {len(results)} overfitting tests")
        return results

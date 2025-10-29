"""
Combinatorial Purged Cross-Validation (CPCV)

Implements López de Prado's Combinatorial Purged Cross-Validation:
- Generates all possible train/test combinations
- Purges overlapping data to prevent leakage
- Embargoes observations after test sets
- Calculates path probabilities
- Computes Probability of Backtest Overfitting (PBO)
- Combinatorial Symmetric Cross-Validation (CSCV)

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple, Set
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import comb


@dataclass
class CPCVConfig:
    """Configuration for CPCV."""
    n_splits: int = 10  # Number of CV splits
    n_test_splits: int = 2  # Number of splits in test set
    purge_pct: float = 0.02  # Percentage to purge
    embargo_pct: float = 0.01  # Percentage to embargo
    n_paths: Optional[int] = None  # Number of paths for PBO (None = all)


@dataclass
class CPCVSplit:
    """Single CPCV train/test split."""
    split_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_indices: np.ndarray
    embargo_indices: np.ndarray


@dataclass
class CPCVResult:
    """Results from CPCV."""
    train_scores: List[float]
    test_scores: List[float]
    splits: List[CPCVSplit]
    mean_train_score: float
    mean_test_score: float
    std_train_score: float
    std_test_score: float
    overfitting_ratio: float
    pbo: Optional[float] = None  # Probability of Backtest Overfitting
    sharpe_ratio: Optional[float] = None
    deflated_sharpe: Optional[float] = None


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    Implements CPCV from López de Prado's book to properly test
    strategy parameters while preventing overfitting.
    """

    def __init__(
        self,
        config: Optional[CPCVConfig] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize CPCV.

        Args:
            config: CPCV configuration
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.config = config or CPCVConfig()
        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        logger.info(
            f"CombinatorialPurgedCV initialized: {self.config.n_splits} splits, "
            f"{self.config.n_test_splits} test splits"
        )

    def generate_splits(
        self,
        data: pd.DataFrame,
        sample_weight: Optional[pd.Series] = None
    ) -> List[CPCVSplit]:
        """
        Generate all combinatorial train/test splits.

        Args:
            data: Data with DatetimeIndex
            sample_weight: Sample weights (e.g., from uniqueness)

        Returns:
            List of CPCVSplit objects
        """
        n_samples = len(data)
        indices = np.arange(n_samples)

        # Split indices into groups
        split_size = n_samples // self.config.n_splits
        split_indices = [
            indices[i * split_size:(i + 1) * split_size]
            for i in range(self.config.n_splits)
        ]

        # Handle remainder
        remainder = n_samples % self.config.n_splits
        if remainder > 0:
            split_indices[-1] = np.append(
                split_indices[-1],
                indices[-remainder:]
            )

        # Generate all combinations of test splits
        test_combinations = list(
            combinations(range(self.config.n_splits), self.config.n_test_splits)
        )

        logger.info(f"Generating {len(test_combinations)} combinatorial splits")

        # Create train/test splits
        splits = []
        for split_id, test_split_ids in enumerate(test_combinations):
            # Test indices
            test_indices = np.concatenate([split_indices[i] for i in test_split_ids])

            # Train indices (all other splits)
            train_split_ids = [
                i for i in range(self.config.n_splits)
                if i not in test_split_ids
            ]
            train_indices = np.concatenate([split_indices[i] for i in train_split_ids])

            # Apply purging
            purge_indices = self._get_purge_indices(
                data, train_indices, test_indices
            )

            # Apply embargo
            embargo_indices = self._get_embargo_indices(
                data, test_indices
            )

            # Remove purged and embargoed indices from training
            train_indices = np.setdiff1d(
                train_indices,
                np.concatenate([purge_indices, embargo_indices])
            )

            splits.append(CPCVSplit(
                split_id=split_id,
                train_indices=train_indices,
                test_indices=test_indices,
                purge_indices=purge_indices,
                embargo_indices=embargo_indices
            ))

        logger.info(f"Generated {len(splits)} CPCV splits")
        return splits

    def cross_validate(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_generator: Callable,
        params: Dict[str, Any],
        metric: str = 'sharpe_ratio',
        sample_weight: Optional[pd.Series] = None
    ) -> CPCVResult:
        """
        Perform combinatorial purged cross-validation.

        Args:
            data: Feature data
            prices: Price data
            strategy_generator: Function to create strategy
            params: Strategy parameters
            metric: Performance metric
            sample_weight: Sample weights

        Returns:
            CPCVResult object
        """
        logger.info("Starting CPCV")

        # Generate splits
        splits = self.generate_splits(data, sample_weight)

        # Evaluate each split
        train_scores = []
        test_scores = []

        if self.n_jobs and self.n_jobs > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_split,
                        split, data, prices, strategy_generator, params, metric
                    ): split
                    for split in splits
                }

                for future in as_completed(futures):
                    train_score, test_score = future.result()
                    train_scores.append(train_score)
                    test_scores.append(test_score)
        else:
            # Sequential evaluation
            for i, split in enumerate(splits):
                train_score, test_score = self._evaluate_split(
                    split, data, prices, strategy_generator, params, metric
                )
                train_scores.append(train_score)
                test_scores.append(test_score)

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(splits)} splits")

        # Calculate statistics
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        std_train = np.std(train_scores)
        std_test = np.std(test_scores)
        overfitting_ratio = mean_test / mean_train if mean_train > 0 else 0

        logger.info(f"CPCV complete: Train={mean_train:.4f}, Test={mean_test:.4f}")

        return CPCVResult(
            train_scores=train_scores,
            test_scores=test_scores,
            splits=splits,
            mean_train_score=mean_train,
            mean_test_score=mean_test,
            std_train_score=std_train,
            std_test_score=std_test,
            overfitting_ratio=overfitting_ratio
        )

    def calculate_pbo(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_generator: Callable,
        param_sets: List[Dict[str, Any]],
        metric: str = 'sharpe_ratio',
        sample_weight: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO).

        Tests multiple parameter sets and calculates the probability
        that the best in-sample performance will underperform out-of-sample.

        Args:
            data: Feature data
            prices: Price data
            strategy_generator: Function to create strategy
            param_sets: List of parameter dictionaries to test
            metric: Performance metric
            sample_weight: Sample weights

        Returns:
            PBO probability (0 to 1)
        """
        logger.info(f"Calculating PBO for {len(param_sets)} parameter sets")

        # Generate a single train/test split for PBO
        n_samples = len(data)
        split_idx = n_samples // 2

        train_data = data.iloc[:split_idx]
        train_prices = prices.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        test_prices = prices.iloc[split_idx:]

        # Evaluate all parameter sets
        train_performances = []
        test_performances = []

        for i, params in enumerate(param_sets):
            try:
                # Train performance
                train_perf = self._evaluate_params(
                    train_data, train_prices, strategy_generator, params, metric
                )

                # Test performance
                test_perf = self._evaluate_params(
                    test_data, test_prices, strategy_generator, params, metric
                )

                train_performances.append(train_perf)
                test_performances.append(test_perf)

            except Exception as e:
                logger.warning(f"Error evaluating params {i}: {e}")
                train_performances.append(-np.inf)
                test_performances.append(-np.inf)

            if (i + 1) % 50 == 0:
                logger.info(f"PBO: Evaluated {i + 1}/{len(param_sets)} parameter sets")

        # Calculate PBO
        train_performances = np.array(train_performances)
        test_performances = np.array(test_performances)

        # Sort by in-sample (train) performance
        sorted_indices = np.argsort(train_performances)[::-1]

        # Split into halves
        n_half = len(sorted_indices) // 2
        top_half_indices = sorted_indices[:n_half]
        bottom_half_indices = sorted_indices[n_half:]

        # Calculate relative rank in OOS
        top_half_oos = test_performances[top_half_indices]
        all_oos = test_performances

        # Count how many in top half underperform median OOS
        median_oos = np.median(all_oos)
        n_underperform = np.sum(top_half_oos < median_oos)

        pbo = n_underperform / len(top_half_oos) if len(top_half_oos) > 0 else 0.5

        logger.info(f"PBO = {pbo:.4f} (higher means more overfitting)")

        return pbo

    def cscv(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_generator: Callable,
        param_sets: List[Dict[str, Any]],
        metric: str = 'sharpe_ratio',
        n_groups: int = 16
    ) -> Tuple[Dict[str, Any], float]:
        """
        Combinatorial Symmetric Cross-Validation (CSCV).

        More robust version of CPCV that tests multiple train/test
        orderings.

        Args:
            data: Feature data
            prices: Price data
            strategy_generator: Function to create strategy
            param_sets: List of parameter sets to evaluate
            metric: Performance metric
            n_groups: Number of groups to split data into

        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Running CSCV with {n_groups} groups")

        n_samples = len(data)
        indices = np.arange(n_samples)

        # Split into groups
        group_size = n_samples // n_groups
        groups = [
            indices[i * group_size:(i + 1) * group_size]
            for i in range(n_groups)
        ]

        # Generate combinations
        n_test_groups = n_groups // 4  # Use 1/4 for testing
        test_combinations = list(combinations(range(n_groups), n_test_groups))

        logger.info(f"Testing {len(param_sets)} param sets across {len(test_combinations)} splits")

        # Evaluate each parameter set
        param_scores = []

        for params in param_sets:
            split_scores = []

            for test_group_ids in test_combinations:
                # Create train/test split
                test_indices = np.concatenate([groups[i] for i in test_group_ids])
                train_group_ids = [i for i in range(n_groups) if i not in test_group_ids]
                train_indices = np.concatenate([groups[i] for i in train_group_ids])

                # Extract data
                train_data = data.iloc[train_indices]
                train_prices = prices.iloc[train_indices]
                test_data = data.iloc[test_indices]
                test_prices = prices.iloc[test_indices]

                # Evaluate
                try:
                    score = self._evaluate_params(
                        test_data, test_prices, strategy_generator, params, metric
                    )
                    split_scores.append(score)
                except Exception as e:
                    logger.debug(f"Error in CSCV split: {e}")
                    split_scores.append(-np.inf)

            # Average across splits
            avg_score = np.mean(split_scores) if split_scores else -np.inf
            param_scores.append((params, avg_score, np.std(split_scores)))

        # Find best parameters
        best_params, best_score, best_std = max(param_scores, key=lambda x: x[1])

        logger.info(f"CSCV complete. Best score: {best_score:.4f} (std: {best_std:.4f})")

        return best_params, best_score

    def _get_purge_indices(
        self,
        data: pd.DataFrame,
        train_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> np.ndarray:
        """
        Get indices to purge from training data.

        Removes training samples that overlap with test period.
        """
        if self.config.purge_pct == 0:
            return np.array([], dtype=int)

        # Calculate purge window
        purge_samples = int(len(data) * self.config.purge_pct)

        # Find training samples close to test set
        test_start = test_indices.min()
        test_end = test_indices.max()

        # Purge training samples within window before test start
        purge_mask = (
            (train_indices >= test_start - purge_samples) &
            (train_indices < test_start)
        )

        purge_indices = train_indices[purge_mask]

        return purge_indices

    def _get_embargo_indices(
        self,
        data: pd.DataFrame,
        test_indices: np.ndarray
    ) -> np.ndarray:
        """
        Get indices to embargo after test period.

        Prevents information leakage from future data.
        """
        if self.config.embargo_pct == 0:
            return np.array([], dtype=int)

        # Calculate embargo window
        embargo_samples = int(len(data) * self.config.embargo_pct)

        # Embargo samples immediately after test end
        test_end = test_indices.max()
        embargo_start = test_end + 1
        embargo_end = min(test_end + embargo_samples, len(data) - 1)

        embargo_indices = np.arange(embargo_start, embargo_end + 1)

        return embargo_indices

    def _evaluate_split(
        self,
        split: CPCVSplit,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_generator: Callable,
        params: Dict[str, Any],
        metric: str
    ) -> Tuple[float, float]:
        """Evaluate a single CPCV split."""
        # Extract train and test data
        train_data = data.iloc[split.train_indices]
        train_prices = prices.iloc[split.train_indices]
        test_data = data.iloc[split.test_indices]
        test_prices = prices.iloc[split.test_indices]

        # Train performance
        train_score = self._evaluate_params(
            train_data, train_prices, strategy_generator, params, metric
        )

        # Test performance
        test_score = self._evaluate_params(
            test_data, test_prices, strategy_generator, params, metric
        )

        return train_score, test_score

    def _evaluate_params(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_generator: Callable,
        params: Dict[str, Any],
        metric: str
    ) -> float:
        """Evaluate strategy parameters on data."""
        try:
            # Create strategy
            strategy = strategy_generator(**params)

            # Generate signals
            signals = strategy.generate_signals(data, prices)

            # Calculate returns
            price_returns = prices.pct_change()
            if isinstance(signals, pd.DataFrame):
                strategy_returns = (signals.shift(1) * price_returns).mean(axis=1)
            else:
                strategy_returns = signals.shift(1) * price_returns

            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) < 10:
                return -np.inf

            # Calculate metric
            if metric == 'sharpe_ratio':
                if strategy_returns.std() > 0:
                    return (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
                return -np.inf
            elif metric == 'sortino_ratio':
                downside = strategy_returns[strategy_returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    return (strategy_returns.mean() / downside.std()) * np.sqrt(252)
                return -np.inf
            elif metric == 'calmar_ratio':
                cumulative = (1 + strategy_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(drawdown.min())
                if max_dd > 0:
                    annual_return = strategy_returns.mean() * 252
                    return annual_return / max_dd
                return -np.inf
            else:
                raise ValueError(f"Unknown metric: {metric}")

        except Exception as e:
            logger.debug(f"Error evaluating params: {e}")
            return -np.inf

    def calculate_deflated_sharpe(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
        expected_sharpe: float = 0.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> float:
        """
        Calculate deflated Sharpe ratio.

        Adjusts Sharpe ratio for multiple testing and non-normality.

        Args:
            observed_sharpe: Observed Sharpe ratio
            n_trials: Number of trials/strategies tested
            n_observations: Number of observations
            expected_sharpe: Expected Sharpe under null hypothesis
            skewness: Return skewness
            kurtosis: Return kurtosis

        Returns:
            Deflated Sharpe ratio (p-value)
        """
        from scipy.stats import norm

        # Adjust for multiple testing (Bonferroni)
        if n_trials > 1:
            # Expected maximum Sharpe under null
            variance_sharpe = (1 + (1 - skewness * observed_sharpe +
                               (kurtosis - 1) / 4 * observed_sharpe ** 2)) / (n_observations - 1)

            # Adjust for multiple testing
            expected_max_sharpe = (
                expected_sharpe +
                np.sqrt(variance_sharpe) *
                ((1 - np.euler_gamma) * norm.ppf(1 - 1.0 / n_trials) +
                 np.euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e)))
            )

            # Calculate deflated Sharpe
            deflated_sharpe = (observed_sharpe - expected_max_sharpe) / np.sqrt(variance_sharpe)

            # Convert to p-value
            p_value = norm.cdf(deflated_sharpe)

            return p_value
        else:
            # No adjustment needed
            return norm.cdf(observed_sharpe * np.sqrt(n_observations))

    def plot_pbo_analysis(
        self,
        train_performances: np.ndarray,
        test_performances: np.ndarray
    ) -> None:
        """
        Plot PBO analysis.

        Args:
            train_performances: In-sample performances
            test_performances: Out-of-sample performances
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        axes[0].scatter(train_performances, test_performances, alpha=0.5)
        axes[0].plot(
            [train_performances.min(), train_performances.max()],
            [train_performances.min(), train_performances.max()],
            'r--', label='No overfitting line'
        )
        axes[0].set_xlabel('In-Sample Performance')
        axes[0].set_ylabel('Out-of-Sample Performance')
        axes[0].set_title('IS vs OOS Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Rank histogram
        sorted_indices = np.argsort(train_performances)[::-1]
        n_half = len(sorted_indices) // 2
        top_half_oos = test_performances[sorted_indices[:n_half]]

        axes[1].hist(top_half_oos, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(
            np.median(test_performances),
            color='r',
            linestyle='--',
            label='Median OOS'
        )
        axes[1].set_xlabel('OOS Performance (Top IS Half)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Top IS Performers OOS')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

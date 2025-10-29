"""
Monte Carlo Simulation Engine

Implements Monte Carlo simulations for strategy validation including:
- Randomization of returns
- Bootstrap sampling
- Permutation tests
- Scenario analysis
"""

from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    mean_metric: float
    std_metric: float
    percentile_5: float
    percentile_95: float
    percentile_99: float
    probability_positive: float
    all_simulations: np.ndarray


class MonteCarloSimulator:
    """Monte Carlo simulator for strategy validation."""

    def __init__(self, n_simulations: int = 1000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"MonteCarloSimulator initialized with {n_simulations} simulations")

    def bootstrap_returns(self, returns: pd.Series, block_size: int = 1) -> SimulationResult:
        """
        Bootstrap returns to estimate confidence intervals.

        Args:
            returns: Historical returns series
            block_size: Block size for block bootstrap (1 = standard bootstrap)

        Returns:
            SimulationResult with distribution statistics
        """
        logger.info(f"Running bootstrap simulation (block_size={block_size})")

        simulated_totals = []

        for i in range(self.n_simulations):
            if block_size == 1:
                # Standard bootstrap
                resampled = np.random.choice(returns.values, size=len(returns), replace=True)
            else:
                # Block bootstrap for time series
                resampled = self._block_bootstrap(returns.values, block_size)

            # Calculate cumulative return
            total_return = (1 + resampled).prod() - 1
            simulated_totals.append(total_return)

        simulated_totals = np.array(simulated_totals)

        return SimulationResult(
            mean_metric=simulated_totals.mean(),
            std_metric=simulated_totals.std(),
            percentile_5=np.percentile(simulated_totals, 5),
            percentile_95=np.percentile(simulated_totals, 95),
            percentile_99=np.percentile(simulated_totals, 99),
            probability_positive=(simulated_totals > 0).mean(),
            all_simulations=simulated_totals
        )

    def _block_bootstrap(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """Perform block bootstrap resampling."""
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))

        resampled = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            block = data[start_idx:start_idx + block_size]
            resampled.extend(block)

        return np.array(resampled[:n])

    def permutation_test(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        metric_func: Callable = lambda x: x.mean()
    ) -> Dict[str, Any]:
        """
        Permutation test to assess statistical significance.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            metric_func: Function to calculate metric from returns

        Returns:
            Test results including p-value
        """
        logger.info("Running permutation test")

        # Calculate observed difference
        strategy_metric = metric_func(strategy_returns)
        benchmark_metric = metric_func(benchmark_returns)
        observed_diff = strategy_metric - benchmark_metric

        # Combine returns
        all_returns = np.concatenate([strategy_returns.values, benchmark_returns.values])
        n_strategy = len(strategy_returns)

        # Permutation test
        permuted_diffs = []
        for _ in range(self.n_simulations):
            # Randomly shuffle
            np.random.shuffle(all_returns)

            # Split
            perm_strategy = all_returns[:n_strategy]
            perm_benchmark = all_returns[n_strategy:]

            # Calculate difference
            diff = metric_func(pd.Series(perm_strategy)) - metric_func(pd.Series(perm_benchmark))
            permuted_diffs.append(diff)

        permuted_diffs = np.array(permuted_diffs)

        # Calculate p-value (two-tailed)
        p_value = (np.abs(permuted_diffs) >= np.abs(observed_diff)).mean()

        logger.info(f"Permutation test p-value: {p_value:.4f}")

        return {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'null_distribution': permuted_diffs,
            'significant': p_value < 0.05
        }

    def monte_carlo_returns(
        self,
        returns: pd.Series,
        method: str = 'parametric'
    ) -> SimulationResult:
        """
        Generate Monte Carlo simulations of returns.

        Args:
            returns: Historical returns
            method: 'parametric' (normal), 'bootstrap', or 'historical'

        Returns:
            SimulationResult
        """
        logger.info(f"Running Monte Carlo simulation (method={method})")

        simulated_totals = []

        if method == 'parametric':
            # Assume normal distribution
            mean = returns.mean()
            std = returns.std()

            for _ in range(self.n_simulations):
                sim_returns = np.random.normal(mean, std, len(returns))
                total_return = (1 + sim_returns).prod() - 1
                simulated_totals.append(total_return)

        elif method == 'bootstrap':
            return self.bootstrap_returns(returns)

        elif method == 'historical':
            # Randomly select historical periods
            for _ in range(self.n_simulations):
                start_idx = np.random.randint(0, len(returns) - 252)
                period_returns = returns.iloc[start_idx:start_idx + 252]
                total_return = (1 + period_returns).prod() - 1
                simulated_totals.append(total_return)

        simulated_totals = np.array(simulated_totals)

        return SimulationResult(
            mean_metric=simulated_totals.mean(),
            std_metric=simulated_totals.std(),
            percentile_5=np.percentile(simulated_totals, 5),
            percentile_95=np.percentile(simulated_totals, 95),
            percentile_99=np.percentile(simulated_totals, 99),
            probability_positive=(simulated_totals > 0).mean(),
            all_simulations=simulated_totals
        )


class SimulationEngine:
    """
    Simulation-based backtesting engine.

    Runs strategies with simulated market conditions and paths.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize simulation engine."""
        self.config = config
        self.n_simulations = config.get('n_simulations', 1000)
        self.random_seed = config.get('random_seed')
        self.monte_carlo = MonteCarloSimulator(self.n_simulations, self.random_seed)

        logger.info("SimulationEngine initialized")

    def run_monte_carlo_backtest(
        self,
        strategy: Any,
        prices: pd.DataFrame,
        method: str = 'bootstrap',
        block_size: int = 20
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo backtest with resampled data.

        Args:
            strategy: Strategy instance
            prices: Historical prices
            method: Resampling method
            block_size: Block size for block bootstrap

        Returns:
            Monte Carlo backtest results
        """
        logger.info(f"Running Monte Carlo backtest with {self.n_simulations} simulations")

        simulation_results = []

        # Calculate returns
        returns = prices.pct_change().dropna()

        for i in range(self.n_simulations):
            if i % 100 == 0:
                logger.debug(f"Simulation {i}/{self.n_simulations}")

            # Resample returns
            if method == 'bootstrap':
                resampled_returns = self._block_bootstrap(returns, block_size)
            elif method == 'parametric':
                resampled_returns = self._parametric_simulation(returns)
            else:
                resampled_returns = returns  # No resampling

            # Reconstruct prices
            sim_prices = self._returns_to_prices(resampled_returns, prices.iloc[0])

            # Run strategy on simulated prices
            signals = strategy.generate_signals(sim_prices)

            # Calculate performance
            strategy_returns = (signals.shift(1) * resampled_returns).sum(axis=1)
            total_return = (1 + strategy_returns).prod() - 1
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

            simulation_results.append({
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'volatility': strategy_returns.std() * np.sqrt(252)
            })

        results_df = pd.DataFrame(simulation_results)

        logger.info(f"Monte Carlo backtest complete")

        return {
            'mean_return': results_df['total_return'].mean(),
            'std_return': results_df['total_return'].std(),
            'mean_sharpe': results_df['sharpe_ratio'].mean(),
            'percentile_5_return': results_df['total_return'].quantile(0.05),
            'percentile_95_return': results_df['total_return'].quantile(0.95),
            'probability_positive': (results_df['total_return'] > 0).mean(),
            'all_results': results_df
        }

    def _block_bootstrap(self, returns: pd.DataFrame, block_size: int) -> pd.DataFrame:
        """Block bootstrap for DataFrame."""
        n = len(returns)
        n_blocks = int(np.ceil(n / block_size))

        indices = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            indices.extend(range(start_idx, start_idx + block_size))

        indices = indices[:n]
        return returns.iloc[indices].reset_index(drop=True)

    def _parametric_simulation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Generate parametric (normal) simulation."""
        means = returns.mean()
        cov = returns.cov()

        simulated = np.random.multivariate_normal(means, cov, len(returns))
        return pd.DataFrame(simulated, columns=returns.columns)

    def _returns_to_prices(self, returns: pd.DataFrame, initial_prices: pd.Series) -> pd.DataFrame:
        """Convert returns to price series."""
        prices = (1 + returns).cumprod()
        for col in prices.columns:
            prices[col] = prices[col] * initial_prices[col]
        return prices

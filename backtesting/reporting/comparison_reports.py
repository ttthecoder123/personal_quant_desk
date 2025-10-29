"""
Strategy comparison and analysis module.

This module provides tools for comparing multiple strategies:
- Side-by-side metrics comparison
- Relative performance charts
- Risk-adjusted performance comparison
- Correlation analysis
- Efficient frontier construction
- Factor exposure comparison
- Statistical significance tests
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats
from scipy.optimize import minimize


@dataclass
class ComparisonReport:
    """Container for strategy comparison results."""
    strategies: List[str]
    metrics_comparison: pd.DataFrame
    correlation_matrix: pd.DataFrame
    statistical_tests: Dict
    rankings: Dict
    relative_performance: pd.DataFrame


class StrategyComparator:
    """
    Compare multiple backtest strategies.

    Provides comprehensive comparison tools to evaluate and rank strategies
    based on various performance metrics.
    """

    def __init__(self):
        """Initialize strategy comparator."""
        logger.info("StrategyComparator initialized")

    def compare_strategies(
        self,
        strategy_results: Dict[str, Dict]
    ) -> ComparisonReport:
        """
        Compare multiple strategies comprehensively.

        Args:
            strategy_results: Dictionary mapping strategy names to results dict
                            Each results dict should contain:
                            - returns: pd.Series
                            - metrics: Dict
                            - equity_curve: pd.Series

        Returns:
            ComparisonReport with all comparison results
        """
        logger.info(f"Comparing {len(strategy_results)} strategies")

        strategy_names = list(strategy_results.keys())

        # Build metrics comparison table
        metrics_df = self.build_metrics_table(strategy_results)

        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(strategy_results)

        # Perform statistical tests
        statistical_tests = self.perform_statistical_tests(strategy_results)

        # Rank strategies
        rankings = self.rank_strategies(metrics_df)

        # Calculate relative performance
        relative_performance = self.calculate_relative_performance(strategy_results)

        return ComparisonReport(
            strategies=strategy_names,
            metrics_comparison=metrics_df,
            correlation_matrix=correlation_matrix,
            statistical_tests=statistical_tests,
            rankings=rankings,
            relative_performance=relative_performance,
        )

    def build_metrics_table(
        self,
        strategy_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Build side-by-side metrics comparison table.

        Args:
            strategy_results: Strategy results dictionary

        Returns:
            DataFrame with metrics comparison
        """
        logger.debug("Building metrics comparison table")

        metrics_data = {}

        for strategy_name, results in strategy_results.items():
            metrics = results.get('metrics', {})
            metrics_data[strategy_name] = metrics

        df = pd.DataFrame(metrics_data).T

        return df

    def calculate_correlation_matrix(
        self,
        strategy_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between strategy returns.

        Args:
            strategy_results: Strategy results dictionary

        Returns:
            Correlation matrix DataFrame
        """
        logger.debug("Calculating correlation matrix")

        returns_dict = {}

        for strategy_name, results in strategy_results.items():
            if 'returns' in results:
                returns_dict[strategy_name] = results['returns']

        if not returns_dict:
            logger.warning("No returns data found for correlation calculation")
            return pd.DataFrame()

        # Align returns on common dates
        returns_df = pd.DataFrame(returns_dict)
        correlation_matrix = returns_df.corr()

        return correlation_matrix

    def perform_statistical_tests(
        self,
        strategy_results: Dict[str, Dict]
    ) -> Dict:
        """
        Perform statistical significance tests between strategies.

        Args:
            strategy_results: Strategy results dictionary

        Returns:
            Dictionary with test results
        """
        logger.debug("Performing statistical tests")

        results = {}
        strategy_names = list(strategy_results.keys())

        # Pairwise t-tests
        results['pairwise_ttests'] = {}

        for i, strategy1 in enumerate(strategy_names):
            for strategy2 in strategy_names[i+1:]:
                if 'returns' not in strategy_results[strategy1] or 'returns' not in strategy_results[strategy2]:
                    continue

                returns1 = strategy_results[strategy1]['returns'].dropna()
                returns2 = strategy_results[strategy2]['returns'].dropna()

                # Align returns
                common_idx = returns1.index.intersection(returns2.index)
                returns1_aligned = returns1.loc[common_idx]
                returns2_aligned = returns2.loc[common_idx]

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(returns1_aligned, returns2_aligned)

                test_key = f"{strategy1}_vs_{strategy2}"
                results['pairwise_ttests'][test_key] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                }

        # Sharpe ratio differences
        results['sharpe_ratio_tests'] = self._test_sharpe_differences(strategy_results)

        return results

    def _test_sharpe_differences(
        self,
        strategy_results: Dict[str, Dict]
    ) -> Dict:
        """
        Test for statistically significant differences in Sharpe ratios.

        Uses Jobson-Korkie test for Sharpe ratio equality.
        """
        tests = {}
        strategy_names = list(strategy_results.keys())

        for i, strategy1 in enumerate(strategy_names):
            for strategy2 in strategy_names[i+1:]:
                if 'returns' not in strategy_results[strategy1] or 'returns' not in strategy_results[strategy2]:
                    continue

                returns1 = strategy_results[strategy1]['returns'].dropna()
                returns2 = strategy_results[strategy2]['returns'].dropna()

                # Calculate Sharpe ratios
                sharpe1 = returns1.mean() / returns1.std() * np.sqrt(252)
                sharpe2 = returns2.mean() / returns2.std() * np.sqrt(252)

                # Jobson-Korkie test statistic
                n = len(returns1)
                rho = returns1.corr(returns2)

                # Variance of Sharpe ratio difference
                var_diff = (1/n) * (
                    2 - 2*rho + 0.5*sharpe1**2 + 0.5*sharpe2**2 -
                    1.5*sharpe1*sharpe2*rho
                )

                if var_diff > 0:
                    z_stat = (sharpe1 - sharpe2) / np.sqrt(var_diff)
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                else:
                    z_stat = 0
                    p_value = 1.0

                test_key = f"{strategy1}_vs_{strategy2}"
                tests[test_key] = {
                    'sharpe1': sharpe1,
                    'sharpe2': sharpe2,
                    'difference': sharpe1 - sharpe2,
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                }

        return tests

    def rank_strategies(self, metrics_df: pd.DataFrame) -> Dict:
        """
        Rank strategies based on various metrics.

        Args:
            metrics_df: Metrics comparison DataFrame

        Returns:
            Dictionary with rankings for each metric
        """
        logger.debug("Ranking strategies")

        rankings = {}

        # Key metrics for ranking
        ranking_metrics = [
            ('sharpe_ratio', True),  # True = higher is better
            ('sortino_ratio', True),
            ('calmar_ratio', True),
            ('total_return', True),
            ('cagr', True),
            ('max_drawdown', False),  # False = lower is better
            ('annualized_volatility', False),
        ]

        for metric, ascending in ranking_metrics:
            if metric in metrics_df.columns:
                ranked = metrics_df[metric].sort_values(ascending=not ascending)
                rankings[metric] = ranked.to_dict()

        # Overall score (composite ranking)
        rankings['overall_score'] = self._calculate_overall_score(metrics_df)

        return rankings

    def _calculate_overall_score(self, metrics_df: pd.DataFrame) -> Dict:
        """
        Calculate overall score combining multiple metrics.

        Uses normalized z-scores to combine metrics.
        """
        score_components = {}

        # Define metrics and their weights
        metric_weights = {
            'sharpe_ratio': 0.25,
            'sortino_ratio': 0.20,
            'calmar_ratio': 0.15,
            'total_return': 0.15,
            'max_drawdown': -0.15,  # Negative weight (lower is better)
            'annualized_volatility': -0.10,
        }

        for metric, weight in metric_weights.items():
            if metric in metrics_df.columns:
                # Normalize to z-scores
                normalized = (metrics_df[metric] - metrics_df[metric].mean()) / metrics_df[metric].std()
                score_components[metric] = normalized * weight

        # Calculate total score
        total_scores = pd.DataFrame(score_components).sum(axis=1)
        ranked_scores = total_scores.sort_values(ascending=False)

        return ranked_scores.to_dict()

    def calculate_relative_performance(
        self,
        strategy_results: Dict[str, Dict],
        benchmark: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate relative performance vs benchmark.

        Args:
            strategy_results: Strategy results dictionary
            benchmark: Strategy to use as benchmark (if None, uses equal-weight)

        Returns:
            DataFrame with relative performance metrics
        """
        logger.debug(f"Calculating relative performance (benchmark: {benchmark})")

        if benchmark is None:
            # Use equal-weight benchmark
            returns_list = []
            for results in strategy_results.values():
                if 'returns' in results:
                    returns_list.append(results['returns'])

            if returns_list:
                benchmark_returns = pd.concat(returns_list, axis=1).mean(axis=1)
            else:
                logger.warning("No returns data for relative performance")
                return pd.DataFrame()
        else:
            if benchmark not in strategy_results:
                logger.warning(f"Benchmark {benchmark} not found")
                return pd.DataFrame()
            benchmark_returns = strategy_results[benchmark]['returns']

        # Calculate relative metrics
        relative_data = {}

        for strategy_name, results in strategy_results.items():
            if 'returns' not in results:
                continue

            strategy_returns = results['returns']

            # Align with benchmark
            common_idx = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_aligned = strategy_returns.loc[common_idx]
            benchmark_aligned = benchmark_returns.loc[common_idx]

            # Calculate metrics
            active_returns = strategy_aligned - benchmark_aligned

            relative_data[strategy_name] = {
                'active_return': active_returns.mean() * 252,
                'tracking_error': active_returns.std() * np.sqrt(252),
                'information_ratio': (active_returns.mean() / active_returns.std() * np.sqrt(252)
                                    if active_returns.std() > 0 else 0),
                'correlation': strategy_aligned.corr(benchmark_aligned),
                'beta': self._calculate_beta(strategy_aligned, benchmark_aligned),
                'alpha': self._calculate_alpha(strategy_aligned, benchmark_aligned),
            }

        return pd.DataFrame(relative_data).T

    def _calculate_beta(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate beta relative to benchmark."""
        covariance = np.cov(returns, benchmark)[0, 1]
        benchmark_variance = np.var(benchmark)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def _calculate_alpha(self, returns: pd.Series, benchmark: pd.Series, rf_rate: float = 0.02) -> float:
        """Calculate alpha (Jensen's alpha)."""
        beta = self._calculate_beta(returns, benchmark)

        portfolio_return = returns.mean() * 252
        benchmark_return = benchmark.mean() * 252

        alpha = portfolio_return - (rf_rate + beta * (benchmark_return - rf_rate))

        return alpha

    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot side-by-side metrics comparison.

        Args:
            metrics_df: Metrics comparison DataFrame
            metrics: List of metrics to plot (None for key metrics)

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                      'total_return', 'max_drawdown']

        # Filter available metrics
        available_metrics = [m for m in metrics if m in metrics_df.columns]

        if not available_metrics:
            logger.warning("No metrics available to plot")
            return plt.figure()

        n_metrics = len(available_metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            data = metrics_df[metric].sort_values(ascending=False)

            colors = ['green' if x > 0 else 'red' for x in data.values]
            data.plot(kind='barh', ax=ax, color=colors, alpha=0.7, edgecolor='black')

            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')

        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            correlation_matrix: Correlation matrix DataFrame

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )

        ax.set_title('Strategy Return Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_relative_performance(
        self,
        strategy_results: Dict[str, Dict],
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot relative equity curves.

        Args:
            strategy_results: Strategy results dictionary
            normalize: Whether to normalize to 100 at start

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        for strategy_name, results in strategy_results.items():
            if 'equity_curve' not in results:
                continue

            equity = results['equity_curve']

            if normalize:
                equity = (equity / equity.iloc[0]) * 100

            ax.plot(equity.index, equity.values, label=strategy_name, linewidth=2)

        ax.set_title('Strategy Comparison - Equity Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value' + (' (Normalized)' if normalize else ''), fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        if not normalize:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig


class EfficientFrontier:
    """
    Calculate and visualize efficient frontier for strategy portfolios.

    Finds optimal combinations of strategies to maximize risk-adjusted returns.
    """

    def __init__(self):
        """Initialize efficient frontier calculator."""
        logger.info("EfficientFrontier initialized")

    def calculate_efficient_frontier(
        self,
        strategy_results: Dict[str, Dict],
        n_points: int = 100,
        allow_shorting: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Calculate efficient frontier for strategy portfolio.

        Args:
            strategy_results: Strategy results dictionary
            n_points: Number of points on frontier
            allow_shorting: Whether to allow negative weights

        Returns:
            Tuple of (returns, volatilities, weights)
        """
        logger.info("Calculating efficient frontier")

        # Build returns matrix
        returns_dict = {}
        for strategy_name, results in strategy_results.items():
            if 'returns' in results:
                returns_dict[strategy_name] = results['returns']

        if len(returns_dict) < 2:
            logger.warning("Need at least 2 strategies for efficient frontier")
            return np.array([]), np.array([]), []

        returns_df = pd.DataFrame(returns_dict).dropna()

        # Calculate expected returns and covariance
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252

        # Generate frontier
        n_assets = len(returns_dict)
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)

        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []

        for target_return in target_returns:
            # Optimize for minimum variance given target return
            weights = self._optimize_portfolio(
                mean_returns,
                cov_matrix,
                target_return,
                allow_shorting
            )

            if weights is not None:
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                frontier_returns.append(portfolio_return)
                frontier_volatilities.append(portfolio_vol)
                frontier_weights.append(weights)

        return (
            np.array(frontier_returns),
            np.array(frontier_volatilities),
            frontier_weights
        )

    def _optimize_portfolio(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: float,
        allow_shorting: bool
    ) -> Optional[np.ndarray]:
        """
        Optimize portfolio for minimum variance given target return.

        Args:
            mean_returns: Expected returns
            cov_matrix: Covariance matrix
            target_return: Target portfolio return
            allow_shorting: Whether to allow negative weights

        Returns:
            Optimal weights or None if optimization fails
        """
        n_assets = len(mean_returns)

        # Objective: minimize variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}  # Target return
        ]

        # Bounds
        if allow_shorting:
            bounds = tuple((-1, 1) for _ in range(n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

        if result.success:
            return result.x
        else:
            return None

    def find_optimal_portfolios(
        self,
        strategy_results: Dict[str, Dict],
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Find key optimal portfolios.

        Args:
            strategy_results: Strategy results dictionary
            risk_free_rate: Risk-free rate for Sharpe ratio

        Returns:
            Dictionary with optimal portfolio specifications
        """
        logger.info("Finding optimal portfolios")

        # Build returns matrix
        returns_dict = {}
        for strategy_name, results in strategy_results.items():
            if 'returns' in results:
                returns_dict[strategy_name] = results['returns']

        returns_df = pd.DataFrame(returns_dict).dropna()
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252

        optimal_portfolios = {}

        # Maximum Sharpe Ratio Portfolio
        max_sharpe_weights = self._maximize_sharpe_ratio(
            mean_returns,
            cov_matrix,
            risk_free_rate
        )

        if max_sharpe_weights is not None:
            optimal_portfolios['max_sharpe'] = {
                'weights': dict(zip(returns_dict.keys(), max_sharpe_weights)),
                'return': np.dot(max_sharpe_weights, mean_returns),
                'volatility': np.sqrt(np.dot(max_sharpe_weights.T,
                                            np.dot(cov_matrix, max_sharpe_weights))),
                'sharpe': (np.dot(max_sharpe_weights, mean_returns) - risk_free_rate) /
                         np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))
            }

        # Minimum Variance Portfolio
        min_var_weights = self._minimize_variance(mean_returns, cov_matrix)

        if min_var_weights is not None:
            optimal_portfolios['min_variance'] = {
                'weights': dict(zip(returns_dict.keys(), min_var_weights)),
                'return': np.dot(min_var_weights, mean_returns),
                'volatility': np.sqrt(np.dot(min_var_weights.T,
                                            np.dot(cov_matrix, min_var_weights))),
            }

        # Equal Weight Portfolio (for comparison)
        equal_weights = np.array([1/len(returns_dict)] * len(returns_dict))
        optimal_portfolios['equal_weight'] = {
            'weights': dict(zip(returns_dict.keys(), equal_weights)),
            'return': np.dot(equal_weights, mean_returns),
            'volatility': np.sqrt(np.dot(equal_weights.T,
                                        np.dot(cov_matrix, equal_weights))),
        }

        return optimal_portfolios

    def _maximize_sharpe_ratio(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
    ) -> Optional[np.ndarray]:
        """Maximize Sharpe ratio."""
        n_assets = len(mean_returns)

        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            if portfolio_vol == 0:
                return 0

            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # Negative because we minimize

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)

        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            return result.x
        else:
            return None

    def _minimize_variance(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Minimize portfolio variance."""
        n_assets = len(mean_returns)

        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            return result.x
        else:
            return None

    def plot_efficient_frontier(
        self,
        frontier_returns: np.ndarray,
        frontier_volatilities: np.ndarray,
        strategy_results: Dict[str, Dict],
        optimal_portfolios: Optional[Dict] = None
    ) -> plt.Figure:
        """
        Plot efficient frontier with individual strategies.

        Args:
            frontier_returns: Array of frontier returns
            frontier_volatilities: Array of frontier volatilities
            strategy_results: Individual strategy results
            optimal_portfolios: Optional dict of optimal portfolios to highlight

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot efficient frontier
        ax.plot(frontier_volatilities * 100, frontier_returns * 100,
               linewidth=3, label='Efficient Frontier', color='blue')

        # Plot individual strategies
        for strategy_name, results in strategy_results.items():
            if 'returns' not in results:
                continue

            returns = results['returns']
            annual_return = returns.mean() * 252 * 100
            annual_vol = returns.std() * np.sqrt(252) * 100

            ax.scatter(annual_vol, annual_return, s=100, alpha=0.6, label=strategy_name)

        # Plot optimal portfolios if provided
        if optimal_portfolios:
            for portfolio_name, portfolio in optimal_portfolios.items():
                vol = portfolio['volatility'] * 100
                ret = portfolio['return'] * 100

                ax.scatter(vol, ret, s=200, marker='*', edgecolors='black',
                          linewidths=2, label=f'Optimal: {portfolio_name}', zorder=5)

        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Efficient Frontier - Risk vs Return', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

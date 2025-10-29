"""
Walk-Forward Optimizer

Implements walk-forward optimization with parameter stability tracking
and integration with the WalkForwardEngine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
from loguru import logger

from .parameter_optimizer import (
    ParameterOptimizer,
    ParameterSpace,
    OptimizationMethod,
    OptimizationResult
)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    train_period: int  # Training window size in days
    test_period: int  # Test window size in days
    window_type: str = 'rolling'  # 'rolling', 'expanding', 'anchored'
    purge_pct: float = 0.0  # Percentage of data to purge between train/test
    embargo_pct: float = 0.01  # Percentage of data to embargo after test
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN
    n_optimization_iter: int = 50  # Iterations per window
    step_size: Optional[int] = None  # Step size for rolling window (default: test_period)
    adaptive_params: bool = False  # Use adaptive parameter updates


@dataclass
class WalkForwardWindow:
    """Single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: Optional[pd.DataFrame] = None
    test_data: Optional[pd.DataFrame] = None


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimized_params: Dict[str, Any]
    train_score: float
    test_score: float
    test_returns: pd.Series
    optimization_history: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization results."""
    window_results: List[WindowResult]
    aggregated_metrics: Dict[str, float]
    parameter_stability: pd.DataFrame
    oos_performance: Dict[str, float]
    config: WalkForwardConfig


class WalkForwardOptimizer:
    """
    Walk-forward optimization with parameter stability tracking.

    Implements:
    - Rolling/expanding/anchored windows
    - Parameter optimization per window
    - Out-of-sample performance tracking
    - Parameter stability analysis
    - Adaptive parameter updates
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        param_space: List[ParameterSpace],
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            config: Walk-forward configuration
            param_space: Parameter space for optimization
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.config = config
        self.param_space = param_space
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Set step size
        if config.step_size is None:
            self.config.step_size = config.test_period

        logger.info(f"WalkForwardOptimizer initialized: {config.window_type} windows")

    def generate_windows(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows.

        Args:
            data: Time series data with DatetimeIndex
            start_date: Start date
            end_date: End date

        Returns:
            List of WalkForwardWindow objects
        """
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]

        windows = []
        window_id = 0

        # Calculate embargo size
        embargo_size = int(self.config.test_period * self.config.embargo_pct)
        purge_size = int(self.config.train_period * self.config.purge_pct)

        if self.config.window_type == 'rolling':
            # Rolling windows
            current_start = start_date

            while True:
                # Calculate train window
                train_end_idx = self._get_date_index(data, current_start) + self.config.train_period
                if train_end_idx >= len(data):
                    break

                train_end = data.index[train_end_idx]

                # Calculate test window with purge
                test_start_idx = train_end_idx + purge_size
                test_end_idx = test_start_idx + self.config.test_period

                if test_end_idx >= len(data):
                    break

                test_start = data.index[test_start_idx]
                test_end = data.index[test_end_idx]

                windows.append(WalkForwardWindow(
                    window_id=window_id,
                    train_start=current_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                ))

                window_id += 1

                # Move to next window
                next_start_idx = self._get_date_index(data, current_start) + self.config.step_size
                if next_start_idx + self.config.train_period >= len(data):
                    break
                current_start = data.index[next_start_idx]

        elif self.config.window_type in ['expanding', 'anchored']:
            # Expanding windows (anchored at start)
            anchor_start = start_date
            current_train_end_idx = self._get_date_index(data, start_date) + self.config.train_period

            while True:
                if current_train_end_idx >= len(data):
                    break

                train_end = data.index[current_train_end_idx]

                # Calculate test window
                test_start_idx = current_train_end_idx + purge_size
                test_end_idx = test_start_idx + self.config.test_period

                if test_end_idx >= len(data):
                    break

                test_start = data.index[test_start_idx]
                test_end = data.index[test_end_idx]

                windows.append(WalkForwardWindow(
                    window_id=window_id,
                    train_start=anchor_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                ))

                window_id += 1

                # Move to next window
                current_train_end_idx += self.config.step_size

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def optimize(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_generator: Callable,
        objective_function_generator: Callable,
        constraints: Optional[List[Callable]] = None,
        verbose: bool = True
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            data: Feature data with DatetimeIndex
            prices: Price data with DatetimeIndex
            strategy_generator: Function to create strategy from parameters
            objective_function_generator: Function to create objective function
            constraints: Parameter constraints
            verbose: Print progress

        Returns:
            WalkForwardResult object
        """
        logger.info("Starting walk-forward optimization")

        # Generate windows
        windows = self.generate_windows(data)

        # Optimize each window
        window_results = []
        previous_params = None

        for i, window in enumerate(windows):
            logger.info(f"Processing window {i + 1}/{len(windows)}: {window.train_start.date()} to {window.test_end.date()}")

            # Extract train and test data
            train_data = data.loc[window.train_start:window.train_end]
            train_prices = prices.loc[window.train_start:window.train_end]
            test_data = data.loc[window.test_start:window.test_end]
            test_prices = prices.loc[window.test_start:window.test_end]

            # Create objective function for training data
            train_objective = objective_function_generator(
                train_data, train_prices, strategy_generator
            )

            # Initialize optimizer
            if self.config.adaptive_params and previous_params is not None:
                # Use adaptive initialization based on previous window
                optimizer = self._create_adaptive_optimizer(
                    train_objective, previous_params
                )
            else:
                optimizer = ParameterOptimizer(
                    objective_function=train_objective,
                    param_space=self.param_space,
                    method=self.config.optimization_method,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state
                )

            # Optimize parameters on training data
            opt_result = optimizer.optimize(
                n_iter=self.config.n_optimization_iter,
                constraints=constraints,
                verbose=False
            )

            optimized_params = opt_result.best_params
            train_score = opt_result.best_score

            # Evaluate on test data (out-of-sample)
            test_objective = objective_function_generator(
                test_data, test_prices, strategy_generator
            )
            test_score = test_objective(optimized_params)

            # Calculate test returns for analysis
            strategy = strategy_generator(**optimized_params)
            test_signals = strategy.generate_signals(test_data, test_prices)
            test_returns = self._calculate_returns(test_signals, test_prices)

            # Calculate additional metrics
            metrics = self._calculate_metrics(test_returns)

            # Store results
            window_result = WindowResult(
                window_id=window.window_id,
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                optimized_params=optimized_params,
                train_score=train_score,
                test_score=test_score,
                test_returns=test_returns,
                optimization_history=opt_result.optimization_history,
                metrics=metrics
            )
            window_results.append(window_result)

            # Update previous params for adaptive mode
            previous_params = optimized_params

            if verbose:
                logger.info(
                    f"Window {i + 1}: Train score={train_score:.4f}, "
                    f"Test score={test_score:.4f}, Sharpe={metrics.get('sharpe_ratio', 0):.2f}"
                )

        # Analyze results
        aggregated_metrics = self._aggregate_metrics(window_results)
        parameter_stability = self._analyze_parameter_stability(window_results)
        oos_performance = self._calculate_oos_performance(window_results)

        logger.info("Walk-forward optimization complete")

        return WalkForwardResult(
            window_results=window_results,
            aggregated_metrics=aggregated_metrics,
            parameter_stability=parameter_stability,
            oos_performance=oos_performance,
            config=self.config
        )

    def _get_date_index(self, data: pd.DataFrame, date: datetime) -> int:
        """Get integer index for date."""
        try:
            return data.index.get_loc(date)
        except KeyError:
            # Find nearest date
            idx = data.index.searchsorted(date)
            return min(idx, len(data) - 1)

    def _calculate_returns(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.Series:
        """Calculate strategy returns from signals."""
        price_returns = prices.pct_change()

        if isinstance(signals, pd.DataFrame):
            # Multi-asset strategy
            strategy_returns = (signals.shift(1) * price_returns).mean(axis=1)
        else:
            # Single asset strategy
            strategy_returns = signals.shift(1) * price_returns

        return strategy_returns.dropna()

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }

        # Total return
        total_return = (1 + returns).prod() - 1

        # Sharpe ratio
        sharpe_ratio = 0.0
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Sortino ratio
        sortino_ratio = 0.0
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(252)

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = 0.0
        if max_drawdown < 0:
            annual_return = returns.mean() * 252
            calmar_ratio = annual_return / abs(max_drawdown)

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }

    def _aggregate_metrics(self, window_results: List[WindowResult]) -> Dict[str, float]:
        """Aggregate metrics across all windows."""
        # Combine all test returns
        all_returns = pd.concat([w.test_returns for w in window_results])

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(all_returns)

        # Window-level statistics
        train_scores = [w.train_score for w in window_results]
        test_scores = [w.test_score for w in window_results]
        sharpes = [w.metrics['sharpe_ratio'] for w in window_results]

        # Overfitting indicators
        avg_train_score = np.mean(train_scores)
        avg_test_score = np.mean(test_scores)
        overfitting_ratio = avg_test_score / avg_train_score if avg_train_score > 0 else 0

        return {
            **overall_metrics,
            'avg_train_score': avg_train_score,
            'avg_test_score': avg_test_score,
            'overfitting_ratio': overfitting_ratio,
            'sharpe_consistency': np.std(sharpes),
            'positive_windows': sum(1 for s in sharpes if s > 0) / len(sharpes),
            'mean_window_sharpe': np.mean(sharpes),
            'median_window_sharpe': np.median(sharpes),
            'worst_window_sharpe': min(sharpes),
            'best_window_sharpe': max(sharpes)
        }

    def _analyze_parameter_stability(
        self,
        window_results: List[WindowResult]
    ) -> pd.DataFrame:
        """
        Analyze parameter stability across windows.

        Returns DataFrame with parameter statistics and stability metrics.
        """
        # Extract parameter history
        param_history = []
        for result in window_results:
            param_dict = result.optimized_params.copy()
            param_dict['window_id'] = result.window_id
            param_dict['test_score'] = result.test_score
            param_dict['sharpe_ratio'] = result.metrics['sharpe_ratio']
            param_history.append(param_dict)

        param_df = pd.DataFrame(param_history)

        # Calculate stability metrics
        param_cols = [
            col for col in param_df.columns
            if col not in ['window_id', 'test_score', 'sharpe_ratio']
        ]

        stability_stats = {}
        for param in param_cols:
            if pd.api.types.is_numeric_dtype(param_df[param]):
                values = param_df[param]
                stability_stats[param] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median(),
                    'coef_variation': values.std() / values.mean() if values.mean() != 0 else np.inf,
                    'range_pct': (values.max() - values.min()) / values.mean() if values.mean() != 0 else np.inf
                }

        logger.info("Parameter stability analysis complete")

        return pd.DataFrame(stability_stats).T

    def _calculate_oos_performance(
        self,
        window_results: List[WindowResult]
    ) -> Dict[str, float]:
        """Calculate out-of-sample performance metrics."""
        # Combine all OOS returns
        oos_returns = pd.concat([w.test_returns for w in window_results])

        # Calculate OOS metrics
        oos_metrics = self._calculate_metrics(oos_returns)

        # Calculate IS/OOS comparison
        is_sharpes = [w.train_score for w in window_results if w.train_score != -np.inf]
        oos_sharpes = [w.metrics['sharpe_ratio'] for w in window_results]

        oos_degradation = 0.0
        if len(is_sharpes) > 0 and np.mean(is_sharpes) > 0:
            oos_degradation = 1 - (np.mean(oos_sharpes) / np.mean(is_sharpes))

        return {
            **oos_metrics,
            'oos_degradation': oos_degradation,
            'avg_oos_sharpe': np.mean(oos_sharpes),
            'oos_sharpe_consistency': np.std(oos_sharpes)
        }

    def _create_adaptive_optimizer(
        self,
        objective_function: Callable,
        previous_params: Dict[str, Any]
    ) -> ParameterOptimizer:
        """
        Create optimizer with adaptive initialization.

        Focuses search around previous optimal parameters.
        """
        # Create narrower parameter space around previous values
        adaptive_space = []
        for param in self.param_space:
            if param.name in previous_params:
                prev_value = previous_params[param.name]

                if param.type == 'continuous':
                    # Narrow range around previous value
                    param_range = param.upper - param.lower
                    new_lower = max(param.lower, prev_value - param_range * 0.2)
                    new_upper = min(param.upper, prev_value + param_range * 0.2)

                    adaptive_param = ParameterSpace(
                        name=param.name,
                        type=param.type,
                        lower=new_lower,
                        upper=new_upper,
                        log_scale=param.log_scale
                    )
                    adaptive_space.append(adaptive_param)
                elif param.type == 'integer':
                    # Narrow range around previous value
                    param_range = param.upper - param.lower
                    new_lower = max(param.lower, int(prev_value - param_range * 0.2))
                    new_upper = min(param.upper, int(prev_value + param_range * 0.2))

                    adaptive_param = ParameterSpace(
                        name=param.name,
                        type=param.type,
                        lower=new_lower,
                        upper=new_upper
                    )
                    adaptive_space.append(adaptive_param)
                else:
                    # Keep categorical as-is
                    adaptive_space.append(param)
            else:
                adaptive_space.append(param)

        return ParameterOptimizer(
            objective_function=objective_function,
            param_space=adaptive_space,
            method=self.config.optimization_method,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

    def plot_parameter_evolution(
        self,
        result: WalkForwardResult,
        param_names: Optional[List[str]] = None
    ) -> None:
        """
        Plot parameter evolution across windows.

        Args:
            result: Walk-forward optimization result
            param_names: List of parameter names to plot (None = all)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        # Extract parameter history
        param_history = []
        for window_result in result.window_results:
            param_dict = window_result.optimized_params.copy()
            param_dict['window_id'] = window_result.window_id
            param_dict['test_score'] = window_result.test_score
            param_history.append(param_dict)

        param_df = pd.DataFrame(param_history)

        # Select parameters to plot
        if param_names is None:
            param_names = [
                col for col in param_df.columns
                if col not in ['window_id', 'test_score'] and
                pd.api.types.is_numeric_dtype(param_df[col])
            ]

        # Create subplots
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3 * n_params))
        if n_params == 1:
            axes = [axes]

        for ax, param_name in zip(axes, param_names):
            ax.plot(param_df['window_id'], param_df[param_name], 'o-')
            ax.set_xlabel('Window ID')
            ax.set_ylabel(param_name)
            ax.set_title(f'{param_name} Evolution')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_performance_metrics(self, result: WalkForwardResult) -> None:
        """
        Plot performance metrics across windows.

        Args:
            result: Walk-forward optimization result
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        # Extract metrics
        window_ids = [w.window_id for w in result.window_results]
        train_scores = [w.train_score for w in result.window_results]
        test_scores = [w.test_score for w in result.window_results]
        sharpes = [w.metrics['sharpe_ratio'] for w in result.window_results]

        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Train vs Test scores
        axes[0].plot(window_ids, train_scores, 'o-', label='Train Score')
        axes[0].plot(window_ids, test_scores, 's-', label='Test Score (OOS)')
        axes[0].set_xlabel('Window ID')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Train vs Test Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # OOS Sharpe ratios
        axes[1].bar(window_ids, sharpes, alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Window ID')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].set_title('Out-of-Sample Sharpe Ratio by Window')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

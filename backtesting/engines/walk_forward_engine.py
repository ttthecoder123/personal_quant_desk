"""
Walk-Forward Backtest Engine

Implements walk-forward analysis with rolling/expanding windows and
out-of-sample testing. Based on López de Prado's methodology.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class WalkForwardWindow:
    """Walk-forward window configuration."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


class WalkForwardEngine:
    """
    Walk-forward backtesting engine.

    Implements proper walk-forward analysis with:
    - Rolling or expanding training windows
    - Out-of-sample testing
    - Parameter optimization within each window
    - Purging and embargo to prevent look-ahead bias
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize walk-forward engine.

        Args:
            config: Engine configuration
        """
        self.config = config
        self.train_period = config.get('train_period', 252)  # Trading days
        self.test_period = config.get('test_period', 63)  # Trading days
        self.window_type = config.get('window_type', 'rolling')  # 'rolling' or 'expanding'
        self.purge_pct = config.get('purge_pct', 0.0)  # Purge overlap
        self.embargo_pct = config.get('embargo_pct', 0.01)  # Embargo after test

        logger.info(f"WalkForwardEngine initialized (type={self.window_type})")

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
            start_date: Start date (default: first date in data)
            end_date: End date (default: last date in data)

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
        embargo_size = int(self.test_period * self.embargo_pct)

        if self.window_type == 'rolling':
            # Rolling windows
            current_start = start_date

            while True:
                # Calculate train window
                train_end_idx = self._get_date_index(data, current_start) + self.train_period
                if train_end_idx >= len(data):
                    break

                train_end = data.index[train_end_idx]

                # Calculate test window with purge
                purge_size = int(self.train_period * self.purge_pct)
                test_start_idx = train_end_idx + purge_size
                test_end_idx = test_start_idx + self.test_period

                if test_end_idx >= len(data):
                    break

                test_start = data.index[test_start_idx]
                test_end = data.index[test_end_idx]

                windows.append(WalkForwardWindow(
                    train_start=current_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id
                ))

                window_id += 1

                # Move to next window (with embargo)
                next_start_idx = test_end_idx + embargo_size
                if next_start_idx >= len(data):
                    break
                current_start = data.index[next_start_idx]

        elif self.window_type == 'expanding':
            # Expanding windows (anchored)
            anchor_start = start_date

            current_train_end_idx = self._get_date_index(data, start_date) + self.train_period

            while True:
                if current_train_end_idx >= len(data):
                    break

                train_end = data.index[current_train_end_idx]

                # Calculate test window
                purge_size = int(self.train_period * self.purge_pct)
                test_start_idx = current_train_end_idx + purge_size
                test_end_idx = test_start_idx + self.test_period

                if test_end_idx >= len(data):
                    break

                test_start = data.index[test_start_idx]
                test_end = data.index[test_end_idx]

                windows.append(WalkForwardWindow(
                    train_start=anchor_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id
                ))

                window_id += 1

                # Move to next window
                current_train_end_idx = test_end_idx + embargo_size

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_generator: Callable,
        optimizer: Optional[Callable] = None,
        param_grid: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward analysis.

        Args:
            data: Feature data
            prices: Price data
            strategy_generator: Function to create strategy from parameters
            optimizer: Optional optimizer function for parameters
            param_grid: Parameter grid for optimization

        Returns:
            Walk-forward results
        """
        logger.info("Starting walk-forward analysis")

        windows = self.generate_windows(data)
        window_results = []

        for window in windows:
            logger.info(f"Processing window {window.window_id + 1}/{len(windows)}")

            # Extract train and test data
            train_data = data.loc[window.train_start:window.train_end]
            train_prices = prices.loc[window.train_start:window.train_end]
            test_data = data.loc[window.test_start:window.test_end]
            test_prices = prices.loc[window.test_start:window.test_end]

            # Optimize parameters on training data (if optimizer provided)
            if optimizer and param_grid:
                logger.debug(f"Optimizing parameters for window {window.window_id}")
                best_params = optimizer(
                    train_data,
                    train_prices,
                    strategy_generator,
                    param_grid
                )
            else:
                best_params = {}

            # Create strategy with optimized parameters
            strategy = strategy_generator(**best_params)

            # Run backtest on test period (out-of-sample)
            test_signals = strategy.generate_signals(test_data, test_prices)
            test_returns = self._calculate_returns(test_signals, test_prices)

            # Calculate metrics
            metrics = self._calculate_window_metrics(test_returns, test_prices)

            window_results.append({
                'window_id': window.window_id,
                'train_start': window.train_start,
                'train_end': window.train_end,
                'test_start': window.test_start,
                'test_end': window.test_end,
                'params': best_params,
                'test_returns': test_returns,
                **metrics
            })

        # Aggregate results
        aggregated = self._aggregate_results(window_results)

        logger.info("Walk-forward analysis complete")

        return {
            'windows': window_results,
            'aggregated_metrics': aggregated,
            'window_type': self.window_type,
            'n_windows': len(windows)
        }

    def _get_date_index(self, data: pd.DataFrame, date: datetime) -> int:
        """Get integer index for date."""
        return data.index.get_loc(date, method='nearest')

    def _calculate_returns(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from signals."""
        price_returns = prices.pct_change()

        # Assume equal weight across instruments
        if isinstance(signals, pd.DataFrame):
            strategy_returns = (signals.shift(1) * price_returns).mean(axis=1)
        else:
            strategy_returns = signals.shift(1) * price_returns

        return strategy_returns.dropna()

    def _calculate_window_metrics(self, returns: pd.Series, prices: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for a single window."""
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }

        # Total return
        total_return = (1 + returns).prod() - 1

        # Sharpe ratio
        sharpe_ratio = 0.0
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }

    def _aggregate_results(self, window_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results across all windows."""
        # Combine all test returns
        all_returns = pd.concat([w['test_returns'] for w in window_results])

        # Calculate overall metrics
        total_return = (1 + all_returns).prod() - 1
        sharpe_ratio = (all_returns.mean() / all_returns.std()) * np.sqrt(252) if all_returns.std() > 0 else 0

        cumulative = (1 + all_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Window statistics
        window_sharpes = [w['sharpe_ratio'] for w in window_results]
        window_returns = [w['total_return'] for w in window_results]

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'mean_window_sharpe': np.mean(window_sharpes),
            'std_window_sharpe': np.std(window_sharpes),
            'mean_window_return': np.mean(window_returns),
            'positive_windows': sum(1 for r in window_returns if r > 0) / len(window_returns),
            'sharpe_consistency': np.std(window_sharpes)  # Lower is more consistent
        }

    def analyze_parameter_stability(
        self,
        window_results: List[Dict]
    ) -> pd.DataFrame:
        """
        Analyze parameter stability across windows.

        Args:
            window_results: Results from walk-forward analysis

        Returns:
            DataFrame with parameter statistics
        """
        if not window_results or 'params' not in window_results[0]:
            return pd.DataFrame()

        # Extract parameters from each window
        param_history = []
        for result in window_results:
            param_dict = result['params'].copy()
            param_dict['window_id'] = result['window_id']
            param_dict['sharpe_ratio'] = result['sharpe_ratio']
            param_history.append(param_dict)

        param_df = pd.DataFrame(param_history)

        # Calculate statistics for each parameter
        param_cols = [col for col in param_df.columns
                     if col not in ['window_id', 'sharpe_ratio']]

        stability_stats = {}
        for param in param_cols:
            if pd.api.types.is_numeric_dtype(param_df[param]):
                stability_stats[param] = {
                    'mean': param_df[param].mean(),
                    'std': param_df[param].std(),
                    'min': param_df[param].min(),
                    'max': param_df[param].max(),
                    'coef_variation': param_df[param].std() / param_df[param].mean() if param_df[param].mean() != 0 else 0
                }

        logger.info("Parameter stability analysis complete")

        return pd.DataFrame(stability_stats).T

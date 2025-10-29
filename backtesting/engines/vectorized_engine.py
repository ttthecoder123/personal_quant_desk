"""
Vectorized Backtest Engine

Fast vectorized backtesting using pandas/numpy operations.
Suitable for rapid strategy prototyping and parameter sweeps.
"""

from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from loguru import logger


class VectorizedEngine:
    """
    Fast vectorized backtesting engine.

    Trades realism for speed - processes entire datasets at once using
    vectorized operations. Ideal for:
    - Rapid strategy prototyping
    - Parameter optimization sweeps
    - Preliminary testing before event-driven validation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vectorized engine.

        Args:
            config: Engine configuration including commission, slippage
        """
        self.config = config
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        self.initial_capital = config.get('initial_capital', 1000000.0)

        logger.info("VectorizedEngine initialized")

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        position_sizing: Optional[Callable] = None,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run vectorized backtest.

        Args:
            signals: DataFrame with signals (-1, 0, 1) for each symbol
            prices: DataFrame with prices for each symbol
            position_sizing: Optional position sizing function
            leverage: Maximum leverage to apply

        Returns:
            Backtest results dictionary
        """
        logger.info("Running vectorized backtest")

        # Align data
        common_index = signals.index.intersection(prices.index)
        signals = signals.loc[common_index]
        prices = prices.loc[common_index]

        # Default position sizing: equal weight
        if position_sizing is None:
            positions = signals * (1.0 / len(signals.columns))
        else:
            positions = signals.apply(position_sizing, axis=1)

        # Apply leverage
        positions = positions * leverage

        # Calculate returns
        returns = prices.pct_change()

        # Strategy returns (position * next period return)
        strategy_returns = (positions.shift(1) * returns).sum(axis=1)

        # Apply costs
        position_changes = positions.diff().abs()
        turnover = position_changes.sum(axis=1)
        costs = turnover * (self.commission + self.slippage)
        net_returns = strategy_returns - costs

        # Calculate equity curve
        equity = (1 + net_returns).cumprod() * self.initial_capital

        # Calculate metrics
        metrics = self._calculate_metrics(net_returns, equity, positions, prices)

        results = {
            'equity_curve': equity,
            'returns': net_returns,
            'positions': positions,
            'turnover': turnover,
            'costs': costs,
            **metrics
        }

        logger.info(f"Backtest complete. Sharpe: {metrics['sharpe_ratio']:.2f}")

        return results

    def run_multi_strategy(
        self,
        strategies: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run multiple strategies with portfolio combination.

        Args:
            strategies: Dict of strategy name -> signals DataFrame
            prices: Price data
            weights: Strategy weights (equal weight if None)

        Returns:
            Combined results
        """
        logger.info(f"Running multi-strategy backtest with {len(strategies)} strategies")

        if weights is None:
            weights = {name: 1.0 / len(strategies) for name in strategies}

        # Run each strategy
        strategy_results = {}
        for name, signals in strategies.items():
            logger.debug(f"Running strategy: {name}")
            result = self.run_backtest(signals, prices)
            strategy_results[name] = result

        # Combine returns
        combined_returns = sum(
            strategy_results[name]['returns'] * weights[name]
            for name in strategies
        )

        # Calculate combined equity
        combined_equity = (1 + combined_returns).cumprod() * self.initial_capital

        # Calculate metrics
        metrics = self._calculate_metrics(
            combined_returns,
            combined_equity,
            None,
            prices
        )

        return {
            'equity_curve': combined_equity,
            'returns': combined_returns,
            'strategy_results': strategy_results,
            'weights': weights,
            **metrics
        }

    def optimize_parameters(
        self,
        signal_generator: Callable,
        prices: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.

        Args:
            signal_generator: Function that generates signals from params
            prices: Price data
            param_grid: Dictionary of parameter names -> values to test
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)

        Returns:
            Optimization results with best parameters
        """
        logger.info(f"Starting parameter optimization on {len(param_grid)} parameters")

        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        results = []
        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))

            try:
                # Generate signals with these parameters
                signals = signal_generator(prices, **params)

                # Run backtest
                backtest_result = self.run_backtest(signals, prices)

                # Store results
                result = {
                    'params': params,
                    'metric_value': backtest_result.get(metric, 0),
                    **{k: v for k, v in backtest_result.items()
                       if k in ['sharpe_ratio', 'total_return', 'max_drawdown']}
                }
                results.append(result)

            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue

        # Find best parameters
        if not results:
            logger.error("No valid results from optimization")
            return {'error': 'No valid results'}

        results_df = pd.DataFrame(results)

        # Maximize metric (or minimize if drawdown)
        if metric == 'max_drawdown':
            best_idx = results_df['metric_value'].argmax()  # Less negative is better
        else:
            best_idx = results_df['metric_value'].argmax()

        best_result = results[best_idx]

        logger.info(f"Best {metric}: {best_result['metric_value']:.4f}")
        logger.info(f"Best params: {best_result['params']}")

        return {
            'best_params': best_result['params'],
            'best_metric': best_result['metric_value'],
            'all_results': results_df,
            'optimization_metric': metric
        }

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        positions: Optional[pd.DataFrame],
        prices: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics."""

        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0
            }

        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Annualization factor
        days = (returns.index[-1] - returns.index[0]).days
        periods_per_year = 252 if days > 365 else len(returns) / (days / 365.25)

        # Sharpe ratio
        sharpe_ratio = 0.0
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

        # Sortino ratio
        sortino_ratio = 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)

        # Maximum drawdown
        cumulative = equity / equity.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = 0.0
        if max_drawdown != 0:
            annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
            calmar_ratio = annualized_return / abs(max_drawdown)

        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'volatility': volatility
        }

    def run_rolling_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        window_size: int = 252,
        step_size: int = 21
    ) -> Dict[str, Any]:
        """
        Run rolling window backtest.

        Args:
            signals: Strategy signals
            prices: Price data
            window_size: Size of rolling window (e.g., 252 trading days)
            step_size: Step between windows (e.g., 21 for monthly)

        Returns:
            Rolling backtest results
        """
        logger.info(f"Running rolling backtest (window={window_size}, step={step_size})")

        results = []
        start_idx = 0

        while start_idx + window_size <= len(signals):
            end_idx = start_idx + window_size

            # Extract window
            window_signals = signals.iloc[start_idx:end_idx]
            window_prices = prices.iloc[start_idx:end_idx]

            # Run backtest on window
            window_result = self.run_backtest(window_signals, window_prices)

            results.append({
                'start_date': window_signals.index[0],
                'end_date': window_signals.index[-1],
                'sharpe_ratio': window_result['sharpe_ratio'],
                'total_return': window_result['total_return'],
                'max_drawdown': window_result['max_drawdown'],
                'volatility': window_result['volatility']
            })

            start_idx += step_size

        results_df = pd.DataFrame(results)
        results_df.set_index('start_date', inplace=True)

        logger.info(f"Completed {len(results)} rolling windows")

        return {
            'rolling_results': results_df,
            'mean_sharpe': results_df['sharpe_ratio'].mean(),
            'std_sharpe': results_df['sharpe_ratio'].std(),
            'mean_return': results_df['total_return'].mean(),
            'consistency': len(results_df[results_df['total_return'] > 0]) / len(results_df)
        }

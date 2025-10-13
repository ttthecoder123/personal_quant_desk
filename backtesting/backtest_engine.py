"""
Backtesting Engine for strategy simulation and performance analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from utils.logger import get_backtest_logger

log = get_backtest_logger()


@dataclass
class BacktestResults:
    """Container for backtest results."""
    total_returns: float
    annualized_returns: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]


class BacktestEngine:
    """Engine for running strategy backtests."""

    def __init__(self, config: dict):
        """Initialize backtest engine."""
        self.config = config
        self.simulation_config = config.get('simulation', {})
        self.metrics_config = config.get('metrics', {})
        self.optimization_config = config.get('optimization', {})

        # Simulation parameters
        self.initial_capital = self.simulation_config.get('initial_capital', 1000000)
        self.commission = self.simulation_config.get('commission', 0.001)
        self.slippage = self.simulation_config.get('slippage', 0.0005)

        # State variables
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def run(
        self,
        strategies: Dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResults:
        """
        Run backtest for given strategies.

        Args:
            strategies: Dictionary of strategy objects
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResults object
        """
        log.info("Starting backtest simulation...")

        # Parse dates
        start = pd.to_datetime(start_date or self.simulation_config.get('start_date'))
        end = pd.to_datetime(end_date or self.simulation_config.get('end_date'))

        # Reset state
        self._reset_state()

        # Load historical data
        market_data = self._load_historical_data(start, end)

        if market_data.empty:
            log.error("No historical data available for backtest")
            return self._generate_empty_results()

        # Run simulation
        results = self._run_simulation(strategies, market_data)

        log.success("Backtest simulation completed")
        return results

    def _reset_state(self):
        """Reset backtest state."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def _load_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load historical market data."""
        log.info(f"Loading historical data from {start_date} to {end_date}")

        # Placeholder for data loading
        # In production, would load from database or data provider
        # For now, generate sample data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame(index=dates)

        # Generate sample price data for multiple symbols
        symbols = ['SPY', 'QQQ', 'AUDUSD', 'WTI', 'GOLD']
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * (1 + returns).cumprod()
            data[f'{symbol}_Close'] = prices
            data[f'{symbol}_Volume'] = np.random.randint(1000000, 10000000, len(dates))

        return data

    def _run_simulation(
        self,
        strategies: Dict,
        market_data: pd.DataFrame
    ) -> BacktestResults:
        """Run the backtest simulation."""
        log.info("Running backtest simulation...")

        # Iterate through each time period
        for i, (date, row) in enumerate(market_data.iterrows()):
            # Prepare current market data
            current_data = self._prepare_market_data(row)

            # Generate signals from strategies
            signals = self._generate_strategy_signals(strategies, current_data, i, market_data)

            # Execute trades based on signals
            self._execute_trades(signals, current_data, date)

            # Update portfolio value
            self._update_portfolio_value(current_data)

            # Record equity curve
            self.equity_curve.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions_value': self.portfolio_value - self.cash
            })

        # Calculate performance metrics
        results = self._calculate_results()
        return results

    def _prepare_market_data(self, row: pd.Series) -> Dict[str, Dict]:
        """Prepare market data for current time step."""
        market_data = {}

        for column in row.index:
            if '_Close' in column:
                symbol = column.replace('_Close', '')
                market_data[symbol] = {
                    'price': row[column],
                    'volume': row.get(f'{symbol}_Volume', 0)
                }

        return market_data

    def _generate_strategy_signals(
        self,
        strategies: Dict,
        current_data: Dict,
        current_index: int,
        full_data: pd.DataFrame
    ) -> List[Dict]:
        """Generate signals from strategies."""
        all_signals = []

        for strategy_name, strategy in strategies.items():
            # Prepare historical data for strategy
            if current_index < 20:  # Need minimum history
                continue

            # Create DataFrame for strategy
            strategy_data = {}
            for symbol in current_data.keys():
                if f'{symbol}_Close' in full_data.columns:
                    historical = full_data[f'{symbol}_Close'].iloc[:current_index+1]
                    df = pd.DataFrame({
                        'Close': historical,
                        'Volume': full_data[f'{symbol}_Volume'].iloc[:current_index+1]
                    })
                    strategy_data[symbol] = df

            # Generate signals (simplified for backtest)
            try:
                # Mock signal generation based on simple rules
                for symbol, data in strategy_data.items():
                    if len(data) < 20:
                        continue

                    # Simple momentum signal
                    returns = data['Close'].pct_change(20)
                    if returns.iloc[-1] > 0.05:  # 5% positive return
                        signal = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'strategy': strategy_name,
                            'price': current_data[symbol]['price'],
                            'strength': min(returns.iloc[-1] / 0.05, 2.0)
                        }
                        all_signals.append(signal)
                    elif returns.iloc[-1] < -0.05:  # 5% negative return
                        signal = {
                            'symbol': symbol,
                            'action': 'SELL',
                            'strategy': strategy_name,
                            'price': current_data[symbol]['price'],
                            'strength': 1.0
                        }
                        all_signals.append(signal)

            except Exception as e:
                log.debug(f"Error generating signals: {str(e)}")

        return all_signals

    def _execute_trades(self, signals: List[Dict], market_data: Dict, date: datetime):
        """Execute trades based on signals."""
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']

            if action == 'BUY':
                # Calculate position size
                position_size = self._calculate_position_size(signal)

                if position_size > 0 and self.cash >= position_size * price:
                    # Execute buy
                    quantity = int(position_size)
                    trade_value = quantity * price
                    commission = trade_value * self.commission
                    slippage_cost = trade_value * self.slippage

                    total_cost = trade_value + commission + slippage_cost

                    if self.cash >= total_cost:
                        self.cash -= total_cost

                        if symbol not in self.positions:
                            self.positions[symbol] = {
                                'quantity': 0,
                                'avg_price': 0
                            }

                        # Update position
                        total_quantity = self.positions[symbol]['quantity'] + quantity
                        total_value = (self.positions[symbol]['quantity'] * self.positions[symbol]['avg_price'] +
                                     quantity * price)

                        self.positions[symbol]['quantity'] = total_quantity
                        self.positions[symbol]['avg_price'] = total_value / total_quantity

                        # Record trade
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': price,
                            'commission': commission,
                            'slippage': slippage_cost,
                            'strategy': signal.get('strategy')
                        })

            elif action == 'SELL' and symbol in self.positions:
                # Execute sell
                quantity = self.positions[symbol]['quantity']
                if quantity > 0:
                    trade_value = quantity * price
                    commission = trade_value * self.commission
                    slippage_cost = trade_value * self.slippage

                    net_proceeds = trade_value - commission - slippage_cost
                    self.cash += net_proceeds

                    # Calculate P&L
                    pnl = (price - self.positions[symbol]['avg_price']) * quantity - commission - slippage_cost

                    # Record trade
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': price,
                        'commission': commission,
                        'slippage': slippage_cost,
                        'pnl': pnl,
                        'strategy': signal.get('strategy')
                    })

                    # Remove position
                    del self.positions[symbol]

    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size for a signal."""
        # Simple equal weight position sizing
        max_positions = 10
        position_value = self.portfolio_value / max_positions
        strength = signal.get('strength', 1.0)

        return (position_value * strength) / signal['price']

    def _update_portfolio_value(self, market_data: Dict):
        """Update portfolio value based on current prices."""
        positions_value = 0

        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                positions_value += position['quantity'] * current_price

        self.portfolio_value = self.cash + positions_value

    def _calculate_results(self) -> BacktestResults:
        """Calculate backtest performance metrics."""
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)

        # Calculate returns
        equity_series = equity_df['portfolio_value']
        returns = equity_series.pct_change().dropna()

        # Calculate metrics
        total_returns = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

        # Annualized returns
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25
        annualized_returns = (1 + total_returns) ** (1/years) - 1 if years > 0 else 0

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe(returns)

        # Sortino ratio
        sortino_ratio = self._calculate_sortino(returns)

        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_series)

        # Trade statistics
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        if not trades_df.empty and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]

            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

            total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0

        # Additional metrics
        metrics = {
            'total_returns': total_returns,
            'annualized_returns': annualized_returns,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades_df),
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'calmar_ratio': annualized_returns / abs(max_drawdown) if max_drawdown != 0 else 0
        }

        return BacktestResults(
            total_returns=total_returns,
            annualized_returns=annualized_returns,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            equity_curve=equity_series,
            trades=trades_df,
            metrics=metrics
        )

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0

        excess_returns = returns - risk_free_rate / 252
        if returns.std() > 0:
            return np.mean(excess_returns) / returns.std() * np.sqrt(252)
        return 0

    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) > 0 and downside_returns.std() > 0:
            return np.mean(excess_returns) / downside_returns.std() * np.sqrt(252)
        return 0

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) == 0:
            return 0

        cumulative_returns = equity_curve / equity_curve.iloc[0]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _generate_empty_results(self) -> BacktestResults:
        """Generate empty results when no data is available."""
        return BacktestResults(
            total_returns=0,
            annualized_returns=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            equity_curve=pd.Series(),
            trades=pd.DataFrame(),
            metrics={}
        )

    def display_results(self, results: BacktestResults):
        """Display backtest results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Returns: {results.total_returns:.2%}")
        print(f"Annualized Returns: {results.annualized_returns:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Total Trades: {results.total_trades}")
        print("="*60)

    async def stop(self):
        """Stop backtest engine."""
        log.info("Stopping backtest engine...")
        # Cleanup if needed
        log.info("Backtest engine stopped")
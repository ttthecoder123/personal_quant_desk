"""
Performance Tracker

Real-time performance monitoring and metrics calculation for strategies.
Computes Sharpe ratio, Sortino ratio, drawdown, and other key metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a strategy or portfolio.

    Attributes:
        total_return: Cumulative return
        annualized_return: Annualized return
        sharpe_ratio: Sharpe ratio (target > 1.0)
        sortino_ratio: Sortino ratio
        max_drawdown: Maximum drawdown (limit 20%)
        current_drawdown: Current drawdown
        volatility: Annualized volatility (target 20%)
        win_rate: Percentage of winning trades
        profit_factor: Gross profit / gross loss
        avg_win: Average winning trade
        avg_loss: Average losing trade
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        avg_holding_period: Average holding period (hours)
        kelly_criterion: Calculated Kelly fraction
        calmar_ratio: Return / max drawdown
        var_95: Value at Risk (95% confidence)
        cvar_95: Conditional Value at Risk (95%)
    """
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_holding_period: float = 0.0
    kelly_criterion: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0


class PerformanceTracker:
    """
    Tracks and calculates performance metrics in real-time.

    Maintains equity curve, trade history, and computes rolling metrics.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize performance tracker.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.returns: List[float] = []
        self.trades: List[Dict] = []
        self.daily_returns: pd.Series = pd.Series(dtype=float)

        logger.info(f"Performance Tracker initialized with ${initial_capital:,.2f}")

    def update_equity(self, timestamp: datetime, equity: float):
        """
        Update equity curve with new value.

        Args:
            timestamp: Current timestamp
            equity: Current equity value
        """
        self.equity_curve.append((timestamp, equity))
        self.current_capital = equity

        # Calculate return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2][1]
            ret = (equity - prev_equity) / prev_equity
            self.returns.append(ret)

    def record_trade(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        size: float,
        side: int,
        pnl: float,
        strategy: str
    ):
        """
        Record a completed trade.

        Args:
            symbol: Instrument symbol
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            side: Position side (1 for long, -1 for short)
            pnl: Realized P&L
            strategy: Strategy name
        """
        holding_period = (exit_time - entry_time).total_seconds() / 3600  # hours

        trade = {
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'side': side,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * size)) * 100,
            'holding_period': holding_period,
            'strategy': strategy
        }

        self.trades.append(trade)

    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        if not self.equity_curve:
            return metrics

        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)

        # Total and annualized return
        metrics.total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        if len(df) > 1:
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            if days > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (252 / days) - 1

        # Calculate returns
        returns = df['equity'].pct_change().dropna()

        if len(returns) > 0:
            # Volatility
            metrics.volatility = returns.std() * np.sqrt(252)  # Annualized

            # Sharpe Ratio (assuming 0% risk-free rate)
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility

            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_dev = downside_returns.std() * np.sqrt(252)
                if downside_dev > 0:
                    metrics.sortino_ratio = metrics.annualized_return / downside_dev

            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            metrics.max_drawdown = abs(drawdown.min())
            metrics.current_drawdown = abs(drawdown.iloc[-1])

            # Calmar Ratio
            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

            # VaR and CVaR (95% confidence)
            metrics.var_95 = returns.quantile(0.05)
            metrics.cvar_95 = returns[returns <= metrics.var_95].mean()

        # Trade statistics
        if self.trades:
            metrics.total_trades = len(self.trades)
            trades_df = pd.DataFrame(self.trades)

            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]

            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)

            # Win rate
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades

            # Average win/loss
            if len(winning_trades) > 0:
                metrics.avg_win = winning_trades['pnl'].mean()
            if len(losing_trades) > 0:
                metrics.avg_loss = abs(losing_trades['pnl'].mean())

            # Profit factor
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0

            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss

            # Average holding period
            metrics.avg_holding_period = trades_df['holding_period'].mean()

            # Kelly Criterion
            if metrics.win_rate > 0 and metrics.avg_loss > 0:
                win_loss_ratio = metrics.avg_win / metrics.avg_loss if metrics.avg_loss > 0 else 0
                metrics.kelly_criterion = metrics.win_rate - ((1 - metrics.win_rate) / win_loss_ratio)
                # Cap Kelly at 25% for safety
                metrics.kelly_criterion = min(metrics.kelly_criterion, 0.25)

        return metrics

    def get_equity_curve_df(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Returns:
            DataFrame with timestamp and equity columns
        """
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        return df

    def get_trades_df(self) -> pd.DataFrame:
        """
        Get trades as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_rolling_sharpe(self, window: int = 60) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            window: Rolling window size

        Returns:
            Series of rolling Sharpe ratios
        """
        if len(self.returns) < window:
            return pd.Series(dtype=float)

        returns_series = pd.Series(self.returns)
        rolling_mean = returns_series.rolling(window).mean()
        rolling_std = returns_series.rolling(window).std()

        # Annualize
        rolling_sharpe = (rolling_mean * np.sqrt(252)) / (rolling_std * np.sqrt(252))
        return rolling_sharpe

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        Calculate monthly returns.

        Returns:
            DataFrame with monthly returns
        """
        df = self.get_equity_curve_df()
        if df.empty:
            return pd.DataFrame()

        df['month'] = df.index.to_period('M')
        monthly = df.groupby('month').agg({
            'equity': ['first', 'last']
        })
        monthly['return'] = (monthly['equity']['last'] - monthly['equity']['first']) / monthly['equity']['first']

        return monthly[['return']]

    def get_drawdown_periods(self) -> List[Dict]:
        """
        Identify drawdown periods.

        Returns:
            List of drawdown period dictionaries
        """
        df = self.get_equity_curve_df()
        if df.empty:
            return []

        returns = df['equity'].pct_change()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        dd_start = None
        dd_max = 0

        for idx, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                dd_start = idx
                dd_max = dd
            elif dd < 0 and in_drawdown:
                dd_max = min(dd_max, dd)
            elif dd >= 0 and in_drawdown:
                drawdown_periods.append({
                    'start': dd_start,
                    'end': idx,
                    'max_drawdown': abs(dd_max),
                    'duration_days': (idx - dd_start).total_seconds() / 86400
                })
                in_drawdown = False
                dd_max = 0

        return drawdown_periods

    def get_performance_summary(self) -> str:
        """
        Get formatted performance summary.

        Returns:
            Formatted string with key metrics
        """
        metrics = self.calculate_metrics()

        summary = f"""
Performance Summary
==================
Total Return:        {metrics.total_return:>10.2%}
Annualized Return:   {metrics.annualized_return:>10.2%}
Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}
Sortino Ratio:       {metrics.sortino_ratio:>10.2f}
Max Drawdown:        {metrics.max_drawdown:>10.2%}
Current Drawdown:    {metrics.current_drawdown:>10.2%}
Volatility:          {metrics.volatility:>10.2%}

Trade Statistics
================
Total Trades:        {metrics.total_trades:>10}
Win Rate:            {metrics.win_rate:>10.2%}
Profit Factor:       {metrics.profit_factor:>10.2f}
Average Win:         ${metrics.avg_win:>10,.2f}
Average Loss:        ${metrics.avg_loss:>10,.2f}
Kelly Criterion:     {metrics.kelly_criterion:>10.2%}

Risk Metrics
============
VaR (95%):           {metrics.var_95:>10.2%}
CVaR (95%):          {metrics.cvar_95:>10.2%}
Calmar Ratio:        {metrics.calmar_ratio:>10.2f}
"""
        return summary

    def reset(self):
        """Reset tracker to initial state."""
        self.current_capital = self.initial_capital
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        self.returns = []
        self.trades = []
        logger.info("Performance Tracker reset")

    def __repr__(self) -> str:
        metrics = self.calculate_metrics()
        return (
            f"PerformanceTracker(capital=${self.current_capital:,.2f}, "
            f"return={metrics.total_return:.2%}, "
            f"sharpe={metrics.sharpe_ratio:.2f}, "
            f"trades={metrics.total_trades})"
        )

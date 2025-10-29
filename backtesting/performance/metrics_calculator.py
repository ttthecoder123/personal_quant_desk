"""
Comprehensive performance metrics calculator for backtesting system.

This module provides extensive performance metrics calculation including:
- Return metrics (total, CAGR, rolling)
- Risk metrics (volatility, downside deviation, tracking error)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar, Omega, Kappa)
- Drawdown analysis (maximum, duration, recovery, Ulcer index)
- Trade statistics (win rate, payoff ratio, profit factor)
- Higher moment analysis (skewness, kurtosis, tail ratios)
- Benchmark comparison (alpha, beta, correlation, information ratio)
"""

from typing import Dict, Optional, Union, Tuple, List
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

from utils.logger import get_backtest_logger

log = get_backtest_logger()


class MetricsCalculator:
    """
    Comprehensive metrics calculator for portfolio performance analysis.

    This class provides methods to calculate a wide range of performance metrics
    commonly used in quantitative finance and portfolio management.
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize the metrics calculator.

        Args:
            returns: Series of portfolio returns
            benchmark_returns: Optional series of benchmark returns
            risk_free_rate: Annual risk-free rate (default: 2%)
            periods_per_year: Trading periods per year (252 for daily, 12 for monthly)
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Validate inputs
        if len(self.returns) == 0:
            log.warning("Empty returns series provided to MetricsCalculator")

        # Align benchmark returns if provided
        if self.benchmark_returns is not None:
            common_idx = self.returns.index.intersection(self.benchmark_returns.index)
            self.returns = self.returns.loc[common_idx]
            self.benchmark_returns = self.benchmark_returns.loc[common_idx]

            if len(common_idx) < len(self.returns):
                log.warning(f"Aligned returns and benchmark: {len(common_idx)} common periods")

    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all available performance metrics.

        Returns:
            Dictionary containing all calculated metrics
        """
        log.info("Calculating comprehensive performance metrics")

        metrics = {}

        try:
            # Return metrics
            metrics.update(self.calculate_return_metrics())

            # Risk metrics
            metrics.update(self.calculate_risk_metrics())

            # Risk-adjusted metrics
            metrics.update(self.calculate_risk_adjusted_metrics())

            # Drawdown metrics
            metrics.update(self.calculate_drawdown_metrics())

            # Trade statistics (if applicable)
            metrics.update(self.calculate_distribution_metrics())

            # Benchmark comparison
            if self.benchmark_returns is not None:
                metrics.update(self.calculate_benchmark_metrics())

            log.success(f"Calculated {len(metrics)} performance metrics")

        except Exception as e:
            log.error(f"Error calculating metrics: {str(e)}")
            raise

        return metrics

    # ===========================
    # Return Metrics
    # ===========================

    def calculate_return_metrics(self) -> Dict[str, float]:
        """Calculate return-based performance metrics."""
        if len(self.returns) == 0:
            return self._empty_return_metrics()

        metrics = {
            'total_return': self.total_return(),
            'cagr': self.cagr(),
            'annualized_return': self.annualized_return(),
            'daily_mean_return': self.returns.mean(),
            'daily_median_return': self.returns.median(),
            'best_day': self.returns.max(),
            'worst_day': self.returns.min(),
            'monthly_mean_return': self.monthly_return_mean(),
            'positive_periods': self.positive_period_ratio(),
            'negative_periods': 1 - self.positive_period_ratio(),
        }

        return metrics

    def total_return(self) -> float:
        """Calculate cumulative total return."""
        if len(self.returns) == 0:
            return 0.0
        return (1 + self.returns).prod() - 1

    def cagr(self) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Returns:
            CAGR as a decimal
        """
        if len(self.returns) == 0:
            return 0.0

        total_return = self.total_return()
        years = len(self.returns) / self.periods_per_year

        if years <= 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1

    def annualized_return(self) -> float:
        """Calculate annualized return (arithmetic)."""
        if len(self.returns) == 0:
            return 0.0
        return self.returns.mean() * self.periods_per_year

    def monthly_return_mean(self) -> float:
        """Calculate mean monthly return."""
        if len(self.returns) == 0:
            return 0.0

        try:
            monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            return monthly_returns.mean()
        except Exception:
            return self.returns.mean() * 21  # Approximate

    def positive_period_ratio(self) -> float:
        """Calculate ratio of positive return periods."""
        if len(self.returns) == 0:
            return 0.0
        return (self.returns > 0).sum() / len(self.returns)

    # ===========================
    # Risk Metrics
    # ===========================

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        if len(self.returns) == 0:
            return self._empty_risk_metrics()

        metrics = {
            'volatility': self.volatility(),
            'annualized_volatility': self.annualized_volatility(),
            'downside_deviation': self.downside_deviation(),
            'semi_variance': self.semi_variance(),
            'upside_variance': self.upside_variance(),
            'downside_risk': self.downside_deviation(),
        }

        # Tracking error if benchmark provided
        if self.benchmark_returns is not None:
            metrics['tracking_error'] = self.tracking_error()
            metrics['active_risk'] = self.tracking_error()

        return metrics

    def volatility(self) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(self.returns) < 2:
            return 0.0
        return self.returns.std()

    def annualized_volatility(self) -> float:
        """Calculate annualized volatility."""
        return self.volatility() * np.sqrt(self.periods_per_year)

    def downside_deviation(self, mar: float = 0.0) -> float:
        """
        Calculate downside deviation (semi-deviation).

        Args:
            mar: Minimum acceptable return (default: 0)

        Returns:
            Downside deviation
        """
        if len(self.returns) == 0:
            return 0.0

        downside_returns = self.returns[self.returns < mar]
        if len(downside_returns) == 0:
            return 0.0

        return np.sqrt(np.mean((downside_returns - mar) ** 2)) * np.sqrt(self.periods_per_year)

    def semi_variance(self, mar: float = 0.0) -> float:
        """Calculate semi-variance (variance of negative returns)."""
        if len(self.returns) == 0:
            return 0.0

        downside_returns = self.returns[self.returns < mar]
        if len(downside_returns) == 0:
            return 0.0

        return np.var(downside_returns - mar)

    def upside_variance(self, mar: float = 0.0) -> float:
        """Calculate upside variance (variance of positive returns)."""
        if len(self.returns) == 0:
            return 0.0

        upside_returns = self.returns[self.returns > mar]
        if len(upside_returns) == 0:
            return 0.0

        return np.var(upside_returns - mar)

    def tracking_error(self) -> float:
        """Calculate tracking error relative to benchmark."""
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        active_returns = self.returns - self.benchmark_returns
        return active_returns.std() * np.sqrt(self.periods_per_year)

    # ===========================
    # Risk-Adjusted Metrics
    # ===========================

    def calculate_risk_adjusted_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        if len(self.returns) == 0:
            return self._empty_risk_adjusted_metrics()

        metrics = {
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'omega_ratio': self.omega_ratio(),
            'kappa_3_ratio': self.kappa_ratio(3),
            'gain_to_pain_ratio': self.gain_to_pain_ratio(),
        }

        # Information ratio if benchmark provided
        if self.benchmark_returns is not None:
            metrics['information_ratio'] = self.information_ratio()

        return metrics

    def sharpe_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            risk_free_rate: Override default risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(self.returns) < 2:
            return 0.0

        rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        excess_returns = self.returns - (rfr / self.periods_per_year)

        if self.returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / self.returns.std()) * np.sqrt(self.periods_per_year)

    def sortino_ratio(self, mar: float = 0.0) -> float:
        """
        Calculate Sortino ratio.

        Args:
            mar: Minimum acceptable return

        Returns:
            Sortino ratio
        """
        if len(self.returns) < 2:
            return 0.0

        excess_returns = self.returns - (self.risk_free_rate / self.periods_per_year)
        downside_dev = self.downside_deviation(mar)

        if downside_dev == 0:
            return 0.0

        return (excess_returns.mean() * self.periods_per_year) / downside_dev

    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (CAGR / Maximum Drawdown)."""
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0
        return self.cagr() / max_dd

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Args:
            threshold: Threshold return

        Returns:
            Omega ratio
        """
        if len(self.returns) == 0:
            return 0.0

        returns_above = self.returns[self.returns > threshold] - threshold
        returns_below = threshold - self.returns[self.returns < threshold]

        if len(returns_below) == 0 or returns_below.sum() == 0:
            return float('inf') if len(returns_above) > 0 else 0.0

        return returns_above.sum() / returns_below.sum()

    def kappa_ratio(self, n: int = 3, mar: float = 0.0) -> float:
        """
        Calculate Kappa ratio (generalized Sortino).

        Args:
            n: Moment order (3 for skewness, 4 for kurtosis)
            mar: Minimum acceptable return

        Returns:
            Kappa ratio
        """
        if len(self.returns) < 2:
            return 0.0

        excess_returns = self.returns - (self.risk_free_rate / self.periods_per_year)
        downside_returns = self.returns[self.returns < mar] - mar

        if len(downside_returns) == 0:
            return 0.0

        lower_partial_moment = ((-downside_returns) ** n).mean() ** (1/n)

        if lower_partial_moment == 0:
            return 0.0

        return (excess_returns.mean() * self.periods_per_year) / (lower_partial_moment * np.sqrt(self.periods_per_year))

    def gain_to_pain_ratio(self) -> float:
        """Calculate gain-to-pain ratio."""
        if len(self.returns) == 0:
            return 0.0

        total_return = self.total_return()
        pain = abs(self.returns[self.returns < 0]).sum()

        if pain == 0:
            return float('inf') if total_return > 0 else 0.0

        return total_return / pain

    def information_ratio(self) -> float:
        """Calculate information ratio relative to benchmark."""
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        active_returns = self.returns - self.benchmark_returns
        tracking_err = self.tracking_error()

        if tracking_err == 0:
            return 0.0

        return (active_returns.mean() * self.periods_per_year) / tracking_err

    # ===========================
    # Drawdown Metrics
    # ===========================

    def calculate_drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown-based metrics."""
        if len(self.returns) == 0:
            return self._empty_drawdown_metrics()

        dd_series = self.drawdown_series()

        metrics = {
            'max_drawdown': self.max_drawdown(),
            'avg_drawdown': dd_series.mean(),
            'max_drawdown_duration': self.max_drawdown_duration(),
            'recovery_factor': self.recovery_factor(),
            'ulcer_index': self.ulcer_index(),
        }

        return metrics

    def drawdown_series(self) -> pd.Series:
        """Calculate drawdown series."""
        if len(self.returns) == 0:
            return pd.Series()

        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        return drawdown

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        dd_series = self.drawdown_series()
        if len(dd_series) == 0:
            return 0.0
        return dd_series.min()

    def max_drawdown_duration(self) -> int:
        """
        Calculate maximum drawdown duration in periods.

        Returns:
            Number of periods for longest drawdown
        """
        if len(self.returns) == 0:
            return 0

        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()

        # Find periods where we're at new highs
        at_high = cum_returns == running_max

        # Calculate duration of each drawdown
        max_duration = 0
        current_duration = 0

        for is_at_high in at_high:
            if is_at_high:
                max_duration = max(max_duration, current_duration)
                current_duration = 0
            else:
                current_duration += 1

        max_duration = max(max_duration, current_duration)

        return max_duration

    def recovery_factor(self) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0
        return self.total_return() / max_dd

    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index (measure of downside volatility).

        Returns:
            Ulcer index
        """
        if len(self.returns) == 0:
            return 0.0

        dd_series = self.drawdown_series()
        squared_dd = dd_series ** 2

        return np.sqrt(squared_dd.mean()) * 100

    # ===========================
    # Distribution Metrics
    # ===========================

    def calculate_distribution_metrics(self) -> Dict[str, float]:
        """Calculate distribution-based metrics."""
        if len(self.returns) < 4:
            return self._empty_distribution_metrics()

        metrics = {
            'skewness': self.skewness(),
            'kurtosis': self.kurtosis(),
            'excess_kurtosis': self.excess_kurtosis(),
            'tail_ratio': self.tail_ratio(),
            'value_at_risk_95': self.value_at_risk(0.05),
            'conditional_var_95': self.conditional_var(0.05),
        }

        return metrics

    def skewness(self) -> float:
        """Calculate skewness of returns distribution."""
        if len(self.returns) < 3:
            return 0.0
        return stats.skew(self.returns, bias=False)

    def kurtosis(self) -> float:
        """Calculate kurtosis of returns distribution."""
        if len(self.returns) < 4:
            return 0.0
        return stats.kurtosis(self.returns, bias=False, fisher=False)

    def excess_kurtosis(self) -> float:
        """Calculate excess kurtosis (kurtosis - 3)."""
        if len(self.returns) < 4:
            return 0.0
        return stats.kurtosis(self.returns, bias=False, fisher=True)

    def tail_ratio(self, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio (right tail / left tail).

        Args:
            percentile: Percentile for tail definition

        Returns:
            Tail ratio
        """
        if len(self.returns) < 20:
            return 0.0

        right_tail = self.returns.quantile(1 - percentile)
        left_tail = abs(self.returns.quantile(percentile))

        if left_tail == 0:
            return 0.0

        return right_tail / left_tail

    def value_at_risk(self, alpha: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR) using historical method.

        Args:
            alpha: Confidence level (0.05 for 95% VaR)

        Returns:
            VaR value (positive number)
        """
        if len(self.returns) == 0:
            return 0.0

        return abs(self.returns.quantile(alpha))

    def conditional_var(self, alpha: float = 0.05) -> float:
        """
        Calculate Conditional VaR (CVaR/Expected Shortfall).

        Args:
            alpha: Confidence level

        Returns:
            CVaR value (positive number)
        """
        if len(self.returns) == 0:
            return 0.0

        var = self.returns.quantile(alpha)
        return abs(self.returns[self.returns <= var].mean())

    # ===========================
    # Benchmark Comparison
    # ===========================

    def calculate_benchmark_metrics(self) -> Dict[str, float]:
        """Calculate metrics relative to benchmark."""
        if self.benchmark_returns is None or len(self.returns) < 2:
            return {}

        alpha, beta = self.alpha_beta()

        metrics = {
            'alpha': alpha,
            'beta': beta,
            'correlation': self.correlation(),
            'r_squared': self.r_squared(),
            'information_ratio': self.information_ratio(),
            'tracking_error': self.tracking_error(),
            'active_return': self.active_return(),
            'up_capture': self.up_capture_ratio(),
            'down_capture': self.down_capture_ratio(),
            'capture_ratio': self.capture_ratio(),
        }

        return metrics

    def alpha_beta(self) -> Tuple[float, float]:
        """
        Calculate alpha and beta using CAPM.

        Returns:
            Tuple of (alpha, beta)
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0, 0.0

        # Calculate beta
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)

        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance

        # Calculate alpha
        portfolio_return = self.returns.mean() * self.periods_per_year
        benchmark_return = self.benchmark_returns.mean() * self.periods_per_year
        rf_rate = self.risk_free_rate

        alpha = portfolio_return - (rf_rate + beta * (benchmark_return - rf_rate))

        return alpha, beta

    def correlation(self) -> float:
        """Calculate correlation with benchmark."""
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        return self.returns.corr(self.benchmark_returns)

    def r_squared(self) -> float:
        """Calculate R-squared relative to benchmark."""
        return self.correlation() ** 2

    def active_return(self) -> float:
        """Calculate annualized active return vs benchmark."""
        if self.benchmark_returns is None:
            return 0.0

        active_returns = self.returns - self.benchmark_returns
        return active_returns.mean() * self.periods_per_year

    def up_capture_ratio(self) -> float:
        """Calculate up-capture ratio."""
        if self.benchmark_returns is None or len(self.returns) == 0:
            return 0.0

        up_periods = self.benchmark_returns > 0

        if up_periods.sum() == 0:
            return 0.0

        portfolio_up = self.returns[up_periods].mean()
        benchmark_up = self.benchmark_returns[up_periods].mean()

        if benchmark_up == 0:
            return 0.0

        return portfolio_up / benchmark_up

    def down_capture_ratio(self) -> float:
        """Calculate down-capture ratio."""
        if self.benchmark_returns is None or len(self.returns) == 0:
            return 0.0

        down_periods = self.benchmark_returns < 0

        if down_periods.sum() == 0:
            return 0.0

        portfolio_down = self.returns[down_periods].mean()
        benchmark_down = self.benchmark_returns[down_periods].mean()

        if benchmark_down == 0:
            return 0.0

        return portfolio_down / benchmark_down

    def capture_ratio(self) -> float:
        """Calculate capture ratio (up-capture / down-capture)."""
        down_capture = self.down_capture_ratio()

        if down_capture == 0:
            return 0.0

        return self.up_capture_ratio() / abs(down_capture)

    # ===========================
    # Rolling Metrics
    # ===========================

    def rolling_metrics(
        self,
        window: int = 252,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics over specified window.

        Args:
            window: Rolling window size
            metrics: List of metrics to calculate (None for all)

        Returns:
            DataFrame with rolling metrics
        """
        if len(self.returns) < window:
            log.warning(f"Insufficient data for rolling metrics (need {window}, have {len(self.returns)})")
            return pd.DataFrame()

        if metrics is None:
            metrics = ['sharpe', 'sortino', 'volatility', 'max_dd']

        rolling_data = {}

        for metric in metrics:
            if metric == 'sharpe':
                rolling_data['sharpe_ratio'] = self._rolling_sharpe(window)
            elif metric == 'sortino':
                rolling_data['sortino_ratio'] = self._rolling_sortino(window)
            elif metric == 'volatility':
                rolling_data['volatility'] = self.returns.rolling(window).std() * np.sqrt(self.periods_per_year)
            elif metric == 'max_dd':
                rolling_data['max_drawdown'] = self._rolling_max_drawdown(window)
            elif metric == 'calmar':
                rolling_data['calmar_ratio'] = self._rolling_calmar(window)

        return pd.DataFrame(rolling_data, index=self.returns.index)

    def _rolling_sharpe(self, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        excess_returns = self.returns - (self.risk_free_rate / self.periods_per_year)
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = self.returns.rolling(window).std()

        return (rolling_mean / rolling_std) * np.sqrt(self.periods_per_year)

    def _rolling_sortino(self, window: int) -> pd.Series:
        """Calculate rolling Sortino ratio."""
        excess_returns = self.returns - (self.risk_free_rate / self.periods_per_year)

        def downside_std(x):
            downside = x[x < 0]
            return downside.std() if len(downside) > 0 else 0

        rolling_mean = excess_returns.rolling(window).mean()
        rolling_dd = self.returns.rolling(window).apply(downside_std, raw=False)

        return (rolling_mean * self.periods_per_year) / (rolling_dd * np.sqrt(self.periods_per_year))

    def _rolling_max_drawdown(self, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        def max_dd(returns_window):
            cum_returns = (1 + returns_window).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            return drawdown.min()

        return self.returns.rolling(window).apply(max_dd, raw=False)

    def _rolling_calmar(self, window: int) -> pd.Series:
        """Calculate rolling Calmar ratio."""
        rolling_return = self.returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (self.periods_per_year / len(x)) - 1,
            raw=False
        )
        rolling_dd = self._rolling_max_drawdown(window)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calmar = rolling_return / abs(rolling_dd)
            calmar = calmar.replace([np.inf, -np.inf], np.nan)

        return calmar

    # ===========================
    # Helper Methods
    # ===========================

    def _empty_return_metrics(self) -> Dict[str, float]:
        """Return empty return metrics dictionary."""
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'annualized_return': 0.0,
            'daily_mean_return': 0.0,
            'daily_median_return': 0.0,
            'best_day': 0.0,
            'worst_day': 0.0,
            'monthly_mean_return': 0.0,
            'positive_periods': 0.0,
            'negative_periods': 0.0,
        }

    def _empty_risk_metrics(self) -> Dict[str, float]:
        """Return empty risk metrics dictionary."""
        return {
            'volatility': 0.0,
            'annualized_volatility': 0.0,
            'downside_deviation': 0.0,
            'semi_variance': 0.0,
            'upside_variance': 0.0,
            'downside_risk': 0.0,
        }

    def _empty_risk_adjusted_metrics(self) -> Dict[str, float]:
        """Return empty risk-adjusted metrics dictionary."""
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'omega_ratio': 0.0,
            'kappa_3_ratio': 0.0,
            'gain_to_pain_ratio': 0.0,
        }

    def _empty_drawdown_metrics(self) -> Dict[str, float]:
        """Return empty drawdown metrics dictionary."""
        return {
            'max_drawdown': 0.0,
            'avg_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'recovery_factor': 0.0,
            'ulcer_index': 0.0,
        }

    def _empty_distribution_metrics(self) -> Dict[str, float]:
        """Return empty distribution metrics dictionary."""
        return {
            'skewness': 0.0,
            'kurtosis': 0.0,
            'excess_kurtosis': 0.0,
            'tail_ratio': 0.0,
            'value_at_risk_95': 0.0,
            'conditional_var_95': 0.0,
        }


def calculate_trade_statistics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trade-level statistics.

    Args:
        trades_df: DataFrame with trade data including 'pnl' column

    Returns:
        Dictionary of trade statistics
    """
    if trades_df.empty or 'pnl' not in trades_df.columns:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'payoff_ratio': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
        }

    trades = trades_df['pnl'].dropna()
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]

    total_trades = len(trades)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)

    win_rate = num_winning / total_trades if total_trades > 0 else 0.0

    avg_win = winning_trades.mean() if num_winning > 0 else 0.0
    avg_loss = abs(losing_trades.mean()) if num_losing > 0 else 0.0
    avg_trade = trades.mean()

    largest_win = winning_trades.max() if num_winning > 0 else 0.0
    largest_loss = abs(losing_trades.min()) if num_losing > 0 else 0.0

    payoff_ratio = avg_win / avg_loss if avg_loss != 0 else 0.0

    total_profit = winning_trades.sum() if num_winning > 0 else 0.0
    total_loss = abs(losing_trades.sum()) if num_losing > 0 else 0.0
    profit_factor = total_profit / total_loss if total_loss != 0 else 0.0

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return {
        'total_trades': total_trades,
        'winning_trades': num_winning,
        'losing_trades': num_losing,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade': avg_trade,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'payoff_ratio': payoff_ratio,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
    }

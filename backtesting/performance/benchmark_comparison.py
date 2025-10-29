"""
Benchmark comparison and analysis for portfolio performance.

This module provides comprehensive benchmark analysis including:
- Absolute and risk-adjusted performance comparison
- Rolling comparison metrics
- Relative strength analysis
- Outperformance periods and statistical significance
- CAPM analysis (alpha, beta, R-squared)
- Up/down capture ratios
- Tracking error analysis
"""

from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import warnings

from utils.logger import get_backtest_logger

log = get_backtest_logger()


class BenchmarkComparison:
    """
    Comprehensive benchmark comparison and analysis.

    Compares portfolio performance against one or more benchmarks
    using various metrics and statistical tests.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        portfolio_name: str = "Portfolio",
        benchmark_name: str = "Benchmark"
    ):
        """
        Initialize benchmark comparison.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            portfolio_name: Name for portfolio
            benchmark_name: Name for benchmark
        """
        # Align returns
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)

        self.portfolio_returns = portfolio_returns.loc[common_idx].dropna()
        self.benchmark_returns = benchmark_returns.loc[common_idx].dropna()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.portfolio_name = portfolio_name
        self.benchmark_name = benchmark_name

        # Calculate active returns
        self.active_returns = self.portfolio_returns - self.benchmark_returns

        if len(self.portfolio_returns) == 0:
            log.warning("No overlapping data between portfolio and benchmark")

        log.info(f"Initialized BenchmarkComparison: {portfolio_name} vs {benchmark_name}")

    def calculate_all_comparisons(self) -> Dict[str, float]:
        """
        Calculate all benchmark comparison metrics.

        Returns:
            Dictionary of comparison metrics
        """
        log.info("Calculating comprehensive benchmark comparisons")

        metrics = {}

        try:
            # Absolute performance
            metrics.update(self.absolute_performance())

            # Risk-adjusted performance
            metrics.update(self.risk_adjusted_comparison())

            # CAPM analysis
            metrics.update(self.capm_analysis())

            # Capture ratios
            metrics.update(self.capture_ratios())

            # Tracking analysis
            metrics.update(self.tracking_analysis())

            # Statistical tests
            metrics.update(self.statistical_tests())

            log.success(f"Calculated {len(metrics)} comparison metrics")

        except Exception as e:
            log.error(f"Error calculating benchmark comparisons: {str(e)}")
            raise

        return metrics

    # ===========================
    # Absolute Performance
    # ===========================

    def absolute_performance(self) -> Dict[str, float]:
        """
        Calculate absolute performance metrics for both portfolio and benchmark.

        Returns:
            Dictionary of absolute performance metrics
        """
        metrics = {}

        # Total returns
        portfolio_total = (1 + self.portfolio_returns).prod() - 1
        benchmark_total = (1 + self.benchmark_returns).prod() - 1

        metrics['portfolio_total_return'] = portfolio_total
        metrics['benchmark_total_return'] = benchmark_total
        metrics['excess_return'] = portfolio_total - benchmark_total

        # Annualized returns
        years = len(self.portfolio_returns) / self.periods_per_year
        if years > 0:
            metrics['portfolio_cagr'] = (1 + portfolio_total) ** (1 / years) - 1
            metrics['benchmark_cagr'] = (1 + benchmark_total) ** (1 / years) - 1
        else:
            metrics['portfolio_cagr'] = 0.0
            metrics['benchmark_cagr'] = 0.0

        # Volatility
        metrics['portfolio_volatility'] = (
            self.portfolio_returns.std() * np.sqrt(self.periods_per_year)
        )
        metrics['benchmark_volatility'] = (
            self.benchmark_returns.std() * np.sqrt(self.periods_per_year)
        )

        # Correlation
        metrics['correlation'] = self.portfolio_returns.corr(self.benchmark_returns)

        return metrics

    def cumulative_performance(self) -> pd.DataFrame:
        """
        Calculate cumulative performance for both portfolio and benchmark.

        Returns:
            DataFrame with cumulative returns
        """
        portfolio_cum = (1 + self.portfolio_returns).cumprod()
        benchmark_cum = (1 + self.benchmark_returns).cumprod()

        df = pd.DataFrame({
            self.portfolio_name: portfolio_cum,
            self.benchmark_name: benchmark_cum,
            'Relative': portfolio_cum / benchmark_cum,
        })

        return df

    # ===========================
    # Risk-Adjusted Performance
    # ===========================

    def risk_adjusted_comparison(self) -> Dict[str, float]:
        """
        Compare risk-adjusted performance metrics.

        Returns:
            Dictionary of risk-adjusted metrics
        """
        metrics = {}

        # Sharpe ratios
        metrics['portfolio_sharpe'] = self._sharpe_ratio(self.portfolio_returns)
        metrics['benchmark_sharpe'] = self._sharpe_ratio(self.benchmark_returns)
        metrics['sharpe_diff'] = metrics['portfolio_sharpe'] - metrics['benchmark_sharpe']

        # Sortino ratios
        metrics['portfolio_sortino'] = self._sortino_ratio(self.portfolio_returns)
        metrics['benchmark_sortino'] = self._sortino_ratio(self.benchmark_returns)
        metrics['sortino_diff'] = metrics['portfolio_sortino'] - metrics['benchmark_sortino']

        # Calmar ratios
        metrics['portfolio_calmar'] = self._calmar_ratio(self.portfolio_returns)
        metrics['benchmark_calmar'] = self._calmar_ratio(self.benchmark_returns)
        metrics['calmar_diff'] = metrics['portfolio_calmar'] - metrics['benchmark_calmar']

        # Information ratio
        metrics['information_ratio'] = self._information_ratio()

        return metrics

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        return (excess_returns.mean() / returns.std()) * np.sqrt(self.periods_per_year)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return (
            (excess_returns.mean() * self.periods_per_year) /
            (downside_returns.std() * np.sqrt(self.periods_per_year))
        )

    def _calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) < 2:
            return 0.0

        # Calculate CAGR
        total_return = (1 + returns).prod() - 1
        years = len(returns) / self.periods_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = abs(drawdown.min())

        if max_dd == 0:
            return 0.0

        return cagr / max_dd

    def _information_ratio(self) -> float:
        """Calculate Information Ratio."""
        if len(self.active_returns) < 2:
            return 0.0

        tracking_error = self.active_returns.std() * np.sqrt(self.periods_per_year)

        if tracking_error == 0:
            return 0.0

        active_return = self.active_returns.mean() * self.periods_per_year

        return active_return / tracking_error

    # ===========================
    # CAPM Analysis
    # ===========================

    def capm_analysis(self) -> Dict[str, float]:
        """
        Perform CAPM analysis (alpha, beta, R-squared).

        Returns:
            Dictionary with CAPM metrics
        """
        if len(self.portfolio_returns) < 2:
            return {
                'alpha': 0.0,
                'alpha_annualized': 0.0,
                'beta': 0.0,
                'r_squared': 0.0,
                'residual_volatility': 0.0,
            }

        # Calculate beta
        covariance = np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1]
        benchmark_variance = self.benchmark_returns.var()

        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance

        # Calculate alpha
        portfolio_mean = self.portfolio_returns.mean() * self.periods_per_year
        benchmark_mean = self.benchmark_returns.mean() * self.periods_per_year
        rf_rate = self.risk_free_rate

        alpha = portfolio_mean - (rf_rate + beta * (benchmark_mean - rf_rate))

        # Calculate R-squared
        correlation = self.portfolio_returns.corr(self.benchmark_returns)
        r_squared = correlation ** 2

        # Calculate residual volatility (idiosyncratic risk)
        predicted_returns = beta * self.benchmark_returns
        residuals = self.portfolio_returns - predicted_returns
        residual_vol = residuals.std() * np.sqrt(self.periods_per_year)

        return {
            'alpha': alpha / self.periods_per_year,  # Daily alpha
            'alpha_annualized': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'residual_volatility': residual_vol,
        }

    def rolling_capm(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling CAPM metrics.

        Args:
            window: Rolling window size

        Returns:
            DataFrame with rolling alpha and beta
        """
        if len(self.portfolio_returns) < window:
            log.warning(f"Insufficient data for rolling CAPM (need {window})")
            return pd.DataFrame()

        rolling_data = {}

        # Rolling beta
        rolling_cov = self.portfolio_returns.rolling(window).cov(self.benchmark_returns)
        rolling_var = self.benchmark_returns.rolling(window).var()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rolling_beta = rolling_cov / rolling_var

        rolling_data['beta'] = rolling_beta

        # Rolling alpha
        portfolio_rolling_mean = self.portfolio_returns.rolling(window).mean() * self.periods_per_year
        benchmark_rolling_mean = self.benchmark_returns.rolling(window).mean() * self.periods_per_year

        rolling_alpha = portfolio_rolling_mean - (
            self.risk_free_rate + rolling_beta * (benchmark_rolling_mean - self.risk_free_rate)
        )

        rolling_data['alpha'] = rolling_alpha

        # Rolling R-squared
        rolling_corr = self.portfolio_returns.rolling(window).corr(self.benchmark_returns)
        rolling_data['r_squared'] = rolling_corr ** 2

        return pd.DataFrame(rolling_data)

    # ===========================
    # Capture Ratios
    # ===========================

    def capture_ratios(self) -> Dict[str, float]:
        """
        Calculate up-capture and down-capture ratios.

        Returns:
            Dictionary with capture ratios
        """
        # Up capture (when benchmark is positive)
        up_periods = self.benchmark_returns > 0
        if up_periods.sum() > 0:
            portfolio_up = self.portfolio_returns[up_periods].mean()
            benchmark_up = self.benchmark_returns[up_periods].mean()
            up_capture = portfolio_up / benchmark_up if benchmark_up != 0 else 0
        else:
            up_capture = 0.0

        # Down capture (when benchmark is negative)
        down_periods = self.benchmark_returns < 0
        if down_periods.sum() > 0:
            portfolio_down = self.portfolio_returns[down_periods].mean()
            benchmark_down = self.benchmark_returns[down_periods].mean()
            down_capture = portfolio_down / benchmark_down if benchmark_down != 0 else 0
        else:
            down_capture = 0.0

        # Capture ratio (up / down)
        if down_capture != 0:
            capture_ratio = up_capture / abs(down_capture)
        else:
            capture_ratio = 0.0

        return {
            'up_capture': up_capture,
            'down_capture': down_capture,
            'capture_ratio': capture_ratio,
        }

    def rolling_capture_ratios(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling capture ratios.

        Args:
            window: Rolling window size

        Returns:
            DataFrame with rolling capture ratios
        """
        if len(self.portfolio_returns) < window:
            return pd.DataFrame()

        def calc_up_capture(idx):
            if idx < window:
                return np.nan

            portfolio_window = self.portfolio_returns.iloc[idx-window:idx]
            benchmark_window = self.benchmark_returns.iloc[idx-window:idx]

            up_periods = benchmark_window > 0
            if up_periods.sum() == 0:
                return np.nan

            portfolio_up = portfolio_window[up_periods].mean()
            benchmark_up = benchmark_window[up_periods].mean()

            return portfolio_up / benchmark_up if benchmark_up != 0 else np.nan

        def calc_down_capture(idx):
            if idx < window:
                return np.nan

            portfolio_window = self.portfolio_returns.iloc[idx-window:idx]
            benchmark_window = self.benchmark_returns.iloc[idx-window:idx]

            down_periods = benchmark_window < 0
            if down_periods.sum() == 0:
                return np.nan

            portfolio_down = portfolio_window[down_periods].mean()
            benchmark_down = benchmark_window[down_periods].mean()

            return portfolio_down / benchmark_down if benchmark_down != 0 else np.nan

        rolling_data = {
            'up_capture': [calc_up_capture(i) for i in range(len(self.portfolio_returns))],
            'down_capture': [calc_down_capture(i) for i in range(len(self.portfolio_returns))],
        }

        df = pd.DataFrame(rolling_data, index=self.portfolio_returns.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['capture_ratio'] = df['up_capture'] / abs(df['down_capture'])

        return df

    # ===========================
    # Tracking Analysis
    # ===========================

    def tracking_analysis(self) -> Dict[str, float]:
        """
        Analyze tracking error and active risk.

        Returns:
            Dictionary with tracking metrics
        """
        # Tracking error
        tracking_error = self.active_returns.std() * np.sqrt(self.periods_per_year)

        # Active return
        active_return = self.active_returns.mean() * self.periods_per_year

        # Information ratio (already calculated, but included for completeness)
        information_ratio = active_return / tracking_error if tracking_error != 0 else 0

        # Active share (approximation based on returns correlation)
        # True active share requires holdings data
        correlation = self.portfolio_returns.corr(self.benchmark_returns)
        active_share_proxy = 1 - correlation

        # Maximum active deviation
        max_active_deviation = abs(self.active_returns).max()

        return {
            'tracking_error': tracking_error,
            'active_return': active_return,
            'information_ratio': information_ratio,
            'active_share_proxy': active_share_proxy,
            'max_active_deviation': max_active_deviation,
        }

    def rolling_tracking_error(self, window: int = 252) -> pd.Series:
        """
        Calculate rolling tracking error.

        Args:
            window: Rolling window size

        Returns:
            Series of rolling tracking error
        """
        if len(self.active_returns) < window:
            return pd.Series()

        rolling_te = self.active_returns.rolling(window).std() * np.sqrt(self.periods_per_year)
        return rolling_te

    # ===========================
    # Outperformance Analysis
    # ===========================

    def outperformance_periods(self) -> Dict[str, any]:
        """
        Analyze periods of outperformance and underperformance.

        Returns:
            Dictionary with outperformance statistics
        """
        # Count periods
        outperform_days = (self.active_returns > 0).sum()
        underperform_days = (self.active_returns < 0).sum()
        neutral_days = (self.active_returns == 0).sum()

        total_days = len(self.active_returns)

        # Percentages
        outperform_pct = outperform_days / total_days if total_days > 0 else 0
        underperform_pct = underperform_days / total_days if total_days > 0 else 0

        # Average magnitude of outperformance/underperformance
        avg_outperform = (
            self.active_returns[self.active_returns > 0].mean()
            if outperform_days > 0 else 0
        )
        avg_underperform = (
            self.active_returns[self.active_returns < 0].mean()
            if underperform_days > 0 else 0
        )

        # Longest streaks
        longest_outperform_streak = self._longest_streak(self.active_returns > 0)
        longest_underperform_streak = self._longest_streak(self.active_returns < 0)

        # Cumulative active return
        cumulative_active = self.active_returns.sum()

        return {
            'outperform_days': outperform_days,
            'underperform_days': underperform_days,
            'neutral_days': neutral_days,
            'outperform_pct': outperform_pct,
            'underperform_pct': underperform_pct,
            'avg_outperform': avg_outperform,
            'avg_underperform': avg_underperform,
            'longest_outperform_streak': longest_outperform_streak,
            'longest_underperform_streak': longest_underperform_streak,
            'cumulative_active_return': cumulative_active,
        }

    def _longest_streak(self, boolean_series: pd.Series) -> int:
        """Calculate longest streak of True values."""
        max_streak = 0
        current_streak = 0

        for value in boolean_series:
            if value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def relative_strength(self) -> pd.Series:
        """
        Calculate relative strength (portfolio cumulative / benchmark cumulative).

        Returns:
            Series of relative strength over time
        """
        portfolio_cum = (1 + self.portfolio_returns).cumprod()
        benchmark_cum = (1 + self.benchmark_returns).cumprod()

        relative_strength = portfolio_cum / benchmark_cum

        return relative_strength

    # ===========================
    # Statistical Tests
    # ===========================

    def statistical_tests(self) -> Dict[str, any]:
        """
        Perform statistical tests on performance difference.

        Returns:
            Dictionary with test results
        """
        results = {}

        # T-test: Is active return significantly different from zero?
        if len(self.active_returns) > 1:
            t_stat, p_value = stats.ttest_1samp(self.active_returns, 0)
            results['t_statistic'] = t_stat
            results['p_value'] = p_value
            results['significant_at_5pct'] = p_value < 0.05
            results['significant_at_1pct'] = p_value < 0.01
        else:
            results['t_statistic'] = np.nan
            results['p_value'] = np.nan
            results['significant_at_5pct'] = False
            results['significant_at_1pct'] = False

        # Sharpe ratio comparison test (Jobson-Korkie test)
        sharpe_diff = self._sharpe_ratio(self.portfolio_returns) - self._sharpe_ratio(self.benchmark_returns)
        results['sharpe_difference'] = sharpe_diff

        # Win rate test (binomial test)
        wins = (self.active_returns > 0).sum()
        total = len(self.active_returns)

        if total > 0:
            from scipy.stats import binom_test
            win_rate = wins / total
            # Test if win rate is significantly different from 0.5
            binom_p_value = binom_test(wins, total, 0.5, alternative='two-sided')

            results['win_rate'] = win_rate
            results['binom_p_value'] = binom_p_value
            results['win_rate_significant'] = binom_p_value < 0.05

        return results

    # ===========================
    # Period Analysis
    # ===========================

    def period_comparison(self, period: str = 'M') -> pd.DataFrame:
        """
        Compare performance by time period.

        Args:
            period: Time period ('M', 'Q', 'Y')

        Returns:
            DataFrame with period comparison
        """
        # Resample both series
        portfolio_period = self.portfolio_returns.resample(period).apply(
            lambda x: (1 + x).prod() - 1
        )
        benchmark_period = self.benchmark_returns.resample(period).apply(
            lambda x: (1 + x).prod() - 1
        )

        comparison = pd.DataFrame({
            self.portfolio_name: portfolio_period,
            self.benchmark_name: benchmark_period,
            'Active': portfolio_period - benchmark_period,
            'Outperformed': portfolio_period > benchmark_period,
        })

        return comparison

    def best_worst_periods(
        self,
        n: int = 10,
        period: str = 'M'
    ) -> Dict[str, pd.DataFrame]:
        """
        Identify best and worst periods for both portfolio and benchmark.

        Args:
            n: Number of periods to return
            period: Time period ('M', 'Q', 'Y')

        Returns:
            Dictionary with best/worst periods DataFrames
        """
        comparison = self.period_comparison(period)

        results = {
            'best_portfolio': comparison.nlargest(n, self.portfolio_name),
            'worst_portfolio': comparison.nsmallest(n, self.portfolio_name),
            'best_active': comparison.nlargest(n, 'Active'),
            'worst_active': comparison.nsmallest(n, 'Active'),
        }

        return results

    # ===========================
    # Summary Reports
    # ===========================

    def comparison_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive comparison summary.

        Returns:
            DataFrame with summary comparison
        """
        metrics = self.calculate_all_comparisons()

        summary_data = []

        # Performance metrics
        summary_data.append({
            'Category': 'Returns',
            'Metric': 'Total Return',
            self.portfolio_name: f"{metrics['portfolio_total_return']:.2%}",
            self.benchmark_name: f"{metrics['benchmark_total_return']:.2%}",
            'Difference': f"{metrics['excess_return']:.2%}",
        })

        summary_data.append({
            'Category': 'Returns',
            'Metric': 'CAGR',
            self.portfolio_name: f"{metrics['portfolio_cagr']:.2%}",
            self.benchmark_name: f"{metrics['benchmark_cagr']:.2%}",
            'Difference': f"{metrics['portfolio_cagr'] - metrics['benchmark_cagr']:.2%}",
        })

        # Risk metrics
        summary_data.append({
            'Category': 'Risk',
            'Metric': 'Volatility',
            self.portfolio_name: f"{metrics['portfolio_volatility']:.2%}",
            self.benchmark_name: f"{metrics['benchmark_volatility']:.2%}",
            'Difference': f"{metrics['portfolio_volatility'] - metrics['benchmark_volatility']:.2%}",
        })

        # Risk-adjusted
        summary_data.append({
            'Category': 'Risk-Adjusted',
            'Metric': 'Sharpe Ratio',
            self.portfolio_name: f"{metrics['portfolio_sharpe']:.2f}",
            self.benchmark_name: f"{metrics['benchmark_sharpe']:.2f}",
            'Difference': f"{metrics['sharpe_diff']:.2f}",
        })

        summary_data.append({
            'Category': 'Risk-Adjusted',
            'Metric': 'Information Ratio',
            self.portfolio_name: f"{metrics['information_ratio']:.2f}",
            self.benchmark_name: '-',
            'Difference': '-',
        })

        # CAPM
        summary_data.append({
            'Category': 'CAPM',
            'Metric': 'Alpha (Ann.)',
            self.portfolio_name: f"{metrics['alpha_annualized']:.2%}",
            self.benchmark_name: '-',
            'Difference': '-',
        })

        summary_data.append({
            'Category': 'CAPM',
            'Metric': 'Beta',
            self.portfolio_name: f"{metrics['beta']:.2f}",
            self.benchmark_name: '1.00',
            'Difference': f"{metrics['beta'] - 1:.2f}",
        })

        # Tracking
        summary_data.append({
            'Category': 'Tracking',
            'Metric': 'Tracking Error',
            self.portfolio_name: f"{metrics['tracking_error']:.2%}",
            self.benchmark_name: '-',
            'Difference': '-',
        })

        # Capture ratios
        summary_data.append({
            'Category': 'Capture',
            'Metric': 'Up Capture',
            self.portfolio_name: f"{metrics['up_capture']:.2%}",
            self.benchmark_name: '100.00%',
            'Difference': f"{(metrics['up_capture'] - 1) * 100:.2f}%",
        })

        summary_data.append({
            'Category': 'Capture',
            'Metric': 'Down Capture',
            self.portfolio_name: f"{metrics['down_capture']:.2%}",
            self.benchmark_name: '100.00%',
            'Difference': f"{(metrics['down_capture'] - 1) * 100:.2f}%",
        })

        return pd.DataFrame(summary_data)


def compare_multiple_benchmarks(
    portfolio_returns: pd.Series,
    benchmarks: Dict[str, pd.Series],
    portfolio_name: str = "Portfolio"
) -> Dict[str, BenchmarkComparison]:
    """
    Compare portfolio against multiple benchmarks.

    Args:
        portfolio_returns: Portfolio return series
        benchmarks: Dictionary of {name: returns} for benchmarks
        portfolio_name: Name for portfolio

    Returns:
        Dictionary of BenchmarkComparison objects
    """
    comparisons = {}

    for benchmark_name, benchmark_returns in benchmarks.items():
        log.info(f"Comparing against {benchmark_name}")

        comparison = BenchmarkComparison(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            portfolio_name=portfolio_name,
            benchmark_name=benchmark_name
        )

        comparisons[benchmark_name] = comparison

    return comparisons


def create_benchmark_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    output_path: Optional[str] = None,
    portfolio_name: str = "Portfolio",
    benchmark_name: str = "Benchmark"
) -> pd.DataFrame:
    """
    Create comprehensive benchmark comparison report.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        output_path: Optional path to save report
        portfolio_name: Portfolio name
        benchmark_name: Benchmark name

    Returns:
        Summary DataFrame
    """
    comparison = BenchmarkComparison(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        portfolio_name=portfolio_name,
        benchmark_name=benchmark_name
    )

    summary = comparison.comparison_summary()

    if output_path:
        try:
            summary.to_excel(output_path, index=False)
            log.success(f"Benchmark report saved to {output_path}")
        except Exception as e:
            log.error(f"Could not save benchmark report: {str(e)}")

    return summary

"""
Regime Analysis Module

Implements regime-specific performance analysis including:
- Performance in bull/bear markets
- Volatility regime performance (VIX-based)
- Crisis period analysis
- Correlation regime performance
- Seasonal performance patterns
- Day-of-week and month-of-year effects
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class RegimeResult:
    """Container for regime analysis results."""

    regime_name: str
    performance_metrics: Dict
    comparison_to_overall: Dict
    statistical_significance: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime_name': self.regime_name,
            'performance_metrics': self.performance_metrics,
            'comparison_to_overall': self.comparison_to_overall,
            'statistical_significance': self.statistical_significance
        }


class RegimeAnalyzer:
    """
    Comprehensive regime-specific performance analysis.

    Analyzes strategy performance across different market regimes
    to ensure robustness and identify potential weaknesses.
    """

    # Predefined crisis periods (can be extended)
    CRISIS_PERIODS = {
        'dot_com_crash': ('2000-03-01', '2002-10-01'),
        'financial_crisis': ('2007-10-01', '2009-03-01'),
        'flash_crash': ('2010-05-06', '2010-05-07'),
        'covid_crash': ('2020-02-19', '2020-03-23'),
        'covid_recovery': ('2020-03-23', '2020-12-31'),
        'rate_hike_2022': ('2022-01-01', '2022-12-31')
    }

    def __init__(self):
        """Initialize regime analyzer."""
        logger.info("Initialized RegimeAnalyzer")

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate performance metrics for a regime.

        Args:
            returns: Return series

        Returns:
            Dictionary of performance metrics
        """
        if len(returns) == 0:
            return {
                'n_observations': 0,
                'total_return': 0,
                'annualized_return': 0,
                'annualized_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        return {
            'n_observations': len(returns),
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }

    def _statistical_comparison(
        self,
        regime_returns: pd.Series,
        overall_returns: pd.Series
    ) -> Dict:
        """
        Perform statistical comparison between regime and overall performance.

        Args:
            regime_returns: Returns in specific regime
            overall_returns: Overall returns

        Returns:
            Dictionary with statistical test results
        """
        if len(regime_returns) < 2:
            return {
                'mean_difference': 0,
                't_statistic': 0,
                'p_value': 1.0,
                'significantly_different': False
            }

        # One-sample t-test comparing regime returns to overall mean
        overall_mean = overall_returns.mean()
        t_stat, p_value = stats.ttest_1samp(regime_returns, overall_mean)

        mean_diff = regime_returns.mean() - overall_mean

        return {
            'mean_difference': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significantly_different': p_value < 0.05,
            'better_than_average': mean_diff > 0
        }

    def bull_bear_analysis(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        bull_threshold: float = 0.0
    ) -> Dict[str, RegimeResult]:
        """
        Analyze performance in bull vs bear markets.

        Args:
            returns: Strategy returns
            market_returns: Market/benchmark returns
            bull_threshold: Threshold for bull market (default 0)

        Returns:
            Dictionary with bull and bear regime results
        """
        logger.debug("Analyzing bull/bear market performance")

        # Align returns
        aligned_returns = returns.reindex(market_returns.index).dropna()
        aligned_market = market_returns.reindex(aligned_returns.index)

        # Define regimes
        bull_mask = aligned_market > bull_threshold
        bear_mask = aligned_market <= bull_threshold

        bull_returns = aligned_returns[bull_mask]
        bear_returns = aligned_returns[bear_mask]

        # Calculate metrics
        bull_metrics = self._calculate_performance_metrics(bull_returns)
        bear_metrics = self._calculate_performance_metrics(bear_returns)
        overall_metrics = self._calculate_performance_metrics(aligned_returns)

        # Statistical comparison
        bull_comparison = self._statistical_comparison(bull_returns, aligned_returns)
        bear_comparison = self._statistical_comparison(bear_returns, aligned_returns)

        results = {
            'bull_market': RegimeResult(
                regime_name='Bull Market',
                performance_metrics=bull_metrics,
                comparison_to_overall={
                    'return_ratio': bull_metrics['annualized_return'] / overall_metrics['annualized_return']
                    if overall_metrics['annualized_return'] != 0 else 0,
                    'sharpe_ratio': bull_metrics['sharpe_ratio'] / overall_metrics['sharpe_ratio']
                    if overall_metrics['sharpe_ratio'] != 0 else 0
                },
                statistical_significance=bull_comparison
            ),
            'bear_market': RegimeResult(
                regime_name='Bear Market',
                performance_metrics=bear_metrics,
                comparison_to_overall={
                    'return_ratio': bear_metrics['annualized_return'] / overall_metrics['annualized_return']
                    if overall_metrics['annualized_return'] != 0 else 0,
                    'sharpe_ratio': bear_metrics['sharpe_ratio'] / overall_metrics['sharpe_ratio']
                    if overall_metrics['sharpe_ratio'] != 0 else 0
                },
                statistical_significance=bear_comparison
            )
        }

        logger.info(f"Bull market: {bull_metrics['n_observations']} periods, "
                   f"Bear market: {bear_metrics['n_observations']} periods")

        return results

    def volatility_regime_analysis(
        self,
        returns: pd.Series,
        vix: Optional[pd.Series] = None,
        lookback: int = 20
    ) -> Dict[str, RegimeResult]:
        """
        Analyze performance across volatility regimes.

        Args:
            returns: Strategy returns
            vix: VIX index (if available)
            lookback: Lookback period for realized volatility calculation

        Returns:
            Dictionary with volatility regime results
        """
        logger.debug("Analyzing volatility regime performance")

        # If VIX provided, use it; otherwise use realized volatility
        if vix is not None:
            aligned_returns = returns.reindex(vix.index).dropna()
            aligned_vix = vix.reindex(aligned_returns.index)
            volatility = aligned_vix
        else:
            aligned_returns = returns
            # Calculate rolling realized volatility
            volatility = returns.rolling(window=lookback).std() * np.sqrt(252)
            volatility = volatility.reindex(aligned_returns.index)

        # Remove NaN values
        valid_mask = ~volatility.isna()
        aligned_returns = aligned_returns[valid_mask]
        volatility = volatility[valid_mask]

        if len(volatility) == 0:
            logger.warning("No valid volatility data")
            return {}

        # Define regimes using percentiles
        low_vol_threshold = volatility.quantile(0.33)
        high_vol_threshold = volatility.quantile(0.67)

        low_vol_mask = volatility <= low_vol_threshold
        med_vol_mask = (volatility > low_vol_threshold) & (volatility <= high_vol_threshold)
        high_vol_mask = volatility > high_vol_threshold

        low_vol_returns = aligned_returns[low_vol_mask]
        med_vol_returns = aligned_returns[med_vol_mask]
        high_vol_returns = aligned_returns[high_vol_mask]

        # Calculate metrics
        low_metrics = self._calculate_performance_metrics(low_vol_returns)
        med_metrics = self._calculate_performance_metrics(med_vol_returns)
        high_metrics = self._calculate_performance_metrics(high_vol_returns)
        overall_metrics = self._calculate_performance_metrics(aligned_returns)

        # Statistical comparison
        low_comparison = self._statistical_comparison(low_vol_returns, aligned_returns)
        med_comparison = self._statistical_comparison(med_vol_returns, aligned_returns)
        high_comparison = self._statistical_comparison(high_vol_returns, aligned_returns)

        results = {
            'low_volatility': RegimeResult(
                regime_name='Low Volatility',
                performance_metrics=low_metrics,
                comparison_to_overall={
                    'volatility_threshold': low_vol_threshold,
                    'return_ratio': low_metrics['annualized_return'] / overall_metrics['annualized_return']
                    if overall_metrics['annualized_return'] != 0 else 0
                },
                statistical_significance=low_comparison
            ),
            'medium_volatility': RegimeResult(
                regime_name='Medium Volatility',
                performance_metrics=med_metrics,
                comparison_to_overall={
                    'volatility_range': (low_vol_threshold, high_vol_threshold),
                    'return_ratio': med_metrics['annualized_return'] / overall_metrics['annualized_return']
                    if overall_metrics['annualized_return'] != 0 else 0
                },
                statistical_significance=med_comparison
            ),
            'high_volatility': RegimeResult(
                regime_name='High Volatility',
                performance_metrics=high_metrics,
                comparison_to_overall={
                    'volatility_threshold': high_vol_threshold,
                    'return_ratio': high_metrics['annualized_return'] / overall_metrics['annualized_return']
                    if overall_metrics['annualized_return'] != 0 else 0
                },
                statistical_significance=high_comparison
            )
        }

        logger.info(f"Low vol: {low_metrics['n_observations']}, "
                   f"Med vol: {med_metrics['n_observations']}, "
                   f"High vol: {high_metrics['n_observations']}")

        return results

    def crisis_period_analysis(
        self,
        returns: pd.Series,
        custom_periods: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> Dict[str, RegimeResult]:
        """
        Analyze performance during crisis periods.

        Args:
            returns: Strategy returns (must have datetime index)
            custom_periods: Optional custom crisis periods to analyze

        Returns:
            Dictionary with crisis period results
        """
        logger.debug("Analyzing crisis period performance")

        # Use predefined + custom periods
        periods = self.CRISIS_PERIODS.copy()
        if custom_periods:
            periods.update(custom_periods)

        results = {}
        overall_metrics = self._calculate_performance_metrics(returns)

        for crisis_name, (start_date, end_date) in periods.items():
            try:
                # Filter returns for crisis period
                crisis_returns = returns[start_date:end_date]

                if len(crisis_returns) == 0:
                    logger.debug(f"No data for {crisis_name}")
                    continue

                # Calculate metrics
                crisis_metrics = self._calculate_performance_metrics(crisis_returns)

                # Statistical comparison
                comparison = self._statistical_comparison(crisis_returns, returns)

                results[crisis_name] = RegimeResult(
                    regime_name=f"Crisis: {crisis_name}",
                    performance_metrics=crisis_metrics,
                    comparison_to_overall={
                        'period': (start_date, end_date),
                        'return_ratio': crisis_metrics['total_return'] / overall_metrics['annualized_return']
                        if overall_metrics['annualized_return'] != 0 else 0,
                        'relative_sharpe': crisis_metrics['sharpe_ratio'] - overall_metrics['sharpe_ratio']
                    },
                    statistical_significance=comparison
                )

                logger.info(f"{crisis_name}: {crisis_metrics['n_observations']} days, "
                           f"return: {crisis_metrics['total_return']:.2%}")

            except Exception as e:
                logger.warning(f"Could not analyze {crisis_name}: {e}")

        return results

    def correlation_regime_analysis(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        lookback: int = 60
    ) -> Dict[str, RegimeResult]:
        """
        Analyze performance across correlation regimes.

        Args:
            returns: Strategy returns
            market_returns: Market returns
            lookback: Lookback period for correlation calculation

        Returns:
            Dictionary with correlation regime results
        """
        logger.debug("Analyzing correlation regime performance")

        # Align returns
        aligned_returns = returns.reindex(market_returns.index).dropna()
        aligned_market = market_returns.reindex(aligned_returns.index)

        # Calculate rolling correlation
        rolling_corr = aligned_returns.rolling(window=lookback).corr(aligned_market)
        rolling_corr = rolling_corr.dropna()

        # Align with valid correlations
        aligned_returns = aligned_returns.reindex(rolling_corr.index)

        if len(rolling_corr) == 0:
            logger.warning("No valid correlation data")
            return {}

        # Define regimes
        low_corr_threshold = rolling_corr.quantile(0.33)
        high_corr_threshold = rolling_corr.quantile(0.67)

        low_corr_mask = rolling_corr <= low_corr_threshold
        med_corr_mask = (rolling_corr > low_corr_threshold) & (rolling_corr <= high_corr_threshold)
        high_corr_mask = rolling_corr > high_corr_threshold

        low_corr_returns = aligned_returns[low_corr_mask]
        med_corr_returns = aligned_returns[med_corr_mask]
        high_corr_returns = aligned_returns[high_corr_mask]

        # Calculate metrics
        low_metrics = self._calculate_performance_metrics(low_corr_returns)
        med_metrics = self._calculate_performance_metrics(med_corr_returns)
        high_metrics = self._calculate_performance_metrics(high_corr_returns)
        overall_metrics = self._calculate_performance_metrics(aligned_returns)

        results = {
            'low_correlation': RegimeResult(
                regime_name='Low Correlation',
                performance_metrics=low_metrics,
                comparison_to_overall={
                    'correlation_threshold': low_corr_threshold,
                    'mean_correlation': rolling_corr[low_corr_mask].mean()
                },
                statistical_significance=self._statistical_comparison(low_corr_returns, aligned_returns)
            ),
            'medium_correlation': RegimeResult(
                regime_name='Medium Correlation',
                performance_metrics=med_metrics,
                comparison_to_overall={
                    'correlation_range': (low_corr_threshold, high_corr_threshold),
                    'mean_correlation': rolling_corr[med_corr_mask].mean()
                },
                statistical_significance=self._statistical_comparison(med_corr_returns, aligned_returns)
            ),
            'high_correlation': RegimeResult(
                regime_name='High Correlation',
                performance_metrics=high_metrics,
                comparison_to_overall={
                    'correlation_threshold': high_corr_threshold,
                    'mean_correlation': rolling_corr[high_corr_mask].mean()
                },
                statistical_significance=self._statistical_comparison(high_corr_returns, aligned_returns)
            )
        }

        return results

    def seasonal_analysis(self, returns: pd.Series) -> Dict[str, RegimeResult]:
        """
        Analyze seasonal performance patterns.

        Args:
            returns: Strategy returns with datetime index

        Returns:
            Dictionary with seasonal results
        """
        logger.debug("Analyzing seasonal performance")

        if not isinstance(returns.index, pd.DatetimeIndex):
            logger.warning("Returns must have datetime index for seasonal analysis")
            return {}

        results = {}
        overall_metrics = self._calculate_performance_metrics(returns)

        # Month-of-year analysis
        monthly_returns = {}
        for month in range(1, 13):
            month_mask = returns.index.month == month
            month_returns = returns[month_mask]
            if len(month_returns) > 0:
                monthly_returns[month] = month_returns

        # Calculate metrics for each month
        for month, month_returns in monthly_returns.items():
            month_name = pd.Timestamp(2000, month, 1).strftime('%B')
            metrics = self._calculate_performance_metrics(month_returns)
            comparison = self._statistical_comparison(month_returns, returns)

            results[f'month_{month:02d}'] = RegimeResult(
                regime_name=f'Month: {month_name}',
                performance_metrics=metrics,
                comparison_to_overall={
                    'return_ratio': metrics['mean_return'] / overall_metrics['mean_return']
                    if overall_metrics['mean_return'] != 0 else 0
                },
                statistical_significance=comparison
            )

        # Day-of-week analysis
        daily_returns = {}
        for day in range(5):  # Monday=0 to Friday=4
            day_mask = returns.index.dayofweek == day
            day_returns = returns[day_mask]
            if len(day_returns) > 0:
                daily_returns[day] = day_returns

        # Calculate metrics for each day
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for day, day_returns in daily_returns.items():
            metrics = self._calculate_performance_metrics(day_returns)
            comparison = self._statistical_comparison(day_returns, returns)

            results[f'day_{day}'] = RegimeResult(
                regime_name=f'Day: {day_names[day]}',
                performance_metrics=metrics,
                comparison_to_overall={
                    'return_ratio': metrics['mean_return'] / overall_metrics['mean_return']
                    if overall_metrics['mean_return'] != 0 else 0
                },
                statistical_significance=comparison
            )

        logger.info(f"Analyzed {len(monthly_returns)} months and {len(daily_returns)} days of week")

        return results

    def run_comprehensive_regime_analysis(
        self,
        returns: pd.Series,
        market_returns: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
        custom_crisis_periods: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> Dict[str, Dict[str, RegimeResult]]:
        """
        Run comprehensive regime analysis.

        Args:
            returns: Strategy returns
            market_returns: Optional market/benchmark returns
            vix: Optional VIX data
            custom_crisis_periods: Optional custom crisis periods

        Returns:
            Dictionary of all regime analysis results
        """
        logger.info("Running comprehensive regime analysis")

        results = {}

        # Bull/Bear analysis
        if market_returns is not None:
            try:
                results['bull_bear'] = self.bull_bear_analysis(returns, market_returns)
                logger.info("Completed bull/bear analysis")
            except Exception as e:
                logger.error(f"Bull/bear analysis failed: {e}")

        # Volatility regime analysis
        try:
            results['volatility_regimes'] = self.volatility_regime_analysis(returns, vix)
            logger.info("Completed volatility regime analysis")
        except Exception as e:
            logger.error(f"Volatility regime analysis failed: {e}")

        # Crisis period analysis
        try:
            results['crisis_periods'] = self.crisis_period_analysis(returns, custom_crisis_periods)
            logger.info("Completed crisis period analysis")
        except Exception as e:
            logger.error(f"Crisis period analysis failed: {e}")

        # Correlation regime analysis
        if market_returns is not None:
            try:
                results['correlation_regimes'] = self.correlation_regime_analysis(returns, market_returns)
                logger.info("Completed correlation regime analysis")
            except Exception as e:
                logger.error(f"Correlation regime analysis failed: {e}")

        # Seasonal analysis
        try:
            results['seasonal'] = self.seasonal_analysis(returns)
            logger.info("Completed seasonal analysis")
        except Exception as e:
            logger.error(f"Seasonal analysis failed: {e}")

        total_regimes = sum(len(v) for v in results.values())
        logger.info(f"Completed comprehensive regime analysis: {total_regimes} regimes analyzed")

        return results

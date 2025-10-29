"""
Advanced risk metrics for portfolio analysis.

This module provides sophisticated risk measurement techniques including:
- Value at Risk (VaR): Historical, Parametric, Monte Carlo, Cornish-Fisher
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Conditional Drawdown at Risk (CDaR)
- Tail risk metrics
- Various risk decomposition methods
"""

from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings

from utils.logger import get_backtest_logger

log = get_backtest_logger()


class RiskMetrics:
    """
    Advanced risk metrics calculator.

    Provides comprehensive risk analysis including VaR, CVaR, tail risk,
    and sophisticated risk decomposition methods.
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        confidence_level: float = 0.95,
        periods_per_year: int = 252
    ):
        """
        Initialize risk metrics calculator.

        Args:
            returns: Series of portfolio returns
            benchmark_returns: Optional benchmark returns
            confidence_level: Confidence level for VaR/CVaR (default: 95%)
            periods_per_year: Trading periods per year
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.periods_per_year = periods_per_year

        if len(self.returns) == 0:
            log.warning("Empty returns series provided to RiskMetrics")

    def calculate_all_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate all risk metrics.

        Returns:
            Dictionary of all calculated risk metrics
        """
        log.info("Calculating comprehensive risk metrics")

        metrics = {}

        try:
            # VaR calculations
            metrics['var_historical'] = self.var_historical()
            metrics['var_parametric'] = self.var_parametric()
            metrics['var_cornish_fisher'] = self.var_cornish_fisher()

            # CVaR/ES
            metrics['cvar_historical'] = self.cvar_historical()
            metrics['cvar_parametric'] = self.cvar_parametric()

            # Drawdown risk
            metrics['cdar'] = self.conditional_drawdown_at_risk()
            metrics['max_drawdown'] = self.max_drawdown()

            # Tail risk
            metrics.update(self.calculate_tail_metrics())

            # Pain metrics
            metrics['pain_index'] = self.pain_index()
            metrics['ulcer_index'] = self.ulcer_index()

            # Beta decomposition
            if self.benchmark_returns is not None:
                metrics.update(self.calculate_beta_metrics())

            log.success(f"Calculated {len(metrics)} risk metrics")

        except Exception as e:
            log.error(f"Error calculating risk metrics: {str(e)}")
            raise

        return metrics

    # ===========================
    # Value at Risk (VaR)
    # ===========================

    def var_historical(self, alpha: Optional[float] = None) -> float:
        """
        Calculate historical Value at Risk.

        Args:
            alpha: Significance level (None uses default)

        Returns:
            VaR value (positive number)
        """
        if len(self.returns) == 0:
            return 0.0

        alpha = alpha if alpha is not None else self.alpha
        return abs(self.returns.quantile(alpha))

    def var_parametric(self, alpha: Optional[float] = None) -> float:
        """
        Calculate parametric VaR assuming normal distribution.

        Args:
            alpha: Significance level

        Returns:
            VaR value (positive number)
        """
        if len(self.returns) < 2:
            return 0.0

        alpha = alpha if alpha is not None else self.alpha

        mean = self.returns.mean()
        std = self.returns.std()

        z_score = stats.norm.ppf(alpha)
        var = -(mean + z_score * std)

        return max(var, 0.0)

    def var_cornish_fisher(self, alpha: Optional[float] = None) -> float:
        """
        Calculate VaR using Cornish-Fisher expansion.

        Adjusts for skewness and kurtosis in the return distribution.

        Args:
            alpha: Significance level

        Returns:
            Modified VaR value
        """
        if len(self.returns) < 4:
            return self.var_parametric(alpha)

        alpha = alpha if alpha is not None else self.alpha

        mean = self.returns.mean()
        std = self.returns.std()
        skew = stats.skew(self.returns, bias=False)
        kurt = stats.kurtosis(self.returns, bias=False, fisher=True)

        # Calculate z-score with Cornish-Fisher modification
        z = stats.norm.ppf(alpha)
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        var = -(mean + z_cf * std)

        return max(var, 0.0)

    def var_monte_carlo(
        self,
        alpha: Optional[float] = None,
        n_simulations: int = 10000,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate VaR using Monte Carlo simulation.

        Args:
            alpha: Significance level
            n_simulations: Number of simulations to run
            time_horizon: Time horizon in periods

        Returns:
            VaR from Monte Carlo
        """
        if len(self.returns) < 2:
            return 0.0

        alpha = alpha if alpha is not None else self.alpha

        mean = self.returns.mean()
        std = self.returns.std()

        # Generate random returns
        np.random.seed(42)
        simulated_returns = np.random.normal(
            mean * time_horizon,
            std * np.sqrt(time_horizon),
            n_simulations
        )

        var = abs(np.percentile(simulated_returns, alpha * 100))

        return var

    # ===========================
    # Conditional VaR (CVaR/ES)
    # ===========================

    def cvar_historical(self, alpha: Optional[float] = None) -> float:
        """
        Calculate historical Conditional VaR (Expected Shortfall).

        Args:
            alpha: Significance level

        Returns:
            CVaR value (positive number)
        """
        if len(self.returns) == 0:
            return 0.0

        alpha = alpha if alpha is not None else self.alpha

        var = -self.returns.quantile(alpha)
        cvar = abs(self.returns[self.returns <= -var].mean())

        return cvar if not np.isnan(cvar) else var

    def cvar_parametric(self, alpha: Optional[float] = None) -> float:
        """
        Calculate parametric CVaR assuming normal distribution.

        Args:
            alpha: Significance level

        Returns:
            CVaR value
        """
        if len(self.returns) < 2:
            return 0.0

        alpha = alpha if alpha is not None else self.alpha

        mean = self.returns.mean()
        std = self.returns.std()

        z_alpha = stats.norm.ppf(alpha)

        # CVaR formula for normal distribution
        cvar = -(mean - std * stats.norm.pdf(z_alpha) / alpha)

        return max(cvar, 0.0)

    def expected_shortfall(self, alpha: Optional[float] = None) -> float:
        """
        Calculate Expected Shortfall (alias for CVaR).

        Args:
            alpha: Significance level

        Returns:
            Expected Shortfall value
        """
        return self.cvar_historical(alpha)

    # ===========================
    # Drawdown Risk
    # ===========================

    def conditional_drawdown_at_risk(
        self,
        alpha: Optional[float] = None
    ) -> float:
        """
        Calculate Conditional Drawdown at Risk (CDaR).

        CDaR is the expected drawdown in the worst (1-alpha)% of cases.

        Args:
            alpha: Confidence level

        Returns:
            CDaR value (positive number)
        """
        if len(self.returns) == 0:
            return 0.0

        alpha = alpha if alpha is not None else self.alpha

        # Calculate drawdown series
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max

        # Get worst drawdowns
        threshold = drawdowns.quantile(alpha)
        worst_drawdowns = drawdowns[drawdowns <= threshold]

        if len(worst_drawdowns) == 0:
            return abs(drawdowns.min())

        return abs(worst_drawdowns.mean())

    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.

        Returns:
            Maximum drawdown (positive number)
        """
        if len(self.returns) == 0:
            return 0.0

        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        return abs(drawdown.min())

    def average_drawdown(self) -> float:
        """
        Calculate average drawdown.

        Returns:
            Average drawdown across all drawdown periods
        """
        if len(self.returns) == 0:
            return 0.0

        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max

        # Only consider negative drawdowns
        negative_dd = drawdowns[drawdowns < 0]

        if len(negative_dd) == 0:
            return 0.0

        return abs(negative_dd.mean())

    # ===========================
    # Tail Risk Metrics
    # ===========================

    def calculate_tail_metrics(self) -> Dict[str, float]:
        """
        Calculate various tail risk metrics.

        Returns:
            Dictionary of tail risk metrics
        """
        if len(self.returns) < 20:
            return {
                'tail_ratio': 0.0,
                'left_tail_var': 0.0,
                'right_tail_var': 0.0,
                'tail_index': 0.0,
            }

        return {
            'tail_ratio': self.tail_ratio(),
            'left_tail_var': self.left_tail_variance(),
            'right_tail_var': self.right_tail_variance(),
            'tail_index': self.tail_index(),
        }

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

        right_tail = abs(self.returns.quantile(1 - percentile))
        left_tail = abs(self.returns.quantile(percentile))

        if left_tail == 0:
            return 0.0

        return right_tail / left_tail

    def left_tail_variance(self, percentile: float = 0.05) -> float:
        """
        Calculate variance of left tail (losses).

        Args:
            percentile: Percentile for tail definition

        Returns:
            Left tail variance
        """
        if len(self.returns) < 20:
            return 0.0

        threshold = self.returns.quantile(percentile)
        left_tail = self.returns[self.returns <= threshold]

        if len(left_tail) == 0:
            return 0.0

        return left_tail.var()

    def right_tail_variance(self, percentile: float = 0.05) -> float:
        """
        Calculate variance of right tail (gains).

        Args:
            percentile: Percentile for tail definition

        Returns:
            Right tail variance
        """
        if len(self.returns) < 20:
            return 0.0

        threshold = self.returns.quantile(1 - percentile)
        right_tail = self.returns[self.returns >= threshold]

        if len(right_tail) == 0:
            return 0.0

        return right_tail.var()

    def tail_index(self) -> float:
        """
        Estimate tail index using Hill estimator.

        Lower values indicate heavier tails.

        Returns:
            Tail index estimate
        """
        if len(self.returns) < 50:
            return 0.0

        # Use negative returns (losses) for tail index
        losses = -self.returns[self.returns < 0].sort_values()

        if len(losses) < 10:
            return 0.0

        # Hill estimator
        k = int(len(losses) * 0.1)  # Use top 10% of losses
        tail_losses = losses.iloc[-k:]

        if len(tail_losses) < 2:
            return 0.0

        log_ratios = np.log(tail_losses / tail_losses.iloc[0])
        tail_idx = log_ratios.sum() / k if k > 0 else 0.0

        return tail_idx

    # ===========================
    # Pain Metrics
    # ===========================

    def pain_index(self) -> float:
        """
        Calculate Pain Index (average drawdown depth).

        Returns:
            Pain index value
        """
        if len(self.returns) == 0:
            return 0.0

        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        # Pain is average of all drawdowns
        pain = abs(drawdown.mean())

        return pain

    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index (RMS of drawdowns).

        Returns:
            Ulcer index value
        """
        if len(self.returns) == 0:
            return 0.0

        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        # Ulcer is square root of mean squared drawdown
        ulcer = np.sqrt((drawdown ** 2).mean()) * 100

        return ulcer

    def pain_ratio(self) -> float:
        """
        Calculate Pain Ratio (return / pain index).

        Returns:
            Pain ratio
        """
        pain = self.pain_index()

        if pain == 0:
            return 0.0

        total_return = (1 + self.returns).prod() - 1
        return total_return / pain

    # ===========================
    # Beta and Correlation Risk
    # ===========================

    def calculate_beta_metrics(self) -> Dict[str, float]:
        """
        Calculate beta and correlation-based risk metrics.

        Returns:
            Dictionary of beta-related metrics
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return {}

        return {
            'beta': self.beta(),
            'beta_bull': self.beta_bull(),
            'beta_bear': self.beta_bear(),
            'correlation': self.correlation(),
            'tracking_error': self.tracking_error(),
            'systematic_risk': self.systematic_risk(),
            'idiosyncratic_risk': self.idiosyncratic_risk(),
        }

    def beta(self) -> float:
        """
        Calculate portfolio beta relative to benchmark.

        Returns:
            Beta coefficient
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = self.benchmark_returns.var()

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def beta_bull(self) -> float:
        """
        Calculate beta during bull markets (benchmark up).

        Returns:
            Bull market beta
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        bull_periods = self.benchmark_returns > 0

        if bull_periods.sum() < 2:
            return 0.0

        bull_returns = self.returns[bull_periods]
        bull_benchmark = self.benchmark_returns[bull_periods]

        covariance = np.cov(bull_returns, bull_benchmark)[0, 1]
        benchmark_variance = bull_benchmark.var()

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def beta_bear(self) -> float:
        """
        Calculate beta during bear markets (benchmark down).

        Returns:
            Bear market beta
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        bear_periods = self.benchmark_returns < 0

        if bear_periods.sum() < 2:
            return 0.0

        bear_returns = self.returns[bear_periods]
        bear_benchmark = self.benchmark_returns[bear_periods]

        covariance = np.cov(bear_returns, bear_benchmark)[0, 1]
        benchmark_variance = bear_benchmark.var()

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def correlation(self) -> float:
        """
        Calculate correlation with benchmark.

        Returns:
            Correlation coefficient
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        return self.returns.corr(self.benchmark_returns)

    def tracking_error(self) -> float:
        """
        Calculate annualized tracking error.

        Returns:
            Tracking error
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        active_returns = self.returns - self.benchmark_returns
        return active_returns.std() * np.sqrt(self.periods_per_year)

    def systematic_risk(self) -> float:
        """
        Calculate systematic risk (beta * benchmark volatility).

        Returns:
            Systematic risk component
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0

        beta = self.beta()
        benchmark_vol = self.benchmark_returns.std() * np.sqrt(self.periods_per_year)

        return abs(beta * benchmark_vol)

    def idiosyncratic_risk(self) -> float:
        """
        Calculate idiosyncratic (specific) risk.

        Returns:
            Idiosyncratic risk component
        """
        if self.benchmark_returns is None or len(self.returns) < 2:
            return self.returns.std() * np.sqrt(self.periods_per_year)

        total_variance = self.returns.var()
        beta = self.beta()
        benchmark_variance = self.benchmark_returns.var()

        systematic_variance = (beta ** 2) * benchmark_variance
        idiosyncratic_variance = max(total_variance - systematic_variance, 0)

        return np.sqrt(idiosyncratic_variance) * np.sqrt(self.periods_per_year)

    # ===========================
    # Risk Decomposition
    # ===========================

    def risk_decomposition(self) -> Dict[str, float]:
        """
        Decompose total risk into components.

        Returns:
            Dictionary with risk decomposition
        """
        total_vol = self.returns.std() * np.sqrt(self.periods_per_year)

        decomp = {
            'total_volatility': total_vol,
            'downside_volatility': self.downside_volatility(),
            'upside_volatility': self.upside_volatility(),
        }

        if self.benchmark_returns is not None:
            decomp['systematic_risk'] = self.systematic_risk()
            decomp['idiosyncratic_risk'] = self.idiosyncratic_risk()
            decomp['tracking_error'] = self.tracking_error()

        return decomp

    def downside_volatility(self, mar: float = 0.0) -> float:
        """
        Calculate downside volatility.

        Args:
            mar: Minimum acceptable return

        Returns:
            Downside volatility
        """
        if len(self.returns) == 0:
            return 0.0

        downside_returns = self.returns[self.returns < mar]

        if len(downside_returns) == 0:
            return 0.0

        return downside_returns.std() * np.sqrt(self.periods_per_year)

    def upside_volatility(self, mar: float = 0.0) -> float:
        """
        Calculate upside volatility.

        Args:
            mar: Minimum acceptable return

        Returns:
            Upside volatility
        """
        if len(self.returns) == 0:
            return 0.0

        upside_returns = self.returns[self.returns > mar]

        if len(upside_returns) == 0:
            return 0.0

        return upside_returns.std() * np.sqrt(self.periods_per_year)

    # ===========================
    # Rolling Risk Metrics
    # ===========================

    def rolling_var(
        self,
        window: int = 252,
        method: str = 'historical'
    ) -> pd.Series:
        """
        Calculate rolling Value at Risk.

        Args:
            window: Rolling window size
            method: 'historical', 'parametric', or 'cornish_fisher'

        Returns:
            Series of rolling VaR values
        """
        if len(self.returns) < window:
            log.warning(f"Insufficient data for rolling VaR (need {window}, have {len(self.returns)})")
            return pd.Series()

        def calculate_var(returns_window):
            calc = RiskMetrics(returns_window, confidence_level=self.confidence_level)
            if method == 'historical':
                return calc.var_historical()
            elif method == 'parametric':
                return calc.var_parametric()
            elif method == 'cornish_fisher':
                return calc.var_cornish_fisher()
            else:
                return calc.var_historical()

        rolling_var = self.returns.rolling(window).apply(
            lambda x: calculate_var(pd.Series(x)),
            raw=False
        )

        return rolling_var

    def rolling_cvar(self, window: int = 252) -> pd.Series:
        """
        Calculate rolling Conditional VaR.

        Args:
            window: Rolling window size

        Returns:
            Series of rolling CVaR values
        """
        if len(self.returns) < window:
            return pd.Series()

        def calculate_cvar(returns_window):
            calc = RiskMetrics(returns_window, confidence_level=self.confidence_level)
            return calc.cvar_historical()

        rolling_cvar = self.returns.rolling(window).apply(
            lambda x: calculate_cvar(pd.Series(x)),
            raw=False
        )

        return rolling_cvar

    def rolling_beta(self, window: int = 252) -> pd.Series:
        """
        Calculate rolling beta.

        Args:
            window: Rolling window size

        Returns:
            Series of rolling beta values
        """
        if self.benchmark_returns is None or len(self.returns) < window:
            return pd.Series()

        def calculate_beta(idx):
            returns_window = self.returns.iloc[idx:idx+window]
            benchmark_window = self.benchmark_returns.iloc[idx:idx+window]

            if len(returns_window) < 2:
                return np.nan

            covariance = np.cov(returns_window, benchmark_window)[0, 1]
            benchmark_variance = benchmark_window.var()

            if benchmark_variance == 0:
                return np.nan

            return covariance / benchmark_variance

        rolling_beta = pd.Series(
            [calculate_beta(i) for i in range(len(self.returns) - window + 1)],
            index=self.returns.index[window-1:]
        )

        return rolling_beta


def calculate_portfolio_var(
    positions: Dict[str, float],
    returns_data: pd.DataFrame,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate portfolio-level Value at Risk.

    Args:
        positions: Dictionary of {asset: weight}
        returns_data: DataFrame with returns for each asset
        confidence_level: Confidence level for VaR
        method: VaR calculation method

    Returns:
        Portfolio VaR
    """
    # Calculate portfolio returns
    weights = pd.Series(positions)
    portfolio_returns = (returns_data * weights).sum(axis=1)

    # Calculate VaR
    risk_calc = RiskMetrics(portfolio_returns, confidence_level=confidence_level)

    if method == 'historical':
        return risk_calc.var_historical()
    elif method == 'parametric':
        return risk_calc.var_parametric()
    elif method == 'cornish_fisher':
        return risk_calc.var_cornish_fisher()
    else:
        return risk_calc.var_historical()


def calculate_marginal_var(
    positions: Dict[str, float],
    returns_data: pd.DataFrame,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate marginal VaR contribution for each position.

    Args:
        positions: Dictionary of {asset: weight}
        returns_data: DataFrame with returns for each asset
        confidence_level: Confidence level for VaR

    Returns:
        Dictionary of marginal VaR for each asset
    """
    marginal_var = {}

    # Calculate base portfolio VaR
    base_var = calculate_portfolio_var(positions, returns_data, confidence_level)

    # Calculate marginal contribution for each asset
    delta = 0.01  # 1% change

    for asset in positions.keys():
        # Increase position slightly
        perturbed_positions = positions.copy()
        perturbed_positions[asset] = positions[asset] * (1 + delta)

        # Renormalize weights
        total_weight = sum(perturbed_positions.values())
        perturbed_positions = {k: v/total_weight for k, v in perturbed_positions.items()}

        # Calculate new VaR
        new_var = calculate_portfolio_var(perturbed_positions, returns_data, confidence_level)

        # Marginal VaR
        marginal_var[asset] = (new_var - base_var) / delta

    return marginal_var

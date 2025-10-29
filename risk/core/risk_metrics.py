"""
Core Risk Metrics Calculation

Implements fundamental risk calculations including:
- Portfolio volatility
- Beta and correlation
- Sharpe/Sortino ratios
- Maximum drawdown
- Value at Risk (VaR)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskMetricsResult:
    """Container for calculated risk metrics"""
    timestamp: datetime
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: Optional[float] = None
    correlation_to_market: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'portfolio_volatility': self.portfolio_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'beta': self.beta,
            'correlation_to_market': self.correlation_to_market
        }


class RiskMetrics:
    """Calculate core risk metrics for portfolio and positions"""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252
    ):
        """
        Initialize risk metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            annualization_factor: Trading days per year (default 252)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Calculate annualized volatility

        Args:
            returns: Return series
            annualize: Whether to annualize the result

        Returns:
            Volatility (annualized if requested)
        """
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(self.annualization_factor)
        return vol

    def calculate_ewma_volatility(
        self,
        returns: pd.Series,
        halflife: int = 25,
        annualize: bool = True
    ) -> float:
        """
        Calculate EWMA volatility (Carver's approach)

        Args:
            returns: Return series
            halflife: Halflife for EWMA (default 25 days)
            annualize: Whether to annualize the result

        Returns:
            EWMA volatility
        """
        ewma_var = returns.ewm(halflife=halflife).var().iloc[-1]
        vol = np.sqrt(ewma_var)
        if annualize:
            vol *= np.sqrt(self.annualization_factor)
        return vol

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate annualized Sharpe ratio

        Args:
            returns: Return series
            risk_free_rate: Override default risk-free rate

        Returns:
            Sharpe ratio
        """
        rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        excess_returns = returns - rfr / self.annualization_factor

        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std()
        sharpe *= np.sqrt(self.annualization_factor)
        return sharpe

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate annualized Sortino ratio (downside deviation)

        Args:
            returns: Return series
            risk_free_rate: Override default risk-free rate

        Returns:
            Sortino ratio
        """
        rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        excess_returns = returns - rfr / self.annualization_factor

        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std()
        sortino *= np.sqrt(self.annualization_factor)
        return sortino

    def calculate_drawdown(
        self,
        equity_curve: pd.Series
    ) -> Tuple[pd.Series, float, float]:
        """
        Calculate drawdown series and metrics

        Args:
            equity_curve: Equity curve series

        Returns:
            Tuple of (drawdown_series, max_drawdown, current_drawdown)
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Maximum drawdown
        max_dd = drawdown.min()

        # Current drawdown
        current_dd = drawdown.iloc[-1]

        return drawdown, max_dd, current_dd

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk

        Args:
            returns: Return series
            confidence_level: Confidence level (0.95 or 0.99)
            method: 'historical' or 'parametric'

        Returns:
            VaR (positive number representing potential loss)
        """
        if method == 'historical':
            var = -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            # Assume normal distribution
            from scipy import stats
            mean = returns.mean()
            std = returns.std()
            var = -(mean + stats.norm.ppf(1 - confidence_level) * std)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return var

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)

        Args:
            returns: Return series
            confidence_level: Confidence level (0.95 or 0.99)

        Returns:
            CVaR (positive number representing expected loss in tail)
        """
        var = self.calculate_var(returns, confidence_level, method='historical')

        # Calculate expected loss beyond VaR
        threshold = -var
        tail_losses = returns[returns <= threshold]

        if len(tail_losses) == 0:
            return var

        cvar = -tail_losses.mean()
        return cvar

    def calculate_beta(
        self,
        returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate beta to market

        Args:
            returns: Portfolio returns
            market_returns: Market returns

        Returns:
            Beta coefficient
        """
        # Align series
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'market': market_returns
        }).dropna()

        if len(aligned_data) < 2:
            return 1.0

        covariance = aligned_data['portfolio'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()

        if market_variance == 0:
            return 1.0

        beta = covariance / market_variance
        return beta

    def calculate_correlation(
        self,
        returns1: pd.Series,
        returns2: pd.Series
    ) -> float:
        """
        Calculate correlation between two return series

        Args:
            returns1: First return series
            returns2: Second return series

        Returns:
            Correlation coefficient
        """
        # Align series
        aligned_data = pd.DataFrame({
            'r1': returns1,
            'r2': returns2
        }).dropna()

        if len(aligned_data) < 2:
            return 0.0

        correlation = aligned_data['r1'].corr(aligned_data['r2'])
        return correlation

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        market_returns: Optional[pd.Series] = None
    ) -> RiskMetricsResult:
        """
        Calculate all risk metrics

        Args:
            returns: Return series
            equity_curve: Equity curve
            market_returns: Optional market returns for beta calculation

        Returns:
            RiskMetricsResult with all calculated metrics
        """
        # Volatility
        vol = self.calculate_volatility(returns)

        # Sharpe and Sortino
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)

        # Drawdown
        _, max_dd, current_dd = self.calculate_drawdown(equity_curve)

        # VaR at different confidence levels
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)

        # CVaR at different confidence levels
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)

        # Beta and correlation (if market returns provided)
        beta = None
        correlation = None
        if market_returns is not None:
            beta = self.calculate_beta(returns, market_returns)
            correlation = self.calculate_correlation(returns, market_returns)

        return RiskMetricsResult(
            timestamp=datetime.now(),
            portfolio_volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            correlation_to_market=correlation
        )

    def calculate_portfolio_volatility(
        self,
        position_returns: pd.DataFrame,
        weights: np.ndarray
    ) -> float:
        """
        Calculate portfolio volatility from position returns and weights

        Args:
            position_returns: DataFrame of position returns
            weights: Array of position weights

        Returns:
            Annualized portfolio volatility
        """
        # Calculate covariance matrix
        cov_matrix = position_returns.cov()

        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

        # Portfolio volatility (annualized)
        portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(self.annualization_factor)

        return portfolio_vol

    def calculate_position_var_contribution(
        self,
        position_returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate each position's contribution to portfolio VaR

        Args:
            position_returns: DataFrame of position returns
            weights: Array of position weights
            confidence_level: VaR confidence level

        Returns:
            Dictionary mapping position to VaR contribution
        """
        # Calculate portfolio returns
        portfolio_returns = (position_returns * weights).sum(axis=1)

        # Calculate portfolio VaR
        portfolio_var = self.calculate_var(portfolio_returns, confidence_level)

        # Calculate marginal VaR for each position
        var_contributions = {}

        for i, col in enumerate(position_returns.columns):
            # Shift weight slightly
            epsilon = 0.001
            weights_up = weights.copy()
            weights_up[i] += epsilon

            # Normalize weights
            weights_up = weights_up / weights_up.sum()

            # Calculate new portfolio VaR
            portfolio_returns_up = (position_returns * weights_up).sum(axis=1)
            var_up = self.calculate_var(portfolio_returns_up, confidence_level)

            # Marginal VaR
            marginal_var = (var_up - portfolio_var) / epsilon

            # Component VaR
            var_contributions[col] = marginal_var * weights[i]

        return var_contributions

"""
Value at Risk (VaR) Models

Implements multiple VaR methodologies:
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Conditional VaR (CVaR/Expected Shortfall)
- Component VaR
- Marginal VaR
- Stressed VaR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize


@dataclass
class VaRResult:
    """Container for VaR calculation results"""
    timestamp: datetime
    method: str
    confidence_level: float
    horizon_days: int
    var: float
    cvar: float
    component_var: Optional[Dict[str, float]] = None
    marginal_var: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'method': self.method,
            'confidence_level': self.confidence_level,
            'horizon_days': self.horizon_days,
            'var': self.var,
            'cvar': self.cvar,
            'component_var': self.component_var,
            'marginal_var': self.marginal_var
        }


class VaRModels:
    """Comprehensive VaR calculation models"""

    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99],
        horizons: List[int] = [1, 10]
    ):
        """
        Initialize VaR models

        Args:
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            horizons: List of time horizons in days (e.g., [1, 10])
        """
        self.confidence_levels = confidence_levels
        self.horizons = horizons
        self.var_history = []

    def historical_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate Historical VaR (percentile method)

        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.95)
            horizon: Time horizon in days

        Returns:
            VaR (positive number representing potential loss)
        """
        # Scale returns to horizon
        if horizon > 1:
            returns_scaled = returns * np.sqrt(horizon)
        else:
            returns_scaled = returns

        # Calculate percentile
        var = -np.percentile(returns_scaled, (1 - confidence_level) * 100)

        return var

    def parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate Parametric VaR (variance-covariance method)

        Assumes returns are normally distributed

        Args:
            returns: Return series
            confidence_level: Confidence level
            horizon: Time horizon in days

        Returns:
            VaR
        """
        mean = returns.mean()
        std = returns.std()

        # Scale to horizon
        mean_scaled = mean * horizon
        std_scaled = std * np.sqrt(horizon)

        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mean_scaled + z_score * std_scaled)

        return var

    def monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 10000
    ) -> float:
        """
        Calculate Monte Carlo VaR

        Args:
            returns: Return series
            confidence_level: Confidence level
            horizon: Time horizon in days
            n_simulations: Number of Monte Carlo simulations

        Returns:
            VaR
        """
        mean = returns.mean()
        std = returns.std()

        # Generate simulated returns
        simulated_returns = np.random.normal(
            mean * horizon,
            std * np.sqrt(horizon),
            n_simulations
        )

        # Calculate VaR from simulations
        var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)

        return var

    def conditional_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        horizon: int = 1,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall)

        Args:
            returns: Return series
            confidence_level: Confidence level
            horizon: Time horizon in days
            method: 'historical' or 'parametric'

        Returns:
            CVaR
        """
        # Scale returns to horizon
        if horizon > 1:
            returns_scaled = returns * np.sqrt(horizon)
        else:
            returns_scaled = returns

        if method == 'historical':
            # Calculate VaR threshold
            var = self.historical_var(returns, confidence_level, horizon)
            threshold = -var

            # Average of losses beyond VaR
            tail_losses = returns_scaled[returns_scaled <= threshold]

            if len(tail_losses) == 0:
                return var

            cvar = -tail_losses.mean()

        elif method == 'parametric':
            # Parametric CVaR for normal distribution
            mean = returns.mean() * horizon
            std = returns.std() * np.sqrt(horizon)

            z_score = stats.norm.ppf(1 - confidence_level)
            pdf_at_z = stats.norm.pdf(z_score)

            cvar = -(mean - std * pdf_at_z / (1 - confidence_level))

        else:
            raise ValueError(f"Unknown CVaR method: {method}")

        return cvar

    def component_var(
        self,
        position_returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate Component VaR (position contribution to portfolio VaR)

        Args:
            position_returns: DataFrame of position returns
            weights: Position weights
            confidence_level: Confidence level
            horizon: Time horizon

        Returns:
            Dictionary mapping position to VaR contribution
        """
        # Calculate portfolio returns
        portfolio_returns = (position_returns * weights).sum(axis=1)

        # Calculate portfolio VaR
        portfolio_var = self.historical_var(portfolio_returns, confidence_level, horizon)

        # Calculate marginal VaR for each position
        component_vars = {}

        for i, col in enumerate(position_returns.columns):
            # Calculate position's marginal contribution
            position_cov_with_portfolio = position_returns[col].cov(portfolio_returns)

            # Component VaR = weight * marginal VaR
            # For historical VaR, use approximation
            component_vars[col] = weights[i] * portfolio_var * (
                position_cov_with_portfolio / portfolio_returns.var()
            )

        return component_vars

    def marginal_var(
        self,
        position_returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate Marginal VaR (change in portfolio VaR from small weight change)

        Args:
            position_returns: DataFrame of position returns
            weights: Position weights
            confidence_level: Confidence level
            horizon: Time horizon

        Returns:
            Dictionary mapping position to marginal VaR
        """
        epsilon = 0.001
        marginal_vars = {}

        # Calculate base portfolio VaR
        portfolio_returns = (position_returns * weights).sum(axis=1)
        base_var = self.historical_var(portfolio_returns, confidence_level, horizon)

        for i, col in enumerate(position_returns.columns):
            # Increase weight by epsilon
            weights_up = weights.copy()
            weights_up[i] += epsilon

            # Renormalize
            weights_up = weights_up / weights_up.sum()

            # Calculate new portfolio VaR
            portfolio_returns_up = (position_returns * weights_up).sum(axis=1)
            var_up = self.historical_var(portfolio_returns_up, confidence_level, horizon)

            # Marginal VaR
            marginal_vars[col] = (var_up - base_var) / epsilon

        return marginal_vars

    def stressed_var(
        self,
        returns: pd.Series,
        stress_period_start: str,
        stress_period_end: str,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate Stressed VaR using historical stress period

        Args:
            returns: Full return series with datetime index
            stress_period_start: Start date of stress period
            stress_period_end: End date of stress period
            confidence_level: Confidence level
            horizon: Time horizon

        Returns:
            Stressed VaR
        """
        # Extract stress period returns
        stress_returns = returns.loc[stress_period_start:stress_period_end]

        if len(stress_returns) == 0:
            # Fallback to regular VaR if stress period not found
            return self.historical_var(returns, confidence_level, horizon)

        # Calculate VaR on stress period
        stressed_var = self.historical_var(stress_returns, confidence_level, horizon)

        return stressed_var

    def calculate_all_var_metrics(
        self,
        returns: pd.Series,
        position_returns: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None
    ) -> List[VaRResult]:
        """
        Calculate VaR using all methods and parameters

        Args:
            returns: Portfolio return series
            position_returns: Individual position returns (for component VaR)
            weights: Position weights (for component VaR)

        Returns:
            List of VaRResult objects
        """
        results = []

        for confidence_level in self.confidence_levels:
            for horizon in self.horizons:
                # Historical VaR
                hist_var = self.historical_var(returns, confidence_level, horizon)
                hist_cvar = self.conditional_var(returns, confidence_level, horizon, 'historical')

                component_var = None
                marginal_var = None

                if position_returns is not None and weights is not None:
                    component_var = self.component_var(
                        position_returns, weights, confidence_level, horizon
                    )
                    marginal_var = self.marginal_var(
                        position_returns, weights, confidence_level, horizon
                    )

                results.append(VaRResult(
                    timestamp=datetime.now(),
                    method='historical',
                    confidence_level=confidence_level,
                    horizon_days=horizon,
                    var=hist_var,
                    cvar=hist_cvar,
                    component_var=component_var,
                    marginal_var=marginal_var
                ))

                # Parametric VaR
                param_var = self.parametric_var(returns, confidence_level, horizon)
                param_cvar = self.conditional_var(returns, confidence_level, horizon, 'parametric')

                results.append(VaRResult(
                    timestamp=datetime.now(),
                    method='parametric',
                    confidence_level=confidence_level,
                    horizon_days=horizon,
                    var=param_var,
                    cvar=param_cvar
                ))

                # Monte Carlo VaR
                mc_var = self.monte_carlo_var(returns, confidence_level, horizon)

                results.append(VaRResult(
                    timestamp=datetime.now(),
                    method='monte_carlo',
                    confidence_level=confidence_level,
                    horizon_days=horizon,
                    var=mc_var,
                    cvar=mc_var  # Approximate CVaR from MC
                ))

        # Store results
        self.var_history.extend(results)

        return results

    def backtest_var(
        self,
        returns: pd.Series,
        var_estimates: pd.Series,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Backtest VaR model (Kupiec test)

        Args:
            returns: Actual returns
            var_estimates: VaR estimates (positive values)
            confidence_level: Confidence level used for VaR

        Returns:
            Dictionary with backtest results
        """
        # Count VaR breaches
        breaches = (returns < -var_estimates).sum()
        total_days = len(returns)

        # Expected number of breaches
        expected_breaches = total_days * (1 - confidence_level)

        # Actual breach rate
        breach_rate = breaches / total_days

        # Kupiec test statistic
        if breaches == 0:
            lr_stat = -2 * np.log((1 - confidence_level) ** total_days)
        elif breaches == total_days:
            lr_stat = -2 * np.log(confidence_level ** total_days)
        else:
            lr_stat = -2 * (
                total_days * np.log(1 - confidence_level) +
                breaches * np.log((1 - confidence_level) / breach_rate) +
                (total_days - breaches) * np.log(confidence_level / (1 - breach_rate))
            )

        # p-value (chi-squared with 1 degree of freedom)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        return {
            'breaches': breaches,
            'total_days': total_days,
            'expected_breaches': expected_breaches,
            'breach_rate': breach_rate,
            'expected_breach_rate': 1 - confidence_level,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'test_passed': p_value > 0.05  # 5% significance level
        }

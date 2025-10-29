"""
Risk Budgeting and Allocation

Implements risk-based position sizing:
- Equal Risk Contribution (ERC)
- Risk parity allocation
- Marginal risk contribution
- Strategy risk limits
- Concentration penalties
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class RiskBudgetResult:
    """Result of risk budget allocation"""
    symbol: str
    allocated_risk: float
    risk_contribution: float
    marginal_risk: float
    position_weight: float
    concentration_penalty: float = 0.0


class RiskBudgeting:
    """
    Risk budgeting and allocation system

    Allocates capital based on risk contributions rather than dollar amounts
    """

    def __init__(
        self,
        total_risk_budget: float = 0.20,
        max_strategy_risk: float = 0.10,
        concentration_threshold: float = 0.25
    ):
        """
        Initialize risk budgeting

        Args:
            total_risk_budget: Total portfolio risk budget (default 20% annual vol)
            max_strategy_risk: Maximum risk per strategy (default 10%)
            concentration_threshold: Threshold for concentration penalties (default 25%)
        """
        self.total_risk_budget = total_risk_budget
        self.max_strategy_risk = max_strategy_risk
        self.concentration_threshold = concentration_threshold

    def calculate_risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk contribution of each position

        Risk Contribution = Weight * Marginal Risk

        Args:
            weights: Position weights
            cov_matrix: Covariance matrix

        Returns:
            Array of risk contributions
        """
        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

        if portfolio_var == 0:
            return np.zeros_like(weights)

        # Portfolio volatility
        portfolio_vol = np.sqrt(portfolio_var)

        # Marginal risk contribution
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol

        # Risk contribution
        risk_contribution = weights * marginal_risk

        return risk_contribution

    def calculate_marginal_risk(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate marginal contribution to risk

        Args:
            weights: Position weights
            cov_matrix: Covariance matrix

        Returns:
            Array of marginal risk contributions
        """
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

        if portfolio_var == 0:
            return np.zeros_like(weights)

        portfolio_vol = np.sqrt(portfolio_var)

        # Marginal risk = d(portfolio_vol) / d(weight_i)
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol

        return marginal_risk

    def equal_risk_contribution(
        self,
        cov_matrix: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Calculate Equal Risk Contribution (ERC) weights

        All positions contribute equally to portfolio risk

        Args:
            cov_matrix: Covariance matrix
            bounds: Weight bounds for each position

        Returns:
            Optimal weights
        """
        n_assets = cov_matrix.shape[0]

        # Objective: minimize sum of squared differences in risk contributions
        def objective(weights):
            risk_contrib = self.calculate_risk_contribution(weights, cov_matrix)

            # Target is equal risk contribution
            target_contrib = risk_contrib.sum() / n_assets

            # Sum of squared deviations
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # Bounds: all weights between 0 and 1
        if bounds is None:
            bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x

    def risk_parity_allocation(
        self,
        cov_matrix: np.ndarray,
        target_vol: float
    ) -> np.ndarray:
        """
        Calculate risk parity allocation

        Args:
            cov_matrix: Covariance matrix
            target_vol: Target portfolio volatility

        Returns:
            Optimal weights
        """
        # Get ERC weights
        erc_weights = self.equal_risk_contribution(cov_matrix)

        # Calculate current portfolio vol
        portfolio_var = np.dot(erc_weights, np.dot(cov_matrix, erc_weights))
        portfolio_vol = np.sqrt(portfolio_var)

        # Scale to target volatility
        if portfolio_vol > 0:
            scale = target_vol / portfolio_vol
            scaled_weights = erc_weights * scale
        else:
            scaled_weights = erc_weights

        return scaled_weights

    def allocate_risk_budget(
        self,
        strategies: List[str],
        strategy_risks: Dict[str, float],
        strategy_returns: pd.DataFrame
    ) -> Dict[str, RiskBudgetResult]:
        """
        Allocate risk budget across strategies

        Args:
            strategies: List of strategy names
            strategy_risks: Dictionary of strategy -> risk (volatility)
            strategy_returns: DataFrame of strategy returns

        Returns:
            Dictionary of strategy -> RiskBudgetResult
        """
        # Calculate covariance matrix
        cov_matrix = strategy_returns.cov().values

        # Equal risk contribution weights
        erc_weights = self.equal_risk_contribution(cov_matrix)

        # Calculate risk contributions
        risk_contrib = self.calculate_risk_contribution(erc_weights, cov_matrix)
        marginal_risk = self.calculate_marginal_risk(erc_weights, cov_matrix)

        # Build results
        results = {}
        for i, strategy in enumerate(strategies):
            # Apply concentration penalty if weight too high
            concentration_penalty = 0.0
            if erc_weights[i] > self.concentration_threshold:
                excess = erc_weights[i] - self.concentration_threshold
                concentration_penalty = excess * 0.5  # 50% penalty on excess

            results[strategy] = RiskBudgetResult(
                symbol=strategy,
                allocated_risk=strategy_risks.get(strategy, 0.0),
                risk_contribution=risk_contrib[i],
                marginal_risk=marginal_risk[i],
                position_weight=erc_weights[i],
                concentration_penalty=concentration_penalty
            )

        return results

    def dynamic_risk_budget_adjustment(
        self,
        base_allocation: Dict[str, float],
        performance_metrics: Dict[str, Dict],
        lookback_sharpe: int = 60
    ) -> Dict[str, float]:
        """
        Dynamically adjust risk budget based on performance

        Args:
            base_allocation: Base risk allocation
            performance_metrics: Dictionary of strategy -> metrics (sharpe, drawdown, etc.)
            lookback_sharpe: Days to calculate Sharpe ratio

        Returns:
            Adjusted risk allocation
        """
        adjusted_allocation = {}

        total_adjustment = 0.0

        for strategy, base_risk in base_allocation.items():
            metrics = performance_metrics.get(strategy, {})

            # Get performance metrics
            sharpe = metrics.get('sharpe_ratio', 0.0)
            drawdown = metrics.get('current_drawdown', 0.0)

            # Adjustment factors
            sharpe_adjustment = 1.0
            if sharpe > 1.5:
                sharpe_adjustment = 1.2  # Increase allocation
            elif sharpe < 0.5:
                sharpe_adjustment = 0.8  # Reduce allocation

            # Drawdown adjustment
            dd_adjustment = 1.0
            if abs(drawdown) > 0.10:
                dd_adjustment = 0.7  # Reduce significantly if in drawdown
            elif abs(drawdown) > 0.05:
                dd_adjustment = 0.85

            # Combined adjustment
            combined_adjustment = sharpe_adjustment * dd_adjustment

            # Apply bounds
            combined_adjustment = np.clip(combined_adjustment, 0.5, 1.5)

            # Adjusted risk
            adjusted_risk = base_risk * combined_adjustment

            # Enforce max strategy risk
            if adjusted_risk > self.max_strategy_risk:
                adjusted_risk = self.max_strategy_risk

            adjusted_allocation[strategy] = adjusted_risk
            total_adjustment += adjusted_risk

        # Normalize to total risk budget
        if total_adjustment > 0:
            scale = self.total_risk_budget / total_adjustment
            adjusted_allocation = {
                k: v * scale
                for k, v in adjusted_allocation.items()
            }

        return adjusted_allocation

    def concentration_penalty(
        self,
        weights: np.ndarray
    ) -> float:
        """
        Calculate concentration penalty

        Uses Herfindahl index

        Args:
            weights: Position weights

        Returns:
            Concentration penalty (0 = perfectly diversified, 1 = concentrated)
        """
        # Herfindahl index
        herfindahl = np.sum(weights ** 2)

        # Normalize (1/n = perfectly diversified, 1 = fully concentrated)
        n = len(weights)
        normalized_h = (herfindahl - 1/n) / (1 - 1/n)

        return normalized_h

    def marginal_var_allocation(
        self,
        weights: np.ndarray,
        returns_df: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> np.ndarray:
        """
        Allocate based on marginal VaR contributions

        Args:
            weights: Current weights
            returns_df: Returns dataframe
            confidence_level: VaR confidence level

        Returns:
            Marginal VaR for each position
        """
        epsilon = 0.001
        marginal_vars = np.zeros(len(weights))

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Calculate base VaR
        base_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # Calculate marginal VaR
        for i in range(len(weights)):
            # Increase weight slightly
            weights_up = weights.copy()
            weights_up[i] += epsilon

            # Normalize
            weights_up = weights_up / weights_up.sum()

            # Calculate new VaR
            portfolio_returns_up = (returns_df * weights_up).sum(axis=1)
            var_up = -np.percentile(portfolio_returns_up, (1 - confidence_level) * 100)

            # Marginal VaR
            marginal_vars[i] = (var_up - base_var) / epsilon

        return marginal_vars

    def enforce_strategy_limits(
        self,
        allocation: Dict[str, float],
        strategy_limits: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Enforce per-strategy risk limits

        Args:
            allocation: Proposed risk allocation
            strategy_limits: Maximum risk per strategy

        Returns:
            Adjusted allocation respecting limits
        """
        adjusted = {}

        for strategy, risk in allocation.items():
            limit = strategy_limits.get(strategy, self.max_strategy_risk)

            if risk > limit:
                adjusted[strategy] = limit
            else:
                adjusted[strategy] = risk

        # Normalize to maintain total risk budget
        total_risk = sum(adjusted.values())
        if total_risk > 0 and total_risk != self.total_risk_budget:
            scale = self.total_risk_budget / total_risk
            adjusted = {k: v * scale for k, v in adjusted.items()}

        return adjusted

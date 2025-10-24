"""
Risk parity allocation for portfolio construction.

Implements multiple risk parity approaches:
- Equal Risk Contribution (ERC)
- Hierarchical Risk Parity (HRP)
- Dynamic risk budget adjustment
- Leverage constraints
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from loguru import logger


class RiskParityAllocator:
    """
    Risk parity portfolio allocator.

    Risk parity aims to allocate risk equally across portfolio components,
    rather than allocating capital equally. This approach provides better
    diversification, especially when asset volatilities differ significantly.

    Attributes:
        target_risk (float): Target portfolio volatility (annualized)
        max_leverage (float): Maximum leverage allowed
        risk_budget (pd.Series): Custom risk budgets for each asset (optional)
        correlation_lookback (int): Lookback period for correlation calculation
    """

    def __init__(
        self,
        target_risk: float = 0.15,
        max_leverage: float = 2.0,
        risk_budget: Optional[pd.Series] = None,
        correlation_lookback: int = 252,
    ):
        """
        Initialize risk parity allocator.

        Args:
            target_risk: Target portfolio volatility (e.g., 0.15 = 15% annualized)
            max_leverage: Maximum allowed leverage
            risk_budget: Custom risk budgets (if None, equal risk allocation)
            correlation_lookback: Days to lookback for correlation estimation
        """
        self.target_risk = target_risk
        self.max_leverage = max_leverage
        self.risk_budget = risk_budget
        self.correlation_lookback = correlation_lookback

        logger.info(
            f"RiskParityAllocator initialized: target_risk={target_risk}, "
            f"max_leverage={max_leverage}"
        )

    def calculate_risk_contributions(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate risk contribution of each asset to portfolio risk.

        Risk contribution of asset i = w_i * (Cov * w)_i / portfolio_risk
        where Cov is the covariance matrix and w is the weights vector.

        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix of returns

        Returns:
            Risk contribution of each asset (sums to 1)
        """
        # Portfolio variance
        portfolio_variance = weights.values @ covariance_matrix.values @ weights.values

        if portfolio_variance <= 0:
            logger.warning("Non-positive portfolio variance, returning equal risk contributions")
            return pd.Series(1.0 / len(weights), index=weights.index)

        portfolio_risk = np.sqrt(portfolio_variance)

        # Marginal risk contribution: Cov * w
        marginal_contrib = covariance_matrix.values @ weights.values

        # Total risk contribution: w_i * marginal_contrib_i
        risk_contrib = weights.values * marginal_contrib / portfolio_risk

        # Normalize to sum to 1
        risk_contrib_series = pd.Series(risk_contrib, index=weights.index)
        risk_contrib_series = risk_contrib_series / risk_contrib_series.sum()

        return risk_contrib_series

    def equal_risk_contribution(
        self,
        covariance_matrix: pd.DataFrame,
        risk_budget: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calculate Equal Risk Contribution (ERC) portfolio weights.

        ERC portfolio allocates risk equally across all assets. Each asset
        contributes the same amount to total portfolio risk.

        Uses convex optimization to minimize the difference between
        actual and target risk contributions.

        Args:
            covariance_matrix: Covariance matrix of asset returns
            risk_budget: Target risk contribution for each asset (equal if None)

        Returns:
            ERC portfolio weights
        """
        n_assets = len(covariance_matrix)

        if n_assets == 0:
            logger.warning("Empty covariance matrix")
            return pd.Series(dtype=float)

        if risk_budget is None:
            # Equal risk budget
            risk_budget = pd.Series(1.0 / n_assets, index=covariance_matrix.index)
        else:
            # Normalize risk budget
            risk_budget = risk_budget / risk_budget.sum()

        try:
            # Define optimization variable
            w = cp.Variable(n_assets)

            # Portfolio variance
            portfolio_variance = cp.quad_form(w, covariance_matrix.values)

            # Risk contributions
            # RC_i = w_i * (Cov * w)_i / sqrt(portfolio_variance)
            marginal_contrib = covariance_matrix.values @ w

            # Objective: Minimize squared deviation from target risk budget
            # Using a squared error formulation for convexity
            risk_contrib_error = 0
            for i in range(n_assets):
                target_rc = risk_budget.iloc[i]
                # Approximate: w_i * marginal_contrib_i ≈ target_rc * portfolio_risk
                risk_contrib_error += cp.square(
                    w[i] * marginal_contrib[i] - target_rc * cp.sqrt(portfolio_variance)
                )

            objective = cp.Minimize(risk_contrib_error)

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,  # Long-only
            ]

            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"ERC optimization status: {problem.status}, using equal weights")
                return pd.Series(1.0 / n_assets, index=covariance_matrix.index)

            # Extract weights
            erc_weights = pd.Series(w.value, index=covariance_matrix.index)
            erc_weights = erc_weights.clip(lower=0)
            erc_weights = erc_weights / erc_weights.sum()

            # Verify risk contributions
            risk_contrib = self.calculate_risk_contributions(erc_weights, covariance_matrix)
            logger.info(
                f"ERC weights calculated: risk_contrib range "
                f"[{risk_contrib.min():.3f}, {risk_contrib.max():.3f}]"
            )

            return erc_weights

        except Exception as e:
            logger.error(f"ERC optimization failed: {e}, using equal weights")
            return pd.Series(1.0 / n_assets, index=covariance_matrix.index)

    def hierarchical_risk_parity(
        self,
        covariance_matrix: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Calculate Hierarchical Risk Parity (HRP) portfolio weights.

        HRP uses hierarchical clustering to group similar assets and
        allocates risk top-down through the hierarchy. This approach
        is more stable than traditional optimization methods.

        Algorithm:
        1. Compute distance matrix from correlations
        2. Perform hierarchical clustering
        3. Recursively bisect and allocate risk

        Args:
            covariance_matrix: Covariance matrix of asset returns
            returns: Historical returns (used if covariance not provided)

        Returns:
            HRP portfolio weights
        """
        n_assets = len(covariance_matrix)

        if n_assets == 0:
            logger.warning("Empty covariance matrix")
            return pd.Series(dtype=float)

        try:
            # Correlation matrix
            volatilities = np.sqrt(np.diag(covariance_matrix))
            correlation_matrix = covariance_matrix.values / np.outer(volatilities, volatilities)
            correlation_matrix = np.clip(correlation_matrix, -1, 1)

            # Distance matrix: d = sqrt(0.5 * (1 - correlation))
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))

            # Hierarchical clustering
            condensed_distance = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distance, method='single')

            # Get cluster order (quasi-diagonalization)
            cluster_order = self._get_cluster_order(linkage_matrix, n_assets)

            # Recursive bisection allocation
            weights = self._recursive_bisection(
                covariance_matrix,
                cluster_order,
            )

            hrp_weights = pd.Series(weights, index=covariance_matrix.index)
            hrp_weights = hrp_weights / hrp_weights.sum()

            logger.info(
                f"HRP weights calculated: {n_assets} assets, "
                f"range [{hrp_weights.min():.3f}, {hrp_weights.max():.3f}]"
            )

            return hrp_weights

        except Exception as e:
            logger.error(f"HRP calculation failed: {e}, using equal weights")
            return pd.Series(1.0 / n_assets, index=covariance_matrix.index)

    def _get_cluster_order(self, linkage_matrix: np.ndarray, n_assets: int) -> List[int]:
        """
        Get quasi-diagonalization order from linkage matrix.

        Args:
            linkage_matrix: Hierarchical clustering linkage matrix
            n_assets: Number of assets

        Returns:
            List of asset indices in cluster order
        """
        # Extract dendrogram ordering
        dend = dendrogram(linkage_matrix, no_plot=True)
        return dend['leaves']

    def _recursive_bisection(
        self,
        covariance_matrix: pd.DataFrame,
        cluster_order: List[int],
    ) -> np.ndarray:
        """
        Recursively allocate weights using bisection.

        Args:
            covariance_matrix: Covariance matrix
            cluster_order: Order of assets from clustering

        Returns:
            Array of portfolio weights
        """
        n_assets = len(cluster_order)
        weights = np.ones(n_assets)

        # Recursive helper
        def _bisect(indices: List[int]) -> None:
            nonlocal weights

            if len(indices) == 1:
                return

            # Split cluster in half
            mid = len(indices) // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]

            # Calculate cluster variances
            left_cov = covariance_matrix.iloc[left_indices, left_indices]
            right_cov = covariance_matrix.iloc[right_indices, right_indices]

            # Equal weight within each cluster for variance calculation
            left_w = np.ones(len(left_indices)) / len(left_indices)
            right_w = np.ones(len(right_indices)) / len(right_indices)

            left_var = left_w @ left_cov.values @ left_w
            right_var = right_w @ right_cov.values @ right_w

            # Inverse variance allocation
            if left_var + right_var > 0:
                left_alloc = 1.0 - left_var / (left_var + right_var)
                right_alloc = 1.0 - right_var / (left_var + right_var)
            else:
                left_alloc = 0.5
                right_alloc = 0.5

            # Update weights
            for idx in left_indices:
                weights[idx] *= left_alloc
            for idx in right_indices:
                weights[idx] *= right_alloc

            # Recurse
            _bisect(left_indices)
            _bisect(right_indices)

        _bisect(cluster_order)
        return weights

    def adjust_for_leverage(
        self,
        weights: pd.Series,
        target_risk: float,
        portfolio_risk: float,
    ) -> pd.Series:
        """
        Adjust weights to target risk level using leverage.

        If portfolio risk < target risk, apply leverage to scale up.
        If portfolio risk > target risk, scale down.

        Args:
            weights: Base portfolio weights
            target_risk: Target portfolio volatility
            portfolio_risk: Current portfolio volatility

        Returns:
            Leverage-adjusted weights
        """
        if portfolio_risk <= 0:
            logger.warning("Non-positive portfolio risk, returning unadjusted weights")
            return weights

        # Calculate required leverage
        leverage = target_risk / portfolio_risk
        leverage = np.clip(leverage, 0.1, self.max_leverage)

        adjusted_weights = weights * leverage

        logger.debug(
            f"Leverage adjustment: {leverage:.3f}x "
            f"(target_risk={target_risk:.3f}, portfolio_risk={portfolio_risk:.3f})"
        )

        return adjusted_weights

    def dynamic_risk_budget(
        self,
        returns: pd.DataFrame,
        market_regime: Optional[str] = None,
    ) -> pd.Series:
        """
        Calculate dynamic risk budget based on market conditions.

        Adjusts risk budget based on:
        - Asset volatility (inverse vol weighting)
        - Market regime (risk-on vs risk-off)
        - Recent performance

        Args:
            returns: Historical returns DataFrame
            market_regime: Optional market regime indicator ('risk_on', 'risk_off')

        Returns:
            Dynamic risk budget for each asset
        """
        # Calculate rolling volatility
        vols = returns.rolling(window=self.correlation_lookback).std()
        recent_vols = vols.iloc[-1]

        # Inverse volatility weighting
        inverse_vols = 1.0 / (recent_vols + 1e-8)
        risk_budget = inverse_vols / inverse_vols.sum()

        # Adjust for market regime
        if market_regime == 'risk_off':
            # Increase allocation to lower-risk assets
            vol_ranks = recent_vols.rank()
            regime_adjustment = 1.0 / (vol_ranks + 1)
            risk_budget = risk_budget * regime_adjustment
            risk_budget = risk_budget / risk_budget.sum()

        elif market_regime == 'risk_on':
            # More equal weighting in risk-on regime
            equal_weight = pd.Series(1.0 / len(risk_budget), index=risk_budget.index)
            risk_budget = 0.7 * risk_budget + 0.3 * equal_weight

        logger.info(
            f"Dynamic risk budget calculated (regime={market_regime}): "
            f"range [{risk_budget.min():.3f}, {risk_budget.max():.3f}]"
        )

        return risk_budget

    def allocate(
        self,
        returns: pd.DataFrame,
        method: str = 'erc',
        risk_budget: Optional[pd.Series] = None,
        market_regime: Optional[str] = None,
    ) -> Dict[str, pd.Series]:
        """
        Main allocation method - applies risk parity allocation.

        Args:
            returns: Historical returns DataFrame (time × assets)
            method: Allocation method ('erc', 'hrp', or 'both')
            risk_budget: Custom risk budget (if None, calculated dynamically)
            market_regime: Market regime for dynamic adjustments

        Returns:
            Dictionary with:
                - weights: Portfolio weights
                - risk_contributions: Risk contribution by asset
                - method: Method used
        """
        logger.info(f"Starting risk parity allocation with method={method}")

        # Calculate covariance matrix
        covariance_matrix = returns.cov()

        # Dynamic risk budget if not provided
        if risk_budget is None and method == 'erc':
            risk_budget = self.dynamic_risk_budget(returns, market_regime)

        # Calculate weights based on method
        if method == 'erc':
            weights = self.equal_risk_contribution(covariance_matrix, risk_budget)
        elif method == 'hrp':
            weights = self.hierarchical_risk_parity(covariance_matrix, returns)
        elif method == 'both':
            # Blend ERC and HRP
            erc_weights = self.equal_risk_contribution(covariance_matrix, risk_budget)
            hrp_weights = self.hierarchical_risk_parity(covariance_matrix, returns)
            weights = 0.5 * erc_weights + 0.5 * hrp_weights
            weights = weights / weights.sum()
        else:
            logger.warning(f"Unknown method {method}, using ERC")
            weights = self.equal_risk_contribution(covariance_matrix, risk_budget)

        # Calculate portfolio risk
        portfolio_variance = weights.values @ covariance_matrix.values @ weights.values
        portfolio_risk = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized

        # Adjust for target risk (apply leverage if needed)
        weights = self.adjust_for_leverage(weights, self.target_risk, portfolio_risk)

        # Recalculate risk after leverage adjustment
        portfolio_variance = weights.values @ covariance_matrix.values @ weights.values
        portfolio_risk = np.sqrt(portfolio_variance) * np.sqrt(252)

        # Calculate final risk contributions
        risk_contributions = self.calculate_risk_contributions(weights, covariance_matrix)

        logger.info(
            f"Risk parity allocation complete: portfolio_risk={portfolio_risk:.3f}, "
            f"total_leverage={weights.sum():.3f}"
        )

        return {
            'weights': weights,
            'risk_contributions': risk_contributions,
            'portfolio_risk': portfolio_risk,
            'method': method,
        }

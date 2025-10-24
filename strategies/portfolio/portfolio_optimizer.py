"""
Portfolio optimizer implementing Carver's systematic portfolio construction.

Based on Rob Carver's "Leveraged Trading" and "Systematic Trading" methodologies,
this module provides portfolio optimization with correlation adjustment, diversification
multipliers, and position inertia for turnover reduction.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger


class PortfolioOptimizer:
    """
    Portfolio optimizer using Carver's systematic approach.

    Key features:
    - Forecast combination with correlation adjustment
    - Diversification multiplier calculation
    - Position inertia for turnover reduction
    - Risk allocation across strategies
    - Instrument weight capping (20% max)
    - Convex optimization using cvxpy

    Attributes:
        max_instrument_weight (float): Maximum weight for single instrument (default: 0.20)
        max_strategy_weight (float): Maximum weight for single strategy (default: 0.40)
        target_risk (float): Target portfolio volatility (default: 0.20 = 20% annualized)
        inertia_factor (float): Position inertia to reduce turnover (default: 0.10)
        min_diversification_mult (float): Minimum diversification multiplier (default: 1.0)
        max_diversification_mult (float): Maximum diversification multiplier (default: 2.5)
    """

    def __init__(
        self,
        max_instrument_weight: float = 0.20,
        max_strategy_weight: float = 0.40,
        target_risk: float = 0.20,
        inertia_factor: float = 0.10,
        min_diversification_mult: float = 1.0,
        max_diversification_mult: float = 2.5,
    ):
        """
        Initialize portfolio optimizer.

        Args:
            max_instrument_weight: Maximum weight for single instrument (e.g., 0.20 = 20%)
            max_strategy_weight: Maximum weight for single strategy
            target_risk: Target portfolio volatility (annualized)
            inertia_factor: Position inertia factor (0.10 = 10% threshold)
            min_diversification_mult: Minimum diversification multiplier
            max_diversification_mult: Maximum diversification multiplier
        """
        self.max_instrument_weight = max_instrument_weight
        self.max_strategy_weight = max_strategy_weight
        self.target_risk = target_risk
        self.inertia_factor = inertia_factor
        self.min_diversification_mult = min_diversification_mult
        self.max_diversification_mult = max_diversification_mult

        logger.info(
            f"PortfolioOptimizer initialized: max_instrument_weight={max_instrument_weight}, "
            f"target_risk={target_risk}, inertia_factor={inertia_factor}"
        )

    def calculate_diversification_multiplier(
        self,
        correlations: pd.DataFrame,
        weights: Optional[pd.Series] = None,
    ) -> float:
        """
        Calculate diversification multiplier (IDM - Instrument Diversification Multiplier).

        The IDM shows how much we can leverage up due to diversification.
        Higher diversification allows higher leverage while maintaining target risk.

        Formula: IDM = 1 / sqrt(w' * C * w)
        where w is weights vector and C is correlation matrix

        Args:
            correlations: Correlation matrix of returns
            weights: Portfolio weights (equal weight if None)

        Returns:
            Diversification multiplier (typically 1.0 to 2.5)
        """
        if correlations.empty:
            logger.warning("Empty correlation matrix, returning min diversification multiplier")
            return self.min_diversification_mult

        n_instruments = len(correlations)

        if weights is None:
            # Equal weight
            weights = pd.Series(1.0 / n_instruments, index=correlations.index)

        # Ensure weights sum to 1
        weights = weights / weights.sum()

        try:
            # Calculate portfolio variance: w' * C * w
            portfolio_variance = weights.values @ correlations.values @ weights.values

            if portfolio_variance <= 0:
                logger.warning(f"Non-positive portfolio variance: {portfolio_variance}")
                return self.min_diversification_mult

            # IDM = 1 / sqrt(portfolio_variance)
            idm = 1.0 / np.sqrt(portfolio_variance)

            # Clip to reasonable bounds
            idm = np.clip(idm, self.min_diversification_mult, self.max_diversification_mult)

            logger.debug(f"Diversification multiplier: {idm:.3f} for {n_instruments} instruments")
            return idm

        except Exception as e:
            logger.error(f"Error calculating diversification multiplier: {e}")
            return self.min_diversification_mult

    def combine_forecasts(
        self,
        forecasts: pd.DataFrame,
        correlations: pd.DataFrame,
        forecast_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Combine multiple forecasts with correlation adjustment.

        Uses Carver's forecast combination methodology:
        1. Apply forecast weights (if specified)
        2. Adjust for correlations between forecasts
        3. Scale to target forecast range (-20 to +20)

        Args:
            forecasts: DataFrame of forecasts (strategies × instruments)
            correlations: Correlation matrix between strategies
            forecast_weights: Optional weights for each strategy

        Returns:
            Combined forecast series
        """
        if forecasts.empty:
            logger.warning("Empty forecasts, returning zeros")
            return pd.Series(0, index=forecasts.columns)

        n_strategies = len(forecasts)

        if forecast_weights is None:
            # Equal weight across strategies
            forecast_weights = pd.Series(1.0 / n_strategies, index=forecasts.index)

        # Normalize weights
        forecast_weights = forecast_weights / forecast_weights.sum()

        # Calculate forecast diversification multiplier (FDM)
        fdm = self.calculate_diversification_multiplier(correlations, forecast_weights)

        # Combine forecasts: weighted average * FDM
        combined = (forecasts.T @ forecast_weights) * fdm

        # Scale to target range (-20 to +20)
        combined = np.clip(combined, -20, 20)

        logger.debug(
            f"Combined {n_strategies} forecasts with FDM={fdm:.3f}, "
            f"range=[{combined.min():.2f}, {combined.max():.2f}]"
        )

        return combined

    def optimize_weights(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
        constraints: Optional[Dict] = None,
    ) -> pd.Series:
        """
        Optimize portfolio weights using convex optimization.

        Objective: Maximize Sharpe ratio subject to constraints
        - Maximum weight per instrument
        - Turnover constraints (position inertia)
        - Long-only constraint
        - Weights sum to 1

        Args:
            expected_returns: Expected returns for each instrument
            covariance_matrix: Covariance matrix of returns
            current_weights: Current portfolio weights (for inertia)
            constraints: Additional constraints dict

        Returns:
            Optimized portfolio weights
        """
        n_assets = len(expected_returns)

        if n_assets == 0:
            logger.warning("No assets to optimize")
            return pd.Series(dtype=float)

        try:
            # Define optimization variable
            w = cp.Variable(n_assets)

            # Objective: Maximize Sharpe ratio (approximately maximize returns / risk)
            # Using a quadratic utility approximation
            risk_aversion = 1.0  # Adjust based on risk preference
            portfolio_return = expected_returns.values @ w
            portfolio_variance = cp.quad_form(w, covariance_matrix.values)
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)

            # Constraints
            constraint_list = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,  # Long-only
                w <= self.max_instrument_weight,  # Max instrument weight
            ]

            # Position inertia constraint (if current weights provided)
            if current_weights is not None and len(current_weights) == n_assets:
                turnover = cp.norm(w - current_weights.values, 1)
                constraint_list.append(turnover <= self.inertia_factor)

            # Additional custom constraints
            if constraints:
                if 'min_weights' in constraints:
                    constraint_list.append(w >= constraints['min_weights'])
                if 'max_weights' in constraints:
                    constraint_list.append(w <= constraints['max_weights'])

            # Solve optimization problem
            problem = cp.Problem(objective, constraint_list)
            problem.solve(solver=cp.ECOS)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"Optimization status: {problem.status}, using equal weights")
                return pd.Series(1.0 / n_assets, index=expected_returns.index)

            # Extract optimal weights
            optimal_weights = pd.Series(w.value, index=expected_returns.index)

            # Ensure non-negative and normalized
            optimal_weights = optimal_weights.clip(lower=0)
            optimal_weights = optimal_weights / optimal_weights.sum()

            logger.info(
                f"Optimization complete: {len(optimal_weights)} assets, "
                f"max_weight={optimal_weights.max():.3f}, "
                f"portfolio_return={portfolio_return.value:.4f}"
            )

            return optimal_weights

        except Exception as e:
            logger.error(f"Optimization failed: {e}, using equal weights")
            return pd.Series(1.0 / n_assets, index=expected_returns.index)

    def allocate_risk(
        self,
        strategy_returns: pd.DataFrame,
        strategy_vols: pd.Series,
        target_allocations: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Allocate risk budget across strategies.

        Each strategy gets a portion of the total risk budget based on:
        - Target allocations (if specified)
        - Inverse volatility weighting
        - Strategy capacity constraints

        Args:
            strategy_returns: Historical returns by strategy
            strategy_vols: Volatility of each strategy
            target_allocations: Target risk allocation (optional)

        Returns:
            Risk-adjusted position sizes by strategy
        """
        n_strategies = len(strategy_vols)

        if n_strategies == 0:
            logger.warning("No strategies to allocate risk")
            return pd.Series(dtype=float)

        if target_allocations is None:
            # Equal risk allocation
            target_allocations = pd.Series(1.0 / n_strategies, index=strategy_vols.index)

        # Normalize target allocations
        target_allocations = target_allocations / target_allocations.sum()

        # Risk-adjusted position sizes: allocation / volatility
        # Higher vol strategies get smaller positions to maintain equal risk
        risk_adjusted_sizes = target_allocations / strategy_vols

        # Normalize to sum to target portfolio risk
        risk_adjusted_sizes = risk_adjusted_sizes / risk_adjusted_sizes.sum() * self.target_risk

        # Cap individual strategy risk
        risk_adjusted_sizes = risk_adjusted_sizes.clip(upper=self.max_strategy_weight * self.target_risk)

        logger.info(
            f"Risk allocated across {n_strategies} strategies: "
            f"range=[{risk_adjusted_sizes.min():.3f}, {risk_adjusted_sizes.max():.3f}]"
        )

        return risk_adjusted_sizes

    def apply_position_inertia(
        self,
        new_weights: pd.Series,
        current_weights: pd.Series,
        threshold: Optional[float] = None,
    ) -> pd.Series:
        """
        Apply position inertia to reduce turnover.

        Only adjust positions if the difference exceeds a threshold.
        This reduces transaction costs from small rebalancing trades.

        Args:
            new_weights: Newly calculated optimal weights
            current_weights: Current portfolio weights
            threshold: Minimum change threshold (uses self.inertia_factor if None)

        Returns:
            Adjusted weights with inertia applied
        """
        if threshold is None:
            threshold = self.inertia_factor

        # Align indices
        all_assets = new_weights.index.union(current_weights.index)
        new_weights = new_weights.reindex(all_assets, fill_value=0)
        current_weights = current_weights.reindex(all_assets, fill_value=0)

        # Calculate weight differences
        weight_diff = (new_weights - current_weights).abs()

        # Keep current weight if change is below threshold
        adjusted_weights = new_weights.copy()
        small_changes = weight_diff < threshold
        adjusted_weights[small_changes] = current_weights[small_changes]

        # Renormalize to sum to 1
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum()

        n_changed = (~small_changes).sum()
        logger.debug(
            f"Position inertia applied: {n_changed}/{len(all_assets)} positions changed "
            f"(threshold={threshold:.3f})"
        )

        return adjusted_weights

    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
        strategy_correlations: Optional[pd.DataFrame] = None,
        instrument_correlations: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Union[pd.Series, float, pd.DataFrame]]:
        """
        Complete portfolio construction pipeline.

        Integrates all components:
        1. Combine forecasts from multiple strategies
        2. Calculate diversification multipliers
        3. Optimize instrument weights
        4. Allocate risk across strategies
        5. Apply position inertia

        Args:
            signals: Strategy signals (strategies × instruments)
            returns: Historical returns (time × instruments)
            current_weights: Current portfolio weights
            strategy_correlations: Correlation matrix between strategies
            instrument_correlations: Correlation matrix between instruments

        Returns:
            Dictionary containing:
                - weights: Final portfolio weights
                - diversification_multiplier: IDM value
                - expected_return: Portfolio expected return
                - expected_risk: Portfolio expected volatility
                - turnover: Portfolio turnover from current weights
        """
        logger.info("Starting portfolio construction")

        # Default correlations if not provided
        if strategy_correlations is None:
            strategy_correlations = signals.T.corr()

        if instrument_correlations is None:
            instrument_correlations = returns.corr()

        # Step 1: Combine forecasts
        combined_signals = self.combine_forecasts(
            signals, strategy_correlations
        )

        # Step 2: Calculate expected returns (using signals as proxy)
        # In practice, you'd use more sophisticated return forecasting
        expected_returns = combined_signals / 20.0  # Scale from [-20, 20] to fraction

        # Step 3: Calculate covariance matrix
        covariance_matrix = returns.cov()

        # Step 4: Calculate diversification multiplier
        idm = self.calculate_diversification_multiplier(instrument_correlations)

        # Step 5: Optimize weights
        optimal_weights = self.optimize_weights(
            expected_returns,
            covariance_matrix,
            current_weights,
        )

        # Step 6: Apply position inertia
        if current_weights is not None:
            final_weights = self.apply_position_inertia(
                optimal_weights, current_weights
            )
        else:
            final_weights = optimal_weights

        # Step 7: Calculate portfolio metrics
        portfolio_return = (final_weights * expected_returns).sum()
        portfolio_variance = final_weights.values @ covariance_matrix.values @ final_weights.values
        portfolio_risk = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized

        # Calculate turnover
        if current_weights is not None:
            turnover = (final_weights - current_weights.reindex(final_weights.index, fill_value=0)).abs().sum()
        else:
            turnover = 0.0

        logger.info(
            f"Portfolio construction complete: "
            f"return={portfolio_return:.4f}, risk={portfolio_risk:.4f}, "
            f"IDM={idm:.3f}, turnover={turnover:.3f}"
        )

        return {
            'weights': final_weights,
            'diversification_multiplier': idm,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'turnover': turnover,
            'covariance_matrix': covariance_matrix,
        }

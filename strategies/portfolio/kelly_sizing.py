"""
Kelly criterion position sizing for optimal growth.

The Kelly criterion determines the optimal position size to maximize
long-term growth rate while managing risk of ruin. This module implements
multiple Kelly variants suitable for systematic trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from loguru import logger


class KellySizer:
    """
    Kelly criterion position sizing calculator.

    Implements multiple Kelly sizing approaches:
    - Classic Kelly for single bets
    - Multi-asset Kelly optimization
    - Fractional Kelly with safety caps
    - Drawdown-adjusted Kelly
    - Meta-label confidence-weighted Kelly

    Attributes:
        max_kelly_fraction (float): Maximum Kelly fraction to use (default: 0.25)
        min_kelly_fraction (float): Minimum Kelly fraction (default: 0.01)
        confidence_scaling (bool): Whether to scale Kelly by meta-label confidence
        drawdown_adjustment (bool): Whether to reduce Kelly during drawdowns
    """

    def __init__(
        self,
        max_kelly_fraction: float = 0.25,
        min_kelly_fraction: float = 0.01,
        confidence_scaling: bool = True,
        drawdown_adjustment: bool = True,
    ):
        """
        Initialize Kelly sizer.

        Args:
            max_kelly_fraction: Maximum Kelly fraction (e.g., 0.25 = quarter Kelly)
            min_kelly_fraction: Minimum Kelly fraction to avoid zero positions
            confidence_scaling: Scale Kelly by meta-label confidence
            drawdown_adjustment: Reduce Kelly during drawdowns
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = min_kelly_fraction
        self.confidence_scaling = confidence_scaling
        self.drawdown_adjustment = drawdown_adjustment

        logger.info(
            f"KellySizer initialized: max_kelly={max_kelly_fraction}, "
            f"confidence_scaling={confidence_scaling}"
        )

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate classic Kelly fraction for a single bet.

        Kelly formula: f = (p*W - (1-p)*L) / (W*L)
        Simplified: f = p - (1-p)/(W/L)
        where:
            p = win rate (probability of winning)
            W = average win amount
            L = average loss amount

        Args:
            win_rate: Probability of winning (0 to 1)
            avg_win: Average win size (positive)
            avg_loss: Average loss size (positive)

        Returns:
            Kelly fraction (capped at max_kelly_fraction)
        """
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win_rate: {win_rate}, returning min Kelly")
            return self.min_kelly_fraction

        if avg_win <= 0 or avg_loss <= 0:
            logger.warning(f"Invalid win/loss: avg_win={avg_win}, avg_loss={avg_loss}")
            return self.min_kelly_fraction

        # Win/loss ratio
        win_loss_ratio = avg_win / avg_loss

        # Kelly fraction: f = p - (1-p)/(W/L)
        kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio

        # Ensure positive and within bounds
        kelly_fraction = np.clip(
            kelly_fraction,
            self.min_kelly_fraction,
            self.max_kelly_fraction
        )

        logger.debug(
            f"Kelly fraction: {kelly_fraction:.3f} "
            f"(win_rate={win_rate:.2f}, W/L={win_loss_ratio:.2f})"
        )

        return kelly_fraction

    def calculate_from_returns(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Calculate Kelly fraction from historical returns.

        Estimates win rate and average win/loss from return distribution.

        Args:
            returns: Series of historical returns

        Returns:
            Kelly fraction
        """
        if len(returns) == 0:
            logger.warning("Empty returns series")
            return self.min_kelly_fraction

        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            logger.warning("No wins or no losses in returns")
            return self.min_kelly_fraction

        # Calculate statistics
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        return self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

    def calculate_from_odds(
        self,
        probability: float,
        odds: float,
    ) -> float:
        """
        Calculate Kelly fraction from probability and odds.

        Alternative formulation: f = (p*odds - 1) / odds
        where odds = payout ratio (e.g., 2.0 means you get 2x your bet)

        Args:
            probability: Probability of winning (0 to 1)
            odds: Payout odds (e.g., 2.0 = 2:1)

        Returns:
            Kelly fraction
        """
        if probability <= 0 or probability >= 1:
            logger.warning(f"Invalid probability: {probability}")
            return self.min_kelly_fraction

        if odds <= 1.0:
            logger.warning(f"Invalid odds: {odds}, must be > 1.0")
            return self.min_kelly_fraction

        # Kelly fraction
        kelly_fraction = (probability * odds - 1) / odds

        kelly_fraction = np.clip(
            kelly_fraction,
            self.min_kelly_fraction,
            self.max_kelly_fraction
        )

        logger.debug(
            f"Kelly fraction from odds: {kelly_fraction:.3f} "
            f"(p={probability:.2f}, odds={odds:.2f})"
        )

        return kelly_fraction

    def multi_asset_kelly(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate Kelly-optimal weights for multiple assets.

        Multi-asset Kelly formula: f = C^(-1) * μ
        where:
            C = covariance matrix
            μ = expected returns vector
            f = Kelly fraction for each asset

        Args:
            expected_returns: Expected return for each asset
            covariance_matrix: Covariance matrix of returns

        Returns:
            Kelly fractions for each asset
        """
        if len(expected_returns) == 0:
            logger.warning("Empty expected returns")
            return pd.Series(dtype=float)

        try:
            # Solve: C * f = μ
            kelly_fractions = np.linalg.solve(
                covariance_matrix.values,
                expected_returns.values
            )

            kelly_series = pd.Series(kelly_fractions, index=expected_returns.index)

            # Cap individual fractions
            kelly_series = kelly_series.clip(
                lower=-self.max_kelly_fraction,
                upper=self.max_kelly_fraction
            )

            # Cap total exposure
            total_exposure = kelly_series.abs().sum()
            if total_exposure > self.max_kelly_fraction:
                kelly_series = kelly_series * (self.max_kelly_fraction / total_exposure)

            logger.info(
                f"Multi-asset Kelly: {len(kelly_series)} assets, "
                f"total_exposure={kelly_series.abs().sum():.3f}"
            )

            return kelly_series

        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed: {e}, using simple Kelly")
            # Fallback to simple Kelly per asset
            kelly_series = expected_returns / covariance_matrix.values.diagonal()
            kelly_series = kelly_series.clip(
                lower=self.min_kelly_fraction,
                upper=self.max_kelly_fraction
            )
            return kelly_series

    def confidence_weighted_kelly(
        self,
        base_kelly: float,
        meta_label_confidence: float,
    ) -> float:
        """
        Adjust Kelly fraction by meta-label confidence.

        Meta-labels provide confidence estimates for predictions.
        Higher confidence → use more of Kelly fraction.
        Lower confidence → use less (more conservative).

        Args:
            base_kelly: Base Kelly fraction
            meta_label_confidence: Confidence from meta-labeling (0 to 1)

        Returns:
            Confidence-adjusted Kelly fraction
        """
        if not self.confidence_scaling:
            return base_kelly

        if meta_label_confidence < 0 or meta_label_confidence > 1:
            logger.warning(f"Invalid confidence: {meta_label_confidence}")
            meta_label_confidence = 0.5

        # Scale Kelly by confidence squared (more conservative)
        adjusted_kelly = base_kelly * (meta_label_confidence ** 2)

        adjusted_kelly = np.clip(
            adjusted_kelly,
            self.min_kelly_fraction,
            self.max_kelly_fraction
        )

        logger.debug(
            f"Confidence-weighted Kelly: {adjusted_kelly:.3f} "
            f"(base={base_kelly:.3f}, confidence={meta_label_confidence:.2f})"
        )

        return adjusted_kelly

    def drawdown_adjusted_kelly(
        self,
        base_kelly: float,
        current_drawdown: float,
        max_acceptable_drawdown: float = 0.20,
    ) -> float:
        """
        Adjust Kelly fraction based on current drawdown.

        Reduce position sizes during drawdowns to avoid compounding losses.
        Increases conservatism as drawdown approaches maximum acceptable level.

        Args:
            base_kelly: Base Kelly fraction
            current_drawdown: Current drawdown (0 to 1, e.g., 0.15 = 15% drawdown)
            max_acceptable_drawdown: Maximum acceptable drawdown threshold

        Returns:
            Drawdown-adjusted Kelly fraction
        """
        if not self.drawdown_adjustment:
            return base_kelly

        current_drawdown = abs(current_drawdown)

        if current_drawdown >= max_acceptable_drawdown:
            logger.warning(
                f"Drawdown {current_drawdown:.1%} exceeds max {max_acceptable_drawdown:.1%}, "
                f"using minimum Kelly"
            )
            return self.min_kelly_fraction

        # Linear reduction in Kelly as drawdown increases
        # At 0% drawdown: use full Kelly
        # At max drawdown: use min Kelly
        drawdown_scalar = 1.0 - (current_drawdown / max_acceptable_drawdown)
        adjusted_kelly = base_kelly * drawdown_scalar

        adjusted_kelly = np.clip(
            adjusted_kelly,
            self.min_kelly_fraction,
            self.max_kelly_fraction
        )

        logger.debug(
            f"Drawdown-adjusted Kelly: {adjusted_kelly:.3f} "
            f"(base={base_kelly:.3f}, drawdown={current_drawdown:.1%})"
        )

        return adjusted_kelly

    def calculate_position_size(
        self,
        kelly_fraction: float,
        account_value: float,
        price: float,
        volatility: Optional[float] = None,
    ) -> int:
        """
        Convert Kelly fraction to actual position size (number of shares/contracts).

        Args:
            kelly_fraction: Kelly fraction (portion of capital)
            account_value: Total account value
            price: Current price per unit
            volatility: Optional volatility for risk-based sizing

        Returns:
            Number of shares/contracts to trade
        """
        if account_value <= 0 or price <= 0:
            logger.warning(f"Invalid account_value or price")
            return 0

        # Capital to allocate
        capital_allocation = kelly_fraction * account_value

        # Basic position size
        position_size = capital_allocation / price

        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            # Scale down for high volatility assets
            vol_scalar = 0.15 / max(volatility, 0.01)  # Target 15% vol
            position_size = position_size * min(vol_scalar, 1.0)

        # Round to integer
        position_size = int(np.floor(position_size))

        logger.debug(
            f"Position size: {position_size} units "
            f"(kelly={kelly_fraction:.3f}, capital=${capital_allocation:.2f})"
        )

        return position_size

    def optimize_strategy_allocations(
        self,
        strategy_returns: pd.DataFrame,
        meta_label_confidences: Optional[pd.DataFrame] = None,
        current_drawdown: float = 0.0,
    ) -> Dict[str, float]:
        """
        Optimize capital allocation across multiple strategies using Kelly.

        Args:
            strategy_returns: Historical returns by strategy (time × strategies)
            meta_label_confidences: Meta-label confidences for each strategy
            current_drawdown: Current portfolio drawdown

        Returns:
            Dictionary of strategy names to Kelly fractions
        """
        logger.info(f"Optimizing Kelly allocations for {len(strategy_returns.columns)} strategies")

        allocations = {}

        for strategy in strategy_returns.columns:
            returns = strategy_returns[strategy].dropna()

            if len(returns) < 30:
                logger.warning(f"Insufficient data for {strategy}, using min Kelly")
                allocations[strategy] = self.min_kelly_fraction
                continue

            # Calculate base Kelly
            base_kelly = self.calculate_from_returns(returns)

            # Apply meta-label confidence if available
            if meta_label_confidences is not None and strategy in meta_label_confidences.columns:
                avg_confidence = meta_label_confidences[strategy].mean()
                base_kelly = self.confidence_weighted_kelly(base_kelly, avg_confidence)

            # Apply drawdown adjustment
            kelly_fraction = self.drawdown_adjusted_kelly(base_kelly, current_drawdown)

            allocations[strategy] = kelly_fraction

        # Normalize to ensure total doesn't exceed max_kelly_fraction
        total_kelly = sum(allocations.values())
        if total_kelly > self.max_kelly_fraction:
            scale_factor = self.max_kelly_fraction / total_kelly
            allocations = {k: v * scale_factor for k, v in allocations.items()}

        logger.info(
            f"Kelly allocations complete: total={sum(allocations.values()):.3f}, "
            f"strategies={len(allocations)}"
        )

        return allocations

    def calculate_kelly_metrics(
        self,
        returns: pd.Series,
        kelly_fraction: float,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a Kelly-sized strategy.

        Args:
            returns: Historical returns
            kelly_fraction: Kelly fraction used

        Returns:
            Dictionary of metrics including growth rate, risk, and drawdown
        """
        if len(returns) == 0:
            return {}

        # Scale returns by Kelly fraction
        kelly_returns = returns * kelly_fraction

        # Growth rate (geometric mean)
        growth_rate = (1 + kelly_returns).prod() ** (252 / len(kelly_returns)) - 1

        # Volatility
        volatility = kelly_returns.std() * np.sqrt(252)

        # Maximum drawdown
        cumulative = (1 + kelly_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio
        sharpe = growth_rate / volatility if volatility > 0 else 0

        metrics = {
            'kelly_fraction': kelly_fraction,
            'growth_rate': growth_rate,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).sum() / len(returns),
        }

        logger.debug(
            f"Kelly metrics: growth={growth_rate:.2%}, vol={volatility:.2%}, "
            f"sharpe={sharpe:.2f}, max_dd={max_drawdown:.2%}"
        )

        return metrics

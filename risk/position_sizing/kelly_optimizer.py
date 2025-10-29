"""
Kelly Criterion Position Sizing

Implements optimal position sizing using Kelly criterion:
- Full Kelly calculation from win rate and payoff
- Fractional Kelly (safety)
- Multi-asset Kelly optimization
- Confidence weighting
- Correlation adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class KellyResult:
    """Result of Kelly calculation"""
    symbol: str
    kelly_fraction: float
    fractional_kelly: float
    win_rate: float
    avg_win: float
    avg_loss: float
    position_size: float
    confidence: float = 1.0


class KellyOptimizer:
    """
    Kelly Criterion optimizer for position sizing

    Kelly% = (Win_Rate * Avg_Win - Loss_Rate * Avg_Loss) / Avg_Win
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_win_rate: float = 0.40,
        max_kelly: float = 0.50
    ):
        """
        Initialize Kelly optimizer

        Args:
            kelly_fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly)
            min_win_rate: Minimum win rate required (default 40%)
            max_kelly: Maximum Kelly fraction allowed (default 50%)
        """
        self.kelly_fraction = kelly_fraction
        self.min_win_rate = min_win_rate
        self.max_kelly = max_kelly

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate full Kelly criterion

        Kelly% = (p*W - (1-p)*L) / W
        where p = win rate, W = avg win, L = avg loss

        Args:
            win_rate: Probability of winning trade
            avg_win: Average win amount (as fraction)
            avg_loss: Average loss amount (as positive fraction)

        Returns:
            Kelly fraction (can be negative if edge is negative)
        """
        if win_rate < self.min_win_rate:
            return 0.0

        if avg_win == 0:
            return 0.0

        # Kelly formula
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Cap at max kelly
        if kelly > self.max_kelly:
            kelly = self.max_kelly

        # No shorting if negative edge
        if kelly < 0:
            kelly = 0.0

        return kelly

    def calculate_from_returns(
        self,
        returns: pd.Series
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Kelly from return series

        Args:
            returns: Return series

        Returns:
            Tuple of (kelly_fraction, win_rate, avg_win, avg_loss)
        """
        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        # Calculate win rate
        total_trades = len(returns)
        if total_trades == 0:
            return 0.0, 0.0, 0.0, 0.0

        win_rate = len(wins) / total_trades

        # Calculate average win/loss
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

        # Calculate Kelly
        kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)

        return kelly, win_rate, avg_win, avg_loss

    def calculate_position_size(
        self,
        returns: pd.Series,
        capital: float,
        price: float,
        confidence: float = 1.0
    ) -> KellyResult:
        """
        Calculate position size using Kelly criterion

        Args:
            returns: Return series
            capital: Available capital
            price: Current price
            confidence: Confidence in the edge (0-1)

        Returns:
            KellyResult with sizing details
        """
        # Calculate Kelly from returns
        kelly, win_rate, avg_win, avg_loss = self.calculate_from_returns(returns)

        # Apply fractional Kelly
        fractional_kelly = kelly * self.kelly_fraction

        # Adjust for confidence
        adjusted_kelly = fractional_kelly * confidence

        # Calculate position value
        position_value = capital * adjusted_kelly

        # Convert to position size
        position_size = position_value / price if price > 0 else 0.0

        return KellyResult(
            symbol="",  # Will be set by caller
            kelly_fraction=kelly,
            fractional_kelly=fractional_kelly,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            position_size=position_size,
            confidence=confidence
        )

    def calculate_with_meta_labels(
        self,
        returns: pd.Series,
        meta_label_confidence: pd.Series,
        capital: float,
        price: float
    ) -> KellyResult:
        """
        Calculate Kelly using meta-label confidence scores

        Args:
            returns: Return series
            meta_label_confidence: ML confidence scores (0-1)
            capital: Available capital
            price: Current price

        Returns:
            KellyResult
        """
        # Use average meta-label confidence as overall confidence
        avg_confidence = meta_label_confidence.mean()

        return self.calculate_position_size(
            returns=returns,
            capital=capital,
            price=price,
            confidence=avg_confidence
        )

    def multi_asset_kelly(
        self,
        returns_df: pd.DataFrame,
        capital: float,
        prices: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, KellyResult]:
        """
        Multi-asset Kelly optimization with correlation adjustment

        Args:
            returns_df: DataFrame of returns (columns = symbols)
            capital: Total capital
            prices: Dictionary of current prices
            correlation_matrix: Correlation matrix (if None, will be calculated)

        Returns:
            Dictionary of symbol -> KellyResult
        """
        # Calculate correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = returns_df.corr()

        results = {}

        # Calculate individual Kelly fractions
        individual_kellys = {}
        for symbol in returns_df.columns:
            kelly, win_rate, avg_win, avg_loss = self.calculate_from_returns(
                returns_df[symbol]
            )
            individual_kellys[symbol] = {
                'kelly': kelly * self.kelly_fraction,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }

        # Adjust for correlations (reduce size for highly correlated assets)
        symbols = list(returns_df.columns)
        for i, symbol in enumerate(symbols):
            # Calculate average correlation to other assets
            correlations = [
                abs(correlation_matrix.loc[symbol, other_symbol])
                for j, other_symbol in enumerate(symbols)
                if i != j
            ]
            avg_correlation = np.mean(correlations) if correlations else 0

            # Reduce Kelly for high correlation
            correlation_adjustment = 1.0 - (avg_correlation * 0.5)  # Max 50% reduction

            # Adjusted Kelly
            adjusted_kelly = individual_kellys[symbol]['kelly'] * correlation_adjustment

            # Calculate position
            position_value = capital * adjusted_kelly
            position_size = position_value / prices.get(symbol, 1) if prices.get(symbol, 0) > 0 else 0

            results[symbol] = KellyResult(
                symbol=symbol,
                kelly_fraction=individual_kellys[symbol]['kelly'],
                fractional_kelly=adjusted_kelly,
                win_rate=individual_kellys[symbol]['win_rate'],
                avg_win=individual_kellys[symbol]['avg_win'],
                avg_loss=individual_kellys[symbol]['avg_loss'],
                position_size=position_size,
                confidence=correlation_adjustment
            )

        return results

    def drawdown_adjusted_kelly(
        self,
        base_kelly: float,
        current_drawdown: float,
        max_drawdown: float = 0.20
    ) -> float:
        """
        Reduce Kelly during drawdowns

        Args:
            base_kelly: Base Kelly fraction
            current_drawdown: Current drawdown (positive number)
            max_drawdown: Maximum allowed drawdown

        Returns:
            Adjusted Kelly fraction
        """
        if current_drawdown <= 0:
            return base_kelly

        # Calculate drawdown ratio
        dd_ratio = current_drawdown / max_drawdown

        # Reduce Kelly proportionally
        if dd_ratio < 0.25:
            # Less than 25% of max DD: no reduction
            adjustment = 1.0
        elif dd_ratio < 0.50:
            # 25-50% of max DD: reduce by 25%
            adjustment = 0.75
        elif dd_ratio < 0.75:
            # 50-75% of max DD: reduce by 50%
            adjustment = 0.50
        else:
            # 75%+ of max DD: reduce by 75%
            adjustment = 0.25

        return base_kelly * adjustment

    def dynamic_kelly_from_regime(
        self,
        base_kelly: float,
        regime: str
    ) -> float:
        """
        Adjust Kelly based on market regime

        Args:
            base_kelly: Base Kelly fraction
            regime: Market regime ('trending', 'mean_reverting', 'volatile', 'calm')

        Returns:
            Regime-adjusted Kelly
        """
        regime_adjustments = {
            'trending': 1.2,        # Increase in trending markets
            'mean_reverting': 1.0,  # Normal in mean-reverting
            'volatile': 0.6,        # Reduce in volatile markets
            'calm': 1.1             # Slight increase in calm markets
        }

        adjustment = regime_adjustments.get(regime, 1.0)
        return base_kelly * adjustment

    def kelly_with_transaction_costs(
        self,
        base_kelly: float,
        avg_win: float,
        avg_loss: float,
        transaction_cost: float
    ) -> float:
        """
        Adjust Kelly for transaction costs

        Args:
            base_kelly: Base Kelly fraction
            avg_win: Average win
            avg_loss: Average loss
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 10bps)

        Returns:
            Cost-adjusted Kelly
        """
        # Reduce avg win and increase avg loss by transaction cost
        adj_avg_win = avg_win - transaction_cost
        adj_avg_loss = avg_loss + transaction_cost

        # Recalculate edge
        if adj_avg_win <= 0:
            return 0.0

        # Simple approximation: reduce Kelly proportionally
        cost_impact = transaction_cost / avg_win
        return base_kelly * (1 - cost_impact * 2)  # 2x to account for both entry and exit

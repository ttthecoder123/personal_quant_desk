"""
Dynamic Position Sizing

Adaptive position sizing based on:
- Market conditions
- Volatility regimes
- Performance metrics
- Correlation changes
- Liquidity constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime types"""
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"
    CRISIS = "crisis"


@dataclass
class DynamicSizingResult:
    """Result of dynamic sizing calculation"""
    symbol: str
    base_size: float
    regime_adjustment: float
    performance_adjustment: float
    correlation_adjustment: float
    liquidity_adjustment: float
    final_size: float


class DynamicSizing:
    """
    Dynamic position sizing that adapts to market conditions

    Combines multiple adjustment factors to determine optimal position size
    """

    def __init__(
        self,
        max_size_multiplier: float = 2.0,
        min_size_multiplier: float = 0.25
    ):
        """
        Initialize dynamic sizing

        Args:
            max_size_multiplier: Maximum size multiplier (default 2.0)
            min_size_multiplier: Minimum size multiplier (default 0.25)
        """
        self.max_size_multiplier = max_size_multiplier
        self.min_size_multiplier = min_size_multiplier

    def detect_volatility_regime(
        self,
        returns: pd.Series,
        lookback: int = 60
    ) -> MarketRegime:
        """
        Detect current volatility regime

        Args:
            returns: Return series
            lookback: Lookback period for regime detection

        Returns:
            MarketRegime
        """
        # Calculate rolling volatility
        recent_vol = returns.tail(lookback).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)

        # Compare recent to historical
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

        # Classify regime
        if vol_ratio < 0.7:
            return MarketRegime.CALM
        elif vol_ratio < 1.3:
            return MarketRegime.NORMAL
        elif vol_ratio < 2.0:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.CRISIS

    def regime_size_adjustment(
        self,
        regime: MarketRegime
    ) -> float:
        """
        Calculate size adjustment based on regime

        Args:
            regime: Current market regime

        Returns:
            Size multiplier
        """
        adjustments = {
            MarketRegime.CALM: 1.3,      # Increase size in calm markets
            MarketRegime.NORMAL: 1.0,    # Normal size
            MarketRegime.VOLATILE: 0.6,  # Reduce size in volatile markets
            MarketRegime.CRISIS: 0.3     # Significantly reduce in crisis
        }

        return adjustments.get(regime, 1.0)

    def performance_size_adjustment(
        self,
        recent_returns: pd.Series,
        lookback: int = 20,
        max_increase: float = 0.5
    ) -> float:
        """
        Adjust size based on recent performance (anti-martingale)

        Args:
            recent_returns: Recent return series
            lookback: Lookback period
            max_increase: Maximum size increase allowed

        Returns:
            Size multiplier
        """
        # Calculate cumulative return over lookback
        cum_return = (1 + recent_returns.tail(lookback)).prod() - 1

        # Positive returns -> slight increase (within limits)
        # Negative returns -> decrease
        if cum_return > 0:
            # Cap increase
            adjustment = 1.0 + min(cum_return * 0.5, max_increase)
        else:
            # Reduce proportionally
            adjustment = 1.0 + cum_return * 0.5  # Will be < 1.0

        # Apply bounds
        adjustment = np.clip(adjustment, self.min_size_multiplier, self.max_size_multiplier)

        return adjustment

    def correlation_size_adjustment(
        self,
        returns_df: pd.DataFrame,
        symbol: str,
        max_correlation: float = 0.85
    ) -> float:
        """
        Adjust size based on correlation with other positions

        High correlation = reduce size

        Args:
            returns_df: DataFrame of all position returns
            symbol: Symbol to calculate adjustment for
            max_correlation: Maximum acceptable correlation

        Returns:
            Size multiplier (< 1.0 if high correlation)
        """
        if symbol not in returns_df.columns:
            return 1.0

        # Calculate correlations with other positions
        correlations = []
        for col in returns_df.columns:
            if col != symbol:
                corr = returns_df[symbol].corr(returns_df[col])
                correlations.append(abs(corr))

        if not correlations:
            return 1.0

        # Average absolute correlation
        avg_corr = np.mean(correlations)

        # Reduce size if correlation too high
        if avg_corr > max_correlation:
            # Significant reduction for very high correlation
            adjustment = 0.5
        elif avg_corr > max_correlation * 0.7:
            # Moderate reduction
            adjustment = 0.75
        else:
            # Normal size
            adjustment = 1.0

        return adjustment

    def liquidity_size_adjustment(
        self,
        volume: float,
        avg_volume: float,
        position_value: float,
        max_volume_pct: float = 0.10
    ) -> float:
        """
        Adjust size based on liquidity

        Args:
            volume: Current volume
            avg_volume: Average volume
            position_value: Proposed position value
            max_volume_pct: Max % of volume to trade

        Returns:
            Size multiplier
        """
        # Calculate volume ratio
        vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Reduce size if volume is low
        if vol_ratio < 0.5:
            liquidity_adj = 0.5  # Significantly reduce
        elif vol_ratio < 0.8:
            liquidity_adj = 0.75  # Moderately reduce
        else:
            liquidity_adj = 1.0  # Normal

        # Also check if position is too large relative to volume
        max_position_value = avg_volume * max_volume_pct
        if position_value > max_position_value:
            size_constraint = max_position_value / position_value
            liquidity_adj = min(liquidity_adj, size_constraint)

        return liquidity_adj

    def calculate_dynamic_size(
        self,
        base_size: float,
        returns: pd.Series,
        all_returns: pd.DataFrame,
        symbol: str,
        volume: float,
        avg_volume: float,
        position_value: float
    ) -> DynamicSizingResult:
        """
        Calculate final position size with all adjustments

        Args:
            base_size: Base position size (from vol targeting or Kelly)
            returns: Return series for the symbol
            all_returns: DataFrame of all position returns
            symbol: Symbol name
            volume: Current volume
            avg_volume: Average volume
            position_value: Proposed position value

        Returns:
            DynamicSizingResult with all adjustments
        """
        # Detect regime
        regime = self.detect_volatility_regime(returns)

        # Calculate all adjustments
        regime_adj = self.regime_size_adjustment(regime)
        performance_adj = self.performance_size_adjustment(returns)
        correlation_adj = self.correlation_size_adjustment(all_returns, symbol)
        liquidity_adj = self.liquidity_size_adjustment(
            volume, avg_volume, position_value
        )

        # Combined adjustment
        total_adjustment = (
            regime_adj *
            performance_adj *
            correlation_adj *
            liquidity_adj
        )

        # Apply bounds
        total_adjustment = np.clip(
            total_adjustment,
            self.min_size_multiplier,
            self.max_size_multiplier
        )

        # Final size
        final_size = base_size * total_adjustment

        return DynamicSizingResult(
            symbol=symbol,
            base_size=base_size,
            regime_adjustment=regime_adj,
            performance_adjustment=performance_adj,
            correlation_adjustment=correlation_adj,
            liquidity_adjustment=liquidity_adj,
            final_size=final_size
        )

    def emergency_size_reduction(
        self,
        current_size: float,
        reduction_level: int
    ) -> float:
        """
        Emergency position size reduction

        Args:
            current_size: Current position size
            reduction_level: 1, 2, or 3 (increasing severity)

        Returns:
            Reduced position size
        """
        reductions = {
            1: 0.75,  # Level 1: Reduce by 25%
            2: 0.50,  # Level 2: Reduce by 50%
            3: 0.00   # Level 3: Flatten
        }

        multiplier = reductions.get(reduction_level, 1.0)
        return current_size * multiplier

    def anti_martingale_sizing(
        self,
        base_size: float,
        winning_streak: int,
        losing_streak: int,
        max_increase: float = 0.5
    ) -> float:
        """
        Anti-martingale: increase after wins, decrease after losses

        Args:
            base_size: Base position size
            winning_streak: Number of consecutive wins
            losing_streak: Number of consecutive losses
            max_increase: Maximum size increase

        Returns:
            Adjusted position size
        """
        if winning_streak > 0:
            # Increase size after wins (capped)
            increase = min(winning_streak * 0.1, max_increase)
            return base_size * (1 + increase)
        elif losing_streak > 0:
            # Decrease size after losses
            decrease = min(losing_streak * 0.15, 0.5)
            return base_size * (1 - decrease)
        else:
            return base_size

    def drawdown_scaling(
        self,
        base_size: float,
        current_drawdown: float,
        max_drawdown: float = 0.20
    ) -> float:
        """
        Scale position size based on drawdown

        Args:
            base_size: Base position size
            current_drawdown: Current drawdown (positive number)
            max_drawdown: Maximum allowed drawdown

        Returns:
            Scaled position size
        """
        if current_drawdown <= 0:
            return base_size

        # Calculate drawdown ratio
        dd_ratio = current_drawdown / max_drawdown

        # Progressive scaling down
        if dd_ratio < 0.25:
            # Less than 25% of max: no scaling
            scale = 1.0
        elif dd_ratio < 0.50:
            # 25-50%: reduce by 20%
            scale = 0.80
        elif dd_ratio < 0.75:
            # 50-75%: reduce by 50%
            scale = 0.50
        else:
            # 75%+: reduce by 75%
            scale = 0.25

        return base_size * scale

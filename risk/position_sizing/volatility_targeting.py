"""
Volatility Targeting Position Sizing

Implements Carver's volatility targeting approach:
- EWMA volatility estimation (25-day halflife)
- Position scaling based on target volatility
- Position inertia to reduce turnover
- Leverage constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class VolatilityTargetResult:
    """Result of volatility targeting calculation"""
    symbol: str
    current_volatility: float
    target_volatility: float
    volatility_scalar: float
    position_size: float
    leverage_used: float


class VolatilityTargeting:
    """
    Carver-style volatility targeting for position sizing

    Position = (Target_Vol / Instrument_Vol) * Capital
    """

    def __init__(
        self,
        target_volatility: float = 0.20,
        ewma_halflife: int = 25,
        max_leverage: float = 2.0,
        inertia_threshold: float = 0.10
    ):
        """
        Initialize volatility targeting

        Args:
            target_volatility: Annual target volatility (default 20%)
            ewma_halflife: Halflife for EWMA volatility (default 25 days)
            max_leverage: Maximum leverage allowed (default 2x)
            inertia_threshold: Minimum change to trigger rebalance (default 10%)
        """
        self.target_volatility = target_volatility
        self.ewma_halflife = ewma_halflife
        self.max_leverage = max_leverage
        self.inertia_threshold = inertia_threshold

    def calculate_ewma_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Calculate EWMA volatility

        Args:
            returns: Return series
            annualize: Whether to annualize (default True)

        Returns:
            EWMA volatility
        """
        # Calculate EWMA variance
        ewma_var = returns.ewm(halflife=self.ewma_halflife).var()

        # Get most recent variance
        current_var = ewma_var.iloc[-1]

        # Calculate volatility
        vol = np.sqrt(current_var)

        # Annualize if requested (assuming daily returns)
        if annualize:
            vol *= np.sqrt(252)

        return vol

    def calculate_volatility_scalar(
        self,
        instrument_volatility: float,
        target_volatility: Optional[float] = None
    ) -> float:
        """
        Calculate volatility scalar

        Scalar = Target_Vol / Instrument_Vol

        Args:
            instrument_volatility: Current instrument volatility
            target_volatility: Override target volatility

        Returns:
            Volatility scalar
        """
        target_vol = target_volatility if target_volatility else self.target_volatility

        if instrument_volatility == 0:
            return 0.0

        scalar = target_vol / instrument_volatility

        # Cap at max leverage
        if scalar > self.max_leverage:
            scalar = self.max_leverage

        return scalar

    def calculate_position_size(
        self,
        returns: pd.Series,
        capital: float,
        price: float,
        target_volatility: Optional[float] = None
    ) -> VolatilityTargetResult:
        """
        Calculate position size using volatility targeting

        Args:
            returns: Return series for the instrument
            capital: Available capital
            price: Current price
            target_volatility: Override target volatility

        Returns:
            VolatilityTargetResult with sizing details
        """
        # Calculate instrument volatility
        instrument_vol = self.calculate_ewma_volatility(returns)

        # Calculate volatility scalar
        vol_scalar = self.calculate_volatility_scalar(
            instrument_vol,
            target_volatility
        )

        # Calculate position value
        position_value = vol_scalar * capital

        # Convert to position size (number of units)
        if price == 0:
            position_size = 0.0
        else:
            position_size = position_value / price

        # Calculate leverage used
        leverage_used = position_value / capital if capital > 0 else 0.0

        return VolatilityTargetResult(
            symbol="",  # Will be set by caller
            current_volatility=instrument_vol,
            target_volatility=target_volatility or self.target_volatility,
            volatility_scalar=vol_scalar,
            position_size=position_size,
            leverage_used=leverage_used
        )

    def apply_inertia(
        self,
        current_position: float,
        target_position: float
    ) -> float:
        """
        Apply position inertia to reduce turnover

        Only change position if the change exceeds threshold

        Args:
            current_position: Current position size
            target_position: Target position size from volatility targeting

        Returns:
            Adjusted target position
        """
        if current_position == 0:
            return target_position

        # Calculate percentage change
        pct_change = abs(target_position - current_position) / abs(current_position)

        # Only change if exceeds threshold
        if pct_change < self.inertia_threshold:
            return current_position
        else:
            return target_position

    def calculate_multi_instrument_positions(
        self,
        returns_dict: Dict[str, pd.Series],
        capital: float,
        prices: Dict[str, float],
        current_positions: Optional[Dict[str, float]] = None,
        target_volatility: Optional[float] = None
    ) -> Dict[str, VolatilityTargetResult]:
        """
        Calculate positions for multiple instruments

        Args:
            returns_dict: Dictionary of symbol -> return series
            capital: Total available capital
            prices: Dictionary of symbol -> current price
            current_positions: Current positions (for inertia)
            target_volatility: Override target volatility

        Returns:
            Dictionary of symbol -> VolatilityTargetResult
        """
        current_pos = current_positions if current_positions else {}
        results = {}

        # Calculate equal capital allocation per instrument
        capital_per_instrument = capital / len(returns_dict)

        for symbol, returns in returns_dict.items():
            # Calculate position size
            result = self.calculate_position_size(
                returns=returns,
                capital=capital_per_instrument,
                price=prices.get(symbol, 0),
                target_volatility=target_volatility
            )

            # Apply inertia if current position exists
            if symbol in current_pos:
                result.position_size = self.apply_inertia(
                    current_pos[symbol],
                    result.position_size
                )

            result.symbol = symbol
            results[symbol] = result

        return results

    def adjust_for_regime(
        self,
        position_size: float,
        volatility_regime: str
    ) -> float:
        """
        Adjust position size based on volatility regime

        Args:
            position_size: Base position size
            volatility_regime: 'low', 'medium', or 'high'

        Returns:
            Adjusted position size
        """
        regime_scalars = {
            'low': 1.2,      # Increase size in low vol
            'medium': 1.0,   # Normal size
            'high': 0.7      # Reduce size in high vol
        }

        scalar = regime_scalars.get(volatility_regime, 1.0)
        return position_size * scalar

    def scale_with_equity(
        self,
        base_position: float,
        initial_capital: float,
        current_capital: float,
        max_scale: float = 1.5
    ) -> float:
        """
        Scale position with equity changes

        Args:
            base_position: Base position size
            initial_capital: Initial capital
            current_capital: Current capital
            max_scale: Maximum scaling factor

        Returns:
            Scaled position size
        """
        if initial_capital == 0:
            return base_position

        # Calculate equity growth
        equity_ratio = current_capital / initial_capital

        # Cap scaling
        scale_factor = min(equity_ratio, max_scale)

        return base_position * scale_factor

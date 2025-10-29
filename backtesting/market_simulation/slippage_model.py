"""
Slippage Model

Implements realistic slippage estimation for backtesting:
- Fixed slippage (basis points)
- Dynamic slippage based on market conditions
- Volume-based slippage
- Volatility-based slippage
- Time-of-day effects
- Adverse selection modeling

Slippage represents the difference between expected execution price
and actual execution price due to market microstructure effects.
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional
import numpy as np
from loguru import logger


class SlippageType(Enum):
    """Slippage model type."""
    FIXED = "FIXED"
    DYNAMIC = "DYNAMIC"
    VOLUME_BASED = "VOLUME_BASED"
    VOLATILITY_BASED = "VOLATILITY_BASED"


@dataclass
class SlippageParameters:
    """
    Slippage model parameters.

    Attributes:
        fixed_bps: Fixed slippage in basis points
        min_bps: Minimum slippage (basis points)
        max_bps: Maximum slippage (basis points)
        volume_factor: Scaling factor for volume-based slippage
        volatility_factor: Scaling factor for volatility-based slippage
        adverse_selection_factor: Adverse selection component
        time_of_day_adjustment: Whether to adjust for time of day
    """
    fixed_bps: float = 5.0  # 5 basis points
    min_bps: float = 1.0
    max_bps: float = 50.0
    volume_factor: float = 0.5
    volatility_factor: float = 1.0
    adverse_selection_factor: float = 0.1
    time_of_day_adjustment: bool = True


class SlippageModel:
    """
    Fixed slippage model.

    Applies constant slippage in basis points, useful for:
    - Simple backtests
    - Conservative cost estimates
    - Baseline comparisons
    """

    def __init__(self, slippage_bps: float = 5.0):
        """
        Initialize fixed slippage model.

        Args:
            slippage_bps: Slippage in basis points (default: 5.0 bps)
        """
        self.slippage_bps = slippage_bps
        logger.info(f"Initialized SlippageModel with {slippage_bps} bps")

    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate fixed slippage.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            price: Reference price
            **kwargs: Additional parameters (ignored for fixed model)

        Returns:
            Dictionary with slippage metrics
        """
        # Fixed slippage as percentage of price
        slippage_pct = self.slippage_bps / 10000.0
        slippage_per_share = price * slippage_pct

        # Total slippage cost
        total_slippage = slippage_per_share * quantity

        # Adverse for buys, favorable for sells (in terms of cost)
        if side.upper() == 'BUY':
            execution_price = price + slippage_per_share
        else:
            execution_price = price - slippage_per_share

        return {
            'slippage_bps': self.slippage_bps,
            'slippage_per_share': slippage_per_share,
            'total_slippage': total_slippage,
            'execution_price': execution_price,
            'reference_price': price
        }


class DynamicSlippageModel:
    """
    Dynamic slippage model based on market conditions.

    Calculates slippage based on:
    - Trade size relative to volume (participation rate)
    - Market volatility
    - Time of day
    - Liquidity conditions
    - Adverse selection
    """

    def __init__(self, params: Optional[SlippageParameters] = None):
        """
        Initialize dynamic slippage model.

        Args:
            params: Slippage parameters
        """
        self.params = params or SlippageParameters()
        logger.info("Initialized DynamicSlippageModel")

    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        volume: Optional[float] = None,
        adv: Optional[float] = None,  # Average Daily Volume
        volatility: Optional[float] = None,
        spread: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        market_state: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate dynamic slippage based on market conditions.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            price: Reference price
            volume: Current period volume
            adv: Average daily volume
            volatility: Historical volatility
            spread: Current bid-ask spread
            timestamp: Trade timestamp
            market_state: Additional market state information
            **kwargs: Additional parameters

        Returns:
            Dictionary with slippage metrics
        """
        # Base slippage
        base_slippage_bps = self.params.fixed_bps

        # Volume-based component
        volume_component = self._calculate_volume_component(quantity, volume, adv)

        # Volatility-based component
        volatility_component = self._calculate_volatility_component(volatility)

        # Spread component
        spread_component = self._calculate_spread_component(spread, price)

        # Time of day adjustment
        time_multiplier = 1.0
        if timestamp and self.params.time_of_day_adjustment:
            time_multiplier = self._get_time_multiplier(timestamp)

        # Adverse selection component
        adverse_selection = self._calculate_adverse_selection(
            quantity, volume, adv, market_state
        )

        # Combine components
        total_slippage_bps = (
            base_slippage_bps +
            volume_component +
            volatility_component +
            spread_component +
            adverse_selection
        ) * time_multiplier

        # Apply bounds
        total_slippage_bps = np.clip(
            total_slippage_bps,
            self.params.min_bps,
            self.params.max_bps
        )

        # Convert to price
        slippage_pct = total_slippage_bps / 10000.0
        slippage_per_share = price * slippage_pct
        total_slippage = slippage_per_share * quantity

        # Execution price
        if side.upper() == 'BUY':
            execution_price = price + slippage_per_share
        else:
            execution_price = price - slippage_per_share

        result = {
            'slippage_bps': total_slippage_bps,
            'slippage_per_share': slippage_per_share,
            'total_slippage': total_slippage,
            'execution_price': execution_price,
            'reference_price': price,
            'components': {
                'base': base_slippage_bps,
                'volume': volume_component,
                'volatility': volatility_component,
                'spread': spread_component,
                'adverse_selection': adverse_selection,
                'time_multiplier': time_multiplier
            }
        }

        logger.debug(
            f"Slippage for {symbol}: {total_slippage_bps:.2f} bps "
            f"({quantity} shares @ ${price:.2f})"
        )

        return result

    def _calculate_volume_component(
        self,
        quantity: float,
        volume: Optional[float],
        adv: Optional[float]
    ) -> float:
        """
        Calculate volume-based slippage component.

        Higher participation rate → higher slippage

        Args:
            quantity: Trade quantity
            volume: Current period volume
            adv: Average daily volume

        Returns:
            Slippage component in basis points
        """
        if adv is None or adv <= 0:
            # Use current volume as proxy if ADV not available
            if volume is None or volume <= 0:
                return 0.0
            adv = volume * 10  # Assume current volume is ~10% of daily

        participation_rate = quantity / adv

        # Non-linear relationship: slippage increases faster for higher participation
        volume_slippage = self.params.volume_factor * (participation_rate ** 0.75) * 100

        return volume_slippage

    def _calculate_volatility_component(self, volatility: Optional[float]) -> float:
        """
        Calculate volatility-based slippage component.

        Higher volatility → wider spreads → higher slippage

        Args:
            volatility: Historical volatility (e.g., daily std dev)

        Returns:
            Slippage component in basis points
        """
        if volatility is None or volatility <= 0:
            return 0.0

        # Convert volatility to basis points contribution
        # Typical daily vol of 1-2% → 1-2 bps additional slippage
        volatility_slippage = self.params.volatility_factor * volatility * 100

        return volatility_slippage

    def _calculate_spread_component(
        self,
        spread: Optional[float],
        price: float
    ) -> float:
        """
        Calculate spread-based slippage component.

        Wider spread → higher slippage (pay spread crossing)

        Args:
            spread: Current bid-ask spread
            price: Reference price

        Returns:
            Slippage component in basis points
        """
        if spread is None or spread <= 0 or price <= 0:
            return 0.0

        # Spread in basis points
        spread_bps = (spread / price) * 10000

        # Assume we pay half the spread on average
        spread_component = spread_bps * 0.5

        return spread_component

    def _get_time_multiplier(self, timestamp: datetime) -> float:
        """
        Get time-of-day slippage multiplier.

        Slippage is typically higher at:
        - Market open (9:30-10:00): High volatility, wider spreads
        - Market close (15:30-16:00): Rebalancing flows, higher impact
        - Lunch (12:00-13:00): Lower liquidity

        Args:
            timestamp: Trade timestamp

        Returns:
            Multiplier (typically 0.8 - 1.5)
        """
        trade_time = timestamp.time()

        # Market open (high slippage)
        if time(9, 30) <= trade_time < time(10, 0):
            return 1.5
        # Early morning (elevated)
        elif time(10, 0) <= trade_time < time(10, 30):
            return 1.2
        # Lunch hour (reduced liquidity)
        elif time(12, 0) <= trade_time < time(13, 0):
            return 1.15
        # Approaching close (increasing)
        elif time(15, 0) <= trade_time < time(15, 30):
            return 1.25
        # Market close (highest slippage)
        elif time(15, 30) <= trade_time <= time(16, 0):
            return 1.5
        # Normal trading hours (baseline)
        else:
            return 1.0

    def _calculate_adverse_selection(
        self,
        quantity: float,
        volume: Optional[float],
        adv: Optional[float],
        market_state: Optional[Dict]
    ) -> float:
        """
        Calculate adverse selection component.

        Adverse selection occurs when informed traders move the market
        before our order completes. More significant for:
        - Larger orders
        - Lower liquidity
        - Trending markets

        Args:
            quantity: Trade quantity
            volume: Current volume
            adv: Average daily volume
            market_state: Market state (trend, momentum, etc.)

        Returns:
            Adverse selection in basis points
        """
        base_adverse = self.params.adverse_selection_factor

        # Size effect
        if adv and adv > 0:
            participation = quantity / adv
            size_multiplier = 1 + participation * 5  # Larger trades → more adverse selection
        else:
            size_multiplier = 1.0

        # Market state effect
        trend_multiplier = 1.0
        if market_state:
            # Higher adverse selection in trending markets
            if 'trend_strength' in market_state:
                trend_strength = abs(market_state['trend_strength'])
                trend_multiplier = 1 + trend_strength * 0.5

        adverse_selection_bps = base_adverse * size_multiplier * trend_multiplier

        return adverse_selection_bps

    def update_parameters(self, **kwargs):
        """
        Update model parameters.

        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
                logger.debug(f"Updated {key} to {value}")


class VolumeBasedSlippageModel(DynamicSlippageModel):
    """
    Slippage model focused on volume/participation rate.

    Simplified model where slippage is primarily a function of
    trade size relative to market volume.
    """

    def __init__(self, volume_factor: float = 0.5):
        """
        Initialize volume-based slippage model.

        Args:
            volume_factor: Scaling factor for volume impact
        """
        params = SlippageParameters(
            fixed_bps=2.0,
            volume_factor=volume_factor,
            volatility_factor=0.0,  # Ignore volatility
            adverse_selection_factor=0.05,
            time_of_day_adjustment=False  # Ignore time effects
        )
        super().__init__(params)
        logger.info(f"Initialized VolumeBasedSlippageModel with factor={volume_factor}")


class VolatilityBasedSlippageModel(DynamicSlippageModel):
    """
    Slippage model focused on volatility.

    Slippage scales primarily with market volatility,
    useful for markets where spread is the dominant cost.
    """

    def __init__(self, volatility_factor: float = 2.0):
        """
        Initialize volatility-based slippage model.

        Args:
            volatility_factor: Scaling factor for volatility impact
        """
        params = SlippageParameters(
            fixed_bps=1.0,
            volume_factor=0.1,  # Minimal volume effect
            volatility_factor=volatility_factor,
            adverse_selection_factor=0.05,
            time_of_day_adjustment=True
        )
        super().__init__(params)
        logger.info(f"Initialized VolatilityBasedSlippageModel with factor={volatility_factor}")


class AdaptiveSlippageModel:
    """
    Adaptive slippage model that learns from execution history.

    Adjusts slippage estimates based on realized slippage from past trades.
    """

    def __init__(self, base_model: Optional[DynamicSlippageModel] = None, learning_rate: float = 0.1):
        """
        Initialize adaptive slippage model.

        Args:
            base_model: Base slippage model to adapt
            learning_rate: Learning rate for parameter updates
        """
        self.base_model = base_model or DynamicSlippageModel()
        self.learning_rate = learning_rate
        self.execution_history: list = []
        self.adaptation_factor = 1.0

        logger.info("Initialized AdaptiveSlippageModel")

    def calculate_slippage(self, **kwargs) -> Dict[str, float]:
        """
        Calculate slippage with adaptation.

        Args:
            **kwargs: Same as DynamicSlippageModel.calculate_slippage

        Returns:
            Slippage metrics
        """
        # Get base slippage
        result = self.base_model.calculate_slippage(**kwargs)

        # Apply adaptation factor
        result['slippage_bps'] *= self.adaptation_factor
        result['slippage_per_share'] *= self.adaptation_factor
        result['total_slippage'] *= self.adaptation_factor

        # Recalculate execution price
        price = kwargs.get('price', 0)
        side = kwargs.get('side', 'BUY')
        slippage_per_share = result['slippage_per_share']

        if side.upper() == 'BUY':
            result['execution_price'] = price + slippage_per_share
        else:
            result['execution_price'] = price - slippage_per_share

        return result

    def record_execution(
        self,
        predicted_slippage_bps: float,
        actual_slippage_bps: float,
        **metadata
    ):
        """
        Record execution and update adaptation.

        Args:
            predicted_slippage_bps: Predicted slippage
            actual_slippage_bps: Actual realized slippage
            **metadata: Additional execution metadata
        """
        self.execution_history.append({
            'predicted': predicted_slippage_bps,
            'actual': actual_slippage_bps,
            'error': actual_slippage_bps - predicted_slippage_bps,
            **metadata
        })

        # Update adaptation factor using exponential moving average
        error_ratio = actual_slippage_bps / predicted_slippage_bps if predicted_slippage_bps > 0 else 1.0
        self.adaptation_factor = (
            (1 - self.learning_rate) * self.adaptation_factor +
            self.learning_rate * error_ratio
        )

        logger.debug(
            f"Updated adaptation factor to {self.adaptation_factor:.3f} "
            f"(error: {actual_slippage_bps - predicted_slippage_bps:.2f} bps)"
        )

    def get_statistics(self) -> Dict[str, float]:
        """
        Get execution statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.execution_history:
            return {}

        errors = [e['error'] for e in self.execution_history]

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_absolute_error': np.mean(np.abs(errors)),
            'adaptation_factor': self.adaptation_factor,
            'num_executions': len(self.execution_history)
        }

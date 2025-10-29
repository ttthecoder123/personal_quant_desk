"""
Spread Models

Implements bid-ask spread cost models for backtesting:
- Fixed spread costs (in ticks or basis points)
- Time-varying spreads (higher at open/close)
- Volume-dependent spreads
- Volatility-based spreads
- Cross-asset spread correlation
- Effective spread calculation
- Implementation shortfall

Spread costs represent the cost of crossing the bid-ask spread,
a significant component of transaction costs especially for less liquid securities.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional, List, Callable
import numpy as np
import pandas as pd
from loguru import logger


class SpreadType(Enum):
    """Type of spread model."""
    FIXED = "FIXED"
    TIME_VARYING = "TIME_VARYING"
    VOLUME_DEPENDENT = "VOLUME_DEPENDENT"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    DYNAMIC = "DYNAMIC"


@dataclass
class SpreadResult:
    """
    Result of spread calculation.

    Attributes:
        quoted_spread: Quoted bid-ask spread
        effective_spread: Effective spread actually paid
        spread_cost: Total cost of crossing spread
        execution_price: Final execution price including spread
        reference_price: Reference price (mid-market)
        spread_bps: Spread in basis points
        spread_breakdown: Breakdown of spread components
    """
    quoted_spread: float
    effective_spread: float
    spread_cost: float
    execution_price: float
    reference_price: float
    spread_bps: float
    spread_breakdown: Dict[str, float] = field(default_factory=dict)


class FixedSpreadModel:
    """
    Fixed spread model.

    Constant bid-ask spread in basis points or ticks.
    Simplest model, good for liquid assets or baseline estimates.
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        min_spread_bps: float = 1.0,
        max_spread_bps: float = 100.0
    ):
        """
        Initialize fixed spread model.

        Args:
            spread_bps: Fixed spread in basis points
            min_spread_bps: Minimum spread
            max_spread_bps: Maximum spread
        """
        self.spread_bps = spread_bps
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps

        logger.info(f"Initialized FixedSpreadModel: {spread_bps} bps")

    def calculate_spread(
        self,
        symbol: str,
        side: str,
        quantity: float,
        mid_price: float,
        **kwargs
    ) -> SpreadResult:
        """
        Calculate fixed spread cost.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            mid_price: Mid-market price
            **kwargs: Additional parameters (ignored)

        Returns:
            SpreadResult object
        """
        spread_bps = np.clip(self.spread_bps, self.min_spread_bps, self.max_spread_bps)

        # Convert to price
        spread_pct = spread_bps / 10000.0
        half_spread = mid_price * spread_pct / 2

        # Execution price
        if side.upper() == 'BUY':
            execution_price = mid_price + half_spread
        else:
            execution_price = mid_price - half_spread

        # Total cost
        spread_cost = half_spread * quantity

        return SpreadResult(
            quoted_spread=mid_price * spread_pct,
            effective_spread=mid_price * spread_pct,
            spread_cost=spread_cost,
            execution_price=execution_price,
            reference_price=mid_price,
            spread_bps=spread_bps
        )


class TimeVaryingSpreadModel:
    """
    Time-varying spread model.

    Spreads vary by time of day:
    - Higher at market open (9:30-10:00)
    - Higher at market close (15:30-16:00)
    - Lower during mid-day
    - Higher during lunch hour (reduced liquidity)

    Based on empirical intraday patterns.
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        open_multiplier: float = 2.0,
        close_multiplier: float = 2.5,
        lunch_multiplier: float = 1.3,
        overnight_multiplier: float = 3.0
    ):
        """
        Initialize time-varying spread model.

        Args:
            base_spread_bps: Base spread during normal hours
            open_multiplier: Multiplier at market open
            close_multiplier: Multiplier at market close
            lunch_multiplier: Multiplier during lunch
            overnight_multiplier: Multiplier for overnight periods
        """
        self.base_spread_bps = base_spread_bps
        self.open_multiplier = open_multiplier
        self.close_multiplier = close_multiplier
        self.lunch_multiplier = lunch_multiplier
        self.overnight_multiplier = overnight_multiplier

        logger.info(
            f"Initialized TimeVaryingSpreadModel: "
            f"base={base_spread_bps} bps"
        )

    def get_time_multiplier(self, timestamp: datetime) -> float:
        """
        Get spread multiplier based on time of day.

        Args:
            timestamp: Trade timestamp

        Returns:
            Spread multiplier
        """
        trade_time = timestamp.time()

        # Market hours: 9:30 - 16:00 ET
        market_open = time(9, 30)
        market_close = time(16, 0)

        # Outside market hours
        if trade_time < market_open or trade_time > market_close:
            return self.overnight_multiplier

        # Market open: 9:30-10:00 (high spread)
        if time(9, 30) <= trade_time < time(10, 0):
            return self.open_multiplier

        # Early morning: 10:00-10:30 (elevated)
        if time(10, 0) <= trade_time < time(10, 30):
            return 1 + (self.open_multiplier - 1) * 0.5

        # Lunch hour: 12:00-13:00 (reduced liquidity)
        if time(12, 0) <= trade_time < time(13, 0):
            return self.lunch_multiplier

        # Approaching close: 15:00-15:30
        if time(15, 0) <= trade_time < time(15, 30):
            return 1 + (self.close_multiplier - 1) * 0.5

        # Market close: 15:30-16:00 (highest spread)
        if time(15, 30) <= trade_time <= time(16, 0):
            return self.close_multiplier

        # Normal trading hours
        return 1.0

    def calculate_spread(
        self,
        symbol: str,
        side: str,
        quantity: float,
        mid_price: float,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> SpreadResult:
        """
        Calculate time-varying spread cost.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            mid_price: Mid-market price
            timestamp: Trade timestamp
            **kwargs: Additional parameters

        Returns:
            SpreadResult object
        """
        # Get time multiplier
        if timestamp:
            time_multiplier = self.get_time_multiplier(timestamp)
        else:
            time_multiplier = 1.0

        # Adjusted spread
        adjusted_spread_bps = self.base_spread_bps * time_multiplier

        # Convert to price
        spread_pct = adjusted_spread_bps / 10000.0
        half_spread = mid_price * spread_pct / 2

        # Execution price
        if side.upper() == 'BUY':
            execution_price = mid_price + half_spread
        else:
            execution_price = mid_price - half_spread

        # Total cost
        spread_cost = half_spread * quantity

        return SpreadResult(
            quoted_spread=mid_price * spread_pct,
            effective_spread=mid_price * spread_pct,
            spread_cost=spread_cost,
            execution_price=execution_price,
            reference_price=mid_price,
            spread_bps=adjusted_spread_bps,
            spread_breakdown={
                'base_spread_bps': self.base_spread_bps,
                'time_multiplier': time_multiplier,
                'adjusted_spread_bps': adjusted_spread_bps
            }
        )


class VolumeDependentSpreadModel:
    """
    Volume-dependent spread model.

    Spread increases with trade size relative to available liquidity.
    Larger trades move through multiple price levels, paying wider spreads.

    Models order book depth and price impact.
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        volume_sensitivity: float = 0.5,
        participation_threshold: float = 0.01  # 1% of daily volume
    ):
        """
        Initialize volume-dependent spread model.

        Args:
            base_spread_bps: Base spread for small trades
            volume_sensitivity: How much spread widens with size
            participation_threshold: Threshold for size penalty
        """
        self.base_spread_bps = base_spread_bps
        self.volume_sensitivity = volume_sensitivity
        self.participation_threshold = participation_threshold

        logger.info(
            f"Initialized VolumeDependentSpreadModel: "
            f"base={base_spread_bps} bps, sensitivity={volume_sensitivity}"
        )

    def calculate_spread(
        self,
        symbol: str,
        side: str,
        quantity: float,
        mid_price: float,
        adv: Optional[float] = None,  # Average daily volume
        current_volume: Optional[float] = None,
        **kwargs
    ) -> SpreadResult:
        """
        Calculate volume-dependent spread cost.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            mid_price: Mid-market price
            adv: Average daily volume
            current_volume: Current period volume
            **kwargs: Additional parameters

        Returns:
            SpreadResult object
        """
        # Estimate participation rate
        if adv and adv > 0:
            participation_rate = quantity / adv
        elif current_volume and current_volume > 0:
            # Assume current volume is ~10% of daily
            estimated_adv = current_volume * 10
            participation_rate = quantity / estimated_adv
        else:
            # Default to small trade
            participation_rate = 0.001

        # Volume penalty: non-linear relationship
        # Small trades pay base spread
        # Large trades pay increasingly wider spreads
        if participation_rate <= self.participation_threshold:
            volume_multiplier = 1.0
        else:
            excess_participation = (
                (participation_rate - self.participation_threshold) /
                self.participation_threshold
            )
            volume_multiplier = 1.0 + self.volume_sensitivity * (excess_participation ** 0.6)

        # Adjusted spread
        adjusted_spread_bps = self.base_spread_bps * volume_multiplier

        # Convert to price
        spread_pct = adjusted_spread_bps / 10000.0
        half_spread = mid_price * spread_pct / 2

        # Execution price
        if side.upper() == 'BUY':
            execution_price = mid_price + half_spread
        else:
            execution_price = mid_price - half_spread

        # Total cost
        spread_cost = half_spread * quantity

        return SpreadResult(
            quoted_spread=mid_price * spread_pct,
            effective_spread=mid_price * spread_pct,
            spread_cost=spread_cost,
            execution_price=execution_price,
            reference_price=mid_price,
            spread_bps=adjusted_spread_bps,
            spread_breakdown={
                'base_spread_bps': self.base_spread_bps,
                'participation_rate': participation_rate,
                'volume_multiplier': volume_multiplier,
                'adjusted_spread_bps': adjusted_spread_bps
            }
        )


class VolatilityBasedSpreadModel:
    """
    Volatility-based spread model.

    Spreads widen during high volatility as market makers increase quotes
    to protect against adverse selection and inventory risk.

    Empirically, spread scales roughly proportional to volatility.
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        volatility_sensitivity: float = 2.0,
        baseline_volatility: float = 0.01  # 1% daily vol
    ):
        """
        Initialize volatility-based spread model.

        Args:
            base_spread_bps: Base spread at baseline volatility
            volatility_sensitivity: How much spread increases with vol
            baseline_volatility: Baseline volatility level
        """
        self.base_spread_bps = base_spread_bps
        self.volatility_sensitivity = volatility_sensitivity
        self.baseline_volatility = baseline_volatility

        logger.info(
            f"Initialized VolatilityBasedSpreadModel: "
            f"base={base_spread_bps} bps, sensitivity={volatility_sensitivity}"
        )

    def calculate_spread(
        self,
        symbol: str,
        side: str,
        quantity: float,
        mid_price: float,
        volatility: Optional[float] = None,  # Recent volatility
        **kwargs
    ) -> SpreadResult:
        """
        Calculate volatility-based spread cost.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            mid_price: Mid-market price
            volatility: Recent realized volatility (daily)
            **kwargs: Additional parameters

        Returns:
            SpreadResult object
        """
        # Use baseline if no volatility provided
        if volatility is None or volatility <= 0:
            volatility = self.baseline_volatility

        # Volatility ratio relative to baseline
        vol_ratio = volatility / self.baseline_volatility

        # Spread scales with volatility
        volatility_multiplier = vol_ratio ** self.volatility_sensitivity

        # Adjusted spread
        adjusted_spread_bps = self.base_spread_bps * volatility_multiplier

        # Convert to price
        spread_pct = adjusted_spread_bps / 10000.0
        half_spread = mid_price * spread_pct / 2

        # Execution price
        if side.upper() == 'BUY':
            execution_price = mid_price + half_spread
        else:
            execution_price = mid_price - half_spread

        # Total cost
        spread_cost = half_spread * quantity

        return SpreadResult(
            quoted_spread=mid_price * spread_pct,
            effective_spread=mid_price * spread_pct,
            spread_cost=spread_cost,
            execution_price=execution_price,
            reference_price=mid_price,
            spread_bps=adjusted_spread_bps,
            spread_breakdown={
                'base_spread_bps': self.base_spread_bps,
                'volatility': volatility,
                'vol_ratio': vol_ratio,
                'volatility_multiplier': volatility_multiplier,
                'adjusted_spread_bps': adjusted_spread_bps
            }
        )


class DynamicSpreadModel:
    """
    Dynamic spread model combining multiple factors.

    Comprehensive model that considers:
    - Time of day effects
    - Trade size relative to volume
    - Market volatility
    - Cross-asset correlations
    - Market regime

    Most realistic model for production backtesting.
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        time_model: Optional[TimeVaryingSpreadModel] = None,
        volume_model: Optional[VolumeDependentSpreadModel] = None,
        volatility_model: Optional[VolatilityBasedSpreadModel] = None,
        time_weight: float = 0.3,
        volume_weight: float = 0.4,
        volatility_weight: float = 0.3
    ):
        """
        Initialize dynamic spread model.

        Args:
            base_spread_bps: Base spread
            time_model: Time-varying component
            volume_model: Volume-dependent component
            volatility_model: Volatility-based component
            time_weight: Weight for time component
            volume_weight: Weight for volume component
            volatility_weight: Weight for volatility component
        """
        self.base_spread_bps = base_spread_bps

        # Component models
        self.time_model = time_model or TimeVaryingSpreadModel(base_spread_bps)
        self.volume_model = volume_model or VolumeDependentSpreadModel(base_spread_bps)
        self.volatility_model = volatility_model or VolatilityBasedSpreadModel(base_spread_bps)

        # Weights (should sum to 1.0)
        total_weight = time_weight + volume_weight + volatility_weight
        self.time_weight = time_weight / total_weight
        self.volume_weight = volume_weight / total_weight
        self.volatility_weight = volatility_weight / total_weight

        logger.info(
            f"Initialized DynamicSpreadModel: base={base_spread_bps} bps, "
            f"weights=({self.time_weight:.2f}, {self.volume_weight:.2f}, {self.volatility_weight:.2f})"
        )

    def calculate_spread(
        self,
        symbol: str,
        side: str,
        quantity: float,
        mid_price: float,
        timestamp: Optional[datetime] = None,
        adv: Optional[float] = None,
        volatility: Optional[float] = None,
        **kwargs
    ) -> SpreadResult:
        """
        Calculate dynamic spread cost.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            mid_price: Mid-market price
            timestamp: Trade timestamp
            adv: Average daily volume
            volatility: Recent volatility
            **kwargs: Additional parameters

        Returns:
            SpreadResult object
        """
        # Calculate component spreads
        time_result = self.time_model.calculate_spread(
            symbol=symbol,
            side=side,
            quantity=quantity,
            mid_price=mid_price,
            timestamp=timestamp,
            **kwargs
        )

        volume_result = self.volume_model.calculate_spread(
            symbol=symbol,
            side=side,
            quantity=quantity,
            mid_price=mid_price,
            adv=adv,
            **kwargs
        )

        volatility_result = self.volatility_model.calculate_spread(
            symbol=symbol,
            side=side,
            quantity=quantity,
            mid_price=mid_price,
            volatility=volatility,
            **kwargs
        )

        # Weighted combination
        combined_spread_bps = (
            time_result.spread_bps * self.time_weight +
            volume_result.spread_bps * self.volume_weight +
            volatility_result.spread_bps * self.volatility_weight
        )

        # Convert to price
        spread_pct = combined_spread_bps / 10000.0
        half_spread = mid_price * spread_pct / 2

        # Execution price
        if side.upper() == 'BUY':
            execution_price = mid_price + half_spread
        else:
            execution_price = mid_price - half_spread

        # Total cost
        spread_cost = half_spread * quantity

        return SpreadResult(
            quoted_spread=mid_price * spread_pct,
            effective_spread=mid_price * spread_pct,
            spread_cost=spread_cost,
            execution_price=execution_price,
            reference_price=mid_price,
            spread_bps=combined_spread_bps,
            spread_breakdown={
                'base_spread_bps': self.base_spread_bps,
                'time_spread_bps': time_result.spread_bps,
                'volume_spread_bps': volume_result.spread_bps,
                'volatility_spread_bps': volatility_result.spread_bps,
                'combined_spread_bps': combined_spread_bps,
                'weights': {
                    'time': self.time_weight,
                    'volume': self.volume_weight,
                    'volatility': self.volatility_weight
                }
            }
        )


class ImplementationShortfallModel:
    """
    Implementation shortfall model.

    Measures the difference between decision price and final execution price,
    including:
    - Delay cost (market moves while deciding)
    - Spread cost (crossing bid-ask)
    - Impact cost (market impact of order)
    - Opportunity cost (missed fills)

    Industry standard for execution cost analysis.
    """

    def __init__(
        self,
        spread_model: Optional[DynamicSpreadModel] = None,
        delay_factor: float = 0.5,
        impact_factor: float = 0.3
    ):
        """
        Initialize implementation shortfall model.

        Args:
            spread_model: Spread model to use
            delay_factor: Scaling for delay cost
            impact_factor: Scaling for market impact
        """
        self.spread_model = spread_model or DynamicSpreadModel()
        self.delay_factor = delay_factor
        self.impact_factor = impact_factor

        logger.info("Initialized ImplementationShortfallModel")

    def calculate_implementation_shortfall(
        self,
        symbol: str,
        side: str,
        quantity: float,
        decision_price: float,
        execution_price: float,
        mid_price_at_execution: float,
        timestamp: datetime,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate implementation shortfall components.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            decision_price: Price when decision was made
            execution_price: Actual execution price
            mid_price_at_execution: Mid-market price at execution
            timestamp: Execution timestamp
            **kwargs: Additional parameters

        Returns:
            Dictionary with shortfall components
        """
        # Total implementation shortfall
        if side.upper() == 'BUY':
            total_shortfall = (execution_price - decision_price) * quantity
            price_drift = (mid_price_at_execution - decision_price) * quantity
        else:
            total_shortfall = (decision_price - execution_price) * quantity
            price_drift = (decision_price - mid_price_at_execution) * quantity

        # Spread cost (crossing bid-ask)
        spread_result = self.spread_model.calculate_spread(
            symbol=symbol,
            side=side,
            quantity=quantity,
            mid_price=mid_price_at_execution,
            timestamp=timestamp,
            **kwargs
        )
        spread_cost = spread_result.spread_cost

        # Decomposition
        delay_cost = price_drift * self.delay_factor
        impact_cost = (total_shortfall - delay_cost - spread_cost) * self.impact_factor
        opportunity_cost = total_shortfall - delay_cost - spread_cost - impact_cost

        return {
            'total_shortfall': total_shortfall,
            'total_shortfall_bps': (total_shortfall / (decision_price * quantity)) * 10000,
            'delay_cost': delay_cost,
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'opportunity_cost': opportunity_cost,
            'components_pct': {
                'delay': delay_cost / total_shortfall if total_shortfall != 0 else 0,
                'spread': spread_cost / total_shortfall if total_shortfall != 0 else 0,
                'impact': impact_cost / total_shortfall if total_shortfall != 0 else 0,
                'opportunity': opportunity_cost / total_shortfall if total_shortfall != 0 else 0
            }
        }


class EffectiveSpreadAnalyzer:
    """
    Effective spread analyzer.

    Analyzes realized spreads from execution history to:
    - Validate spread models
    - Detect execution quality issues
    - Calibrate model parameters
    - Provide benchmarks
    """

    def __init__(self):
        """Initialize effective spread analyzer."""
        self.execution_history: List[Dict] = []
        logger.info("Initialized EffectiveSpreadAnalyzer")

    def record_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        execution_price: float,
        mid_price: float,
        timestamp: datetime,
        **metadata
    ):
        """
        Record execution for analysis.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Quantity executed
            execution_price: Execution price
            mid_price: Mid-market price at execution
            timestamp: Execution timestamp
            **metadata: Additional metadata
        """
        # Calculate effective spread
        if side.upper() == 'BUY':
            effective_spread = 2 * (execution_price - mid_price)
        else:
            effective_spread = 2 * (mid_price - execution_price)

        effective_spread_bps = (effective_spread / mid_price) * 10000

        self.execution_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'execution_price': execution_price,
            'mid_price': mid_price,
            'effective_spread': effective_spread,
            'effective_spread_bps': effective_spread_bps,
            **metadata
        })

    def get_statistics(
        self,
        symbol: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get effective spread statistics.

        Args:
            symbol: Filter by symbol (optional)
            lookback_days: Lookback period in days (optional)

        Returns:
            Dictionary with statistics
        """
        if not self.execution_history:
            return {}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.execution_history)

        # Apply filters
        if symbol:
            df = df[df['symbol'] == symbol]

        if lookback_days:
            cutoff = datetime.now() - pd.Timedelta(days=lookback_days)
            df = df[df['timestamp'] >= cutoff]

        if df.empty:
            return {}

        # Calculate statistics
        stats = {
            'mean_effective_spread_bps': df['effective_spread_bps'].mean(),
            'median_effective_spread_bps': df['effective_spread_bps'].median(),
            'std_effective_spread_bps': df['effective_spread_bps'].std(),
            'min_effective_spread_bps': df['effective_spread_bps'].min(),
            'max_effective_spread_bps': df['effective_spread_bps'].max(),
            'total_executions': len(df),
            'total_spread_cost': df['effective_spread'].sum() / 2  # Half-spread
        }

        return stats

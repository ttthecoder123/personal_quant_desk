"""
Execution algorithms for systematic trading.

Implements various execution strategies to minimize market impact
and slippage while executing large orders over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class ExecutionMode(Enum):
    """Execution mode enum."""
    AGGRESSIVE = "AGGRESSIVE"  # Prioritize speed over cost
    PASSIVE = "PASSIVE"        # Prioritize cost over speed
    BALANCED = "BALANCED"      # Balance speed and cost


@dataclass
class ExecutionSlice:
    """
    Represents a slice of an order for execution.

    Attributes:
        quantity: Quantity to execute in this slice
        target_time: Target execution time
        price_limit: Optional price limit
        urgency: Urgency score (0 to 1)
    """
    quantity: float
    target_time: datetime
    price_limit: Optional[float] = None
    urgency: float = 0.5


class ExecutionAlgorithm:
    """
    Execution algorithms for order execution.

    Implements:
    - TWAP (Time-Weighted Average Price)
    - VWAP (Volume-Weighted Average Price)
    - Aggressive/passive execution modes
    - Iceberg order slicing
    - Smart order routing

    Attributes:
        default_mode (ExecutionMode): Default execution mode
        max_participation_rate (float): Maximum market participation rate
        min_slice_size (float): Minimum slice size
        enable_dark_pools (bool): Whether to use dark pools
    """

    def __init__(
        self,
        default_mode: ExecutionMode = ExecutionMode.BALANCED,
        max_participation_rate: float = 0.10,
        min_slice_size: float = 10.0,
        enable_dark_pools: bool = False,
    ):
        """
        Initialize execution algorithm.

        Args:
            default_mode: Default execution mode
            max_participation_rate: Max participation in market volume (e.g., 0.10 = 10%)
            min_slice_size: Minimum size for order slices
            enable_dark_pools: Whether to route to dark pools
        """
        self.default_mode = default_mode
        self.max_participation_rate = max_participation_rate
        self.min_slice_size = min_slice_size
        self.enable_dark_pools = enable_dark_pools

        logger.info(
            f"ExecutionAlgorithm initialized: mode={default_mode.value}, "
            f"max_participation={max_participation_rate}"
        )

    def twap_schedule(
        self,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 5,
    ) -> List[ExecutionSlice]:
        """
        Generate TWAP (Time-Weighted Average Price) execution schedule.

        Divides order into equal slices executed at regular intervals.
        Minimizes timing risk and reduces market impact.

        Args:
            total_quantity: Total quantity to execute
            start_time: Start time for execution
            end_time: End time for execution
            interval_minutes: Minutes between slices

        Returns:
            List of execution slices
        """
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        n_slices = max(1, int(duration / interval_minutes))

        # Equal quantity per slice
        slice_quantity = total_quantity / n_slices

        if slice_quantity < self.min_slice_size:
            # Adjust number of slices to meet minimum size
            n_slices = max(1, int(total_quantity / self.min_slice_size))
            slice_quantity = total_quantity / n_slices

        # Generate schedule
        schedule = []
        current_time = start_time

        for i in range(n_slices):
            # Last slice gets remaining quantity
            if i == n_slices - 1:
                qty = total_quantity - (slice_quantity * i)
            else:
                qty = slice_quantity

            schedule.append(ExecutionSlice(
                quantity=qty,
                target_time=current_time,
                urgency=0.5  # Balanced urgency
            ))

            current_time += timedelta(minutes=interval_minutes)

        logger.info(
            f"TWAP schedule: {n_slices} slices, "
            f"slice_size={slice_quantity:.2f}, "
            f"duration={duration:.0f}min"
        )

        return schedule

    def vwap_schedule(
        self,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        volume_profile: Optional[pd.Series] = None,
    ) -> List[ExecutionSlice]:
        """
        Generate VWAP (Volume-Weighted Average Price) execution schedule.

        Distributes order according to expected volume profile.
        Executes more during high-volume periods to minimize impact.

        Args:
            total_quantity: Total quantity to execute
            start_time: Start time
            end_time: End time
            volume_profile: Expected volume by time (if None, uses typical profile)

        Returns:
            List of execution slices
        """
        # Use typical intraday volume profile if not provided
        if volume_profile is None:
            volume_profile = self._get_typical_volume_profile(start_time, end_time)

        if volume_profile.empty:
            logger.warning("Empty volume profile, falling back to TWAP")
            return self.twap_schedule(total_quantity, start_time, end_time)

        # Normalize volume profile to sum to 1
        volume_weights = volume_profile / volume_profile.sum()

        # Calculate quantities per slice
        schedule = []
        for timestamp, weight in volume_weights.items():
            qty = total_quantity * weight

            if qty < self.min_slice_size:
                continue

            schedule.append(ExecutionSlice(
                quantity=qty,
                target_time=timestamp,
                urgency=weight  # Higher urgency during high volume
            ))

        logger.info(
            f"VWAP schedule: {len(schedule)} slices following volume profile"
        )

        return schedule

    def _get_typical_volume_profile(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 5,
    ) -> pd.Series:
        """
        Generate typical U-shaped intraday volume profile.

        Volume is typically higher at market open and close.

        Args:
            start_time: Start time
            end_time: End time
            interval_minutes: Interval in minutes

        Returns:
            Volume profile series
        """
        # Generate time points
        duration = (end_time - start_time).total_seconds() / 60
        n_points = max(1, int(duration / interval_minutes))

        times = [start_time + timedelta(minutes=i * interval_minutes) for i in range(n_points)]

        # U-shaped volume profile (higher at start and end)
        # Using a quadratic function
        volumes = []
        for i in range(n_points):
            # Normalize position (0 to 1)
            x = i / max(1, n_points - 1)

            # U-shape: high at 0 and 1, low at 0.5
            volume = 1.0 + 2.0 * (x - 0.5) ** 2

            volumes.append(volume)

        profile = pd.Series(volumes, index=times)

        return profile

    def iceberg_slicing(
        self,
        total_quantity: float,
        display_quantity: float,
        schedule: Optional[List[ExecutionSlice]] = None,
    ) -> List[ExecutionSlice]:
        """
        Create iceberg order slices.

        Shows only a small portion of the order to the market while
        hiding the full size. Reduces market impact.

        Args:
            total_quantity: Total order quantity
            display_quantity: Quantity to display per slice
            schedule: Optional existing schedule to modify

        Returns:
            List of iceberg slices
        """
        if display_quantity >= total_quantity:
            logger.warning("Display quantity >= total quantity, no iceberg needed")
            return [ExecutionSlice(quantity=total_quantity, target_time=datetime.now())]

        # Calculate number of slices
        n_slices = int(np.ceil(total_quantity / display_quantity))

        slices = []
        remaining = total_quantity

        for i in range(n_slices):
            qty = min(display_quantity, remaining)

            slices.append(ExecutionSlice(
                quantity=qty,
                target_time=datetime.now() + timedelta(seconds=i * 30),  # 30 sec apart
                urgency=0.3  # Lower urgency for iceberg
            ))

            remaining -= qty

        logger.info(
            f"Iceberg slicing: {n_slices} slices of {display_quantity} "
            f"(total={total_quantity})"
        )

        return slices

    def participation_rate_strategy(
        self,
        total_quantity: float,
        expected_market_volume: float,
        target_participation: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate execution rate based on market participation target.

        Args:
            total_quantity: Total quantity to execute
            expected_market_volume: Expected market volume
            target_participation: Target participation rate (uses max if None)

        Returns:
            Dictionary with execution parameters
        """
        if target_participation is None:
            target_participation = self.max_participation_rate

        # Clip to max participation
        target_participation = min(target_participation, self.max_participation_rate)

        # Calculate execution rate
        max_quantity = expected_market_volume * target_participation

        if max_quantity >= total_quantity:
            # Can execute all in one period
            execution_rate = total_quantity
            n_periods = 1
        else:
            # Need multiple periods
            execution_rate = max_quantity
            n_periods = int(np.ceil(total_quantity / execution_rate))

        strategy = {
            'participation_rate': target_participation,
            'execution_rate': execution_rate,
            'n_periods': n_periods,
            'estimated_duration_minutes': n_periods * 5,  # Assume 5-min periods
        }

        logger.info(
            f"Participation strategy: {target_participation:.1%} participation, "
            f"{n_periods} periods"
        )

        return strategy

    def adaptive_execution(
        self,
        total_quantity: float,
        current_price: float,
        limit_price: float,
        urgency: float = 0.5,
        market_impact_model: Optional[callable] = None,
    ) -> List[ExecutionSlice]:
        """
        Adaptive execution based on price, urgency, and market impact.

        Adjusts execution speed based on:
        - Distance from limit price
        - Urgency level
        - Estimated market impact

        Args:
            total_quantity: Total quantity to execute
            current_price: Current market price
            limit_price: Limit price
            urgency: Urgency score (0 to 1, higher = faster execution)
            market_impact_model: Optional function to estimate market impact

        Returns:
            List of execution slices
        """
        # Calculate price pressure (how far from limit)
        price_distance = abs(current_price - limit_price) / limit_price

        # Adjust urgency based on price
        if price_distance > 0.02:  # More than 2% away
            # Slow down if price is unfavorable
            adjusted_urgency = urgency * 0.5
        else:
            # Speed up if price is favorable
            adjusted_urgency = min(1.0, urgency * 1.5)

        # Calculate number of slices based on urgency
        if adjusted_urgency > 0.8:
            n_slices = 2  # Aggressive
        elif adjusted_urgency > 0.5:
            n_slices = 5  # Balanced
        else:
            n_slices = 10  # Passive

        # Create schedule
        slice_quantity = total_quantity / n_slices
        schedule = []
        current_time = datetime.now()

        for i in range(n_slices):
            # Calculate time spacing based on urgency
            time_interval = int(60 / max(adjusted_urgency, 0.1))  # seconds

            schedule.append(ExecutionSlice(
                quantity=slice_quantity if i < n_slices - 1 else total_quantity - (slice_quantity * i),
                target_time=current_time + timedelta(seconds=i * time_interval),
                price_limit=limit_price,
                urgency=adjusted_urgency
            ))

        logger.info(
            f"Adaptive execution: {n_slices} slices, "
            f"urgency={adjusted_urgency:.2f}, "
            f"price_distance={price_distance:.2%}"
        )

        return schedule

    def smart_order_routing(
        self,
        quantity: float,
        symbol: str,
        venue_liquidity: Dict[str, float],
        venue_fees: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Smart order routing across multiple venues.

        Optimally routes order to minimize costs and maximize fill probability.

        Args:
            quantity: Order quantity
            symbol: Trading symbol
            venue_liquidity: Available liquidity by venue
            venue_fees: Trading fees by venue (in basis points)

        Returns:
            Dictionary mapping venue to quantity allocation
        """
        # Simple routing: prioritize by liquidity and fees
        venues = list(venue_liquidity.keys())

        # Calculate score for each venue (higher is better)
        venue_scores = {}
        for venue in venues:
            liquidity = venue_liquidity[venue]
            fee = venue_fees.get(venue, 10.0)  # Default 10 bps

            # Score = liquidity / (1 + fee/100)
            # Higher liquidity and lower fees → higher score
            score = liquidity / (1 + fee / 100)
            venue_scores[venue] = score

        # Sort venues by score
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)

        # Allocate quantity
        allocation = {}
        remaining = quantity

        for venue, score in sorted_venues:
            available = venue_liquidity[venue]
            allocated = min(available, remaining)

            if allocated > 0:
                allocation[venue] = allocated
                remaining -= allocated

            if remaining <= 0:
                break

        if remaining > 0:
            logger.warning(
                f"Insufficient liquidity: {remaining:.2f} units unallocated"
            )

        logger.info(
            f"Smart routing: {len(allocation)} venues, "
            f"{sum(allocation.values()):.2f}/{quantity:.2f} allocated"
        )

        return allocation

    def execute_with_mode(
        self,
        total_quantity: float,
        mode: Optional[ExecutionMode] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ExecutionSlice]:
        """
        Execute order with specified mode.

        Args:
            total_quantity: Total quantity to execute
            mode: Execution mode (uses default if None)
            start_time: Start time (uses now if None)
            end_time: End time (inferred from mode if None)

        Returns:
            Execution schedule
        """
        if mode is None:
            mode = self.default_mode

        if start_time is None:
            start_time = datetime.now()

        # Determine execution parameters based on mode
        if mode == ExecutionMode.AGGRESSIVE:
            # Fast execution: 5 minutes
            if end_time is None:
                end_time = start_time + timedelta(minutes=5)
            interval = 1  # 1-minute slices
            urgency = 0.9

        elif mode == ExecutionMode.PASSIVE:
            # Slow execution: 60 minutes
            if end_time is None:
                end_time = start_time + timedelta(minutes=60)
            interval = 5  # 5-minute slices
            urgency = 0.3

        else:  # BALANCED
            # Moderate execution: 20 minutes
            if end_time is None:
                end_time = start_time + timedelta(minutes=20)
            interval = 3  # 3-minute slices
            urgency = 0.5

        # Generate TWAP schedule
        schedule = self.twap_schedule(
            total_quantity,
            start_time,
            end_time,
            interval_minutes=interval
        )

        # Adjust urgency
        for slice in schedule:
            slice.urgency = urgency

        logger.info(
            f"Executing with {mode.value} mode: {len(schedule)} slices, "
            f"urgency={urgency}"
        )

        return schedule

    def estimate_execution_time(
        self,
        total_quantity: float,
        expected_volume: float,
        urgency: float = 0.5,
    ) -> float:
        """
        Estimate time required to execute order.

        Args:
            total_quantity: Total quantity to execute
            expected_volume: Expected market volume per period
            urgency: Urgency level (0 to 1)

        Returns:
            Estimated time in minutes
        """
        # Adjust participation rate based on urgency
        participation = self.max_participation_rate * (0.5 + urgency)

        # Calculate execution rate
        execution_rate = expected_volume * participation

        if execution_rate <= 0:
            logger.warning("Invalid execution rate, returning max time")
            return 60.0

        # Estimated time = quantity / rate
        estimated_time = total_quantity / execution_rate

        # Assume each period is 5 minutes
        estimated_minutes = estimated_time * 5

        logger.debug(
            f"Execution time estimate: {estimated_minutes:.1f} minutes "
            f"(participation={participation:.1%})"
        )

        return estimated_minutes

    def calculate_execution_shortfall(
        self,
        initial_price: float,
        execution_prices: List[float],
        quantities: List[float],
        side: str = 'BUY',
    ) -> Dict[str, float]:
        """
        Calculate implementation shortfall (slippage + opportunity cost).

        Args:
            initial_price: Decision price (when order was decided)
            execution_prices: Actual execution prices
            quantities: Quantities executed at each price
            side: Order side ('BUY' or 'SELL')

        Returns:
            Dictionary with shortfall metrics
        """
        if len(execution_prices) != len(quantities):
            logger.warning("Mismatched execution prices and quantities")
            return {}

        total_quantity = sum(quantities)
        if total_quantity == 0:
            return {}

        # Calculate average execution price
        total_cost = sum(p * q for p, q in zip(execution_prices, quantities))
        avg_execution_price = total_cost / total_quantity

        # Calculate slippage
        if side.upper() == 'BUY':
            slippage = avg_execution_price - initial_price
        else:  # SELL
            slippage = initial_price - avg_execution_price

        slippage_bps = (slippage / initial_price) * 10000

        # Calculate arrival cost (vs decision price)
        arrival_cost = abs(slippage) * total_quantity

        metrics = {
            'initial_price': initial_price,
            'avg_execution_price': avg_execution_price,
            'slippage': slippage,
            'slippage_bps': slippage_bps,
            'arrival_cost': arrival_cost,
            'total_quantity': total_quantity,
        }

        logger.info(
            f"Implementation shortfall: {slippage_bps:.2f} bps, "
            f"cost=${arrival_cost:.2f}"
        )

        return metrics

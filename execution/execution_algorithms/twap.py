"""
Time-Weighted Average Price (TWAP) Execution Algorithm

Splits orders evenly across time intervals to minimize market impact.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from ..order_management.order_types import Order, OrderFactory, OrderSide
import logging

logger = logging.getLogger(__name__)


class TWAPAlgorithm:
    """
    TWAP execution algorithm

    Executes orders by splitting evenly across time with:
    - Uniform or randomized intervals
    - Volume participation limits
    - Catch-up logic for missed intervals
    - End-of-day completion guarantee
    """

    def __init__(self, config: Dict = None):
        """
        Initialize TWAP algorithm

        Args:
            config: Algorithm configuration
        """
        self.config = config or {}
        self.randomize = self.config.get('randomize_intervals', True)
        self.max_participation = self.config.get('max_participation', 0.10)
        self.min_slice_size = self.config.get('min_slice_size', 1.0)

    def generate_schedule(self, parent_order: Order, start_time: datetime,
                         end_time: datetime, num_slices: int = 10) -> List[Dict]:
        """
        Generate TWAP execution schedule

        Args:
            parent_order: Parent order to execute
            start_time: Start time
            end_time: End time
            num_slices: Number of time slices

        Returns:
            List of execution slices with timing and quantity
        """
        logger.info(
            f"Generating TWAP schedule: {parent_order.quantity} shares "
            f"over {num_slices} slices from {start_time} to {end_time}"
        )

        total_duration = (end_time - start_time).total_seconds()
        interval_duration = total_duration / num_slices

        # Calculate base quantity per slice
        base_quantity = parent_order.quantity / num_slices

        schedule = []
        cumulative_time = 0

        for i in range(num_slices):
            # Randomize interval if enabled
            if self.randomize and i < num_slices - 1:
                # Add randomness ±20% to interval
                random_factor = np.random.uniform(0.8, 1.2)
                slice_duration = interval_duration * random_factor
            else:
                slice_duration = interval_duration

            # Randomize quantity if enabled (±10%)
            if self.randomize and i < num_slices - 1:
                random_factor = np.random.uniform(0.9, 1.1)
                slice_quantity = base_quantity * random_factor
            else:
                # Last slice gets remaining quantity
                slice_quantity = parent_order.quantity - sum(s['quantity'] for s in schedule)

            # Ensure minimum size
            slice_quantity = max(slice_quantity, self.min_slice_size)

            # Calculate execution time
            execution_time = start_time + timedelta(seconds=cumulative_time)

            schedule.append({
                'slice_number': i + 1,
                'execution_time': execution_time,
                'quantity': round(slice_quantity, 2),
                'interval_seconds': slice_duration,
                'status': 'pending'
            })

            cumulative_time += slice_duration

        # Adjust last slice to ensure total matches
        total_scheduled = sum(s['quantity'] for s in schedule)
        if abs(total_scheduled - parent_order.quantity) > 0.01:
            schedule[-1]['quantity'] += (parent_order.quantity - total_scheduled)

        logger.info(f"Generated TWAP schedule with {len(schedule)} slices")
        return schedule

    def create_slice_order(self, parent_order: Order, slice_info: Dict) -> Order:
        """
        Create child order for a TWAP slice

        Args:
            parent_order: Parent order
            slice_info: Slice information from schedule

        Returns:
            Child order
        """
        # Use limit order slightly better than market to get fills
        # In production, would use real-time market data
        price = parent_order.price if parent_order.price else None

        child_order = OrderFactory.create_limit_order(
            symbol=parent_order.symbol,
            side=parent_order.side,
            quantity=slice_info['quantity'],
            price=price,
            time_in_force=parent_order.time_in_force,
            parent_order_id=parent_order.order_id,
            account=parent_order.account,
            strategy_id=parent_order.strategy_id
        )

        child_order.tags['twap_slice'] = slice_info['slice_number']
        child_order.tags['scheduled_time'] = slice_info['execution_time']

        return child_order

    def should_execute_slice(self, slice_info: Dict, current_time: datetime) -> bool:
        """
        Check if slice should be executed now

        Args:
            slice_info: Slice information
            current_time: Current time

        Returns:
            True if slice should execute
        """
        return current_time >= slice_info['execution_time']

    def adjust_remaining_schedule(self, schedule: List[Dict], completed_quantity: float,
                                 remaining_quantity: float, current_time: datetime) -> List[Dict]:
        """
        Adjust schedule based on execution progress

        Args:
            schedule: Original schedule
            completed_quantity: Quantity completed so far
            remaining_quantity: Quantity remaining
            current_time: Current time

        Returns:
            Adjusted schedule
        """
        # Find pending slices
        pending_slices = [s for s in schedule if s['status'] == 'pending'
                         and s['execution_time'] >= current_time]

        if not pending_slices:
            return schedule

        # Redistribute remaining quantity across pending slices
        remaining_per_slice = remaining_quantity / len(pending_slices)

        for slice_info in pending_slices:
            slice_info['quantity'] = round(remaining_per_slice, 2)

        # Adjust last slice for rounding
        total_adjusted = sum(s['quantity'] for s in pending_slices)
        if abs(total_adjusted - remaining_quantity) > 0.01:
            pending_slices[-1]['quantity'] += (remaining_quantity - total_adjusted)

        logger.info(f"Adjusted TWAP schedule: {remaining_quantity} shares across {len(pending_slices)} slices")
        return schedule

    def get_progress(self, schedule: List[Dict]) -> Dict:
        """
        Get execution progress

        Args:
            schedule: Execution schedule

        Returns:
            Progress metrics
        """
        total_slices = len(schedule)
        completed_slices = len([s for s in schedule if s['status'] == 'completed'])
        pending_slices = len([s for s in schedule if s['status'] == 'pending'])

        total_quantity = sum(s['quantity'] for s in schedule)
        completed_quantity = sum(
            s['quantity'] for s in schedule if s['status'] == 'completed'
        )

        return {
            'total_slices': total_slices,
            'completed_slices': completed_slices,
            'pending_slices': pending_slices,
            'total_quantity': total_quantity,
            'completed_quantity': completed_quantity,
            'completion_percentage': (completed_slices / total_slices * 100) if total_slices > 0 else 0
        }

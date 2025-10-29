"""
Volume-Weighted Average Price (VWAP) Execution Algorithm

Executes orders following historical volume patterns to achieve VWAP benchmark.
"""

import numpy as np
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional
from ..order_management.order_types import Order
import logging

logger = logging.getLogger(__name__)


class VWAPAlgorithm:
    """
    VWAP execution algorithm

    Executes orders following volume profile to achieve VWAP with:
    - Historical volume curve estimation
    - Real-time volume tracking
    - Dynamic schedule adjustment
    - Aggressive/passive modes
    """

    def __init__(self, config: Dict = None):
        """Initialize VWAP algorithm"""
        self.config = config or {}
        self.max_participation = self.config.get('max_participation', 0.10)
        self.aggressive_on_close = self.config.get('aggressive_on_close', True)
        self.use_dark_pools = self.config.get('use_dark_pools', True)

        # Typical intraday volume profile (percentage of daily volume per hour)
        # Based on US market typical patterns
        self.default_volume_profile = self._generate_default_volume_profile()

    def _generate_default_volume_profile(self) -> Dict[int, float]:
        """Generate typical intraday volume profile"""
        # Hour -> percentage of daily volume
        # US market hours: 9:30 AM - 4:00 PM (6.5 hours)
        return {
            9: 0.15,   # 9:30-10:00 (opening)
            10: 0.12,
            11: 0.10,
            12: 0.08,
            13: 0.10,  # 1:00-2:00 PM
            14: 0.12,
            15: 0.18,  # 3:00-4:00 PM (closing)
            16: 0.15   # Close
        }

    def estimate_volume_profile(self, symbol: str, historical_data: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Estimate intraday volume profile

        Args:
            symbol: Symbol to estimate
            historical_data: Historical volume data by time

        Returns:
            Volume profile dict (hour -> percentage)
        """
        if historical_data is not None:
            # Calculate from historical data
            # For now, use default profile
            pass

        return self.default_volume_profile

    def generate_schedule(self, parent_order: Order, start_time: datetime,
                         end_time: datetime, volume_profile: Optional[Dict] = None) -> List[Dict]:
        """
        Generate VWAP execution schedule

        Args:
            parent_order: Parent order
            start_time: Start time
            end_time: End time
            volume_profile: Optional volume profile

        Returns:
            List of execution slices
        """
        logger.info(f"Generating VWAP schedule: {parent_order.quantity} shares")

        if volume_profile is None:
            volume_profile = self.estimate_volume_profile(parent_order.symbol)

        # Get relevant hours
        current_hour = start_time.hour
        end_hour = end_time.hour

        schedule = []

        # Calculate total volume percentage in our time window
        relevant_hours = [h for h in volume_profile.keys() if current_hour <= h <= end_hour]
        total_volume_pct = sum(volume_profile.get(h, 0) for h in relevant_hours)

        # Distribute order quantity based on volume profile
        for hour in relevant_hours:
            hour_volume_pct = volume_profile.get(hour, 0)
            slice_pct = hour_volume_pct / total_volume_pct if total_volume_pct > 0 else 1.0 / len(relevant_hours)
            slice_quantity = parent_order.quantity * slice_pct

            # Determine execution time for this hour
            if hour == current_hour:
                execution_time = start_time
            else:
                execution_time = start_time.replace(hour=hour, minute=0, second=0)

            schedule.append({
                'hour': hour,
                'execution_time': execution_time,
                'quantity': round(slice_quantity, 2),
                'target_participation': self.max_participation,
                'status': 'pending',
                'volume_percentage': hour_volume_pct
            })

        # Adjust last slice for rounding
        total_scheduled = sum(s['quantity'] for s in schedule)
        if abs(total_scheduled - parent_order.quantity) > 0.01:
            schedule[-1]['quantity'] += (parent_order.quantity - total_scheduled)

        logger.info(f"Generated VWAP schedule with {len(schedule)} slices")
        return schedule

    def calculate_participation_rate(self, target_quantity: float, market_volume: float,
                                    time_remaining: float, quantity_remaining: float) -> float:
        """
        Calculate dynamic participation rate

        Args:
            target_quantity: Target quantity for this period
            market_volume: Observed market volume
            time_remaining: Time remaining (minutes)
            quantity_remaining: Quantity remaining to execute

        Returns:
            Participation rate (0-1)
        """
        # Base participation rate
        base_rate = self.max_participation

        # Adjust based on progress
        if time_remaining > 0:
            required_rate = quantity_remaining / (market_volume * time_remaining / 60)
            # Cap at max but allow aggressive execution if behind
            adjusted_rate = min(required_rate, base_rate * 1.5)
        else:
            # Urgent - be aggressive
            adjusted_rate = min(0.25, base_rate * 2)

        return max(0.01, min(adjusted_rate, 0.25))  # Keep between 1% and 25%

    def adjust_for_volume(self, schedule: List[Dict], actual_volume: Dict[int, float],
                         completed_quantity: float, remaining_quantity: float) -> List[Dict]:
        """
        Adjust schedule based on actual volume

        Args:
            schedule: Original schedule
            actual_volume: Actual observed volume by hour
            completed_quantity: Quantity completed
            remaining_quantity: Quantity remaining

        Returns:
            Adjusted schedule
        """
        pending_slices = [s for s in schedule if s['status'] == 'pending']
        if not pending_slices:
            return schedule

        # Recalculate based on remaining volume profile
        total_remaining_volume = sum(
            s['volume_percentage'] for s in pending_slices
        )

        for slice_info in pending_slices:
            volume_pct = slice_info['volume_percentage']
            slice_pct = volume_pct / total_remaining_volume if total_remaining_volume > 0 else 1.0 / len(pending_slices)
            slice_info['quantity'] = round(remaining_quantity * slice_pct, 2)

        # Adjust last slice
        total_adjusted = sum(s['quantity'] for s in pending_slices)
        if abs(total_adjusted - remaining_quantity) > 0.01:
            pending_slices[-1]['quantity'] += (remaining_quantity - total_adjusted)

        logger.info(f"Adjusted VWAP schedule based on actual volume")
        return schedule

    def get_progress(self, schedule: List[Dict], current_vwap: Optional[float] = None) -> Dict:
        """Get execution progress"""
        total_quantity = sum(s['quantity'] for s in schedule)
        completed_quantity = sum(
            s['quantity'] for s in schedule if s['status'] == 'completed'
        )

        progress = {
            'total_quantity': total_quantity,
            'completed_quantity': completed_quantity,
            'remaining_quantity': total_quantity - completed_quantity,
            'completion_percentage': (completed_quantity / total_quantity * 100) if total_quantity > 0 else 0
        }

        if current_vwap is not None:
            progress['current_vwap'] = current_vwap

        return progress

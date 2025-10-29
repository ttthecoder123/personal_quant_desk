"""
Implementation Shortfall Algorithm (Almgren-Chriss)

Optimal execution minimizing implementation shortfall with risk aversion.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from ..order_management.order_types import Order
import logging

logger = logging.getLogger(__name__)


class ImplementationShortfallAlgorithm:
    """
    Implementation Shortfall algorithm based on Almgren-Chriss framework

    Minimizes implementation shortfall = actual cost - decision price
    with explicit risk aversion parameter
    """

    def __init__(self, config: Dict = None):
        """Initialize IS algorithm"""
        self.config = config or {}
        self.risk_aversion = self.config.get('risk_aversion', 1.0)
        self.permanent_impact = self.config.get('permanent_impact_factor', 0.1)
        self.temporary_impact = self.config.get('temporary_impact_factor', 0.01)

    def generate_schedule(self, parent_order: Order, start_time: datetime,
                         end_time: datetime, volatility: float = 0.02) -> List[Dict]:
        """
        Generate optimal execution schedule using Almgren-Chriss

        Args:
            parent_order: Parent order
            start_time: Start time
            end_time: End time
            volatility: Price volatility

        Returns:
            Optimal execution schedule
        """
        logger.info(f"Generating IS schedule with risk aversion={self.risk_aversion}")

        total_duration = (end_time - start_time).total_seconds() / 3600  # hours
        num_periods = max(int(total_duration * 4), 5)  # 15-min intervals

        # Almgren-Chriss optimal trajectory
        trajectory = self._calculate_optimal_trajectory(
            parent_order.quantity,
            num_periods,
            volatility
        )

        schedule = []
        interval_seconds = total_duration * 3600 / num_periods

        for i in range(num_periods):
            execution_time = start_time + timedelta(seconds=i * interval_seconds)
            quantity = trajectory[i]

            schedule.append({
                'period': i + 1,
                'execution_time': execution_time,
                'quantity': round(quantity, 2),
                'status': 'pending'
            })

        logger.info(f"Generated IS schedule with {len(schedule)} periods")
        return schedule

    def _calculate_optimal_trajectory(self, total_quantity: float,
                                     num_periods: int, volatility: float) -> np.ndarray:
        """
        Calculate optimal execution trajectory

        Uses simplified Almgren-Chriss formulation
        """
        # Time decay parameter
        kappa = np.sqrt(self.risk_aversion * volatility**2 / self.temporary_impact)

        # Optimal trajectory (exponential decay)
        t = np.arange(num_periods)
        remaining = total_quantity * np.sinh(kappa * (num_periods - t)) / np.sinh(kappa * num_periods)

        # Convert to execution quantities
        quantities = np.diff(np.concatenate(([total_quantity], remaining)))

        return np.abs(quantities)

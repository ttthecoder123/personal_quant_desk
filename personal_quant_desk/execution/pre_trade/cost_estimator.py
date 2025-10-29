"""Pre-trade cost estimation"""
import numpy as np
from typing import Dict
class CostEstimator:
    def __init__(self, config: Dict = None):
        self.config = config or {}
    def estimate_cost(self, symbol: str, quantity: float, price: float) -> Dict:
        """Estimate total execution cost"""
        spread_cost = quantity * price * 0.0005  # 5 bps
        impact_cost = quantity * price * 0.001  # 10 bps for impact
        commission = quantity * 0.005
        return {
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'commission': commission,
            'total_cost': spread_cost + impact_cost + commission,
            'cost_bps': ((spread_cost + impact_cost + commission) / (quantity * price)) * 10000
        }

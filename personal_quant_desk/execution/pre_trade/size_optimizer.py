"""Order sizing optimization"""
from typing import Dict, List
class SizeOptimizer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
    def optimize_sizing(self, total_quantity: float, constraints: Dict) -> List[float]:
        return [total_quantity]

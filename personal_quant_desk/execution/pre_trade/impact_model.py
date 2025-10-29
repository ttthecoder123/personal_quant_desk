"""Market impact estimation models"""
import numpy as np
from typing import Dict
class ImpactModel:
    def __init__(self, config: Dict = None):
        self.config = config or {}
    def estimate_impact(self, quantity: float, adv: float, volatility: float = 0.02) -> Dict:
        """Estimate market impact using square-root model"""
        participation = quantity / adv if adv > 0 else 0
        temporary_impact = volatility * np.sqrt(participation) if participation > 0 else 0
        permanent_impact = volatility * participation * 0.1
        return {
            'temporary_impact_bps': temporary_impact * 10000,
            'permanent_impact_bps': permanent_impact * 10000,
            'total_impact_bps': (temporary_impact + permanent_impact) * 10000
        }

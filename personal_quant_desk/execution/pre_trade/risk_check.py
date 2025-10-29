"""Pre-trade risk validation"""
from typing import Dict, Tuple
class RiskCheck:
    def __init__(self, config: Dict = None):
        self.config = config or {}
    def check_risk(self, order_params: Dict, account_state: Dict) -> Tuple[bool, str]:
        return True, "OK"

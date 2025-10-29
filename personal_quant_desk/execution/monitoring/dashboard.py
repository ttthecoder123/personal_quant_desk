"""Execution dashboard"""
from typing import Dict
class ExecutionDashboard:
    def __init__(self, config: Dict = None):
        self.config = config or {}
    def get_dashboard(self) -> Dict:
        return {'status': 'running', 'orders': [], 'positions': {}}

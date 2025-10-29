"""Execution alert system"""
from typing import Dict, List, Callable
import logging
logger = logging.getLogger(__name__)
class AlertSystem:
    """Generate and manage execution alerts"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.alerts = []
        self.callbacks: List[Callable] = []
    def add_alert(self, alert_type: str, message: str, severity: str = 'INFO'):
        alert = {'type': alert_type, 'message': message, 'severity': severity}
        self.alerts.append(alert)
        logger.warning(f"Alert: {alert}")
        for callback in self.callbacks:
            callback(alert)
    def get_alerts(self) -> List[Dict]:
        return self.alerts
    def register_callback(self, callback: Callable):
        self.callbacks.append(callback)

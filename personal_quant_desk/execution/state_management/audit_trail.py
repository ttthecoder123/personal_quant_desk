"""Compliance audit trail"""
from typing import Dict, List
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
class AuditTrail:
    """Maintain compliance audit trail"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.trail: List[Dict] = []
    def log_event(self, event_type: str, event_data: Dict):
        """Log audit event"""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'data': event_data
        }
        self.trail.append(event)
        logger.info(f"Audit: {event_type}")
    def get_trail(self, start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """Get audit trail"""
        if not start_time and not end_time:
            return self.trail
        filtered = [e for e in self.trail
                   if (not start_time or e['timestamp'] >= start_time)
                   and (not end_time or e['timestamp'] <= end_time)]
        return filtered

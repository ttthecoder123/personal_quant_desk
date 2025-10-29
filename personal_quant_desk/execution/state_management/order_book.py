"""Internal order book management"""
from typing import Dict, List, Optional
from datetime import datetime
class OrderBook:
    """Manage internal order book"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.orders: Dict[str, Dict] = {}
    def add_order(self, order_id: str, order_data: Dict):
        self.orders[order_id] = order_data
    def get_order(self, order_id: str) -> Optional[Dict]:
        return self.orders.get(order_id)
    def get_active_orders(self) -> List[Dict]:
        return [o for o in self.orders.values() if o.get('status') in ['SUBMITTED', 'PARTIAL']]

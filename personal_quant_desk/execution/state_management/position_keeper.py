"""Position tracking and reconciliation"""
from typing import Dict
import logging
logger = logging.getLogger(__name__)
class PositionKeeper:
    """Track and reconcile positions across brokers"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.positions: Dict[str, float] = {}
    def update_position(self, symbol: str, quantity: float):
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        logger.info(f"Position updated: {symbol} = {self.positions[symbol]}")
    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0)
    def get_all_positions(self) -> Dict[str, float]:
        return self.positions.copy()
    def reconcile_positions(self, broker_positions: Dict[str, float]):
        """Reconcile with broker positions"""
        discrepancies = {}
        for symbol, broker_qty in broker_positions.items():
            our_qty = self.positions.get(symbol, 0)
            if abs(broker_qty - our_qty) > 0.01:
                discrepancies[symbol] = {'ours': our_qty, 'broker': broker_qty}
        if discrepancies:
            logger.warning(f"Position discrepancies: {discrepancies}")
        return discrepancies

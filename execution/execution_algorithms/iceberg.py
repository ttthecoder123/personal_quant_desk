"""Iceberg order execution with hidden quantity"""

from typing import Dict
from ..order_management.order_types import IcebergOrder
import logging

logger = logging.getLogger(__name__)


class IcebergExecutor:
    """Executes iceberg orders with hidden quantity"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def execute(self, order: IcebergOrder) -> Dict:
        """Execute iceberg order"""
        logger.info(f"Executing iceberg: display {order.display_quantity} of {order.quantity}")
        return {'status': 'executing'}

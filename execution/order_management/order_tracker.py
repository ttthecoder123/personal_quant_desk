"""
Order lifecycle tracking and monitoring

Tracks orders from submission to completion with:
- Real-time status updates
- Fill notifications
- Performance metrics
- Stuck order detection
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from ..order_management.order_types import Order, OrderStatus, Fill
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class OrderTracker:
    """
    Tracks order lifecycle and performance

    Monitors:
    - Order status changes
    - Fill events
    - Order aging
    - Performance metrics
    """

    def __init__(self, config: Dict = None):
        """
        Initialize order tracker

        Args:
            config: Tracker configuration
        """
        self.config = config or {}

        # Order storage
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []

        # Status callbacks
        self.status_callbacks: List[Callable] = []
        self.fill_callbacks: List[Callable] = []

        # Performance metrics
        self.metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_fills': 0,
            'total_value_traded': 0.0
        }

        # Alert thresholds
        self.stuck_order_threshold = self.config.get('stuck_order_threshold_minutes', 30)
        self.partial_fill_threshold = self.config.get('partial_fill_threshold_minutes', 15)

    def track_order(self, order: Order):
        """
        Start tracking an order

        Args:
            order: Order to track
        """
        logger.info(f"Tracking order {order.order_id} - {order.symbol} {order.side.value} {order.quantity}")

        self.active_orders[order.order_id] = order
        self.metrics['total_orders'] += 1

        # Trigger callbacks
        for callback in self.status_callbacks:
            try:
                callback(order, None, order.status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def update_order_status(self, order_id: str, new_status: OrderStatus,
                           message: Optional[str] = None):
        """
        Update order status

        Args:
            order_id: Order ID
            new_status: New order status
            message: Optional status message
        """
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return

        order = self.active_orders[order_id]
        old_status = order.status

        order.update_status(new_status, message)
        logger.info(f"Order {order_id} status: {old_status.value} -> {new_status.value}")

        # Update metrics
        if new_status == OrderStatus.FILLED:
            self.metrics['filled_orders'] += 1
        elif new_status == OrderStatus.CANCELLED:
            self.metrics['cancelled_orders'] += 1
        elif new_status == OrderStatus.REJECTED:
            self.metrics['rejected_orders'] += 1

        # Move to completed if terminal state
        if order.is_complete:
            self._complete_order(order)

        # Trigger callbacks
        for callback in self.status_callbacks:
            try:
                callback(order, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def add_fill(self, order_id: str, fill: Fill):
        """
        Add a fill to an order

        Args:
            order_id: Order ID
            fill: Fill information
        """
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return

        order = self.active_orders[order_id]
        order.add_fill(fill)

        logger.info(
            f"Fill for order {order_id}: {fill.quantity} @ {fill.price} "
            f"({order.filled_quantity}/{order.quantity} filled)"
        )

        # Update metrics
        self.metrics['total_fills'] += 1
        self.metrics['total_value_traded'] += fill.total_value

        # Check if complete
        if order.is_complete:
            self._complete_order(order)

        # Trigger callbacks
        for callback in self.fill_callbacks:
            try:
                callback(order, fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.active_orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol"""
        orders = list(self.active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders by status"""
        return [o for o in self.active_orders.values() if o.status == status]

    def cancel_order(self, order_id: str, reason: str = "User cancelled"):
        """
        Cancel an order

        Args:
            order_id: Order ID
            reason: Cancellation reason
        """
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found")
            return

        order = self.active_orders[order_id]
        if order.is_complete:
            logger.warning(f"Order {order_id} already complete")
            return

        self.update_order_status(order_id, OrderStatus.PENDING_CANCEL, reason)

    def _complete_order(self, order: Order):
        """Move order from active to completed"""
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
            self.completed_orders.append(order)

            # Keep only recent completed orders
            if len(self.completed_orders) > 10000:
                self.completed_orders = self.completed_orders[-5000:]

    def check_stuck_orders(self) -> List[Order]:
        """
        Check for stuck orders

        Returns:
            List of potentially stuck orders
        """
        stuck_orders = []
        now = datetime.now()

        for order in self.active_orders.values():
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.PENDING_SUBMIT]:
                age_minutes = (now - order.create_time).total_seconds() / 60

                if age_minutes > self.stuck_order_threshold:
                    logger.warning(
                        f"Stuck order detected: {order.order_id} "
                        f"({order.symbol}, age: {age_minutes:.1f} min)"
                    )
                    stuck_orders.append(order)

            elif order.status == OrderStatus.PARTIALLY_FILLED:
                age_minutes = (now - order.last_update_time).total_seconds() / 60

                if age_minutes > self.partial_fill_threshold:
                    logger.warning(
                        f"Stalled partial fill: {order.order_id} "
                        f"({order.symbol}, {order.fill_percentage:.1f}% filled)"
                    )
                    stuck_orders.append(order)

        return stuck_orders

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        metrics = self.metrics.copy()

        # Calculate fill rate
        if metrics['total_orders'] > 0:
            metrics['fill_rate'] = metrics['filled_orders'] / metrics['total_orders']
            metrics['rejection_rate'] = metrics['rejected_orders'] / metrics['total_orders']
        else:
            metrics['fill_rate'] = 0.0
            metrics['rejection_rate'] = 0.0

        # Active orders count
        metrics['active_orders'] = len(self.active_orders)

        return metrics

    def get_symbol_metrics(self, symbol: str) -> Dict:
        """Get metrics for a specific symbol"""
        symbol_orders = [
            o for o in self.completed_orders + list(self.active_orders.values())
            if o.symbol == symbol
        ]

        metrics = {
            'total_orders': len(symbol_orders),
            'filled_orders': len([o for o in symbol_orders if o.status == OrderStatus.FILLED]),
            'total_quantity': sum(o.quantity for o in symbol_orders),
            'filled_quantity': sum(o.filled_quantity for o in symbol_orders),
            'total_value': sum(o.average_fill_price * o.filled_quantity for o in symbol_orders)
        }

        return metrics

    def register_status_callback(self, callback: Callable):
        """
        Register callback for status updates

        Callback signature: callback(order, old_status, new_status)
        """
        self.status_callbacks.append(callback)

    def register_fill_callback(self, callback: Callable):
        """
        Register callback for fills

        Callback signature: callback(order, fill)
        """
        self.fill_callbacks.append(callback)

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history"""
        orders = self.completed_orders[-limit:]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def cleanup_old_orders(self, days: int = 7):
        """Remove old completed orders"""
        cutoff = datetime.now() - timedelta(days=days)
        self.completed_orders = [
            o for o in self.completed_orders
            if o.last_update_time and o.last_update_time > cutoff
        ]
        logger.info(f"Cleaned up old orders, {len(self.completed_orders)} remaining")

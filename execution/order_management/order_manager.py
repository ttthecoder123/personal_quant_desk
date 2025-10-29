"""
Core Order Management System

Orchestrates:
- Order submission and lifecycle
- Validation
- Routing
- Tracking
- Parent-child relationships
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from ..order_management.order_types import (
    Order, OrderStatus, OrderType, Fill, TWAPOrder, VWAPOrder
)
from ..order_management.order_validator import OrderValidator, ValidationResult
from ..order_management.order_router import OrderRouter, RoutingDecision
from ..order_management.order_tracker import OrderTracker
import logging
import uuid

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Core Order Management System

    Central coordinator for:
    - Order submission and validation
    - Order routing
    - Lifecycle management
    - Parent-child order relationships
    - Order modification and cancellation
    """

    def __init__(self, config: Dict = None):
        """
        Initialize order manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize subsystems
        self.validator = OrderValidator(self.config.get('validation', {}))
        self.router = OrderRouter(self.config.get('routing', {}))
        self.tracker = OrderTracker(self.config.get('tracking', {}))

        # Parent-child relationships
        self.parent_orders: Dict[str, Order] = {}
        self.child_orders: Dict[str, List[Order]] = {}

        # Broker connectors (to be injected)
        self.broker_connectors = {}

        logger.info("Order Manager initialized")

    def submit_order(self, order: Order, account_state: Dict = None,
                    validate: bool = True) -> Tuple[bool, str]:
        """
        Submit an order

        Args:
            order: Order to submit
            account_state: Current account state
            validate: Whether to validate order

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Submitting order {order.order_id}: {order.symbol} {order.side.value} {order.quantity}")

        # Validate order
        if validate:
            validation = self.validator.validate_order(order, account_state)
            if not validation.valid:
                logger.error(f"Order validation failed: {validation}")
                order.update_status(OrderStatus.REJECTED, str(validation))
                return False, f"Validation failed: {validation}"

            if validation.warnings:
                logger.warning(f"Order warnings: {', '.join(validation.warnings)}")

        # Route order
        routing = self.router.route_order(order, account_state)
        logger.info(f"Routing decision: {routing}")

        order.broker = routing.broker
        order.venue = routing.venue

        # Handle algo orders (split into child orders)
        if order.order_type in [OrderType.TWAP, OrderType.VWAP, OrderType.IMPLEMENTATION_SHORTFALL]:
            return self._submit_algo_order(order)

        # Submit to broker
        success, message = self._send_to_broker(order)

        if success:
            # Track order
            self.tracker.track_order(order)
            self.validator.record_order(order)
            logger.info(f"Order {order.order_id} submitted successfully")
        else:
            order.update_status(OrderStatus.REJECTED, message)
            logger.error(f"Order {order.order_id} rejected: {message}")

        return success, message

    def modify_order(self, order_id: str, modifications: Dict) -> Tuple[bool, str]:
        """
        Modify an existing order

        Args:
            order_id: Order ID to modify
            modifications: Dict of fields to modify

        Returns:
            Tuple of (success, message)
        """
        order = self.tracker.get_order(order_id)
        if not order:
            return False, f"Order {order_id} not found"

        if order.is_complete:
            return False, f"Order {order_id} is complete and cannot be modified"

        logger.info(f"Modifying order {order_id}: {modifications}")

        # Apply modifications
        old_order = order.to_dict()

        for field, value in modifications.items():
            if hasattr(order, field):
                setattr(order, field, value)

        # Validate modified order
        validation = self.validator.validate_order(order)
        if not validation.valid:
            # Revert changes
            for field, value in old_order.items():
                if hasattr(order, field):
                    setattr(order, field, value)
            return False, f"Modified order validation failed: {validation}"

        # Send modification to broker
        if order.broker and order.broker in self.broker_connectors:
            connector = self.broker_connectors[order.broker]
            success, message = connector.modify_order(order)

            if success:
                logger.info(f"Order {order_id} modified successfully")
                return True, "Order modified"
            else:
                # Revert on failure
                for field, value in old_order.items():
                    if hasattr(order, field):
                        setattr(order, field, value)
                return False, message

        return False, "Broker connector not available"

    def cancel_order(self, order_id: str, reason: str = "User cancelled") -> Tuple[bool, str]:
        """
        Cancel an order

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            Tuple of (success, message)
        """
        order = self.tracker.get_order(order_id)
        if not order:
            return False, f"Order {order_id} not found"

        if order.is_complete:
            return False, f"Order {order_id} is already complete"

        logger.info(f"Cancelling order {order_id}: {reason}")

        # Update status
        self.tracker.update_order_status(order_id, OrderStatus.PENDING_CANCEL, reason)

        # Send cancellation to broker
        if order.broker and order.broker in self.broker_connectors:
            connector = self.broker_connectors[order.broker]
            success, message = connector.cancel_order(order)

            if success:
                self.tracker.update_order_status(order_id, OrderStatus.CANCELLED, reason)
                logger.info(f"Order {order_id} cancelled successfully")
                return True, "Order cancelled"
            else:
                return False, message

        return False, "Broker connector not available"

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Tuple[bool, str]]:
        """
        Cancel all active orders

        Args:
            symbol: Optional symbol filter

        Returns:
            Dict of order_id -> (success, message)
        """
        orders = self.tracker.get_active_orders(symbol)
        results = {}

        logger.info(f"Cancelling {len(orders)} orders" + (f" for {symbol}" if symbol else ""))

        for order in orders:
            success, message = self.cancel_order(order.order_id, "Bulk cancellation")
            results[order.order_id] = (success, message)

        return results

    def handle_fill(self, order_id: str, fill: Fill):
        """
        Handle a fill notification from broker

        Args:
            order_id: Order ID
            fill: Fill information
        """
        logger.info(f"Handling fill for order {order_id}: {fill.quantity} @ {fill.price}")

        self.tracker.add_fill(order_id, fill)

        # Check if parent order completed
        order = self.tracker.get_order(order_id)
        if order and order.parent_order_id:
            self._check_parent_completion(order.parent_order_id)

    def handle_status_update(self, order_id: str, status: OrderStatus, message: str = ""):
        """
        Handle status update from broker

        Args:
            order_id: Order ID
            status: New status
            message: Status message
        """
        logger.info(f"Status update for order {order_id}: {status.value}")
        self.tracker.update_order_status(order_id, status, message)

    def _submit_algo_order(self, order: Order) -> Tuple[bool, str]:
        """
        Submit an algorithmic order by splitting into child orders

        Args:
            order: Algo order to submit

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Submitting algo order {order.order_id}: {order.order_type.value}")

        # Store as parent order
        self.parent_orders[order.order_id] = order
        self.child_orders[order.order_id] = []

        # Mark as submitted
        order.update_status(OrderStatus.SUBMITTED)
        self.tracker.track_order(order)

        # Child orders will be created and submitted by execution algorithms
        # This is handled by the execution engine

        return True, f"Algo order {order.order_type.value} accepted"

    def _send_to_broker(self, order: Order) -> Tuple[bool, str]:
        """
        Send order to broker

        Args:
            order: Order to send

        Returns:
            Tuple of (success, message)
        """
        if not order.broker:
            return False, "No broker assigned"

        if order.broker not in self.broker_connectors:
            return False, f"Broker connector {order.broker} not available"

        connector = self.broker_connectors[order.broker]

        try:
            order.update_status(OrderStatus.PENDING_SUBMIT)
            success, message = connector.submit_order(order)

            if success:
                order.update_status(OrderStatus.SUBMITTED)

            return success, message

        except Exception as e:
            logger.error(f"Error submitting to broker: {e}")
            return False, str(e)

    def _check_parent_completion(self, parent_order_id: str):
        """Check if parent order is complete"""
        if parent_order_id not in self.child_orders:
            return

        children = self.child_orders[parent_order_id]
        all_complete = all(child.is_complete for child in children)

        if all_complete:
            parent = self.parent_orders[parent_order_id]

            # Aggregate fills
            total_filled = sum(child.filled_quantity for child in children)
            parent.filled_quantity = total_filled

            if total_filled >= parent.quantity:
                parent.update_status(OrderStatus.FILLED)
                logger.info(f"Parent order {parent_order_id} completed")

    def add_broker_connector(self, name: str, connector):
        """Add a broker connector"""
        self.broker_connectors[name] = connector
        logger.info(f"Added broker connector: {name}")

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.tracker.get_order(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders"""
        return self.tracker.get_active_orders(symbol)

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history"""
        return self.tracker.get_order_history(symbol, limit)

    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.tracker.get_performance_metrics()

    def check_system_health(self) -> Dict:
        """Check OMS health"""
        stuck_orders = self.tracker.check_stuck_orders()

        health = {
            'active_orders': len(self.tracker.active_orders),
            'stuck_orders': len(stuck_orders),
            'metrics': self.get_metrics(),
            'broker_status': {
                name: connector.is_connected()
                for name, connector in self.broker_connectors.items()
            }
        }

        return health

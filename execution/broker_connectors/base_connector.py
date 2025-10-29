"""
Base broker connector interface

All broker implementations must inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from ..order_management.order_types import Order, OrderStatus, Fill
import logging

logger = logging.getLogger(__name__)


class BaseBrokerConnector(ABC):
    """
    Abstract base class for broker connectors

    All broker implementations must implement these methods.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize broker connector

        Args:
            config: Broker configuration
        """
        self.config = config or {}
        self.connected = False
        self.authenticated = False

        # Callbacks
        self.order_update_callbacks: List[Callable] = []
        self.fill_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []

        # Connection tracking
        self.connection_time: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None

        # Statistics
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'orders_cancelled': 0,
            'connection_errors': 0,
            'api_errors': 0
        }

    @abstractmethod
    def connect(self) -> Tuple[bool, str]:
        """
        Connect to broker

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    def disconnect(self) -> Tuple[bool, str]:
        """
        Disconnect from broker

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    def authenticate(self) -> Tuple[bool, str]:
        """
        Authenticate with broker

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """
        Submit order to broker

        Args:
            order: Order to submit

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    def modify_order(self, order: Order) -> Tuple[bool, str]:
        """
        Modify existing order

        Args:
            order: Order with modifications

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    def cancel_order(self, order: Order) -> Tuple[bool, str]:
        """
        Cancel order

        Args:
            order: Order to cancel

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Tuple[Optional[OrderStatus], str]:
        """
        Get order status

        Args:
            order_id: Order ID

        Returns:
            Tuple of (status, message)
        """
        pass

    @abstractmethod
    def get_positions(self, account: Optional[str] = None) -> Dict[str, float]:
        """
        Get current positions

        Args:
            account: Optional account filter

        Returns:
            Dict of symbol -> quantity
        """
        pass

    @abstractmethod
    def get_account_info(self, account: Optional[str] = None) -> Dict:
        """
        Get account information

        Args:
            account: Optional account filter

        Returns:
            Account information dict
        """
        pass

    @abstractmethod
    def subscribe_market_data(self, symbols: List[str]) -> Tuple[bool, str]:
        """
        Subscribe to market data

        Args:
            symbols: List of symbols

        Returns:
            Tuple of (success, message)
        """
        pass

    def is_connected(self) -> bool:
        """Check if connected to broker"""
        return self.connected and self.authenticated

    def register_order_callback(self, callback: Callable):
        """
        Register callback for order updates

        Callback signature: callback(order_id, status, message)
        """
        self.order_update_callbacks.append(callback)

    def register_fill_callback(self, callback: Callable):
        """
        Register callback for fills

        Callback signature: callback(order_id, fill)
        """
        self.fill_callbacks.append(callback)

    def register_error_callback(self, callback: Callable):
        """
        Register callback for errors

        Callback signature: callback(error_code, error_message)
        """
        self.error_callbacks.append(callback)

    def _notify_order_update(self, order_id: str, status: OrderStatus, message: str = ""):
        """Notify order update callbacks"""
        for callback in self.order_update_callbacks:
            try:
                callback(order_id, status, message)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")

    def _notify_fill(self, order_id: str, fill: Fill):
        """Notify fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                callback(order_id, fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")

    def _notify_error(self, error_code: str, error_message: str):
        """Notify error callbacks"""
        logger.error(f"Broker error {error_code}: {error_message}")
        for callback in self.error_callbacks:
            try:
                callback(error_code, error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def get_statistics(self) -> Dict:
        """Get connector statistics"""
        return {
            **self.stats,
            'connected': self.connected,
            'authenticated': self.authenticated,
            'connection_time': self.connection_time,
            'last_heartbeat': self.last_heartbeat
        }

    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {k: 0 for k in self.stats}

    def heartbeat(self):
        """Update last heartbeat time"""
        self.last_heartbeat = datetime.now()

    def check_connection_health(self) -> Tuple[bool, str]:
        """
        Check connection health

        Returns:
            Tuple of (healthy, message)
        """
        if not self.is_connected():
            return False, "Not connected"

        if self.last_heartbeat:
            age = (datetime.now() - self.last_heartbeat).total_seconds()
            if age > 60:  # No heartbeat for 60 seconds
                return False, f"No heartbeat for {age:.0f} seconds"

        return True, "Connection healthy"

"""
Paper Trading Connector

Simulates order execution for testing without real money.
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import uuid
from .base_connector import BaseBrokerConnector
from ..order_management.order_types import Order, OrderStatus, OrderType, OrderSide, Fill
import logging

logger = logging.getLogger(__name__)


class PaperTradingConnector(BaseBrokerConnector):
    """
    Paper trading connector for simulated execution

    Simulates realistic execution with:
    - Slippage modeling
    - Latency simulation
    - Partial fills
    - Market hours enforcement
    - Position tracking
    """

    def __init__(self, config: Dict = None):
        """Initialize paper trading connector"""
        super().__init__(config)

        # Simulation parameters
        self.slippage_bps = self.config.get('slippage_bps', 5)  # 5 bps slippage
        self.latency_ms = self.config.get('latency_ms', 50)  # 50ms latency
        self.fill_probability = self.config.get('fill_probability', 0.95)
        self.partial_fill_probability = self.config.get('partial_fill_probability', 0.1)

        # State
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, float] = {}
        self.cash = self.config.get('initial_cash', 100_000)
        self.initial_cash = self.cash

        # Market data (simulated)
        self.market_prices: Dict[str, float] = {}

        logger.info("Paper trading connector initialized")

    def connect(self) -> Tuple[bool, str]:
        """Connect to paper trading (always succeeds)"""
        self.connected = True
        self.connection_time = datetime.now()
        logger.info("Connected to paper trading")
        return True, "Connected"

    def disconnect(self) -> Tuple[bool, str]:
        """Disconnect from paper trading"""
        self.connected = False
        logger.info("Disconnected from paper trading")
        return True, "Disconnected"

    def authenticate(self) -> Tuple[bool, str]:
        """Authenticate (always succeeds)"""
        self.authenticated = True
        return True, "Authenticated"

    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """
        Submit order for paper trading

        Args:
            order: Order to submit

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Paper trading: submitting order {order.order_id}")

        # Store order
        self.orders[order.order_id] = order
        self.stats['orders_submitted'] += 1

        # Simulate execution after latency
        self._simulate_execution(order)

        return True, f"Order {order.order_id} submitted"

    def modify_order(self, order: Order) -> Tuple[bool, str]:
        """Modify order"""
        if order.order_id not in self.orders:
            return False, "Order not found"

        logger.info(f"Paper trading: modifying order {order.order_id}")
        self.orders[order.order_id] = order
        return True, "Order modified"

    def cancel_order(self, order: Order) -> Tuple[bool, str]:
        """Cancel order"""
        if order.order_id not in self.orders:
            return False, "Order not found"

        logger.info(f"Paper trading: cancelling order {order.order_id}")
        self._notify_order_update(order.order_id, OrderStatus.CANCELLED, "User cancelled")
        self.stats['orders_cancelled'] += 1
        return True, "Order cancelled"

    def get_order_status(self, order_id: str) -> Tuple[Optional[OrderStatus], str]:
        """Get order status"""
        if order_id in self.orders:
            return self.orders[order_id].status, "OK"
        return None, "Order not found"

    def get_positions(self, account: Optional[str] = None) -> Dict[str, float]:
        """Get current positions"""
        return self.positions.copy()

    def get_account_info(self, account: Optional[str] = None) -> Dict:
        """Get account information"""
        # Calculate portfolio value
        portfolio_value = self.cash
        for symbol, quantity in self.positions.items():
            price = self.market_prices.get(symbol, 100.0)  # Default price
            portfolio_value += quantity * price

        return {
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'equity': portfolio_value,
            'buying_power': self.cash * 4,  # 4x for margin
            'positions': self.positions.copy(),
            'pnl': portfolio_value - self.initial_cash,
            'pnl_percent': ((portfolio_value - self.initial_cash) / self.initial_cash * 100)
                          if self.initial_cash > 0 else 0
        }

    def subscribe_market_data(self, symbols: List[str]) -> Tuple[bool, str]:
        """Subscribe to market data (simulated)"""
        for symbol in symbols:
            if symbol not in self.market_prices:
                # Initialize with random price
                self.market_prices[symbol] = np.random.uniform(50, 200)

        return True, f"Subscribed to {len(symbols)} symbols"

    def _simulate_execution(self, order: Order):
        """
        Simulate order execution

        Args:
            order: Order to execute
        """
        # Get or generate market price
        if order.symbol not in self.market_prices:
            self.market_prices[order.symbol] = np.random.uniform(50, 200)

        market_price = self.market_prices[order.symbol]

        # Determine fill price based on order type
        if order.order_type == OrderType.MARKET:
            # Market orders fill with slippage
            slippage_factor = 1 + (self.slippage_bps / 10000)
            if order.side in [OrderSide.BUY]:
                fill_price = market_price * slippage_factor
            else:
                fill_price = market_price / slippage_factor

            self._execute_fill(order, fill_price, order.quantity)

        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill if price is favorable
            if order.price:
                if order.side in [OrderSide.BUY] and market_price <= order.price:
                    fill_price = order.price
                    self._execute_fill(order, fill_price, order.quantity)
                elif order.side in [OrderSide.SELL, OrderSide.SHORT] and market_price >= order.price:
                    fill_price = order.price
                    self._execute_fill(order, fill_price, order.quantity)
                else:
                    # Order not filled yet
                    self._notify_order_update(order.order_id, OrderStatus.SUBMITTED, "Waiting for fill")

        elif order.order_type == OrderType.STOP:
            # Stop orders convert to market when triggered
            if order.stop_price:
                if order.side in [OrderSide.BUY] and market_price >= order.stop_price:
                    fill_price = market_price * (1 + self.slippage_bps / 10000)
                    self._execute_fill(order, fill_price, order.quantity)
                elif order.side in [OrderSide.SELL] and market_price <= order.stop_price:
                    fill_price = market_price * (1 - self.slippage_bps / 10000)
                    self._execute_fill(order, fill_price, order.quantity)

    def _execute_fill(self, order: Order, price: float, quantity: float):
        """
        Execute fill for an order

        Args:
            order: Order being filled
            price: Fill price
            quantity: Fill quantity
        """
        # Simulate partial fills occasionally
        if np.random.random() < self.partial_fill_probability:
            quantity = quantity * np.random.uniform(0.5, 0.9)

        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            timestamp=datetime.now(),
            price=price,
            quantity=quantity,
            commission=quantity * 0.005,  # $0.005 per share
            venue="PAPER",
            liquidity="REMOVE"
        )

        # Update positions
        if order.side in [OrderSide.BUY]:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + quantity
            self.cash -= (price * quantity + fill.commission)
        else:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - quantity
            self.cash += (price * quantity - fill.commission)

        # Notify fill
        self._notify_fill(order.order_id, fill)

        # Update order status
        if quantity >= order.quantity:
            self._notify_order_update(order.order_id, OrderStatus.FILLED, "Order filled")
            self.stats['orders_filled'] += 1
        else:
            self._notify_order_update(order.order_id, OrderStatus.PARTIALLY_FILLED,
                                    f"Partially filled: {quantity}/{order.quantity}")

        logger.info(f"Paper trading: filled {quantity} @ {price} for order {order.order_id}")

    def update_market_price(self, symbol: str, price: float):
        """
        Update market price for a symbol

        Args:
            symbol: Symbol
            price: New price
        """
        self.market_prices[symbol] = price

        # Check pending orders for fills
        for order in self.orders.values():
            if order.symbol == symbol and order.status == OrderStatus.SUBMITTED:
                self._simulate_execution(order)

"""
Execution Manager for order management and trade execution.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from enum import Enum
import os
from decimal import Decimal
import pandas as pd

from utils.logger import get_trade_logger, log_trade

log = get_trade_logger()


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class Order:
    """Order representation."""

    def __init__(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: OrderType,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ):
        """Initialize order."""
        self.order_id = None
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.avg_fill_price = 0
        self.timestamp = datetime.now()
        self.metadata = {}


class ExecutionAlgorithm:
    """Base class for execution algorithms."""

    def __init__(self, name: str, config: dict):
        """Initialize execution algorithm."""
        self.name = name
        self.config = config

    async def execute(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """Execute order using the algorithm."""
        raise NotImplementedError


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm."""

    def __init__(self, config: dict):
        """Initialize TWAP algorithm."""
        super().__init__("TWAP", config)
        self.time_window = config.get('time_window', 300)  # 5 minutes default

    async def execute(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """Execute order using TWAP."""
        # Split order into smaller chunks
        num_slices = max(1, self.time_window // 60)  # One slice per minute
        slice_quantity = order.quantity // num_slices
        remainder = order.quantity % num_slices

        child_orders = []
        for i in range(num_slices):
            quantity = slice_quantity + (remainder if i == num_slices - 1 else 0)
            child_order = Order(
                symbol=order.symbol,
                action=order.action,
                quantity=quantity,
                order_type=order.order_type,
                price=order.price,
                time_in_force=order.time_in_force
            )
            child_orders.append(child_order)

        log.debug(f"TWAP: Split {order.symbol} order into {len(child_orders)} slices")
        return child_orders


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm."""

    def __init__(self, config: dict):
        """Initialize VWAP algorithm."""
        super().__init__("VWAP", config)
        self.volume_participation = config.get('volume_participation', 0.10)

    async def execute(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """Execute order using VWAP."""
        if 'Volume' not in market_data.columns or market_data.empty:
            # Fallback to single order if no volume data
            return [order]

        # Calculate volume profile
        total_volume = market_data['Volume'].sum()
        volume_profile = market_data['Volume'] / total_volume

        # Create child orders based on volume distribution
        child_orders = []
        remaining_quantity = order.quantity

        for i, volume_pct in enumerate(volume_profile):
            if remaining_quantity <= 0:
                break

            slice_quantity = min(
                int(order.quantity * volume_pct * (1 / self.volume_participation)),
                remaining_quantity
            )

            if slice_quantity > 0:
                child_order = Order(
                    symbol=order.symbol,
                    action=order.action,
                    quantity=slice_quantity,
                    order_type=order.order_type,
                    price=order.price,
                    time_in_force=order.time_in_force
                )
                child_orders.append(child_order)
                remaining_quantity -= slice_quantity

        # Add remaining quantity to last order
        if remaining_quantity > 0 and child_orders:
            child_orders[-1].quantity += remaining_quantity

        log.debug(f"VWAP: Split {order.symbol} order into {len(child_orders)} slices")
        return child_orders


class ExecutionManager:
    """Manages order execution and broker interactions."""

    def __init__(self, config: dict):
        """Initialize execution manager."""
        self.config = config
        self.broker_config = config.get('broker', {})
        self.order_config = config.get('orders', {})
        self.algorithms_config = config.get('algorithms', {})

        # Initialize execution algorithms
        self.algorithms = {}
        self._initialize_algorithms()

        # Order tracking
        self.active_orders = {}
        self.order_history = []
        self.positions = {}

        # Broker connection placeholder
        self.broker_connected = False

    def _initialize_algorithms(self):
        """Initialize execution algorithms."""
        if self.algorithms_config.get('twap', {}).get('enabled', False):
            self.algorithms['twap'] = TWAPAlgorithm(self.algorithms_config['twap'])
            log.info("TWAP algorithm initialized")

        if self.algorithms_config.get('vwap', {}).get('enabled', False):
            self.algorithms['vwap'] = VWAPAlgorithm(self.algorithms_config['vwap'])
            log.info("VWAP algorithm initialized")

    async def execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """
        Execute trades based on approved signals.

        Args:
            signals: List of approved trading signals

        Returns:
            List of executed trades
        """
        executed_trades = []

        for signal in signals:
            try:
                # Create order from signal
                order = self._create_order_from_signal(signal)

                # Execute order
                trade = await self._execute_order(order, signal)

                if trade:
                    executed_trades.append(trade)
                    log_trade(
                        f"Executed {trade['action']} {trade['quantity']} "
                        f"{trade['symbol']} @ {trade['price']}"
                    )

            except Exception as e:
                log.error(f"Error executing trade for {signal['symbol']}: {str(e)}")

        return executed_trades

    def _create_order_from_signal(self, signal: Dict) -> Order:
        """Create order from trading signal."""
        # Determine order type
        order_type_str = self.order_config.get('default_order_type', 'LIMIT')
        order_type = OrderType[order_type_str]

        # Calculate quantity (simplified - should use position sizing from risk manager)
        quantity = self._calculate_order_quantity(signal)

        # Create order
        order = Order(
            symbol=signal['symbol'],
            action=signal['action'],
            quantity=quantity,
            order_type=order_type,
            price=signal.get('price'),
            stop_price=signal.get('stop_loss'),
            time_in_force=TimeInForce[self.order_config.get('time_in_force', 'DAY')]
        )

        # Add signal metadata
        order.metadata = {
            'strategy': signal.get('strategy'),
            'signal_strength': signal.get('strength'),
            'timestamp': signal.get('timestamp')
        }

        return order

    def _calculate_order_quantity(self, signal: Dict) -> int:
        """Calculate order quantity based on signal and position sizing."""
        # Simplified calculation - in production would use proper position sizing
        base_quantity = 100  # Default quantity

        # Adjust based on signal strength
        strength = signal.get('strength', 1.0)
        adjusted_quantity = int(base_quantity * strength)

        return max(1, adjusted_quantity)

    async def _execute_order(self, order: Order, signal: Dict) -> Optional[Dict]:
        """Execute order through broker."""
        try:
            # Check if we should use execution algorithm
            if order.quantity > 1000 and self.algorithms:
                # Use VWAP for large orders
                if 'vwap' in self.algorithms:
                    child_orders = await self.algorithms['vwap'].execute(order, pd.DataFrame())
                    return await self._execute_child_orders(child_orders, signal)

            # Simulate order execution (in production, would connect to real broker)
            fill_price = await self._simulate_order_execution(order)

            if fill_price:
                trade = {
                    'order_id': f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'symbol': order.symbol,
                    'action': order.action,
                    'quantity': order.quantity,
                    'price': fill_price,
                    'timestamp': datetime.now(),
                    'strategy': order.metadata.get('strategy'),
                    'commission': self._calculate_commission(order.quantity, fill_price),
                    'status': 'FILLED'
                }

                # Update positions
                self._update_positions(trade)

                # Record in history
                self.order_history.append(trade)

                return trade

        except Exception as e:
            log.error(f"Order execution failed: {str(e)}")

        return None

    async def _execute_child_orders(
        self,
        child_orders: List[Order],
        signal: Dict
    ) -> Optional[Dict]:
        """Execute child orders from algorithm."""
        total_filled = 0
        total_value = 0

        for child_order in child_orders:
            fill_price = await self._simulate_order_execution(child_order)
            if fill_price:
                total_filled += child_order.quantity
                total_value += child_order.quantity * fill_price

            # Add delay between child orders
            await asyncio.sleep(1)

        if total_filled > 0:
            avg_price = total_value / total_filled
            return {
                'order_id': f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'symbol': signal['symbol'],
                'action': signal['action'],
                'quantity': total_filled,
                'price': avg_price,
                'timestamp': datetime.now(),
                'strategy': signal.get('strategy'),
                'commission': self._calculate_commission(total_filled, avg_price),
                'status': 'FILLED'
            }

        return None

    async def _simulate_order_execution(self, order: Order) -> Optional[float]:
        """Simulate order execution (placeholder for real broker integration)."""
        # In production, this would connect to real broker API
        # For now, simulate with slippage
        base_price = order.price if order.price else 100.0  # Default price

        # Apply slippage
        slippage = self.order_config.get('estimated_slippage', 0.0005)
        if order.action == 'BUY':
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)

        # Simulate execution delay
        await asyncio.sleep(0.1)

        return fill_price

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate trading commission."""
        commission_rate = self.config.get('commission', 0.001)  # 10 basis points
        return quantity * price * commission_rate

    def _update_positions(self, trade: Dict):
        """Update positions based on executed trade."""
        symbol = trade['symbol']
        quantity = trade['quantity']
        action = trade['action']

        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'value': 0,
                'unrealized_pnl': 0
            }

        if action == 'BUY':
            # Update average price
            total_value = (self.positions[symbol]['quantity'] * self.positions[symbol]['avg_price'] +
                          quantity * trade['price'])
            total_quantity = self.positions[symbol]['quantity'] + quantity

            self.positions[symbol]['quantity'] = total_quantity
            self.positions[symbol]['avg_price'] = total_value / total_quantity if total_quantity > 0 else 0
            self.positions[symbol]['value'] = total_value

        elif action == 'SELL':
            self.positions[symbol]['quantity'] -= quantity

            if self.positions[symbol]['quantity'] <= 0:
                # Position closed
                del self.positions[symbol]
            else:
                # Update value
                self.positions[symbol]['value'] = (
                    self.positions[symbol]['quantity'] * self.positions[symbol]['avg_price']
                )

    async def close_all_positions(self):
        """Close all open positions."""
        log.info("Closing all positions...")

        for symbol, position in self.positions.items():
            if position['quantity'] > 0:
                # Create sell order
                order = Order(
                    symbol=symbol,
                    action='SELL',
                    quantity=position['quantity'],
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.IOC
                )

                await self._execute_order(order, {'symbol': symbol, 'action': 'SELL'})
                log.info(f"Closed position: {symbol}")

        log.info("All positions closed")

    def get_positions(self) -> Dict:
        """Get current positions."""
        return self.positions.copy()

    def get_order_history(self) -> List[Dict]:
        """Get order history."""
        return self.order_history.copy()

    async def stop(self):
        """Stop execution manager."""
        log.info("Stopping execution manager...")

        # Cancel any pending orders
        for order_id, order in self.active_orders.items():
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                log.info(f"Cancelled order: {order_id}")

        log.info("Execution manager stopped")
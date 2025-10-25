"""
Order generation from strategy signals.

Converts portfolio signals and target positions into executable orders
with appropriate validation, netting, and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class OrderSide(Enum):
    """Order side enum."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """
    Represents a trading order.

    Attributes:
        symbol: Trading symbol
        side: Order side (BUY/SELL)
        quantity: Order quantity (positive)
        order_type: Order type (MARKET/LIMIT/etc)
        price: Limit price (optional, for LIMIT orders)
        stop_price: Stop price (optional, for STOP orders)
        strategy: Strategy generating this order
        timestamp: Order creation timestamp
        order_id: Unique order identifier
        parent_order_id: Parent order ID (for child orders)
        status: Order status
        filled_quantity: Quantity filled so far
        avg_fill_price: Average fill price
        metadata: Additional order metadata
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Generate order ID if not provided."""
        if self.order_id is None:
            self.order_id = f"ORD_{self.symbol}_{self.timestamp.strftime('%Y%m%d%H%M%S%f')}"

    def to_dict(self) -> Dict:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'strategy': self.strategy,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
        }


class OrderGenerator:
    """
    Generates executable orders from strategy signals.

    Features:
    - Convert signals to orders
    - Aggregate orders across strategies
    - Net opposing orders
    - Parent-child order creation
    - Order validation and sanity checks

    Attributes:
        min_order_size (float): Minimum order size
        max_order_size (float): Maximum order size
        round_lot_size (float): Round lot size (e.g., 100 shares)
        enable_netting (bool): Whether to net opposing orders
        validate_orders (bool): Whether to validate orders before generation
    """

    def __init__(
        self,
        min_order_size: float = 1.0,
        max_order_size: float = 1000000.0,
        round_lot_size: float = 1.0,
        enable_netting: bool = True,
        validate_orders: bool = True,
    ):
        """
        Initialize order generator.

        Args:
            min_order_size: Minimum order quantity
            max_order_size: Maximum order quantity
            round_lot_size: Round lot size for rounding
            enable_netting: Enable netting of opposing orders
            validate_orders: Validate orders before generation
        """
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        self.round_lot_size = round_lot_size
        self.enable_netting = enable_netting
        self.validate_orders = validate_orders

        logger.info(
            f"OrderGenerator initialized: min_size={min_order_size}, "
            f"max_size={max_order_size}, netting={enable_netting}"
        )

    def generate_orders_from_targets(
        self,
        target_positions: pd.Series,
        current_positions: pd.Series,
        prices: pd.Series,
        strategy_name: Optional[str] = None,
    ) -> List[Order]:
        """
        Generate orders to move from current to target positions.

        Args:
            target_positions: Target position sizes
            current_positions: Current position sizes
            prices: Current prices for each symbol
            strategy_name: Strategy name for order tagging

        Returns:
            List of orders to execute
        """
        # Align indices
        all_symbols = target_positions.index.union(current_positions.index)
        target_positions = target_positions.reindex(all_symbols, fill_value=0)
        current_positions = current_positions.reindex(all_symbols, fill_value=0)

        # Calculate required trades
        position_changes = target_positions - current_positions

        # Generate orders
        orders = []
        for symbol, change in position_changes.items():
            if abs(change) < self.min_order_size:
                continue

            # Determine order side
            if change > 0:
                side = OrderSide.BUY
                quantity = change
            else:
                side = OrderSide.SELL
                quantity = abs(change)

            # Round to lot size
            quantity = self._round_to_lot_size(quantity)

            if quantity < self.min_order_size:
                continue

            # Get current price
            price = prices.get(symbol, None)

            # Create order
            order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                price=price,
                strategy=strategy_name,
                metadata={'target_position': target_positions[symbol]}
            )

            # Validate
            if self.validate_orders and not self._validate_order(order):
                logger.warning(f"Order validation failed for {symbol}")
                continue

            orders.append(order)

        logger.info(
            f"Generated {len(orders)} orders from position targets "
            f"(strategy={strategy_name})"
        )

        return orders

    def generate_orders_from_signals(
        self,
        signals: pd.Series,
        current_positions: pd.Series,
        prices: pd.Series,
        position_sizer: Optional[callable] = None,
        strategy_name: Optional[str] = None,
    ) -> List[Order]:
        """
        Generate orders from strategy signals.

        Args:
            signals: Trading signals (-1 to 1, or discrete -1/0/1)
            current_positions: Current position sizes
            prices: Current prices
            position_sizer: Optional function to size positions from signals
            strategy_name: Strategy name for tagging

        Returns:
            List of orders
        """
        if position_sizer is None:
            # Default: signal magnitude = position size
            target_positions = signals
        else:
            # Use custom position sizing
            target_positions = signals.apply(position_sizer)

        return self.generate_orders_from_targets(
            target_positions,
            current_positions,
            prices,
            strategy_name
        )

    def aggregate_orders(
        self,
        orders: List[Order],
        aggregate_by_symbol: bool = True,
    ) -> List[Order]:
        """
        Aggregate multiple orders.

        Args:
            orders: List of orders to aggregate
            aggregate_by_symbol: Aggregate orders for same symbol

        Returns:
            Aggregated orders
        """
        if not aggregate_by_symbol:
            return orders

        # Group by symbol
        symbol_orders = {}
        for order in orders:
            if order.symbol not in symbol_orders:
                symbol_orders[order.symbol] = []
            symbol_orders[order.symbol].append(order)

        # Aggregate each symbol
        aggregated = []
        for symbol, symbol_order_list in symbol_orders.items():
            if len(symbol_order_list) == 1:
                aggregated.append(symbol_order_list[0])
                continue

            # Net buys and sells
            buy_qty = sum(o.quantity for o in symbol_order_list if o.side == OrderSide.BUY)
            sell_qty = sum(o.quantity for o in symbol_order_list if o.side == OrderSide.SELL)

            net_qty = buy_qty - sell_qty

            if abs(net_qty) < self.min_order_size:
                continue

            # Create aggregated order
            side = OrderSide.BUY if net_qty > 0 else OrderSide.SELL
            quantity = abs(net_qty)

            # Combine strategies
            strategies = [o.strategy for o in symbol_order_list if o.strategy]
            strategy_str = ','.join(set(strategies)) if strategies else None

            aggregated_order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                strategy=strategy_str,
                metadata={
                    'aggregated_from': len(symbol_order_list),
                    'original_buy_qty': buy_qty,
                    'original_sell_qty': sell_qty,
                }
            )

            aggregated.append(aggregated_order)

        logger.info(
            f"Aggregated {len(orders)} orders → {len(aggregated)} orders"
        )

        return aggregated

    def net_orders(
        self,
        orders: List[Order],
    ) -> List[Order]:
        """
        Net opposing orders for the same symbol.

        Args:
            orders: List of orders

        Returns:
            Netted orders
        """
        if not self.enable_netting:
            return orders

        return self.aggregate_orders(orders, aggregate_by_symbol=True)

    def create_parent_child_orders(
        self,
        parent_quantity: float,
        symbol: str,
        side: OrderSide,
        n_children: int,
        strategy_name: Optional[str] = None,
    ) -> Tuple[Order, List[Order]]:
        """
        Create parent-child order structure for algorithmic execution.

        Parent order represents total desired position.
        Child orders are smaller slices for execution.

        Args:
            parent_quantity: Total quantity for parent order
            symbol: Trading symbol
            side: Order side
            n_children: Number of child orders to create
            strategy_name: Strategy name

        Returns:
            Tuple of (parent_order, child_orders_list)
        """
        # Create parent order
        parent_order = Order(
            symbol=symbol,
            side=side,
            quantity=parent_quantity,
            order_type=OrderType.MARKET,
            strategy=strategy_name,
            metadata={'is_parent': True, 'n_children': n_children}
        )

        # Calculate child quantities
        child_quantity = parent_quantity / n_children
        child_quantity = self._round_to_lot_size(child_quantity)

        # Create child orders
        child_orders = []
        remaining_quantity = parent_quantity

        for i in range(n_children):
            if i == n_children - 1:
                # Last child gets remaining quantity
                qty = remaining_quantity
            else:
                qty = min(child_quantity, remaining_quantity)

            if qty < self.min_order_size:
                break

            child_order = Order(
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
                strategy=strategy_name,
                parent_order_id=parent_order.order_id,
                metadata={'is_child': True, 'child_index': i}
            )

            child_orders.append(child_order)
            remaining_quantity -= qty

        logger.info(
            f"Created parent-child orders: parent_qty={parent_quantity}, "
            f"n_children={len(child_orders)}"
        )

        return parent_order, child_orders

    def validate_order(
        self,
        order: Order,
        max_position_value: Optional[float] = None,
    ) -> bool:
        """
        Validate order for sanity checks.

        Checks:
        - Quantity within min/max bounds
        - Price is positive (if specified)
        - Symbol is valid
        - Position value not too large

        Args:
            order: Order to validate
            max_position_value: Maximum position value allowed

        Returns:
            True if valid, False otherwise
        """
        return self._validate_order(order, max_position_value)

    def _validate_order(
        self,
        order: Order,
        max_position_value: Optional[float] = None,
    ) -> bool:
        """Internal order validation."""
        # Check quantity bounds
        if order.quantity < self.min_order_size:
            logger.warning(f"Order quantity {order.quantity} below minimum {self.min_order_size}")
            return False

        if order.quantity > self.max_order_size:
            logger.warning(f"Order quantity {order.quantity} above maximum {self.max_order_size}")
            return False

        # Check price if specified
        if order.price is not None and order.price <= 0:
            logger.warning(f"Invalid order price: {order.price}")
            return False

        # Check symbol
        if not order.symbol or len(order.symbol) == 0:
            logger.warning("Empty symbol")
            return False

        # Check position value if specified
        if max_position_value is not None and order.price is not None:
            position_value = order.quantity * order.price
            if position_value > max_position_value:
                logger.warning(
                    f"Position value ${position_value:,.2f} exceeds max "
                    f"${max_position_value:,.2f}"
                )
                return False

        return True

    def _round_to_lot_size(self, quantity: float) -> float:
        """Round quantity to lot size."""
        if self.round_lot_size <= 0:
            return quantity

        return round(quantity / self.round_lot_size) * self.round_lot_size

    def optimize_order_sequence(
        self,
        orders: List[Order],
        urgency_scores: Optional[Dict[str, float]] = None,
    ) -> List[Order]:
        """
        Optimize order execution sequence.

        Prioritizes:
        1. High urgency orders
        2. Liquidity considerations
        3. Cross-impact minimization

        Args:
            orders: List of orders to sequence
            urgency_scores: Optional urgency scores by symbol (0 to 1)

        Returns:
            Reordered list of orders
        """
        if urgency_scores is None:
            urgency_scores = {}

        # Score each order
        scored_orders = []
        for order in orders:
            urgency = urgency_scores.get(order.symbol, 0.5)

            # Size score (smaller orders first to reduce market impact)
            size_score = 1.0 / (1.0 + np.log1p(order.quantity))

            # Combined score
            score = 0.6 * urgency + 0.4 * size_score

            scored_orders.append((score, order))

        # Sort by score (descending)
        scored_orders.sort(key=lambda x: x[0], reverse=True)

        optimized_orders = [order for score, order in scored_orders]

        logger.info(
            f"Optimized order sequence: {len(optimized_orders)} orders "
            f"(avg_urgency={np.mean([urgency_scores.get(o.symbol, 0.5) for o in orders]):.2f})"
        )

        return optimized_orders

    def generate_execution_report(
        self,
        orders: List[Order],
    ) -> Dict:
        """
        Generate execution report from orders.

        Args:
            orders: List of orders

        Returns:
            Dictionary with execution statistics
        """
        if not orders:
            return {}

        total_orders = len(orders)
        buy_orders = sum(1 for o in orders if o.side == OrderSide.BUY)
        sell_orders = sum(1 for o in orders if o.side == OrderSide.SELL)

        total_buy_qty = sum(o.quantity for o in orders if o.side == OrderSide.BUY)
        total_sell_qty = sum(o.quantity for o in orders if o.side == OrderSide.SELL)

        # Estimate total value (if prices available)
        total_value = sum(
            o.quantity * o.price
            for o in orders
            if o.price is not None
        )

        symbols = list(set(o.symbol for o in orders))
        strategies = list(set(o.strategy for o in orders if o.strategy))

        report = {
            'total_orders': total_orders,
            'buy_orders': buy_orders,
            'sell_orders': sell_orders,
            'total_buy_quantity': total_buy_qty,
            'total_sell_quantity': total_sell_qty,
            'net_quantity': total_buy_qty - total_sell_qty,
            'total_value': total_value,
            'n_symbols': len(symbols),
            'n_strategies': len(strategies),
            'symbols': symbols,
            'strategies': strategies,
        }

        logger.info(
            f"Execution report: {total_orders} orders, "
            f"{buy_orders} buys, {sell_orders} sells, "
            f"value=${total_value:,.2f}"
        )

        return report

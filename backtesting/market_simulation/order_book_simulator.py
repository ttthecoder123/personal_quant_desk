"""
Order Book Simulator

Implements realistic Level 2 order book simulation with:
- Order book reconstruction
- Queue position modeling
- FIFO matching logic
- Market/limit order handling
- Bid-ask spread calculation
- Order book imbalance metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import numpy as np
from loguru import logger


class OrderSide(Enum):
    """Order side enumeration."""
    BID = "BID"
    ASK = "ASK"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class BookLevel:
    """
    Single price level in the order book.

    Attributes:
        price: Price level
        quantity: Total quantity at this level
        order_count: Number of orders at this level
        queue: FIFO queue of orders (for realistic queue position modeling)
    """
    price: float
    quantity: float = 0.0
    order_count: int = 0
    queue: Deque[Tuple[str, float]] = field(default_factory=deque)  # (order_id, quantity)

    def add_order(self, order_id: str, quantity: float):
        """Add order to this level."""
        self.queue.append((order_id, quantity))
        self.quantity += quantity
        self.order_count += 1

    def remove_order(self, order_id: str) -> Optional[float]:
        """
        Remove order from this level.

        Returns:
            Quantity removed if found, None otherwise
        """
        for i, (oid, qty) in enumerate(self.queue):
            if oid == order_id:
                self.queue.remove((oid, qty))
                self.quantity -= qty
                self.order_count -= 1
                return qty
        return None

    def match_quantity(self, quantity: float) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Match incoming quantity against this level using FIFO.

        Args:
            quantity: Quantity to match

        Returns:
            Tuple of (remaining_quantity, matched_orders)
        """
        matched = []
        remaining = quantity

        while remaining > 0 and self.queue:
            order_id, order_qty = self.queue[0]

            if order_qty <= remaining:
                # Fully fill this order
                self.queue.popleft()
                matched.append((order_id, order_qty))
                remaining -= order_qty
                self.quantity -= order_qty
                self.order_count -= 1
            else:
                # Partially fill this order
                matched.append((order_id, remaining))
                self.queue[0] = (order_id, order_qty - remaining)
                self.quantity -= remaining
                remaining = 0

        return remaining, matched


@dataclass
class OrderBook:
    """
    Level 2 order book for a single symbol.

    Maintains separate bid and ask sides with full queue position tracking.
    """
    symbol: str
    timestamp: datetime
    bids: Dict[float, BookLevel] = field(default_factory=dict)
    asks: Dict[float, BookLevel] = field(default_factory=dict)
    last_trade_price: float = 0.0
    last_trade_quantity: float = 0.0

    def add_order(self, order_id: str, side: OrderSide, price: float, quantity: float):
        """
        Add limit order to book.

        Args:
            order_id: Unique order identifier
            side: BID or ASK
            price: Limit price
            quantity: Order quantity
        """
        book = self.bids if side == OrderSide.BID else self.asks

        if price not in book:
            book[price] = BookLevel(price=price)

        book[price].add_order(order_id, quantity)
        logger.debug(f"Added {side.value} order {order_id} @ {price} x {quantity} to {self.symbol}")

    def cancel_order(self, order_id: str, side: OrderSide, price: float) -> bool:
        """
        Cancel order from book.

        Args:
            order_id: Order to cancel
            side: Order side
            price: Price level

        Returns:
            True if order was found and cancelled
        """
        book = self.bids if side == OrderSide.BID else self.asks

        if price in book:
            qty = book[price].remove_order(order_id)
            if qty is not None:
                if book[price].quantity == 0:
                    del book[price]
                logger.debug(f"Cancelled {side.value} order {order_id} @ {price}")
                return True

        return False

    def execute_market_order(self, side: OrderSide, quantity: float) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Execute market order against the book.

        Args:
            side: BID (buy) or ASK (sell)
            quantity: Order quantity

        Returns:
            Tuple of (unfilled_quantity, fills) where fills is [(price, quantity), ...]
        """
        # Market buy orders match against asks, market sells against bids
        book = self.asks if side == OrderSide.BID else self.bids

        if not book:
            logger.warning(f"No liquidity for {side.value} market order on {self.symbol}")
            return quantity, []

        # Sort price levels (ascending for asks, descending for bids)
        price_levels = sorted(book.keys()) if side == OrderSide.BID else sorted(book.keys(), reverse=True)

        fills = []
        remaining = quantity

        for price in price_levels:
            if remaining <= 0:
                break

            level = book[price]
            matched_qty = min(remaining, level.quantity)

            # Match at this level
            unfilled, matched = level.match_quantity(matched_qty)

            if matched:
                fills.append((price, matched_qty - unfilled))
                remaining = unfilled

            # Remove empty level
            if level.quantity == 0:
                del book[price]

        # Record last trade
        if fills:
            self.last_trade_price = fills[-1][0]
            self.last_trade_quantity = sum(qty for _, qty in fills)

        return remaining, fills

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return min(self.asks.keys()) if self.asks else None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_spread_bps(self) -> Optional[float]:
        """Get bid-ask spread in basis points."""
        spread = self.get_spread()
        mid = self.get_mid_price()

        if spread is not None and mid is not None and mid > 0:
            return (spread / mid) * 10000
        return None

    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get order book depth.

        Args:
            levels: Number of levels to return

        Returns:
            Dictionary with 'bids' and 'asks' containing [(price, quantity), ...]
        """
        bid_levels = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_levels = sorted(self.asks.keys())[:levels]

        return {
            'bids': [(price, self.bids[price].quantity) for price in bid_levels],
            'asks': [(price, self.asks[price].quantity) for price in ask_levels]
        }

    def get_volume_at_price(self, price: float, side: OrderSide) -> float:
        """Get total volume at specific price level."""
        book = self.bids if side == OrderSide.BID else self.asks
        return book[price].quantity if price in book else 0.0

    def get_total_volume(self, side: OrderSide, max_levels: Optional[int] = None) -> float:
        """
        Get total volume on one side of the book.

        Args:
            side: BID or ASK
            max_levels: Maximum number of levels to include (None = all)

        Returns:
            Total volume
        """
        book = self.bids if side == OrderSide.BID else self.asks

        if max_levels is None:
            return sum(level.quantity for level in book.values())

        # Get top N levels
        prices = sorted(book.keys(), reverse=(side == OrderSide.BID))[:max_levels]
        return sum(book[price].quantity for price in prices)

    def calculate_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance.

        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Args:
            levels: Number of levels to consider

        Returns:
            Imbalance ratio in [-1, 1]
        """
        bid_volume = self.get_total_volume(OrderSide.BID, max_levels=levels)
        ask_volume = self.get_total_volume(OrderSide.ASK, max_levels=levels)

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / total_volume

    def calculate_weighted_mid(self, levels: int = 5) -> Optional[float]:
        """
        Calculate volume-weighted mid price.

        Args:
            levels: Number of levels to consider

        Returns:
            Weighted mid price
        """
        bid_volume = self.get_total_volume(OrderSide.BID, max_levels=levels)
        ask_volume = self.get_total_volume(OrderSide.ASK, max_levels=levels)
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return (best_bid + best_ask) / 2.0

        return (best_bid * ask_volume + best_ask * bid_volume) / total_volume


class OrderBookSimulator:
    """
    Simulates realistic order book dynamics for backtesting.

    Features:
    - Multiple symbol support
    - Level 2 order book reconstruction
    - Queue position modeling
    - FIFO matching
    - Market impact estimation
    """

    def __init__(self, symbols: List[str], initial_spread_bps: float = 10.0):
        """
        Initialize order book simulator.

        Args:
            symbols: List of symbols to simulate
            initial_spread_bps: Initial bid-ask spread in basis points
        """
        self.symbols = symbols
        self.initial_spread_bps = initial_spread_bps
        self.order_books: Dict[str, OrderBook] = {}
        self.order_counter = 0

        logger.info(f"Initialized OrderBookSimulator for {len(symbols)} symbols")

    def initialize_book(self, symbol: str, mid_price: float, timestamp: datetime,
                       depth_levels: int = 10, volume_per_level: float = 1000.0):
        """
        Initialize order book with synthetic liquidity.

        Args:
            symbol: Symbol to initialize
            mid_price: Current mid price
            timestamp: Current timestamp
            depth_levels: Number of price levels to create
            volume_per_level: Base volume per level
        """
        book = OrderBook(symbol=symbol, timestamp=timestamp)

        # Calculate spread
        spread = (self.initial_spread_bps / 10000) * mid_price
        half_spread = spread / 2.0

        # Create bid levels
        for i in range(depth_levels):
            price = mid_price - half_spread - (i * half_spread * 0.2)
            # Volume decreases with distance from mid
            volume = volume_per_level * (1.0 - i * 0.05)
            order_id = f"INIT_BID_{symbol}_{i}"
            book.add_order(order_id, OrderSide.BID, price, volume)

        # Create ask levels
        for i in range(depth_levels):
            price = mid_price + half_spread + (i * half_spread * 0.2)
            volume = volume_per_level * (1.0 - i * 0.05)
            order_id = f"INIT_ASK_{symbol}_{i}"
            book.add_order(order_id, OrderSide.ASK, price, volume)

        self.order_books[symbol] = book
        logger.debug(f"Initialized order book for {symbol} at mid={mid_price:.2f}, spread={spread:.4f}")

    def update_book(self, symbol: str, mid_price: float, timestamp: datetime):
        """
        Update order book to reflect new mid price.

        Args:
            symbol: Symbol to update
            mid_price: New mid price
            timestamp: Current timestamp
        """
        if symbol not in self.order_books:
            self.initialize_book(symbol, mid_price, timestamp)
            return

        book = self.order_books[symbol]
        old_mid = book.get_mid_price()

        if old_mid is None:
            self.initialize_book(symbol, mid_price, timestamp)
            return

        # Shift book by price change
        price_change = mid_price - old_mid

        if abs(price_change) > 0.0001:  # Only update if meaningful change
            # Rebuild book at new price level
            self.initialize_book(symbol, mid_price, timestamp)

    def submit_order(self, symbol: str, order_id: str, side: OrderSide,
                    order_type: OrderType, quantity: float,
                    price: Optional[float] = None) -> Dict:
        """
        Submit order to the book.

        Args:
            symbol: Symbol
            order_id: Unique order ID
            side: BID or ASK
            order_type: MARKET or LIMIT
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)

        Returns:
            Execution result dictionary
        """
        if symbol not in self.order_books:
            logger.error(f"No order book for {symbol}")
            return {'status': 'REJECTED', 'reason': 'NO_BOOK'}

        book = self.order_books[symbol]

        if order_type == OrderType.MARKET:
            return self._execute_market_order(book, order_id, side, quantity)
        else:
            return self._submit_limit_order(book, order_id, side, price, quantity)

    def _execute_market_order(self, book: OrderBook, order_id: str,
                             side: OrderSide, quantity: float) -> Dict:
        """Execute market order against the book."""
        unfilled, fills = book.execute_market_order(side, quantity)

        if not fills:
            return {
                'status': 'REJECTED',
                'reason': 'NO_LIQUIDITY',
                'order_id': order_id
            }

        # Calculate average fill price
        total_qty = sum(qty for _, qty in fills)
        avg_price = sum(price * qty for price, qty in fills) / total_qty if total_qty > 0 else 0.0

        status = 'FILLED' if unfilled == 0 else 'PARTIALLY_FILLED'

        return {
            'status': status,
            'order_id': order_id,
            'fills': fills,
            'avg_fill_price': avg_price,
            'filled_quantity': total_qty,
            'unfilled_quantity': unfilled
        }

    def _submit_limit_order(self, book: OrderBook, order_id: str,
                           side: OrderSide, price: float, quantity: float) -> Dict:
        """Submit limit order to the book."""
        # Check if order is immediately marketable
        best_bid = book.get_best_bid()
        best_ask = book.get_best_ask()

        is_marketable = False
        if side == OrderSide.BID and best_ask is not None:
            is_marketable = price >= best_ask
        elif side == OrderSide.ASK and best_bid is not None:
            is_marketable = price <= best_bid

        if is_marketable:
            # Execute as aggressive limit order
            unfilled, fills = book.execute_market_order(side, quantity)

            if unfilled > 0:
                # Rest goes into book
                book.add_order(order_id, side, price, unfilled)

            total_qty = sum(qty for _, qty in fills)
            avg_price = sum(p * q for p, q in fills) / total_qty if total_qty > 0 else price

            return {
                'status': 'PARTIALLY_FILLED' if unfilled > 0 else 'FILLED',
                'order_id': order_id,
                'fills': fills,
                'avg_fill_price': avg_price,
                'filled_quantity': total_qty,
                'unfilled_quantity': unfilled,
                'resting_quantity': unfilled
            }
        else:
            # Add to book
            book.add_order(order_id, side, price, quantity)

            return {
                'status': 'ACCEPTED',
                'order_id': order_id,
                'resting_quantity': quantity
            }

    def cancel_order(self, symbol: str, order_id: str, side: OrderSide, price: float) -> bool:
        """
        Cancel resting order.

        Args:
            symbol: Symbol
            order_id: Order to cancel
            side: Order side
            price: Price level

        Returns:
            True if cancelled successfully
        """
        if symbol not in self.order_books:
            return False

        return self.order_books[symbol].cancel_order(order_id, side, price)

    def get_book_state(self, symbol: str) -> Optional[Dict]:
        """
        Get current order book state.

        Args:
            symbol: Symbol

        Returns:
            Dictionary with book statistics
        """
        if symbol not in self.order_books:
            return None

        book = self.order_books[symbol]

        return {
            'symbol': symbol,
            'timestamp': book.timestamp,
            'best_bid': book.get_best_bid(),
            'best_ask': book.get_best_ask(),
            'mid_price': book.get_mid_price(),
            'spread': book.get_spread(),
            'spread_bps': book.get_spread_bps(),
            'imbalance': book.calculate_imbalance(),
            'weighted_mid': book.calculate_weighted_mid(),
            'depth': book.get_depth(),
            'bid_volume': book.get_total_volume(OrderSide.BID, max_levels=5),
            'ask_volume': book.get_total_volume(OrderSide.ASK, max_levels=5)
        }

    def estimate_market_impact(self, symbol: str, side: OrderSide, quantity: float) -> Dict:
        """
        Estimate market impact of an order.

        Args:
            symbol: Symbol
            side: Order side
            quantity: Order quantity

        Returns:
            Impact statistics dictionary
        """
        if symbol not in self.order_books:
            return {'error': 'NO_BOOK'}

        book = self.order_books[symbol]

        # Simulate execution to estimate impact
        # Create a copy of the relevant side
        source_book = book.asks if side == OrderSide.BID else book.bids
        price_levels = sorted(source_book.keys()) if side == OrderSide.BID else sorted(source_book.keys(), reverse=True)

        remaining = quantity
        total_cost = 0.0
        levels_consumed = 0

        for price in price_levels:
            if remaining <= 0:
                break

            available = source_book[price].quantity
            executed = min(remaining, available)
            total_cost += price * executed
            remaining -= executed
            levels_consumed += 1

        filled_qty = quantity - remaining
        avg_price = total_cost / filled_qty if filled_qty > 0 else 0.0

        # Calculate impact relative to mid
        mid_price = book.get_mid_price()
        impact_bps = 0.0
        if mid_price and mid_price > 0:
            impact_bps = abs((avg_price - mid_price) / mid_price) * 10000

        return {
            'avg_price': avg_price,
            'filled_quantity': filled_qty,
            'unfilled_quantity': remaining,
            'levels_consumed': levels_consumed,
            'impact_bps': impact_bps,
            'mid_price': mid_price
        }

"""
Fill Simulator

Implements realistic order fill simulation for backtesting:
- Fill probability based on order aggressiveness
- Partial fill modeling
- Fill price determination
- Time-weighted fill profiles
- Priority queue simulation
- Liquidity-based fill rates

Based on realistic market microstructure and empirical fill patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


class OrderAggressiveness(Enum):
    """Order aggressiveness level."""
    MARKET = "MARKET"  # Crosses spread
    AGGRESSIVE_LIMIT = "AGGRESSIVE_LIMIT"  # Near or at best price
    PASSIVE_LIMIT = "PASSIVE_LIMIT"  # Away from best price
    FAR_LIMIT = "FAR_LIMIT"  # Far from best price


class FillStatus(Enum):
    """Fill status."""
    UNFILLED = "UNFILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"


@dataclass
class FillProbability:
    """
    Fill probability parameters.

    Determines likelihood of order execution based on:
    - Order type and aggressiveness
    - Queue position
    - Market conditions
    """
    market_fill_prob: float = 1.0  # Market orders almost always fill
    aggressive_limit_prob: float = 0.8  # Aggressive limits high probability
    passive_limit_prob: float = 0.4  # Passive limits moderate probability
    far_limit_prob: float = 0.1  # Far limits low probability

    # Time-based fill probability (cumulative over time)
    def get_time_weighted_probability(self, seconds_elapsed: float, base_prob: float) -> float:
        """
        Get fill probability adjusted for time elapsed.

        Probability increases over time (more likely to fill if resting longer).

        Args:
            seconds_elapsed: Seconds since order submission
            base_prob: Base fill probability

        Returns:
            Time-adjusted fill probability
        """
        # Exponential increase in fill probability
        # After 60 seconds, probability approaches 1.0 for aggressive orders
        time_factor = 1 - np.exp(-seconds_elapsed / 60.0)
        adjusted_prob = base_prob + (1 - base_prob) * time_factor * 0.5

        return min(adjusted_prob, 1.0)


@dataclass
class FillResult:
    """
    Result of a fill simulation.

    Attributes:
        status: Fill status
        filled_quantity: Quantity filled
        unfilled_quantity: Quantity not filled
        fill_price: Average fill price
        fills: List of individual fills [(timestamp, price, quantity)]
        queue_position: Position in queue (if limit order)
        time_to_fill: Time taken to fill (seconds)
    """
    status: FillStatus
    filled_quantity: float = 0.0
    unfilled_quantity: float = 0.0
    fill_price: float = 0.0
    fills: List[Tuple[datetime, float, float]] = field(default_factory=list)
    queue_position: Optional[int] = None
    time_to_fill: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class FillSimulator:
    """
    Simulates realistic order fills for backtesting.

    Features:
    - Probabilistic fill modeling
    - Partial fills based on order size and liquidity
    - Queue position tracking
    - Time-weighted fill profiles
    - Price improvement for passive orders
    """

    def __init__(
        self,
        fill_probability: Optional[FillProbability] = None,
        enable_partial_fills: bool = True,
        min_fill_size: float = 1.0
    ):
        """
        Initialize fill simulator.

        Args:
            fill_probability: Fill probability parameters
            enable_partial_fills: Whether to simulate partial fills
            min_fill_size: Minimum fill size (shares)
        """
        self.fill_probability = fill_probability or FillProbability()
        self.enable_partial_fills = enable_partial_fills
        self.min_fill_size = min_fill_size

        # Track resting orders
        self.resting_orders: Dict[str, Dict] = {}

        logger.info("Initialized FillSimulator")

    def simulate_market_order_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        timestamp: datetime,
        market_price: float,
        available_liquidity: Optional[float] = None,
        depth: Optional[List[Tuple[float, float]]] = None
    ) -> FillResult:
        """
        Simulate market order fill.

        Market orders typically fill immediately but may experience:
        - Partial fills if quantity exceeds available liquidity
        - Multiple price levels if large order
        - Higher average price due to walking the book

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            timestamp: Order timestamp
            market_price: Current market price
            available_liquidity: Available liquidity at best price
            depth: Order book depth [(price, quantity), ...]

        Returns:
            FillResult
        """
        fills = []
        remaining = quantity

        # Use depth if provided, otherwise assume sufficient liquidity
        if depth and len(depth) > 0:
            # Fill against order book levels
            for price, available in depth:
                if remaining <= 0:
                    break

                fill_qty = min(remaining, available)
                fills.append((timestamp, price, fill_qty))
                remaining -= fill_qty

        else:
            # Simple fill at market price
            if available_liquidity and available_liquidity < quantity:
                # Partial fill
                fills.append((timestamp, market_price, available_liquidity))
                remaining = quantity - available_liquidity
            else:
                # Full fill
                fills.append((timestamp, market_price, quantity))
                remaining = 0

        # Calculate average fill price
        total_filled = sum(qty for _, _, qty in fills)
        avg_price = sum(price * qty for _, price, qty in fills) / total_filled if total_filled > 0 else market_price

        status = FillStatus.FILLED if remaining == 0 else FillStatus.PARTIALLY_FILLED

        logger.debug(
            f"Market order {order_id}: {status.value}, "
            f"filled={total_filled}/{quantity} @ avg_price={avg_price:.2f}"
        )

        return FillResult(
            status=status,
            filled_quantity=total_filled,
            unfilled_quantity=remaining,
            fill_price=avg_price,
            fills=fills,
            time_to_fill=0.0  # Market orders fill immediately
        )

    def simulate_limit_order_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        timestamp: datetime,
        market_price: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        queue_position: Optional[int] = None,
        time_elapsed: float = 0.0
    ) -> FillResult:
        """
        Simulate limit order fill.

        Limit orders fill based on:
        - Price aggressiveness relative to market
        - Queue position
        - Time elapsed
        - Market volatility

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            limit_price: Limit price
            timestamp: Order timestamp
            market_price: Current market price
            best_bid: Best bid price
            best_ask: Best ask price
            queue_position: Position in queue at this price
            time_elapsed: Time since order submission (seconds)

        Returns:
            FillResult
        """
        # Determine aggressiveness
        aggressiveness = self._determine_aggressiveness(
            side, limit_price, market_price, best_bid, best_ask
        )

        # Get base fill probability
        base_prob = self._get_base_fill_probability(aggressiveness)

        # Adjust for time elapsed
        fill_prob = self.fill_probability.get_time_weighted_probability(time_elapsed, base_prob)

        # Adjust for queue position
        if queue_position is not None and queue_position > 0:
            # Deeper in queue → lower fill probability
            queue_adjustment = np.exp(-queue_position / 10.0)
            fill_prob *= queue_adjustment

        # Determine if order fills
        if np.random.random() > fill_prob:
            # No fill
            logger.debug(f"Limit order {order_id} did not fill (prob={fill_prob:.3f})")
            return FillResult(
                status=FillStatus.UNFILLED,
                unfilled_quantity=quantity,
                queue_position=queue_position
            )

        # Determine fill quantity
        if self.enable_partial_fills and quantity > 100:
            # Partial fill possible for large orders
            fill_ratio = self._determine_fill_ratio(aggressiveness, time_elapsed)
            filled_qty = max(quantity * fill_ratio, self.min_fill_size)
            filled_qty = min(filled_qty, quantity)
        else:
            filled_qty = quantity

        # Determine fill price (may get price improvement)
        fill_price = self._determine_fill_price(
            side, limit_price, market_price, aggressiveness
        )

        fills = [(timestamp, fill_price, filled_qty)]
        remaining = quantity - filled_qty

        status = FillStatus.FILLED if remaining == 0 else FillStatus.PARTIALLY_FILLED

        logger.debug(
            f"Limit order {order_id}: {status.value}, "
            f"filled={filled_qty}/{quantity} @ {fill_price:.2f} "
            f"(aggressiveness={aggressiveness.value}, prob={fill_prob:.3f})"
        )

        return FillResult(
            status=status,
            filled_quantity=filled_qty,
            unfilled_quantity=remaining,
            fill_price=fill_price,
            fills=fills,
            queue_position=queue_position,
            time_to_fill=time_elapsed
        )

    def submit_resting_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        timestamp: datetime,
        queue_position: Optional[int] = None
    ):
        """
        Submit a resting limit order.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            limit_price: Limit price
            timestamp: Submission timestamp
            queue_position: Initial queue position
        """
        self.resting_orders[order_id] = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'limit_price': limit_price,
            'timestamp': timestamp,
            'queue_position': queue_position or 0,
            'filled_quantity': 0.0
        }

        logger.debug(f"Submitted resting order {order_id} for {symbol}")

    def update_resting_order(
        self,
        order_id: str,
        current_time: datetime,
        market_price: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None
    ) -> Optional[FillResult]:
        """
        Update resting order and check for fills.

        Args:
            order_id: Order to update
            current_time: Current timestamp
            market_price: Current market price
            best_bid: Best bid price
            best_ask: Best ask price

        Returns:
            FillResult if order filled, None otherwise
        """
        if order_id not in self.resting_orders:
            return None

        order = self.resting_orders[order_id]
        time_elapsed = (current_time - order['timestamp']).total_seconds()

        # Simulate potential fill
        result = self.simulate_limit_order_fill(
            order_id=order_id,
            symbol=order['symbol'],
            side=order['side'],
            quantity=order['quantity'] - order['filled_quantity'],
            limit_price=order['limit_price'],
            timestamp=current_time,
            market_price=market_price,
            best_bid=best_bid,
            best_ask=best_ask,
            queue_position=order['queue_position'],
            time_elapsed=time_elapsed
        )

        if result.status != FillStatus.UNFILLED:
            # Update order
            order['filled_quantity'] += result.filled_quantity

            if order['filled_quantity'] >= order['quantity']:
                # Fully filled, remove from resting orders
                del self.resting_orders[order_id]
                logger.debug(f"Order {order_id} fully filled and removed")

            return result

        return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a resting order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancelled, False if not found
        """
        if order_id in self.resting_orders:
            del self.resting_orders[order_id]
            logger.debug(f"Cancelled order {order_id}")
            return True
        return False

    def _determine_aggressiveness(
        self,
        side: str,
        limit_price: float,
        market_price: float,
        best_bid: Optional[float],
        best_ask: Optional[float]
    ) -> OrderAggressiveness:
        """
        Determine order aggressiveness.

        Args:
            side: Order side
            limit_price: Limit price
            market_price: Market price
            best_bid: Best bid
            best_ask: Best ask

        Returns:
            OrderAggressiveness level
        """
        if side.upper() == 'BUY':
            # For buys, higher price = more aggressive
            if best_ask and limit_price >= best_ask:
                return OrderAggressiveness.AGGRESSIVE_LIMIT
            elif best_bid and limit_price >= best_bid:
                return OrderAggressiveness.AGGRESSIVE_LIMIT
            elif limit_price >= market_price * 0.995:  # Within 0.5%
                return OrderAggressiveness.PASSIVE_LIMIT
            else:
                return OrderAggressiveness.FAR_LIMIT
        else:  # SELL
            # For sells, lower price = more aggressive
            if best_bid and limit_price <= best_bid:
                return OrderAggressiveness.AGGRESSIVE_LIMIT
            elif best_ask and limit_price <= best_ask:
                return OrderAggressiveness.AGGRESSIVE_LIMIT
            elif limit_price <= market_price * 1.005:  # Within 0.5%
                return OrderAggressiveness.PASSIVE_LIMIT
            else:
                return OrderAggressiveness.FAR_LIMIT

    def _get_base_fill_probability(self, aggressiveness: OrderAggressiveness) -> float:
        """Get base fill probability for aggressiveness level."""
        if aggressiveness == OrderAggressiveness.MARKET:
            return self.fill_probability.market_fill_prob
        elif aggressiveness == OrderAggressiveness.AGGRESSIVE_LIMIT:
            return self.fill_probability.aggressive_limit_prob
        elif aggressiveness == OrderAggressiveness.PASSIVE_LIMIT:
            return self.fill_probability.passive_limit_prob
        else:  # FAR_LIMIT
            return self.fill_probability.far_limit_prob

    def _determine_fill_ratio(self, aggressiveness: OrderAggressiveness, time_elapsed: float) -> float:
        """
        Determine what fraction of order fills.

        More aggressive orders and longer waiting time → higher fill ratio.

        Args:
            aggressiveness: Order aggressiveness
            time_elapsed: Time since submission

        Returns:
            Fill ratio (0.0 to 1.0)
        """
        # Base fill ratio by aggressiveness
        base_ratios = {
            OrderAggressiveness.MARKET: 1.0,
            OrderAggressiveness.AGGRESSIVE_LIMIT: 0.8,
            OrderAggressiveness.PASSIVE_LIMIT: 0.5,
            OrderAggressiveness.FAR_LIMIT: 0.3
        }

        base_ratio = base_ratios.get(aggressiveness, 0.5)

        # Increase fill ratio with time
        time_factor = min(time_elapsed / 300.0, 0.3)  # Up to 30% increase over 5 minutes
        fill_ratio = min(base_ratio + time_factor, 1.0)

        return fill_ratio

    def _determine_fill_price(
        self,
        side: str,
        limit_price: float,
        market_price: float,
        aggressiveness: OrderAggressiveness
    ) -> float:
        """
        Determine fill price.

        Passive orders may get price improvement.

        Args:
            side: Order side
            limit_price: Limit price
            market_price: Market price
            aggressiveness: Order aggressiveness

        Returns:
            Fill price
        """
        # Aggressive orders fill at limit price (or worse for market orders)
        if aggressiveness in [OrderAggressiveness.MARKET, OrderAggressiveness.AGGRESSIVE_LIMIT]:
            return limit_price

        # Passive orders may get price improvement
        if aggressiveness == OrderAggressiveness.PASSIVE_LIMIT:
            # 30% chance of improvement
            if np.random.random() < 0.3:
                improvement = (limit_price - market_price) * 0.2
                if side.upper() == 'BUY':
                    return limit_price - abs(improvement)
                else:
                    return limit_price + abs(improvement)

        return limit_price

    def get_resting_orders(self, symbol: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get all resting orders.

        Args:
            symbol: Filter by symbol (None = all symbols)

        Returns:
            Dictionary of resting orders
        """
        if symbol:
            return {
                order_id: order
                for order_id, order in self.resting_orders.items()
                if order['symbol'] == symbol
            }
        return self.resting_orders.copy()

    def get_statistics(self) -> Dict[str, any]:
        """
        Get fill simulator statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'num_resting_orders': len(self.resting_orders),
            'enable_partial_fills': self.enable_partial_fills,
            'min_fill_size': self.min_fill_size
        }


class TimeWeightedFillSimulator(FillSimulator):
    """
    Fill simulator with time-weighted fill profiles.

    Models realistic fill patterns over time, such as:
    - TWAP (Time-Weighted Average Price) execution
    - VWAP (Volume-Weighted Average Price) execution
    """

    def __init__(self, **kwargs):
        """Initialize time-weighted fill simulator."""
        super().__init__(**kwargs)
        logger.info("Initialized TimeWeightedFillSimulator")

    def simulate_twap_execution(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        num_slices: int,
        price_series: List[Tuple[datetime, float]]
    ) -> List[FillResult]:
        """
        Simulate TWAP execution.

        Divides order into equal slices over time.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            total_quantity: Total quantity to execute
            start_time: Start time
            end_time: End time
            num_slices: Number of slices
            price_series: List of (timestamp, price) tuples

        Returns:
            List of FillResults for each slice
        """
        slice_qty = total_quantity / num_slices
        time_delta = (end_time - start_time) / num_slices

        results = []
        current_time = start_time

        for i in range(num_slices):
            # Find price at this time
            price = self._interpolate_price(current_time, price_series)

            # Simulate fill for this slice
            result = self.simulate_market_order_fill(
                order_id=f"TWAP_{i}",
                symbol=symbol,
                side=side,
                quantity=slice_qty,
                timestamp=current_time,
                market_price=price
            )

            results.append(result)
            current_time += time_delta

        logger.info(
            f"TWAP execution completed: {num_slices} slices, "
            f"avg_price={np.mean([r.fill_price for r in results]):.2f}"
        )

        return results

    def _interpolate_price(self, timestamp: datetime, price_series: List[Tuple[datetime, float]]) -> float:
        """Interpolate price at given timestamp."""
        if not price_series:
            return 0.0

        # Find surrounding prices
        before = None
        after = None

        for ts, price in price_series:
            if ts <= timestamp:
                before = (ts, price)
            elif ts > timestamp and after is None:
                after = (ts, price)
                break

        # Return interpolated price
        if before and after:
            time_diff = (after[0] - before[0]).total_seconds()
            if time_diff > 0:
                weight = (timestamp - before[0]).total_seconds() / time_diff
                return before[1] * (1 - weight) + after[1] * weight
            return before[1]
        elif before:
            return before[1]
        elif after:
            return after[1]
        else:
            return price_series[0][1]

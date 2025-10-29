"""
Event-Driven Backtest Engine

Implements realistic event-driven simulation with proper order lifecycle,
market simulation, and state management. Follows López de Prado's recommendations
for realistic backtesting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from collections import deque
from loguru import logger


class EventType(Enum):
    """Types of events in the simulation."""
    MARKET = "MARKET"  # New market data
    SIGNAL = "SIGNAL"  # Strategy signal
    ORDER = "ORDER"  # New order
    FILL = "FILL"  # Order fill
    RISK = "RISK"  # Risk limit breach
    REBALANCE = "REBALANCE"  # Periodic rebalancing


@dataclass
class Event:
    """Base class for all events."""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority = processed first

    def __lt__(self, other):
        """Compare events for priority queue."""
        if self.timestamp == other.timestamp:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


@dataclass
class MarketEvent(Event):
    """Market data update event."""
    def __init__(self, timestamp: datetime, symbol: str, data: Dict[str, float]):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.MARKET,
            data={'symbol': symbol, 'market_data': data},
            priority=10
        )


@dataclass
class SignalEvent(Event):
    """Strategy signal event."""
    def __init__(self, timestamp: datetime, symbol: str, signal: str,
                 strength: float, strategy: str, metadata: Optional[Dict] = None):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.SIGNAL,
            data={
                'symbol': symbol,
                'signal': signal,
                'strength': strength,
                'strategy': strategy,
                'metadata': metadata or {}
            },
            priority=5
        )


@dataclass
class OrderEvent(Event):
    """Order event (new, modify, cancel)."""
    def __init__(self, timestamp: datetime, order_id: str, symbol: str,
                 side: str, quantity: float, order_type: str,
                 price: Optional[float] = None, metadata: Optional[Dict] = None):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.ORDER,
            data={
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'metadata': metadata or {}
            },
            priority=3
        )


@dataclass
class FillEvent(Event):
    """Order fill event."""
    def __init__(self, timestamp: datetime, order_id: str, symbol: str,
                 side: str, quantity: float, fill_price: float,
                 commission: float, slippage: float, metadata: Optional[Dict] = None):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.FILL,
            data={
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'fill_price': fill_price,
                'commission': commission,
                'slippage': slippage,
                'metadata': metadata or {}
            },
            priority=1
        )


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    order_type: str  # MARKET, LIMIT, STOP
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    side: str  # LONG or SHORT
    avg_entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PortfolioState:
    """Portfolio state manager."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos.quantity * current_prices.get(pos.symbol, 0.0)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def update_position(self, symbol: str, quantity: float, price: float,
                       side: str, timestamp: datetime):
        """Update or create position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Update existing position
            total_quantity = pos.quantity + quantity
            if total_quantity != 0:
                pos.avg_entry_price = (
                    (pos.quantity * pos.avg_entry_price + quantity * price) /
                    total_quantity
                )
                pos.quantity = total_quantity
            else:
                # Position closed
                del self.positions[symbol]
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=abs(quantity),
                side=side,
                avg_entry_price=price,
                entry_time=timestamp
            )

    def record_trade(self, fill_data: Dict):
        """Record executed trade."""
        self.trades.append(fill_data)

    def record_equity(self, timestamp: datetime, total_value: float):
        """Record equity curve point."""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash
        })


class EventEngine:
    """
    Event-driven backtesting engine.

    Processes events in chronological order, maintaining realistic
    market conditions and order execution.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize event engine.

        Args:
            config: Engine configuration
        """
        self.config = config
        self.event_queue: deque = deque()
        self.handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }

        # State
        self.current_time: Optional[datetime] = None
        self.portfolio: Optional[PortfolioState] = None
        self.market_data: Dict[str, Dict] = {}
        self.order_counter = 0

        # Components (to be injected)
        self.strategy = None
        self.risk_manager = None
        self.execution_handler = None
        self.market_simulator = None

        # Audit trail
        self.event_history: List[Event] = []

        logger.info("EventEngine initialized")

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler."""
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")

    def add_event(self, event: Event):
        """Add event to queue (maintains chronological order)."""
        # Insert event in chronological order
        inserted = False
        for i, existing_event in enumerate(self.event_queue):
            if event < existing_event:
                self.event_queue.insert(i, event)
                inserted = True
                break
        if not inserted:
            self.event_queue.append(event)

    def process_events(self) -> bool:
        """
        Process all events for current timestamp.

        Returns:
            True if events were processed
        """
        if not self.event_queue:
            return False

        current_timestamp = self.event_queue[0].timestamp
        events_to_process = []

        # Collect all events for current timestamp
        while self.event_queue and self.event_queue[0].timestamp == current_timestamp:
            events_to_process.append(self.event_queue.popleft())

        # Process events by priority
        events_to_process.sort(key=lambda e: e.priority, reverse=True)

        for event in events_to_process:
            self._process_event(event)
            self.event_history.append(event)

        return True

    def _process_event(self, event: Event):
        """Process single event."""
        self.current_time = event.timestamp

        # Call registered handlers
        for handler in self.handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in handler for {event.event_type}: {e}")

    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        market_data: pd.DataFrame,
        strategy: Any,
        risk_manager: Any = None,
        initial_capital: float = 1000000.0
    ) -> Dict[str, Any]:
        """
        Run complete backtest.

        Args:
            start_date: Start date
            end_date: End date
            market_data: Historical market data
            strategy: Strategy instance
            risk_manager: Risk manager instance
            initial_capital: Starting capital

        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting event-driven backtest: {start_date} to {end_date}")

        # Initialize
        self.portfolio = PortfolioState(initial_capital)
        self.strategy = strategy
        self.risk_manager = risk_manager

        # Register default handlers
        self._register_default_handlers()

        # Generate market events from data
        self._generate_market_events(market_data, start_date, end_date)

        # Process all events
        event_count = 0
        while self.event_queue:
            if self.process_events():
                event_count += 1

                # Record equity curve
                if self.current_time:
                    current_prices = self._get_current_prices()
                    total_value = self.portfolio.get_total_value(current_prices)
                    self.portfolio.record_equity(self.current_time, total_value)

        logger.info(f"Backtest completed. Processed {event_count} time steps")

        # Generate results
        results = self._generate_results()
        return results

    def _register_default_handlers(self):
        """Register default event handlers."""
        self.register_handler(EventType.MARKET, self._handle_market_event)
        self.register_handler(EventType.SIGNAL, self._handle_signal_event)
        self.register_handler(EventType.ORDER, self._handle_order_event)
        self.register_handler(EventType.FILL, self._handle_fill_event)

    def _handle_market_event(self, event: MarketEvent):
        """Handle market data update."""
        symbol = event.data['symbol']
        self.market_data[symbol] = event.data['market_data']

        # Generate signals from strategy
        if self.strategy:
            signal = self.strategy.generate_signal(
                symbol=symbol,
                market_data=self.market_data[symbol],
                timestamp=event.timestamp
            )

            if signal:
                signal_event = SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal=signal['action'],
                    strength=signal.get('strength', 1.0),
                    strategy=self.strategy.name,
                    metadata=signal.get('metadata', {})
                )
                self.add_event(signal_event)

    def _handle_signal_event(self, event: SignalEvent):
        """Handle strategy signal."""
        # Risk check
        if self.risk_manager:
            can_trade, reason = self.risk_manager.check_pre_trade(
                symbol=event.data['symbol'],
                signal=event.data['signal'],
                portfolio_state=self.portfolio
            )
            if not can_trade:
                logger.debug(f"Signal rejected: {reason}")
                return

        # Generate order
        order_id = f"ORD_{self.order_counter:08d}"
        self.order_counter += 1

        order_event = OrderEvent(
            timestamp=event.timestamp,
            order_id=order_id,
            symbol=event.data['symbol'],
            side=event.data['signal'],
            quantity=self._calculate_order_quantity(event),
            order_type='MARKET'
        )
        self.add_event(order_event)

    def _handle_order_event(self, event: OrderEvent):
        """Handle order submission."""
        order = Order(
            order_id=event.data['order_id'],
            timestamp=event.timestamp,
            symbol=event.data['symbol'],
            side=event.data['side'],
            quantity=event.data['quantity'],
            order_type=event.data['order_type'],
            price=event.data.get('price')
        )

        self.portfolio.orders[order.order_id] = order

        # Simulate order execution (instant for market orders in this simple version)
        fill_event = self._simulate_fill(order, event.timestamp)
        if fill_event:
            self.add_event(fill_event)

    def _handle_fill_event(self, event: FillEvent):
        """Handle order fill."""
        fill_data = event.data

        # Update portfolio
        side_multiplier = 1 if fill_data['side'] == 'BUY' else -1
        quantity = fill_data['quantity'] * side_multiplier

        self.portfolio.update_position(
            symbol=fill_data['symbol'],
            quantity=quantity,
            price=fill_data['fill_price'],
            side='LONG' if fill_data['side'] == 'BUY' else 'SHORT',
            timestamp=event.timestamp
        )

        # Update cash
        cost = fill_data['quantity'] * fill_data['fill_price']
        total_cost = cost + fill_data['commission'] + fill_data['slippage']

        if fill_data['side'] == 'BUY':
            self.portfolio.cash -= total_cost
        else:
            self.portfolio.cash += cost - fill_data['commission'] - fill_data['slippage']

        # Record trade
        self.portfolio.record_trade(fill_data)

        # Update order status
        order_id = fill_data['order_id']
        if order_id in self.portfolio.orders:
            self.portfolio.orders[order_id].status = OrderStatus.FILLED

    def _generate_market_events(self, market_data: pd.DataFrame,
                               start_date: datetime, end_date: datetime):
        """Generate market events from data."""
        for timestamp, row in market_data.iterrows():
            if timestamp < start_date or timestamp > end_date:
                continue

            # Create market event for each symbol
            for col in row.index:
                if '_Close' in col:
                    symbol = col.replace('_Close', '')
                    market_event = MarketEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        data={
                            'close': row[col],
                            'open': row.get(f'{symbol}_Open', row[col]),
                            'high': row.get(f'{symbol}_High', row[col]),
                            'low': row.get(f'{symbol}_Low', row[col]),
                            'volume': row.get(f'{symbol}_Volume', 0)
                        }
                    )
                    self.add_event(market_event)

    def _simulate_fill(self, order: Order, timestamp: datetime) -> Optional[FillEvent]:
        """Simulate order fill (simplified)."""
        if order.symbol not in self.market_data:
            return None

        # Get current price
        current_price = self.market_data[order.symbol].get('close', 0)
        if current_price == 0:
            return None

        # Simple slippage model (0.05%)
        slippage_pct = 0.0005
        slippage = order.quantity * current_price * slippage_pct
        fill_price = current_price * (1 + slippage_pct if order.side == 'BUY' else 1 - slippage_pct)

        # Commission (0.1%)
        commission_pct = 0.001
        commission = order.quantity * fill_price * commission_pct

        return FillEvent(
            timestamp=timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage
        )

    def _calculate_order_quantity(self, signal_event: SignalEvent) -> float:
        """Calculate order quantity from signal."""
        # Simplified position sizing (10% of portfolio)
        if not self.portfolio:
            return 0.0

        symbol = signal_event.data['symbol']
        if symbol not in self.market_data:
            return 0.0

        price = self.market_data[symbol].get('close', 0)
        if price == 0:
            return 0.0

        current_prices = self._get_current_prices()
        portfolio_value = self.portfolio.get_total_value(current_prices)

        position_value = portfolio_value * 0.10  # 10% allocation
        quantity = position_value / price

        return quantity

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        return {
            symbol: data.get('close', 0.0)
            for symbol, data in self.market_data.items()
        }

    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results."""
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        if equity_df.empty:
            return {'error': 'No data generated'}

        equity_df.set_index('timestamp', inplace=True)

        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()

        # Calculate metrics
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1

        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Maximum drawdown
        cumulative = equity_df['equity'] / equity_df['equity'].iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': equity_df['equity'].iloc[-1],
            'total_trades': len(self.portfolio.trades),
            'equity_curve': equity_df,
            'trades': pd.DataFrame(self.portfolio.trades),
            'positions': list(self.portfolio.positions.values())
        }

        return results

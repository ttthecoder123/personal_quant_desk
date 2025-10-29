"""
Order type definitions for the OMS

Implements comprehensive order types including:
- Market, Limit, Stop, Stop-Limit
- Advanced orders: Iceberg, TWAP, VWAP, Pegged
- Conditional orders: OCO, Bracket, Trailing Stop
- Time-in-force: GTC, IOC, FOK, Day
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"
    ADAPTIVE = "ADAPTIVE"
    PEGGED = "PEGGED"
    TRAILING_STOP = "TRAILING_STOP"
    BRACKET = "BRACKET"
    ONE_CANCELS_OTHER = "OCO"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "NEW"
    PENDING_SUBMIT = "PENDING_SUBMIT"
    PENDING_CANCEL = "PENDING_CANCEL"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    REPLACED = "REPLACED"


class TimeInForce(Enum):
    """Time-in-force enumeration"""
    DAY = "DAY"
    GTC = "GTC"  # Good-till-cancelled
    IOC = "IOC"  # Immediate-or-cancel
    FOK = "FOK"  # Fill-or-kill
    GTD = "GTD"  # Good-till-date
    OPG = "OPG"  # At the open
    CLS = "CLS"  # At the close


class PegType(Enum):
    """Peg type for pegged orders"""
    MID = "MID"  # Mid-point
    PRIMARY = "PRIMARY"  # Primary peg
    MARKET = "MARKET"  # Market peg
    BEST_BID = "BEST_BID"
    BEST_ASK = "BEST_ASK"


@dataclass
class Fill:
    """Represents an order fill"""
    fill_id: str
    order_id: str
    timestamp: datetime
    price: float
    quantity: float
    commission: float = 0.0
    venue: Optional[str] = None
    exchange_id: Optional[str] = None
    liquidity: Optional[str] = None  # "ADD" or "REMOVE"

    @property
    def total_value(self) -> float:
        """Total fill value including commission"""
        return self.price * self.quantity + self.commission


@dataclass
class Order:
    """Base order class with comprehensive attributes"""

    # Identification
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_order_id: Optional[str] = None

    # Security
    symbol: str = ""
    exchange: Optional[str] = None
    asset_class: str = "EQUITY"

    # Order specifications
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None

    # Time-in-force
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: Optional[datetime] = None

    # Status tracking
    status: OrderStatus = OrderStatus.NEW
    create_time: datetime = field(default_factory=datetime.now)
    submit_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None

    # Execution tracking
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fills: List[Fill] = field(default_factory=list)

    # Routing
    broker: Optional[str] = None
    venue: Optional[str] = None
    routing_instructions: Dict[str, Any] = field(default_factory=dict)

    # Algorithm parameters (for algo orders)
    algo_params: Dict[str, Any] = field(default_factory=dict)

    # Risk and validation
    account: Optional[str] = None
    strategy_id: Optional[str] = None

    # Metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    # Rejection/error tracking
    reject_reason: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate order after initialization"""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if self.price is None or self.price <= 0:
                raise ValueError(f"{self.order_type} orders require a valid price")

        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError(f"{self.order_type} orders require a valid stop price")

    @property
    def remaining_quantity(self) -> float:
        """Quantity remaining to be filled"""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Percentage of order filled"""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0.0

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]

    @property
    def is_active(self) -> bool:
        """Check if order is active"""
        return self.status in [
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED
        ]

    def add_fill(self, fill: Fill):
        """Add a fill to the order"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity

        # Update average fill price
        total_value = sum(f.price * f.quantity for f in self.fills)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.last_update_time = datetime.now()

    def update_status(self, new_status: OrderStatus, message: Optional[str] = None):
        """Update order status"""
        old_status = self.status
        self.status = new_status
        self.last_update_time = datetime.now()

        if new_status == OrderStatus.REJECTED:
            self.reject_reason = message
        elif new_status in [OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
            self.error_message = message

        if new_status == OrderStatus.SUBMITTED and old_status == OrderStatus.PENDING_SUBMIT:
            self.submit_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'create_time': self.create_time.isoformat(),
            'fills': [f.__dict__ for f in self.fills]
        }


@dataclass
class IcebergOrder(Order):
    """Iceberg order with hidden quantity"""
    display_quantity: float = 0.0
    refresh_on_fill: bool = True
    randomize_display: bool = True
    min_display: Optional[float] = None
    max_display: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.ICEBERG
        if self.display_quantity <= 0:
            self.display_quantity = min(self.quantity * 0.1, 100)
        if self.display_quantity > self.quantity:
            raise ValueError("Display quantity cannot exceed total quantity")


@dataclass
class TWAPOrder(Order):
    """Time-Weighted Average Price order"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    num_slices: int = 10
    randomize_intervals: bool = True
    max_participation: float = 0.10
    min_slice_size: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.TWAP
        if self.start_time is None:
            self.start_time = datetime.now()


@dataclass
class VWAPOrder(Order):
    """Volume-Weighted Average Price order"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participation: float = 0.10
    aggressive_on_close: bool = True
    use_dark_pools: bool = True
    min_fill_size: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.VWAP
        if self.start_time is None:
            self.start_time = datetime.now()


@dataclass
class ImplementationShortfallOrder(Order):
    """Implementation Shortfall / Almgren-Chriss order"""
    risk_aversion: float = 1.0
    urgency: float = 0.5  # 0 = patient, 1 = urgent
    permanent_impact_factor: float = 0.1
    temporary_impact_factor: float = 0.01
    volatility: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.IMPLEMENTATION_SHORTFALL


@dataclass
class BracketOrder(Order):
    """Bracket order with profit target and stop loss"""
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    trailing_stop_percent: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.BRACKET
        if self.take_profit_price is None and self.stop_loss_price is None:
            raise ValueError("Bracket order requires at least take profit or stop loss")


@dataclass
class OCOOrder:
    """One-Cancels-Other order"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    primary_order: Order = None
    secondary_order: Order = None

    def __post_init__(self):
        if self.primary_order is None or self.secondary_order is None:
            raise ValueError("OCO order requires both primary and secondary orders")
        self.primary_order.parent_order_id = self.order_id
        self.secondary_order.parent_order_id = self.order_id


class OrderFactory:
    """Factory for creating orders"""

    @staticmethod
    def create_market_order(symbol: str, side: OrderSide, quantity: float, **kwargs) -> Order:
        """Create a market order"""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            **kwargs
        )

    @staticmethod
    def create_limit_order(symbol: str, side: OrderSide, quantity: float, price: float, **kwargs) -> Order:
        """Create a limit order"""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=OrderType.LIMIT,
            **kwargs
        )

    @staticmethod
    def create_stop_order(symbol: str, side: OrderSide, quantity: float, stop_price: float, **kwargs) -> Order:
        """Create a stop order"""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            stop_price=stop_price,
            order_type=OrderType.STOP,
            **kwargs
        )

    @staticmethod
    def create_twap_order(symbol: str, side: OrderSide, quantity: float,
                         duration_minutes: int = 60, **kwargs) -> TWAPOrder:
        """Create a TWAP order"""
        start_time = datetime.now()
        from datetime import timedelta
        end_time = start_time + timedelta(minutes=duration_minutes)

        return TWAPOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            **kwargs
        )

    @staticmethod
    def create_vwap_order(symbol: str, side: OrderSide, quantity: float,
                         end_time: Optional[datetime] = None, **kwargs) -> VWAPOrder:
        """Create a VWAP order"""
        return VWAPOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            end_time=end_time,
            **kwargs
        )

    @staticmethod
    def create_iceberg_order(symbol: str, side: OrderSide, quantity: float,
                            display_quantity: float, price: float, **kwargs) -> IcebergOrder:
        """Create an iceberg order"""
        return IcebergOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            display_quantity=display_quantity,
            price=price,
            **kwargs
        )

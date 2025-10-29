"""
Order validation for pre-submission checks

Validates orders against:
- Risk limits (position, exposure, leverage)
- Capital availability
- Regulatory compliance
- Market hours
- Order constraints
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import logging
from ..order_management.order_types import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of order validation"""

    def __init__(self, valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)

    def __str__(self):
        if self.valid:
            return "Valid" + (f" (Warnings: {', '.join(self.warnings)})" if self.warnings else "")
        return f"Invalid: {', '.join(self.errors)}"


class OrderValidator:
    """
    Validates orders before submission

    Performs comprehensive pre-trade validation including:
    - Risk limit checks
    - Capital availability
    - Regulatory compliance
    - Market conditions
    - Order constraints
    """

    def __init__(self, config: Dict = None):
        """
        Initialize order validator

        Args:
            config: Validation configuration
        """
        self.config = config or {}

        # Default limits
        self.max_order_value = self.config.get('max_order_value', 1_000_000)
        self.max_position_size = self.config.get('max_position_size', 10_000)
        self.max_daily_trades = self.config.get('max_daily_trades', 100)
        self.max_leverage = self.config.get('max_leverage', 4.0)
        self.min_order_value = self.config.get('min_order_value', 1.0)

        # Pattern day trader settings
        self.pdt_enabled = self.config.get('pdt_enabled', True)
        self.pdt_threshold = self.config.get('pdt_threshold', 25_000)

        # Market hours
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        self.allow_premarket = self.config.get('allow_premarket', False)
        self.allow_afterhours = self.config.get('allow_afterhours', False)

        # Tracking
        self.order_history: List[Order] = []
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

    def validate_order(self, order: Order, account_state: Dict = None) -> ValidationResult:
        """
        Validate an order comprehensively

        Args:
            order: Order to validate
            account_state: Current account state

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True)

        # Reset daily counters if new day
        self._reset_daily_counters()

        # Run all validation checks
        self._validate_order_fields(order, result)
        self._validate_symbol(order, result)
        self._validate_quantity(order, result)
        self._validate_price(order, result)
        self._validate_market_hours(order, result)
        self._validate_order_value(order, result)

        if account_state:
            self._validate_capital(order, account_state, result)
            self._validate_position_limits(order, account_state, result)
            self._validate_leverage(order, account_state, result)
            self._validate_pdt_rules(order, account_state, result)

        self._validate_daily_limits(order, result)
        self._validate_wash_trade(order, result)
        self._check_duplicate(order, result)

        if result.valid:
            logger.info(f"Order {order.order_id} validated successfully")
        else:
            logger.warning(f"Order {order.order_id} validation failed: {result}")

        return result

    def _validate_order_fields(self, order: Order, result: ValidationResult):
        """Validate required order fields"""
        if not order.symbol:
            result.add_error("Symbol is required")

        if order.quantity <= 0:
            result.add_error("Quantity must be positive")

        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                result.add_error(f"{order.order_type} requires valid price")

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                result.add_error(f"{order.order_type} requires valid stop price")

    def _validate_symbol(self, order: Order, result: ValidationResult):
        """Validate symbol"""
        # Check symbol format
        if len(order.symbol) > 10:
            result.add_error(f"Invalid symbol: {order.symbol}")

        # Check for common invalid symbols
        invalid_symbols = ['TEST', 'INVALID', 'NULL']
        if order.symbol.upper() in invalid_symbols:
            result.add_error(f"Invalid symbol: {order.symbol}")

    def _validate_quantity(self, order: Order, result: ValidationResult):
        """Validate order quantity"""
        if order.quantity <= 0:
            result.add_error("Quantity must be positive")

        # Check for fractional shares (if not allowed)
        if not self.config.get('allow_fractional', False):
            if order.quantity != int(order.quantity):
                result.add_error("Fractional shares not allowed")

        # Warn on very large quantities
        if order.quantity > 100_000:
            result.add_warning(f"Very large quantity: {order.quantity}")

    def _validate_price(self, order: Order, result: ValidationResult):
        """Validate order price"""
        if order.price is not None:
            if order.price <= 0:
                result.add_error("Price must be positive")

            # Price reasonability check
            if order.price > 1_000_000:
                result.add_warning(f"Unusually high price: {order.price}")

            if order.price < 0.01:
                result.add_warning(f"Unusually low price: {order.price}")

        if order.stop_price is not None:
            if order.stop_price <= 0:
                result.add_error("Stop price must be positive")

    def _validate_market_hours(self, order: Order, result: ValidationResult):
        """Validate market hours"""
        now = datetime.now().time()

        # Check if market is open
        is_market_hours = self.market_open <= now <= self.market_close

        if not is_market_hours:
            # Check extended hours
            is_premarket = time(4, 0) <= now < self.market_open
            is_afterhours = self.market_close < now <= time(20, 0)

            if is_premarket and not self.allow_premarket:
                result.add_error("Pre-market trading not allowed")
            elif is_afterhours and not self.allow_afterhours:
                result.add_error("After-hours trading not allowed")
            elif not (is_premarket or is_afterhours):
                result.add_error("Market is closed")

        # Check weekend
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            result.add_warning("Market closed on weekends")

    def _validate_order_value(self, order: Order, result: ValidationResult):
        """Validate order value"""
        # Estimate order value
        price = order.price if order.price else 0  # Market orders need market price
        order_value = price * order.quantity

        if price > 0:  # Only check if we have a price
            if order_value < self.min_order_value:
                result.add_error(f"Order value {order_value:.2f} below minimum {self.min_order_value}")

            if order_value > self.max_order_value:
                result.add_error(f"Order value {order_value:.2f} exceeds maximum {self.max_order_value}")

    def _validate_capital(self, order: Order, account_state: Dict, result: ValidationResult):
        """Validate sufficient capital"""
        buying_power = account_state.get('buying_power', 0)
        cash = account_state.get('cash', 0)

        # Estimate required capital
        price = order.price if order.price else account_state.get('last_price', 0)
        required_capital = price * order.quantity

        if order.side in [OrderSide.BUY]:
            if required_capital > buying_power:
                result.add_error(
                    f"Insufficient buying power: required {required_capital:.2f}, "
                    f"available {buying_power:.2f}"
                )

            if required_capital > cash and not self.config.get('allow_margin', False):
                result.add_warning("Order may use margin")

    def _validate_position_limits(self, order: Order, account_state: Dict, result: ValidationResult):
        """Validate position limits"""
        current_position = account_state.get('positions', {}).get(order.symbol, 0)

        # Calculate new position
        if order.side in [OrderSide.BUY]:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity

        # Check position limits
        if abs(new_position) > self.max_position_size:
            result.add_error(
                f"Position limit exceeded: new position {new_position}, "
                f"max {self.max_position_size}"
            )

        # Check for excessive concentration
        portfolio_value = account_state.get('portfolio_value', 1)
        position_value = abs(new_position) * (order.price or 0)
        concentration = position_value / portfolio_value if portfolio_value > 0 else 0

        if concentration > 0.25:
            result.add_warning(f"High concentration: {concentration:.1%} in {order.symbol}")

    def _validate_leverage(self, order: Order, account_state: Dict, result: ValidationResult):
        """Validate leverage limits"""
        equity = account_state.get('equity', 1)
        total_value = account_state.get('total_position_value', 0)

        # Calculate new total value
        order_value = (order.price or 0) * order.quantity
        if order.side in [OrderSide.BUY]:
            new_total_value = total_value + order_value
        else:
            new_total_value = total_value - order_value

        # Calculate leverage
        leverage = new_total_value / equity if equity > 0 else 0

        if leverage > self.max_leverage:
            result.add_error(
                f"Leverage limit exceeded: {leverage:.2f}x, max {self.max_leverage}x"
            )

    def _validate_pdt_rules(self, order: Order, account_state: Dict, result: ValidationResult):
        """Validate Pattern Day Trader rules"""
        if not self.pdt_enabled:
            return

        equity = account_state.get('equity', 0)
        day_trades_count = account_state.get('day_trades_count', 0)

        # Check if under PDT threshold
        if equity < self.pdt_threshold:
            if day_trades_count >= 3:
                result.add_error(
                    f"Pattern Day Trader rule: account under ${self.pdt_threshold:,.0f} "
                    f"limited to 3 day trades per 5 days"
                )

    def _validate_daily_limits(self, order: Order, result: ValidationResult):
        """Validate daily trading limits"""
        if self.daily_trade_count >= self.max_daily_trades:
            result.add_error(
                f"Daily trade limit reached: {self.daily_trade_count}/{self.max_daily_trades}"
            )

    def _validate_wash_trade(self, order: Order, result: ValidationResult):
        """Check for potential wash trades"""
        # Look for recent trades in same symbol
        recent_trades = [
            o for o in self.order_history[-100:]
            if o.symbol == order.symbol
            and (datetime.now() - o.create_time).seconds < 30
        ]

        # Check for buy-sell or sell-buy within 30 seconds
        if recent_trades:
            last_trade = recent_trades[-1]
            if (order.side == OrderSide.BUY and last_trade.side == OrderSide.SELL) or \
               (order.side == OrderSide.SELL and last_trade.side == OrderSide.BUY):
                result.add_warning(
                    f"Potential wash trade detected: opposite trade in {order.symbol} "
                    f"within 30 seconds"
                )

    def _check_duplicate(self, order: Order, result: ValidationResult):
        """Check for duplicate orders"""
        # Look for identical recent orders
        duplicates = [
            o for o in self.order_history[-50:]
            if o.symbol == order.symbol
            and o.side == order.side
            and o.quantity == order.quantity
            and abs(o.price - order.price) < 0.01 if o.price and order.price else False
            and (datetime.now() - o.create_time).seconds < 10
        ]

        if duplicates:
            result.add_warning(
                f"Potential duplicate order: similar order in {order.symbol} "
                f"within 10 seconds"
            )

    def _reset_daily_counters(self):
        """Reset daily counters at start of new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = today

    def record_order(self, order: Order):
        """Record order for validation history"""
        self.order_history.append(order)
        self.daily_trade_count += 1

        # Keep only recent history
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-500:]

    def update_config(self, config: Dict):
        """Update validator configuration"""
        self.config.update(config)

        # Update limits
        self.max_order_value = self.config.get('max_order_value', self.max_order_value)
        self.max_position_size = self.config.get('max_position_size', self.max_position_size)
        self.max_daily_trades = self.config.get('max_daily_trades', self.max_daily_trades)
        self.max_leverage = self.config.get('max_leverage', self.max_leverage)

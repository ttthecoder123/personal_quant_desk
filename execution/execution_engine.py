"""
Main Execution Engine

Central coordinator for the entire execution and OMS system.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Order management
from .order_management import (
    OrderManager, Order, OrderStatus, OrderType, Fill
)

# Execution algorithms
from .execution_algorithms import (
    TWAPAlgorithm, VWAPAlgorithm, ImplementationShortfallAlgorithm
)

# Broker connectors
from .broker_connectors import (
    BaseBrokerConnector, PaperTradingConnector
)

# Smart routing
from .smart_routing.venue_selector import VenueSelector

# Pre-trade analytics
from .pre_trade.cost_estimator import CostEstimator
from .pre_trade.impact_model import ImpactModel

# Post-trade analytics
from .post_trade.tca_engine import TCAEngine
from .post_trade.execution_analytics import ExecutionAnalytics

# Monitoring
from .monitoring.real_time_monitor import RealTimeMonitor
from .monitoring.alert_system import AlertSystem

# State management
from .state_management.position_keeper import PositionKeeper
from .state_management.order_book import OrderBook
from .state_management.audit_trail import AuditTrail

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Main Execution Engine

    Central coordinator integrating:
    - Order management system
    - Execution algorithms
    - Broker connectivity
    - Smart routing
    - Pre-trade and post-trade analytics
    - Real-time monitoring
    - State management

    Integrates with:
    - Step 5: Strategy signals and sizing
    - Step 6: Risk limits and checks
    - Step 7: Backtested parameters
    """

    def __init__(self, config: Dict = None):
        """
        Initialize execution engine

        Args:
            config: Engine configuration
        """
        self.config = config or {}

        logger.info("Initializing Execution Engine...")

        # Core OMS
        self.order_manager = OrderManager(self.config.get('order_management', {}))

        # Execution algorithms
        self.twap_algo = TWAPAlgorithm(self.config.get('twap', {}))
        self.vwap_algo = VWAPAlgorithm(self.config.get('vwap', {}))
        self.is_algo = ImplementationShortfallAlgorithm(self.config.get('is', {}))

        # Smart routing
        self.venue_selector = VenueSelector(self.config.get('routing', {}))

        # Pre-trade analytics
        self.cost_estimator = CostEstimator(self.config.get('pre_trade', {}))
        self.impact_model = ImpactModel(self.config.get('pre_trade', {}))

        # Post-trade analytics
        self.tca_engine = TCAEngine(self.config.get('post_trade', {}))
        self.execution_analytics = ExecutionAnalytics(self.config.get('post_trade', {}))

        # Monitoring
        self.monitor = RealTimeMonitor(self.config.get('monitoring', {}))
        self.alert_system = AlertSystem(self.config.get('alerts', {}))

        # State management
        self.position_keeper = PositionKeeper(self.config.get('positions', {}))
        self.order_book = OrderBook(self.config.get('order_book', {}))
        self.audit_trail = AuditTrail(self.config.get('audit', {}))

        # Broker connectors
        self._initialize_brokers()

        # Register callbacks
        self._register_callbacks()

        # State
        self.running = False
        self.start_time: Optional[datetime] = None

        logger.info("Execution Engine initialized successfully")

    def _initialize_brokers(self):
        """Initialize broker connectors"""
        # Default to paper trading
        paper_broker = PaperTradingConnector(self.config.get('paper_trading', {}))
        paper_broker.connect()
        paper_broker.authenticate()
        self.order_manager.add_broker_connector('paper_trading', paper_broker)

        logger.info("Broker connectors initialized")

    def _register_callbacks(self):
        """Register callbacks for order updates"""
        def on_order_update(order_id: str, status: OrderStatus, message: str):
            logger.info(f"Order update: {order_id} -> {status.value}")
            self.monitor.update_metrics('last_order_update', datetime.now().timestamp())

        def on_fill(order_id: str, fill: Fill):
            logger.info(f"Fill: {order_id} - {fill.quantity} @ {fill.price}")
            # Update positions
            order = self.order_manager.get_order(order_id)
            if order:
                qty_change = fill.quantity if order.side.value in ['BUY'] else -fill.quantity
                self.position_keeper.update_position(order.symbol, qty_change)

        # Register with all brokers
        for broker in self.order_manager.broker_connectors.values():
            broker.register_order_callback(on_order_update)
            broker.register_fill_callback(on_fill)

    def start(self):
        """Start execution engine"""
        logger.info("Starting Execution Engine...")
        self.running = True
        self.start_time = datetime.now()
        self.audit_trail.log_event('ENGINE_START', {'timestamp': self.start_time})

    def stop(self):
        """Stop execution engine"""
        logger.info("Stopping Execution Engine...")
        self.running = False
        self.audit_trail.log_event('ENGINE_STOP', {'timestamp': datetime.now()})

    # ====================
    # ORDER SUBMISSION
    # ====================

    def submit_order(self, order: Order, validate: bool = True) -> Tuple[bool, str]:
        """
        Submit order for execution

        Args:
            order: Order to submit
            validate: Whether to validate

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Submitting order: {order.symbol} {order.side.value} {order.quantity}")

        # Get account state
        account_state = self._get_account_state()

        # Pre-trade analysis
        if order.price and validate:
            cost_estimate = self.cost_estimator.estimate_cost(
                order.symbol, order.quantity, order.price
            )
            logger.info(f"Estimated cost: {cost_estimate['cost_bps']:.2f} bps")

        # Submit through order manager
        success, message = self.order_manager.submit_order(order, account_state, validate)

        # Log to audit trail
        self.audit_trail.log_event('ORDER_SUBMIT', {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'success': success
        })

        return success, message

    def modify_order(self, order_id: str, modifications: Dict) -> Tuple[bool, str]:
        """Modify order"""
        logger.info(f"Modifying order {order_id}")
        success, message = self.order_manager.modify_order(order_id, modifications)
        self.audit_trail.log_event('ORDER_MODIFY', {'order_id': order_id, 'success': success})
        return success, message

    def cancel_order(self, order_id: str, reason: str = "User cancelled") -> Tuple[bool, str]:
        """Cancel order"""
        logger.info(f"Cancelling order {order_id}")
        success, message = self.order_manager.cancel_order(order_id, reason)
        self.audit_trail.log_event('ORDER_CANCEL', {'order_id': order_id, 'success': success})
        return success, message

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Tuple[bool, str]]:
        """Cancel all orders"""
        logger.info("Cancelling all orders")
        return self.order_manager.cancel_all_orders(symbol)

    def emergency_close_all(self) -> Dict:
        """Emergency close all positions"""
        logger.warning("EMERGENCY CLOSE ALL POSITIONS")
        self.alert_system.add_alert('EMERGENCY', 'Emergency close all triggered', 'CRITICAL')

        results = {}
        positions = self.position_keeper.get_all_positions()

        for symbol, quantity in positions.items():
            if abs(quantity) > 0:
                # Create market order to close
                from .order_management.order_types import OrderFactory, OrderSide
                side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
                close_order = OrderFactory.create_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity)
                )
                success, message = self.submit_order(close_order, validate=False)
                results[symbol] = (success, message)

        self.audit_trail.log_event('EMERGENCY_CLOSE', {'positions': positions})
        return results

    # ====================
    # QUERIES
    # ====================

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.order_manager.get_order(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders"""
        return self.order_manager.get_active_orders(symbol)

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history"""
        return self.order_manager.get_order_history(symbol, limit)

    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.position_keeper.get_all_positions()

    def get_position(self, symbol: str) -> float:
        """Get position for symbol"""
        return self.position_keeper.get_position(symbol)

    # ====================
    # ANALYTICS
    # ====================

    def generate_tca_report(self, order_id: str) -> Dict:
        """
        Generate Transaction Cost Analysis report

        Args:
            order_id: Order ID to analyze

        Returns:
            TCA report
        """
        order = self.order_manager.get_order(order_id)
        if not order:
            return {'error': 'Order not found'}

        # Use decision price as benchmark
        benchmark_price = order.price if order.price else 0

        tca = self.tca_engine.analyze_execution(order_id, order.fills, benchmark_price)

        logger.info(f"TCA Report for {order_id}: slippage = {tca.get('slippage_bps', 0):.2f} bps")
        return tca

    def get_execution_analytics(self) -> Dict:
        """Get execution quality analytics"""
        return self.execution_analytics.analyze_quality([])

    # ====================
    # SYSTEM STATUS
    # ====================

    def system_health_check(self) -> Dict:
        """
        Comprehensive system health check

        Returns:
            Health status dict
        """
        health = {
            'status': 'healthy' if self.running else 'stopped',
            'start_time': self.start_time,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'oms_health': self.order_manager.check_system_health(),
            'active_orders': len(self.order_manager.get_active_orders()),
            'positions': len([p for p in self.position_keeper.get_all_positions().values() if abs(p) > 0]),
            'alerts': len(self.alert_system.get_alerts())
        }

        logger.info(f"Health check: {health['status']}")
        return health

    def get_dashboard_data(self) -> Dict:
        """Get real-time dashboard data"""
        return {
            'health': self.system_health_check(),
            'active_orders': [o.to_dict() for o in self.get_active_orders()],
            'positions': self.get_positions(),
            'metrics': self.order_manager.get_metrics(),
            'alerts': self.alert_system.get_alerts()
        }

    def _get_account_state(self) -> Dict:
        """Get current account state"""
        # Get from primary broker (paper trading for now)
        broker = self.order_manager.broker_connectors.get('paper_trading')
        if broker:
            account_info = broker.get_account_info()
            account_info['positions'] = self.position_keeper.get_all_positions()
            return account_info
        return {}

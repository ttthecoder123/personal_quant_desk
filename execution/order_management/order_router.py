"""
Smart Order Router

Routes orders to optimal brokers/venues based on:
- Cost optimization
- Latency requirements
- Liquidity availability
- Regulatory compliance
"""

from typing import Dict, List, Optional
from ..order_management.order_types import Order, OrderType
import logging

logger = logging.getLogger(__name__)


class RoutingDecision:
    """Routing decision for an order"""

    def __init__(self, broker: str, venue: Optional[str] = None,
                 split: Optional[List[Dict]] = None, reason: str = ""):
        self.broker = broker
        self.venue = venue
        self.split = split  # List of {broker, venue, quantity} for split orders
        self.reason = reason
        self.timestamp = None

    def __str__(self):
        if self.split:
            return f"Split routing: {len(self.split)} parts - {self.reason}"
        return f"Route to {self.broker}" + (f"/{self.venue}" if self.venue else "") + f" - {self.reason}"


class OrderRouter:
    """
    Smart order routing engine

    Routes orders to optimal execution venues based on:
    - Asset class and symbol
    - Order characteristics
    - Cost analysis
    - Latency requirements
    - Available liquidity
    """

    def __init__(self, config: Dict = None):
        """
        Initialize order router

        Args:
            config: Router configuration
        """
        self.config = config or {}

        # Broker capabilities
        self.broker_capabilities = self.config.get('broker_capabilities', {
            'interactive_brokers': {
                'asset_classes': ['EQUITY', 'OPTION', 'FUTURE', 'FOREX', 'CRYPTO'],
                'exchanges': ['NASDAQ', 'NYSE', 'ARCA', 'BATS'],
                'max_order_size': 1_000_000,
                'commission_per_share': 0.005,
                'supports_algo': True,
                'avg_latency_ms': 50
            },
            'alpaca': {
                'asset_classes': ['EQUITY', 'CRYPTO'],
                'exchanges': ['NASDAQ', 'NYSE'],
                'max_order_size': 100_000,
                'commission_per_share': 0.0,
                'supports_algo': False,
                'avg_latency_ms': 100
            },
            'paper_trading': {
                'asset_classes': ['EQUITY', 'OPTION', 'FUTURE', 'FOREX', 'CRYPTO'],
                'exchanges': ['ALL'],
                'max_order_size': float('inf'),
                'commission_per_share': 0.0,
                'supports_algo': True,
                'avg_latency_ms': 1
            }
        })

        # Routing rules
        self.routing_rules = self.config.get('routing_rules', {})

        # Performance tracking
        self.broker_performance = {}

    def route_order(self, order: Order, account_state: Dict = None) -> RoutingDecision:
        """
        Determine optimal routing for an order

        Args:
            order: Order to route
            account_state: Current account state

        Returns:
            RoutingDecision with routing instructions
        """
        logger.info(f"Routing order {order.order_id} for {order.symbol}")

        # Check for manual routing override
        if order.broker:
            logger.info(f"Using manual broker override: {order.broker}")
            return RoutingDecision(
                broker=order.broker,
                venue=order.venue,
                reason="Manual override"
            )

        # Apply routing rules
        decision = self._apply_routing_rules(order)
        if decision:
            return decision

        # Asset class routing
        eligible_brokers = self._filter_by_asset_class(order)
        if not eligible_brokers:
            logger.error(f"No eligible brokers for asset class {order.asset_class}")
            return RoutingDecision(broker='paper_trading', reason="Fallback - no eligible brokers")

        # Filter by capabilities
        eligible_brokers = self._filter_by_capabilities(order, eligible_brokers)

        # Order size routing
        if order.quantity > 10_000:
            eligible_brokers = self._filter_by_size(order, eligible_brokers)

        # Select best broker
        if len(eligible_brokers) == 1:
            broker = eligible_brokers[0]
            return RoutingDecision(
                broker=broker,
                reason=f"Only eligible broker for {order.asset_class}"
            )

        # Cost-based routing
        best_broker = self._select_by_cost(order, eligible_brokers)
        if best_broker:
            return RoutingDecision(
                broker=best_broker,
                reason="Lowest estimated cost"
            )

        # Latency-based routing (for urgent orders)
        if order.order_type == OrderType.MARKET:
            best_broker = self._select_by_latency(eligible_brokers)
            return RoutingDecision(
                broker=best_broker,
                reason="Lowest latency for market order"
            )

        # Default routing
        broker = eligible_brokers[0]
        return RoutingDecision(broker=broker, reason="Default routing")

    def _apply_routing_rules(self, order: Order) -> Optional[RoutingDecision]:
        """Apply custom routing rules"""
        for rule_name, rule in self.routing_rules.items():
            if self._matches_rule(order, rule):
                logger.info(f"Applied routing rule: {rule_name}")
                return RoutingDecision(
                    broker=rule.get('broker'),
                    venue=rule.get('venue'),
                    reason=f"Rule: {rule_name}"
                )
        return None

    def _matches_rule(self, order: Order, rule: Dict) -> bool:
        """Check if order matches routing rule"""
        # Symbol match
        if 'symbols' in rule:
            if order.symbol not in rule['symbols']:
                return False

        # Asset class match
        if 'asset_class' in rule:
            if order.asset_class != rule['asset_class']:
                return False

        # Order type match
        if 'order_types' in rule:
            if order.order_type not in rule['order_types']:
                return False

        return True

    def _filter_by_asset_class(self, order: Order) -> List[str]:
        """Filter brokers by asset class support"""
        eligible = []
        for broker, caps in self.broker_capabilities.items():
            if order.asset_class in caps['asset_classes']:
                eligible.append(broker)
        return eligible

    def _filter_by_capabilities(self, order: Order, brokers: List[str]) -> List[str]:
        """Filter brokers by required capabilities"""
        eligible = []

        for broker in brokers:
            caps = self.broker_capabilities[broker]

            # Check algo support for algo orders
            if order.order_type in [OrderType.TWAP, OrderType.VWAP, OrderType.IMPLEMENTATION_SHORTFALL]:
                if not caps.get('supports_algo', False):
                    continue

            eligible.append(broker)

        return eligible

    def _filter_by_size(self, order: Order, brokers: List[str]) -> List[str]:
        """Filter brokers by order size limits"""
        eligible = []
        for broker in brokers:
            caps = self.broker_capabilities[broker]
            if order.quantity <= caps['max_order_size']:
                eligible.append(broker)
        return eligible if eligible else brokers  # Return all if none qualify

    def _select_by_cost(self, order: Order, brokers: List[str]) -> Optional[str]:
        """Select broker with lowest cost"""
        best_broker = None
        best_cost = float('inf')

        for broker in brokers:
            caps = self.broker_capabilities[broker]
            estimated_cost = order.quantity * caps['commission_per_share']

            if estimated_cost < best_cost:
                best_cost = estimated_cost
                best_broker = broker

        return best_broker

    def _select_by_latency(self, brokers: List[str]) -> str:
        """Select broker with lowest latency"""
        best_broker = brokers[0]
        best_latency = float('inf')

        for broker in brokers:
            caps = self.broker_capabilities[broker]
            latency = caps['avg_latency_ms']

            if latency < best_latency:
                best_latency = latency
                best_broker = broker

        return best_broker

    def update_broker_performance(self, broker: str, metrics: Dict):
        """Update broker performance metrics"""
        if broker not in self.broker_performance:
            self.broker_performance[broker] = {
                'orders': 0,
                'fills': 0,
                'rejections': 0,
                'avg_latency': 0,
                'avg_slippage': 0
            }

        perf = self.broker_performance[broker]
        perf['orders'] += 1

        if 'filled' in metrics:
            perf['fills'] += 1
        if 'rejected' in metrics:
            perf['rejections'] += 1
        if 'latency' in metrics:
            perf['avg_latency'] = (perf['avg_latency'] * (perf['orders'] - 1) + metrics['latency']) / perf['orders']

    def get_broker_performance(self, broker: str) -> Dict:
        """Get performance statistics for a broker"""
        return self.broker_performance.get(broker, {})

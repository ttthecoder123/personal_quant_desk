"""
Execution Monitor

Fill rate monitoring, slippage tracking, rejection rate monitoring,
order latency tracking, venue performance, algorithm performance,
and cost analysis.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np


@dataclass
class OrderExecution:
    """Order execution information."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', etc.
    quantity: float
    limit_price: Optional[float]
    fill_quantity: float
    fill_price: float
    status: str  # 'filled', 'partial', 'rejected', 'cancelled'
    venue: str
    algorithm: Optional[str] = None
    latency_ms: Optional[float] = None
    rejection_reason: Optional[str] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate."""
        return self.fill_quantity / self.quantity if self.quantity > 0 else 0.0

    @property
    def slippage(self) -> Optional[float]:
        """Calculate slippage."""
        if self.limit_price is None or self.fill_quantity == 0:
            return None
        if self.side == 'buy':
            return self.fill_price - self.limit_price
        else:
            return self.limit_price - self.fill_price

    @property
    def slippage_bps(self) -> Optional[float]:
        """Calculate slippage in basis points."""
        slippage = self.slippage
        if slippage is None or self.limit_price == 0:
            return None
        return (slippage / self.limit_price) * 10000

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == 'filled'

    @property
    def is_rejected(self) -> bool:
        """Check if order is rejected."""
        return self.status == 'rejected'


@dataclass
class VenuePerformance:
    """Performance metrics for a trading venue."""
    venue: str
    total_orders: int
    filled_orders: int
    rejected_orders: int
    avg_fill_rate: float
    avg_latency_ms: float
    avg_slippage_bps: float
    rejection_rate: float
    total_volume: float
    timestamp: datetime


@dataclass
class AlgorithmPerformance:
    """Performance metrics for an execution algorithm."""
    algorithm: str
    total_orders: int
    filled_orders: int
    avg_fill_rate: float
    avg_slippage_bps: float
    avg_latency_ms: float
    participation_rate: float
    timestamp: datetime


class ExecutionMonitor:
    """
    Comprehensive execution monitoring.

    Features:
    - Fill rate monitoring
    - Slippage tracking
    - Rejection rate monitoring
    - Order latency tracking
    - Venue performance analysis
    - Algorithm performance analysis
    - Cost analysis (commission, slippage)
    - Market impact estimation
    - Execution quality metrics
    """

    def __init__(self, update_interval: int = 5):
        """
        Initialize execution monitor.

        Args:
            update_interval: Seconds between metric updates
        """
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Execution tracking
        self.executions: deque = deque(maxlen=100000)
        self.pending_orders: Dict[str, OrderExecution] = {}

        # Performance metrics
        self.venue_performance: Dict[str, VenuePerformance] = {}
        self.algorithm_performance: Dict[str, AlgorithmPerformance] = {}
        self.performance_history: deque = deque(maxlen=10000)

        # Statistics
        self.fill_rate_history: deque = deque(maxlen=1000)
        self.slippage_history: deque = deque(maxlen=1000)
        self.latency_history: deque = deque(maxlen=1000)
        self.rejection_history: deque = deque(maxlen=1000)
        self.cost_history: deque = deque(maxlen=1000)

        # Alerts
        self.alerts: deque = deque(maxlen=1000)

        # Thresholds
        self.max_slippage_bps = 10.0
        self.max_latency_ms = 1000.0
        self.max_rejection_rate = 0.05

    def start(self):
        """Start execution monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop execution monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def record_execution(self, execution: OrderExecution):
        """
        Record an order execution.

        Args:
            execution: OrderExecution object
        """
        with self.lock:
            self.executions.append(execution)

            # Update pending orders
            if execution.order_id in self.pending_orders:
                del self.pending_orders[execution.order_id]

            # Check for alerts
            self._check_execution_alerts(execution)

    def record_order_submission(self, order_id: str, execution: OrderExecution):
        """
        Record order submission for latency tracking.

        Args:
            order_id: Order ID
            execution: OrderExecution object
        """
        with self.lock:
            self.pending_orders[order_id] = execution

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._calculate_metrics()
                self._calculate_venue_performance()
                self._calculate_algorithm_performance()
                self._calculate_cost_analysis()

                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in execution monitor loop: {e}")

    def _calculate_metrics(self):
        """Calculate execution metrics."""
        with self.lock:
            if not self.executions:
                return

            # Get recent executions (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_executions = [e for e in self.executions if e.timestamp >= cutoff]

            if not recent_executions:
                return

            # Fill rate
            filled = [e for e in recent_executions if e.is_filled]
            fill_rate = len(filled) / len(recent_executions) if recent_executions else 0

            self.fill_rate_history.append({
                'timestamp': datetime.now(),
                'fill_rate': fill_rate,
                'total_orders': len(recent_executions),
                'filled_orders': len(filled)
            })

            # Slippage
            slippages = [e.slippage_bps for e in recent_executions if e.slippage_bps is not None]
            avg_slippage = np.mean(slippages) if slippages else 0

            self.slippage_history.append({
                'timestamp': datetime.now(),
                'avg_slippage_bps': avg_slippage,
                'max_slippage_bps': max(slippages) if slippages else 0,
                'min_slippage_bps': min(slippages) if slippages else 0
            })

            # Latency
            latencies = [e.latency_ms for e in recent_executions if e.latency_ms is not None]
            avg_latency = np.mean(latencies) if latencies else 0

            self.latency_history.append({
                'timestamp': datetime.now(),
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max(latencies) if latencies else 0,
                'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
                'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0
            })

            # Rejections
            rejected = [e for e in recent_executions if e.is_rejected]
            rejection_rate = len(rejected) / len(recent_executions) if recent_executions else 0

            self.rejection_history.append({
                'timestamp': datetime.now(),
                'rejection_rate': rejection_rate,
                'total_rejections': len(rejected),
                'rejection_reasons': self._count_rejection_reasons(rejected)
            })

    def _calculate_venue_performance(self):
        """Calculate performance by venue."""
        with self.lock:
            if not self.executions:
                return

            # Get recent executions (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_executions = [e for e in self.executions if e.timestamp >= cutoff]

            # Group by venue
            venue_executions = defaultdict(list)
            for execution in recent_executions:
                venue_executions[execution.venue].append(execution)

            # Calculate metrics for each venue
            for venue, executions in venue_executions.items():
                total_orders = len(executions)
                filled_orders = sum(1 for e in executions if e.is_filled)
                rejected_orders = sum(1 for e in executions if e.is_rejected)

                fill_rates = [e.fill_rate for e in executions]
                avg_fill_rate = np.mean(fill_rates) if fill_rates else 0

                latencies = [e.latency_ms for e in executions if e.latency_ms is not None]
                avg_latency = np.mean(latencies) if latencies else 0

                slippages = [e.slippage_bps for e in executions if e.slippage_bps is not None]
                avg_slippage = np.mean(slippages) if slippages else 0

                rejection_rate = rejected_orders / total_orders if total_orders > 0 else 0

                total_volume = sum(e.fill_quantity * e.fill_price for e in executions)

                self.venue_performance[venue] = VenuePerformance(
                    venue=venue,
                    total_orders=total_orders,
                    filled_orders=filled_orders,
                    rejected_orders=rejected_orders,
                    avg_fill_rate=avg_fill_rate,
                    avg_latency_ms=avg_latency,
                    avg_slippage_bps=avg_slippage,
                    rejection_rate=rejection_rate,
                    total_volume=total_volume,
                    timestamp=datetime.now()
                )

    def _calculate_algorithm_performance(self):
        """Calculate performance by algorithm."""
        with self.lock:
            if not self.executions:
                return

            # Get recent executions (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_executions = [e for e in self.executions if e.timestamp >= cutoff and e.algorithm]

            # Group by algorithm
            algo_executions = defaultdict(list)
            for execution in recent_executions:
                if execution.algorithm:
                    algo_executions[execution.algorithm].append(execution)

            # Calculate metrics for each algorithm
            for algorithm, executions in algo_executions.items():
                total_orders = len(executions)
                filled_orders = sum(1 for e in executions if e.is_filled)

                fill_rates = [e.fill_rate for e in executions]
                avg_fill_rate = np.mean(fill_rates) if fill_rates else 0

                slippages = [e.slippage_bps for e in executions if e.slippage_bps is not None]
                avg_slippage = np.mean(slippages) if slippages else 0

                latencies = [e.latency_ms for e in executions if e.latency_ms is not None]
                avg_latency = np.mean(latencies) if latencies else 0

                # Calculate participation rate (simplified)
                participation_rate = sum(e.fill_quantity for e in executions) / sum(e.quantity for e in executions) if executions else 0

                self.algorithm_performance[algorithm] = AlgorithmPerformance(
                    algorithm=algorithm,
                    total_orders=total_orders,
                    filled_orders=filled_orders,
                    avg_fill_rate=avg_fill_rate,
                    avg_slippage_bps=avg_slippage,
                    avg_latency_ms=avg_latency,
                    participation_rate=participation_rate,
                    timestamp=datetime.now()
                )

    def _calculate_cost_analysis(self):
        """Calculate execution costs."""
        with self.lock:
            if not self.executions:
                return

            # Get recent executions (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_executions = [e for e in self.executions if e.timestamp >= cutoff]

            if not recent_executions:
                return

            # Calculate costs
            total_commission = sum(e.commission for e in recent_executions)
            total_volume = sum(e.fill_quantity * e.fill_price for e in recent_executions)

            # Slippage cost
            slippage_costs = []
            for e in recent_executions:
                if e.slippage is not None:
                    cost = e.slippage * e.fill_quantity
                    slippage_costs.append(cost)

            total_slippage_cost = sum(slippage_costs)

            # Calculate basis points
            commission_bps = (total_commission / total_volume * 10000) if total_volume > 0 else 0
            slippage_bps = (total_slippage_cost / total_volume * 10000) if total_volume > 0 else 0
            total_cost_bps = commission_bps + slippage_bps

            self.cost_history.append({
                'timestamp': datetime.now(),
                'total_commission': total_commission,
                'total_slippage_cost': total_slippage_cost,
                'total_volume': total_volume,
                'commission_bps': commission_bps,
                'slippage_bps': slippage_bps,
                'total_cost_bps': total_cost_bps
            })

    def _check_execution_alerts(self, execution: OrderExecution):
        """
        Check execution for alert conditions.

        Args:
            execution: OrderExecution object
        """
        # Slippage alert
        if execution.slippage_bps is not None and abs(execution.slippage_bps) > self.max_slippage_bps:
            self.alerts.append({
                'timestamp': datetime.now(),
                'type': 'high_slippage',
                'severity': 'warning',
                'message': f"High slippage on {execution.symbol}: {execution.slippage_bps:.2f} bps",
                'execution': execution
            })

        # Latency alert
        if execution.latency_ms is not None and execution.latency_ms > self.max_latency_ms:
            self.alerts.append({
                'timestamp': datetime.now(),
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"High latency on {execution.symbol}: {execution.latency_ms:.2f} ms",
                'execution': execution
            })

        # Rejection alert
        if execution.is_rejected:
            self.alerts.append({
                'timestamp': datetime.now(),
                'type': 'order_rejection',
                'severity': 'critical',
                'message': f"Order rejected for {execution.symbol}: {execution.rejection_reason}",
                'execution': execution
            })

    def _count_rejection_reasons(self, rejected_executions: List[OrderExecution]) -> Dict[str, int]:
        """
        Count rejection reasons.

        Args:
            rejected_executions: List of rejected executions

        Returns:
            Dictionary of rejection reason counts
        """
        reasons = defaultdict(int)
        for execution in rejected_executions:
            if execution.rejection_reason:
                reasons[execution.rejection_reason] += 1
        return dict(reasons)

    def get_fill_rate_metrics(self) -> Dict[str, Any]:
        """
        Get fill rate metrics.

        Returns:
            Dictionary of fill rate metrics
        """
        with self.lock:
            if not self.fill_rate_history:
                return {}
            return self.fill_rate_history[-1]

    def get_slippage_metrics(self) -> Dict[str, Any]:
        """
        Get slippage metrics.

        Returns:
            Dictionary of slippage metrics
        """
        with self.lock:
            if not self.slippage_history:
                return {}
            return self.slippage_history[-1]

    def get_latency_metrics(self) -> Dict[str, Any]:
        """
        Get latency metrics.

        Returns:
            Dictionary of latency metrics
        """
        with self.lock:
            if not self.latency_history:
                return {}
            return self.latency_history[-1]

    def get_rejection_metrics(self) -> Dict[str, Any]:
        """
        Get rejection metrics.

        Returns:
            Dictionary of rejection metrics
        """
        with self.lock:
            if not self.rejection_history:
                return {}
            return self.rejection_history[-1]

    def get_cost_metrics(self) -> Dict[str, Any]:
        """
        Get cost metrics.

        Returns:
            Dictionary of cost metrics
        """
        with self.lock:
            if not self.cost_history:
                return {}
            return self.cost_history[-1]

    def get_venue_performance(self, venue: Optional[str] = None) -> Dict[str, VenuePerformance]:
        """
        Get venue performance metrics.

        Args:
            venue: Optional specific venue

        Returns:
            Dictionary of venue performance
        """
        with self.lock:
            if venue:
                return {venue: self.venue_performance.get(venue)}
            return self.venue_performance.copy()

    def get_algorithm_performance(self, algorithm: Optional[str] = None) -> Dict[str, AlgorithmPerformance]:
        """
        Get algorithm performance metrics.

        Args:
            algorithm: Optional specific algorithm

        Returns:
            Dictionary of algorithm performance
        """
        with self.lock:
            if algorithm:
                return {algorithm: self.algorithm_performance.get(algorithm)}
            return self.algorithm_performance.copy()

    def get_alerts(self, alert_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            alert_type: Filter by alert type
            limit: Maximum number of alerts

        Returns:
            List of alerts
        """
        with self.lock:
            alerts = list(self.alerts)
            if alert_type:
                alerts = [a for a in alerts if a['type'] == alert_type]
            return alerts[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            fill_rate = self.get_fill_rate_metrics()
            slippage = self.get_slippage_metrics()
            latency = self.get_latency_metrics()
            rejection = self.get_rejection_metrics()
            cost = self.get_cost_metrics()

            return {
                'total_executions': len(self.executions),
                'pending_orders': len(self.pending_orders),
                'fill_rate': fill_rate.get('fill_rate', 0),
                'avg_slippage_bps': slippage.get('avg_slippage_bps', 0),
                'avg_latency_ms': latency.get('avg_latency_ms', 0),
                'rejection_rate': rejection.get('rejection_rate', 0),
                'total_cost_bps': cost.get('total_cost_bps', 0),
                'total_venues': len(self.venue_performance),
                'total_algorithms': len(self.algorithm_performance),
                'alerts_count': len(self.alerts),
                'timestamp': datetime.now().isoformat()
            }

    def get_execution_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """
        Get executions as DataFrame.

        Args:
            hours: Hours of history to include

        Returns:
            DataFrame of executions
        """
        with self.lock:
            if not self.executions:
                return pd.DataFrame()

            cutoff = datetime.now() - timedelta(hours=hours)
            recent_executions = [e for e in self.executions if e.timestamp >= cutoff]

            data = []
            for execution in recent_executions:
                data.append({
                    'timestamp': execution.timestamp,
                    'order_id': execution.order_id,
                    'symbol': execution.symbol,
                    'side': execution.side,
                    'order_type': execution.order_type,
                    'quantity': execution.quantity,
                    'fill_quantity': execution.fill_quantity,
                    'fill_rate': execution.fill_rate,
                    'fill_price': execution.fill_price,
                    'slippage_bps': execution.slippage_bps,
                    'latency_ms': execution.latency_ms,
                    'status': execution.status,
                    'venue': execution.venue,
                    'algorithm': execution.algorithm,
                    'commission': execution.commission
                })

            return pd.DataFrame(data)

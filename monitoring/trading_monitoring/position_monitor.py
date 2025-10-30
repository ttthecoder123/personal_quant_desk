"""
Position Monitor

Real-time position tracking, position limit monitoring, exposure calculation,
position age tracking, concentration monitoring, cross-broker reconciliation,
and position drift detection.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np


@dataclass
class Position:
    """Position information."""
    symbol: str
    quantity: float
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    broker: str
    timestamp: datetime
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        """Calculate market value."""
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == 'long':
            return self.quantity * (self.current_price - self.entry_price)
        else:
            return abs(self.quantity) * (self.entry_price - self.current_price)

    @property
    def age(self) -> timedelta:
        """Calculate position age."""
        return datetime.now() - self.timestamp


@dataclass
class PositionLimit:
    """Position limit configuration."""
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    max_quantity: Optional[float] = None
    max_notional: Optional[float] = None
    max_concentration: Optional[float] = None  # As percentage
    max_age_hours: Optional[int] = None


@dataclass
class PositionAlert:
    """Position alert information."""
    alert_type: str
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: datetime
    position: Optional[Position] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionMonitor:
    """
    Comprehensive position monitoring.

    Features:
    - Real-time position tracking
    - Position limit monitoring
    - Exposure calculation (gross, net, delta)
    - Position age tracking
    - Concentration monitoring
    - Cross-broker reconciliation
    - Position drift detection
    - Position limit alerts
    """

    def __init__(self, update_interval: int = 1):
        """
        Initialize position monitor.

        Args:
            update_interval: Seconds between position updates
        """
        self.update_interval = update_interval
        self.positions: Dict[Tuple[str, str], Position] = {}  # (symbol, broker) -> Position
        self.position_history: deque = deque(maxlen=10000)
        self.limits: List[PositionLimit] = []
        self.alerts: deque = deque(maxlen=1000)
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Tracking metrics
        self.exposure_history: deque = deque(maxlen=1000)
        self.concentration_history: deque = deque(maxlen=1000)
        self.drift_events: deque = deque(maxlen=500)

        # Configuration
        self.drift_threshold = 0.05  # 5% drift threshold
        self.reconciliation_interval = 60  # seconds
        self.last_reconciliation = datetime.now()

    def start(self):
        """Start position monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop position monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def update_position(self, position: Position):
        """
        Update a position.

        Args:
            position: Position object
        """
        with self.lock:
            key = (position.symbol, position.broker)
            old_position = self.positions.get(key)

            # Check for drift if position exists
            if old_position:
                self._check_drift(old_position, position)

            self.positions[key] = position
            self.position_history.append({
                'timestamp': datetime.now(),
                'symbol': position.symbol,
                'broker': position.broker,
                'quantity': position.quantity,
                'price': position.current_price,
                'market_value': position.market_value
            })

    def remove_position(self, symbol: str, broker: str):
        """
        Remove a position.

        Args:
            symbol: Position symbol
            broker: Broker name
        """
        with self.lock:
            key = (symbol, broker)
            if key in self.positions:
                del self.positions[key]

    def add_limit(self, limit: PositionLimit):
        """
        Add a position limit.

        Args:
            limit: PositionLimit object
        """
        with self.lock:
            self.limits.append(limit)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_limits()
                self._calculate_exposures()
                self._check_concentration()
                self._check_position_ages()

                # Periodic reconciliation
                if (datetime.now() - self.last_reconciliation).total_seconds() > self.reconciliation_interval:
                    self._reconcile_positions()
                    self.last_reconciliation = datetime.now()

                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in position monitor loop: {e}")

    def _check_limits(self):
        """Check position limits."""
        with self.lock:
            for position in self.positions.values():
                for limit in self.limits:
                    # Check symbol-specific limits
                    if limit.symbol and limit.symbol != position.symbol:
                        continue

                    # Check strategy-specific limits
                    if limit.strategy and limit.strategy != position.strategy:
                        continue

                    # Check quantity limit
                    if limit.max_quantity and abs(position.quantity) > limit.max_quantity:
                        self._create_alert(
                            alert_type='quantity_limit',
                            severity='critical',
                            message=f"Position {position.symbol} exceeds quantity limit: {abs(position.quantity)} > {limit.max_quantity}",
                            position=position
                        )

                    # Check notional limit
                    if limit.max_notional and position.market_value > limit.max_notional:
                        self._create_alert(
                            alert_type='notional_limit',
                            severity='critical',
                            message=f"Position {position.symbol} exceeds notional limit: {position.market_value} > {limit.max_notional}",
                            position=position
                        )

    def _calculate_exposures(self):
        """Calculate various exposure metrics."""
        with self.lock:
            if not self.positions:
                return

            gross_exposure = sum(abs(p.market_value) for p in self.positions.values())
            net_exposure = sum(
                p.market_value if p.side == 'long' else -p.market_value
                for p in self.positions.values()
            )
            long_exposure = sum(
                p.market_value for p in self.positions.values() if p.side == 'long'
            )
            short_exposure = sum(
                p.market_value for p in self.positions.values() if p.side == 'short'
            )

            self.exposure_history.append({
                'timestamp': datetime.now(),
                'gross_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'net_delta': net_exposure / gross_exposure if gross_exposure > 0 else 0
            })

    def _check_concentration(self):
        """Check position concentration."""
        with self.lock:
            if not self.positions:
                return

            total_exposure = sum(abs(p.market_value) for p in self.positions.values())
            if total_exposure == 0:
                return

            # By symbol
            symbol_concentration = defaultdict(float)
            for position in self.positions.values():
                symbol_concentration[position.symbol] += abs(position.market_value)

            max_symbol_concentration = max(symbol_concentration.values()) / total_exposure if symbol_concentration else 0

            # By strategy
            strategy_concentration = defaultdict(float)
            for position in self.positions.values():
                if position.strategy:
                    strategy_concentration[position.strategy] += abs(position.market_value)

            max_strategy_concentration = max(strategy_concentration.values()) / total_exposure if strategy_concentration else 0

            self.concentration_history.append({
                'timestamp': datetime.now(),
                'max_symbol_concentration': max_symbol_concentration,
                'max_strategy_concentration': max_strategy_concentration,
                'symbol_concentrations': dict(symbol_concentration),
                'strategy_concentrations': dict(strategy_concentration)
            })

            # Check concentration limits
            for limit in self.limits:
                if limit.max_concentration:
                    if limit.symbol:
                        conc = symbol_concentration.get(limit.symbol, 0) / total_exposure
                        if conc > limit.max_concentration / 100:
                            self._create_alert(
                                alert_type='concentration_limit',
                                severity='warning',
                                message=f"Symbol {limit.symbol} concentration {conc*100:.1f}% exceeds limit {limit.max_concentration}%",
                                metadata={'concentration': conc}
                            )

    def _check_position_ages(self):
        """Check position ages against limits."""
        with self.lock:
            for position in self.positions.values():
                age_hours = position.age.total_seconds() / 3600

                # Check against limits
                for limit in self.limits:
                    if limit.max_age_hours and limit.max_age_hours > 0:
                        if limit.symbol and limit.symbol != position.symbol:
                            continue
                        if limit.strategy and limit.strategy != position.strategy:
                            continue

                        if age_hours > limit.max_age_hours:
                            self._create_alert(
                                alert_type='age_limit',
                                severity='warning',
                                message=f"Position {position.symbol} age {age_hours:.1f}h exceeds limit {limit.max_age_hours}h",
                                position=position,
                                metadata={'age_hours': age_hours}
                            )

    def _check_drift(self, old_position: Position, new_position: Position):
        """
        Check for position drift.

        Args:
            old_position: Previous position
            new_position: New position
        """
        quantity_drift = abs(new_position.quantity - old_position.quantity) / abs(old_position.quantity) if old_position.quantity != 0 else 0

        if quantity_drift > self.drift_threshold:
            self.drift_events.append({
                'timestamp': datetime.now(),
                'symbol': new_position.symbol,
                'broker': new_position.broker,
                'old_quantity': old_position.quantity,
                'new_quantity': new_position.quantity,
                'drift_percent': quantity_drift * 100
            })

    def _reconcile_positions(self):
        """Reconcile positions across brokers."""
        with self.lock:
            # Group positions by symbol
            symbol_positions = defaultdict(list)
            for (symbol, broker), position in self.positions.items():
                symbol_positions[symbol].append(position)

            # Check for discrepancies
            for symbol, positions in symbol_positions.items():
                if len(positions) > 1:
                    total_quantity = sum(p.quantity for p in positions)
                    quantities = [p.quantity for p in positions]
                    std_dev = np.std(quantities) if len(quantities) > 1 else 0

                    if std_dev > 0:
                        self._create_alert(
                            alert_type='reconciliation',
                            severity='warning',
                            message=f"Position discrepancy for {symbol} across brokers. Total: {total_quantity}, StdDev: {std_dev:.2f}",
                            metadata={
                                'symbol': symbol,
                                'total_quantity': total_quantity,
                                'broker_quantities': {p.broker: p.quantity for p in positions}
                            }
                        )

    def _create_alert(self, alert_type: str, severity: str, message: str,
                     position: Optional[Position] = None, metadata: Optional[Dict] = None):
        """
        Create a position alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            position: Related position
            metadata: Additional metadata
        """
        alert = PositionAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            position=position,
            metadata=metadata or {}
        )
        self.alerts.append(alert)

    def get_position(self, symbol: str, broker: str) -> Optional[Position]:
        """
        Get a specific position.

        Args:
            symbol: Position symbol
            broker: Broker name

        Returns:
            Position or None
        """
        with self.lock:
            return self.positions.get((symbol, broker))

    def get_all_positions(self) -> List[Position]:
        """
        Get all positions.

        Returns:
            List of positions
        """
        with self.lock:
            return list(self.positions.values())

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get all positions for a symbol.

        Args:
            symbol: Symbol to filter

        Returns:
            List of positions
        """
        with self.lock:
            return [p for p in self.positions.values() if p.symbol == symbol]

    def get_positions_by_strategy(self, strategy: str) -> List[Position]:
        """
        Get all positions for a strategy.

        Args:
            strategy: Strategy to filter

        Returns:
            List of positions
        """
        with self.lock:
            return [p for p in self.positions.values() if p.strategy == strategy]

    def get_exposures(self) -> Dict[str, float]:
        """
        Get current exposures.

        Returns:
            Dictionary of exposure metrics
        """
        with self.lock:
            if not self.exposure_history:
                return {}
            return self.exposure_history[-1]

    def get_concentration(self) -> Dict[str, Any]:
        """
        Get current concentration metrics.

        Returns:
            Dictionary of concentration metrics
        """
        with self.lock:
            if not self.concentration_history:
                return {}
            return self.concentration_history[-1]

    def get_alerts(self, severity: Optional[str] = None, limit: int = 100) -> List[PositionAlert]:
        """
        Get recent alerts.

        Args:
            severity: Filter by severity
            limit: Maximum number of alerts

        Returns:
            List of alerts
        """
        with self.lock:
            alerts = list(self.alerts)
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts[-limit:]

    def get_drift_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent drift events.

        Args:
            limit: Maximum number of events

        Returns:
            List of drift events
        """
        with self.lock:
            return list(self.drift_events)[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get position summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            positions = list(self.positions.values())
            if not positions:
                return {
                    'total_positions': 0,
                    'total_symbols': 0,
                    'gross_exposure': 0,
                    'net_exposure': 0,
                    'long_exposure': 0,
                    'short_exposure': 0,
                    'total_unrealized_pnl': 0,
                    'alerts_count': len(self.alerts),
                    'critical_alerts': sum(1 for a in self.alerts if a.severity == 'critical')
                }

            gross_exposure = sum(abs(p.market_value) for p in positions)
            net_exposure = sum(
                p.market_value if p.side == 'long' else -p.market_value
                for p in positions
            )
            long_exposure = sum(p.market_value for p in positions if p.side == 'long')
            short_exposure = sum(p.market_value for p in positions if p.side == 'short')
            total_unrealized = sum(p.unrealized_pnl for p in positions)

            unique_symbols = len(set(p.symbol for p in positions))

            return {
                'total_positions': len(positions),
                'total_symbols': unique_symbols,
                'gross_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'net_delta': net_exposure / gross_exposure if gross_exposure > 0 else 0,
                'total_unrealized_pnl': total_unrealized,
                'alerts_count': len(self.alerts),
                'critical_alerts': sum(1 for a in self.alerts if a.severity == 'critical'),
                'drift_events': len(self.drift_events),
                'timestamp': datetime.now().isoformat()
            }

    def get_position_dataframe(self) -> pd.DataFrame:
        """
        Get positions as DataFrame.

        Returns:
            DataFrame of positions
        """
        with self.lock:
            if not self.positions:
                return pd.DataFrame()

            data = []
            for position in self.positions.values():
                data.append({
                    'symbol': position.symbol,
                    'broker': position.broker,
                    'quantity': position.quantity,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'age_hours': position.age.total_seconds() / 3600,
                    'strategy': position.strategy,
                    'timestamp': position.timestamp
                })

            return pd.DataFrame(data)

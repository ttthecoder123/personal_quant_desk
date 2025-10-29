"""
Dynamic Threshold Management System

Manages adaptive thresholds for risk alerts:
- Static threshold definitions
- Dynamic threshold calculation based on historical data
- Statistical threshold detection (Z-score, percentile-based)
- Regime-aware thresholds
- Breach detection and monitoring
- Threshold optimization
- Backtesting threshold effectiveness
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import warnings

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - limited threshold functionality")


class ThresholdType(Enum):
    """Types of thresholds"""
    STATIC = "static"  # Fixed threshold
    DYNAMIC = "dynamic"  # Adaptive based on history
    PERCENTILE = "percentile"  # Percentile-based
    ZSCORE = "zscore"  # Z-score based
    EWMA = "ewma"  # Exponentially weighted moving average
    BOLLINGER = "bollinger"  # Bollinger band style
    REGIME_AWARE = "regime_aware"  # Different thresholds by regime


class BoundaryType(Enum):
    """Threshold boundary types"""
    UPPER = "upper"  # Upper bound (value should be below)
    LOWER = "lower"  # Lower bound (value should be above)
    BOTH = "both"  # Both upper and lower bounds
    RANGE = "range"  # Value should be within range


class BreachSeverity(Enum):
    """Severity of threshold breach"""
    WARNING = "warning"  # Close to threshold
    BREACH = "breach"  # Threshold exceeded
    CRITICAL = "critical"  # Significantly exceeded


@dataclass
class Threshold:
    """Threshold configuration"""
    threshold_id: str
    name: str
    metric_name: str
    threshold_type: ThresholdType
    boundary_type: BoundaryType
    value: float
    warning_buffer: float = 0.1  # 10% buffer for warnings
    critical_multiplier: float = 1.5  # 150% for critical
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'threshold_id': self.threshold_id,
            'name': self.name,
            'metric_name': self.metric_name,
            'threshold_type': self.threshold_type.value,
            'boundary_type': self.boundary_type.value,
            'value': self.value,
            'warning_buffer': self.warning_buffer,
            'critical_multiplier': self.critical_multiplier,
            'enabled': self.enabled,
            'metadata': self.metadata
        }


@dataclass
class DynamicThresholdConfig:
    """Configuration for dynamic threshold calculation"""
    lookback_window: int = 252  # Days to look back
    percentile: float = 95.0  # Percentile for threshold
    zscore: float = 2.0  # Z-score multiplier
    ewma_halflife: int = 25  # Halflife for EWMA
    bollinger_std: float = 2.0  # Standard deviations for Bollinger
    update_frequency: int = 1  # Days between updates
    min_samples: int = 30  # Minimum samples required


@dataclass
class BreachEvent:
    """Record of a threshold breach"""
    breach_id: str
    threshold_id: str
    threshold_name: str
    metric_name: str
    timestamp: datetime
    value: float
    threshold_value: float
    deviation: float
    deviation_pct: float
    severity: BreachSeverity
    boundary_type: BoundaryType

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'breach_id': self.breach_id,
            'threshold_id': self.threshold_id,
            'threshold_name': self.threshold_name,
            'metric_name': self.metric_name,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'threshold_value': self.threshold_value,
            'deviation': self.deviation,
            'deviation_pct': self.deviation_pct,
            'severity': self.severity.value,
            'boundary_type': self.boundary_type.value
        }


class ThresholdManager:
    """
    Dynamic threshold management and breach detection

    Features:
    - Static and dynamic threshold management
    - Multiple threshold calculation methods
    - Automatic threshold updates
    - Breach detection and tracking
    - Historical analysis
    - Threshold optimization
    """

    def __init__(
        self,
        thresholds: Optional[List[Threshold]] = None,
        dynamic_config: Optional[DynamicThresholdConfig] = None
    ):
        """
        Initialize threshold manager

        Args:
            thresholds: List of threshold configurations
            dynamic_config: Configuration for dynamic thresholds
        """
        self.thresholds = thresholds if thresholds is not None else self._get_default_thresholds()
        self.threshold_dict = {t.threshold_id: t for t in self.thresholds}

        self.dynamic_config = dynamic_config if dynamic_config is not None else DynamicThresholdConfig()

        # Historical data for dynamic thresholds
        self.metric_history: Dict[str, deque] = {}  # metric_name -> deque of (timestamp, value)

        # Breach tracking
        self.breach_history: List[BreachEvent] = []
        self.active_breaches: Dict[str, BreachEvent] = {}  # threshold_id -> BreachEvent

        # Update tracking
        self.last_update: Dict[str, datetime] = {}  # threshold_id -> last_update_time

        # Statistics
        self.breach_stats = {
            'total_breaches': 0,
            'by_severity': {'warning': 0, 'breach': 0, 'critical': 0},
            'by_threshold': {},
            'by_metric': {}
        }

    def check_threshold(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> List[BreachEvent]:
        """
        Check value against all thresholds for the metric

        Args:
            metric_name: Metric name
            value: Current value
            timestamp: Timestamp (defaults to now)

        Returns:
            List of breach events (empty if no breaches)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Store in history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = deque(
                maxlen=self.dynamic_config.lookback_window
            )
        self.metric_history[metric_name].append((timestamp, value))

        breaches = []

        # Check against all thresholds for this metric
        for threshold in self.thresholds:
            if threshold.metric_name != metric_name or not threshold.enabled:
                continue

            # Update dynamic thresholds if needed
            if threshold.threshold_type != ThresholdType.STATIC:
                self._update_dynamic_threshold(threshold)

            # Check for breach
            breach = self._check_single_threshold(threshold, value, timestamp)
            if breach:
                breaches.append(breach)
                self._record_breach(breach)

        return breaches

    def check_multiple_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, List[BreachEvent]]:
        """
        Check multiple metrics at once

        Args:
            metrics: Dictionary of metric_name -> value
            timestamp: Timestamp (defaults to now)

        Returns:
            Dictionary of metric_name -> list of breaches
        """
        results = {}

        for metric_name, value in metrics.items():
            breaches = self.check_threshold(metric_name, value, timestamp)
            if breaches:
                results[metric_name] = breaches

        return results

    def add_threshold(self, threshold: Threshold):
        """
        Add a new threshold

        Args:
            threshold: Threshold to add
        """
        self.thresholds.append(threshold)
        self.threshold_dict[threshold.threshold_id] = threshold

    def update_threshold(
        self,
        threshold_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update threshold configuration

        Args:
            threshold_id: Threshold ID
            updates: Dictionary of fields to update

        Returns:
            Success status
        """
        if threshold_id not in self.threshold_dict:
            return False

        threshold = self.threshold_dict[threshold_id]

        for field, value in updates.items():
            if hasattr(threshold, field):
                setattr(threshold, field, value)

        return True

    def get_threshold(self, threshold_id: str) -> Optional[Threshold]:
        """
        Get threshold by ID

        Args:
            threshold_id: Threshold ID

        Returns:
            Threshold or None
        """
        return self.threshold_dict.get(threshold_id)

    def get_active_breaches(
        self,
        metric_name: Optional[str] = None
    ) -> List[BreachEvent]:
        """
        Get all active breaches

        Args:
            metric_name: Optional filter by metric

        Returns:
            List of active breaches
        """
        breaches = list(self.active_breaches.values())

        if metric_name:
            breaches = [b for b in breaches if b.metric_name == metric_name]

        return breaches

    def get_breach_history(
        self,
        metric_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_severity: Optional[BreachSeverity] = None
    ) -> List[BreachEvent]:
        """
        Get historical breaches with filtering

        Args:
            metric_name: Optional metric filter
            start_date: Optional start date
            end_date: Optional end date
            min_severity: Optional minimum severity

        Returns:
            List of historical breaches
        """
        breaches = self.breach_history

        if metric_name:
            breaches = [b for b in breaches if b.metric_name == metric_name]

        if start_date:
            breaches = [b for b in breaches if b.timestamp >= start_date]

        if end_date:
            breaches = [b for b in breaches if b.timestamp <= end_date]

        if min_severity:
            severity_order = {
                BreachSeverity.WARNING: 0,
                BreachSeverity.BREACH: 1,
                BreachSeverity.CRITICAL: 2
            }
            min_level = severity_order[min_severity]
            breaches = [
                b for b in breaches
                if severity_order[b.severity] >= min_level
            ]

        return breaches

    def calculate_dynamic_threshold(
        self,
        metric_name: str,
        method: ThresholdType
    ) -> Optional[float]:
        """
        Calculate dynamic threshold for a metric

        Args:
            metric_name: Metric name
            method: Calculation method

        Returns:
            Calculated threshold value or None
        """
        if metric_name not in self.metric_history:
            return None

        history = self.metric_history[metric_name]
        if len(history) < self.dynamic_config.min_samples:
            return None

        # Extract values
        values = np.array([v for _, v in history])

        if method == ThresholdType.PERCENTILE:
            return np.percentile(values, self.dynamic_config.percentile)

        elif method == ThresholdType.ZSCORE:
            mean = np.mean(values)
            std = np.std(values)
            return mean + self.dynamic_config.zscore * std

        elif method == ThresholdType.EWMA:
            # Calculate EWMA
            series = pd.Series(values)
            ewma = series.ewm(halflife=self.dynamic_config.ewma_halflife).mean()
            return ewma.iloc[-1]

        elif method == ThresholdType.BOLLINGER:
            # Upper Bollinger band
            series = pd.Series(values)
            rolling_mean = series.rolling(window=20).mean()
            rolling_std = series.rolling(window=20).std()
            upper_band = rolling_mean + (self.dynamic_config.bollinger_std * rolling_std)
            return upper_band.iloc[-1]

        return None

    def optimize_threshold(
        self,
        metric_name: str,
        target_breach_rate: float = 0.05
    ) -> Optional[float]:
        """
        Optimize threshold to achieve target breach rate

        Args:
            metric_name: Metric name
            target_breach_rate: Desired breach rate (e.g., 0.05 = 5%)

        Returns:
            Optimized threshold value or None
        """
        if metric_name not in self.metric_history:
            return None

        history = self.metric_history[metric_name]
        if len(history) < self.dynamic_config.min_samples:
            return None

        values = np.array([v for _, v in history])

        # Calculate percentile corresponding to target breach rate
        percentile = (1 - target_breach_rate) * 100

        return np.percentile(values, percentile)

    def backtest_threshold(
        self,
        threshold_id: str,
        test_threshold_value: float
    ) -> Dict[str, Any]:
        """
        Backtest a threshold value against historical data

        Args:
            threshold_id: Threshold ID
            test_threshold_value: Threshold value to test

        Returns:
            Dictionary with backtest results
        """
        threshold = self.threshold_dict.get(threshold_id)
        if not threshold:
            return {}

        metric_name = threshold.metric_name
        if metric_name not in self.metric_history:
            return {}

        history = self.metric_history[metric_name]
        values = [v for _, v in history]

        # Count breaches
        breaches = []
        for timestamp, value in history:
            if threshold.boundary_type == BoundaryType.UPPER:
                if value > test_threshold_value:
                    breaches.append((timestamp, value))
            elif threshold.boundary_type == BoundaryType.LOWER:
                if value < test_threshold_value:
                    breaches.append((timestamp, value))

        breach_rate = len(breaches) / len(values) if values else 0

        # Calculate breach statistics
        if breaches:
            breach_values = [v for _, v in breaches]
            avg_deviation = np.mean([abs(v - test_threshold_value) for v in breach_values])
            max_deviation = max([abs(v - test_threshold_value) for v in breach_values])
        else:
            avg_deviation = 0
            max_deviation = 0

        return {
            'threshold_value': test_threshold_value,
            'total_samples': len(values),
            'total_breaches': len(breaches),
            'breach_rate': breach_rate,
            'avg_deviation': avg_deviation,
            'max_deviation': max_deviation
        }

    def _check_single_threshold(
        self,
        threshold: Threshold,
        value: float,
        timestamp: datetime
    ) -> Optional[BreachEvent]:
        """Check value against a single threshold"""
        threshold_value = threshold.value
        boundary_type = threshold.boundary_type

        # Determine if breached
        breached = False
        deviation = 0.0

        if boundary_type == BoundaryType.UPPER:
            if value > threshold_value:
                breached = True
                deviation = value - threshold_value

        elif boundary_type == BoundaryType.LOWER:
            if value < threshold_value:
                breached = True
                deviation = threshold_value - value

        if not breached:
            # Clear any active breach for this threshold
            if threshold.threshold_id in self.active_breaches:
                del self.active_breaches[threshold.threshold_id]
            return None

        # Determine severity
        deviation_pct = (abs(deviation) / threshold_value * 100) if threshold_value != 0 else 0

        if deviation_pct < threshold.warning_buffer * 100:
            severity = BreachSeverity.WARNING
        elif deviation_pct < (threshold.critical_multiplier - 1) * 100:
            severity = BreachSeverity.BREACH
        else:
            severity = BreachSeverity.CRITICAL

        # Create breach event
        breach_id = self._generate_breach_id(threshold.threshold_id, timestamp)

        breach = BreachEvent(
            breach_id=breach_id,
            threshold_id=threshold.threshold_id,
            threshold_name=threshold.name,
            metric_name=threshold.metric_name,
            timestamp=timestamp,
            value=value,
            threshold_value=threshold_value,
            deviation=deviation,
            deviation_pct=deviation_pct,
            severity=severity,
            boundary_type=boundary_type
        )

        return breach

    def _record_breach(self, breach: BreachEvent):
        """Record a breach event"""
        self.breach_history.append(breach)
        self.active_breaches[breach.threshold_id] = breach

        # Update statistics
        self.breach_stats['total_breaches'] += 1
        self.breach_stats['by_severity'][breach.severity.value] += 1

        if breach.threshold_name not in self.breach_stats['by_threshold']:
            self.breach_stats['by_threshold'][breach.threshold_name] = 0
        self.breach_stats['by_threshold'][breach.threshold_name] += 1

        if breach.metric_name not in self.breach_stats['by_metric']:
            self.breach_stats['by_metric'][breach.metric_name] = 0
        self.breach_stats['by_metric'][breach.metric_name] += 1

    def _update_dynamic_threshold(self, threshold: Threshold):
        """Update dynamic threshold based on recent data"""
        # Check if update is needed
        last_update = self.last_update.get(threshold.threshold_id)
        if last_update:
            days_since_update = (datetime.now() - last_update).days
            if days_since_update < self.dynamic_config.update_frequency:
                return

        # Calculate new threshold
        new_value = self.calculate_dynamic_threshold(
            threshold.metric_name,
            threshold.threshold_type
        )

        if new_value is not None:
            threshold.value = new_value
            self.last_update[threshold.threshold_id] = datetime.now()

    def _generate_breach_id(self, threshold_id: str, timestamp: datetime) -> str:
        """Generate unique breach ID"""
        time_str = timestamp.strftime('%Y%m%d%H%M%S%f')
        return f"BREACH_{threshold_id}_{time_str}"

    def _get_default_thresholds(self) -> List[Threshold]:
        """Get default threshold configurations"""
        return [
            Threshold(
                threshold_id="THR_VAR_95",
                name="VaR 95% Limit",
                metric_name="var_95",
                threshold_type=ThresholdType.STATIC,
                boundary_type=BoundaryType.UPPER,
                value=0.02,
                warning_buffer=0.8,
                critical_multiplier=1.5
            ),
            Threshold(
                threshold_id="THR_DRAWDOWN",
                name="Maximum Drawdown",
                metric_name="drawdown",
                threshold_type=ThresholdType.STATIC,
                boundary_type=BoundaryType.UPPER,
                value=0.20,
                warning_buffer=0.75,
                critical_multiplier=1.25
            ),
            Threshold(
                threshold_id="THR_VOLATILITY",
                name="Portfolio Volatility",
                metric_name="volatility",
                threshold_type=ThresholdType.DYNAMIC,
                boundary_type=BoundaryType.UPPER,
                value=0.30,
                warning_buffer=0.9,
                critical_multiplier=1.5
            ),
            Threshold(
                threshold_id="THR_LEVERAGE",
                name="Leverage Limit",
                metric_name="leverage",
                threshold_type=ThresholdType.STATIC,
                boundary_type=BoundaryType.UPPER,
                value=2.0,
                warning_buffer=0.9,
                critical_multiplier=1.3
            ),
            Threshold(
                threshold_id="THR_SHARPE",
                name="Minimum Sharpe Ratio",
                metric_name="sharpe_ratio",
                threshold_type=ThresholdType.DYNAMIC,
                boundary_type=BoundaryType.LOWER,
                value=0.5,
                warning_buffer=0.8,
                critical_multiplier=0.5
            )
        ]

    def get_threshold_summary(self) -> str:
        """
        Generate text summary of thresholds and breaches

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("THRESHOLD MANAGER SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append(f"Total Thresholds: {len(self.thresholds)}")
        lines.append(f"Active Thresholds: {sum(1 for t in self.thresholds if t.enabled)}")
        lines.append(f"Active Breaches: {len(self.active_breaches)}")
        lines.append("")

        lines.append("BREACH STATISTICS:")
        lines.append(f"  Total Breaches: {self.breach_stats['total_breaches']}")
        lines.append(f"  Warnings: {self.breach_stats['by_severity']['warning']}")
        lines.append(f"  Breaches: {self.breach_stats['by_severity']['breach']}")
        lines.append(f"  Critical: {self.breach_stats['by_severity']['critical']}")
        lines.append("")

        if self.active_breaches:
            lines.append("ACTIVE BREACHES:")
            for breach in self.active_breaches.values():
                lines.append(
                    f"  [{breach.severity.value.upper():<8}] {breach.threshold_name}: "
                    f"{breach.value:.4f} (threshold: {breach.threshold_value:.4f})"
                )
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def export_thresholds(self, filepath: str):
        """
        Export threshold configurations to JSON

        Args:
            filepath: Output file path
        """
        import json

        threshold_dicts = [t.to_dict() for t in self.thresholds]

        with open(filepath, 'w') as f:
            json.dump(threshold_dicts, f, indent=2)

    def import_thresholds(self, filepath: str):
        """
        Import threshold configurations from JSON

        Args:
            filepath: Input file path
        """
        import json

        with open(filepath, 'r') as f:
            threshold_dicts = json.load(f)

        for td in threshold_dicts:
            threshold = Threshold(
                threshold_id=td['threshold_id'],
                name=td['name'],
                metric_name=td['metric_name'],
                threshold_type=ThresholdType(td['threshold_type']),
                boundary_type=BoundaryType(td['boundary_type']),
                value=td['value'],
                warning_buffer=td.get('warning_buffer', 0.1),
                critical_multiplier=td.get('critical_multiplier', 1.5),
                enabled=td.get('enabled', True),
                metadata=td.get('metadata', {})
            )
            self.add_threshold(threshold)

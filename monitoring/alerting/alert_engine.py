"""
Alert Engine - Rule-based and ML-based alert generation.

Features:
- Rule-based alert generation
- ML-based anomaly detection
- Threshold monitoring
- Composite alert conditions
- Alert correlation
- Root cause analysis
- Alert prioritization
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from scipy import stats
import hashlib
import json


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertCategory(Enum):
    """Alert categories."""
    SYSTEM = "system"
    TRADING = "trading"
    DATA = "data"
    PERFORMANCE = "performance"
    SECURITY = "security"
    NETWORK = "network"
    RESOURCE = "resource"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: datetime
    source: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    root_cause: Optional[str] = None
    related_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'tags': self.tags,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'root_cause': self.root_cause,
            'related_alerts': self.related_alerts
        }


@dataclass
class MetricDataPoint:
    """Metric data point for anomaly detection."""
    timestamp: datetime
    value: float
    metric_name: str
    tags: Dict[str, str] = field(default_factory=dict)


class AnomalyDetector:
    """ML-based anomaly detection for metrics."""

    def __init__(self, window_size: int = 100, sensitivity: float = 3.0):
        """
        Initialize anomaly detector.

        Args:
            window_size: Number of points to consider for baseline
            sensitivity: Standard deviations for anomaly threshold
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.lock = threading.Lock()

    def add_data_point(self, metric_name: str, value: float):
        """Add data point for metric."""
        with self.lock:
            self.metric_history[metric_name].append(value)
            self._update_baseline_stats(metric_name)

    def _update_baseline_stats(self, metric_name: str):
        """Update baseline statistics for metric."""
        history = list(self.metric_history[metric_name])
        if len(history) < 10:  # Need minimum data
            return

        self.baseline_stats[metric_name] = {
            'mean': np.mean(history),
            'std': np.std(history),
            'median': np.median(history),
            'q1': np.percentile(history, 25),
            'q3': np.percentile(history, 75)
        }

    def is_anomaly(self, metric_name: str, value: float) -> tuple[bool, Optional[str]]:
        """
        Check if value is an anomaly.

        Returns:
            Tuple of (is_anomaly, reason)
        """
        with self.lock:
            if metric_name not in self.baseline_stats:
                return False, None

            stats = self.baseline_stats[metric_name]
            mean = stats['mean']
            std = stats['std']

            # Z-score method
            if std > 0:
                z_score = abs((value - mean) / std)
                if z_score > self.sensitivity:
                    return True, f"Z-score: {z_score:.2f} (threshold: {self.sensitivity})"

            # IQR method for additional validation
            iqr = stats['q3'] - stats['q1']
            if iqr > 0:
                lower_bound = stats['q1'] - 1.5 * iqr
                upper_bound = stats['q3'] + 1.5 * iqr
                if value < lower_bound or value > upper_bound:
                    return True, f"Outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"

            return False, None

    def get_baseline_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for metric."""
        with self.lock:
            return self.baseline_stats.get(metric_name)


class RootCauseAnalyzer:
    """Analyzes root causes of alerts."""

    def __init__(self):
        """Initialize root cause analyzer."""
        self.dependency_graph: Dict[str, List[str]] = {}
        self.recent_events: deque = deque(maxlen=1000)
        self.lock = threading.Lock()

    def add_dependency(self, component: str, depends_on: List[str]):
        """
        Define component dependencies.

        Args:
            component: Component name
            depends_on: List of components it depends on
        """
        with self.lock:
            self.dependency_graph[component] = depends_on

    def record_event(self, component: str, event_type: str, timestamp: datetime):
        """Record system event for analysis."""
        with self.lock:
            self.recent_events.append({
                'component': component,
                'event_type': event_type,
                'timestamp': timestamp
            })

    def analyze(self, alert: Alert) -> Optional[str]:
        """
        Analyze root cause of alert.

        Returns:
            Root cause description or None
        """
        with self.lock:
            # Check for recent events in related components
            source = alert.source
            related_components = self.dependency_graph.get(source, [])

            # Look for events in last 5 minutes
            recent_threshold = datetime.now() - timedelta(minutes=5)
            relevant_events = [
                e for e in self.recent_events
                if e['timestamp'] > recent_threshold and
                   e['component'] in related_components
            ]

            if relevant_events:
                # Group by component
                events_by_component = defaultdict(list)
                for event in relevant_events:
                    events_by_component[event['component']].append(event)

                # Build root cause message
                causes = []
                for component, events in events_by_component.items():
                    event_types = [e['event_type'] for e in events]
                    causes.append(f"{component}: {', '.join(event_types)}")

                return f"Possible root causes: {'; '.join(causes)}"

            return None


class AlertCorrelator:
    """Correlates related alerts."""

    def __init__(self, correlation_window: int = 60):
        """
        Initialize alert correlator.

        Args:
            correlation_window: Seconds to correlate alerts within
        """
        self.correlation_window = correlation_window
        self.recent_alerts: deque = deque(maxlen=1000)
        self.correlation_groups: Dict[str, List[str]] = {}
        self.lock = threading.Lock()

    def add_alert(self, alert: Alert) -> Optional[str]:
        """
        Add alert and check for correlations.

        Returns:
            Correlation ID if correlated, None otherwise
        """
        with self.lock:
            # Find similar recent alerts
            correlation_threshold = datetime.now() - timedelta(seconds=self.correlation_window)
            similar_alerts = [
                a for a in self.recent_alerts
                if a.timestamp > correlation_threshold and
                   self._are_similar(alert, a)
            ]

            if similar_alerts:
                # Use existing correlation ID
                correlation_id = similar_alerts[0].correlation_id
                if correlation_id:
                    alert.correlation_id = correlation_id
                    self.correlation_groups[correlation_id].append(alert.id)
                    alert.related_alerts = self.correlation_groups[correlation_id].copy()
                else:
                    # Create new correlation group
                    correlation_id = self._generate_correlation_id(alert)
                    alert.correlation_id = correlation_id
                    self.correlation_groups[correlation_id] = [similar_alerts[0].id, alert.id]
                    similar_alerts[0].correlation_id = correlation_id
                    alert.related_alerts = [similar_alerts[0].id]

            self.recent_alerts.append(alert)
            return alert.correlation_id

    def _are_similar(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are similar."""
        # Same category and severity
        if alert1.category != alert2.category:
            return False

        if alert1.severity != alert2.severity:
            return False

        # Same metric or source
        if alert1.metric_name and alert2.metric_name:
            if alert1.metric_name == alert2.metric_name:
                return True

        if alert1.source == alert2.source:
            return True

        return False

    def _generate_correlation_id(self, alert: Alert) -> str:
        """Generate unique correlation ID."""
        data = f"{alert.category.value}_{alert.severity.value}_{alert.metric_name or alert.source}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


class AlertEngine:
    """
    Main alert engine for rule-based and ML-based alert generation.

    Features:
    - Rule-based threshold monitoring
    - ML-based anomaly detection
    - Composite alert conditions
    - Alert correlation and prioritization
    - Root cause analysis
    """

    def __init__(self):
        """Initialize alert engine."""
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.lock = threading.Lock()

        # Sub-components
        self.anomaly_detector = AnomalyDetector()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.alert_correlator = AlertCorrelator()

        # Rules and conditions
        self.threshold_rules: Dict[str, Dict[str, Any]] = {}
        self.composite_conditions: Dict[str, Callable] = {}

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Metrics
        self.metrics_buffer: Dict[str, List[MetricDataPoint]] = defaultdict(list)

    def register_threshold_rule(
        self,
        metric_name: str,
        threshold: float,
        operator: str,
        severity: AlertSeverity,
        category: AlertCategory,
        message_template: str
    ):
        """
        Register threshold-based alert rule.

        Args:
            metric_name: Metric to monitor
            threshold: Threshold value
            operator: Comparison operator (>, <, >=, <=, ==)
            severity: Alert severity
            category: Alert category
            message_template: Alert message template
        """
        self.threshold_rules[metric_name] = {
            'threshold': threshold,
            'operator': operator,
            'severity': severity,
            'category': category,
            'message_template': message_template
        }

    def register_composite_condition(
        self,
        condition_name: str,
        condition_func: Callable[[Dict[str, float]], bool],
        severity: AlertSeverity,
        category: AlertCategory,
        message_template: str
    ):
        """
        Register composite alert condition.

        Args:
            condition_name: Unique condition name
            condition_func: Function that evaluates condition
            severity: Alert severity
            category: Alert category
            message_template: Alert message template
        """
        self.composite_conditions[condition_name] = {
            'func': condition_func,
            'severity': severity,
            'category': category,
            'message_template': message_template
        }

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Add callback to be called when alert is generated.

        Args:
            callback: Function to call with alert
        """
        self.alert_callbacks.append(callback)

    def ingest_metric(
        self,
        metric_name: str,
        value: float,
        source: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Ingest metric value and check for alerts.

        Args:
            metric_name: Metric name
            value: Metric value
            source: Source component
            tags: Optional tags
        """
        timestamp = datetime.now()
        tags = tags or {}

        # Store metric
        data_point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            metric_name=metric_name,
            tags=tags
        )
        self.metrics_buffer[metric_name].append(data_point)

        # Update anomaly detector
        self.anomaly_detector.add_data_point(metric_name, value)

        # Check threshold rules
        self._check_threshold_rules(metric_name, value, source, tags)

        # Check for anomalies
        self._check_anomalies(metric_name, value, source, tags)

        # Check composite conditions
        self._check_composite_conditions(source)

    def _check_threshold_rules(
        self,
        metric_name: str,
        value: float,
        source: str,
        tags: Dict[str, str]
    ):
        """Check threshold-based rules."""
        if metric_name not in self.threshold_rules:
            return

        rule = self.threshold_rules[metric_name]
        threshold = rule['threshold']
        operator = rule['operator']

        # Evaluate threshold
        triggered = False
        if operator == '>':
            triggered = value > threshold
        elif operator == '<':
            triggered = value < threshold
        elif operator == '>=':
            triggered = value >= threshold
        elif operator == '<=':
            triggered = value <= threshold
        elif operator == '==':
            triggered = abs(value - threshold) < 1e-6

        if triggered:
            # Generate alert
            alert = Alert(
                id=self._generate_alert_id(),
                title=f"Threshold exceeded: {metric_name}",
                message=rule['message_template'].format(
                    metric_name=metric_name,
                    value=value,
                    threshold=threshold
                ),
                severity=rule['severity'],
                category=rule['category'],
                timestamp=datetime.now(),
                source=source,
                metric_name=metric_name,
                metric_value=value,
                threshold=threshold,
                tags=tags
            )

            self._emit_alert(alert)

    def _check_anomalies(
        self,
        metric_name: str,
        value: float,
        source: str,
        tags: Dict[str, str]
    ):
        """Check for anomalies using ML detector."""
        is_anomaly, reason = self.anomaly_detector.is_anomaly(metric_name, value)

        if is_anomaly:
            baseline_stats = self.anomaly_detector.get_baseline_stats(metric_name)

            alert = Alert(
                id=self._generate_alert_id(),
                title=f"Anomaly detected: {metric_name}",
                message=f"Anomalous value detected for {metric_name}: {value:.2f}. {reason}",
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.PERFORMANCE,
                timestamp=datetime.now(),
                source=source,
                metric_name=metric_name,
                metric_value=value,
                threshold=baseline_stats.get('mean') if baseline_stats else None,
                tags=tags,
                metadata={
                    'baseline_stats': baseline_stats,
                    'anomaly_reason': reason
                }
            )

            self._emit_alert(alert)

    def _check_composite_conditions(self, source: str):
        """Check composite alert conditions."""
        # Get latest metric values
        latest_metrics = {}
        for metric_name, data_points in self.metrics_buffer.items():
            if data_points:
                latest_metrics[metric_name] = data_points[-1].value

        # Evaluate each condition
        for condition_name, condition in self.composite_conditions.items():
            try:
                if condition['func'](latest_metrics):
                    alert = Alert(
                        id=self._generate_alert_id(),
                        title=f"Composite condition triggered: {condition_name}",
                        message=condition['message_template'].format(**latest_metrics),
                        severity=condition['severity'],
                        category=condition['category'],
                        timestamp=datetime.now(),
                        source=source,
                        metadata={'condition_name': condition_name}
                    )

                    self._emit_alert(alert)
            except Exception as e:
                print(f"Error evaluating composite condition {condition_name}: {e}")

    def _emit_alert(self, alert: Alert):
        """Emit alert with correlation and root cause analysis."""
        with self.lock:
            # Correlate with recent alerts
            correlation_id = self.alert_correlator.add_alert(alert)

            # Perform root cause analysis
            root_cause = self.root_cause_analyzer.analyze(alert)
            if root_cause:
                alert.root_cause = root_cause

            # Store alert
            self.alerts.append(alert)
            self.alert_history.append(alert)
            self.active_alerts[alert.id] = alert

            # Call callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{timestamp}_{time.time()}".encode()).hexdigest()[:16]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge and clear an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if acknowledged, False if not found
        """
        with self.lock:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                return True
            return False

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            severity: Filter by severity
            category: Filter by category

        Returns:
            List of active alerts
        """
        with self.lock:
            alerts = list(self.active_alerts.values())

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            if category:
                alerts = [a for a in alerts if a.category == category]

            # Sort by severity and timestamp
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.HIGH: 1,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.LOW: 3,
                AlertSeverity.INFO: 4
            }

            alerts.sort(key=lambda a: (severity_order[a.severity], a.timestamp), reverse=True)
            return alerts

    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        with self.lock:
            alerts = list(self.alert_history)

            if start_time:
                alerts = [a for a in alerts if a.timestamp >= start_time]

            if end_time:
                alerts = [a for a in alerts if a.timestamp <= end_time]

            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            return alerts[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1

            # Count by category
            category_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                category_counts[alert.category.value] += 1

            # Historical stats
            total_alerts_24h = len([
                a for a in self.alert_history
                if a.timestamp > datetime.now() - timedelta(hours=24)
            ])

            return {
                'active_alerts': len(self.active_alerts),
                'total_alerts_24h': total_alerts_24h,
                'severity_distribution': dict(severity_counts),
                'category_distribution': dict(category_counts),
                'correlation_groups': len(self.alert_correlator.correlation_groups)
            }

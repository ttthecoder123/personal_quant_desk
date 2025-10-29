"""
Alert Management System

Comprehensive alert management for risk monitoring:
- Multi-level priority system (Critical/High/Medium/Low)
- Alert deduplication
- Escalation rules
- Acknowledgment workflow
- Alert routing and distribution
- Historical alert tracking
- Alert aggregation and batching
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import hashlib
import json
from pathlib import Path


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Urgent attention needed
    MEDIUM = "medium"  # Should be addressed soon
    LOW = "low"  # Informational
    INFO = "info"  # General information


class AlertCategory(Enum):
    """Alert categories"""
    RISK_LIMIT = "risk_limit"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    VAR_BREACH = "var_breach"
    POSITION_LIMIT = "position_limit"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    timestamp: datetime
    priority: AlertPriority
    category: AlertCategory
    title: str
    message: str
    source: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_count: int = 0
    last_escalated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalation_count': self.escalation_count,
            'last_escalated_at': self.last_escalated_at.isoformat() if self.last_escalated_at else None,
            'metadata': self.metadata
        }


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    rule_name: str
    category: AlertCategory
    priority: AlertPriority
    condition: str  # Python expression
    message_template: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    escalation_delay_seconds: int = 1800  # 30 minutes
    auto_resolve: bool = False
    notification_channels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'category': self.category.value,
            'priority': self.priority.value,
            'condition': self.condition,
            'message_template': self.message_template,
            'enabled': self.enabled,
            'cooldown_seconds': self.cooldown_seconds,
            'escalation_delay_seconds': self.escalation_delay_seconds,
            'auto_resolve': self.auto_resolve,
            'notification_channels': self.notification_channels
        }


@dataclass
class EscalationRule:
    """Escalation rule for alerts"""
    rule_id: str
    from_priority: AlertPriority
    to_priority: AlertPriority
    condition: str  # e.g., "unacknowledged for 30 minutes"
    delay_seconds: int
    notification_channels: List[str] = field(default_factory=list)


class AlertManager:
    """
    Comprehensive alert management system

    Features:
    - Multi-priority alert handling
    - Deduplication to prevent alert spam
    - Automatic escalation for unhandled alerts
    - Acknowledgment and resolution workflow
    - Alert aggregation and batching
    - Historical tracking
    - Integration with notification channels
    """

    def __init__(
        self,
        alert_rules: Optional[List[AlertRule]] = None,
        escalation_rules: Optional[List[EscalationRule]] = None,
        storage_dir: str = "./alerts"
    ):
        """
        Initialize alert manager

        Args:
            alert_rules: List of alert rules
            escalation_rules: List of escalation rules
            storage_dir: Directory for alert storage
        """
        self.alert_rules = alert_rules if alert_rules is not None else self._get_default_rules()
        self.escalation_rules = escalation_rules if escalation_rules is not None else []

        # Storage
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}  # alert_id -> Alert
        self.alert_history: deque = deque(maxlen=10000)

        # Deduplication tracking
        self.alert_signatures: Dict[str, datetime] = {}  # signature -> last_seen
        self.alert_counts: Dict[str, int] = defaultdict(int)  # signature -> count

        # Notification callbacks
        self.notification_callbacks: Dict[str, Callable] = {}

        # Statistics
        self.alert_stats = {
            'total_generated': 0,
            'total_deduplicated': 0,
            'total_acknowledged': 0,
            'total_resolved': 0,
            'total_escalated': 0,
            'by_priority': defaultdict(int),
            'by_category': defaultdict(int)
        }

    def create_alert(
        self,
        priority: AlertPriority,
        category: AlertCategory,
        title: str,
        message: str,
        source: str,
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """
        Create a new alert with deduplication

        Args:
            priority: Alert priority
            category: Alert category
            title: Alert title
            message: Alert message
            source: Alert source/component
            metric_name: Optional metric name
            current_value: Optional current metric value
            threshold_value: Optional threshold value
            metadata: Optional additional metadata

        Returns:
            Created Alert or None if deduplicated
        """
        timestamp = datetime.now()

        # Generate alert signature for deduplication
        signature = self._generate_signature(
            category, title, metric_name, source
        )

        # Check for deduplication
        if self._should_deduplicate(signature, priority):
            self.alert_stats['total_deduplicated'] += 1
            self.alert_counts[signature] += 1
            return None

        # Create alert
        alert_id = self._generate_alert_id(timestamp)

        alert = Alert(
            alert_id=alert_id,
            timestamp=timestamp,
            priority=priority,
            category=category,
            title=title,
            message=message,
            source=source,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            metadata=metadata or {}
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Update tracking
        self.alert_signatures[signature] = timestamp
        self.alert_counts[signature] += 1

        # Update statistics
        self.alert_stats['total_generated'] += 1
        self.alert_stats['by_priority'][priority.value] += 1
        self.alert_stats['by_category'][category.value] += 1

        # Trigger notifications
        self._trigger_notifications(alert)

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User acknowledging

        Returns:
            Success status
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()

        self.alert_stats['total_acknowledged'] += 1

        return True

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert

        Args:
            alert_id: Alert ID to resolve
            resolved_by: Optional user resolving

        Returns:
            Success status
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()

        # Remove from active alerts
        del self.active_alerts[alert_id]

        self.alert_stats['total_resolved'] += 1

        return True

    def escalate_alert(
        self,
        alert_id: str,
        reason: str = "Auto-escalated due to timeout"
    ) -> bool:
        """
        Escalate an alert to higher priority

        Args:
            alert_id: Alert ID to escalate
            reason: Escalation reason

        Returns:
            Success status
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]

        # Increase priority
        current_priority_order = [
            AlertPriority.INFO,
            AlertPriority.LOW,
            AlertPriority.MEDIUM,
            AlertPriority.HIGH,
            AlertPriority.CRITICAL
        ]

        current_idx = current_priority_order.index(alert.priority)
        if current_idx < len(current_priority_order) - 1:
            alert.priority = current_priority_order[current_idx + 1]

        alert.status = AlertStatus.ESCALATED
        alert.escalation_count += 1
        alert.last_escalated_at = datetime.now()

        if 'escalation_history' not in alert.metadata:
            alert.metadata['escalation_history'] = []

        alert.metadata['escalation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'new_priority': alert.priority.value
        })

        self.alert_stats['total_escalated'] += 1

        # Trigger escalation notifications
        self._trigger_notifications(alert, escalation=True)

        return True

    def check_escalations(self):
        """
        Check all active alerts for escalation conditions
        """
        current_time = datetime.now()

        for alert_id, alert in list(self.active_alerts.items()):
            # Skip if already acknowledged or escalated recently
            if alert.status == AlertStatus.ACKNOWLEDGED:
                continue

            if alert.last_escalated_at:
                time_since_escalation = (current_time - alert.last_escalated_at).total_seconds()
                if time_since_escalation < 1800:  # 30 minutes
                    continue

            # Check escalation conditions
            time_active = (current_time - alert.timestamp).total_seconds()

            # Critical alerts: escalate if unacknowledged for 5 minutes
            if alert.priority == AlertPriority.CRITICAL and time_active > 300:
                self.escalate_alert(alert_id, "Critical alert unacknowledged for 5+ minutes")

            # High alerts: escalate if unacknowledged for 15 minutes
            elif alert.priority == AlertPriority.HIGH and time_active > 900:
                self.escalate_alert(alert_id, "High priority alert unacknowledged for 15+ minutes")

            # Medium alerts: escalate if unacknowledged for 30 minutes
            elif alert.priority == AlertPriority.MEDIUM and time_active > 1800:
                self.escalate_alert(alert_id, "Medium priority alert unacknowledged for 30+ minutes")

    def get_active_alerts(
        self,
        priority: Optional[AlertPriority] = None,
        category: Optional[AlertCategory] = None,
        status: Optional[AlertStatus] = None
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering

        Args:
            priority: Filter by priority
            category: Filter by category
            status: Filter by status

        Returns:
            List of matching alerts
        """
        alerts = list(self.active_alerts.values())

        if priority:
            alerts = [a for a in alerts if a.priority == priority]

        if category:
            alerts = [a for a in alerts if a.category == category]

        if status:
            alerts = [a for a in alerts if a.status == status]

        # Sort by priority and timestamp
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3,
            AlertPriority.INFO: 4
        }

        alerts.sort(key=lambda a: (priority_order[a.priority], a.timestamp))

        return alerts

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alert status

        Returns:
            Dictionary with alert summary
        """
        active_alerts = list(self.active_alerts.values())

        by_priority = defaultdict(int)
        by_category = defaultdict(int)
        by_status = defaultdict(int)

        for alert in active_alerts:
            by_priority[alert.priority.value] += 1
            by_category[alert.category.value] += 1
            by_status[alert.status.value] += 1

        return {
            'total_active': len(active_alerts),
            'by_priority': dict(by_priority),
            'by_category': dict(by_category),
            'by_status': dict(by_status),
            'unacknowledged': sum(
                1 for a in active_alerts
                if a.status == AlertStatus.ACTIVE
            ),
            'oldest_active': min(
                (a.timestamp for a in active_alerts),
                default=None
            )
        }

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive alert statistics

        Returns:
            Dictionary with statistics
        """
        return {
            **self.alert_stats,
            'active_count': len(self.active_alerts),
            'history_count': len(self.alert_history)
        }

    def register_notification_callback(
        self,
        channel_name: str,
        callback: Callable[[Alert], None]
    ):
        """
        Register a notification callback for a channel

        Args:
            channel_name: Channel identifier
            callback: Callback function taking Alert as parameter
        """
        self.notification_callbacks[channel_name] = callback

    def batch_alerts(
        self,
        alerts: List[Alert],
        batch_by: str = "category"
    ) -> Dict[str, List[Alert]]:
        """
        Batch alerts for aggregated notifications

        Args:
            alerts: List of alerts to batch
            batch_by: Batching key ('category', 'priority', 'source')

        Returns:
            Dictionary of batched alerts
        """
        batches = defaultdict(list)

        for alert in alerts:
            if batch_by == "category":
                key = alert.category.value
            elif batch_by == "priority":
                key = alert.priority.value
            elif batch_by == "source":
                key = alert.source
            else:
                key = "unknown"

            batches[key].append(alert)

        return dict(batches)

    def export_alerts(
        self,
        filepath: str,
        include_resolved: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Export alerts to JSON file

        Args:
            filepath: Output file path
            include_resolved: Include resolved alerts
            start_date: Filter start date
            end_date: Filter end date
        """
        alerts = list(self.active_alerts.values())

        if include_resolved:
            alerts.extend([a for a in self.alert_history if a.status == AlertStatus.RESOLVED])

        # Filter by date
        if start_date:
            alerts = [a for a in alerts if a.timestamp >= start_date]
        if end_date:
            alerts = [a for a in alerts if a.timestamp <= end_date]

        # Convert to dict
        alert_dicts = [a.to_dict() for a in alerts]

        with open(filepath, 'w') as f:
            json.dump(alert_dicts, f, indent=2)

    def _should_deduplicate(
        self,
        signature: str,
        priority: AlertPriority
    ) -> bool:
        """Check if alert should be deduplicated"""
        if signature not in self.alert_signatures:
            return False

        last_seen = self.alert_signatures[signature]
        time_since = (datetime.now() - last_seen).total_seconds()

        # Different cooldown based on priority
        cooldown_map = {
            AlertPriority.CRITICAL: 60,  # 1 minute
            AlertPriority.HIGH: 300,  # 5 minutes
            AlertPriority.MEDIUM: 600,  # 10 minutes
            AlertPriority.LOW: 1800,  # 30 minutes
            AlertPriority.INFO: 3600  # 1 hour
        }

        cooldown = cooldown_map.get(priority, 300)

        return time_since < cooldown

    def _generate_signature(
        self,
        category: AlertCategory,
        title: str,
        metric_name: Optional[str],
        source: str
    ) -> str:
        """Generate alert signature for deduplication"""
        sig_string = f"{category.value}:{title}:{metric_name}:{source}"
        return hashlib.md5(sig_string.encode()).hexdigest()

    def _generate_alert_id(self, timestamp: datetime) -> str:
        """Generate unique alert ID"""
        time_str = timestamp.strftime('%Y%m%d%H%M%S%f')
        return f"ALERT_{time_str}"

    def _trigger_notifications(
        self,
        alert: Alert,
        escalation: bool = False
    ):
        """Trigger notifications for alert"""
        for channel_name, callback in self.notification_callbacks.items():
            try:
                callback(alert)
            except Exception as e:
                # Log error but don't fail
                print(f"Notification error on channel {channel_name}: {e}")

    def _get_default_rules(self) -> List[AlertRule]:
        """Get default alert rules"""
        return [
            AlertRule(
                rule_id="VAR001",
                rule_name="VaR Limit Breach",
                category=AlertCategory.VAR_BREACH,
                priority=AlertPriority.HIGH,
                condition="var_95 > 0.02",
                message_template="VaR (95%) exceeded limit: {current_value:.2%} > {threshold_value:.2%}",
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="DD001",
                rule_name="Maximum Drawdown Breach",
                category=AlertCategory.DRAWDOWN,
                priority=AlertPriority.CRITICAL,
                condition="abs(drawdown) > 0.20",
                message_template="Drawdown exceeded maximum: {current_value:.2%}",
                cooldown_seconds=60
            ),
            AlertRule(
                rule_id="LEV001",
                rule_name="Leverage Limit Breach",
                category=AlertCategory.LEVERAGE,
                priority=AlertPriority.HIGH,
                condition="leverage > 2.0",
                message_template="Leverage exceeded limit: {current_value:.2f}x > {threshold_value:.2f}x",
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="VOL001",
                rule_name="High Volatility Warning",
                category=AlertCategory.VOLATILITY,
                priority=AlertPriority.MEDIUM,
                condition="volatility > 0.40",
                message_template="Portfolio volatility elevated: {current_value:.2%}",
                cooldown_seconds=600
            )
        ]

    def generate_alert_report(self) -> str:
        """
        Generate text report of current alert status

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ALERT MANAGER STATUS")
        lines.append("=" * 70)
        lines.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        summary = self.get_alert_summary()
        lines.append(f"Active Alerts: {summary['total_active']}")
        lines.append(f"Unacknowledged: {summary['unacknowledged']}")
        lines.append("")

        # By priority
        lines.append("BY PRIORITY:")
        for priority in [AlertPriority.CRITICAL, AlertPriority.HIGH, AlertPriority.MEDIUM, AlertPriority.LOW]:
            count = summary['by_priority'].get(priority.value, 0)
            lines.append(f"  {priority.value.upper():<12}: {count:>4}")
        lines.append("")

        # Recent alerts
        recent_alerts = self.get_active_alerts()[:10]
        if recent_alerts:
            lines.append("RECENT ACTIVE ALERTS:")
            for alert in recent_alerts:
                status_str = "ACK" if alert.status == AlertStatus.ACKNOWLEDGED else "NEW"
                lines.append(
                    f"  [{alert.priority.value.upper():<8}] [{status_str:<3}] "
                    f"{alert.timestamp.strftime('%H:%M:%S')} - {alert.title}"
                )
        lines.append("")

        # Statistics
        stats = self.get_alert_statistics()
        lines.append("STATISTICS:")
        lines.append(f"  Total Generated: {stats['total_generated']}")
        lines.append(f"  Total Deduplicated: {stats['total_deduplicated']}")
        lines.append(f"  Total Acknowledged: {stats['total_acknowledged']}")
        lines.append(f"  Total Resolved: {stats['total_resolved']}")
        lines.append(f"  Total Escalated: {stats['total_escalated']}")

        lines.append("=" * 70)

        return "\n".join(lines)

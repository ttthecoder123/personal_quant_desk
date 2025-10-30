"""
Alert Rules - Rule definitions and management.

Features:
- Critical/high/medium/low priority alerts
- Custom rule definitions
- Dynamic threshold adjustment
- Time-based rules
- Rule validation
"""

import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class RulePriority(Enum):
    """Rule priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleOperator(Enum):
    """Comparison operators."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN_RANGE = "in_range"
    OUT_OF_RANGE = "out_of_range"


class AggregationMethod(Enum):
    """Aggregation methods for time-based rules."""
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    COUNT = "count"
    PERCENTILE = "percentile"


@dataclass
class TimeWindow:
    """Time window for rule evaluation."""
    start_time: time
    end_time: time
    days_of_week: List[int] = field(default_factory=lambda: list(range(7)))  # 0=Monday

    def is_active(self, dt: datetime) -> bool:
        """Check if current time is within window."""
        if dt.weekday() not in self.days_of_week:
            return False

        current_time = dt.time()
        if self.start_time <= self.end_time:
            return self.start_time <= current_time <= self.end_time
        else:
            # Window crosses midnight
            return current_time >= self.start_time or current_time <= self.end_time


@dataclass
class ThresholdConfig:
    """Threshold configuration with dynamic adjustment."""
    value: float
    operator: RuleOperator
    adjustment_enabled: bool = False
    adjustment_factor: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def get_current_threshold(self, baseline_value: Optional[float] = None) -> float:
        """Get current threshold value with dynamic adjustment."""
        if self.adjustment_enabled and baseline_value is not None:
            adjusted = baseline_value * self.adjustment_factor

            # Apply bounds
            if self.min_value is not None:
                adjusted = max(adjusted, self.min_value)
            if self.max_value is not None:
                adjusted = min(adjusted, self.max_value)

            return adjusted

        return self.value


@dataclass
class AlertRule:
    """Alert rule definition."""
    id: str
    name: str
    description: str
    priority: RulePriority
    enabled: bool = True

    # Metric configuration
    metric_name: str = ""
    threshold: Optional[ThresholdConfig] = None

    # Time-based configuration
    time_window: Optional[TimeWindow] = None
    aggregation_method: Optional[AggregationMethod] = None
    aggregation_window_seconds: int = 60

    # Custom condition
    custom_condition: Optional[str] = None  # Python expression

    # Alert configuration
    alert_message: str = ""
    alert_metadata: Dict[str, Any] = field(default_factory=dict)

    # Rate limiting
    cooldown_seconds: int = 300  # 5 minutes
    max_alerts_per_hour: int = 10

    # Tracking
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    trigger_count_reset_time: Optional[datetime] = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def can_trigger(self) -> bool:
        """Check if rule can trigger based on rate limits."""
        now = datetime.now()

        # Check cooldown
        if self.last_triggered:
            elapsed = (now - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False

        # Check hourly limit
        if self.trigger_count_reset_time:
            if now - self.trigger_count_reset_time > timedelta(hours=1):
                # Reset counter
                self.trigger_count = 0
                self.trigger_count_reset_time = now
        else:
            self.trigger_count_reset_time = now

        if self.trigger_count >= self.max_alerts_per_hour:
            return False

        return True

    def record_trigger(self):
        """Record that rule was triggered."""
        self.last_triggered = datetime.now()
        self.trigger_count += 1

    def is_active(self) -> bool:
        """Check if rule is currently active."""
        if not self.enabled:
            return False

        if self.time_window:
            return self.time_window.is_active(datetime.now())

        return True


@dataclass
class RuleValidationResult:
    """Result of rule validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RuleValidator:
    """Validates alert rules."""

    @staticmethod
    def validate_rule(rule: AlertRule) -> RuleValidationResult:
        """
        Validate alert rule.

        Args:
            rule: Rule to validate

        Returns:
            Validation result
        """
        result = RuleValidationResult(valid=True)

        # Basic validation
        if not rule.name:
            result.errors.append("Rule name is required")
            result.valid = False

        if not rule.metric_name and not rule.custom_condition:
            result.errors.append("Either metric_name or custom_condition is required")
            result.valid = False

        # Threshold validation
        if rule.metric_name and not rule.threshold:
            result.errors.append("Threshold is required for metric-based rules")
            result.valid = False

        if rule.threshold:
            if rule.threshold.adjustment_enabled:
                if rule.threshold.adjustment_factor <= 0:
                    result.errors.append("Adjustment factor must be positive")
                    result.valid = False

                if rule.threshold.min_value is not None and rule.threshold.max_value is not None:
                    if rule.threshold.min_value > rule.threshold.max_value:
                        result.errors.append("Min value cannot be greater than max value")
                        result.valid = False

        # Custom condition validation
        if rule.custom_condition:
            try:
                # Try to compile the expression
                compile(rule.custom_condition, '<string>', 'eval')
            except SyntaxError as e:
                result.errors.append(f"Invalid custom condition: {e}")
                result.valid = False

        # Rate limiting validation
        if rule.cooldown_seconds < 0:
            result.errors.append("Cooldown cannot be negative")
            result.valid = False

        if rule.max_alerts_per_hour < 1:
            result.errors.append("Max alerts per hour must be at least 1")
            result.valid = False

        # Warnings
        if rule.cooldown_seconds > 3600:
            result.warnings.append("Cooldown is greater than 1 hour")

        if not rule.alert_message:
            result.warnings.append("Alert message is empty")

        return result


class AlertRules:
    """
    Alert rules management system.

    Features:
    - Rule CRUD operations
    - Rule validation
    - Dynamic threshold adjustment
    - Time-based rule activation
    - Rule prioritization
    """

    def __init__(self):
        """Initialize alert rules manager."""
        self.rules: Dict[str, AlertRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}  # Group name -> rule IDs
        self.lock = threading.Lock()
        self.validator = RuleValidator()

        # Baseline metrics for dynamic thresholds
        self.baseline_metrics: Dict[str, float] = {}

        # Pre-defined rule templates
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        # Critical: System resource exhaustion
        self.add_rule(AlertRule(
            id="cpu_critical",
            name="Critical CPU Usage",
            description="CPU usage exceeds critical threshold",
            priority=RulePriority.CRITICAL,
            metric_name="system.cpu.percent",
            threshold=ThresholdConfig(
                value=95.0,
                operator=RuleOperator.GREATER_THAN
            ),
            alert_message="CRITICAL: CPU usage at {value:.1f}% (threshold: {threshold}%)",
            cooldown_seconds=300
        ))

        self.add_rule(AlertRule(
            id="memory_critical",
            name="Critical Memory Usage",
            description="Memory usage exceeds critical threshold",
            priority=RulePriority.CRITICAL,
            metric_name="system.memory.percent",
            threshold=ThresholdConfig(
                value=95.0,
                operator=RuleOperator.GREATER_THAN
            ),
            alert_message="CRITICAL: Memory usage at {value:.1f}% (threshold: {threshold}%)",
            cooldown_seconds=300
        ))

        # High: Trading system errors
        self.add_rule(AlertRule(
            id="order_failure_high",
            name="High Order Failure Rate",
            description="Order failure rate exceeds threshold",
            priority=RulePriority.HIGH,
            metric_name="trading.order_failure_rate",
            threshold=ThresholdConfig(
                value=0.05,
                operator=RuleOperator.GREATER_THAN
            ),
            alert_message="HIGH: Order failure rate at {value:.2%} (threshold: {threshold:.2%})",
            cooldown_seconds=600
        ))

        # Medium: Performance degradation
        self.add_rule(AlertRule(
            id="latency_medium",
            name="Elevated Latency",
            description="System latency exceeds normal range",
            priority=RulePriority.MEDIUM,
            metric_name="system.latency.p95",
            threshold=ThresholdConfig(
                value=100.0,
                operator=RuleOperator.GREATER_THAN,
                adjustment_enabled=True,
                adjustment_factor=2.0,
                min_value=50.0,
                max_value=500.0
            ),
            alert_message="MEDIUM: Latency elevated at {value:.1f}ms (threshold: {threshold:.1f}ms)",
            cooldown_seconds=900
        ))

        # Low: Information alerts
        self.add_rule(AlertRule(
            id="position_count_low",
            name="Position Count Alert",
            description="Number of open positions exceeds normal",
            priority=RulePriority.LOW,
            metric_name="trading.position_count",
            threshold=ThresholdConfig(
                value=50.0,
                operator=RuleOperator.GREATER_THAN
            ),
            alert_message="INFO: Position count at {value} (threshold: {threshold})",
            cooldown_seconds=1800
        ))

    def add_rule(self, rule: AlertRule) -> bool:
        """
        Add new alert rule.

        Args:
            rule: Alert rule to add

        Returns:
            True if added successfully, False otherwise
        """
        # Validate rule
        validation = self.validator.validate_rule(rule)
        if not validation.valid:
            print(f"Rule validation failed: {validation.errors}")
            return False

        with self.lock:
            if rule.id in self.rules:
                print(f"Rule {rule.id} already exists")
                return False

            self.rules[rule.id] = rule
            return True

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update existing rule.

        Args:
            rule_id: Rule ID to update
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully
        """
        with self.lock:
            if rule_id not in self.rules:
                return False

            rule = self.rules[rule_id]

            # Apply updates
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)

            rule.updated_at = datetime.now()

            # Re-validate
            validation = self.validator.validate_rule(rule)
            if not validation.valid:
                print(f"Updated rule validation failed: {validation.errors}")
                return False

            return True

    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete rule.

        Args:
            rule_id: Rule ID to delete

        Returns:
            True if deleted successfully
        """
        with self.lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                # Remove from groups
                for group_rules in self.rule_groups.values():
                    if rule_id in group_rules:
                        group_rules.remove(rule_id)
                return True
            return False

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """
        Get rule by ID.

        Args:
            rule_id: Rule ID

        Returns:
            Alert rule or None
        """
        with self.lock:
            return self.rules.get(rule_id)

    def get_all_rules(self, enabled_only: bool = False) -> List[AlertRule]:
        """
        Get all rules.

        Args:
            enabled_only: Only return enabled rules

        Returns:
            List of rules
        """
        with self.lock:
            rules = list(self.rules.values())

            if enabled_only:
                rules = [r for r in rules if r.enabled]

            return rules

    def get_active_rules(self) -> List[AlertRule]:
        """
        Get currently active rules.

        Returns:
            List of active rules
        """
        with self.lock:
            return [r for r in self.rules.values() if r.is_active()]

    def enable_rule(self, rule_id: str) -> bool:
        """Enable rule."""
        with self.lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = True
                return True
            return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable rule."""
        with self.lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = False
                return True
            return False

    def add_to_group(self, group_name: str, rule_id: str) -> bool:
        """
        Add rule to group.

        Args:
            group_name: Group name
            rule_id: Rule ID

        Returns:
            True if added successfully
        """
        with self.lock:
            if rule_id not in self.rules:
                return False

            if group_name not in self.rule_groups:
                self.rule_groups[group_name] = []

            if rule_id not in self.rule_groups[group_name]:
                self.rule_groups[group_name].append(rule_id)

            return True

    def get_group_rules(self, group_name: str) -> List[AlertRule]:
        """
        Get all rules in a group.

        Args:
            group_name: Group name

        Returns:
            List of rules
        """
        with self.lock:
            rule_ids = self.rule_groups.get(group_name, [])
            return [self.rules[rid] for rid in rule_ids if rid in self.rules]

    def update_baseline_metric(self, metric_name: str, value: float):
        """
        Update baseline metric for dynamic threshold adjustment.

        Args:
            metric_name: Metric name
            value: Baseline value
        """
        with self.lock:
            self.baseline_metrics[metric_name] = value

    def evaluate_rule(
        self,
        rule: AlertRule,
        metric_value: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate if rule should trigger.

        Args:
            rule: Rule to evaluate
            metric_value: Current metric value
            context: Additional context for custom conditions

        Returns:
            Tuple of (should_trigger, reason)
        """
        if not rule.is_active():
            return False, "Rule not active"

        if not rule.can_trigger():
            return False, "Rate limit exceeded"

        # Evaluate metric-based rule
        if rule.metric_name and rule.threshold:
            if metric_value is None:
                return False, "No metric value provided"

            # Get threshold with dynamic adjustment
            baseline = self.baseline_metrics.get(rule.metric_name)
            threshold = rule.threshold.get_current_threshold(baseline)

            # Compare based on operator
            triggered = self._compare_value(
                metric_value,
                threshold,
                rule.threshold.operator
            )

            if triggered:
                return True, f"Metric {rule.metric_name} = {metric_value} {rule.threshold.operator.value} {threshold}"

        # Evaluate custom condition
        if rule.custom_condition:
            try:
                context = context or {}
                result = eval(rule.custom_condition, {"__builtins__": {}}, context)
                if result:
                    return True, f"Custom condition triggered: {rule.custom_condition}"
            except Exception as e:
                print(f"Error evaluating custom condition: {e}")
                return False, f"Evaluation error: {e}"

        return False, None

    def _compare_value(
        self,
        value: float,
        threshold: float,
        operator: RuleOperator
    ) -> bool:
        """Compare value against threshold."""
        if operator == RuleOperator.GREATER_THAN:
            return value > threshold
        elif operator == RuleOperator.LESS_THAN:
            return value < threshold
        elif operator == RuleOperator.GREATER_EQUAL:
            return value >= threshold
        elif operator == RuleOperator.LESS_EQUAL:
            return value <= threshold
        elif operator == RuleOperator.EQUAL:
            return abs(value - threshold) < 1e-6
        elif operator == RuleOperator.NOT_EQUAL:
            return abs(value - threshold) >= 1e-6

        return False

    def export_rules(self, rule_ids: Optional[List[str]] = None) -> str:
        """
        Export rules to JSON.

        Args:
            rule_ids: Optional list of rule IDs to export

        Returns:
            JSON string of rules
        """
        with self.lock:
            if rule_ids:
                rules = [self.rules[rid] for rid in rule_ids if rid in self.rules]
            else:
                rules = list(self.rules.values())

            # Convert to dict
            rules_dict = []
            for rule in rules:
                rule_dict = {
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'priority': rule.priority.value,
                    'enabled': rule.enabled,
                    'metric_name': rule.metric_name,
                    'alert_message': rule.alert_message,
                    'cooldown_seconds': rule.cooldown_seconds,
                    'max_alerts_per_hour': rule.max_alerts_per_hour
                }

                if rule.threshold:
                    rule_dict['threshold'] = {
                        'value': rule.threshold.value,
                        'operator': rule.threshold.operator.value,
                        'adjustment_enabled': rule.threshold.adjustment_enabled,
                        'adjustment_factor': rule.threshold.adjustment_factor
                    }

                rules_dict.append(rule_dict)

            return json.dumps(rules_dict, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rule statistics.

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            total_rules = len(self.rules)
            enabled_rules = sum(1 for r in self.rules.values() if r.enabled)
            active_rules = len([r for r in self.rules.values() if r.is_active()])

            # Count by priority
            priority_counts = {}
            for priority in RulePriority:
                priority_counts[priority.value] = sum(
                    1 for r in self.rules.values()
                    if r.priority == priority
                )

            return {
                'total_rules': total_rules,
                'enabled_rules': enabled_rules,
                'active_rules': active_rules,
                'disabled_rules': total_rules - enabled_rules,
                'priority_distribution': priority_counts,
                'rule_groups': len(self.rule_groups)
            }

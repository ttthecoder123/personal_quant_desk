"""
Alert Suppression - Intelligent alert filtering and deduplication.

Features:
- Duplicate detection
- Alert correlation
- Maintenance mode
- Scheduled suppression
- Alert fatigue prevention
- Smart grouping
- Rate limiting
"""

import threading
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import re


class SuppressionType(Enum):
    """Types of alert suppression."""
    DUPLICATE = "duplicate"
    MAINTENANCE = "maintenance"
    SCHEDULED = "scheduled"
    RATE_LIMIT = "rate_limit"
    CORRELATION = "correlation"
    MANUAL = "manual"


@dataclass
class SuppressionRule:
    """Suppression rule definition."""
    rule_id: str
    name: str
    suppression_type: SuppressionType
    enabled: bool = True

    # Matching criteria
    severity_levels: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    source_patterns: Optional[List[str]] = None
    metric_patterns: Optional[List[str]] = None
    tag_patterns: Optional[Dict[str, str]] = None

    # Time-based suppression
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    daily_start: Optional[time] = None
    daily_end: Optional[time] = None
    days_of_week: Optional[List[int]] = None

    # Rate limiting
    max_alerts_per_window: Optional[int] = None
    window_seconds: int = 300

    # Grouping
    group_by: Optional[List[str]] = None
    group_window_seconds: int = 300

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MaintenanceWindow:
    """Maintenance window definition."""
    window_id: str
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    affected_components: List[str] = field(default_factory=list)
    affected_categories: List[str] = field(default_factory=list)
    suppress_all: bool = False
    created_by: Optional[str] = None


@dataclass
class AlertFingerprint:
    """Unique fingerprint for alert deduplication."""
    fingerprint: str
    alert_id: str
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int = 1
    suppressed_count: int = 0


@dataclass
class AlertGroup:
    """Group of related alerts."""
    group_id: str
    alerts: List[str] = field(default_factory=list)
    fingerprints: Set[str] = field(default_factory=set)
    first_alert_time: datetime = field(default_factory=datetime.now)
    last_alert_time: datetime = field(default_factory=datetime.now)
    suppression_count: int = 0


@dataclass
class SuppressionDecision:
    """Result of suppression evaluation."""
    should_suppress: bool
    reasons: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    group_id: Optional[str] = None
    fingerprint: Optional[str] = None


class DuplicateDetector:
    """Detects duplicate alerts."""

    def __init__(self, dedup_window_seconds: int = 300):
        """
        Initialize duplicate detector.

        Args:
            dedup_window_seconds: Time window for deduplication
        """
        self.dedup_window_seconds = dedup_window_seconds
        self.fingerprints: Dict[str, AlertFingerprint] = {}
        self.lock = threading.Lock()

    def generate_fingerprint(
        self,
        alert_id: str,
        severity: str,
        category: str,
        source: str,
        metric_name: Optional[str] = None,
        message: Optional[str] = None
    ) -> str:
        """
        Generate unique fingerprint for alert.

        Args:
            alert_id: Alert ID
            severity: Alert severity
            category: Alert category
            source: Alert source
            metric_name: Optional metric name
            message: Optional message for fingerprinting

        Returns:
            Fingerprint hash
        """
        # Create fingerprint from key components
        fingerprint_data = f"{severity}|{category}|{source}"

        if metric_name:
            fingerprint_data += f"|{metric_name}"

        # Extract key parts from message (remove dynamic values)
        if message:
            # Remove numbers and timestamps
            cleaned_message = re.sub(r'\d+', 'X', message)
            cleaned_message = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', cleaned_message)
            cleaned_message = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', cleaned_message)
            fingerprint_data += f"|{cleaned_message[:100]}"

        return hashlib.md5(fingerprint_data.encode()).hexdigest()

    def is_duplicate(
        self,
        fingerprint: str,
        alert_id: str
    ) -> Tuple[bool, Optional[AlertFingerprint]]:
        """
        Check if alert is a duplicate.

        Args:
            fingerprint: Alert fingerprint
            alert_id: Alert ID

        Returns:
            Tuple of (is_duplicate, existing_fingerprint)
        """
        with self.lock:
            now = datetime.now()

            # Clean up old fingerprints
            expired = []
            for fp, data in self.fingerprints.items():
                if (now - data.last_seen).total_seconds() > self.dedup_window_seconds:
                    expired.append(fp)

            for fp in expired:
                del self.fingerprints[fp]

            # Check for duplicate
            if fingerprint in self.fingerprints:
                existing = self.fingerprints[fingerprint]
                existing.last_seen = now
                existing.occurrence_count += 1
                existing.suppressed_count += 1
                return True, existing

            # Record new fingerprint
            self.fingerprints[fingerprint] = AlertFingerprint(
                fingerprint=fingerprint,
                alert_id=alert_id,
                first_seen=now,
                last_seen=now
            )

            return False, None


class AlertGrouper:
    """Groups related alerts together."""

    def __init__(self, group_window_seconds: int = 300):
        """
        Initialize alert grouper.

        Args:
            group_window_seconds: Time window for grouping
        """
        self.group_window_seconds = group_window_seconds
        self.groups: Dict[str, AlertGroup] = {}
        self.alert_to_group: Dict[str, str] = {}
        self.lock = threading.Lock()

    def add_to_group(
        self,
        alert_id: str,
        fingerprint: str,
        group_key: str
    ) -> Tuple[bool, str]:
        """
        Add alert to appropriate group.

        Args:
            alert_id: Alert ID
            fingerprint: Alert fingerprint
            group_key: Key for grouping

        Returns:
            Tuple of (is_suppressed, group_id)
        """
        with self.lock:
            now = datetime.now()

            # Clean up old groups
            expired = []
            for gid, group in self.groups.items():
                if (now - group.last_alert_time).total_seconds() > self.group_window_seconds:
                    expired.append(gid)

            for gid in expired:
                group = self.groups[gid]
                for alert in group.alerts:
                    if alert in self.alert_to_group:
                        del self.alert_to_group[alert]
                del self.groups[gid]

            # Find or create group
            group_hash = hashlib.md5(group_key.encode()).hexdigest()[:16]

            if group_hash in self.groups:
                # Add to existing group
                group = self.groups[group_hash]
                group.alerts.append(alert_id)
                group.fingerprints.add(fingerprint)
                group.last_alert_time = now
                group.suppression_count += 1
                self.alert_to_group[alert_id] = group_hash

                # Suppress if not the first alert in group
                return True, group_hash
            else:
                # Create new group
                group = AlertGroup(
                    group_id=group_hash,
                    alerts=[alert_id],
                    fingerprints={fingerprint},
                    first_alert_time=now,
                    last_alert_time=now
                )
                self.groups[group_hash] = group
                self.alert_to_group[alert_id] = group_hash

                # Don't suppress first alert in group
                return False, group_hash

    def get_group(self, group_id: str) -> Optional[AlertGroup]:
        """Get alert group by ID."""
        with self.lock:
            return self.groups.get(group_id)


class RateLimiter:
    """Rate limits alerts per source/metric."""

    def __init__(self):
        """Initialize rate limiter."""
        self.alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()

    def should_limit(
        self,
        key: str,
        max_alerts: int,
        window_seconds: int
    ) -> bool:
        """
        Check if alert should be rate limited.

        Args:
            key: Rate limit key
            max_alerts: Maximum alerts in window
            window_seconds: Time window in seconds

        Returns:
            True if should be limited
        """
        with self.lock:
            now = datetime.now()
            threshold = now - timedelta(seconds=window_seconds)

            # Get timestamps for this key
            timestamps = self.alert_counts[key]

            # Remove old timestamps
            while timestamps and timestamps[0] < threshold:
                timestamps.popleft()

            # Check limit
            if len(timestamps) >= max_alerts:
                return True

            # Record new alert
            timestamps.append(now)
            return False


class AlertSuppression:
    """
    Intelligent alert suppression system.

    Features:
    - Duplicate detection and deduplication
    - Alert correlation and grouping
    - Maintenance window support
    - Scheduled suppression
    - Rate limiting
    - Alert fatigue prevention
    """

    def __init__(self):
        """Initialize alert suppression."""
        self.suppression_rules: List[SuppressionRule] = []
        self.maintenance_windows: Dict[str, MaintenanceWindow] = {}
        self.lock = threading.Lock()

        # Sub-components
        self.duplicate_detector = DuplicateDetector()
        self.alert_grouper = AlertGrouper()
        self.rate_limiter = RateLimiter()

        # Statistics
        self.suppression_stats = defaultdict(int)
        self.suppression_history: deque = deque(maxlen=10000)

    def add_rule(self, rule: SuppressionRule):
        """
        Add suppression rule.

        Args:
            rule: Suppression rule to add
        """
        with self.lock:
            # Insert by priority (more specific rules first)
            self.suppression_rules.append(rule)
            self.suppression_rules.sort(
                key=lambda r: (
                    len(r.source_patterns or []) +
                    len(r.metric_patterns or []) +
                    len(r.tag_patterns or {})
                ),
                reverse=True
            )

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove suppression rule.

        Args:
            rule_id: Rule ID to remove

        Returns:
            True if removed successfully
        """
        with self.lock:
            for i, rule in enumerate(self.suppression_rules):
                if rule.rule_id == rule_id:
                    del self.suppression_rules[i]
                    return True
            return False

    def add_maintenance_window(self, window: MaintenanceWindow):
        """
        Add maintenance window.

        Args:
            window: Maintenance window to add
        """
        with self.lock:
            self.maintenance_windows[window.window_id] = window

    def remove_maintenance_window(self, window_id: str) -> bool:
        """
        Remove maintenance window.

        Args:
            window_id: Window ID to remove

        Returns:
            True if removed successfully
        """
        with self.lock:
            if window_id in self.maintenance_windows:
                del self.maintenance_windows[window_id]
                return True
            return False

    def should_suppress(
        self,
        alert_id: str,
        severity: str,
        category: str,
        source: str,
        metric_name: Optional[str] = None,
        message: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> SuppressionDecision:
        """
        Evaluate if alert should be suppressed.

        Args:
            alert_id: Alert ID
            severity: Alert severity
            category: Alert category
            source: Alert source
            metric_name: Optional metric name
            message: Optional message
            tags: Optional tags

        Returns:
            Suppression decision
        """
        tags = tags or {}
        decision = SuppressionDecision(should_suppress=False)

        # Check maintenance windows
        if self._is_in_maintenance(source, category):
            decision.should_suppress = True
            decision.reasons.append("Maintenance window active")
            self.suppression_stats[SuppressionType.MAINTENANCE.value] += 1
            self._record_suppression(alert_id, SuppressionType.MAINTENANCE)
            return decision

        # Check duplicate detection
        fingerprint = self.duplicate_detector.generate_fingerprint(
            alert_id, severity, category, source, metric_name, message
        )
        is_duplicate, existing = self.duplicate_detector.is_duplicate(fingerprint, alert_id)

        decision.fingerprint = fingerprint

        if is_duplicate:
            decision.should_suppress = True
            decision.reasons.append(f"Duplicate alert (seen {existing.occurrence_count} times)")
            self.suppression_stats[SuppressionType.DUPLICATE.value] += 1
            self._record_suppression(alert_id, SuppressionType.DUPLICATE)
            return decision

        # Check suppression rules
        with self.lock:
            for rule in self.suppression_rules:
                if not rule.enabled:
                    continue

                if self._matches_rule(rule, severity, category, source, metric_name, tags):
                    decision.matched_rules.append(rule.rule_id)

                    # Check time-based suppression
                    if not self._is_time_active(rule):
                        decision.should_suppress = True
                        decision.reasons.append(f"Scheduled suppression: {rule.name}")
                        self.suppression_stats[SuppressionType.SCHEDULED.value] += 1
                        self._record_suppression(alert_id, SuppressionType.SCHEDULED)
                        return decision

                    # Check rate limiting
                    if rule.max_alerts_per_window:
                        rate_key = self._get_rate_key(source, metric_name)
                        if self.rate_limiter.should_limit(
                            rate_key,
                            rule.max_alerts_per_window,
                            rule.window_seconds
                        ):
                            decision.should_suppress = True
                            decision.reasons.append(f"Rate limit exceeded: {rule.name}")
                            self.suppression_stats[SuppressionType.RATE_LIMIT.value] += 1
                            self._record_suppression(alert_id, SuppressionType.RATE_LIMIT)
                            return decision

                    # Check grouping
                    if rule.group_by:
                        group_key = self._get_group_key(
                            rule.group_by,
                            severity, category, source, metric_name, tags
                        )
                        is_grouped, group_id = self.alert_grouper.add_to_group(
                            alert_id, fingerprint, group_key
                        )
                        decision.group_id = group_id

                        if is_grouped:
                            decision.should_suppress = True
                            decision.reasons.append(f"Grouped with existing alerts: {group_id}")
                            self.suppression_stats[SuppressionType.CORRELATION.value] += 1
                            self._record_suppression(alert_id, SuppressionType.CORRELATION)
                            return decision

        return decision

    def _is_in_maintenance(self, source: str, category: str) -> bool:
        """Check if source/category is in maintenance window."""
        with self.lock:
            now = datetime.now()

            for window in self.maintenance_windows.values():
                # Check time window
                if not (window.start_time <= now <= window.end_time):
                    continue

                # Check if applies to this alert
                if window.suppress_all:
                    return True

                if source in window.affected_components:
                    return True

                if category in window.affected_categories:
                    return True

            return False

    def _matches_rule(
        self,
        rule: SuppressionRule,
        severity: str,
        category: str,
        source: str,
        metric_name: Optional[str],
        tags: Dict[str, str]
    ) -> bool:
        """Check if alert matches suppression rule."""
        # Check severity
        if rule.severity_levels and severity not in rule.severity_levels:
            return False

        # Check category
        if rule.categories and category not in rule.categories:
            return False

        # Check source patterns
        if rule.source_patterns:
            matched = False
            for pattern in rule.source_patterns:
                if re.match(pattern, source):
                    matched = True
                    break
            if not matched:
                return False

        # Check metric patterns
        if rule.metric_patterns and metric_name:
            matched = False
            for pattern in rule.metric_patterns:
                if re.match(pattern, metric_name):
                    matched = True
                    break
            if not matched:
                return False

        # Check tag patterns
        if rule.tag_patterns:
            for key, pattern in rule.tag_patterns.items():
                if key not in tags:
                    return False
                if not re.match(pattern, tags[key]):
                    return False

        return True

    def _is_time_active(self, rule: SuppressionRule) -> bool:
        """Check if rule is currently active (not suppressed by time)."""
        now = datetime.now()

        # Check absolute time window
        if rule.start_time and rule.end_time:
            if not (rule.start_time <= now <= rule.end_time):
                return False

        # Check daily time window
        if rule.daily_start and rule.daily_end:
            current_time = now.time()
            if rule.daily_start <= rule.daily_end:
                if not (rule.daily_start <= current_time <= rule.daily_end):
                    return False
            else:
                # Crosses midnight
                if not (current_time >= rule.daily_start or current_time <= rule.daily_end):
                    return False

        # Check days of week
        if rule.days_of_week:
            if now.weekday() not in rule.days_of_week:
                return False

        return True

    def _get_rate_key(self, source: str, metric_name: Optional[str]) -> str:
        """Generate rate limiting key."""
        if metric_name:
            return f"{source}|{metric_name}"
        return source

    def _get_group_key(
        self,
        group_by: List[str],
        severity: str,
        category: str,
        source: str,
        metric_name: Optional[str],
        tags: Dict[str, str]
    ) -> str:
        """Generate grouping key."""
        components = []

        for field in group_by:
            if field == 'severity':
                components.append(severity)
            elif field == 'category':
                components.append(category)
            elif field == 'source':
                components.append(source)
            elif field == 'metric':
                components.append(metric_name or 'none')
            elif field.startswith('tag:'):
                tag_name = field[4:]
                components.append(tags.get(tag_name, 'none'))

        return '|'.join(components)

    def _record_suppression(self, alert_id: str, suppression_type: SuppressionType):
        """Record suppression in history."""
        self.suppression_history.append({
            'alert_id': alert_id,
            'type': suppression_type.value,
            'timestamp': datetime.now().isoformat()
        })

    def get_active_maintenance_windows(self) -> List[MaintenanceWindow]:
        """
        Get currently active maintenance windows.

        Returns:
            List of active maintenance windows
        """
        with self.lock:
            now = datetime.now()
            return [
                window for window in self.maintenance_windows.values()
                if window.start_time <= now <= window.end_time
            ]

    def get_suppression_statistics(self) -> Dict[str, any]:
        """
        Get suppression statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            total_suppressed = sum(self.suppression_stats.values())

            # Recent suppression rate
            hour_ago = datetime.now() - timedelta(hours=1)
            recent_suppressions = [
                s for s in self.suppression_history
                if datetime.fromisoformat(s['timestamp']) > hour_ago
            ]

            return {
                'total_suppressed': total_suppressed,
                'suppression_by_type': dict(self.suppression_stats),
                'active_maintenance_windows': len(self.get_active_maintenance_windows()),
                'active_rules': sum(1 for r in self.suppression_rules if r.enabled),
                'total_rules': len(self.suppression_rules),
                'recent_suppressions_1h': len(recent_suppressions),
                'unique_fingerprints': len(self.duplicate_detector.fingerprints),
                'active_groups': len(self.alert_grouper.groups)
            }

    def reset_statistics(self):
        """Reset suppression statistics."""
        with self.lock:
            self.suppression_stats.clear()
            self.suppression_history.clear()

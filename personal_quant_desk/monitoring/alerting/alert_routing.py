"""
Alert Routing - Intelligent alert routing and distribution.

Features:
- Role-based routing
- Severity-based routing
- Time-based routing
- Geographic routing
- Escalation paths
- On-call scheduling
- Alert load balancing
"""

import threading
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib


class UserRole(Enum):
    """User roles for routing."""
    ENGINEER = "engineer"
    OPERATIONS = "operations"
    TRADING = "trading"
    MANAGEMENT = "management"
    ONCALL = "oncall"
    BACKUP = "backup"


class RoutingStrategy(Enum):
    """Alert routing strategies."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    BROADCAST = "broadcast"
    PRIMARY_BACKUP = "primary_backup"


@dataclass
class User:
    """User/recipient for alerts."""
    user_id: str
    name: str
    email: str
    phone: Optional[str] = None
    roles: Set[UserRole] = field(default_factory=set)
    timezone: str = "UTC"
    max_alerts_per_hour: int = 20
    notification_preferences: Dict[str, bool] = field(default_factory=dict)

    # Availability
    available: bool = True
    on_call: bool = False
    quiet_hours_start: Optional[time] = None
    quiet_hours_end: Optional[time] = None


@dataclass
class Team:
    """Team of users."""
    team_id: str
    name: str
    members: List[str] = field(default_factory=list)  # user_ids
    routing_strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    primary_user_id: Optional[str] = None
    backup_user_ids: List[str] = field(default_factory=list)


@dataclass
class RoutingRule:
    """Routing rule definition."""
    rule_id: str
    name: str
    priority: int = 0  # Higher priority evaluated first

    # Matching criteria
    severity_levels: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    tags: Optional[Dict[str, str]] = None
    source_patterns: Optional[List[str]] = None

    # Routing target
    target_user_ids: List[str] = field(default_factory=list)
    target_team_ids: List[str] = field(default_factory=list)
    target_roles: List[UserRole] = field(default_factory=list)

    # Time-based routing
    active_hours_start: Optional[time] = None
    active_hours_end: Optional[time] = None
    active_days: List[int] = field(default_factory=lambda: list(range(7)))  # 0=Monday

    # Geographic routing
    regions: Optional[List[str]] = None

    enabled: bool = True


@dataclass
class OnCallSchedule:
    """On-call schedule entry."""
    schedule_id: str
    user_id: str
    team_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    rotation_hours: int = 24  # Hours before rotation


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    alert_id: str
    recipients: List[str]  # user_ids
    channels: Dict[str, List[str]]  # channel_type -> user_ids
    routing_path: List[str]  # Rule IDs that matched
    timestamp: datetime = field(default_factory=datetime.now)


class LoadBalancer:
    """Load balances alerts across recipients."""

    def __init__(self):
        """Initialize load balancer."""
        self.alert_counts: Dict[str, int] = defaultdict(int)  # user_id -> count
        self.last_recipient_index: Dict[str, int] = defaultdict(int)  # team_id -> index
        self.lock = threading.Lock()

    def get_next_recipient(
        self,
        candidates: List[str],
        team_id: str,
        strategy: RoutingStrategy
    ) -> str:
        """
        Get next recipient based on strategy.

        Args:
            candidates: List of candidate user IDs
            team_id: Team ID for round-robin tracking
            strategy: Routing strategy

        Returns:
            Selected user ID
        """
        if not candidates:
            raise ValueError("No candidates available")

        with self.lock:
            if strategy == RoutingStrategy.ROUND_ROBIN:
                idx = self.last_recipient_index[team_id]
                selected = candidates[idx % len(candidates)]
                self.last_recipient_index[team_id] = (idx + 1) % len(candidates)
                return selected

            elif strategy == RoutingStrategy.LOAD_BALANCED:
                # Select user with fewest alerts
                selected = min(
                    candidates,
                    key=lambda uid: self.alert_counts.get(uid, 0)
                )
                return selected

            elif strategy == RoutingStrategy.PRIORITY_BASED:
                # First candidate is highest priority
                return candidates[0]

            else:
                return candidates[0]

    def record_alert(self, user_id: str):
        """Record alert sent to user."""
        with self.lock:
            self.alert_counts[user_id] += 1

    def reset_counts(self):
        """Reset alert counts (for hourly reset)."""
        with self.lock:
            self.alert_counts.clear()


class AlertRouter:
    """
    Intelligent alert routing system.

    Features:
    - Role-based routing
    - Severity and category-based routing
    - Time-based routing with on-call schedules
    - Geographic routing
    - Load balancing across recipients
    - Escalation path management
    """

    def __init__(self):
        """Initialize alert router."""
        self.users: Dict[str, User] = {}
        self.teams: Dict[str, Team] = {}
        self.routing_rules: List[RoutingRule] = []
        self.on_call_schedules: List[OnCallSchedule] = []
        self.lock = threading.Lock()

        self.load_balancer = LoadBalancer()

        # Routing history
        self.routing_history: deque = deque(maxlen=10000)

        # Alert rate tracking
        self.user_alert_counts: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

    def register_user(self, user: User):
        """
        Register user for alert routing.

        Args:
            user: User to register
        """
        with self.lock:
            self.users[user.user_id] = user

    def unregister_user(self, user_id: str) -> bool:
        """
        Unregister user.

        Args:
            user_id: User ID to unregister

        Returns:
            True if unregistered successfully
        """
        with self.lock:
            if user_id in self.users:
                del self.users[user_id]
                return True
            return False

    def create_team(self, team: Team):
        """
        Create team.

        Args:
            team: Team to create
        """
        with self.lock:
            self.teams[team.team_id] = team

    def add_routing_rule(self, rule: RoutingRule):
        """
        Add routing rule.

        Args:
            rule: Routing rule to add
        """
        with self.lock:
            # Insert in priority order
            inserted = False
            for i, existing_rule in enumerate(self.routing_rules):
                if rule.priority > existing_rule.priority:
                    self.routing_rules.insert(i, rule)
                    inserted = True
                    break

            if not inserted:
                self.routing_rules.append(rule)

    def remove_routing_rule(self, rule_id: str) -> bool:
        """
        Remove routing rule.

        Args:
            rule_id: Rule ID to remove

        Returns:
            True if removed successfully
        """
        with self.lock:
            for i, rule in enumerate(self.routing_rules):
                if rule.rule_id == rule_id:
                    del self.routing_rules[i]
                    return True
            return False

    def set_on_call(self, schedule: OnCallSchedule):
        """
        Set on-call schedule.

        Args:
            schedule: On-call schedule
        """
        with self.lock:
            self.on_call_schedules.append(schedule)

            # Update user on-call status
            if schedule.user_id in self.users:
                self.users[schedule.user_id].on_call = True

    def get_on_call_users(self, team_id: Optional[str] = None) -> List[str]:
        """
        Get currently on-call users.

        Args:
            team_id: Optional team ID to filter by

        Returns:
            List of on-call user IDs
        """
        with self.lock:
            now = datetime.now()
            on_call_users = []

            for schedule in self.on_call_schedules:
                # Check if schedule is active
                if schedule.end_time and now > schedule.end_time:
                    continue

                if team_id and schedule.team_id != team_id:
                    continue

                on_call_users.append(schedule.user_id)

            return on_call_users

    def route_alert(
        self,
        alert_id: str,
        severity: str,
        category: str,
        source: str,
        tags: Optional[Dict[str, str]] = None
    ) -> RoutingDecision:
        """
        Route alert to appropriate recipients.

        Args:
            alert_id: Alert ID
            severity: Alert severity
            category: Alert category
            source: Alert source
            tags: Optional alert tags

        Returns:
            Routing decision
        """
        tags = tags or {}
        matched_rules = []
        recipient_set: Set[str] = set()

        with self.lock:
            # Evaluate routing rules
            for rule in self.routing_rules:
                if not rule.enabled:
                    continue

                if self._matches_rule(rule, severity, category, source, tags):
                    matched_rules.append(rule.rule_id)

                    # Add recipients from rule
                    recipients = self._get_recipients_from_rule(rule)
                    recipient_set.update(recipients)

            # If no rules matched, use default routing
            if not recipient_set:
                recipient_set.update(self._get_default_recipients(severity))

            # Filter out unavailable users and respect rate limits
            final_recipients = self._filter_recipients(list(recipient_set))

            # Determine notification channels
            channels = self._determine_channels(final_recipients, severity)

            # Create routing decision
            decision = RoutingDecision(
                alert_id=alert_id,
                recipients=final_recipients,
                channels=channels,
                routing_path=matched_rules
            )

            # Record routing
            self.routing_history.append(decision)

            # Update alert counts and load balancer
            for user_id in final_recipients:
                self.user_alert_counts[user_id].append(datetime.now())
                self.load_balancer.record_alert(user_id)

            return decision

    def _matches_rule(
        self,
        rule: RoutingRule,
        severity: str,
        category: str,
        source: str,
        tags: Dict[str, str]
    ) -> bool:
        """Check if alert matches routing rule."""
        # Check severity
        if rule.severity_levels and severity not in rule.severity_levels:
            return False

        # Check category
        if rule.categories and category not in rule.categories:
            return False

        # Check tags
        if rule.tags:
            for key, value in rule.tags.items():
                if tags.get(key) != value:
                    return False

        # Check source patterns
        if rule.source_patterns:
            import re
            matched = False
            for pattern in rule.source_patterns:
                if re.match(pattern, source):
                    matched = True
                    break
            if not matched:
                return False

        # Check time-based routing
        if not self._is_time_active(rule):
            return False

        return True

    def _is_time_active(self, rule: RoutingRule) -> bool:
        """Check if rule is active based on time."""
        now = datetime.now()

        # Check day of week
        if now.weekday() not in rule.active_days:
            return False

        # Check time of day
        if rule.active_hours_start and rule.active_hours_end:
            current_time = now.time()
            if rule.active_hours_start <= rule.active_hours_end:
                if not (rule.active_hours_start <= current_time <= rule.active_hours_end):
                    return False
            else:
                # Crosses midnight
                if not (current_time >= rule.active_hours_start or
                       current_time <= rule.active_hours_end):
                    return False

        return True

    def _get_recipients_from_rule(self, rule: RoutingRule) -> List[str]:
        """Get recipient user IDs from rule."""
        recipients: Set[str] = set()

        # Direct user assignments
        recipients.update(rule.target_user_ids)

        # Team assignments
        for team_id in rule.target_team_ids:
            if team_id in self.teams:
                team = self.teams[team_id]

                if team.routing_strategy == RoutingStrategy.BROADCAST:
                    recipients.update(team.members)
                elif team.routing_strategy == RoutingStrategy.PRIMARY_BACKUP:
                    if team.primary_user_id:
                        recipients.add(team.primary_user_id)
                else:
                    # Use load balancer
                    if team.members:
                        recipient = self.load_balancer.get_next_recipient(
                            team.members,
                            team_id,
                            team.routing_strategy
                        )
                        recipients.add(recipient)

        # Role-based assignments
        for role in rule.target_roles:
            for user in self.users.values():
                if role in user.roles:
                    recipients.add(user.user_id)

        return list(recipients)

    def _get_default_recipients(self, severity: str) -> List[str]:
        """Get default recipients based on severity."""
        # Critical/High alerts go to on-call
        if severity in ['critical', 'high']:
            on_call = self.get_on_call_users()
            if on_call:
                return on_call

        # Otherwise, send to operations team
        if 'operations' in self.teams:
            return self.teams['operations'].members

        # Fallback to all users with operations role
        return [
            uid for uid, user in self.users.items()
            if UserRole.OPERATIONS in user.roles
        ]

    def _filter_recipients(self, recipients: List[str]) -> List[str]:
        """Filter recipients based on availability and rate limits."""
        filtered = []
        now = datetime.now()

        for user_id in recipients:
            if user_id not in self.users:
                continue

            user = self.users[user_id]

            # Check availability
            if not user.available:
                continue

            # Check quiet hours
            if user.quiet_hours_start and user.quiet_hours_end:
                current_time = now.time()
                if user.quiet_hours_start <= user.quiet_hours_end:
                    if user.quiet_hours_start <= current_time <= user.quiet_hours_end:
                        continue
                else:
                    if current_time >= user.quiet_hours_start or \
                       current_time <= user.quiet_hours_end:
                        continue

            # Check rate limits
            recent_alerts = [
                t for t in self.user_alert_counts[user_id]
                if t > now - timedelta(hours=1)
            ]
            if len(recent_alerts) >= user.max_alerts_per_hour:
                continue

            filtered.append(user_id)

        return filtered

    def _determine_channels(
        self,
        recipients: List[str],
        severity: str
    ) -> Dict[str, List[str]]:
        """Determine notification channels for recipients."""
        channels = defaultdict(list)

        for user_id in recipients:
            if user_id not in self.users:
                continue

            user = self.users[user_id]
            prefs = user.notification_preferences

            # Email is default
            if prefs.get('email', True):
                channels['email'].append(user_id)

            # Critical/High alerts use multiple channels
            if severity in ['critical', 'high']:
                if user.phone and prefs.get('sms', True):
                    channels['sms'].append(user_id)

                if prefs.get('phone', False):
                    channels['phone'].append(user_id)

                if prefs.get('slack', True):
                    channels['slack'].append(user_id)

            # Medium alerts use email and slack
            elif severity == 'medium':
                if prefs.get('slack', True):
                    channels['slack'].append(user_id)

            # Dashboard alerts for all
            channels['dashboard'].append(user_id)

        return dict(channels)

    def get_routing_statistics(self) -> Dict[str, any]:
        """
        Get routing statistics.

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            # Count alerts per user in last hour
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)

            user_alert_counts = {}
            for user_id, timestamps in self.user_alert_counts.items():
                recent = [t for t in timestamps if t > hour_ago]
                user_alert_counts[user_id] = len(recent)

            # Most loaded users
            most_loaded = sorted(
                user_alert_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            return {
                'total_users': len(self.users),
                'available_users': sum(1 for u in self.users.values() if u.available),
                'on_call_users': len(self.get_on_call_users()),
                'total_teams': len(self.teams),
                'routing_rules': len(self.routing_rules),
                'active_rules': sum(1 for r in self.routing_rules if r.enabled),
                'alerts_last_hour': sum(user_alert_counts.values()),
                'most_loaded_users': [
                    {'user_id': uid, 'alert_count': count}
                    for uid, count in most_loaded
                ]
            }

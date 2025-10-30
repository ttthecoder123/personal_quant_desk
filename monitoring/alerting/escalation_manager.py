"""
Escalation Manager - Alert escalation and response tracking.

Features:
- Automatic escalation
- Time-based escalation
- Response tracking
- Acknowledgment handling
- Escalation chains
- Override mechanisms
- Audit trail
"""

import threading
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json


class AlertStatus(Enum):
    """Alert status in escalation lifecycle."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class EscalationLevel(Enum):
    """Escalation levels."""
    L1 = "l1"  # First responder
    L2 = "l2"  # Team lead
    L3 = "l3"  # Manager
    L4 = "l4"  # Executive
    OVERRIDE = "override"  # Manual override


@dataclass
class EscalationPolicy:
    """Escalation policy definition."""
    policy_id: str
    name: str
    description: str

    # Escalation chain
    escalation_chain: List[Dict[str, any]] = field(default_factory=list)
    # Each entry: {'level': EscalationLevel, 'recipients': [...], 'timeout_minutes': int}

    # Conditions
    severity_levels: Optional[List[str]] = None
    categories: Optional[List[str]] = None

    # Configuration
    auto_escalate: bool = True
    require_acknowledgment: bool = True
    acknowledgment_timeout_minutes: int = 15

    enabled: bool = True


@dataclass
class AlertIncident:
    """Alert incident tracking."""
    incident_id: str
    alert_id: str
    status: AlertStatus
    severity: str
    category: str

    # Escalation tracking
    current_level: EscalationLevel = EscalationLevel.L1
    escalation_count: int = 0
    last_escalation_time: Optional[datetime] = None

    # Response tracking
    assigned_to: Optional[str] = None  # user_id
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    escalation_deadline: Optional[datetime] = None

    # Metadata
    notes: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class EscalationEvent:
    """Escalation event for audit trail."""
    event_id: str
    incident_id: str
    event_type: str  # 'created', 'acknowledged', 'escalated', 'resolved', etc.
    from_level: Optional[EscalationLevel] = None
    to_level: Optional[EscalationLevel] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, any] = field(default_factory=dict)


@dataclass
class Response:
    """User response to alert."""
    response_id: str
    incident_id: str
    user_id: str
    action: str  # 'acknowledged', 'resolved', 'commented', etc.
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class EscalationChain:
    """Manages escalation chain execution."""

    def __init__(self, policy: EscalationPolicy):
        """
        Initialize escalation chain.

        Args:
            policy: Escalation policy to execute
        """
        self.policy = policy
        self.current_step = 0
        self.started_at: Optional[datetime] = None
        self.lock = threading.Lock()

    def start(self) -> Dict[str, any]:
        """
        Start escalation chain.

        Returns:
            First escalation step
        """
        with self.lock:
            self.started_at = datetime.now()
            self.current_step = 0
            return self._get_current_step()

    def get_next_step(self) -> Optional[Dict[str, any]]:
        """
        Get next escalation step.

        Returns:
            Next step or None if at end of chain
        """
        with self.lock:
            self.current_step += 1
            if self.current_step >= len(self.policy.escalation_chain):
                return None
            return self._get_current_step()

    def _get_current_step(self) -> Optional[Dict[str, any]]:
        """Get current escalation step."""
        if self.current_step < len(self.policy.escalation_chain):
            return self.policy.escalation_chain[self.current_step]
        return None


class EscalationManager:
    """
    Alert escalation management system.

    Features:
    - Automatic time-based escalation
    - Escalation policies and chains
    - Response tracking and acknowledgment
    - Override mechanisms
    - Complete audit trail
    """

    def __init__(self):
        """Initialize escalation manager."""
        self.incidents: Dict[str, AlertIncident] = {}
        self.policies: Dict[str, EscalationPolicy] = {}
        self.escalation_chains: Dict[str, EscalationChain] = {}
        self.responses: Dict[str, List[Response]] = {}  # incident_id -> responses
        self.audit_trail: deque = deque(maxlen=100000)
        self.lock = threading.Lock()

        # Callbacks
        self.escalation_callbacks: List[Callable] = []

        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Initialize default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default escalation policies."""
        # Critical alert policy
        critical_policy = EscalationPolicy(
            policy_id="critical_default",
            name="Critical Alert Escalation",
            description="Default escalation for critical alerts",
            escalation_chain=[
                {
                    'level': EscalationLevel.L1,
                    'recipients': ['oncall'],
                    'timeout_minutes': 5
                },
                {
                    'level': EscalationLevel.L2,
                    'recipients': ['team_lead'],
                    'timeout_minutes': 10
                },
                {
                    'level': EscalationLevel.L3,
                    'recipients': ['manager'],
                    'timeout_minutes': 15
                },
                {
                    'level': EscalationLevel.L4,
                    'recipients': ['executive'],
                    'timeout_minutes': 30
                }
            ],
            severity_levels=['critical'],
            acknowledgment_timeout_minutes=5
        )
        self.add_policy(critical_policy)

        # High alert policy
        high_policy = EscalationPolicy(
            policy_id="high_default",
            name="High Alert Escalation",
            description="Default escalation for high priority alerts",
            escalation_chain=[
                {
                    'level': EscalationLevel.L1,
                    'recipients': ['oncall'],
                    'timeout_minutes': 15
                },
                {
                    'level': EscalationLevel.L2,
                    'recipients': ['team_lead'],
                    'timeout_minutes': 30
                },
                {
                    'level': EscalationLevel.L3,
                    'recipients': ['manager'],
                    'timeout_minutes': 60
                }
            ],
            severity_levels=['high'],
            acknowledgment_timeout_minutes=15
        )
        self.add_policy(high_policy)

        # Medium alert policy
        medium_policy = EscalationPolicy(
            policy_id="medium_default",
            name="Medium Alert Escalation",
            description="Default escalation for medium priority alerts",
            escalation_chain=[
                {
                    'level': EscalationLevel.L1,
                    'recipients': ['oncall'],
                    'timeout_minutes': 30
                },
                {
                    'level': EscalationLevel.L2,
                    'recipients': ['team_lead'],
                    'timeout_minutes': 120
                }
            ],
            severity_levels=['medium'],
            acknowledgment_timeout_minutes=30
        )
        self.add_policy(medium_policy)

    def add_policy(self, policy: EscalationPolicy):
        """
        Add escalation policy.

        Args:
            policy: Escalation policy to add
        """
        with self.lock:
            self.policies[policy.policy_id] = policy

    def remove_policy(self, policy_id: str) -> bool:
        """
        Remove escalation policy.

        Args:
            policy_id: Policy ID to remove

        Returns:
            True if removed successfully
        """
        with self.lock:
            if policy_id in self.policies:
                del self.policies[policy_id]
                return True
            return False

    def create_incident(
        self,
        alert_id: str,
        severity: str,
        category: str
    ) -> AlertIncident:
        """
        Create incident from alert.

        Args:
            alert_id: Alert ID
            severity: Alert severity
            category: Alert category

        Returns:
            Created incident
        """
        import hashlib
        import time

        with self.lock:
            # Generate incident ID
            incident_id = hashlib.md5(
                f"{alert_id}_{time.time()}".encode()
            ).hexdigest()[:16]

            # Find matching policy
            policy = self._find_matching_policy(severity, category)

            # Create incident
            incident = AlertIncident(
                incident_id=incident_id,
                alert_id=alert_id,
                status=AlertStatus.NEW,
                severity=severity,
                category=category
            )

            # Calculate escalation deadline
            if policy:
                timeout = policy.acknowledgment_timeout_minutes
                incident.escalation_deadline = (
                    datetime.now() + timedelta(minutes=timeout)
                )

                # Start escalation chain
                chain = EscalationChain(policy)
                first_step = chain.start()
                self.escalation_chains[incident_id] = chain

                if first_step:
                    incident.current_level = first_step['level']

            self.incidents[incident_id] = incident

            # Record event
            self._record_event(
                incident_id=incident_id,
                event_type='created',
                details={
                    'alert_id': alert_id,
                    'severity': severity,
                    'category': category
                }
            )

            # Trigger callbacks
            self._trigger_callbacks(incident, 'created')

            return incident

    def acknowledge_incident(
        self,
        incident_id: str,
        user_id: str,
        message: Optional[str] = None
    ) -> bool:
        """
        Acknowledge incident.

        Args:
            incident_id: Incident ID
            user_id: User acknowledging
            message: Optional message

        Returns:
            True if acknowledged successfully
        """
        with self.lock:
            if incident_id not in self.incidents:
                return False

            incident = self.incidents[incident_id]

            if incident.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED]:
                return False

            # Update incident
            incident.status = AlertStatus.ACKNOWLEDGED
            incident.acknowledged_by = user_id
            incident.acknowledged_at = datetime.now()
            incident.assigned_to = user_id
            incident.updated_at = datetime.now()

            # Record response
            response = Response(
                response_id=self._generate_id(),
                incident_id=incident_id,
                user_id=user_id,
                action='acknowledged',
                message=message
            )
            if incident_id not in self.responses:
                self.responses[incident_id] = []
            self.responses[incident_id].append(response)

            # Record event
            self._record_event(
                incident_id=incident_id,
                event_type='acknowledged',
                user_id=user_id,
                details={'message': message}
            )

            # Trigger callbacks
            self._trigger_callbacks(incident, 'acknowledged')

            return True

    def resolve_incident(
        self,
        incident_id: str,
        user_id: str,
        message: Optional[str] = None
    ) -> bool:
        """
        Resolve incident.

        Args:
            incident_id: Incident ID
            user_id: User resolving
            message: Optional resolution message

        Returns:
            True if resolved successfully
        """
        with self.lock:
            if incident_id not in self.incidents:
                return False

            incident = self.incidents[incident_id]

            if incident.status == AlertStatus.CLOSED:
                return False

            # Update incident
            incident.status = AlertStatus.RESOLVED
            incident.resolved_by = user_id
            incident.resolved_at = datetime.now()
            incident.updated_at = datetime.now()

            if message:
                incident.notes.append(f"[{datetime.now().isoformat()}] {user_id}: {message}")

            # Record response
            response = Response(
                response_id=self._generate_id(),
                incident_id=incident_id,
                user_id=user_id,
                action='resolved',
                message=message
            )
            if incident_id not in self.responses:
                self.responses[incident_id] = []
            self.responses[incident_id].append(response)

            # Record event
            self._record_event(
                incident_id=incident_id,
                event_type='resolved',
                user_id=user_id,
                details={'message': message}
            )

            # Trigger callbacks
            self._trigger_callbacks(incident, 'resolved')

            return True

    def escalate_incident(
        self,
        incident_id: str,
        override_level: Optional[EscalationLevel] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Escalate incident to next level.

        Args:
            incident_id: Incident ID
            override_level: Optional override level
            user_id: Optional user initiating escalation

        Returns:
            True if escalated successfully
        """
        with self.lock:
            if incident_id not in self.incidents:
                return False

            incident = self.incidents[incident_id]
            from_level = incident.current_level

            if override_level:
                # Manual override
                to_level = override_level
                incident.current_level = to_level
            else:
                # Automatic escalation
                if incident_id in self.escalation_chains:
                    chain = self.escalation_chains[incident_id]
                    next_step = chain.get_next_step()

                    if not next_step:
                        # End of escalation chain
                        return False

                    to_level = next_step['level']
                    incident.current_level = to_level
                    incident.escalation_deadline = (
                        datetime.now() + timedelta(minutes=next_step['timeout_minutes'])
                    )
                else:
                    return False

            # Update incident
            incident.status = AlertStatus.ESCALATED
            incident.escalation_count += 1
            incident.last_escalation_time = datetime.now()
            incident.updated_at = datetime.now()

            # Record event
            self._record_event(
                incident_id=incident_id,
                event_type='escalated',
                from_level=from_level,
                to_level=to_level,
                user_id=user_id,
                details={'override': override_level is not None}
            )

            # Trigger callbacks
            self._trigger_callbacks(incident, 'escalated', {'to_level': to_level})

            return True

    def add_note(
        self,
        incident_id: str,
        user_id: str,
        note: str
    ) -> bool:
        """
        Add note to incident.

        Args:
            incident_id: Incident ID
            user_id: User adding note
            note: Note text

        Returns:
            True if added successfully
        """
        with self.lock:
            if incident_id not in self.incidents:
                return False

            incident = self.incidents[incident_id]
            timestamp = datetime.now().isoformat()
            incident.notes.append(f"[{timestamp}] {user_id}: {note}")
            incident.updated_at = datetime.now()

            # Record response
            response = Response(
                response_id=self._generate_id(),
                incident_id=incident_id,
                user_id=user_id,
                action='commented',
                message=note
            )
            if incident_id not in self.responses:
                self.responses[incident_id] = []
            self.responses[incident_id].append(response)

            return True

    def get_incident(self, incident_id: str) -> Optional[AlertIncident]:
        """
        Get incident by ID.

        Args:
            incident_id: Incident ID

        Returns:
            Incident or None
        """
        with self.lock:
            return self.incidents.get(incident_id)

    def get_active_incidents(
        self,
        severity: Optional[str] = None
    ) -> List[AlertIncident]:
        """
        Get active incidents.

        Args:
            severity: Optional severity filter

        Returns:
            List of active incidents
        """
        with self.lock:
            incidents = [
                i for i in self.incidents.values()
                if i.status not in [AlertStatus.RESOLVED, AlertStatus.CLOSED]
            ]

            if severity:
                incidents = [i for i in incidents if i.severity == severity]

            # Sort by creation time
            incidents.sort(key=lambda i: i.created_at, reverse=True)
            return incidents

    def get_incident_responses(self, incident_id: str) -> List[Response]:
        """
        Get all responses for incident.

        Args:
            incident_id: Incident ID

        Returns:
            List of responses
        """
        with self.lock:
            return self.responses.get(incident_id, []).copy()

    def get_audit_trail(
        self,
        incident_id: Optional[str] = None,
        limit: int = 100
    ) -> List[EscalationEvent]:
        """
        Get audit trail.

        Args:
            incident_id: Optional incident ID filter
            limit: Maximum events to return

        Returns:
            List of events
        """
        with self.lock:
            events = list(self.audit_trail)

            if incident_id:
                events = [e for e in events if e.incident_id == incident_id]

            events.sort(key=lambda e: e.timestamp, reverse=True)
            return events[:limit]

    def start_monitoring(self):
        """Start background monitoring for auto-escalation."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitoring_loop(self):
        """Background monitoring loop for auto-escalation."""
        import time

        while self.monitoring_active:
            try:
                self._check_escalation_timeouts()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Error in escalation monitoring: {e}")

    def _check_escalation_timeouts(self):
        """Check for incidents that need escalation."""
        now = datetime.now()

        with self.lock:
            for incident in list(self.incidents.values()):
                # Skip resolved/closed
                if incident.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED]:
                    continue

                # Check if escalation deadline passed
                if incident.escalation_deadline and now > incident.escalation_deadline:
                    # Get policy
                    if incident.incident_id in self.escalation_chains:
                        policy = self.escalation_chains[incident.incident_id].policy
                        if policy.auto_escalate:
                            self.escalate_incident(incident.incident_id)

    def _find_matching_policy(
        self,
        severity: str,
        category: str
    ) -> Optional[EscalationPolicy]:
        """Find matching escalation policy."""
        for policy in self.policies.values():
            if not policy.enabled:
                continue

            if policy.severity_levels and severity not in policy.severity_levels:
                continue

            if policy.categories and category not in policy.categories:
                continue

            return policy

        return None

    def _record_event(
        self,
        incident_id: str,
        event_type: str,
        from_level: Optional[EscalationLevel] = None,
        to_level: Optional[EscalationLevel] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """Record event in audit trail."""
        event = EscalationEvent(
            event_id=self._generate_id(),
            incident_id=incident_id,
            event_type=event_type,
            from_level=from_level,
            to_level=to_level,
            user_id=user_id,
            details=details or {}
        )
        self.audit_trail.append(event)

    def _trigger_callbacks(
        self,
        incident: AlertIncident,
        event_type: str,
        data: Optional[Dict] = None
    ):
        """Trigger escalation callbacks."""
        for callback in self.escalation_callbacks:
            try:
                callback(incident, event_type, data or {})
            except Exception as e:
                print(f"Error in escalation callback: {e}")

    def add_callback(self, callback: Callable):
        """
        Add escalation callback.

        Args:
            callback: Function(incident, event_type, data)
        """
        self.escalation_callbacks.append(callback)

    def _generate_id(self) -> str:
        """Generate unique ID."""
        import hashlib
        import time
        return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:16]

    def get_statistics(self) -> Dict[str, any]:
        """
        Get escalation statistics.

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            total_incidents = len(self.incidents)
            active_incidents = len([
                i for i in self.incidents.values()
                if i.status not in [AlertStatus.RESOLVED, AlertStatus.CLOSED]
            ])

            # Status distribution
            status_counts = {}
            for status in AlertStatus:
                status_counts[status.value] = sum(
                    1 for i in self.incidents.values()
                    if i.status == status
                )

            # Escalation level distribution
            level_counts = {}
            for level in EscalationLevel:
                level_counts[level.value] = sum(
                    1 for i in self.incidents.values()
                    if i.current_level == level and
                       i.status not in [AlertStatus.RESOLVED, AlertStatus.CLOSED]
                )

            # Average response time
            ack_times = [
                (i.acknowledged_at - i.created_at).total_seconds()
                for i in self.incidents.values()
                if i.acknowledged_at
            ]
            avg_ack_time = sum(ack_times) / len(ack_times) if ack_times else 0

            # Average resolution time
            resolution_times = [
                (i.resolved_at - i.created_at).total_seconds()
                for i in self.incidents.values()
                if i.resolved_at
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0

            return {
                'total_incidents': total_incidents,
                'active_incidents': active_incidents,
                'resolved_incidents': status_counts.get('resolved', 0),
                'status_distribution': status_counts,
                'level_distribution': level_counts,
                'total_policies': len(self.policies),
                'avg_acknowledgment_time_seconds': avg_ack_time,
                'avg_resolution_time_seconds': avg_resolution_time
            }

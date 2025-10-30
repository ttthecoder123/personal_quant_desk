"""
Dependency Monitor

Monitors external dependencies including APIs, databases,
message queues, caches, and third-party services.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
from enum import Enum


class DependencyType(Enum):
    """Types of dependencies."""
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class DependencyStatus:
    """Status of a dependency."""
    name: str
    dependency_type: DependencyType
    is_available: bool
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class DependencyMonitor:
    """
    Comprehensive dependency monitoring.

    Features:
    - API health monitoring
    - Database connectivity
    - Cache availability
    - Message queue health
    - File system access
    - Third-party service status
    """

    def __init__(self, check_interval: int = 30):
        """
        Initialize dependency monitor.

        Args:
            check_interval: Seconds between checks
        """
        self.check_interval = check_interval
        self.dependencies: Dict[str, Dict[str, Any]] = {}
        self.status_history: Dict[str, deque] = {}
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self):
        """Start dependency monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop dependency monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def register_dependency(self, name: str, dependency_type: DependencyType,
                          check_func: Callable[[], tuple], critical: bool = True):
        """
        Register a dependency for monitoring.

        Args:
            name: Dependency name
            dependency_type: Type of dependency
            check_func: Function that returns (is_available, response_time_ms, error_msg)
            critical: Whether this is a critical dependency
        """
        with self.lock:
            self.dependencies[name] = {
                'type': dependency_type,
                'check_func': check_func,
                'critical': critical
            }
            self.status_history[name] = deque(maxlen=1000)

    def unregister_dependency(self, name: str):
        """
        Unregister a dependency.

        Args:
            name: Dependency name
        """
        with self.lock:
            if name in self.dependencies:
                del self.dependencies[name]
            if name in self.status_history:
                del self.status_history[name]

    def check_dependency(self, name: str) -> Optional[DependencyStatus]:
        """
        Check a specific dependency.

        Args:
            name: Dependency name

        Returns:
            DependencyStatus or None
        """
        with self.lock:
            if name not in self.dependencies:
                return None

            dep_info = self.dependencies[name]

        # Run check outside lock
        start_time = time.time()
        try:
            is_available, response_time_ms, error_msg = dep_info['check_func']()
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            is_available = False
            error_msg = str(e)

        status = DependencyStatus(
            name=name,
            dependency_type=dep_info['type'],
            is_available=is_available,
            response_time_ms=response_time_ms,
            last_check=datetime.now(),
            error_message=error_msg
        )

        with self.lock:
            self.status_history[name].append(status)

        return status

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_dependencies()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in dependency monitor loop: {e}")

    def _check_all_dependencies(self):
        """Check all registered dependencies."""
        with self.lock:
            dep_names = list(self.dependencies.keys())

        for name in dep_names:
            try:
                self.check_dependency(name)
            except Exception as e:
                print(f"Error checking dependency {name}: {e}")

    def get_dependency_status(self, name: str) -> Optional[DependencyStatus]:
        """
        Get current status of a dependency.

        Args:
            name: Dependency name

        Returns:
            DependencyStatus or None
        """
        with self.lock:
            if name not in self.status_history or not self.status_history[name]:
                return None
            return self.status_history[name][-1]

    def get_dependency_health(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get health metrics for a dependency.

        Args:
            name: Dependency name
            window_minutes: Time window

        Returns:
            Dictionary with health metrics
        """
        with self.lock:
            if name not in self.status_history:
                return {}

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent = [s for s in self.status_history[name] if s.last_check >= cutoff]

            if not recent:
                return {}

            total = len(recent)
            available = sum(1 for s in recent if s.is_available)
            uptime_percent = (available / total) * 100

            response_times = [s.response_time_ms for s in recent if s.is_available]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            return {
                'name': name,
                'uptime_percent': uptime_percent,
                'total_checks': total,
                'successful_checks': available,
                'failed_checks': total - available,
                'avg_response_time_ms': avg_response_time,
                'last_status': recent[-1].is_available,
                'last_error': recent[-1].error_message if not recent[-1].is_available else None
            }

    def get_critical_dependencies(self) -> List[str]:
        """
        Get list of critical dependencies.

        Returns:
            List of dependency names
        """
        with self.lock:
            return [name for name, info in self.dependencies.items() if info['critical']]

    def get_unavailable_dependencies(self) -> List[DependencyStatus]:
        """
        Get list of currently unavailable dependencies.

        Returns:
            List of DependencyStatus objects
        """
        with self.lock:
            unavailable = []
            for name in self.dependencies.keys():
                if self.status_history[name]:
                    latest = self.status_history[name][-1]
                    if not latest.is_available:
                        unavailable.append(latest)
            return unavailable

    def get_critical_failures(self) -> List[DependencyStatus]:
        """
        Get critical dependencies that are failing.

        Returns:
            List of DependencyStatus objects
        """
        critical_names = self.get_critical_dependencies()
        unavailable = self.get_unavailable_dependencies()
        return [s for s in unavailable if s.name in critical_names]

    def is_all_critical_available(self) -> bool:
        """
        Check if all critical dependencies are available.

        Returns:
            True if all critical dependencies are up
        """
        return len(self.get_critical_failures()) == 0

    def get_summary(self) -> Dict[str, Any]:
        """
        Get dependency monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total = len(self.dependencies)
            critical = len(self.get_critical_dependencies())
            unavailable = len(self.get_unavailable_dependencies())
            critical_failures = len(self.get_critical_failures())

            return {
                'total_dependencies': total,
                'critical_dependencies': critical,
                'unavailable_dependencies': unavailable,
                'critical_failures': critical_failures,
                'all_critical_available': critical_failures == 0,
                'timestamp': datetime.now().isoformat()
            }

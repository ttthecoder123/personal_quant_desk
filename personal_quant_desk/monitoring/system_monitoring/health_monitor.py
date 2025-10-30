"""
System Health Monitor

Monitors system health including heartbeats, service availability,
database connectivity, API health, queues, threads, and memory leaks.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import socket
import requests


@dataclass
class HealthStatus:
    """Health status for a component."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartbeatStatus:
    """Heartbeat status for a service."""
    service: str
    last_heartbeat: datetime
    expected_interval: timedelta
    missed_count: int = 0

    def is_alive(self) -> bool:
        """Check if service is still alive."""
        elapsed = datetime.now() - self.last_heartbeat
        return elapsed < (self.expected_interval * 2)


class HealthMonitor:
    """
    Comprehensive system health monitoring.

    Features:
    - Heartbeat monitoring for all components
    - Service availability checks
    - Database connectivity monitoring
    - API endpoint health checks
    - Queue depth monitoring
    - Thread pool monitoring
    - Memory leak detection
    - File system monitoring
    - Process zombie detection
    """

    def __init__(self, check_interval: int = 10):
        """
        Initialize health monitor.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self.heartbeats: Dict[str, HeartbeatStatus] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.memory_baseline: Dict[int, float] = {}
        self.memory_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Health check configurations
        self.endpoints: Dict[str, str] = {}
        self.database_connections: Dict[str, Any] = {}
        self.queue_monitors: Dict[str, Any] = {}

    def start(self):
        """Start health monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def register_heartbeat(self, service: str, interval_seconds: int = 60):
        """
        Register a service for heartbeat monitoring.

        Args:
            service: Service name
            interval_seconds: Expected heartbeat interval
        """
        with self.lock:
            self.heartbeats[service] = HeartbeatStatus(
                service=service,
                last_heartbeat=datetime.now(),
                expected_interval=timedelta(seconds=interval_seconds)
            )

    def record_heartbeat(self, service: str):
        """
        Record a heartbeat for a service.

        Args:
            service: Service name
        """
        with self.lock:
            if service in self.heartbeats:
                heartbeat = self.heartbeats[service]
                heartbeat.last_heartbeat = datetime.now()
                heartbeat.missed_count = 0

    def register_endpoint(self, name: str, url: str):
        """
        Register an API endpoint for health checks.

        Args:
            name: Endpoint name
            url: Endpoint URL
        """
        self.endpoints[name] = url

    def register_database(self, name: str, connection: Any):
        """
        Register a database connection for monitoring.

        Args:
            name: Database name
            connection: Database connection object
        """
        self.database_connections[name] = connection

    def register_queue(self, name: str, queue: Any):
        """
        Register a queue for depth monitoring.

        Args:
            name: Queue name
            queue: Queue object
        """
        self.queue_monitors[name] = queue

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_health()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in health monitor loop: {e}")

    def _check_all_health(self):
        """Perform all health checks."""
        # Check heartbeats
        self._check_heartbeats()

        # Check endpoints
        self._check_endpoints()

        # Check databases
        self._check_databases()

        # Check queues
        self._check_queues()

        # Check system resources
        self._check_system_resources()

        # Check for memory leaks
        self._check_memory_leaks()

        # Check for zombie processes
        self._check_zombie_processes()

    def _check_heartbeats(self):
        """Check all registered heartbeats."""
        with self.lock:
            now = datetime.now()
            for service, heartbeat in self.heartbeats.items():
                is_alive = heartbeat.is_alive()
                status = 'healthy' if is_alive else 'unhealthy'

                if not is_alive:
                    heartbeat.missed_count += 1

                self.health_status[f"heartbeat_{service}"] = HealthStatus(
                    component=f"heartbeat_{service}",
                    status=status,
                    last_check=now,
                    response_time_ms=0,
                    error_message=f"Missed {heartbeat.missed_count} heartbeats" if not is_alive else None
                )

    def _check_endpoints(self):
        """Check all registered API endpoints."""
        for name, url in self.endpoints.items():
            start_time = time.time()
            try:
                response = requests.get(url, timeout=5)
                response_time = (time.time() - start_time) * 1000

                status = 'healthy' if response.status_code == 200 else 'degraded'
                error_msg = None if response.status_code == 200 else f"Status code: {response.status_code}"

                self.health_status[f"endpoint_{name}"] = HealthStatus(
                    component=f"endpoint_{name}",
                    status=status,
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    error_message=error_msg,
                    metadata={'status_code': response.status_code}
                )
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.health_status[f"endpoint_{name}"] = HealthStatus(
                    component=f"endpoint_{name}",
                    status='unhealthy',
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    error_message=str(e)
                )

    def _check_databases(self):
        """Check all registered database connections."""
        for name, connection in self.database_connections.items():
            start_time = time.time()
            try:
                # Try to execute a simple query
                if hasattr(connection, 'execute'):
                    connection.execute('SELECT 1')
                elif hasattr(connection, 'ping'):
                    connection.ping()

                response_time = (time.time() - start_time) * 1000

                self.health_status[f"database_{name}"] = HealthStatus(
                    component=f"database_{name}",
                    status='healthy',
                    last_check=datetime.now(),
                    response_time_ms=response_time
                )
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.health_status[f"database_{name}"] = HealthStatus(
                    component=f"database_{name}",
                    status='unhealthy',
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    error_message=str(e)
                )

    def _check_queues(self):
        """Check all registered queue depths."""
        for name, queue in self.queue_monitors.items():
            try:
                depth = queue.qsize() if hasattr(queue, 'qsize') else 0

                # Consider queue unhealthy if it's very full
                status = 'healthy'
                if depth > 10000:
                    status = 'unhealthy'
                elif depth > 5000:
                    status = 'degraded'

                self.health_status[f"queue_{name}"] = HealthStatus(
                    component=f"queue_{name}",
                    status=status,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    metadata={'queue_depth': depth}
                )
            except Exception as e:
                self.health_status[f"queue_{name}"] = HealthStatus(
                    component=f"queue_{name}",
                    status='unhealthy',
                    last_check=datetime.now(),
                    response_time_ms=0,
                    error_message=str(e)
                )

    def _check_system_resources(self):
        """Check system resource availability."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = 'healthy'
            if cpu_percent > 90:
                cpu_status = 'unhealthy'
            elif cpu_percent > 80:
                cpu_status = 'degraded'

            self.health_status['system_cpu'] = HealthStatus(
                component='system_cpu',
                status=cpu_status,
                last_check=datetime.now(),
                response_time_ms=0,
                metadata={'cpu_percent': cpu_percent}
            )

            # Memory usage
            memory = psutil.virtual_memory()
            mem_status = 'healthy'
            if memory.percent > 90:
                mem_status = 'unhealthy'
            elif memory.percent > 80:
                mem_status = 'degraded'

            self.health_status['system_memory'] = HealthStatus(
                component='system_memory',
                status=mem_status,
                last_check=datetime.now(),
                response_time_ms=0,
                metadata={
                    'memory_percent': memory.percent,
                    'available_gb': memory.available / (1024**3)
                }
            )

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_status = 'healthy'
            if disk.percent > 90:
                disk_status = 'unhealthy'
            elif disk.percent > 80:
                disk_status = 'degraded'

            self.health_status['system_disk'] = HealthStatus(
                component='system_disk',
                status=disk_status,
                last_check=datetime.now(),
                response_time_ms=0,
                metadata={
                    'disk_percent': disk.percent,
                    'free_gb': disk.free / (1024**3)
                }
            )
        except Exception as e:
            print(f"Error checking system resources: {e}")

    def _check_memory_leaks(self):
        """Check for memory leaks in processes."""
        try:
            current_process = psutil.Process()
            pid = current_process.pid
            memory_mb = current_process.memory_info().rss / (1024**2)

            # Record memory history
            self.memory_history[pid].append(memory_mb)

            # Set baseline if not set
            if pid not in self.memory_baseline:
                self.memory_baseline[pid] = memory_mb

            # Check for memory leak (continuous growth)
            if len(self.memory_history[pid]) >= 10:
                recent_avg = sum(list(self.memory_history[pid])[-10:]) / 10
                growth = recent_avg - self.memory_baseline[pid]
                growth_rate = growth / self.memory_baseline[pid] * 100

                status = 'healthy'
                error_msg = None
                if growth_rate > 50:  # 50% growth
                    status = 'degraded'
                    error_msg = f"Memory growth: {growth_rate:.1f}%"

                self.health_status['memory_leak_detection'] = HealthStatus(
                    component='memory_leak_detection',
                    status=status,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    error_message=error_msg,
                    metadata={
                        'current_memory_mb': memory_mb,
                        'baseline_memory_mb': self.memory_baseline[pid],
                        'growth_rate_percent': growth_rate
                    }
                )
        except Exception as e:
            print(f"Error checking memory leaks: {e}")

    def _check_zombie_processes(self):
        """Check for zombie processes."""
        try:
            zombie_count = 0
            for proc in psutil.process_iter(['pid', 'status']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            status = 'healthy' if zombie_count == 0 else 'degraded'
            error_msg = f"Found {zombie_count} zombie processes" if zombie_count > 0 else None

            self.health_status['zombie_processes'] = HealthStatus(
                component='zombie_processes',
                status=status,
                last_check=datetime.now(),
                response_time_ms=0,
                error_message=error_msg,
                metadata={'zombie_count': zombie_count}
            )
        except Exception as e:
            print(f"Error checking zombie processes: {e}")

    def get_health_status(self, component: Optional[str] = None) -> Dict[str, HealthStatus]:
        """
        Get health status for components.

        Args:
            component: Optional specific component name

        Returns:
            Dictionary of health statuses
        """
        with self.lock:
            if component:
                return {component: self.health_status.get(component)}
            return self.health_status.copy()

    def is_healthy(self) -> bool:
        """
        Check if all components are healthy.

        Returns:
            True if all components are healthy
        """
        with self.lock:
            return all(
                status.status == 'healthy'
                for status in self.health_status.values()
            )

    def get_unhealthy_components(self) -> List[HealthStatus]:
        """
        Get list of unhealthy components.

        Returns:
            List of unhealthy component statuses
        """
        with self.lock:
            return [
                status for status in self.health_status.values()
                if status.status in ['unhealthy', 'degraded']
            ]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get health summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total = len(self.health_status)
            healthy = sum(1 for s in self.health_status.values() if s.status == 'healthy')
            degraded = sum(1 for s in self.health_status.values() if s.status == 'degraded')
            unhealthy = sum(1 for s in self.health_status.values() if s.status == 'unhealthy')

            return {
                'total_components': total,
                'healthy': healthy,
                'degraded': degraded,
                'unhealthy': unhealthy,
                'overall_status': 'healthy' if unhealthy == 0 and degraded == 0 else 'degraded' if unhealthy == 0 else 'unhealthy',
                'last_check': datetime.now().isoformat()
            }

"""
System Risk Monitoring

Monitors system health and infrastructure:
- System latency (execution, data feed)
- Memory usage and thresholds
- CPU utilization
- System anomaly detection
- Connectivity issue tracking
- Backup system status monitoring
- Comprehensive system health reports
"""

import numpy as np
import pandas as pd
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from loguru import logger


class SystemStatus(Enum):
    """System status types"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ComponentType(Enum):
    """System component types"""
    EXECUTION_ENGINE = "execution_engine"
    DATA_FEED = "data_feed"
    DATABASE = "database"
    RISK_ENGINE = "risk_engine"
    STRATEGY_ENGINE = "strategy_engine"
    API_GATEWAY = "api_gateway"


class ConnectionStatus(Enum):
    """Connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNSTABLE = "unstable"
    RECONNECTING = "reconnecting"


@dataclass
class LatencyMeasurement:
    """System latency measurement"""
    timestamp: datetime
    component: ComponentType
    operation: str
    latency_ms: float
    is_slow: bool
    percentile_rank: Optional[float] = None


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    timestamp: datetime
    total_mb: float
    available_mb: float
    used_mb: float
    usage_pct: float
    is_high: bool
    swap_used_mb: float
    swap_pct: float


@dataclass
class CPUMetrics:
    """CPU usage metrics"""
    timestamp: datetime
    cpu_pct: float
    cpu_count: int
    per_cpu_pct: List[float]
    load_avg_1min: float
    load_avg_5min: float
    load_avg_15min: float
    is_high: bool


@dataclass
class SystemAnomaly:
    """System anomaly detection"""
    timestamp: datetime
    anomaly_type: str
    component: Optional[ComponentType]
    description: str
    severity: str  # 'low', 'medium', 'high'
    metric_value: float
    expected_range: Tuple[float, float]


@dataclass
class ConnectivityStatus:
    """Connectivity status tracking"""
    timestamp: datetime
    endpoint: str
    component: ComponentType
    status: ConnectionStatus
    latency_ms: Optional[float]
    last_successful_connection: datetime
    consecutive_failures: int
    uptime_pct: float


@dataclass
class BackupSystemStatus:
    """Backup system status"""
    timestamp: datetime
    system_name: str
    is_active: bool
    is_ready: bool
    last_health_check: datetime
    health_check_passed: bool
    failover_capable: bool
    sync_lag_seconds: Optional[float]


@dataclass
class SystemAlert:
    """System risk alert"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    alert_type: str
    component: Optional[ComponentType]
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    requires_failover: bool = False


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    timestamp: datetime
    overall_status: SystemStatus
    latency_summary: Dict[str, float]
    memory_metrics: MemoryMetrics
    cpu_metrics: CPUMetrics
    connectivity_status: Dict[str, ConnectivityStatus]
    backup_systems: Dict[str, BackupSystemStatus]
    recent_anomalies: List[SystemAnomaly]
    alerts: List[SystemAlert]
    uptime_hours: float
    recommendations: List[str]


@dataclass
class PerformanceBaseline:
    """System performance baseline"""
    component: ComponentType
    operation: str
    baseline_latency_ms: float
    baseline_cpu_pct: float
    baseline_memory_mb: float
    established_date: datetime
    sample_size: int


@dataclass
class SystemRiskConfig:
    """Configuration for system risk monitoring"""
    # Latency thresholds (milliseconds)
    max_execution_latency_ms: float = 100.0
    critical_execution_latency_ms: float = 500.0
    max_data_feed_latency_ms: float = 1000.0
    critical_data_feed_latency_ms: float = 5000.0

    # Memory thresholds
    memory_warning_pct: float = 75.0
    memory_critical_pct: float = 90.0
    swap_warning_pct: float = 50.0

    # CPU thresholds
    cpu_warning_pct: float = 75.0
    cpu_critical_pct: float = 90.0
    load_warning_multiplier: float = 1.5  # x CPU count

    # Connectivity thresholds
    max_consecutive_failures: int = 3
    min_uptime_pct: float = 99.0
    connection_timeout_ms: float = 5000.0

    # Monitoring settings
    latency_window_size: int = 100
    anomaly_detection_std: float = 3.0
    health_check_interval_minutes: int = 5


class SystemRisk:
    """
    Monitor and manage system risk

    Features:
    - Real-time latency monitoring
    - Resource usage tracking
    - Anomaly detection
    - Connectivity monitoring
    - Backup system management
    - Health reporting
    """

    def __init__(self, config: Optional[SystemRiskConfig] = None):
        """
        Initialize system risk monitor

        Args:
            config: System risk configuration (optional)
        """
        self.config = config if config is not None else SystemRiskConfig()

        # Latency tracking
        self.latency_history: Dict[str, deque] = {}

        # Resource metrics history
        self.memory_history: deque = deque(maxlen=1000)
        self.cpu_history: deque = deque(maxlen=1000)

        # Connectivity tracking
        self.connectivity_stats: Dict[str, Dict] = {}

        # Backup systems
        self.backup_systems: Dict[str, BackupSystemStatus] = {}

        # Anomaly tracking
        self.anomalies: List[SystemAnomaly] = []

        # Alert tracking
        self.alerts: List[SystemAlert] = []

        # Performance baselines
        self.baselines: Dict[str, PerformanceBaseline] = {}

        # System start time
        self.start_time = datetime.now()

        logger.info("SystemRisk monitor initialized")

    def record_latency(
        self,
        timestamp: datetime,
        component: ComponentType,
        operation: str,
        latency_ms: float
    ) -> LatencyMeasurement:
        """
        Record system latency measurement

        Args:
            timestamp: Measurement timestamp
            component: System component
            operation: Operation name
            latency_ms: Latency in milliseconds

        Returns:
            LatencyMeasurement record
        """
        key = f"{component.value}:{operation}"

        # Initialize history if needed
        if key not in self.latency_history:
            self.latency_history[key] = deque(
                maxlen=self.config.latency_window_size
            )

        # Determine if slow based on component type
        if component == ComponentType.EXECUTION_ENGINE:
            is_slow = latency_ms > self.config.max_execution_latency_ms
        elif component == ComponentType.DATA_FEED:
            is_slow = latency_ms > self.config.max_data_feed_latency_ms
        else:
            is_slow = latency_ms > 1000.0  # Default 1 second

        # Calculate percentile rank if we have history
        percentile_rank = None
        if self.latency_history[key]:
            history_values = list(self.latency_history[key])
            percentile_rank = (
                sum(1 for v in history_values if v < latency_ms) / len(history_values)
            ) * 100

        measurement = LatencyMeasurement(
            timestamp=timestamp,
            component=component,
            operation=operation,
            latency_ms=latency_ms,
            is_slow=is_slow,
            percentile_rank=percentile_rank
        )

        # Store in history
        self.latency_history[key].append(latency_ms)

        # Check for alerts
        self._check_latency_alerts(measurement)

        # Check for anomalies
        self._check_latency_anomaly(key, latency_ms)

        return measurement

    def measure_memory_usage(self) -> MemoryMetrics:
        """
        Measure current memory usage

        Returns:
            MemoryMetrics
        """
        # Get memory info
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        total_mb = memory.total / (1024 ** 2)
        available_mb = memory.available / (1024 ** 2)
        used_mb = memory.used / (1024 ** 2)
        usage_pct = memory.percent

        swap_used_mb = swap.used / (1024 ** 2)
        swap_pct = swap.percent

        is_high = usage_pct > self.config.memory_warning_pct

        metrics = MemoryMetrics(
            timestamp=datetime.now(),
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=used_mb,
            usage_pct=usage_pct,
            is_high=is_high,
            swap_used_mb=swap_used_mb,
            swap_pct=swap_pct
        )

        self.memory_history.append(metrics)

        # Check for alerts
        self._check_memory_alerts(metrics)

        return metrics

    def measure_cpu_usage(self) -> CPUMetrics:
        """
        Measure current CPU usage

        Returns:
            CPUMetrics
        """
        # Get CPU info
        cpu_pct = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        per_cpu_pct = psutil.cpu_percent(interval=0.1, percpu=True)

        # Get load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
            load_avg_1min, load_avg_5min, load_avg_15min = load_avg
        except (AttributeError, OSError):
            # Windows doesn't have getloadavg
            load_avg_1min = cpu_pct / 100 * cpu_count
            load_avg_5min = load_avg_1min
            load_avg_15min = load_avg_1min

        is_high = cpu_pct > self.config.cpu_warning_pct

        metrics = CPUMetrics(
            timestamp=datetime.now(),
            cpu_pct=cpu_pct,
            cpu_count=cpu_count,
            per_cpu_pct=per_cpu_pct,
            load_avg_1min=load_avg_1min,
            load_avg_5min=load_avg_5min,
            load_avg_15min=load_avg_15min,
            is_high=is_high
        )

        self.cpu_history.append(metrics)

        # Check for alerts
        self._check_cpu_alerts(metrics)

        return metrics

    def record_connection_attempt(
        self,
        timestamp: datetime,
        endpoint: str,
        component: ComponentType,
        success: bool,
        latency_ms: Optional[float] = None
    ):
        """
        Record connection attempt

        Args:
            timestamp: Attempt timestamp
            endpoint: Connection endpoint
            component: Component type
            success: Whether connection succeeded
            latency_ms: Connection latency (optional)
        """
        key = f"{component.value}:{endpoint}"

        # Initialize tracking if needed
        if key not in self.connectivity_stats:
            self.connectivity_stats[key] = {
                'endpoint': endpoint,
                'component': component,
                'total_attempts': 0,
                'successful_attempts': 0,
                'last_success': None,
                'consecutive_failures': 0,
                'latencies': deque(maxlen=100)
            }

        stats = self.connectivity_stats[key]
        stats['total_attempts'] += 1

        if success:
            stats['successful_attempts'] += 1
            stats['last_success'] = timestamp
            stats['consecutive_failures'] = 0
            if latency_ms:
                stats['latencies'].append(latency_ms)
        else:
            stats['consecutive_failures'] += 1

            # Check for critical failures
            if stats['consecutive_failures'] >= self.config.max_consecutive_failures:
                self.alerts.append(SystemAlert(
                    timestamp=timestamp,
                    severity='critical',
                    alert_type='connectivity_failure',
                    component=component,
                    message=f"Connection failures to {endpoint}: {stats['consecutive_failures']} consecutive",
                    value=float(stats['consecutive_failures']),
                    threshold=float(self.config.max_consecutive_failures),
                    requires_failover=True
                ))

    def get_connectivity_status(
        self,
        endpoint: str,
        component: ComponentType
    ) -> Optional[ConnectivityStatus]:
        """
        Get connectivity status for endpoint

        Args:
            endpoint: Connection endpoint
            component: Component type

        Returns:
            ConnectivityStatus or None
        """
        key = f"{component.value}:{endpoint}"

        if key not in self.connectivity_stats:
            return None

        stats = self.connectivity_stats[key]

        # Determine status
        if stats['consecutive_failures'] >= self.config.max_consecutive_failures:
            status = ConnectionStatus.DISCONNECTED
        elif stats['consecutive_failures'] > 0:
            status = ConnectionStatus.UNSTABLE
        else:
            status = ConnectionStatus.CONNECTED

        # Calculate uptime
        uptime_pct = (
            (stats['successful_attempts'] / stats['total_attempts'] * 100)
            if stats['total_attempts'] > 0 else 0.0
        )

        # Average latency
        latencies = list(stats['latencies'])
        avg_latency = np.mean(latencies) if latencies else None

        return ConnectivityStatus(
            timestamp=datetime.now(),
            endpoint=endpoint,
            component=component,
            status=status,
            latency_ms=avg_latency,
            last_successful_connection=stats['last_success'],
            consecutive_failures=stats['consecutive_failures'],
            uptime_pct=uptime_pct
        )

    def register_backup_system(
        self,
        system_name: str,
        component: ComponentType,
        is_active: bool = False
    ):
        """
        Register a backup system

        Args:
            system_name: Name of backup system
            component: Component type
            is_active: Whether currently active
        """
        self.backup_systems[system_name] = BackupSystemStatus(
            timestamp=datetime.now(),
            system_name=system_name,
            is_active=is_active,
            is_ready=True,
            last_health_check=datetime.now(),
            health_check_passed=True,
            failover_capable=True,
            sync_lag_seconds=0.0
        )

        logger.info(f"Registered backup system: {system_name}")

    def update_backup_system_health(
        self,
        system_name: str,
        health_check_passed: bool,
        is_ready: bool = True,
        sync_lag_seconds: Optional[float] = None
    ):
        """
        Update backup system health status

        Args:
            system_name: Name of backup system
            health_check_passed: Whether health check passed
            is_ready: Whether system is ready for failover
            sync_lag_seconds: Replication lag (optional)
        """
        if system_name not in self.backup_systems:
            logger.warning(f"Backup system {system_name} not registered")
            return

        backup = self.backup_systems[system_name]
        backup.last_health_check = datetime.now()
        backup.health_check_passed = health_check_passed
        backup.is_ready = is_ready
        if sync_lag_seconds is not None:
            backup.sync_lag_seconds = sync_lag_seconds

        # Alert if backup not ready
        if not health_check_passed or not is_ready:
            self.alerts.append(SystemAlert(
                timestamp=datetime.now(),
                severity='warning',
                alert_type='backup_system_issue',
                component=None,
                message=f"Backup system {system_name} health issue",
                requires_failover=False
            ))

    def activate_backup_system(self, system_name: str) -> bool:
        """
        Activate backup system (failover)

        Args:
            system_name: Name of backup system

        Returns:
            True if activated successfully
        """
        if system_name not in self.backup_systems:
            logger.error(f"Backup system {system_name} not found")
            return False

        backup = self.backup_systems[system_name]

        if not backup.is_ready or not backup.failover_capable:
            logger.error(f"Backup system {system_name} not ready for failover")
            return False

        backup.is_active = True
        backup.timestamp = datetime.now()

        logger.critical(f"FAILOVER: Activated backup system {system_name}")

        self.alerts.append(SystemAlert(
            timestamp=datetime.now(),
            severity='critical',
            alert_type='failover_activated',
            component=None,
            message=f"Failover to backup system: {system_name}",
            requires_failover=False
        ))

        return True

    def _check_latency_alerts(self, measurement: LatencyMeasurement):
        """Check latency for alerts"""
        component = measurement.component
        latency = measurement.latency_ms

        # Determine thresholds based on component
        if component == ComponentType.EXECUTION_ENGINE:
            warning_threshold = self.config.max_execution_latency_ms
            critical_threshold = self.config.critical_execution_latency_ms
        elif component == ComponentType.DATA_FEED:
            warning_threshold = self.config.max_data_feed_latency_ms
            critical_threshold = self.config.critical_data_feed_latency_ms
        else:
            warning_threshold = 1000.0
            critical_threshold = 5000.0

        # Generate alerts
        if latency > critical_threshold:
            self.alerts.append(SystemAlert(
                timestamp=measurement.timestamp,
                severity='critical',
                alert_type='critical_latency',
                component=component,
                message=f"{component.value} critical latency: {latency:.1f}ms",
                value=latency,
                threshold=critical_threshold
            ))
        elif latency > warning_threshold:
            self.alerts.append(SystemAlert(
                timestamp=measurement.timestamp,
                severity='warning',
                alert_type='high_latency',
                component=component,
                message=f"{component.value} high latency: {latency:.1f}ms",
                value=latency,
                threshold=warning_threshold
            ))

    def _check_memory_alerts(self, metrics: MemoryMetrics):
        """Check memory usage for alerts"""
        if metrics.usage_pct > self.config.memory_critical_pct:
            self.alerts.append(SystemAlert(
                timestamp=metrics.timestamp,
                severity='critical',
                alert_type='critical_memory',
                component=None,
                message=f"Critical memory usage: {metrics.usage_pct:.1f}%",
                value=metrics.usage_pct,
                threshold=self.config.memory_critical_pct
            ))
        elif metrics.usage_pct > self.config.memory_warning_pct:
            self.alerts.append(SystemAlert(
                timestamp=metrics.timestamp,
                severity='warning',
                alert_type='high_memory',
                component=None,
                message=f"High memory usage: {metrics.usage_pct:.1f}%",
                value=metrics.usage_pct,
                threshold=self.config.memory_warning_pct
            ))

        # Check swap
        if metrics.swap_pct > self.config.swap_warning_pct:
            self.alerts.append(SystemAlert(
                timestamp=metrics.timestamp,
                severity='warning',
                alert_type='high_swap',
                component=None,
                message=f"High swap usage: {metrics.swap_pct:.1f}%",
                value=metrics.swap_pct,
                threshold=self.config.swap_warning_pct
            ))

    def _check_cpu_alerts(self, metrics: CPUMetrics):
        """Check CPU usage for alerts"""
        if metrics.cpu_pct > self.config.cpu_critical_pct:
            self.alerts.append(SystemAlert(
                timestamp=metrics.timestamp,
                severity='critical',
                alert_type='critical_cpu',
                component=None,
                message=f"Critical CPU usage: {metrics.cpu_pct:.1f}%",
                value=metrics.cpu_pct,
                threshold=self.config.cpu_critical_pct
            ))
        elif metrics.cpu_pct > self.config.cpu_warning_pct:
            self.alerts.append(SystemAlert(
                timestamp=metrics.timestamp,
                severity='warning',
                alert_type='high_cpu',
                component=None,
                message=f"High CPU usage: {metrics.cpu_pct:.1f}%",
                value=metrics.cpu_pct,
                threshold=self.config.cpu_warning_pct
            ))

        # Check load average
        load_threshold = metrics.cpu_count * self.config.load_warning_multiplier
        if metrics.load_avg_1min > load_threshold:
            self.alerts.append(SystemAlert(
                timestamp=metrics.timestamp,
                severity='warning',
                alert_type='high_load',
                component=None,
                message=f"High load average: {metrics.load_avg_1min:.2f}",
                value=metrics.load_avg_1min,
                threshold=load_threshold
            ))

    def _check_latency_anomaly(self, key: str, latency_ms: float):
        """Check for latency anomalies"""
        if key not in self.latency_history or len(self.latency_history[key]) < 30:
            return

        history = list(self.latency_history[key])
        mean = np.mean(history[:-1])  # Exclude current
        std = np.std(history[:-1])

        if std == 0:
            return

        z_score = abs((latency_ms - mean) / std)

        if z_score > self.config.anomaly_detection_std:
            component_str, operation = key.split(':', 1)

            anomaly = SystemAnomaly(
                timestamp=datetime.now(),
                anomaly_type='latency_spike',
                component=ComponentType(component_str),
                description=f"Latency anomaly in {operation}: {latency_ms:.1f}ms (z={z_score:.2f})",
                severity='high' if z_score > 5 else 'medium',
                metric_value=latency_ms,
                expected_range=(mean - 3*std, mean + 3*std)
            )

            self.anomalies.append(anomaly)

    def generate_system_health_report(self) -> SystemHealthReport:
        """
        Generate comprehensive system health report

        Returns:
            SystemHealthReport with all metrics
        """
        # Latency summary
        latency_summary = {}
        for key, history in self.latency_history.items():
            if history:
                latency_summary[key] = {
                    'avg_ms': np.mean(list(history)),
                    'p95_ms': np.percentile(list(history), 95),
                    'max_ms': np.max(list(history))
                }

        # Current memory and CPU
        memory_metrics = self.measure_memory_usage()
        cpu_metrics = self.measure_cpu_usage()

        # Connectivity status
        connectivity_status = {}
        for key, stats in self.connectivity_stats.items():
            component_str, endpoint = key.split(':', 1)
            component = ComponentType(component_str)
            status = self.get_connectivity_status(endpoint, component)
            if status:
                connectivity_status[key] = status

        # Recent anomalies
        recent_anomalies = self.anomalies[-20:]

        # Recent alerts
        cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        # Calculate uptime
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        # Determine overall status
        overall_status = self._determine_overall_status(
            memory_metrics, cpu_metrics, connectivity_status, recent_alerts
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            memory_metrics, cpu_metrics, connectivity_status, recent_anomalies
        )

        return SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=overall_status,
            latency_summary=latency_summary,
            memory_metrics=memory_metrics,
            cpu_metrics=cpu_metrics,
            connectivity_status=connectivity_status,
            backup_systems=self.backup_systems.copy(),
            recent_anomalies=recent_anomalies,
            alerts=recent_alerts,
            uptime_hours=uptime_hours,
            recommendations=recommendations
        )

    def _determine_overall_status(
        self,
        memory: MemoryMetrics,
        cpu: CPUMetrics,
        connectivity: Dict,
        alerts: List[SystemAlert]
    ) -> SystemStatus:
        """Determine overall system status"""
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        if critical_alerts:
            return SystemStatus.CRITICAL

        # Check resources
        if (memory.usage_pct > self.config.memory_critical_pct or
            cpu.cpu_pct > self.config.cpu_critical_pct):
            return SystemStatus.CRITICAL

        # Check for degradation
        warning_alerts = [a for a in alerts if a.severity == 'warning']
        if (len(warning_alerts) > 5 or
            memory.usage_pct > self.config.memory_warning_pct or
            cpu.cpu_pct > self.config.cpu_warning_pct):
            return SystemStatus.DEGRADED

        return SystemStatus.HEALTHY

    def _generate_recommendations(
        self,
        memory: MemoryMetrics,
        cpu: CPUMetrics,
        connectivity: Dict,
        anomalies: List
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Memory recommendations
        if memory.usage_pct > self.config.memory_warning_pct:
            recommendations.append(
                f"High memory usage ({memory.usage_pct:.1f}%) - consider scaling resources"
            )

        # CPU recommendations
        if cpu.cpu_pct > self.config.cpu_warning_pct:
            recommendations.append(
                f"High CPU usage ({cpu.cpu_pct:.1f}%) - review process efficiency"
            )

        # Connectivity recommendations
        disconnected = [
            k for k, v in connectivity.items()
            if v.status == ConnectionStatus.DISCONNECTED
        ]
        if disconnected:
            recommendations.append(
                f"Connection issues detected - review: {', '.join(disconnected[:3])}"
            )

        # Anomaly recommendations
        if len(anomalies) > 10:
            recommendations.append(
                f"Multiple anomalies detected ({len(anomalies)}) - investigate system behavior"
            )

        if not recommendations:
            recommendations.append("System health is optimal")

        return recommendations

    def get_recent_alerts(
        self,
        component: Optional[ComponentType] = None,
        severity: Optional[str] = None,
        hours: int = 24
    ) -> List[SystemAlert]:
        """Get recent alerts with optional filters"""
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        if component:
            alerts = [a for a in alerts if a.component == component]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def clear_old_data(self, days: int = 30):
        """Clear old data to manage memory"""
        cutoff = datetime.now() - timedelta(days=days)

        # Clear old anomalies
        self.anomalies = [a for a in self.anomalies if a.timestamp >= cutoff]

        # Clear old alerts
        self.alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        logger.info(f"Cleared system data older than {days} days")

    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            'platform': psutil.LINUX if hasattr(psutil, 'LINUX') else 'unknown',
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
            'python_version': __import__('sys').version,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }

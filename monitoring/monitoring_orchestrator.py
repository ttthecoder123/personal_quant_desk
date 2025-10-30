"""
Monitoring Orchestrator

Central coordination of all monitoring, alerting, logging, and operational
capabilities for the Personal Quant Desk trading system.
"""

import threading
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

# System Monitoring
from .system_monitoring.health_monitor import HealthMonitor
from .system_monitoring.performance_monitor import PerformanceMonitor
from .system_monitoring.resource_monitor import ResourceMonitor
from .system_monitoring.network_monitor import NetworkMonitor
from .system_monitoring.process_monitor import ProcessMonitor
from .system_monitoring.dependency_monitor import DependencyMonitor

# Data Monitoring
from .data_monitoring.feed_monitor import FeedMonitor
from .data_monitoring.quality_monitor import QualityMonitor
from .data_monitoring.latency_monitor import LatencyMonitor
from .data_monitoring.completeness_monitor import CompletenessMonitor
from .data_monitoring.anomaly_detector import AnomalyDetector

# Trading Monitoring
from .trading_monitoring.position_monitor import PositionMonitor
from .trading_monitoring.pnl_monitor import PnLMonitor
from .trading_monitoring.execution_monitor import ExecutionMonitor
from .trading_monitoring.signal_monitor import SignalMonitor
from .trading_monitoring.strategy_monitor import StrategyMonitor
from .trading_monitoring.risk_monitor import RiskMonitor

# Alerting
from .alerting.alert_engine import AlertEngine
from .alerting.alert_rules import AlertRules
from .alerting.alert_routing import AlertRouter
from .alerting.escalation_manager import EscalationManager
from .alerting.notification_channels import NotificationChannels
from .alerting.alert_suppression import AlertSuppression

# Logging
from .logging.structured_logger import StructuredLogger
from .logging.log_aggregator import LogAggregator
from .logging.log_analyzer import LogAnalyzer
from .logging.audit_logger import AuditLogger
from .logging.performance_logger import PerformanceLogger
from .logging.error_tracker import ErrorTracker

# Dashboards
from .dashboards.dashboard_server import DashboardServer

# Reporting
from .reporting.daily_reports import DailyReports
from .reporting.performance_reports import PerformanceReports
from .reporting.risk_reports import RiskReports
from .reporting.compliance_reports import ComplianceReports
from .reporting.incident_reports import IncidentReports
from .reporting.report_scheduler import ReportScheduler

# Diagnostics
from .diagnostics.system_diagnostics import SystemDiagnostics
from .diagnostics.performance_profiler import PerformanceProfiler
from .diagnostics.bottleneck_analyzer import BottleneckAnalyzer

# Maintenance
from .maintenance.backup_manager import BackupManager
from .maintenance.cleanup_manager import CleanupManager
from .maintenance.maintenance_window import MaintenanceWindow

# Metrics
from .metrics.metric_collector import MetricCollector
from .metrics.metric_aggregator import MetricAggregator
from .metrics.metric_api import MetricAPI


@dataclass
class MonitoringStatus:
    """Overall monitoring system status."""
    is_running: bool
    start_time: Optional[datetime]
    components_active: int
    components_total: int
    alerts_active: int
    errors_last_hour: int
    health_score: float  # 0-100


class MonitoringOrchestrator:
    """
    Central monitoring coordination system.

    Responsibilities:
    - Initialize all monitoring components
    - Coordinate monitoring activities
    - Manage alert flow
    - Generate reports
    - Handle maintenance windows
    - Provide monitoring API
    - Manage monitoring state
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize monitoring orchestrator.

        Args:
            config_path: Path to monitoring configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "monitoring_config.yaml"

        self.config = self._load_config(config_path)

        # State
        self.running = False
        self.start_time: Optional[datetime] = None
        self.lock = threading.Lock()

        # Initialize logging first
        self._init_logging()

        # Initialize monitoring components
        self._init_system_monitors()
        self._init_data_monitors()
        self._init_trading_monitors()
        self._init_alerting()
        self._init_dashboards()
        self._init_reporting()
        self._init_diagnostics()
        self._init_maintenance()
        self._init_metrics()

        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None

        self.logger.info("Monitoring orchestrator initialized")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'monitoring': {
                'intervals': {
                    'system_health': 10,
                    'performance_metrics': 30,
                    'resource_usage': 60,
                },
                'thresholds': {
                    'cpu': {'warning': 75, 'critical': 90},
                    'memory': {'warning': 80, 'critical': 90},
                }
            }
        }

    def _init_logging(self):
        """Initialize logging system."""
        self.logger = StructuredLogger(
            name='monitoring_orchestrator',
            level=self.config.get('monitoring', {}).get('logging', {}).get('level', 'INFO')
        )
        self.log_aggregator = LogAggregator()
        self.log_analyzer = LogAnalyzer()
        self.audit_logger = AuditLogger()
        self.performance_logger = PerformanceLogger()
        self.error_tracker = ErrorTracker()

    def _init_system_monitors(self):
        """Initialize system monitoring components."""
        intervals = self.config.get('monitoring', {}).get('intervals', {})

        self.health_monitor = HealthMonitor(
            check_interval=intervals.get('system_health', 10)
        )
        self.performance_monitor = PerformanceMonitor()
        self.resource_monitor = ResourceMonitor(
            sample_interval=intervals.get('resource_usage', 60)
        )
        self.network_monitor = NetworkMonitor(
            check_interval=intervals.get('network_check', 30)
        )
        self.process_monitor = ProcessMonitor(
            check_interval=intervals.get('process_check', 15)
        )
        self.dependency_monitor = DependencyMonitor(
            check_interval=intervals.get('dependency_check', 30)
        )

    def _init_data_monitors(self):
        """Initialize data monitoring components."""
        intervals = self.config.get('monitoring', {}).get('intervals', {})

        self.feed_monitor = FeedMonitor(
            check_interval=intervals.get('feed_check', 1)
        )
        self.quality_monitor = QualityMonitor()
        self.latency_monitor = LatencyMonitor()
        self.completeness_monitor = CompletenessMonitor()
        self.anomaly_detector = AnomalyDetector()

    def _init_trading_monitors(self):
        """Initialize trading monitoring components."""
        intervals = self.config.get('monitoring', {}).get('intervals', {})

        self.position_monitor = PositionMonitor()
        self.pnl_monitor = PnLMonitor()
        self.execution_monitor = ExecutionMonitor()
        self.signal_monitor = SignalMonitor()
        self.strategy_monitor = StrategyMonitor()
        self.risk_monitor = RiskMonitor()

    def _init_alerting(self):
        """Initialize alerting system."""
        alert_config_path = Path(__file__).parent / "config" / "alert_config.yaml"
        try:
            with open(alert_config_path, 'r') as f:
                alert_config = yaml.safe_load(f)
        except:
            alert_config = {}

        self.alert_engine = AlertEngine()
        self.alert_rules = AlertRules(alert_config.get('alert_rules', {}))
        self.alert_router = AlertRouter(alert_config.get('routing', {}))
        self.escalation_manager = EscalationManager(alert_config.get('escalation', {}))
        self.notification_channels = NotificationChannels(
            self.config.get('monitoring', {}).get('integrations', {})
        )
        self.alert_suppression = AlertSuppression(alert_config.get('suppression', {}))

    def _init_dashboards(self):
        """Initialize dashboard server."""
        dashboard_config_path = Path(__file__).parent / "config" / "dashboard_config.yaml"
        try:
            with open(dashboard_config_path, 'r') as f:
                dashboard_config = yaml.safe_load(f)
        except:
            dashboard_config = {}

        self.dashboard_server = DashboardServer(
            config=dashboard_config,
            orchestrator=self
        )

    def _init_reporting(self):
        """Initialize reporting system."""
        self.daily_reports = DailyReports()
        self.performance_reports = PerformanceReports()
        self.risk_reports = RiskReports()
        self.compliance_reports = ComplianceReports()
        self.incident_reports = IncidentReports()
        self.report_scheduler = ReportScheduler()

    def _init_diagnostics(self):
        """Initialize diagnostic tools."""
        self.system_diagnostics = SystemDiagnostics()
        self.performance_profiler = PerformanceProfiler()
        self.bottleneck_analyzer = BottleneckAnalyzer()

    def _init_maintenance(self):
        """Initialize maintenance systems."""
        self.backup_manager = BackupManager()
        self.cleanup_manager = CleanupManager()
        self.maintenance_window = MaintenanceWindow()

    def _init_metrics(self):
        """Initialize metrics collection and aggregation."""
        self.metric_collector = MetricCollector()
        self.metric_aggregator = MetricAggregator()
        self.metric_api = MetricAPI()

    def start(self):
        """Start all monitoring systems."""
        with self.lock:
            if self.running:
                self.logger.warning("Monitoring already running")
                return

            self.logger.info("Starting monitoring orchestrator")
            self.running = True
            self.start_time = datetime.now()

            # Start all monitors
            self._start_all_monitors()

            # Start alerting
            self.alert_engine.start()

            # Start dashboards
            self.dashboard_server.start()

            # Start maintenance
            self.backup_manager.start()
            self.cleanup_manager.start()

            # Start metrics collection
            self.metric_collector.start()
            self.metric_aggregator.start()

            # Start orchestrator monitoring thread
            self.monitor_thread = threading.Thread(target=self._orchestrator_loop, daemon=True)
            self.monitor_thread.start()

            self.logger.info("Monitoring orchestrator started successfully")
            self.audit_logger.log_event("monitoring_started", {
                'timestamp': datetime.now().isoformat()
            })

    def stop(self):
        """Stop all monitoring systems."""
        with self.lock:
            if not self.running:
                return

            self.logger.info("Stopping monitoring orchestrator")
            self.running = False

            # Stop all monitors
            self._stop_all_monitors()

            # Stop alerting
            self.alert_engine.stop()

            # Stop dashboards
            self.dashboard_server.stop()

            # Stop maintenance
            self.backup_manager.stop()
            self.cleanup_manager.stop()

            # Stop metrics
            self.metric_collector.stop()
            self.metric_aggregator.stop()

            # Wait for orchestrator thread
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)

            self.logger.info("Monitoring orchestrator stopped")
            self.audit_logger.log_event("monitoring_stopped", {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            })

    def _start_all_monitors(self):
        """Start all monitoring components."""
        # System monitors
        self.health_monitor.start()
        self.performance_monitor.start()
        self.resource_monitor.start()
        self.network_monitor.start()
        self.process_monitor.start()
        self.dependency_monitor.start()

        # Data monitors
        self.feed_monitor.start()
        self.quality_monitor.start()
        self.latency_monitor.start()
        self.completeness_monitor.start()
        self.anomaly_detector.start()

        # Trading monitors
        self.position_monitor.start()
        self.pnl_monitor.start()
        self.execution_monitor.start()
        self.signal_monitor.start()
        self.strategy_monitor.start()
        self.risk_monitor.start()

    def _stop_all_monitors(self):
        """Stop all monitoring components."""
        # System monitors
        self.health_monitor.stop()
        self.performance_monitor.stop()
        self.resource_monitor.stop()
        self.network_monitor.stop()
        self.process_monitor.stop()
        self.dependency_monitor.stop()

        # Data monitors
        self.feed_monitor.stop()
        self.quality_monitor.stop()
        self.latency_monitor.stop()
        self.completeness_monitor.stop()
        self.anomaly_detector.stop()

        # Trading monitors
        self.position_monitor.stop()
        self.pnl_monitor.stop()
        self.execution_monitor.stop()
        self.signal_monitor.stop()
        self.strategy_monitor.stop()
        self.risk_monitor.stop()

    def _orchestrator_loop(self):
        """Main orchestration loop."""
        while self.running:
            try:
                # Collect metrics from all monitors
                self._collect_metrics()

                # Check for alerts
                self._process_alerts()

                # Update dashboards
                self._update_dashboards()

                # Run scheduled tasks
                self._run_scheduled_tasks()

                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Error in orchestrator loop: {e}")
                self.error_tracker.record_error(e)

    def _collect_metrics(self):
        """Collect metrics from all monitoring components."""
        try:
            # System metrics
            self.metric_collector.collect('health', self.health_monitor.get_summary())
            self.metric_collector.collect('performance', self.performance_monitor.get_summary())
            self.metric_collector.collect('resources', self.resource_monitor.get_summary())

            # Data metrics
            self.metric_collector.collect('feeds', self.feed_monitor.get_summary())
            self.metric_collector.collect('data_quality', self.quality_monitor.get_summary())

            # Trading metrics
            self.metric_collector.collect('positions', self.position_monitor.get_summary())
            self.metric_collector.collect('pnl', self.pnl_monitor.get_summary())
            self.metric_collector.collect('execution', self.execution_monitor.get_summary())

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")

    def _process_alerts(self):
        """Process and route alerts."""
        try:
            # Get all potential alerts
            alerts = []

            # Check system health
            unhealthy = self.health_monitor.get_unhealthy_components()
            for component in unhealthy:
                alerts.append({
                    'type': 'system_health',
                    'severity': 'critical' if component.status == 'unhealthy' else 'warning',
                    'message': f"Component {component.component} is {component.status}",
                    'details': component
                })

            # Process through alert engine
            for alert in alerts:
                if not self.alert_suppression.should_suppress(alert):
                    self.alert_engine.process_alert(alert)
                    self.alert_router.route_alert(alert)

        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")

    def _update_dashboards(self):
        """Update dashboard data."""
        try:
            dashboard_data = {
                'system': self.get_system_status(),
                'trading': self.get_trading_status(),
                'risk': self.get_risk_status(),
                'performance': self.get_performance_status()
            }
            self.dashboard_server.update_data(dashboard_data)
        except Exception as e:
            self.logger.error(f"Error updating dashboards: {e}")

    def _run_scheduled_tasks(self):
        """Run scheduled tasks (reports, backups, etc.)."""
        try:
            # Run scheduled reports
            self.report_scheduler.run_due_reports()

            # Check for maintenance windows
            if self.maintenance_window.is_maintenance_time():
                self._handle_maintenance_mode()

        except Exception as e:
            self.logger.error(f"Error running scheduled tasks: {e}")

    def _handle_maintenance_mode(self):
        """Handle maintenance mode operations."""
        self.logger.info("Entering maintenance mode")
        self.alert_suppression.enable_maintenance_mode()
        # Additional maintenance tasks here

    # Public API Methods

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            'health': self.health_monitor.get_summary(),
            'performance': self.performance_monitor.get_summary(),
            'resources': self.resource_monitor.get_summary(),
            'network': self.network_monitor.get_summary()
        }

    def get_trading_status(self) -> Dict[str, Any]:
        """Get trading system status."""
        return {
            'positions': self.position_monitor.get_summary(),
            'pnl': self.pnl_monitor.get_summary(),
            'execution': self.execution_monitor.get_summary(),
            'strategies': self.strategy_monitor.get_summary()
        }

    def get_risk_status(self) -> Dict[str, Any]:
        """Get risk status."""
        return self.risk_monitor.get_summary()

    def get_performance_status(self) -> Dict[str, Any]:
        """Get performance status."""
        return {
            'strategies': self.strategy_monitor.get_performance_summary(),
            'execution': self.execution_monitor.get_performance_summary()
        }

    def get_monitoring_status(self) -> MonitoringStatus:
        """Get monitoring system status."""
        with self.lock:
            # Count active components
            total = 22  # Total number of monitoring components
            active = sum([
                self.health_monitor.running,
                self.performance_monitor.running,
                self.resource_monitor.running,
                self.network_monitor.running,
                self.process_monitor.running,
                self.dependency_monitor.running,
                self.feed_monitor.running,
                self.quality_monitor.running,
                self.latency_monitor.running,
                self.completeness_monitor.running,
                self.anomaly_detector.running,
                self.position_monitor.running,
                self.pnl_monitor.running,
                self.execution_monitor.running,
                self.signal_monitor.running,
                self.strategy_monitor.running,
                self.risk_monitor.running,
                self.alert_engine.running,
                self.dashboard_server.running,
                self.backup_manager.running,
                self.cleanup_manager.running,
                self.metric_collector.running
            ])

            health_score = (active / total * 100) if total > 0 else 0

            return MonitoringStatus(
                is_running=self.running,
                start_time=self.start_time,
                components_active=active,
                components_total=total,
                alerts_active=len(self.alert_engine.get_active_alerts()) if hasattr(self.alert_engine, 'get_active_alerts') else 0,
                errors_last_hour=self.error_tracker.get_error_count(hours=1) if hasattr(self.error_tracker, 'get_error_count') else 0,
                health_score=health_score
            )

    def generate_report(self, report_type: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a monitoring report.

        Args:
            report_type: Type of report (daily, performance, risk, compliance, incident)
            **kwargs: Additional parameters for report generation

        Returns:
            Generated report data
        """
        if report_type == 'daily':
            return self.daily_reports.generate(**kwargs)
        elif report_type == 'performance':
            return self.performance_reports.generate(**kwargs)
        elif report_type == 'risk':
            return self.risk_reports.generate(**kwargs)
        elif report_type == 'compliance':
            return self.compliance_reports.generate(**kwargs)
        elif report_type == 'incident':
            return self.incident_reports.generate(**kwargs)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def run_diagnostics(self, diagnostic_type: str = 'full') -> Dict[str, Any]:
        """
        Run system diagnostics.

        Args:
            diagnostic_type: Type of diagnostic (full, quick, targeted)

        Returns:
            Diagnostic results
        """
        if diagnostic_type == 'full':
            return self.system_diagnostics.run_full_diagnostics()
        elif diagnostic_type == 'quick':
            return self.system_diagnostics.run_quick_checks()
        else:
            return self.system_diagnostics.run_targeted_diagnostics(diagnostic_type)

    def trigger_backup(self) -> bool:
        """
        Trigger manual backup.

        Returns:
            True if backup initiated successfully
        """
        try:
            self.backup_manager.trigger_backup()
            return True
        except Exception as e:
            self.logger.error(f"Error triggering backup: {e}")
            return False

    def get_metrics(self, metric_name: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Query collected metrics.

        Args:
            metric_name: Optional specific metric name
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Metric data
        """
        return self.metric_api.query(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time
        )

"""
Monitoring Service for system health, performance, and alerting.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import smtplib
import pandas as pd
import numpy as np
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests
import psutil

from utils.logger import log, log_alert

log = log.bind(name="Monitoring")


class AlertManager:
    """Manages different types of alerts."""

    def __init__(self, config: dict):
        """Initialize alert manager."""
        self.config = config
        self.email_config = config.get('alerts', {}).get('email', {})
        self.slack_config = config.get('alerts', {}).get('slack', {})

    async def send_email_alert(self, subject: str, message: str):
        """Send email alert."""
        if not self.email_config.get('enabled', False):
            return

        try:
            smtp_server = os.getenv('SMTP_SERVER')
            smtp_port = os.getenv('SMTP_PORT', 587)
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASSWORD')
            recipients = self.email_config.get('recipients', [])

            if not all([smtp_server, smtp_user, smtp_password, recipients]):
                log.warning("Email configuration incomplete - skipping email alert")
                return

            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Trading System Alert: {subject}"

            body = f"""
Trading System Alert

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Subject: {subject}

Message:
{message}

---
Quantitative Trading System
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            text = msg.as_string()
            server.sendmail(smtp_user, recipients, text)
            server.quit()

            log.info(f"Email alert sent: {subject}")

        except Exception as e:
            log.error(f"Failed to send email alert: {str(e)}")

    async def send_slack_alert(self, subject: str, message: str):
        """Send Slack alert."""
        if not self.slack_config.get('enabled', False):
            return

        try:
            webhook_url = os.getenv('SLACK_WEBHOOK')
            if not webhook_url:
                log.warning("Slack webhook not configured")
                return

            payload = {
                'text': f"🚨 Trading System Alert: {subject}",
                'attachments': [{
                    'color': 'danger',
                    'fields': [
                        {
                            'title': 'Time',
                            'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        },
                        {
                            'title': 'Message',
                            'value': message,
                            'short': False
                        }
                    ]
                }]
            }

            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            log.info(f"Slack alert sent: {subject}")

        except Exception as e:
            log.error(f"Failed to send Slack alert: {str(e)}")

    async def send_alert(self, subject: str, message: str):
        """Send alert through all configured channels."""
        await asyncio.gather(
            self.send_email_alert(subject, message),
            self.send_slack_alert(subject, message)
        )


class PerformanceMonitor:
    """Monitors trading system performance metrics."""

    def __init__(self, config: dict):
        """Initialize performance monitor."""
        self.config = config
        self.thresholds = config.get('thresholds', {})
        self.metrics_history = []

    async def calculate_metrics(self, portfolio_value: float, positions: Dict) -> Dict[str, float]:
        """Calculate current performance metrics."""
        metrics = {
            'portfolio_value': portfolio_value,
            'num_positions': len(positions),
            'cash_percentage': 0,  # Would calculate from actual positions
            'daily_pnl': 0,
            'daily_return': 0,
            'current_drawdown': 0
        }

        # Calculate additional metrics if history is available
        if self.metrics_history:
            previous_value = self.metrics_history[-1].get('portfolio_value', portfolio_value)
            metrics['daily_pnl'] = portfolio_value - previous_value
            metrics['daily_return'] = (portfolio_value / previous_value - 1) if previous_value > 0 else 0

        # Calculate drawdown
        if self.metrics_history:
            values = [m['portfolio_value'] for m in self.metrics_history] + [portfolio_value]
            peak = max(values)
            metrics['current_drawdown'] = (peak - portfolio_value) / peak if peak > 0 else 0

        # Store metrics
        metrics['timestamp'] = datetime.now()
        self.metrics_history.append(metrics)

        # Keep only recent history (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.metrics_history = [
            m for m in self.metrics_history
            if m['timestamp'] > cutoff_date
        ]

        return metrics

    async def check_alert_conditions(self, metrics: Dict[str, float], alert_manager: AlertManager):
        """Check if any alert conditions are met."""
        # Drawdown alert
        drawdown_threshold = self.thresholds.get('drawdown_alert', 0.10)
        if metrics['current_drawdown'] > drawdown_threshold:
            await alert_manager.send_alert(
                "High Drawdown Alert",
                f"Current drawdown: {metrics['current_drawdown']:.2%} "
                f"(threshold: {drawdown_threshold:.2%})"
            )

        # Daily loss alert
        daily_loss_threshold = self.thresholds.get('daily_loss_alert', 0.05)
        if metrics['daily_return'] < -daily_loss_threshold:
            await alert_manager.send_alert(
                "Large Daily Loss",
                f"Daily return: {metrics['daily_return']:.2%} "
                f"(threshold: {-daily_loss_threshold:.2%})"
            )

        # Position limit alert
        position_limit_threshold = self.thresholds.get('position_limit_alert', 0.90)
        max_positions = 20  # Would get from config
        if metrics['num_positions'] / max_positions > position_limit_threshold:
            await alert_manager.send_alert(
                "Position Limit Alert",
                f"Current positions: {metrics['num_positions']}/{max_positions} "
                f"({metrics['num_positions'] / max_positions:.1%})"
            )


class SystemMonitor:
    """Monitors system health and resources."""

    def __init__(self):
        """Initialize system monitor."""
        self.alerts_sent = set()

    async def check_system_health(self, alert_manager: AlertManager):
        """Check system health metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90 and 'cpu_high' not in self.alerts_sent:
            await alert_manager.send_alert(
                "High CPU Usage",
                f"CPU usage: {cpu_percent:.1f}%"
            )
            self.alerts_sent.add('cpu_high')
        elif cpu_percent < 70 and 'cpu_high' in self.alerts_sent:
            self.alerts_sent.remove('cpu_high')

        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90 and 'memory_high' not in self.alerts_sent:
            await alert_manager.send_alert(
                "High Memory Usage",
                f"Memory usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)"
            )
            self.alerts_sent.add('memory_high')
        elif memory.percent < 70 and 'memory_high' in self.alerts_sent:
            self.alerts_sent.remove('memory_high')

        # Disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > 90 and 'disk_high' not in self.alerts_sent:
            await alert_manager.send_alert(
                "High Disk Usage",
                f"Disk usage: {disk.percent:.1f}% ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)"
            )
            self.alerts_sent.add('disk_high')
        elif disk.percent < 70 and 'disk_high' in self.alerts_sent:
            self.alerts_sent.remove('disk_high')


class ReportGenerator:
    """Generates performance reports."""

    def __init__(self, config: dict):
        """Initialize report generator."""
        self.config = config
        self.reporting_config = config.get('reporting', {})

    async def generate_daily_report(
        self,
        metrics: Dict[str, float],
        positions: Dict,
        trades: List[Dict]
    ) -> str:
        """Generate daily performance report."""
        report = f"""
DAILY TRADING REPORT
Date: {datetime.now().strftime('%Y-%m-%d')}

PERFORMANCE METRICS:
- Portfolio Value: ${metrics.get('portfolio_value', 0):,.2f}
- Daily P&L: ${metrics.get('daily_pnl', 0):,.2f}
- Daily Return: {metrics.get('daily_return', 0):.2%}
- Current Drawdown: {metrics.get('current_drawdown', 0):.2%}

POSITIONS:
- Number of Positions: {len(positions)}
- Active Symbols: {', '.join(positions.keys()) if positions else 'None'}

TRADING ACTIVITY:
- Trades Today: {len([t for t in trades if t.get('timestamp', datetime.now()).date() == datetime.now().date()])}

SYSTEM STATUS:
- Status: Operational
- Uptime: Running
- Last Update: {datetime.now().strftime('%H:%M:%S')}

---
Quantitative Trading System
        """

        return report.strip()

    async def send_daily_report(
        self,
        metrics: Dict[str, float],
        positions: Dict,
        trades: List[Dict],
        alert_manager: AlertManager
    ):
        """Send daily report."""
        if not self.reporting_config.get('daily_report', False):
            return

        try:
            report = await self.generate_daily_report(metrics, positions, trades)

            # Send as email (not alert)
            if alert_manager.email_config.get('enabled', False):
                await alert_manager.send_email_alert("Daily Report", report)

            log.info("Daily report sent")

        except Exception as e:
            log.error(f"Failed to send daily report: {str(e)}")


class MonitoringService:
    """Main monitoring service orchestrator."""

    def __init__(self, config: dict):
        """Initialize monitoring service."""
        self.config = config
        self.alert_manager = AlertManager(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.system_monitor = SystemMonitor()
        self.report_generator = ReportGenerator(config)

        self.running = False
        self.last_metrics = {}
        self.last_report_time = None

    async def start(self):
        """Start monitoring service."""
        log.info("Starting monitoring service...")
        self.running = True

        # Start monitoring tasks
        asyncio.create_task(self.monitoring_loop())
        asyncio.create_task(self.reporting_loop())

        log.success("Monitoring service started")

    async def monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check system health
                await self.system_monitor.check_system_health(self.alert_manager)

                # Sleep for 60 seconds between checks
                await asyncio.sleep(60)

            except Exception as e:
                log.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)

    async def reporting_loop(self):
        """Daily reporting loop."""
        while self.running:
            try:
                now = datetime.now()
                report_time = self.config.get('reporting', {}).get('report_time', '17:00')
                report_hour, report_minute = map(int, report_time.split(':'))

                # Check if it's time to send daily report
                if (now.hour == report_hour and now.minute == report_minute and
                    (not self.last_report_time or
                     self.last_report_time.date() != now.date())):

                    await self.send_daily_report()
                    self.last_report_time = now

                # Sleep for 60 seconds
                await asyncio.sleep(60)

            except Exception as e:
                log.error(f"Error in reporting loop: {str(e)}")
                await asyncio.sleep(60)

    async def update_metrics(self, portfolio_value: float = None, positions: Dict = None):
        """Update performance metrics."""
        try:
            if portfolio_value is None or positions is None:
                log.debug("No metrics to update")
                return

            # Calculate current metrics
            metrics = await self.performance_monitor.calculate_metrics(
                portfolio_value,
                positions
            )

            # Check alert conditions
            await self.performance_monitor.check_alert_conditions(
                metrics,
                self.alert_manager
            )

            # Store metrics
            self.last_metrics = metrics

            log.debug("Metrics updated")

        except Exception as e:
            log.error(f"Error updating metrics: {str(e)}")

    async def send_alert(self, subject: str, message: str):
        """Send alert through monitoring service."""
        await self.alert_manager.send_alert(subject, message)
        log_alert(f"Alert sent: {subject}")

    async def send_daily_report(self):
        """Send daily report."""
        try:
            # Get positions and trades (would come from execution manager)
            positions = {}  # Placeholder
            trades = []    # Placeholder

            await self.report_generator.send_daily_report(
                self.last_metrics,
                positions,
                trades,
                self.alert_manager
            )

        except Exception as e:
            log.error(f"Error sending daily report: {str(e)}")

    async def stop(self):
        """Stop monitoring service."""
        log.info("Stopping monitoring service...")
        self.running = False
        log.info("Monitoring service stopped")
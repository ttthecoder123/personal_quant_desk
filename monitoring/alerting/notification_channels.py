"""
Notification Channels - Multi-channel alert delivery.

Features:
- Email notifications
- SMS notifications
- Slack integration
- PagerDuty integration
- Phone calls
- Mobile push notifications
- Dashboard alerts
- Webhook integrations
"""

import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from abc import ABC, abstractmethod


class ChannelType(Enum):
    """Notification channel types."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    PHONE = "phone"
    PUSH = "push"
    DASHBOARD = "dashboard"
    WEBHOOK = "webhook"


class DeliveryStatus(Enum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel."""
    channel_type: ChannelType
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    retry_delay_seconds: int = 60
    rate_limit_per_minute: int = 60


@dataclass
class Notification:
    """Notification message."""
    notification_id: str
    alert_id: str
    channel: ChannelType
    recipient: str
    subject: str
    message: str
    priority: str = "normal"
    status: DeliveryStatus = DeliveryStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)


class NotificationChannel(ABC):
    """Base class for notification channels."""

    def __init__(self, config: NotificationConfig):
        """
        Initialize notification channel.

        Args:
            config: Channel configuration
        """
        self.config = config
        self.lock = threading.Lock()
        self.delivery_history: deque = deque(maxlen=1000)
        self.rate_limit_timestamps: deque = deque(maxlen=100)

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """
        Send notification.

        Args:
            notification: Notification to send

        Returns:
            True if sent successfully
        """
        pass

    def can_send(self) -> bool:
        """Check if channel can send based on rate limits."""
        with self.lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Remove old timestamps
            while self.rate_limit_timestamps and \
                  self.rate_limit_timestamps[0] < minute_ago:
                self.rate_limit_timestamps.popleft()

            # Check rate limit
            if len(self.rate_limit_timestamps) >= self.config.rate_limit_per_minute:
                return False

            return True

    def record_send(self):
        """Record that a message was sent."""
        with self.lock:
            self.rate_limit_timestamps.append(datetime.now())

    def record_delivery(self, notification: Notification):
        """Record delivery result."""
        with self.lock:
            self.delivery_history.append({
                'notification_id': notification.notification_id,
                'status': notification.status.value,
                'timestamp': datetime.now()
            })


class EmailChannel(NotificationChannel):
    """Email notification channel."""

    def send(self, notification: Notification) -> bool:
        """Send email notification."""
        try:
            if not self.can_send():
                notification.error_message = "Rate limit exceeded"
                return False

            smtp_config = self.config.config
            smtp_server = smtp_config.get('server')
            smtp_port = smtp_config.get('port', 587)
            smtp_user = smtp_config.get('username')
            smtp_password = smtp_config.get('password')
            from_addr = smtp_config.get('from_address', smtp_user)

            if not all([smtp_server, smtp_user, smtp_password]):
                notification.error_message = "Email configuration incomplete"
                return False

            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_addr
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject

            # Add priority headers
            if notification.priority == 'critical':
                msg['X-Priority'] = '1'
                msg['Importance'] = 'high'

            # Create HTML body
            html_body = f"""
            <html>
                <head></head>
                <body>
                    <h2>{notification.subject}</h2>
                    <p><strong>Priority:</strong> {notification.priority.upper()}</p>
                    <p><strong>Time:</strong> {notification.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr>
                    <pre>{notification.message}</pre>
                    <hr>
                    <p><small>Alert ID: {notification.alert_id}</small></p>
                </body>
            </html>
            """

            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            server.quit()

            notification.status = DeliveryStatus.SENT
            notification.delivered_at = datetime.now()
            self.record_send()
            self.record_delivery(notification)

            return True

        except Exception as e:
            notification.error_message = str(e)
            notification.status = DeliveryStatus.FAILED
            return False


class SlackChannel(NotificationChannel):
    """Slack notification channel."""

    def send(self, notification: Notification) -> bool:
        """Send Slack notification."""
        try:
            if not self.can_send():
                notification.error_message = "Rate limit exceeded"
                return False

            webhook_url = self.config.config.get('webhook_url')
            if not webhook_url:
                notification.error_message = "Slack webhook URL not configured"
                return False

            # Determine color based on priority
            color_map = {
                'critical': 'danger',
                'high': 'warning',
                'medium': '#FFA500',
                'low': 'good'
            }
            color = color_map.get(notification.priority, 'good')

            # Build Slack message
            payload = {
                'text': f":rotating_light: *{notification.subject}*",
                'attachments': [{
                    'color': color,
                    'fields': [
                        {
                            'title': 'Priority',
                            'value': notification.priority.upper(),
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': notification.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        },
                        {
                            'title': 'Alert ID',
                            'value': notification.alert_id,
                            'short': True
                        },
                        {
                            'title': 'Message',
                            'value': notification.message,
                            'short': False
                        }
                    ],
                    'footer': 'Alert System',
                    'ts': int(notification.created_at.timestamp())
                }]
            }

            # Add metadata if present
            if notification.metadata:
                metadata_str = '\n'.join([
                    f"*{k}:* {v}"
                    for k, v in notification.metadata.items()
                ])
                payload['attachments'][0]['fields'].append({
                    'title': 'Additional Info',
                    'value': metadata_str,
                    'short': False
                })

            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            notification.status = DeliveryStatus.SENT
            notification.delivered_at = datetime.now()
            self.record_send()
            self.record_delivery(notification)

            return True

        except Exception as e:
            notification.error_message = str(e)
            notification.status = DeliveryStatus.FAILED
            return False


class SMSChannel(NotificationChannel):
    """SMS notification channel (Twilio)."""

    def send(self, notification: Notification) -> bool:
        """Send SMS notification."""
        try:
            if not self.can_send():
                notification.error_message = "Rate limit exceeded"
                return False

            # Get Twilio config
            account_sid = self.config.config.get('account_sid')
            auth_token = self.config.config.get('auth_token')
            from_number = self.config.config.get('from_number')

            if not all([account_sid, auth_token, from_number]):
                notification.error_message = "Twilio configuration incomplete"
                return False

            # Build SMS message (keep it short)
            sms_body = f"[{notification.priority.upper()}] {notification.subject}\n{notification.message[:100]}"

            # Send via Twilio API
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
            data = {
                'From': from_number,
                'To': notification.recipient,
                'Body': sms_body
            }

            response = requests.post(
                url,
                data=data,
                auth=(account_sid, auth_token),
                timeout=10
            )
            response.raise_for_status()

            notification.status = DeliveryStatus.SENT
            notification.delivered_at = datetime.now()
            self.record_send()
            self.record_delivery(notification)

            return True

        except Exception as e:
            notification.error_message = str(e)
            notification.status = DeliveryStatus.FAILED
            return False


class PagerDutyChannel(NotificationChannel):
    """PagerDuty notification channel."""

    def send(self, notification: Notification) -> bool:
        """Send PagerDuty notification."""
        try:
            if not self.can_send():
                notification.error_message = "Rate limit exceeded"
                return False

            integration_key = self.config.config.get('integration_key')
            if not integration_key:
                notification.error_message = "PagerDuty integration key not configured"
                return False

            # Map priority to PagerDuty severity
            severity_map = {
                'critical': 'critical',
                'high': 'error',
                'medium': 'warning',
                'low': 'info'
            }
            severity = severity_map.get(notification.priority, 'info')

            # Build PagerDuty event
            payload = {
                'routing_key': integration_key,
                'event_action': 'trigger',
                'payload': {
                    'summary': notification.subject,
                    'severity': severity,
                    'source': 'alert_system',
                    'timestamp': notification.created_at.isoformat(),
                    'custom_details': {
                        'message': notification.message,
                        'alert_id': notification.alert_id,
                        'priority': notification.priority,
                        **notification.metadata
                    }
                }
            }

            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            notification.status = DeliveryStatus.SENT
            notification.delivered_at = datetime.now()
            self.record_send()
            self.record_delivery(notification)

            return True

        except Exception as e:
            notification.error_message = str(e)
            notification.status = DeliveryStatus.FAILED
            return False


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""

    def send(self, notification: Notification) -> bool:
        """Send webhook notification."""
        try:
            if not self.can_send():
                notification.error_message = "Rate limit exceeded"
                return False

            webhook_url = self.config.config.get('url')
            if not webhook_url:
                notification.error_message = "Webhook URL not configured"
                return False

            # Build payload
            payload = {
                'notification_id': notification.notification_id,
                'alert_id': notification.alert_id,
                'subject': notification.subject,
                'message': notification.message,
                'priority': notification.priority,
                'timestamp': notification.created_at.isoformat(),
                'metadata': notification.metadata
            }

            # Get additional config
            headers = self.config.config.get('headers', {})
            method = self.config.config.get('method', 'POST').upper()

            # Send webhook
            response = requests.request(
                method,
                webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            notification.status = DeliveryStatus.SENT
            notification.delivered_at = datetime.now()
            self.record_send()
            self.record_delivery(notification)

            return True

        except Exception as e:
            notification.error_message = str(e)
            notification.status = DeliveryStatus.FAILED
            return False


class DashboardChannel(NotificationChannel):
    """Dashboard notification channel (in-memory queue)."""

    def __init__(self, config: NotificationConfig):
        """Initialize dashboard channel."""
        super().__init__(config)
        self.dashboard_queue: deque = deque(maxlen=100)

    def send(self, notification: Notification) -> bool:
        """Add notification to dashboard queue."""
        try:
            with self.lock:
                self.dashboard_queue.append({
                    'id': notification.notification_id,
                    'alert_id': notification.alert_id,
                    'subject': notification.subject,
                    'message': notification.message,
                    'priority': notification.priority,
                    'timestamp': notification.created_at.isoformat()
                })

            notification.status = DeliveryStatus.DELIVERED
            notification.delivered_at = datetime.now()
            self.record_delivery(notification)

            return True

        except Exception as e:
            notification.error_message = str(e)
            notification.status = DeliveryStatus.FAILED
            return False

    def get_notifications(self, limit: int = 50) -> List[Dict]:
        """Get recent dashboard notifications."""
        with self.lock:
            return list(self.dashboard_queue)[-limit:]


class NotificationChannels:
    """
    Multi-channel notification management system.

    Features:
    - Multiple notification channels (Email, SMS, Slack, PagerDuty, etc.)
    - Automatic retry with exponential backoff
    - Rate limiting per channel
    - Delivery tracking and status
    - Channel fallback
    """

    def __init__(self):
        """Initialize notification channels."""
        self.channels: Dict[ChannelType, NotificationChannel] = {}
        self.notifications: Dict[str, Notification] = {}
        self.pending_queue: deque = deque()
        self.lock = threading.Lock()

        # Retry management
        self.retry_thread: Optional[threading.Thread] = None
        self.retry_active = False

        # Statistics
        self.stats = defaultdict(lambda: defaultdict(int))

    def register_channel(
        self,
        channel_type: ChannelType,
        channel: NotificationChannel
    ):
        """
        Register notification channel.

        Args:
            channel_type: Type of channel
            channel: Channel instance
        """
        with self.lock:
            self.channels[channel_type] = channel

    def configure_email(self, config: Dict[str, Any]):
        """Configure email channel."""
        channel_config = NotificationConfig(
            channel_type=ChannelType.EMAIL,
            config=config
        )
        self.register_channel(ChannelType.EMAIL, EmailChannel(channel_config))

    def configure_slack(self, webhook_url: str):
        """Configure Slack channel."""
        channel_config = NotificationConfig(
            channel_type=ChannelType.SLACK,
            config={'webhook_url': webhook_url}
        )
        self.register_channel(ChannelType.SLACK, SlackChannel(channel_config))

    def configure_sms(self, config: Dict[str, Any]):
        """Configure SMS channel."""
        channel_config = NotificationConfig(
            channel_type=ChannelType.SMS,
            config=config
        )
        self.register_channel(ChannelType.SMS, SMSChannel(channel_config))

    def configure_pagerduty(self, integration_key: str):
        """Configure PagerDuty channel."""
        channel_config = NotificationConfig(
            channel_type=ChannelType.PAGERDUTY,
            config={'integration_key': integration_key}
        )
        self.register_channel(ChannelType.PAGERDUTY, PagerDutyChannel(channel_config))

    def configure_webhook(self, url: str, headers: Optional[Dict] = None):
        """Configure webhook channel."""
        channel_config = NotificationConfig(
            channel_type=ChannelType.WEBHOOK,
            config={'url': url, 'headers': headers or {}}
        )
        self.register_channel(ChannelType.WEBHOOK, WebhookChannel(channel_config))

    def configure_dashboard(self):
        """Configure dashboard channel."""
        channel_config = NotificationConfig(
            channel_type=ChannelType.DASHBOARD,
            rate_limit_per_minute=1000  # High limit for dashboard
        )
        self.register_channel(ChannelType.DASHBOARD, DashboardChannel(channel_config))

    def send_notification(
        self,
        alert_id: str,
        channel: ChannelType,
        recipient: str,
        subject: str,
        message: str,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Send notification through specified channel.

        Args:
            alert_id: Alert ID
            channel: Notification channel
            recipient: Recipient identifier
            subject: Notification subject
            message: Notification message
            priority: Priority level
            metadata: Optional metadata

        Returns:
            Notification ID or None if failed
        """
        import hashlib
        import time

        # Generate notification ID
        notification_id = hashlib.md5(
            f"{alert_id}_{channel.value}_{time.time()}".encode()
        ).hexdigest()[:16]

        # Create notification
        notification = Notification(
            notification_id=notification_id,
            alert_id=alert_id,
            channel=channel,
            recipient=recipient,
            subject=subject,
            message=message,
            priority=priority,
            metadata=metadata or {}
        )

        with self.lock:
            self.notifications[notification_id] = notification

        # Try to send
        success = self._send_notification(notification)

        if not success and notification.attempts < self.channels[channel].config.max_retries:
            # Add to retry queue
            with self.lock:
                self.pending_queue.append(notification_id)

        return notification_id

    def _send_notification(self, notification: Notification) -> bool:
        """Send notification through channel."""
        channel = self.channels.get(notification.channel)
        if not channel:
            notification.error_message = "Channel not configured"
            notification.status = DeliveryStatus.FAILED
            return False

        notification.attempts += 1
        notification.last_attempt = datetime.now()

        success = channel.send(notification)

        # Update statistics
        with self.lock:
            self.stats[notification.channel.value]['total'] += 1
            if success:
                self.stats[notification.channel.value]['success'] += 1
            else:
                self.stats[notification.channel.value]['failed'] += 1

        return success

    def start_retry_service(self):
        """Start background retry service."""
        if self.retry_active:
            return

        self.retry_active = True
        self.retry_thread = threading.Thread(
            target=self._retry_loop,
            daemon=True
        )
        self.retry_thread.start()

    def stop_retry_service(self):
        """Stop background retry service."""
        self.retry_active = False
        if self.retry_thread:
            self.retry_thread.join(timeout=5)

    def _retry_loop(self):
        """Background retry loop."""
        import time

        while self.retry_active:
            try:
                with self.lock:
                    if not self.pending_queue:
                        continue

                    notification_id = self.pending_queue.popleft()

                if notification_id in self.notifications:
                    notification = self.notifications[notification_id]
                    channel = self.channels.get(notification.channel)

                    if not channel:
                        continue

                    # Check if enough time has passed
                    if notification.last_attempt:
                        elapsed = (datetime.now() - notification.last_attempt).total_seconds()
                        if elapsed < channel.config.retry_delay_seconds:
                            # Put back in queue
                            with self.lock:
                                self.pending_queue.append(notification_id)
                            continue

                    # Retry send
                    success = self._send_notification(notification)

                    if not success and notification.attempts < channel.config.max_retries:
                        # Re-queue for another retry
                        with self.lock:
                            self.pending_queue.append(notification_id)

                time.sleep(1)

            except Exception as e:
                print(f"Error in retry loop: {e}")
                time.sleep(5)

    def get_notification_status(self, notification_id: str) -> Optional[Notification]:
        """Get notification status."""
        with self.lock:
            return self.notifications.get(notification_id)

    def get_dashboard_notifications(self, limit: int = 50) -> List[Dict]:
        """Get dashboard notifications."""
        channel = self.channels.get(ChannelType.DASHBOARD)
        if isinstance(channel, DashboardChannel):
            return channel.get_notifications(limit)
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get notification statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            channel_stats = dict(self.stats)

            # Calculate success rates
            for channel, stats in channel_stats.items():
                total = stats.get('total', 0)
                success = stats.get('success', 0)
                stats['success_rate'] = (success / total * 100) if total > 0 else 0

            return {
                'channels': channel_stats,
                'pending_retries': len(self.pending_queue),
                'total_notifications': len(self.notifications),
                'configured_channels': list(self.channels.keys())
            }

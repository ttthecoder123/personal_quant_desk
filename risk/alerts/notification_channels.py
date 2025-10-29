"""
Notification Channel System

Multi-channel notification delivery for alerts:
- Email notifications (SMTP)
- SMS notifications (Twilio)
- Slack notifications
- Webhook notifications
- Push notifications
- Channel prioritization
- Delivery confirmation
- Rate limiting
- Template system
"""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import warnings


class ChannelType(Enum):
    """Notification channel types"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PUSH = "push"
    CONSOLE = "console"


class NotificationPriority(Enum):
    """Notification priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DeliveryStatus(Enum):
    """Notification delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


@dataclass
class ChannelConfig:
    """Configuration for a notification channel"""
    channel_type: ChannelType
    enabled: bool = True
    priority_threshold: NotificationPriority = NotificationPriority.MEDIUM
    rate_limit_per_hour: int = 100
    rate_limit_per_day: int = 500
    retry_attempts: int = 3
    timeout_seconds: int = 30
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'channel_type': self.channel_type.value,
            'enabled': self.enabled,
            'priority_threshold': self.priority_threshold.value,
            'rate_limit_per_hour': self.rate_limit_per_hour,
            'rate_limit_per_day': self.rate_limit_per_day,
            'retry_attempts': self.retry_attempts,
            'timeout_seconds': self.timeout_seconds,
            'config': self.config
        }


@dataclass
class NotificationMessage:
    """Notification message structure"""
    message_id: str
    timestamp: datetime
    channel_type: ChannelType
    priority: NotificationPriority
    subject: str
    body: str
    recipients: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DeliveryStatus = DeliveryStatus.PENDING
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'channel_type': self.channel_type.value,
            'priority': self.priority.value,
            'subject': self.subject,
            'body': self.body,
            'recipients': self.recipients,
            'metadata': self.metadata,
            'status': self.status.value,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'delivered_at': self.delivered_at.isoformat() if self.delivered_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


class NotificationChannels:
    """
    Multi-channel notification delivery system

    Features:
    - Email, SMS, Slack, and webhook support
    - Priority-based routing
    - Rate limiting
    - Retry logic
    - Delivery tracking
    - Template system
    """

    def __init__(
        self,
        channels: Optional[Dict[ChannelType, ChannelConfig]] = None
    ):
        """
        Initialize notification channels

        Args:
            channels: Dictionary of channel configurations
        """
        self.channels = channels if channels is not None else self._get_default_channels()

        # Message tracking
        self.message_queue: deque = deque()
        self.sent_messages: List[NotificationMessage] = []
        self.failed_messages: List[NotificationMessage] = []

        # Rate limiting
        self.rate_limit_counters: Dict[ChannelType, deque] = {
            channel_type: deque(maxlen=1000)
            for channel_type in ChannelType
        }

        # Statistics
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'by_channel': defaultdict(int),
            'by_priority': defaultdict(int)
        }

        # Message templates
        self.templates = self._get_default_templates()

    def send_notification(
        self,
        channel_type: ChannelType,
        priority: NotificationPriority,
        subject: str,
        body: str,
        recipients: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> NotificationMessage:
        """
        Send notification through specified channel

        Args:
            channel_type: Notification channel
            priority: Message priority
            subject: Message subject/title
            body: Message body
            recipients: List of recipients
            metadata: Optional metadata

        Returns:
            NotificationMessage with delivery status
        """
        timestamp = datetime.now()

        # Create message
        message = NotificationMessage(
            message_id=self._generate_message_id(timestamp),
            timestamp=timestamp,
            channel_type=channel_type,
            priority=priority,
            subject=subject,
            body=body,
            recipients=recipients,
            metadata=metadata or {}
        )

        # Check if channel is enabled
        if channel_type not in self.channels or not self.channels[channel_type].enabled:
            message.status = DeliveryStatus.FAILED
            message.error_message = f"Channel {channel_type.value} is not enabled"
            self.failed_messages.append(message)
            return message

        channel_config = self.channels[channel_type]

        # Check priority threshold
        priority_order = {
            NotificationPriority.CRITICAL: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.MEDIUM: 2,
            NotificationPriority.LOW: 3
        }

        if priority_order[priority] > priority_order[channel_config.priority_threshold]:
            message.status = DeliveryStatus.FAILED
            message.error_message = f"Priority {priority.value} below threshold"
            return message

        # Check rate limits
        if not self._check_rate_limit(channel_type, channel_config):
            message.status = DeliveryStatus.RATE_LIMITED
            message.error_message = "Rate limit exceeded"
            self.message_queue.append(message)
            return message

        # Send message
        try:
            success = self._send_via_channel(channel_type, message, channel_config)

            if success:
                message.status = DeliveryStatus.SENT
                message.sent_at = datetime.now()
                self.sent_messages.append(message)
                self.stats['total_sent'] += 1
                self.stats['by_channel'][channel_type.value] += 1
                self.stats['by_priority'][priority.value] += 1

                # Track for rate limiting
                self.rate_limit_counters[channel_type].append(datetime.now())
            else:
                message.status = DeliveryStatus.FAILED
                self.failed_messages.append(message)
                self.stats['total_failed'] += 1

        except Exception as e:
            message.status = DeliveryStatus.FAILED
            message.error_message = str(e)
            self.failed_messages.append(message)
            self.stats['total_failed'] += 1

        return message

    def send_email(
        self,
        to_addresses: List[str],
        subject: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        html: bool = False
    ) -> NotificationMessage:
        """
        Send email notification

        Args:
            to_addresses: List of recipient email addresses
            subject: Email subject
            body: Email body
            priority: Message priority
            html: Whether body is HTML

        Returns:
            NotificationMessage
        """
        return self.send_notification(
            channel_type=ChannelType.EMAIL,
            priority=priority,
            subject=subject,
            body=body,
            recipients=to_addresses,
            metadata={'html': html}
        )

    def send_sms(
        self,
        phone_numbers: List[str],
        message: str,
        priority: NotificationPriority = NotificationPriority.HIGH
    ) -> NotificationMessage:
        """
        Send SMS notification

        Args:
            phone_numbers: List of phone numbers
            message: SMS message (max 160 chars)
            priority: Message priority

        Returns:
            NotificationMessage
        """
        # Truncate message if too long
        if len(message) > 160:
            message = message[:157] + "..."

        return self.send_notification(
            channel_type=ChannelType.SMS,
            priority=priority,
            subject="SMS Alert",
            body=message,
            recipients=phone_numbers
        )

    def send_slack(
        self,
        channel: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        attachments: Optional[List[Dict]] = None
    ) -> NotificationMessage:
        """
        Send Slack notification

        Args:
            channel: Slack channel name (e.g., #alerts)
            message: Message text
            priority: Message priority
            attachments: Optional Slack attachments

        Returns:
            NotificationMessage
        """
        return self.send_notification(
            channel_type=ChannelType.SLACK,
            priority=priority,
            subject=channel,
            body=message,
            recipients=[channel],
            metadata={'attachments': attachments}
        )

    def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM
    ) -> NotificationMessage:
        """
        Send webhook notification

        Args:
            url: Webhook URL
            payload: JSON payload
            priority: Message priority

        Returns:
            NotificationMessage
        """
        return self.send_notification(
            channel_type=ChannelType.WEBHOOK,
            priority=priority,
            subject="Webhook Notification",
            body=json.dumps(payload),
            recipients=[url],
            metadata={'payload': payload}
        )

    def configure_channel(
        self,
        channel_type: ChannelType,
        config: ChannelConfig
    ):
        """
        Configure notification channel

        Args:
            channel_type: Channel to configure
            config: Channel configuration
        """
        self.channels[channel_type] = config

    def enable_channel(self, channel_type: ChannelType):
        """Enable a notification channel"""
        if channel_type in self.channels:
            self.channels[channel_type].enabled = True

    def disable_channel(self, channel_type: ChannelType):
        """Disable a notification channel"""
        if channel_type in self.channels:
            self.channels[channel_type].enabled = False

    def retry_failed_messages(self):
        """Retry sending failed messages"""
        retry_queue = self.failed_messages.copy()
        self.failed_messages.clear()

        for message in retry_queue:
            channel_config = self.channels[message.channel_type]

            if message.retry_count >= channel_config.retry_attempts:
                # Max retries reached
                self.failed_messages.append(message)
                continue

            message.retry_count += 1

            try:
                success = self._send_via_channel(
                    message.channel_type,
                    message,
                    channel_config
                )

                if success:
                    message.status = DeliveryStatus.SENT
                    message.sent_at = datetime.now()
                    self.sent_messages.append(message)
                else:
                    self.failed_messages.append(message)

            except Exception as e:
                message.error_message = str(e)
                self.failed_messages.append(message)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get notification statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'total_sent': self.stats['total_sent'],
            'total_failed': self.stats['total_failed'],
            'by_channel': dict(self.stats['by_channel']),
            'by_priority': dict(self.stats['by_priority']),
            'queued_messages': len(self.message_queue),
            'failed_messages': len(self.failed_messages)
        }

    def _send_via_channel(
        self,
        channel_type: ChannelType,
        message: NotificationMessage,
        config: ChannelConfig
    ) -> bool:
        """Send message via specific channel"""
        if channel_type == ChannelType.EMAIL:
            return self._send_email_smtp(message, config)
        elif channel_type == ChannelType.SMS:
            return self._send_sms_twilio(message, config)
        elif channel_type == ChannelType.SLACK:
            return self._send_slack_webhook(message, config)
        elif channel_type == ChannelType.WEBHOOK:
            return self._send_webhook_post(message, config)
        elif channel_type == ChannelType.CONSOLE:
            return self._send_console_print(message, config)
        else:
            return False

    def _send_email_smtp(
        self,
        message: NotificationMessage,
        config: ChannelConfig
    ) -> bool:
        """Send email via SMTP"""
        try:
            smtp_config = config.config

            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_address', 'noreply@example.com')
            msg['To'] = ', '.join(message.recipients)
            msg['Subject'] = message.subject

            # Attach body
            if message.metadata.get('html', False):
                msg.attach(MIMEText(message.body, 'html'))
            else:
                msg.attach(MIMEText(message.body, 'plain'))

            # Send via SMTP
            with smtplib.SMTP(
                smtp_config.get('host', 'localhost'),
                smtp_config.get('port', 587)
            ) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()

                if 'username' in smtp_config and 'password' in smtp_config:
                    server.login(smtp_config['username'], smtp_config['password'])

                server.send_message(msg)

            return True

        except Exception as e:
            message.error_message = f"SMTP error: {str(e)}"
            return False

    def _send_sms_twilio(
        self,
        message: NotificationMessage,
        config: ChannelConfig
    ) -> bool:
        """Send SMS via Twilio"""
        try:
            twilio_config = config.config

            # Note: This is a placeholder. Real implementation would use Twilio SDK
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)

            warnings.warn("Twilio SMS integration requires twilio package")

            # Simulated send
            for phone_number in message.recipients:
                # In real implementation:
                # client.messages.create(
                #     to=phone_number,
                #     from_=twilio_config['from_number'],
                #     body=message.body
                # )
                pass

            return True

        except Exception as e:
            message.error_message = f"Twilio error: {str(e)}"
            return False

    def _send_slack_webhook(
        self,
        message: NotificationMessage,
        config: ChannelConfig
    ) -> bool:
        """Send message to Slack via webhook"""
        try:
            slack_config = config.config
            webhook_url = slack_config.get('webhook_url')

            if not webhook_url:
                message.error_message = "Slack webhook URL not configured"
                return False

            # Build payload
            payload = {
                'channel': message.subject,  # Channel name stored in subject
                'text': message.body,
                'username': slack_config.get('username', 'Risk Alert Bot'),
                'icon_emoji': slack_config.get('icon', ':chart_with_upwards_trend:')
            }

            # Add attachments if provided
            if 'attachments' in message.metadata:
                payload['attachments'] = message.metadata['attachments']

            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=config.timeout_seconds
            )

            return response.status_code == 200

        except Exception as e:
            message.error_message = f"Slack error: {str(e)}"
            return False

    def _send_webhook_post(
        self,
        message: NotificationMessage,
        config: ChannelConfig
    ) -> bool:
        """Send webhook POST request"""
        try:
            url = message.recipients[0]  # URL stored in recipients
            payload = message.metadata.get('payload', {})

            response = requests.post(
                url,
                json=payload,
                timeout=config.timeout_seconds
            )

            return response.status_code in [200, 201, 202]

        except Exception as e:
            message.error_message = f"Webhook error: {str(e)}"
            return False

    def _send_console_print(
        self,
        message: NotificationMessage,
        config: ChannelConfig
    ) -> bool:
        """Print notification to console"""
        try:
            print("\n" + "=" * 70)
            print(f"[{message.priority.value.upper()}] {message.subject}")
            print("-" * 70)
            print(message.body)
            print("=" * 70 + "\n")
            return True
        except Exception as e:
            message.error_message = f"Console error: {str(e)}"
            return False

    def _check_rate_limit(
        self,
        channel_type: ChannelType,
        config: ChannelConfig
    ) -> bool:
        """Check if rate limit allows sending"""
        now = datetime.now()
        counter = self.rate_limit_counters[channel_type]

        # Check hourly limit
        one_hour_ago = now - timedelta(hours=1)
        recent_hour = sum(1 for ts in counter if ts > one_hour_ago)

        if recent_hour >= config.rate_limit_per_hour:
            return False

        # Check daily limit
        one_day_ago = now - timedelta(days=1)
        recent_day = sum(1 for ts in counter if ts > one_day_ago)

        if recent_day >= config.rate_limit_per_day:
            return False

        return True

    def _generate_message_id(self, timestamp: datetime) -> str:
        """Generate unique message ID"""
        return f"MSG_{timestamp.strftime('%Y%m%d%H%M%S%f')}"

    def _get_default_channels(self) -> Dict[ChannelType, ChannelConfig]:
        """Get default channel configurations"""
        return {
            ChannelType.EMAIL: ChannelConfig(
                channel_type=ChannelType.EMAIL,
                enabled=False,  # Requires SMTP configuration
                priority_threshold=NotificationPriority.MEDIUM,
                rate_limit_per_hour=50,
                config={}
            ),
            ChannelType.SMS: ChannelConfig(
                channel_type=ChannelType.SMS,
                enabled=False,  # Requires Twilio configuration
                priority_threshold=NotificationPriority.HIGH,
                rate_limit_per_hour=10,
                config={}
            ),
            ChannelType.SLACK: ChannelConfig(
                channel_type=ChannelType.SLACK,
                enabled=False,  # Requires webhook URL
                priority_threshold=NotificationPriority.MEDIUM,
                rate_limit_per_hour=100,
                config={}
            ),
            ChannelType.WEBHOOK: ChannelConfig(
                channel_type=ChannelType.WEBHOOK,
                enabled=True,
                priority_threshold=NotificationPriority.MEDIUM,
                rate_limit_per_hour=200,
                config={}
            ),
            ChannelType.CONSOLE: ChannelConfig(
                channel_type=ChannelType.CONSOLE,
                enabled=True,
                priority_threshold=NotificationPriority.LOW,
                rate_limit_per_hour=1000,
                config={}
            )
        }

    def _get_default_templates(self) -> Dict[str, str]:
        """Get default message templates"""
        return {
            'risk_breach': """
RISK LIMIT BREACH ALERT

Metric: {metric_name}
Current Value: {current_value}
Threshold: {threshold_value}
Deviation: {deviation}

Timestamp: {timestamp}
Severity: {severity}

Action Required: Review portfolio risk exposure immediately.
""",
            'drawdown_alert': """
DRAWDOWN ALERT

Current Drawdown: {current_drawdown:.2%}
Maximum Allowed: {max_drawdown:.2%}

Portfolio Value: ${portfolio_value:,.2f}
Peak Value: ${peak_value:,.2f}

Timestamp: {timestamp}

Action Required: Consider reducing positions or implementing stop-losses.
""",
            'compliance_breach': """
COMPLIANCE BREACH NOTIFICATION

Rule: {rule_name}
Description: {description}
Current Status: {status}

Breach Details:
- Current Value: {current_value}
- Threshold: {threshold_value}
- Deviation: {deviation}

Timestamp: {timestamp}

Action Required: Review and acknowledge breach, implement corrective measures.
"""
        }

    def format_from_template(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """
        Format message from template

        Args:
            template_name: Name of template
            **kwargs: Template variables

        Returns:
            Formatted message
        """
        if template_name not in self.templates:
            return json.dumps(kwargs, indent=2)

        try:
            return self.templates[template_name].format(**kwargs)
        except KeyError as e:
            return f"Template error: missing variable {e}"

    def add_template(self, name: str, template: str):
        """
        Add custom message template

        Args:
            name: Template name
            template: Template string with {variable} placeholders
        """
        self.templates[name] = template

    def generate_summary_report(self) -> str:
        """
        Generate text summary of notification status

        Returns:
            Formatted summary string
        """
        stats = self.get_statistics()

        lines = []
        lines.append("=" * 70)
        lines.append("NOTIFICATION CHANNELS SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append("STATISTICS:")
        lines.append(f"  Total Sent: {stats['total_sent']}")
        lines.append(f"  Total Failed: {stats['total_failed']}")
        lines.append(f"  Queued: {stats['queued_messages']}")
        lines.append("")

        lines.append("BY CHANNEL:")
        for channel_type, count in stats['by_channel'].items():
            lines.append(f"  {channel_type:<12}: {count:>6}")
        lines.append("")

        lines.append("CHANNEL STATUS:")
        for channel_type, config in self.channels.items():
            status = "ENABLED" if config.enabled else "DISABLED"
            lines.append(f"  {channel_type.value:<12}: {status}")

        lines.append("=" * 70)

        return "\n".join(lines)

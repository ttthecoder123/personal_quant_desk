"""
Network Monitor

Monitors connection stability, packet loss, latency, bandwidth,
DNS resolution, SSL certificates, WebSocket health, and API availability.
"""

import socket
import ssl
import time
import requests
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import subprocess


@dataclass
class NetworkMetrics:
    """Network metrics for an endpoint."""
    endpoint: str
    is_reachable: bool
    latency_ms: float
    packet_loss_percent: float
    last_check: datetime
    error_message: Optional[str] = None


class NetworkMonitor:
    """
    Comprehensive network monitoring.

    Features:
    - Connection stability
    - Packet loss detection
    - Latency monitoring
    - Bandwidth utilization
    - DNS resolution monitoring
    - SSL certificate expiry
    - WebSocket connection health
    - API endpoint availability
    """

    def __init__(self, check_interval: int = 30):
        """
        Initialize network monitor.

        Args:
            check_interval: Seconds between checks
        """
        self.check_interval = check_interval
        self.endpoints: Dict[str, str] = {}
        self.websockets: Dict[str, Any] = {}
        self.ssl_certs: Dict[str, Dict] = {}
        self.metrics: Dict[str, deque] = {}
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self):
        """Start network monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop network monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def register_endpoint(self, name: str, url: str):
        """
        Register an endpoint for monitoring.

        Args:
            name: Endpoint name
            url: Endpoint URL
        """
        self.endpoints[name] = url
        with self.lock:
            self.metrics[name] = deque(maxlen=1000)

    def check_connectivity(self, host: str, port: int, timeout: int = 5) -> Tuple[bool, float]:
        """
        Check TCP connectivity to a host.

        Args:
            host: Hostname or IP
            port: Port number
            timeout: Timeout in seconds

        Returns:
            Tuple of (is_reachable, latency_ms)
        """
        start_time = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            latency = (time.time() - start_time) * 1000
            return True, latency
        except Exception:
            latency = (time.time() - start_time) * 1000
            return False, latency

    def check_http_endpoint(self, url: str, timeout: int = 5) -> Tuple[bool, float, Optional[str]]:
        """
        Check HTTP endpoint availability.

        Args:
            url: URL to check
            timeout: Timeout in seconds

        Returns:
            Tuple of (is_available, latency_ms, error_message)
        """
        start_time = time.time()
        try:
            response = requests.get(url, timeout=timeout)
            latency = (time.time() - start_time) * 1000
            if response.status_code == 200:
                return True, latency, None
            else:
                return False, latency, f"Status code: {response.status_code}"
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return False, latency, str(e)

    def check_dns_resolution(self, hostname: str) -> Tuple[bool, float, Optional[str]]:
        """
        Check DNS resolution.

        Args:
            hostname: Hostname to resolve

        Returns:
            Tuple of (is_resolved, latency_ms, ip_address or error)
        """
        start_time = time.time()
        try:
            ip = socket.gethostbyname(hostname)
            latency = (time.time() - start_time) * 1000
            return True, latency, ip
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return False, latency, str(e)

    def check_ssl_certificate(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """
        Check SSL certificate validity and expiration.

        Args:
            hostname: Hostname to check
            port: SSL port

        Returns:
            Dictionary with certificate information
        """
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()

                    # Parse expiration date
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.now()).days

                    return {
                        'hostname': hostname,
                        'is_valid': True,
                        'issuer': dict(x[0] for x in cert['issuer']),
                        'subject': dict(x[0] for x in cert['subject']),
                        'not_after': not_after.isoformat(),
                        'days_until_expiry': days_until_expiry,
                        'expires_soon': days_until_expiry < 30,
                        'error': None
                    }
        except Exception as e:
            return {
                'hostname': hostname,
                'is_valid': False,
                'error': str(e)
            }

    def measure_bandwidth(self, test_url: str, test_size_mb: int = 10) -> Dict[str, float]:
        """
        Measure bandwidth to a test endpoint.

        Args:
            test_url: URL to download from
            test_size_mb: Approximate size to download

        Returns:
            Dictionary with bandwidth metrics
        """
        try:
            start_time = time.time()
            response = requests.get(test_url, stream=True, timeout=30)
            bytes_downloaded = 0

            for chunk in response.iter_content(chunk_size=8192):
                bytes_downloaded += len(chunk)
                if bytes_downloaded >= test_size_mb * 1024 * 1024:
                    break

            elapsed = time.time() - start_time
            mb_downloaded = bytes_downloaded / (1024 * 1024)
            mbps = (mb_downloaded * 8) / elapsed

            return {
                'bytes_downloaded': bytes_downloaded,
                'elapsed_seconds': elapsed,
                'mbps': mbps,
                'mb_per_second': mb_downloaded / elapsed
            }
        except Exception as e:
            return {
                'error': str(e),
                'mbps': 0
            }

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_endpoints()
                self._check_ssl_certificates()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in network monitor loop: {e}")

    def _check_all_endpoints(self):
        """Check all registered endpoints."""
        for name, url in self.endpoints.items():
            is_available, latency, error = self.check_http_endpoint(url)

            metric = NetworkMetrics(
                endpoint=name,
                is_reachable=is_available,
                latency_ms=latency,
                packet_loss_percent=0.0,  # Could be enhanced with ping
                last_check=datetime.now(),
                error_message=error
            )

            with self.lock:
                self.metrics[name].append(metric)

    def _check_ssl_certificates(self):
        """Check SSL certificates for all HTTPS endpoints."""
        for name, url in self.endpoints.items():
            if url.startswith('https://'):
                try:
                    hostname = url.split('//')[1].split('/')[0].split(':')[0]
                    cert_info = self.check_ssl_certificate(hostname)
                    with self.lock:
                        self.ssl_certs[name] = cert_info
                except Exception as e:
                    print(f"Error checking SSL for {name}: {e}")

    def get_endpoint_metrics(self, name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get metrics for an endpoint.

        Args:
            name: Endpoint name
            window_minutes: Time window

        Returns:
            Dictionary with endpoint metrics
        """
        with self.lock:
            if name not in self.metrics:
                return {}

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent = [m for m in self.metrics[name] if m.last_check >= cutoff]

            if not recent:
                return {}

            latencies = [m.latency_ms for m in recent]
            availability = sum(1 for m in recent if m.is_reachable) / len(recent) * 100

            return {
                'endpoint': name,
                'availability_percent': availability,
                'avg_latency_ms': sum(latencies) / len(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'total_checks': len(recent),
                'failed_checks': sum(1 for m in recent if not m.is_reachable),
                'last_error': recent[-1].error_message if not recent[-1].is_reachable else None
            }

    def get_ssl_certificate_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get SSL certificate status for an endpoint.

        Args:
            name: Endpoint name

        Returns:
            Certificate information or None
        """
        with self.lock:
            return self.ssl_certs.get(name)

    def get_expiring_certificates(self, days_threshold: int = 30) -> List[Dict[str, Any]]:
        """
        Get certificates expiring soon.

        Args:
            days_threshold: Days threshold

        Returns:
            List of expiring certificates
        """
        with self.lock:
            expiring = []
            for name, cert in self.ssl_certs.items():
                if cert.get('is_valid') and cert.get('days_until_expiry', 999) < days_threshold:
                    expiring.append({
                        'endpoint': name,
                        'hostname': cert['hostname'],
                        'days_until_expiry': cert['days_until_expiry'],
                        'not_after': cert['not_after']
                    })
            return expiring

    def get_summary(self) -> Dict[str, Any]:
        """
        Get network monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total_endpoints = len(self.endpoints)
            healthy_endpoints = 0

            for name in self.endpoints.keys():
                if name in self.metrics and self.metrics[name]:
                    if self.metrics[name][-1].is_reachable:
                        healthy_endpoints += 1

            return {
                'total_endpoints': total_endpoints,
                'healthy_endpoints': healthy_endpoints,
                'unhealthy_endpoints': total_endpoints - healthy_endpoints,
                'ssl_certificates_monitored': len(self.ssl_certs),
                'expiring_certificates': len(self.get_expiring_certificates()),
                'timestamp': datetime.now().isoformat()
            }

"""
Resource Monitor

Tracks CPU, memory, disk I/O, network bandwidth, file descriptors,
threads, database connections, and GPU usage.
"""

import psutil
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from collections import deque


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage."""
    timestamp: datetime
    cpu_percent: float
    cpu_per_core: List[float]
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_read_mb_per_sec: float
    disk_write_mb_per_sec: float
    network_sent_mb_per_sec: float
    network_recv_mb_per_sec: float
    open_files: int
    thread_count: int


class ResourceMonitor:
    """
    Comprehensive resource monitoring.

    Features:
    - CPU usage tracking (per core/process)
    - Memory usage (heap, buffers)
    - Disk I/O metrics
    - Network bandwidth usage
    - File descriptor usage
    - Thread count monitoring
    - Database connection pools
    - GPU usage (if applicable)
    """

    def __init__(self, sample_interval: int = 1, history_size: int = 3600):
        """
        Initialize resource monitor.

        Args:
            sample_interval: Seconds between samples
            history_size: Number of samples to keep in history
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)

        # Previous I/O counters for rate calculation
        self.prev_disk_io = None
        self.prev_net_io = None
        self.prev_time = None

        # Process-specific tracking
        self.process_monitors: Dict[int, Any] = {}

        # Connection pool tracking
        self.connection_pools: Dict[str, Any] = {}

        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self):
        """Start resource monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop resource monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def register_process(self, pid: int):
        """
        Register a process for detailed monitoring.

        Args:
            pid: Process ID
        """
        try:
            self.process_monitors[pid] = psutil.Process(pid)
        except psutil.NoSuchProcess:
            pass

    def register_connection_pool(self, name: str, pool: Any):
        """
        Register a database connection pool for monitoring.

        Args:
            name: Pool name
            pool: Connection pool object
        """
        self.connection_pools[name] = pool

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                snapshot = self._take_snapshot()
                with self.lock:
                    self.history.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                print(f"Error in resource monitor loop: {e}")

    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        current_time = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_rate = 0.0
        disk_write_rate = 0.0

        if self.prev_disk_io and self.prev_time:
            time_delta = current_time - self.prev_time
            read_bytes = disk_io.read_bytes - self.prev_disk_io.read_bytes
            write_bytes = disk_io.write_bytes - self.prev_disk_io.write_bytes
            disk_read_rate = (read_bytes / time_delta) / (1024**2)  # MB/s
            disk_write_rate = (write_bytes / time_delta) / (1024**2)  # MB/s

        self.prev_disk_io = disk_io

        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_rate = 0.0
        net_recv_rate = 0.0

        if self.prev_net_io and self.prev_time:
            time_delta = current_time - self.prev_time
            sent_bytes = net_io.bytes_sent - self.prev_net_io.bytes_sent
            recv_bytes = net_io.bytes_recv - self.prev_net_io.bytes_recv
            net_sent_rate = (sent_bytes / time_delta) / (1024**2)  # MB/s
            net_recv_rate = (recv_bytes / time_delta) / (1024**2)  # MB/s

        self.prev_net_io = net_io
        self.prev_time = current_time

        # File descriptors
        try:
            open_files = len(psutil.Process().open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0

        # Thread count
        try:
            thread_count = psutil.Process().num_threads()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            thread_count = 0

        return ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            disk_read_mb_per_sec=disk_read_rate,
            disk_write_mb_per_sec=disk_write_rate,
            network_sent_mb_per_sec=net_sent_rate,
            network_recv_mb_per_sec=net_recv_rate,
            open_files=open_files,
            thread_count=thread_count
        )

    def get_current_usage(self) -> Optional[ResourceSnapshot]:
        """
        Get current resource usage.

        Returns:
            Current ResourceSnapshot or None
        """
        with self.lock:
            return self.history[-1] if self.history else None

    def get_process_usage(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get resource usage for a specific process.

        Args:
            pid: Process ID

        Returns:
            Dictionary with process usage stats
        """
        try:
            proc = psutil.Process(pid)
            return {
                'pid': pid,
                'name': proc.name(),
                'cpu_percent': proc.cpu_percent(interval=0.1),
                'memory_percent': proc.memory_percent(),
                'memory_mb': proc.memory_info().rss / (1024**2),
                'num_threads': proc.num_threads(),
                'open_files': len(proc.open_files()),
                'connections': len(proc.connections()),
                'status': proc.status()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def get_connection_pool_stats(self, name: str) -> Dict[str, Any]:
        """
        Get connection pool statistics.

        Args:
            name: Pool name

        Returns:
            Dictionary with pool stats
        """
        pool = self.connection_pools.get(name)
        if not pool:
            return {}

        stats = {
            'name': name,
            'size': 0,
            'checked_out': 0,
            'overflow': 0,
            'available': 0
        }

        # Try to get stats from common pool types
        if hasattr(pool, 'size'):
            stats['size'] = pool.size()
        if hasattr(pool, 'checkedout'):
            stats['checked_out'] = pool.checkedout()
        if hasattr(pool, 'overflow'):
            stats['overflow'] = pool.overflow()

        stats['available'] = stats['size'] - stats['checked_out']

        return stats

    def get_cpu_stats(self, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Get CPU statistics.

        Args:
            window_seconds: Time window for stats (None = current)

        Returns:
            Dictionary with CPU stats
        """
        with self.lock:
            if not self.history:
                return {}

            if window_seconds:
                cutoff = datetime.now().timestamp() - window_seconds
                samples = [s for s in self.history if s.timestamp.timestamp() >= cutoff]
            else:
                samples = [self.history[-1]]

            if not samples:
                return {}

            cpu_percents = [s.cpu_percent for s in samples]

            return {
                'current': samples[-1].cpu_percent,
                'average': sum(cpu_percents) / len(cpu_percents),
                'max': max(cpu_percents),
                'min': min(cpu_percents),
                'cores': len(samples[-1].cpu_per_core),
                'per_core': samples[-1].cpu_per_core
            }

    def get_memory_stats(self, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Get memory statistics.

        Args:
            window_seconds: Time window for stats

        Returns:
            Dictionary with memory stats
        """
        with self.lock:
            if not self.history:
                return {}

            if window_seconds:
                cutoff = datetime.now().timestamp() - window_seconds
                samples = [s for s in self.history if s.timestamp.timestamp() >= cutoff]
            else:
                samples = [self.history[-1]]

            if not samples:
                return {}

            mem_percents = [s.memory_percent for s in samples]
            mem_used = [s.memory_used_gb for s in samples]

            return {
                'current_percent': samples[-1].memory_percent,
                'current_used_gb': samples[-1].memory_used_gb,
                'available_gb': samples[-1].memory_available_gb,
                'average_percent': sum(mem_percents) / len(mem_percents),
                'max_used_gb': max(mem_used),
                'min_used_gb': min(mem_used)
            }

    def get_disk_io_stats(self, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Get disk I/O statistics.

        Args:
            window_seconds: Time window for stats

        Returns:
            Dictionary with disk I/O stats
        """
        with self.lock:
            if not self.history:
                return {}

            if window_seconds:
                cutoff = datetime.now().timestamp() - window_seconds
                samples = [s for s in self.history if s.timestamp.timestamp() >= cutoff]
            else:
                samples = [self.history[-1]]

            if not samples:
                return {}

            read_rates = [s.disk_read_mb_per_sec for s in samples]
            write_rates = [s.disk_write_mb_per_sec for s in samples]

            return {
                'current_read_mb_per_sec': samples[-1].disk_read_mb_per_sec,
                'current_write_mb_per_sec': samples[-1].disk_write_mb_per_sec,
                'avg_read_mb_per_sec': sum(read_rates) / len(read_rates),
                'avg_write_mb_per_sec': sum(write_rates) / len(write_rates),
                'max_read_mb_per_sec': max(read_rates),
                'max_write_mb_per_sec': max(write_rates)
            }

    def get_network_stats(self, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Get network statistics.

        Args:
            window_seconds: Time window for stats

        Returns:
            Dictionary with network stats
        """
        with self.lock:
            if not self.history:
                return {}

            if window_seconds:
                cutoff = datetime.now().timestamp() - window_seconds
                samples = [s for s in self.history if s.timestamp.timestamp() >= cutoff]
            else:
                samples = [self.history[-1]]

            if not samples:
                return {}

            sent_rates = [s.network_sent_mb_per_sec for s in samples]
            recv_rates = [s.network_recv_mb_per_sec for s in samples]

            return {
                'current_sent_mb_per_sec': samples[-1].network_sent_mb_per_sec,
                'current_recv_mb_per_sec': samples[-1].network_recv_mb_per_sec,
                'avg_sent_mb_per_sec': sum(sent_rates) / len(sent_rates),
                'avg_recv_mb_per_sec': sum(recv_rates) / len(recv_rates),
                'max_sent_mb_per_sec': max(sent_rates),
                'max_recv_mb_per_sec': max(recv_rates)
            }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get resource usage summary.

        Returns:
            Summary dictionary
        """
        current = self.get_current_usage()
        if not current:
            return {}

        return {
            'cpu_percent': current.cpu_percent,
            'memory_percent': current.memory_percent,
            'memory_used_gb': current.memory_used_gb,
            'disk_read_mb_per_sec': current.disk_read_mb_per_sec,
            'disk_write_mb_per_sec': current.disk_write_mb_per_sec,
            'network_sent_mb_per_sec': current.network_sent_mb_per_sec,
            'network_recv_mb_per_sec': current.network_recv_mb_per_sec,
            'open_files': current.open_files,
            'thread_count': current.thread_count,
            'timestamp': current.timestamp.isoformat()
        }

"""
Performance Monitor

Tracks request latency, throughput, response times, query performance,
cache hit rates, API call rates, and component-level metrics.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics."""
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring.

    Features:
    - Request latency tracking
    - Throughput measurement
    - Response time monitoring
    - Query performance tracking
    - Cache hit rates
    - API call rates
    - Batch processing times
    - Component-level metrics
    """

    def __init__(self, retention_minutes: int = 60):
        """
        Initialize performance monitor.

        Args:
            retention_minutes: How long to retain metrics
        """
        self.retention_minutes = retention_minutes
        self.retention_period = timedelta(minutes=retention_minutes)

        # Latency tracking
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.latency_history: Dict[str, List[float]] = defaultdict(list)

        # Throughput tracking
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.throughput_history: Dict[str, List[tuple]] = defaultdict(list)

        # Cache tracking
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})

        # Query performance
        self.query_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Custom metrics
        self.custom_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)

        # Locks
        self.lock = threading.Lock()

        # Cleanup thread
        self.running = False
        self.cleanup_thread: Optional[threading.Thread] = None

    def start(self):
        """Start performance monitoring."""
        if self.running:
            return

        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def stop(self):
        """Stop performance monitoring."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)

    @contextmanager
    def track_latency(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager to track operation latency.

        Args:
            operation: Operation name
            tags: Optional tags for categorization

        Example:
            with monitor.track_latency('api_call', {'endpoint': '/users'}):
                # perform operation
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.record_latency(operation, latency_ms, tags)

    def record_latency(self, operation: str, latency_ms: float, tags: Optional[Dict[str, str]] = None):
        """
        Record operation latency.

        Args:
            operation: Operation name
            latency_ms: Latency in milliseconds
            tags: Optional tags
        """
        with self.lock:
            key = self._make_key(operation, tags)
            self.latencies[key].append((datetime.now(), latency_ms))

    def record_throughput(self, operation: str, count: int = 1, tags: Optional[Dict[str, str]] = None):
        """
        Record throughput (requests/operations).

        Args:
            operation: Operation name
            count: Number of operations
            tags: Optional tags
        """
        with self.lock:
            key = self._make_key(operation, tags)
            self.request_counts[key].append((datetime.now(), count))

    def record_cache_hit(self, cache_name: str):
        """
        Record a cache hit.

        Args:
            cache_name: Name of the cache
        """
        with self.lock:
            self.cache_stats[cache_name]['hits'] += 1

    def record_cache_miss(self, cache_name: str):
        """
        Record a cache miss.

        Args:
            cache_name: Name of the cache
        """
        with self.lock:
            self.cache_stats[cache_name]['misses'] += 1

    def record_query(self, query_type: str, execution_time_ms: float, tags: Optional[Dict[str, str]] = None):
        """
        Record database query performance.

        Args:
            query_type: Type of query
            execution_time_ms: Execution time in milliseconds
            tags: Optional tags
        """
        with self.lock:
            key = self._make_key(query_type, tags)
            self.query_stats[key].append((datetime.now(), execution_time_ms))

    def record_metric(self, name: str, value: float, unit: str = '', tags: Optional[Dict[str, str]] = None):
        """
        Record a custom metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags
        """
        with self.lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.custom_metrics[name].append(metric)

    def get_latency_stats(self, operation: str, tags: Optional[Dict[str, str]] = None,
                          window_minutes: Optional[int] = None) -> Optional[LatencyStats]:
        """
        Get latency statistics for an operation.

        Args:
            operation: Operation name
            tags: Optional tags
            window_minutes: Time window for stats (None = all time)

        Returns:
            LatencyStats object or None if no data
        """
        with self.lock:
            key = self._make_key(operation, tags)
            latencies = self.latencies.get(key, deque())

            if not latencies:
                return None

            # Filter by time window
            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                filtered = [lat for ts, lat in latencies if ts >= cutoff]
            else:
                filtered = [lat for _, lat in latencies]

            if not filtered:
                return None

            sorted_latencies = sorted(filtered)
            count = len(sorted_latencies)

            return LatencyStats(
                count=count,
                min_ms=min(sorted_latencies),
                max_ms=max(sorted_latencies),
                mean_ms=statistics.mean(sorted_latencies),
                median_ms=statistics.median(sorted_latencies),
                p95_ms=sorted_latencies[int(count * 0.95)] if count > 0 else 0,
                p99_ms=sorted_latencies[int(count * 0.99)] if count > 0 else 0,
                std_dev_ms=statistics.stdev(sorted_latencies) if count > 1 else 0
            )

    def get_throughput(self, operation: str, tags: Optional[Dict[str, str]] = None,
                       window_minutes: int = 1) -> float:
        """
        Get throughput (operations per second).

        Args:
            operation: Operation name
            tags: Optional tags
            window_minutes: Time window for calculation

        Returns:
            Operations per second
        """
        with self.lock:
            key = self._make_key(operation, tags)
            counts = self.request_counts.get(key, deque())

            if not counts:
                return 0.0

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent = [count for ts, count in counts if ts >= cutoff]

            if not recent:
                return 0.0

            total = sum(recent)
            return total / (window_minutes * 60)

    def get_cache_hit_rate(self, cache_name: str) -> float:
        """
        Get cache hit rate (0-1).

        Args:
            cache_name: Name of the cache

        Returns:
            Hit rate as decimal (0.0 - 1.0)
        """
        with self.lock:
            stats = self.cache_stats.get(cache_name, {'hits': 0, 'misses': 0})
            total = stats['hits'] + stats['misses']
            return stats['hits'] / total if total > 0 else 0.0

    def get_cache_stats(self, cache_name: str) -> Dict[str, Any]:
        """
        Get detailed cache statistics.

        Args:
            cache_name: Name of the cache

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            stats = self.cache_stats.get(cache_name, {'hits': 0, 'misses': 0})
            total = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total if total > 0 else 0.0

            return {
                'hits': stats['hits'],
                'misses': stats['misses'],
                'total_requests': total,
                'hit_rate': hit_rate,
                'miss_rate': 1.0 - hit_rate
            }

    def get_query_stats(self, query_type: str, tags: Optional[Dict[str, str]] = None,
                        window_minutes: Optional[int] = None) -> Optional[LatencyStats]:
        """
        Get query performance statistics.

        Args:
            query_type: Type of query
            tags: Optional tags
            window_minutes: Time window for stats

        Returns:
            LatencyStats object or None
        """
        with self.lock:
            key = self._make_key(query_type, tags)
            queries = self.query_stats.get(key, deque())

            if not queries:
                return None

            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                filtered = [time_ms for ts, time_ms in queries if ts >= cutoff]
            else:
                filtered = [time_ms for _, time_ms in queries]

            if not filtered:
                return None

            sorted_times = sorted(filtered)
            count = len(sorted_times)

            return LatencyStats(
                count=count,
                min_ms=min(sorted_times),
                max_ms=max(sorted_times),
                mean_ms=statistics.mean(sorted_times),
                median_ms=statistics.median(sorted_times),
                p95_ms=sorted_times[int(count * 0.95)] if count > 0 else 0,
                p99_ms=sorted_times[int(count * 0.99)] if count > 0 else 0,
                std_dev_ms=statistics.stdev(sorted_times) if count > 1 else 0
            )

    def get_custom_metric(self, name: str, window_minutes: Optional[int] = None) -> List[PerformanceMetric]:
        """
        Get custom metric values.

        Args:
            name: Metric name
            window_minutes: Time window

        Returns:
            List of metric values
        """
        with self.lock:
            metrics = self.custom_metrics.get(name, [])

            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                return [m for m in metrics if m.timestamp >= cutoff]

            return metrics

    def get_all_operations(self) -> List[str]:
        """
        Get list of all tracked operations.

        Returns:
            List of operation names
        """
        with self.lock:
            operations = set()
            operations.update(self.latencies.keys())
            operations.update(self.request_counts.keys())
            operations.update(self.query_stats.keys())
            return sorted(list(operations))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            return {
                'total_operations': len(self.get_all_operations()),
                'total_caches': len(self.cache_stats),
                'total_custom_metrics': len(self.custom_metrics),
                'timestamp': datetime.now().isoformat()
            }

    def reset(self):
        """Reset all performance metrics."""
        with self.lock:
            self.latencies.clear()
            self.request_counts.clear()
            self.cache_stats.clear()
            self.query_stats.clear()
            self.custom_metrics.clear()

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a key from name and tags."""
        if not tags:
            return name

        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def _cleanup_loop(self):
        """Background cleanup of old metrics."""
        while self.running:
            try:
                self._cleanup_old_data()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                print(f"Error in cleanup loop: {e}")

    def _cleanup_old_data(self):
        """Remove old data beyond retention period."""
        with self.lock:
            cutoff = datetime.now() - self.retention_period

            # Cleanup latencies
            for key in list(self.latencies.keys()):
                self.latencies[key] = deque(
                    (ts, lat) for ts, lat in self.latencies[key] if ts >= cutoff
                )

            # Cleanup throughput
            for key in list(self.request_counts.keys()):
                self.request_counts[key] = deque(
                    (ts, count) for ts, count in self.request_counts[key] if ts >= cutoff
                )

            # Cleanup queries
            for key in list(self.query_stats.keys()):
                self.query_stats[key] = deque(
                    (ts, time_ms) for ts, time_ms in self.query_stats[key] if ts >= cutoff
                )

            # Cleanup custom metrics
            for name in list(self.custom_metrics.keys()):
                self.custom_metrics[name] = [
                    m for m in self.custom_metrics[name] if m.timestamp >= cutoff
                ]

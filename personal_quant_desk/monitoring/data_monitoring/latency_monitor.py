"""
Latency Monitor

Feed-to-database latency, end-to-end data latency, processing pipeline latency,
inter-component latency, geographic latency, and peak vs average latency tracking.
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import numpy as np


@dataclass
class LatencyMeasurement:
    """Individual latency measurement."""
    component: str
    operation: str
    latency_ms: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Statistical summary of latency."""
    component: str
    operation: str
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    std_dev_ms: float
    window_start: datetime
    window_end: datetime


@dataclass
class EndToEndLatency:
    """End-to-end latency tracking."""
    trace_id: str
    components: List[Tuple[str, float]]  # (component, latency_ms)
    total_latency_ms: float
    start_time: datetime
    end_time: datetime


@dataclass
class ComponentLatency:
    """Latency between components."""
    source: str
    destination: str
    latency_ms: float
    timestamp: datetime


class LatencyMonitor:
    """
    Comprehensive latency monitoring.

    Features:
    - Feed-to-database latency tracking
    - End-to-end data path latency
    - Processing pipeline stage latencies
    - Inter-component communication latency
    - Geographic/regional latency
    - Peak vs average latency analysis
    - Percentile-based SLA tracking
    - Latency breakdown and attribution
    - Historical trend analysis
    """

    def __init__(self, retention_minutes: int = 60, sample_rate: float = 1.0):
        """
        Initialize latency monitor.

        Args:
            retention_minutes: How long to retain measurements
            sample_rate: Sampling rate (0-1, 1 = track all)
        """
        self.retention_minutes = retention_minutes
        self.retention_period = timedelta(minutes=retention_minutes)
        self.sample_rate = sample_rate

        # Latency measurements
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # End-to-end tracking
        self.e2e_traces: Dict[str, Dict[str, Any]] = {}
        self.e2e_completed: deque = deque(maxlen=1000)

        # Component-to-component latency
        self.component_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Geographic latency
        self.geo_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Pipeline stage latencies
        self.pipeline_latencies: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )

        # Peak tracking
        self.peak_latencies: Dict[str, float] = {}
        self.peak_times: Dict[str, datetime] = {}

        # Feed-to-database specific
        self.feed_to_db_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        self.cleanup_thread: Optional[threading.Thread] = None

    def start(self):
        """Start latency monitoring."""
        if self.running:
            return

        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def stop(self):
        """Stop latency monitoring."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)

    @contextmanager
    def measure(self, component: str, operation: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager to measure latency.

        Args:
            component: Component name
            operation: Operation name
            tags: Optional tags for categorization

        Example:
            with monitor.measure('database', 'query', {'table': 'trades'}):
                # perform operation
                pass
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.record_latency(component, operation, latency_ms, tags)

    def record_latency(self, component: str, operation: str, latency_ms: float,
                       tags: Optional[Dict[str, str]] = None):
        """
        Record a latency measurement.

        Args:
            component: Component name
            operation: Operation name
            latency_ms: Latency in milliseconds
            tags: Optional tags
        """
        # Sample if configured
        if self.sample_rate < 1.0 and np.random.random() > self.sample_rate:
            return

        with self.lock:
            key = self._make_key(component, operation, tags)
            timestamp = datetime.now()

            measurement = LatencyMeasurement(
                component=component,
                operation=operation,
                latency_ms=latency_ms,
                timestamp=timestamp,
                tags=tags or {}
            )

            self.latencies[key].append(measurement)

            # Track peak
            if key not in self.peak_latencies or latency_ms > self.peak_latencies[key]:
                self.peak_latencies[key] = latency_ms
                self.peak_times[key] = timestamp

    def record_feed_to_db_latency(self, feed_name: str, symbol: str, latency_ms: float):
        """
        Record feed-to-database latency.

        Args:
            feed_name: Name of the feed
            symbol: Symbol
            latency_ms: Latency in milliseconds
        """
        with self.lock:
            key = f"{feed_name}:{symbol}"
            self.feed_to_db_latencies[key].append((datetime.now(), latency_ms))

    def start_trace(self, trace_id: str):
        """
        Start an end-to-end latency trace.

        Args:
            trace_id: Unique trace identifier
        """
        with self.lock:
            self.e2e_traces[trace_id] = {
                'start_time': datetime.now(),
                'components': [],
                'last_timestamp': time.perf_counter()
            }

    def record_trace_component(self, trace_id: str, component: str):
        """
        Record a component in an end-to-end trace.

        Args:
            trace_id: Trace identifier
            component: Component name
        """
        with self.lock:
            if trace_id not in self.e2e_traces:
                return

            trace = self.e2e_traces[trace_id]
            current_time = time.perf_counter()
            latency_ms = (current_time - trace['last_timestamp']) * 1000

            trace['components'].append((component, latency_ms))
            trace['last_timestamp'] = current_time

    def end_trace(self, trace_id: str):
        """
        End an end-to-end latency trace.

        Args:
            trace_id: Trace identifier
        """
        with self.lock:
            if trace_id not in self.e2e_traces:
                return

            trace = self.e2e_traces[trace_id]
            end_time = datetime.now()

            total_latency = sum(lat for _, lat in trace['components'])

            e2e = EndToEndLatency(
                trace_id=trace_id,
                components=trace['components'],
                total_latency_ms=total_latency,
                start_time=trace['start_time'],
                end_time=end_time
            )

            self.e2e_completed.append(e2e)
            del self.e2e_traces[trace_id]

    def record_component_latency(self, source: str, destination: str, latency_ms: float):
        """
        Record inter-component latency.

        Args:
            source: Source component
            destination: Destination component
            latency_ms: Latency in milliseconds
        """
        with self.lock:
            key = f"{source}->{destination}"
            latency = ComponentLatency(
                source=source,
                destination=destination,
                latency_ms=latency_ms,
                timestamp=datetime.now()
            )
            self.component_latencies[key].append(latency)

    def record_geo_latency(self, region: str, latency_ms: float):
        """
        Record geographic latency.

        Args:
            region: Region name
            latency_ms: Latency in milliseconds
        """
        with self.lock:
            self.geo_latencies[region].append((datetime.now(), latency_ms))

    def record_pipeline_stage(self, pipeline: str, stage: str, latency_ms: float):
        """
        Record pipeline stage latency.

        Args:
            pipeline: Pipeline name
            stage: Stage name
            latency_ms: Latency in milliseconds
        """
        with self.lock:
            self.pipeline_latencies[pipeline][stage].append((datetime.now(), latency_ms))

    def get_latency_stats(self, component: str, operation: str,
                          tags: Optional[Dict[str, str]] = None,
                          window_minutes: Optional[int] = None) -> Optional[LatencyStats]:
        """
        Get latency statistics.

        Args:
            component: Component name
            operation: Operation name
            tags: Optional tags
            window_minutes: Time window (None = all time)

        Returns:
            LatencyStats or None if no data
        """
        with self.lock:
            key = self._make_key(component, operation, tags)
            measurements = self.latencies.get(key, deque())

            if not measurements:
                return None

            # Filter by time window
            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                measurements = [m for m in measurements if m.timestamp >= cutoff]

            if not measurements:
                return None

            latencies = [m.latency_ms for m in measurements]
            sorted_latencies = sorted(latencies)
            count = len(sorted_latencies)

            window_start = min(m.timestamp for m in measurements)
            window_end = max(m.timestamp for m in measurements)

            return LatencyStats(
                component=component,
                operation=operation,
                count=count,
                min_ms=min(sorted_latencies),
                max_ms=max(sorted_latencies),
                mean_ms=statistics.mean(sorted_latencies),
                median_ms=statistics.median(sorted_latencies),
                p50_ms=sorted_latencies[int(count * 0.50)] if count > 0 else 0,
                p95_ms=sorted_latencies[int(count * 0.95)] if count > 0 else 0,
                p99_ms=sorted_latencies[int(count * 0.99)] if count > 0 else 0,
                p999_ms=sorted_latencies[int(count * 0.999)] if count > 0 else 0,
                std_dev_ms=statistics.stdev(sorted_latencies) if count > 1 else 0,
                window_start=window_start,
                window_end=window_end
            )

    def get_feed_to_db_stats(self, feed_name: str, symbol: Optional[str] = None,
                             window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get feed-to-database latency statistics.

        Args:
            feed_name: Name of the feed
            symbol: Optional specific symbol
            window_minutes: Time window

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)

            if symbol:
                key = f"{feed_name}:{symbol}"
                data = self.feed_to_db_latencies.get(key, deque())
            else:
                # Aggregate all symbols for this feed
                data = []
                for key, values in self.feed_to_db_latencies.items():
                    if key.startswith(f"{feed_name}:"):
                        data.extend(values)

            # Filter by time window
            recent = [lat for ts, lat in data if ts >= cutoff]

            if not recent:
                return {}

            sorted_latencies = sorted(recent)
            count = len(sorted_latencies)

            return {
                'count': count,
                'min_ms': min(sorted_latencies),
                'max_ms': max(sorted_latencies),
                'mean_ms': statistics.mean(sorted_latencies),
                'median_ms': statistics.median(sorted_latencies),
                'p95_ms': sorted_latencies[int(count * 0.95)] if count > 0 else 0,
                'p99_ms': sorted_latencies[int(count * 0.99)] if count > 0 else 0
            }

    def get_e2e_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get end-to-end latency statistics.

        Args:
            window_minutes: Time window

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent = [e for e in self.e2e_completed if e.end_time >= cutoff]

            if not recent:
                return {}

            total_latencies = [e.total_latency_ms for e in recent]
            sorted_latencies = sorted(total_latencies)
            count = len(sorted_latencies)

            # Component breakdown
            component_times = defaultdict(list)
            for e2e in recent:
                for component, latency in e2e.components:
                    component_times[component].append(latency)

            component_stats = {}
            for component, latencies in component_times.items():
                component_stats[component] = {
                    'mean_ms': statistics.mean(latencies),
                    'max_ms': max(latencies),
                    'contribution_pct': (sum(latencies) / sum(total_latencies)) * 100
                }

            return {
                'count': count,
                'min_ms': min(sorted_latencies),
                'max_ms': max(sorted_latencies),
                'mean_ms': statistics.mean(sorted_latencies),
                'median_ms': statistics.median(sorted_latencies),
                'p95_ms': sorted_latencies[int(count * 0.95)] if count > 0 else 0,
                'p99_ms': sorted_latencies[int(count * 0.99)] if count > 0 else 0,
                'component_breakdown': component_stats
            }

    def get_component_latency_stats(self, source: str, destination: str,
                                   window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get inter-component latency statistics.

        Args:
            source: Source component
            destination: Destination component
            window_minutes: Time window

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            key = f"{source}->{destination}"
            data = self.component_latencies.get(key, deque())

            if not data:
                return {}

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent = [c.latency_ms for c in data if c.timestamp >= cutoff]

            if not recent:
                return {}

            sorted_latencies = sorted(recent)
            count = len(sorted_latencies)

            return {
                'count': count,
                'min_ms': min(sorted_latencies),
                'max_ms': max(sorted_latencies),
                'mean_ms': statistics.mean(sorted_latencies),
                'median_ms': statistics.median(sorted_latencies),
                'p95_ms': sorted_latencies[int(count * 0.95)] if count > 0 else 0,
                'p99_ms': sorted_latencies[int(count * 0.99)] if count > 0 else 0
            }

    def get_geo_latency_stats(self, region: str, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get geographic latency statistics.

        Args:
            region: Region name
            window_minutes: Time window

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            data = self.geo_latencies.get(region, deque())

            if not data:
                return {}

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent = [lat for ts, lat in data if ts >= cutoff]

            if not recent:
                return {}

            sorted_latencies = sorted(recent)
            count = len(sorted_latencies)

            return {
                'region': region,
                'count': count,
                'min_ms': min(sorted_latencies),
                'max_ms': max(sorted_latencies),
                'mean_ms': statistics.mean(sorted_latencies),
                'median_ms': statistics.median(sorted_latencies),
                'p95_ms': sorted_latencies[int(count * 0.95)] if count > 0 else 0,
                'p99_ms': sorted_latencies[int(count * 0.99)] if count > 0 else 0
            }

    def get_pipeline_stats(self, pipeline: str, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get pipeline latency statistics.

        Args:
            pipeline: Pipeline name
            window_minutes: Time window

        Returns:
            Dictionary with statistics per stage
        """
        with self.lock:
            if pipeline not in self.pipeline_latencies:
                return {}

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            stage_stats = {}

            for stage, data in self.pipeline_latencies[pipeline].items():
                recent = [lat for ts, lat in data if ts >= cutoff]

                if recent:
                    sorted_latencies = sorted(recent)
                    count = len(sorted_latencies)

                    stage_stats[stage] = {
                        'count': count,
                        'min_ms': min(sorted_latencies),
                        'max_ms': max(sorted_latencies),
                        'mean_ms': statistics.mean(sorted_latencies),
                        'p95_ms': sorted_latencies[int(count * 0.95)] if count > 0 else 0,
                        'p99_ms': sorted_latencies[int(count * 0.99)] if count > 0 else 0
                    }

            # Calculate total pipeline latency
            if stage_stats:
                total_mean = sum(s['mean_ms'] for s in stage_stats.values())
                total_p95 = sum(s['p95_ms'] for s in stage_stats.values())

                return {
                    'pipeline': pipeline,
                    'stages': stage_stats,
                    'total_mean_ms': total_mean,
                    'total_p95_ms': total_p95
                }

            return {}

    def get_peak_latencies(self, window_hours: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get peak latencies.

        Args:
            window_hours: Time window in hours (None = all time)

        Returns:
            Dictionary with peak latencies
        """
        with self.lock:
            peaks = {}

            cutoff = None
            if window_hours:
                cutoff = datetime.now() - timedelta(hours=window_hours)

            for key, peak_latency in self.peak_latencies.items():
                peak_time = self.peak_times[key]

                if cutoff and peak_time < cutoff:
                    continue

                peaks[key] = {
                    'peak_latency_ms': peak_latency,
                    'peak_time': peak_time.isoformat(),
                    'time_ago_seconds': (datetime.now() - peak_time).total_seconds()
                }

            return peaks

    def get_summary(self) -> Dict[str, Any]:
        """
        Get latency monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total_measurements = sum(len(measurements) for measurements in self.latencies.values())
            total_e2e_traces = len(self.e2e_completed)

            # Get overall average latency
            all_latencies = []
            for measurements in self.latencies.values():
                all_latencies.extend([m.latency_ms for m in measurements])

            avg_latency = statistics.mean(all_latencies) if all_latencies else 0

            return {
                'total_measurements': total_measurements,
                'total_e2e_traces': total_e2e_traces,
                'average_latency_ms': avg_latency,
                'monitored_operations': len(self.latencies),
                'timestamp': datetime.now().isoformat()
            }

    def _make_key(self, component: str, operation: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a key from component, operation, and tags."""
        key = f"{component}:{operation}"
        if tags:
            tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
            key = f"{key}[{tag_str}]"
        return key

    def _cleanup_loop(self):
        """Background cleanup of old measurements."""
        while self.running:
            try:
                self._cleanup_old_data()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                print(f"Error in cleanup loop: {e}")

    def _cleanup_old_data(self):
        """Remove old measurements beyond retention period."""
        with self.lock:
            cutoff = datetime.now() - self.retention_period

            # Cleanup latencies
            for key in list(self.latencies.keys()):
                filtered = deque(
                    (m for m in self.latencies[key] if m.timestamp >= cutoff),
                    maxlen=self.latencies[key].maxlen
                )
                self.latencies[key] = filtered

            # Cleanup feed-to-db latencies
            for key in list(self.feed_to_db_latencies.keys()):
                filtered = deque(
                    ((ts, lat) for ts, lat in self.feed_to_db_latencies[key] if ts >= cutoff),
                    maxlen=self.feed_to_db_latencies[key].maxlen
                )
                self.feed_to_db_latencies[key] = filtered

            # Cleanup geo latencies
            for region in list(self.geo_latencies.keys()):
                filtered = deque(
                    ((ts, lat) for ts, lat in self.geo_latencies[region] if ts >= cutoff),
                    maxlen=self.geo_latencies[region].maxlen
                )
                self.geo_latencies[region] = filtered

            # Cleanup pipeline latencies
            for pipeline in list(self.pipeline_latencies.keys()):
                for stage in list(self.pipeline_latencies[pipeline].keys()):
                    filtered = deque(
                        ((ts, lat) for ts, lat in self.pipeline_latencies[pipeline][stage] if ts >= cutoff),
                        maxlen=self.pipeline_latencies[pipeline][stage].maxlen
                    )
                    self.pipeline_latencies[pipeline][stage] = filtered

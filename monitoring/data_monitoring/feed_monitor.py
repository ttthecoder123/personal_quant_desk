"""
Feed Monitor

Real-time feed status monitoring, feed latency tracking, gap detection,
sequence number monitoring, heartbeat monitoring, symbol coverage tracking,
feed quality metrics, and backup feed switching.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum


class FeedStatus(Enum):
    """Feed connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    SWITCHING = "switching"
    BACKUP = "backup"


@dataclass
class FeedHealthMetric:
    """Feed health metrics."""
    feed_name: str
    status: FeedStatus
    last_message_time: datetime
    messages_received: int
    latency_ms: float
    gap_count: int
    sequence_errors: int
    heartbeat_missed: int
    symbols_active: int
    quality_score: float  # 0-100
    is_primary: bool
    timestamp: datetime


@dataclass
class FeedGap:
    """Data gap detection."""
    feed_name: str
    symbol: str
    gap_start: datetime
    gap_end: Optional[datetime]
    expected_sequence: int
    received_sequence: int
    gap_size: int
    recovered: bool = False


@dataclass
class SequenceTracker:
    """Sequence number tracking."""
    symbol: str
    last_sequence: int
    expected_sequence: int
    errors: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class HeartbeatTracker:
    """Heartbeat monitoring."""
    feed_name: str
    last_heartbeat: datetime
    expected_interval: timedelta
    missed_count: int = 0
    consecutive_misses: int = 0

    def is_alive(self) -> bool:
        """Check if feed is alive."""
        elapsed = datetime.now() - self.last_heartbeat
        return elapsed < (self.expected_interval * 2)


class FeedMonitor:
    """
    Comprehensive feed monitoring.

    Features:
    - Real-time feed status tracking
    - Feed latency monitoring
    - Gap detection and tracking
    - Sequence number validation
    - Heartbeat monitoring
    - Symbol coverage tracking
    - Feed quality metrics
    - Automatic backup feed switching
    - Historical feed statistics
    """

    def __init__(self, check_interval: int = 1):
        """
        Initialize feed monitor.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval

        # Feed status tracking
        self.feeds: Dict[str, FeedHealthMetric] = {}
        self.feed_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3600))

        # Latency tracking
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Gap detection
        self.gaps: Dict[str, List[FeedGap]] = defaultdict(list)
        self.active_gaps: Dict[str, FeedGap] = {}

        # Sequence tracking
        self.sequences: Dict[str, Dict[str, SequenceTracker]] = defaultdict(dict)

        # Heartbeat tracking
        self.heartbeats: Dict[str, HeartbeatTracker] = {}

        # Symbol coverage
        self.symbol_coverage: Dict[str, Set[str]] = defaultdict(set)
        self.expected_symbols: Dict[str, Set[str]] = defaultdict(set)

        # Feed configuration
        self.primary_feeds: Dict[str, str] = {}  # data_type -> feed_name
        self.backup_feeds: Dict[str, List[str]] = defaultdict(list)

        # Quality metrics
        self.quality_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start(self):
        """Start feed monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop feed monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def register_feed(self, feed_name: str, heartbeat_interval: int = 30,
                      is_primary: bool = True, data_type: str = 'default'):
        """
        Register a feed for monitoring.

        Args:
            feed_name: Name of the feed
            heartbeat_interval: Expected heartbeat interval in seconds
            is_primary: Whether this is a primary feed
            data_type: Type of data (quotes, trades, etc.)
        """
        with self.lock:
            self.feeds[feed_name] = FeedHealthMetric(
                feed_name=feed_name,
                status=FeedStatus.CONNECTED,
                last_message_time=datetime.now(),
                messages_received=0,
                latency_ms=0.0,
                gap_count=0,
                sequence_errors=0,
                heartbeat_missed=0,
                symbols_active=0,
                quality_score=100.0,
                is_primary=is_primary,
                timestamp=datetime.now()
            )

            self.heartbeats[feed_name] = HeartbeatTracker(
                feed_name=feed_name,
                last_heartbeat=datetime.now(),
                expected_interval=timedelta(seconds=heartbeat_interval)
            )

            if is_primary:
                self.primary_feeds[data_type] = feed_name

    def register_backup_feed(self, data_type: str, backup_feed: str):
        """
        Register a backup feed.

        Args:
            data_type: Type of data
            backup_feed: Name of backup feed
        """
        with self.lock:
            self.backup_feeds[data_type].append(backup_feed)

    def record_message(self, feed_name: str, symbol: str, sequence: Optional[int] = None,
                       latency_ms: Optional[float] = None):
        """
        Record a message received from feed.

        Args:
            feed_name: Name of the feed
            symbol: Symbol received
            sequence: Sequence number if available
            latency_ms: Message latency if available
        """
        with self.lock:
            if feed_name not in self.feeds:
                return

            feed = self.feeds[feed_name]
            feed.last_message_time = datetime.now()
            feed.messages_received += 1

            # Update symbol coverage
            self.symbol_coverage[feed_name].add(symbol)
            feed.symbols_active = len(self.symbol_coverage[feed_name])

            # Track latency
            if latency_ms is not None:
                self.latencies[feed_name].append((datetime.now(), latency_ms))
                feed.latency_ms = latency_ms

            # Track sequence
            if sequence is not None:
                self._track_sequence(feed_name, symbol, sequence)

    def record_heartbeat(self, feed_name: str):
        """
        Record a heartbeat from feed.

        Args:
            feed_name: Name of the feed
        """
        with self.lock:
            if feed_name in self.heartbeats:
                heartbeat = self.heartbeats[feed_name]
                heartbeat.last_heartbeat = datetime.now()
                heartbeat.consecutive_misses = 0

    def set_expected_symbols(self, feed_name: str, symbols: Set[str]):
        """
        Set expected symbols for a feed.

        Args:
            feed_name: Name of the feed
            symbols: Set of expected symbols
        """
        with self.lock:
            self.expected_symbols[feed_name] = symbols

    def _track_sequence(self, feed_name: str, symbol: str, sequence: int):
        """Track sequence numbers for gap detection."""
        feed_key = f"{feed_name}:{symbol}"

        if feed_key not in self.sequences[feed_name]:
            self.sequences[feed_name][feed_key] = SequenceTracker(
                symbol=symbol,
                last_sequence=sequence,
                expected_sequence=sequence + 1
            )
            return

        tracker = self.sequences[feed_name][feed_key]

        # Check for gap
        if sequence != tracker.expected_sequence:
            gap_size = sequence - tracker.expected_sequence

            # Record gap
            gap = FeedGap(
                feed_name=feed_name,
                symbol=symbol,
                gap_start=tracker.last_update,
                gap_end=datetime.now(),
                expected_sequence=tracker.expected_sequence,
                received_sequence=sequence,
                gap_size=gap_size
            )
            self.gaps[feed_name].append(gap)

            # Update feed metrics
            if feed_name in self.feeds:
                self.feeds[feed_name].gap_count += 1
                self.feeds[feed_name].sequence_errors += 1

            # Track error
            tracker.errors.append({
                'timestamp': datetime.now(),
                'expected': tracker.expected_sequence,
                'received': sequence,
                'gap': gap_size
            })

        tracker.last_sequence = sequence
        tracker.expected_sequence = sequence + 1
        tracker.last_update = datetime.now()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_feed_health()
                self._check_heartbeats()
                self._update_quality_scores()
                self._check_for_failover()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in feed monitor loop: {e}")

    def _check_feed_health(self):
        """Check health of all feeds."""
        with self.lock:
            now = datetime.now()

            for feed_name, feed in self.feeds.items():
                # Check if feed is receiving data
                time_since_message = now - feed.last_message_time

                if time_since_message > timedelta(seconds=60):
                    feed.status = FeedStatus.DISCONNECTED
                elif time_since_message > timedelta(seconds=30):
                    feed.status = FeedStatus.DEGRADED
                else:
                    feed.status = FeedStatus.CONNECTED

                # Update timestamp
                feed.timestamp = now

                # Store in history
                self.feed_history[feed_name].append({
                    'timestamp': now,
                    'status': feed.status,
                    'messages': feed.messages_received,
                    'latency': feed.latency_ms,
                    'quality': feed.quality_score
                })

    def _check_heartbeats(self):
        """Check heartbeats for all feeds."""
        with self.lock:
            for feed_name, heartbeat in self.heartbeats.items():
                if not heartbeat.is_alive():
                    heartbeat.missed_count += 1
                    heartbeat.consecutive_misses += 1

                    if feed_name in self.feeds:
                        self.feeds[feed_name].heartbeat_missed = heartbeat.missed_count

                        # Mark as disconnected if too many consecutive misses
                        if heartbeat.consecutive_misses >= 3:
                            self.feeds[feed_name].status = FeedStatus.DISCONNECTED

    def _update_quality_scores(self):
        """Update quality scores for all feeds."""
        with self.lock:
            for feed_name, feed in self.feeds.items():
                score = 100.0

                # Deduct for status
                if feed.status == FeedStatus.DEGRADED:
                    score -= 20
                elif feed.status == FeedStatus.DISCONNECTED:
                    score -= 50

                # Deduct for latency
                if feed.latency_ms > 100:
                    score -= min(30, (feed.latency_ms - 100) / 10)

                # Deduct for gaps
                recent_gaps = len([g for g in self.gaps[feed_name]
                                 if g.gap_end and (datetime.now() - g.gap_end) < timedelta(minutes=5)])
                score -= min(20, recent_gaps * 2)

                # Deduct for heartbeat misses
                score -= min(20, feed.heartbeat_missed * 5)

                # Deduct for symbol coverage
                if feed_name in self.expected_symbols:
                    expected = len(self.expected_symbols[feed_name])
                    actual = feed.symbols_active
                    if expected > 0:
                        coverage_ratio = actual / expected
                        if coverage_ratio < 0.8:
                            score -= (1.0 - coverage_ratio) * 20

                feed.quality_score = max(0.0, score)
                self.quality_scores[feed_name].append((datetime.now(), score))

    def _check_for_failover(self):
        """Check if failover to backup feed is needed."""
        with self.lock:
            for data_type, primary_feed in self.primary_feeds.items():
                if primary_feed not in self.feeds:
                    continue

                primary = self.feeds[primary_feed]

                # Check if failover is needed
                needs_failover = (
                    primary.status == FeedStatus.DISCONNECTED or
                    primary.quality_score < 30.0
                )

                if needs_failover and data_type in self.backup_feeds:
                    backup_feeds = self.backup_feeds[data_type]

                    # Find best backup feed
                    best_backup = None
                    best_score = 0.0

                    for backup_name in backup_feeds:
                        if backup_name in self.feeds:
                            backup = self.feeds[backup_name]
                            if backup.status == FeedStatus.CONNECTED and backup.quality_score > best_score:
                                best_backup = backup_name
                                best_score = backup.quality_score

                    if best_backup:
                        self._switch_to_backup(data_type, primary_feed, best_backup)

    def _switch_to_backup(self, data_type: str, primary: str, backup: str):
        """
        Switch from primary to backup feed.

        Args:
            data_type: Type of data
            primary: Primary feed name
            backup: Backup feed name
        """
        if primary in self.feeds:
            self.feeds[primary].status = FeedStatus.SWITCHING

        if backup in self.feeds:
            self.feeds[backup].status = FeedStatus.BACKUP
            self.feeds[backup].is_primary = True

        # Update primary feed mapping
        old_primary = self.primary_feeds[data_type]
        self.primary_feeds[data_type] = backup

        # Mark old primary as backup
        if old_primary in self.feeds:
            self.feeds[old_primary].is_primary = False

    def get_feed_status(self, feed_name: Optional[str] = None) -> Dict[str, FeedHealthMetric]:
        """
        Get feed status.

        Args:
            feed_name: Optional specific feed name

        Returns:
            Dictionary of feed health metrics
        """
        with self.lock:
            if feed_name:
                return {feed_name: self.feeds.get(feed_name)}
            return self.feeds.copy()

    def get_latency_stats(self, feed_name: str, window_minutes: int = 5) -> Dict[str, float]:
        """
        Get latency statistics for a feed.

        Args:
            feed_name: Name of the feed
            window_minutes: Time window for stats

        Returns:
            Dictionary with latency statistics
        """
        with self.lock:
            if feed_name not in self.latencies:
                return {}

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent = [lat for ts, lat in self.latencies[feed_name] if ts >= cutoff]

            if not recent:
                return {}

            return {
                'min_ms': min(recent),
                'max_ms': max(recent),
                'avg_ms': sum(recent) / len(recent),
                'median_ms': sorted(recent)[len(recent) // 2],
                'count': len(recent)
            }

    def get_gaps(self, feed_name: str, window_minutes: Optional[int] = None) -> List[FeedGap]:
        """
        Get data gaps for a feed.

        Args:
            feed_name: Name of the feed
            window_minutes: Time window (None = all)

        Returns:
            List of gaps
        """
        with self.lock:
            gaps = self.gaps.get(feed_name, [])

            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                gaps = [g for g in gaps if g.gap_start >= cutoff]

            return gaps

    def get_symbol_coverage(self, feed_name: str) -> Dict[str, Any]:
        """
        Get symbol coverage statistics.

        Args:
            feed_name: Name of the feed

        Returns:
            Dictionary with coverage statistics
        """
        with self.lock:
            active = self.symbol_coverage.get(feed_name, set())
            expected = self.expected_symbols.get(feed_name, set())

            missing = expected - active if expected else set()

            return {
                'active_symbols': len(active),
                'expected_symbols': len(expected),
                'coverage_percent': (len(active) / len(expected) * 100) if expected else 100.0,
                'missing_symbols': list(missing)
            }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get feed monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total_feeds = len(self.feeds)
            connected = sum(1 for f in self.feeds.values() if f.status == FeedStatus.CONNECTED)
            degraded = sum(1 for f in self.feeds.values() if f.status == FeedStatus.DEGRADED)
            disconnected = sum(1 for f in self.feeds.values() if f.status == FeedStatus.DISCONNECTED)

            total_messages = sum(f.messages_received for f in self.feeds.values())
            total_gaps = sum(f.gap_count for f in self.feeds.values())

            avg_quality = sum(f.quality_score for f in self.feeds.values()) / total_feeds if total_feeds > 0 else 0

            return {
                'total_feeds': total_feeds,
                'connected': connected,
                'degraded': degraded,
                'disconnected': disconnected,
                'total_messages': total_messages,
                'total_gaps': total_gaps,
                'average_quality_score': avg_quality,
                'timestamp': datetime.now().isoformat()
            }

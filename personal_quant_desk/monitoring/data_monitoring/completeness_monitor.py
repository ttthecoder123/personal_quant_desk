"""
Completeness Monitor

Missing data detection, coverage analysis, historical data gaps, symbol availability,
field completeness, and time series continuity monitoring.
"""

import threading
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np


@dataclass
class DataGap:
    """Represents a gap in data."""
    symbol: str
    field: str
    gap_start: datetime
    gap_end: Optional[datetime]
    expected_count: int
    actual_count: int
    gap_size: int
    gap_type: str  # 'temporal', 'sequence', 'field'
    severity: str  # 'minor', 'moderate', 'critical'
    filled: bool = False


@dataclass
class SymbolCoverage:
    """Symbol coverage statistics."""
    symbol: str
    total_fields: int
    complete_fields: int
    partial_fields: int
    missing_fields: int
    completeness_pct: float
    last_update: datetime
    update_frequency: Optional[float] = None  # Updates per minute


@dataclass
class FieldCompleteness:
    """Field completeness tracking."""
    symbol: str
    field: str
    expected_updates: int
    actual_updates: int
    missing_updates: int
    completeness_pct: float
    last_update: Optional[datetime]
    update_interval: Optional[timedelta]


@dataclass
class TimeSeriesContinuity:
    """Time series continuity metrics."""
    symbol: str
    field: str
    start_time: datetime
    end_time: datetime
    expected_points: int
    actual_points: int
    gaps_count: int
    largest_gap_seconds: float
    continuity_score: float  # 0-100


class CompletenessMonitor:
    """
    Comprehensive data completeness monitoring.

    Features:
    - Missing data detection
    - Coverage analysis (symbols, fields)
    - Historical data gap detection
    - Symbol availability tracking
    - Field completeness scoring
    - Time series continuity checks
    - Expected vs actual update tracking
    - Data density analysis
    - Coverage trend monitoring
    """

    def __init__(self, check_interval: int = 60):
        """
        Initialize completeness monitor.

        Args:
            check_interval: Seconds between completeness checks
        """
        self.check_interval = check_interval

        # Data tracking
        self.data_points: Dict[str, Dict[str, List[datetime]]] = defaultdict(lambda: defaultdict(list))
        self.last_updates: Dict[str, Dict[str, datetime]] = defaultdict(dict)

        # Expected configuration
        self.expected_symbols: Set[str] = set()
        self.expected_fields: Dict[str, Set[str]] = defaultdict(set)
        self.expected_intervals: Dict[str, Dict[str, timedelta]] = defaultdict(dict)

        # Gap tracking
        self.gaps: Dict[str, List[DataGap]] = defaultdict(list)
        self.active_gaps: Dict[str, DataGap] = {}

        # Coverage tracking
        self.symbol_coverage: Dict[str, SymbolCoverage] = {}
        self.field_completeness: Dict[str, Dict[str, FieldCompleteness]] = defaultdict(dict)

        # Time series continuity
        self.continuity_metrics: Dict[str, Dict[str, TimeSeriesContinuity]] = defaultdict(dict)

        # Update frequency tracking
        self.update_counts: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))

        # Historical data
        self.historical_coverage: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals

        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start(self):
        """Start completeness monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop completeness monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def configure_expectations(self, symbol: str, fields: Set[str],
                              update_intervals: Optional[Dict[str, timedelta]] = None):
        """
        Configure expected data for a symbol.

        Args:
            symbol: Symbol name
            fields: Set of expected fields
            update_intervals: Expected update interval per field
        """
        with self.lock:
            self.expected_symbols.add(symbol)
            self.expected_fields[symbol] = fields

            if update_intervals:
                for field, interval in update_intervals.items():
                    self.expected_intervals[symbol][field] = interval

    def record_data_point(self, symbol: str, field: str, timestamp: Optional[datetime] = None):
        """
        Record a data point.

        Args:
            symbol: Symbol name
            field: Field name
            timestamp: Data timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self.lock:
            # Record timestamp
            self.data_points[symbol][field].append(timestamp)
            self.last_updates[symbol][field] = timestamp

            # Track update frequency
            self.update_counts[symbol][field].append(timestamp)

            # Check if this closes an active gap
            gap_key = f"{symbol}:{field}"
            if gap_key in self.active_gaps:
                gap = self.active_gaps[gap_key]
                gap.gap_end = timestamp
                gap.filled = True
                self.gaps[symbol].append(gap)
                del self.active_gaps[gap_key]

    def detect_gaps(self, symbol: str, field: Optional[str] = None,
                    window_hours: int = 24) -> List[DataGap]:
        """
        Detect gaps in data.

        Args:
            symbol: Symbol name
            field: Optional specific field
            window_hours: Time window to check

        Returns:
            List of detected gaps
        """
        with self.lock:
            gaps = []
            cutoff = datetime.now() - timedelta(hours=window_hours)

            fields_to_check = [field] if field else self.expected_fields.get(symbol, set())

            for fld in fields_to_check:
                if fld not in self.data_points[symbol]:
                    # Complete missing field
                    gap = DataGap(
                        symbol=symbol,
                        field=fld,
                        gap_start=cutoff,
                        gap_end=None,
                        expected_count=self._calculate_expected_count(symbol, fld, window_hours),
                        actual_count=0,
                        gap_size=self._calculate_expected_count(symbol, fld, window_hours),
                        gap_type='field',
                        severity='critical'
                    )
                    gaps.append(gap)
                    continue

                # Get timestamps in window
                timestamps = [ts for ts in self.data_points[symbol][fld] if ts >= cutoff]

                if not timestamps:
                    continue

                # Check for temporal gaps
                timestamps = sorted(timestamps)

                # Check expected interval
                if symbol in self.expected_intervals and fld in self.expected_intervals[symbol]:
                    expected_interval = self.expected_intervals[symbol][fld]
                    max_gap = expected_interval * 3  # Allow 3x the expected interval

                    for i in range(len(timestamps) - 1):
                        gap_duration = timestamps[i + 1] - timestamps[i]

                        if gap_duration > max_gap:
                            gap_size = int(gap_duration / expected_interval)
                            severity = self._determine_severity(gap_duration, expected_interval)

                            gap = DataGap(
                                symbol=symbol,
                                field=fld,
                                gap_start=timestamps[i],
                                gap_end=timestamps[i + 1],
                                expected_count=gap_size,
                                actual_count=0,
                                gap_size=gap_size,
                                gap_type='temporal',
                                severity=severity,
                                filled=True
                            )
                            gaps.append(gap)

            return gaps

    def calculate_coverage(self, symbol: Optional[str] = None) -> Dict[str, SymbolCoverage]:
        """
        Calculate symbol coverage.

        Args:
            symbol: Optional specific symbol

        Returns:
            Dictionary of symbol coverage
        """
        with self.lock:
            coverage = {}

            symbols_to_check = [symbol] if symbol else self.expected_symbols

            for sym in symbols_to_check:
                if sym not in self.expected_fields:
                    continue

                expected_fields = self.expected_fields[sym]
                total_fields = len(expected_fields)

                complete_fields = 0
                partial_fields = 0
                missing_fields = 0

                for field in expected_fields:
                    if field in self.data_points[sym]:
                        # Check if recently updated
                        if field in self.last_updates[sym]:
                            last_update = self.last_updates[sym][field]
                            age = datetime.now() - last_update

                            if age < timedelta(minutes=5):
                                complete_fields += 1
                            else:
                                partial_fields += 1
                        else:
                            partial_fields += 1
                    else:
                        missing_fields += 1

                completeness_pct = (complete_fields / total_fields * 100) if total_fields > 0 else 0

                # Calculate update frequency
                update_freq = None
                if sym in self.update_counts:
                    recent_updates = sum(len(updates) for updates in self.update_counts[sym].values())
                    if recent_updates > 0:
                        update_freq = recent_updates / len(self.update_counts[sym])

                coverage[sym] = SymbolCoverage(
                    symbol=sym,
                    total_fields=total_fields,
                    complete_fields=complete_fields,
                    partial_fields=partial_fields,
                    missing_fields=missing_fields,
                    completeness_pct=completeness_pct,
                    last_update=max(self.last_updates[sym].values()) if self.last_updates[sym] else datetime.now(),
                    update_frequency=update_freq
                )

            return coverage

    def calculate_field_completeness(self, symbol: str, field: str,
                                    window_hours: int = 1) -> FieldCompleteness:
        """
        Calculate field completeness.

        Args:
            symbol: Symbol name
            field: Field name
            window_hours: Time window

        Returns:
            FieldCompleteness object
        """
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=window_hours)

            # Get actual updates
            timestamps = [ts for ts in self.data_points[symbol].get(field, []) if ts >= cutoff]
            actual_updates = len(timestamps)

            # Calculate expected updates
            expected_updates = self._calculate_expected_count(symbol, field, window_hours)

            # Calculate missing
            missing_updates = max(0, expected_updates - actual_updates)

            # Completeness percentage
            completeness_pct = (actual_updates / expected_updates * 100) if expected_updates > 0 else 0

            # Last update
            last_update = self.last_updates[symbol].get(field)

            # Update interval
            update_interval = None
            if symbol in self.expected_intervals and field in self.expected_intervals[symbol]:
                update_interval = self.expected_intervals[symbol][field]

            return FieldCompleteness(
                symbol=symbol,
                field=field,
                expected_updates=expected_updates,
                actual_updates=actual_updates,
                missing_updates=missing_updates,
                completeness_pct=completeness_pct,
                last_update=last_update,
                update_interval=update_interval
            )

    def analyze_time_series_continuity(self, symbol: str, field: str,
                                       window_hours: int = 24) -> TimeSeriesContinuity:
        """
        Analyze time series continuity.

        Args:
            symbol: Symbol name
            field: Field name
            window_hours: Time window

        Returns:
            TimeSeriesContinuity object
        """
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=window_hours)

            # Get timestamps in window
            timestamps = sorted([ts for ts in self.data_points[symbol].get(field, []) if ts >= cutoff])

            if not timestamps:
                return TimeSeriesContinuity(
                    symbol=symbol,
                    field=field,
                    start_time=cutoff,
                    end_time=datetime.now(),
                    expected_points=self._calculate_expected_count(symbol, field, window_hours),
                    actual_points=0,
                    gaps_count=0,
                    largest_gap_seconds=0.0,
                    continuity_score=0.0
                )

            start_time = timestamps[0]
            end_time = timestamps[-1]
            actual_points = len(timestamps)

            # Calculate expected points
            expected_points = self._calculate_expected_count(symbol, field, window_hours)

            # Detect gaps
            gaps_count = 0
            largest_gap = 0.0

            if symbol in self.expected_intervals and field in self.expected_intervals[symbol]:
                expected_interval = self.expected_intervals[symbol][field]
                max_gap = expected_interval * 2

                for i in range(len(timestamps) - 1):
                    gap = timestamps[i + 1] - timestamps[i]
                    if gap > max_gap:
                        gaps_count += 1
                        largest_gap = max(largest_gap, gap.total_seconds())

            # Calculate continuity score
            completeness = (actual_points / expected_points) if expected_points > 0 else 0
            gap_penalty = min(gaps_count * 5, 30)  # Max 30% penalty for gaps
            continuity_score = max(0, (completeness * 100) - gap_penalty)

            return TimeSeriesContinuity(
                symbol=symbol,
                field=field,
                start_time=start_time,
                end_time=end_time,
                expected_points=expected_points,
                actual_points=actual_points,
                gaps_count=gaps_count,
                largest_gap_seconds=largest_gap,
                continuity_score=continuity_score
            )

    def get_missing_symbols(self) -> Set[str]:
        """
        Get symbols with no data.

        Returns:
            Set of missing symbols
        """
        with self.lock:
            missing = set()
            for symbol in self.expected_symbols:
                if symbol not in self.data_points or not self.data_points[symbol]:
                    missing.add(symbol)
            return missing

    def get_missing_fields(self, symbol: str) -> Set[str]:
        """
        Get missing fields for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            Set of missing fields
        """
        with self.lock:
            if symbol not in self.expected_fields:
                return set()

            expected = self.expected_fields[symbol]
            actual = set(self.data_points[symbol].keys())

            return expected - actual

    def get_stale_data(self, max_age_minutes: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Get stale data (not updated recently).

        Args:
            max_age_minutes: Maximum age in minutes

        Returns:
            Dictionary of stale data with age in seconds
        """
        with self.lock:
            stale = {}
            cutoff = datetime.now() - timedelta(minutes=max_age_minutes)

            for symbol in self.expected_symbols:
                if symbol not in self.last_updates:
                    continue

                stale_fields = {}
                for field, last_update in self.last_updates[symbol].items():
                    if last_update < cutoff:
                        age_seconds = (datetime.now() - last_update).total_seconds()
                        stale_fields[field] = age_seconds

                if stale_fields:
                    stale[symbol] = stale_fields

            return stale

    def get_data_density(self, symbol: str, field: str, window_hours: int = 1) -> float:
        """
        Calculate data density (ratio of actual to expected updates).

        Args:
            symbol: Symbol name
            field: Field name
            window_hours: Time window

        Returns:
            Density ratio (0-1)
        """
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=window_hours)

            actual = len([ts for ts in self.data_points[symbol].get(field, []) if ts >= cutoff])
            expected = self._calculate_expected_count(symbol, field, window_hours)

            return min(1.0, actual / expected) if expected > 0 else 0.0

    def _calculate_expected_count(self, symbol: str, field: str, window_hours: int) -> int:
        """Calculate expected number of updates."""
        if symbol not in self.expected_intervals or field not in self.expected_intervals[symbol]:
            # Default assumption: 1 update per minute
            return window_hours * 60

        interval = self.expected_intervals[symbol][field]
        window_seconds = window_hours * 3600
        return int(window_seconds / interval.total_seconds())

    def _determine_severity(self, gap_duration: timedelta, expected_interval: timedelta) -> str:
        """Determine gap severity."""
        ratio = gap_duration / expected_interval

        if ratio > 10:
            return 'critical'
        elif ratio > 5:
            return 'moderate'
        else:
            return 'minor'

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_completeness()
                self._detect_new_gaps()
                self._update_coverage_history()
                threading.Event().wait(self.check_interval)
            except Exception as e:
                print(f"Error in completeness monitor loop: {e}")

    def _check_completeness(self):
        """Check completeness for all symbols."""
        with self.lock:
            for symbol in self.expected_symbols:
                coverage = self.calculate_coverage(symbol)
                self.symbol_coverage.update(coverage)

                # Update field completeness
                if symbol in self.expected_fields:
                    for field in self.expected_fields[symbol]:
                        completeness = self.calculate_field_completeness(symbol, field)
                        self.field_completeness[symbol][field] = completeness

    def _detect_new_gaps(self):
        """Detect new gaps in data."""
        with self.lock:
            now = datetime.now()

            for symbol in self.expected_symbols:
                if symbol not in self.expected_intervals:
                    continue

                for field, interval in self.expected_intervals[symbol].items():
                    # Check if we should have received an update
                    if field in self.last_updates[symbol]:
                        last_update = self.last_updates[symbol][field]
                        expected_update = last_update + interval

                        if now > expected_update + interval:  # Allow 1 interval grace period
                            gap_key = f"{symbol}:{field}"

                            if gap_key not in self.active_gaps:
                                # New gap detected
                                gap = DataGap(
                                    symbol=symbol,
                                    field=field,
                                    gap_start=expected_update,
                                    gap_end=None,
                                    expected_count=1,
                                    actual_count=0,
                                    gap_size=1,
                                    gap_type='temporal',
                                    severity=self._determine_severity(now - expected_update, interval)
                                )
                                self.active_gaps[gap_key] = gap

    def _update_coverage_history(self):
        """Update historical coverage data."""
        with self.lock:
            coverage = self.calculate_coverage()

            if coverage:
                avg_completeness = np.mean([c.completeness_pct for c in coverage.values()])

                self.historical_coverage.append({
                    'timestamp': datetime.now(),
                    'average_completeness': avg_completeness,
                    'total_symbols': len(coverage),
                    'complete_symbols': sum(1 for c in coverage.values() if c.completeness_pct == 100)
                })

    def get_summary(self) -> Dict[str, Any]:
        """
        Get completeness monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total_symbols = len(self.expected_symbols)
            total_gaps = sum(len(gaps) for gaps in self.gaps.values())
            active_gaps = len(self.active_gaps)

            coverage = self.calculate_coverage()
            avg_completeness = np.mean([c.completeness_pct for c in coverage.values()]) if coverage else 0

            missing_symbols = len(self.get_missing_symbols())

            return {
                'total_symbols': total_symbols,
                'missing_symbols': missing_symbols,
                'average_completeness': avg_completeness,
                'total_gaps': total_gaps,
                'active_gaps': active_gaps,
                'timestamp': datetime.now().isoformat()
            }

"""
Quality Monitor

Data quality scoring, outlier detection, missing value tracking, consistency checks,
cross-validation between sources, corporate action validation, and price sanity checks.
"""

import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy import stats
from enum import Enum


class QualityIssueType(Enum):
    """Types of data quality issues."""
    OUTLIER = "outlier"
    MISSING_VALUE = "missing_value"
    INCONSISTENT = "inconsistent"
    INVALID_PRICE = "invalid_price"
    STALE_DATA = "stale_data"
    DUPLICATE = "duplicate"
    CORPORATE_ACTION = "corporate_action"
    CROSS_VALIDATION_FAIL = "cross_validation_fail"


@dataclass
class QualityIssue:
    """Data quality issue."""
    issue_type: QualityIssueType
    symbol: str
    field: str
    value: Any
    expected_value: Optional[Any]
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: datetime
    resolved: bool = False


@dataclass
class QualityScore:
    """Quality score for a symbol or field."""
    symbol: str
    field: Optional[str]
    score: float  # 0-100
    completeness: float  # 0-100
    accuracy: float  # 0-100
    consistency: float  # 0-100
    timeliness: float  # 0-100
    issues_count: int
    timestamp: datetime


@dataclass
class OutlierStats:
    """Outlier detection statistics."""
    symbol: str
    field: str
    value: float
    mean: float
    std_dev: float
    z_score: float
    is_outlier: bool
    timestamp: datetime


class QualityMonitor:
    """
    Comprehensive data quality monitoring.

    Features:
    - Data quality scoring (0-100 scale)
    - Statistical outlier detection (z-score, IQR)
    - Missing value tracking
    - Consistency checks across time and sources
    - Cross-validation between data sources
    - Corporate action validation
    - Price sanity checks (negative, zero, spike detection)
    - Duplicate detection
    - Staleness detection
    - Field completeness tracking
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize quality monitor.

        Args:
            window_size: Number of samples to keep for statistical analysis
        """
        self.window_size = window_size

        # Quality scores
        self.quality_scores: Dict[str, QualityScore] = {}
        self.score_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Issues tracking
        self.issues: Dict[str, List[QualityIssue]] = defaultdict(list)
        self.active_issues: Dict[str, List[QualityIssue]] = defaultdict(list)

        # Data history for statistical analysis
        self.data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Missing value tracking
        self.missing_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.expected_updates: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Cross-validation
        self.source_data: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))

        # Corporate actions
        self.corporate_actions: Dict[str, List[Dict]] = defaultdict(list)

        # Price ranges
        self.price_ranges: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Duplicate detection
        self.seen_data: Dict[str, Set[Tuple]] = defaultdict(set)

        # Last update times
        self.last_updates: Dict[str, datetime] = {}

        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start(self):
        """Start quality monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop quality monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def record_data(self, symbol: str, field: str, value: Any, source: Optional[str] = None,
                    timestamp: Optional[datetime] = None):
        """
        Record data for quality monitoring.

        Args:
            symbol: Symbol
            field: Field name
            value: Field value
            source: Data source name
            timestamp: Data timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self.lock:
            key = f"{symbol}:{field}"

            # Check for missing values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                self.missing_counts[symbol][field] += 1
                self._create_issue(
                    QualityIssueType.MISSING_VALUE,
                    symbol, field, value, None,
                    'medium',
                    f"Missing value for {field}"
                )
                return

            # Check for duplicates
            data_tuple = (symbol, field, value, timestamp.isoformat())
            if data_tuple in self.seen_data[key]:
                self._create_issue(
                    QualityIssueType.DUPLICATE,
                    symbol, field, value, None,
                    'low',
                    f"Duplicate data point for {field}"
                )
            self.seen_data[key].add(data_tuple)

            # Record data
            self.data_history[key].append({
                'value': value,
                'timestamp': timestamp
            })

            # Update last update time
            self.last_updates[key] = timestamp

            # Track by source for cross-validation
            if source:
                source_key = f"{symbol}:{field}:{source}"
                self.source_data[symbol][source_key].append({
                    'value': value,
                    'timestamp': timestamp
                })

            # Expected updates counter
            self.expected_updates[symbol][field] += 1

            # Check for price-specific issues
            if field in ['price', 'bid', 'ask', 'last', 'close', 'open', 'high', 'low']:
                self._check_price_sanity(symbol, field, value)

    def check_outliers(self, symbol: str, field: str, method: str = 'zscore',
                       threshold: float = 3.0) -> List[OutlierStats]:
        """
        Check for outliers in data.

        Args:
            symbol: Symbol
            field: Field name
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold for outlier detection

        Returns:
            List of outlier statistics
        """
        with self.lock:
            key = f"{symbol}:{field}"
            if key not in self.data_history or len(self.data_history[key]) < 10:
                return []

            # Extract numeric values
            values = [d['value'] for d in self.data_history[key] if isinstance(d['value'], (int, float))]
            if len(values) < 10:
                return []

            values_array = np.array(values)
            outliers = []

            if method == 'zscore':
                mean = np.mean(values_array)
                std_dev = np.std(values_array)

                if std_dev > 0:
                    z_scores = np.abs((values_array - mean) / std_dev)

                    for i, (value, z_score) in enumerate(zip(values, z_scores)):
                        is_outlier = z_score > threshold

                        if is_outlier:
                            outlier = OutlierStats(
                                symbol=symbol,
                                field=field,
                                value=value,
                                mean=mean,
                                std_dev=std_dev,
                                z_score=z_score,
                                is_outlier=True,
                                timestamp=self.data_history[key][i]['timestamp']
                            )
                            outliers.append(outlier)

                            # Create issue
                            self._create_issue(
                                QualityIssueType.OUTLIER,
                                symbol, field, value, mean,
                                'medium' if z_score < 5 else 'high',
                                f"Outlier detected: {value} (z-score: {z_score:.2f})"
                            )

            elif method == 'iqr':
                q1 = np.percentile(values_array, 25)
                q3 = np.percentile(values_array, 75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)

                for i, value in enumerate(values):
                    if value < lower_bound or value > upper_bound:
                        outlier = OutlierStats(
                            symbol=symbol,
                            field=field,
                            value=value,
                            mean=np.mean(values_array),
                            std_dev=np.std(values_array),
                            z_score=0.0,
                            is_outlier=True,
                            timestamp=self.data_history[key][i]['timestamp']
                        )
                        outliers.append(outlier)

            return outliers

    def cross_validate(self, symbol: str, field: str, max_deviation: float = 0.01) -> Dict[str, Any]:
        """
        Cross-validate data across sources.

        Args:
            symbol: Symbol
            field: Field name
            max_deviation: Maximum allowed deviation between sources

        Returns:
            Validation results
        """
        with self.lock:
            if symbol not in self.source_data:
                return {'validated': False, 'reason': 'No data'}

            # Get all sources for this symbol/field
            sources = {k: v for k, v in self.source_data[symbol].items()
                      if k.startswith(f"{symbol}:{field}:")}

            if len(sources) < 2:
                return {'validated': True, 'reason': 'Single source'}

            # Get most recent values from each source
            source_values = {}
            for source_key, data in sources.items():
                if data:
                    source_name = source_key.split(':')[-1]
                    source_values[source_name] = data[-1]['value']

            if len(source_values) < 2:
                return {'validated': False, 'reason': 'Insufficient data'}

            # Calculate deviation
            values = list(source_values.values())
            mean_value = np.mean(values)
            max_dev = max(abs(v - mean_value) / mean_value for v in values if mean_value != 0)

            validated = max_dev <= max_deviation

            if not validated:
                self._create_issue(
                    QualityIssueType.CROSS_VALIDATION_FAIL,
                    symbol, field, source_values, None,
                    'high',
                    f"Cross-validation failed: deviation {max_dev:.2%}"
                )

            return {
                'validated': validated,
                'sources': source_values,
                'mean': mean_value,
                'max_deviation': max_dev,
                'threshold': max_deviation
            }

    def check_completeness(self, symbol: str, field: Optional[str] = None) -> Dict[str, float]:
        """
        Check data completeness.

        Args:
            symbol: Symbol
            field: Optional specific field

        Returns:
            Completeness percentages
        """
        with self.lock:
            if field:
                expected = self.expected_updates[symbol].get(field, 0)
                missing = self.missing_counts[symbol].get(field, 0)
                completeness = ((expected - missing) / expected * 100) if expected > 0 else 100.0
                return {field: completeness}

            # All fields
            completeness = {}
            for field_name in self.expected_updates[symbol].keys():
                expected = self.expected_updates[symbol][field_name]
                missing = self.missing_counts[symbol][field_name]
                completeness[field_name] = ((expected - missing) / expected * 100) if expected > 0 else 100.0

            return completeness

    def _check_price_sanity(self, symbol: str, field: str, value: float):
        """Check price sanity."""
        # Check for negative prices
        if value < 0:
            self._create_issue(
                QualityIssueType.INVALID_PRICE,
                symbol, field, value, None,
                'critical',
                f"Negative price: {value}"
            )
            return

        # Check for zero prices
        if value == 0:
            self._create_issue(
                QualityIssueType.INVALID_PRICE,
                symbol, field, value, None,
                'high',
                f"Zero price detected"
            )
            return

        # Check for price spikes
        key = f"{symbol}:{field}"
        if key in self.data_history and len(self.data_history[key]) > 1:
            prev_value = self.data_history[key][-1]['value']
            if isinstance(prev_value, (int, float)) and prev_value > 0:
                change_pct = abs(value - prev_value) / prev_value

                if change_pct > 0.2:  # 20% spike
                    self._create_issue(
                        QualityIssueType.INVALID_PRICE,
                        symbol, field, value, prev_value,
                        'high' if change_pct > 0.5 else 'medium',
                        f"Price spike: {change_pct:.1%} change"
                    )

        # Update price range
        if symbol not in self.price_ranges:
            self.price_ranges[symbol] = {'min': value, 'max': value}
        else:
            self.price_ranges[symbol]['min'] = min(self.price_ranges[symbol]['min'], value)
            self.price_ranges[symbol]['max'] = max(self.price_ranges[symbol]['max'], value)

    def _create_issue(self, issue_type: QualityIssueType, symbol: str, field: str,
                      value: Any, expected_value: Optional[Any], severity: str,
                      description: str):
        """Create a quality issue."""
        issue = QualityIssue(
            issue_type=issue_type,
            symbol=symbol,
            field=field,
            value=value,
            expected_value=expected_value,
            severity=severity,
            description=description,
            timestamp=datetime.now()
        )

        self.issues[symbol].append(issue)
        self.active_issues[symbol].append(issue)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._update_quality_scores()
                self._check_staleness()
                self._cleanup_old_issues()
                threading.Event().wait(5)  # Sleep for 5 seconds
            except Exception as e:
                print(f"Error in quality monitor loop: {e}")

    def _update_quality_scores(self):
        """Update quality scores for all symbols."""
        with self.lock:
            for symbol in self.expected_updates.keys():
                for field in self.expected_updates[symbol].keys():
                    key = f"{symbol}:{field}"

                    # Calculate component scores
                    completeness = self.check_completeness(symbol, field).get(field, 100.0)

                    # Accuracy (based on outliers and invalid values)
                    recent_issues = [i for i in self.active_issues[symbol]
                                   if i.field == field and
                                   (datetime.now() - i.timestamp) < timedelta(hours=1)]
                    outlier_issues = sum(1 for i in recent_issues if i.issue_type == QualityIssueType.OUTLIER)
                    invalid_issues = sum(1 for i in recent_issues if i.issue_type == QualityIssueType.INVALID_PRICE)
                    accuracy = max(0, 100 - (outlier_issues * 5) - (invalid_issues * 10))

                    # Consistency (based on cross-validation failures)
                    consistency_issues = sum(1 for i in recent_issues
                                           if i.issue_type == QualityIssueType.CROSS_VALIDATION_FAIL)
                    consistency = max(0, 100 - (consistency_issues * 15))

                    # Timeliness (based on staleness)
                    if key in self.last_updates:
                        age = (datetime.now() - self.last_updates[key]).total_seconds()
                        if age < 60:
                            timeliness = 100.0
                        elif age < 300:
                            timeliness = 80.0
                        elif age < 900:
                            timeliness = 60.0
                        else:
                            timeliness = 40.0
                    else:
                        timeliness = 0.0

                    # Overall score (weighted average)
                    score = (completeness * 0.3 + accuracy * 0.3 + consistency * 0.2 + timeliness * 0.2)

                    quality_score = QualityScore(
                        symbol=symbol,
                        field=field,
                        score=score,
                        completeness=completeness,
                        accuracy=accuracy,
                        consistency=consistency,
                        timeliness=timeliness,
                        issues_count=len(recent_issues),
                        timestamp=datetime.now()
                    )

                    self.quality_scores[key] = quality_score
                    self.score_history[key].append((datetime.now(), score))

    def _check_staleness(self):
        """Check for stale data."""
        with self.lock:
            now = datetime.now()
            staleness_threshold = timedelta(minutes=5)

            for key, last_update in self.last_updates.items():
                age = now - last_update

                if age > staleness_threshold:
                    symbol, field = key.split(':', 1)
                    self._create_issue(
                        QualityIssueType.STALE_DATA,
                        symbol, field, age.total_seconds(), None,
                        'medium',
                        f"Stale data: last update {age.total_seconds():.0f}s ago"
                    )

    def _cleanup_old_issues(self):
        """Clean up old resolved issues."""
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=24)

            for symbol in list(self.issues.keys()):
                # Keep recent issues and unresolved issues
                self.issues[symbol] = [
                    i for i in self.issues[symbol]
                    if i.timestamp >= cutoff or not i.resolved
                ]

                self.active_issues[symbol] = [
                    i for i in self.active_issues[symbol]
                    if not i.resolved
                ]

    def get_quality_score(self, symbol: str, field: Optional[str] = None) -> Dict[str, QualityScore]:
        """
        Get quality scores.

        Args:
            symbol: Symbol
            field: Optional specific field

        Returns:
            Dictionary of quality scores
        """
        with self.lock:
            if field:
                key = f"{symbol}:{field}"
                return {key: self.quality_scores.get(key)}

            # All fields for symbol
            return {k: v for k, v in self.quality_scores.items() if k.startswith(f"{symbol}:")}

    def get_issues(self, symbol: str, active_only: bool = True,
                   severity: Optional[str] = None) -> List[QualityIssue]:
        """
        Get quality issues.

        Args:
            symbol: Symbol
            active_only: Only return active (unresolved) issues
            severity: Filter by severity

        Returns:
            List of quality issues
        """
        with self.lock:
            issues = self.active_issues[symbol] if active_only else self.issues[symbol]

            if severity:
                issues = [i for i in issues if i.severity == severity]

            return issues

    def get_summary(self) -> Dict[str, Any]:
        """
        Get quality monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total_symbols = len(self.quality_scores)
            avg_score = np.mean([s.score for s in self.quality_scores.values()]) if self.quality_scores else 0

            total_issues = sum(len(issues) for issues in self.active_issues.values())
            critical_issues = sum(
                sum(1 for i in issues if i.severity == 'critical')
                for issues in self.active_issues.values()
            )

            return {
                'total_symbols': total_symbols,
                'average_quality_score': avg_score,
                'total_active_issues': total_issues,
                'critical_issues': critical_issues,
                'timestamp': datetime.now().isoformat()
            }

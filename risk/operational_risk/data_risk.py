"""
Data Risk Monitoring

Monitors data quality and reliability:
- Data quality scores from ingestion system
- Data latency and freshness tracking
- Statistical anomaly detection
- Missing data pattern analysis
- Data source reliability monitoring
- Data quality alerts
- Automated fallback protocols
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
from scipy import stats
from loguru import logger


class DataQualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class DataSourceStatus(Enum):
    """Data source status"""
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    FALLBACK = "fallback"


class AnomalyType(Enum):
    """Types of data anomalies"""
    OUTLIER = "outlier"
    GAP = "gap"
    STALENESS = "staleness"
    INCONSISTENCY = "inconsistency"
    FORMAT_ERROR = "format_error"


@dataclass
class DataQualityScore:
    """Data quality score record"""
    timestamp: datetime
    instrument: str
    source: str
    composite_score: float
    completeness: float
    consistency: float
    timeliness: float
    accuracy: float
    quality_level: DataQualityLevel


@dataclass
class DataLatency:
    """Data latency metrics"""
    timestamp: datetime
    instrument: str
    source: str
    expected_update_time: datetime
    actual_update_time: datetime
    latency_seconds: float
    is_late: bool
    staleness_hours: float


@dataclass
class DataAnomaly:
    """Data anomaly detection result"""
    timestamp: datetime
    instrument: str
    anomaly_type: AnomalyType
    field: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # 'low', 'medium', 'high'
    description: str


@dataclass
class MissingDataPattern:
    """Missing data pattern analysis"""
    timestamp: datetime
    instrument: str
    field: str
    missing_count: int
    total_count: int
    missing_pct: float
    consecutive_missing: int
    pattern_type: str  # 'random', 'consecutive', 'periodic'


@dataclass
class DataSourceReliability:
    """Data source reliability metrics"""
    timestamp: datetime
    source: str
    uptime_pct: float
    error_count: int
    total_requests: int
    error_rate: float
    avg_response_time_ms: float
    status: DataSourceStatus
    last_successful_update: datetime
    consecutive_failures: int


@dataclass
class DataRiskAlert:
    """Data risk alert"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    alert_type: str
    instrument: Optional[str]
    source: Optional[str]
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    requires_fallback: bool = False


@dataclass
class FallbackProtocol:
    """Fallback protocol execution"""
    timestamp: datetime
    trigger_reason: str
    primary_source: str
    fallback_source: str
    instrument: str
    success: bool
    fallback_quality_score: Optional[float] = None


@dataclass
class DataRiskReport:
    """Comprehensive data risk report"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    quality_summary: Dict[str, float]
    source_reliability: Dict[str, DataSourceReliability]
    recent_anomalies: List[DataAnomaly]
    missing_data_summary: Dict[str, MissingDataPattern]
    latency_issues: List[DataLatency]
    active_fallbacks: List[FallbackProtocol]
    alerts: List[DataRiskAlert]
    recommendations: List[str]


@dataclass
class DataRiskConfig:
    """Configuration for data risk monitoring"""
    # Quality thresholds
    min_quality_score: float = 85.0
    critical_quality_score: float = 70.0
    min_completeness: float = 0.95
    min_consistency: float = 0.90

    # Latency thresholds
    max_latency_seconds: float = 300.0  # 5 minutes
    critical_latency_seconds: float = 900.0  # 15 minutes
    max_staleness_hours: float = 24.0

    # Anomaly detection
    outlier_std_threshold: float = 3.0
    outlier_iqr_multiplier: float = 3.0
    max_consecutive_missing: int = 5

    # Source reliability
    min_uptime_pct: float = 99.0
    max_error_rate: float = 0.05  # 5%
    max_consecutive_failures: int = 3

    # Monitoring windows
    quality_window_hours: int = 24
    reliability_window_hours: int = 168  # 1 week
    anomaly_detection_window: int = 100


class DataRisk:
    """
    Monitor and manage data risk

    Features:
    - Quality score monitoring
    - Latency tracking
    - Anomaly detection
    - Missing data analysis
    - Source reliability monitoring
    - Automated fallback protocols
    """

    def __init__(self, config: Optional[DataRiskConfig] = None):
        """
        Initialize data risk monitor

        Args:
            config: Data risk configuration (optional)
        """
        self.config = config if config is not None else DataRiskConfig()

        # Quality score history
        self.quality_scores: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Latency tracking
        self.latency_records: deque = deque(maxlen=1000)

        # Anomaly tracking
        self.anomalies: List[DataAnomaly] = []

        # Source reliability tracking
        self.source_stats: Dict[str, Dict] = defaultdict(
            lambda: {
                'requests': 0,
                'errors': 0,
                'response_times': deque(maxlen=100),
                'last_success': None,
                'consecutive_failures': 0
            }
        )

        # Active fallback protocols
        self.active_fallbacks: List[FallbackProtocol] = []

        # Alert tracking
        self.alerts: List[DataRiskAlert] = []

        # Historical data for anomaly detection
        self.historical_data: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.config.anomaly_detection_window))
        )

        logger.info("DataRisk monitor initialized")

    def record_quality_score(
        self,
        timestamp: datetime,
        instrument: str,
        source: str,
        composite_score: float,
        completeness: float,
        consistency: float,
        timeliness: float,
        accuracy: float
    ) -> DataQualityScore:
        """
        Record data quality score

        Args:
            timestamp: Timestamp
            instrument: Instrument symbol
            source: Data source
            composite_score: Overall quality score (0-100)
            completeness: Completeness score (0-100)
            consistency: Consistency score (0-100)
            timeliness: Timeliness score (0-100)
            accuracy: Accuracy score (0-100)

        Returns:
            DataQualityScore record
        """
        # Determine quality level
        if composite_score >= 95:
            quality_level = DataQualityLevel.EXCELLENT
        elif composite_score >= 85:
            quality_level = DataQualityLevel.GOOD
        elif composite_score >= 70:
            quality_level = DataQualityLevel.FAIR
        elif composite_score >= 50:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.CRITICAL

        score = DataQualityScore(
            timestamp=timestamp,
            instrument=instrument,
            source=source,
            composite_score=composite_score,
            completeness=completeness,
            consistency=consistency,
            timeliness=timeliness,
            accuracy=accuracy,
            quality_level=quality_level
        )

        # Store in history
        key = f"{instrument}:{source}"
        self.quality_scores[key].append(score)

        # Check for alerts
        self._check_quality_alerts(score)

        logger.debug(f"Recorded quality score for {instrument}: {composite_score:.1f}")

        return score

    def record_data_update(
        self,
        timestamp: datetime,
        instrument: str,
        source: str,
        expected_update_time: datetime,
        data: pd.DataFrame
    ) -> DataLatency:
        """
        Record data update and calculate latency

        Args:
            timestamp: Update timestamp
            instrument: Instrument symbol
            source: Data source
            expected_update_time: When data was expected
            data: Updated data

        Returns:
            DataLatency metrics
        """
        # Calculate latency
        latency_seconds = (timestamp - expected_update_time).total_seconds()
        is_late = latency_seconds > self.config.max_latency_seconds

        # Calculate staleness (time since last data point)
        if not data.empty and hasattr(data.index, 'max'):
            last_data_time = data.index.max()
            if isinstance(last_data_time, pd.Timestamp):
                staleness_hours = (timestamp - last_data_time).total_seconds() / 3600
            else:
                staleness_hours = 0.0
        else:
            staleness_hours = 0.0

        latency = DataLatency(
            timestamp=timestamp,
            instrument=instrument,
            source=source,
            expected_update_time=expected_update_time,
            actual_update_time=timestamp,
            latency_seconds=latency_seconds,
            is_late=is_late,
            staleness_hours=staleness_hours
        )

        self.latency_records.append(latency)

        # Check for alerts
        self._check_latency_alerts(latency)

        # Detect anomalies in data
        self._detect_data_anomalies(timestamp, instrument, data)

        return latency

    def detect_anomalies(
        self,
        timestamp: datetime,
        instrument: str,
        field: str,
        value: float
    ) -> Optional[DataAnomaly]:
        """
        Detect statistical anomalies in data

        Args:
            timestamp: Timestamp
            instrument: Instrument symbol
            field: Data field name
            value: Field value

        Returns:
            DataAnomaly if detected, None otherwise
        """
        # Store value in history
        self.historical_data[instrument][field].append(value)

        # Need sufficient history
        history = list(self.historical_data[instrument][field])
        if len(history) < 30:
            return None

        # Calculate statistics
        mean = np.mean(history[:-1])  # Exclude current value
        std = np.std(history[:-1])
        q1 = np.percentile(history[:-1], 25)
        q3 = np.percentile(history[:-1], 75)
        iqr = q3 - q1

        # Z-score test
        z_score = abs((value - mean) / (std + 1e-10))
        is_outlier_zscore = z_score > self.config.outlier_std_threshold

        # IQR test
        lower_bound = q1 - self.config.outlier_iqr_multiplier * iqr
        upper_bound = q3 + self.config.outlier_iqr_multiplier * iqr
        is_outlier_iqr = value < lower_bound or value > upper_bound

        # Detect anomaly
        if is_outlier_zscore and is_outlier_iqr:
            severity = 'high' if z_score > 5 else 'medium'

            anomaly = DataAnomaly(
                timestamp=timestamp,
                instrument=instrument,
                anomaly_type=AnomalyType.OUTLIER,
                field=field,
                value=value,
                expected_range=(lower_bound, upper_bound),
                severity=severity,
                description=f"Outlier detected: {field}={value:.4f} (z-score={z_score:.2f})"
            )

            self.anomalies.append(anomaly)

            # Generate alert for high severity
            if severity == 'high':
                self.alerts.append(DataRiskAlert(
                    timestamp=timestamp,
                    severity='warning',
                    alert_type='data_anomaly',
                    instrument=instrument,
                    source=None,
                    message=f"High severity anomaly in {field}: {value:.4f}",
                    value=z_score,
                    threshold=self.config.outlier_std_threshold
                ))

            return anomaly

        return None

    def _detect_data_anomalies(
        self,
        timestamp: datetime,
        instrument: str,
        data: pd.DataFrame
    ):
        """Detect anomalies in data update"""
        if data.empty:
            return

        # Check common fields
        for field in ['Close', 'Open', 'High', 'Low', 'Volume']:
            if field in data.columns:
                latest_value = data[field].iloc[-1]
                if not pd.isna(latest_value):
                    self.detect_anomalies(timestamp, instrument, field, latest_value)

    def analyze_missing_data(
        self,
        timestamp: datetime,
        instrument: str,
        data: pd.DataFrame,
        field: str
    ) -> Optional[MissingDataPattern]:
        """
        Analyze missing data patterns

        Args:
            timestamp: Timestamp
            instrument: Instrument symbol
            data: Data to analyze
            field: Field to check

        Returns:
            MissingDataPattern analysis
        """
        if field not in data.columns:
            return None

        series = data[field]
        missing_mask = series.isna()

        missing_count = missing_mask.sum()
        total_count = len(series)
        missing_pct = (missing_count / total_count) * 100 if total_count > 0 else 0.0

        # Find consecutive missing values
        consecutive_missing = 0
        current_consecutive = 0
        for is_missing in missing_mask:
            if is_missing:
                current_consecutive += 1
                consecutive_missing = max(consecutive_missing, current_consecutive)
            else:
                current_consecutive = 0

        # Determine pattern type
        if missing_count == 0:
            pattern_type = 'none'
        elif consecutive_missing == missing_count:
            pattern_type = 'consecutive'
        elif consecutive_missing > 5:
            pattern_type = 'clustered'
        else:
            pattern_type = 'random'

        pattern = MissingDataPattern(
            timestamp=timestamp,
            instrument=instrument,
            field=field,
            missing_count=missing_count,
            total_count=total_count,
            missing_pct=missing_pct,
            consecutive_missing=consecutive_missing,
            pattern_type=pattern_type
        )

        # Check for alerts
        if consecutive_missing > self.config.max_consecutive_missing:
            self.alerts.append(DataRiskAlert(
                timestamp=timestamp,
                severity='warning',
                alert_type='missing_data',
                instrument=instrument,
                source=None,
                message=f"Consecutive missing data in {field}: {consecutive_missing} periods",
                value=float(consecutive_missing),
                threshold=float(self.config.max_consecutive_missing)
            ))

        return pattern

    def record_source_request(
        self,
        source: str,
        success: bool,
        response_time_ms: float
    ):
        """
        Record data source request

        Args:
            source: Data source name
            success: Whether request succeeded
            response_time_ms: Response time in milliseconds
        """
        stats = self.source_stats[source]

        stats['requests'] += 1
        stats['response_times'].append(response_time_ms)

        if success:
            stats['last_success'] = datetime.now()
            stats['consecutive_failures'] = 0
        else:
            stats['errors'] += 1
            stats['consecutive_failures'] += 1

            # Check for critical failures
            if stats['consecutive_failures'] >= self.config.max_consecutive_failures:
                self.alerts.append(DataRiskAlert(
                    timestamp=datetime.now(),
                    severity='critical',
                    alert_type='source_failure',
                    instrument=None,
                    source=source,
                    message=f"Source {source} has {stats['consecutive_failures']} consecutive failures",
                    value=float(stats['consecutive_failures']),
                    threshold=float(self.config.max_consecutive_failures),
                    requires_fallback=True
                ))

    def calculate_source_reliability(
        self,
        source: str,
        lookback_hours: Optional[int] = None
    ) -> DataSourceReliability:
        """
        Calculate data source reliability metrics

        Args:
            source: Data source name
            lookback_hours: Hours to look back (optional)

        Returns:
            DataSourceReliability metrics
        """
        stats = self.source_stats[source]

        total_requests = stats['requests']
        error_count = stats['errors']

        # Calculate error rate
        error_rate = error_count / total_requests if total_requests > 0 else 0.0

        # Calculate uptime
        uptime_pct = ((total_requests - error_count) / total_requests * 100) if total_requests > 0 else 0.0

        # Average response time
        response_times = list(stats['response_times'])
        avg_response_time = np.mean(response_times) if response_times else 0.0

        # Determine status
        if stats['consecutive_failures'] >= self.config.max_consecutive_failures:
            status = DataSourceStatus.OFFLINE
        elif error_rate > self.config.max_error_rate:
            status = DataSourceStatus.DEGRADED
        elif uptime_pct < self.config.min_uptime_pct:
            status = DataSourceStatus.DEGRADED
        else:
            status = DataSourceStatus.ONLINE

        return DataSourceReliability(
            timestamp=datetime.now(),
            source=source,
            uptime_pct=uptime_pct,
            error_count=error_count,
            total_requests=total_requests,
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time,
            status=status,
            last_successful_update=stats['last_success'],
            consecutive_failures=stats['consecutive_failures']
        )

    def execute_fallback_protocol(
        self,
        timestamp: datetime,
        trigger_reason: str,
        primary_source: str,
        fallback_source: str,
        instrument: str
    ) -> FallbackProtocol:
        """
        Execute fallback protocol to alternative data source

        Args:
            timestamp: Timestamp
            trigger_reason: Reason for fallback
            primary_source: Primary data source
            fallback_source: Fallback data source
            instrument: Instrument symbol

        Returns:
            FallbackProtocol record
        """
        # In production, this would actually fetch from fallback source
        # For now, we simulate
        success = True
        fallback_quality_score = 85.0  # Simulated

        protocol = FallbackProtocol(
            timestamp=timestamp,
            trigger_reason=trigger_reason,
            primary_source=primary_source,
            fallback_source=fallback_source,
            instrument=instrument,
            success=success,
            fallback_quality_score=fallback_quality_score
        )

        self.active_fallbacks.append(protocol)

        logger.warning(
            f"Executed fallback protocol: {primary_source} -> {fallback_source} "
            f"for {instrument}. Reason: {trigger_reason}"
        )

        return protocol

    def _check_quality_alerts(self, score: DataQualityScore):
        """Check quality score for alerts"""
        if score.composite_score < self.config.critical_quality_score:
            self.alerts.append(DataRiskAlert(
                timestamp=score.timestamp,
                severity='critical',
                alert_type='critical_quality',
                instrument=score.instrument,
                source=score.source,
                message=f"Critical data quality: {score.composite_score:.1f}",
                value=score.composite_score,
                threshold=self.config.critical_quality_score,
                requires_fallback=True
            ))
        elif score.composite_score < self.config.min_quality_score:
            self.alerts.append(DataRiskAlert(
                timestamp=score.timestamp,
                severity='warning',
                alert_type='low_quality',
                instrument=score.instrument,
                source=score.source,
                message=f"Low data quality: {score.composite_score:.1f}",
                value=score.composite_score,
                threshold=self.config.min_quality_score
            ))

    def _check_latency_alerts(self, latency: DataLatency):
        """Check latency for alerts"""
        if latency.latency_seconds > self.config.critical_latency_seconds:
            self.alerts.append(DataRiskAlert(
                timestamp=latency.timestamp,
                severity='critical',
                alert_type='critical_latency',
                instrument=latency.instrument,
                source=latency.source,
                message=f"Critical data latency: {latency.latency_seconds:.0f}s",
                value=latency.latency_seconds,
                threshold=self.config.critical_latency_seconds
            ))
        elif latency.latency_seconds > self.config.max_latency_seconds:
            self.alerts.append(DataRiskAlert(
                timestamp=latency.timestamp,
                severity='warning',
                alert_type='high_latency',
                instrument=latency.instrument,
                source=latency.source,
                message=f"High data latency: {latency.latency_seconds:.0f}s",
                value=latency.latency_seconds,
                threshold=self.config.max_latency_seconds
            ))

        # Check staleness
        if latency.staleness_hours > self.config.max_staleness_hours:
            self.alerts.append(DataRiskAlert(
                timestamp=latency.timestamp,
                severity='warning',
                alert_type='stale_data',
                instrument=latency.instrument,
                source=latency.source,
                message=f"Stale data: {latency.staleness_hours:.1f} hours old",
                value=latency.staleness_hours,
                threshold=self.config.max_staleness_hours
            ))

    def generate_data_risk_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> DataRiskReport:
        """
        Generate comprehensive data risk report

        Args:
            period_start: Report period start (optional)
            period_end: Report period end (optional)

        Returns:
            DataRiskReport with all metrics
        """
        if period_end is None:
            period_end = datetime.now()
        if period_start is None:
            period_start = period_end - timedelta(hours=24)

        # Quality summary
        quality_summary = {}
        for key, scores in self.quality_scores.items():
            recent_scores = [
                s.composite_score for s in scores
                if period_start <= s.timestamp <= period_end
            ]
            if recent_scores:
                quality_summary[key] = np.mean(recent_scores)

        # Source reliability
        source_reliability = {}
        for source in self.source_stats.keys():
            source_reliability[source] = self.calculate_source_reliability(source)

        # Recent anomalies
        recent_anomalies = [
            a for a in self.anomalies
            if period_start <= a.timestamp <= period_end
        ][-20:]  # Last 20

        # Missing data summary (placeholder)
        missing_data_summary = {}

        # Latency issues
        latency_issues = [
            l for l in self.latency_records
            if period_start <= l.timestamp <= period_end and l.is_late
        ]

        # Active fallbacks
        active_fallbacks = [
            f for f in self.active_fallbacks
            if period_start <= f.timestamp <= period_end
        ]

        # Recent alerts
        recent_alerts = [
            a for a in self.alerts
            if period_start <= a.timestamp <= period_end
        ]

        # Recommendations
        recommendations = self._generate_recommendations(
            quality_summary, source_reliability, recent_anomalies, latency_issues
        )

        return DataRiskReport(
            timestamp=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            quality_summary=quality_summary,
            source_reliability=source_reliability,
            recent_anomalies=recent_anomalies,
            missing_data_summary=missing_data_summary,
            latency_issues=latency_issues,
            active_fallbacks=active_fallbacks,
            alerts=recent_alerts,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        quality_summary: Dict,
        source_reliability: Dict,
        anomalies: List,
        latency_issues: List
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Quality issues
        low_quality = [k for k, v in quality_summary.items() if v < self.config.min_quality_score]
        if low_quality:
            recommendations.append(f"Review data quality for: {', '.join(low_quality[:3])}")

        # Source issues
        degraded_sources = [
            s for s, r in source_reliability.items()
            if r.status in [DataSourceStatus.DEGRADED, DataSourceStatus.OFFLINE]
        ]
        if degraded_sources:
            recommendations.append(f"Investigate degraded sources: {', '.join(degraded_sources)}")

        # Anomalies
        if len(anomalies) > 10:
            recommendations.append(f"High anomaly count ({len(anomalies)}) - review data pipeline")

        # Latency
        if len(latency_issues) > 5:
            recommendations.append("Address data latency issues - consider alternative sources")

        if not recommendations:
            recommendations.append("Data quality and reliability are healthy")

        return recommendations

    def get_recent_alerts(
        self,
        instrument: Optional[str] = None,
        source: Optional[str] = None,
        hours: int = 24
    ) -> List[DataRiskAlert]:
        """Get recent alerts with optional filters"""
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        if instrument:
            alerts = [a for a in alerts if a.instrument == instrument]

        if source:
            alerts = [a for a in alerts if a.source == source]

        return alerts

    def clear_old_data(self, days: int = 30):
        """Clear old data to manage memory"""
        cutoff = datetime.now() - timedelta(days=days)

        # Clear old anomalies
        self.anomalies = [a for a in self.anomalies if a.timestamp >= cutoff]

        # Clear old alerts
        self.alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        logger.info(f"Cleared data older than {days} days")

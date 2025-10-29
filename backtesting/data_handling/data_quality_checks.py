"""
Data Quality Checks

Implements comprehensive data quality validation for backtesting:
- Missing data detection and handling
- Outlier detection (statistical methods)
- Price jump detection (splits, errors)
- Volume anomaly detection
- Data consistency checks (OHLC relationships)
- Staleness detection
- Quality scoring

High-quality data is essential for reliable backtesting.
Bad data leads to bad results. This module helps identify and
handle data quality issues before they affect your backtest.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


class QualityIssueType(Enum):
    """Type of data quality issue."""
    MISSING_DATA = "MISSING_DATA"
    OUTLIER = "OUTLIER"
    PRICE_JUMP = "PRICE_JUMP"
    VOLUME_ANOMALY = "VOLUME_ANOMALY"
    OHLC_INCONSISTENCY = "OHLC_INCONSISTENCY"
    STALE_DATA = "STALE_DATA"
    DUPLICATE_DATA = "DUPLICATE_DATA"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    ZERO_VOLUME = "ZERO_VOLUME"


class Severity(Enum):
    """Severity of data quality issue."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class QualityIssue:
    """
    Data quality issue.

    Attributes:
        symbol: Trading symbol
        issue_type: Type of issue
        severity: Severity level
        timestamp: When issue occurred
        description: Issue description
        affected_fields: Fields affected
        suggested_action: Suggested remediation
    """
    symbol: str
    issue_type: QualityIssueType
    severity: Severity
    timestamp: datetime
    description: str
    affected_fields: List[str] = field(default_factory=list)
    suggested_action: str = ""


@dataclass
class QualityReport:
    """
    Data quality report.

    Attributes:
        symbol: Trading symbol
        start_date: Analysis start date
        end_date: Analysis end date
        total_records: Total number of records
        issues: List of quality issues
        quality_score: Overall quality score (0-100)
        passed: Whether data passed quality checks
    """
    symbol: str
    start_date: datetime
    end_date: datetime
    total_records: int
    issues: List[QualityIssue] = field(default_factory=list)
    quality_score: float = 100.0
    passed: bool = True

    def add_issue(self, issue: QualityIssue):
        """Add issue to report."""
        self.issues.append(issue)

        # Deduct from quality score based on severity
        if issue.severity == Severity.INFO:
            self.quality_score -= 0.1
        elif issue.severity == Severity.WARNING:
            self.quality_score -= 1.0
        elif issue.severity == Severity.ERROR:
            self.quality_score -= 5.0
        elif issue.severity == Severity.CRITICAL:
            self.quality_score -= 10.0
            self.passed = False

        self.quality_score = max(0.0, self.quality_score)


class MissingDataDetector:
    """
    Detect missing data in time series.

    Identifies:
    - Gaps in timestamps
    - Missing values (NaN)
    - Incomplete records
    """

    def __init__(self, expected_frequency: str = 'D'):
        """
        Initialize missing data detector.

        Args:
            expected_frequency: Expected data frequency
        """
        self.expected_frequency = expected_frequency
        logger.info(f"Initialized MissingDataDetector: freq={expected_frequency}")

    def check(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> List[QualityIssue]:
        """
        Check for missing data.

        Args:
            data: DataFrame to check
            symbol: Trading symbol

        Returns:
            List of quality issues
        """
        issues = []

        if data.empty:
            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.MISSING_DATA,
                severity=Severity.CRITICAL,
                timestamp=datetime.now(),
                description="DataFrame is empty",
                suggested_action="Verify data source"
            ))
            return issues

        # Check for missing timestamps
        if isinstance(data.index, pd.DatetimeIndex):
            expected_range = pd.date_range(
                start=data.index.min(),
                end=data.index.max(),
                freq=self.expected_frequency
            )

            missing_dates = expected_range.difference(data.index)

            if len(missing_dates) > 0:
                missing_pct = (len(missing_dates) / len(expected_range)) * 100

                severity = Severity.INFO
                if missing_pct > 10:
                    severity = Severity.WARNING
                if missing_pct > 25:
                    severity = Severity.ERROR

                issues.append(QualityIssue(
                    symbol=symbol,
                    issue_type=QualityIssueType.MISSING_DATA,
                    severity=severity,
                    timestamp=missing_dates[0] if len(missing_dates) > 0 else datetime.now(),
                    description=f"Missing {len(missing_dates)} timestamps ({missing_pct:.1f}%)",
                    suggested_action="Use forward-fill or interpolation"
                ))

        # Check for NaN values
        for col in data.columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                nan_pct = (nan_count / len(data)) * 100

                severity = Severity.WARNING
                if nan_pct > 10:
                    severity = Severity.ERROR

                issues.append(QualityIssue(
                    symbol=symbol,
                    issue_type=QualityIssueType.MISSING_DATA,
                    severity=severity,
                    timestamp=data[data[col].isna()].index[0],
                    description=f"Column '{col}' has {nan_count} NaN values ({nan_pct:.1f}%)",
                    affected_fields=[col],
                    suggested_action="Fill or drop missing values"
                ))

        return issues


class OutlierDetector:
    """
    Detect statistical outliers in data.

    Uses multiple methods:
    - Z-score (standard deviations from mean)
    - IQR (interquartile range)
    - Modified Z-score (using median)
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        use_log_returns: bool = True
    ):
        """
        Initialize outlier detector.

        Args:
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            use_log_returns: Use log returns for price analysis
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.use_log_returns = use_log_returns

        logger.info(
            f"Initialized OutlierDetector: "
            f"z={z_threshold}, iqr={iqr_multiplier}"
        )

    def check(
        self,
        data: pd.DataFrame,
        symbol: str,
        columns: Optional[List[str]] = None
    ) -> List[QualityIssue]:
        """
        Check for outliers.

        Args:
            data: DataFrame to check
            symbol: Trading symbol
            columns: Columns to check (None = numeric columns)

        Returns:
            List of quality issues
        """
        issues = []

        if data.empty:
            return issues

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in data.columns:
                continue

            series = data[col].dropna()

            if len(series) < 10:  # Not enough data
                continue

            # For price columns, check returns instead
            if col in ['open', 'high', 'low', 'close', 'price']:
                if self.use_log_returns:
                    series = np.log(series / series.shift(1)).dropna()
                else:
                    series = series.pct_change().dropna()

            # Z-score method
            z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
            outliers_z = series[z_scores > self.z_threshold]

            if len(outliers_z) > 0:
                for idx in outliers_z.index:
                    issues.append(QualityIssue(
                        symbol=symbol,
                        issue_type=QualityIssueType.OUTLIER,
                        severity=Severity.WARNING,
                        timestamp=idx,
                        description=f"Outlier in '{col}': {data.loc[idx, col]:.4f} (Z={z_scores[series.index.get_loc(idx)]:.2f})",
                        affected_fields=[col],
                        suggested_action="Review for data errors or legitimate events"
                    ))

            # IQR method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            outliers_iqr = series[(series < lower_bound) | (series > upper_bound)]

            # Report severe IQR outliers not already caught by Z-score
            severe_outliers = outliers_iqr[
                ~outliers_iqr.index.isin(outliers_z.index)
            ]

            if len(severe_outliers) > 0:
                # Report up to 5 worst
                worst = severe_outliers.nlargest(5)
                for idx in worst.index:
                    issues.append(QualityIssue(
                        symbol=symbol,
                        issue_type=QualityIssueType.OUTLIER,
                        severity=Severity.INFO,
                        timestamp=idx,
                        description=f"IQR outlier in '{col}': {data.loc[idx, col]:.4f}",
                        affected_fields=[col],
                        suggested_action="Review for data errors"
                    ))

        return issues


class PriceJumpDetector:
    """
    Detect unusual price jumps.

    Identifies:
    - Potential data errors
    - Unaccounted stock splits
    - Fat-finger errors
    """

    def __init__(
        self,
        jump_threshold: float = 0.50,  # 50% jump
        split_threshold: float = 0.90  # 90% drop/gain (likely split)
    ):
        """
        Initialize price jump detector.

        Args:
            jump_threshold: Threshold for unusual jumps
            split_threshold: Threshold for potential splits
        """
        self.jump_threshold = jump_threshold
        self.split_threshold = split_threshold

        logger.info(
            f"Initialized PriceJumpDetector: "
            f"jump={jump_threshold*100:.0f}%, split={split_threshold*100:.0f}%"
        )

    def check(
        self,
        data: pd.DataFrame,
        symbol: str,
        price_column: str = 'close'
    ) -> List[QualityIssue]:
        """
        Check for price jumps.

        Args:
            data: DataFrame to check
            symbol: Trading symbol
            price_column: Price column to check

        Returns:
            List of quality issues
        """
        issues = []

        if data.empty or price_column not in data.columns:
            return issues

        prices = data[price_column].dropna()

        if len(prices) < 2:
            return issues

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Find large jumps
        large_jumps = returns[np.abs(returns) > self.jump_threshold]

        for idx in large_jumps.index:
            ret = returns[idx]

            # Classify as split or error
            if np.abs(ret) > self.split_threshold:
                issue_type = QualityIssueType.PRICE_JUMP
                severity = Severity.ERROR
                description = f"Potential unadjusted split: {ret*100:.1f}% change"
                action = "Check for stock splits and adjust data"
            else:
                issue_type = QualityIssueType.PRICE_JUMP
                severity = Severity.WARNING
                description = f"Large price jump: {ret*100:.1f}% change"
                action = "Review for data errors or legitimate events"

            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=issue_type,
                severity=severity,
                timestamp=idx,
                description=description,
                affected_fields=[price_column],
                suggested_action=action
            ))

        return issues


class VolumeAnomalyDetector:
    """
    Detect volume anomalies.

    Unusual volume can indicate:
    - Data errors
    - Corporate events
    - Market activity spikes
    """

    def __init__(
        self,
        spike_threshold: float = 10.0,  # 10x average
        zero_volume_severity: Severity = Severity.WARNING
    ):
        """
        Initialize volume anomaly detector.

        Args:
            spike_threshold: Multiplier for volume spike
            zero_volume_severity: Severity for zero volume
        """
        self.spike_threshold = spike_threshold
        self.zero_volume_severity = zero_volume_severity

        logger.info(
            f"Initialized VolumeAnomalyDetector: spike={spike_threshold}x"
        )

    def check(
        self,
        data: pd.DataFrame,
        symbol: str,
        volume_column: str = 'volume'
    ) -> List[QualityIssue]:
        """
        Check for volume anomalies.

        Args:
            data: DataFrame to check
            symbol: Trading symbol
            volume_column: Volume column to check

        Returns:
            List of quality issues
        """
        issues = []

        if data.empty or volume_column not in data.columns:
            return issues

        volume = data[volume_column].dropna()

        if len(volume) == 0:
            return issues

        # Check for zero volume
        zero_volume = volume[volume == 0]
        if len(zero_volume) > 0:
            zero_pct = (len(zero_volume) / len(volume)) * 100

            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.ZERO_VOLUME,
                severity=self.zero_volume_severity,
                timestamp=zero_volume.index[0],
                description=f"Zero volume on {len(zero_volume)} days ({zero_pct:.1f}%)",
                affected_fields=[volume_column],
                suggested_action="Review for holidays or data errors"
            ))

        # Check for volume spikes
        avg_volume = volume.rolling(window=20, min_periods=5).mean()
        volume_ratio = volume / avg_volume

        spikes = volume_ratio[volume_ratio > self.spike_threshold]

        for idx in spikes.index:
            ratio = volume_ratio[idx]
            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.VOLUME_ANOMALY,
                severity=Severity.INFO,
                timestamp=idx,
                description=f"Volume spike: {ratio:.1f}x average ({volume[idx]:,.0f})",
                affected_fields=[volume_column],
                suggested_action="Review for corporate events or data errors"
            ))

        return issues


class OHLCConsistencyChecker:
    """
    Check OHLC (Open-High-Low-Close) consistency.

    Validates:
    - High >= Low
    - High >= Open, Close
    - Low <= Open, Close
    """

    def __init__(self):
        """Initialize OHLC consistency checker."""
        logger.info("Initialized OHLCConsistencyChecker")

    def check(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> List[QualityIssue]:
        """
        Check OHLC consistency.

        Args:
            data: DataFrame to check
            symbol: Trading symbol

        Returns:
            List of quality issues
        """
        issues = []

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return issues

        # Check: High >= Low
        violations = data[data['high'] < data['low']]
        for idx in violations.index:
            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.OHLC_INCONSISTENCY,
                severity=Severity.CRITICAL,
                timestamp=idx,
                description=f"High < Low: H={data.loc[idx, 'high']:.2f}, L={data.loc[idx, 'low']:.2f}",
                affected_fields=['high', 'low'],
                suggested_action="Fix data error"
            ))

        # Check: High >= Open
        violations = data[data['high'] < data['open']]
        for idx in violations.index:
            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.OHLC_INCONSISTENCY,
                severity=Severity.ERROR,
                timestamp=idx,
                description=f"High < Open: H={data.loc[idx, 'high']:.2f}, O={data.loc[idx, 'open']:.2f}",
                affected_fields=['high', 'open'],
                suggested_action="Fix data error"
            ))

        # Check: High >= Close
        violations = data[data['high'] < data['close']]
        for idx in violations.index:
            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.OHLC_INCONSISTENCY,
                severity=Severity.ERROR,
                timestamp=idx,
                description=f"High < Close: H={data.loc[idx, 'high']:.2f}, C={data.loc[idx, 'close']:.2f}",
                affected_fields=['high', 'close'],
                suggested_action="Fix data error"
            ))

        # Check: Low <= Open
        violations = data[data['low'] > data['open']]
        for idx in violations.index:
            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.OHLC_INCONSISTENCY,
                severity=Severity.ERROR,
                timestamp=idx,
                description=f"Low > Open: L={data.loc[idx, 'low']:.2f}, O={data.loc[idx, 'open']:.2f}",
                affected_fields=['low', 'open'],
                suggested_action="Fix data error"
            ))

        # Check: Low <= Close
        violations = data[data['low'] > data['close']]
        for idx in violations.index:
            issues.append(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.OHLC_INCONSISTENCY,
                severity=Severity.ERROR,
                timestamp=idx,
                description=f"Low > Close: L={data.loc[idx, 'low']:.2f}, C={data.loc[idx, 'close']:.2f}",
                affected_fields=['low', 'close'],
                suggested_action="Fix data error"
            ))

        return issues


class StaleDataDetector:
    """
    Detect stale (unchanged) data.

    Identifies periods where data hasn't updated,
    which may indicate feed issues.
    """

    def __init__(self, max_unchanged_periods: int = 5):
        """
        Initialize stale data detector.

        Args:
            max_unchanged_periods: Maximum consecutive unchanged periods
        """
        self.max_unchanged_periods = max_unchanged_periods
        logger.info(
            f"Initialized StaleDataDetector: "
            f"max_unchanged={max_unchanged_periods}"
        )

    def check(
        self,
        data: pd.DataFrame,
        symbol: str,
        columns: Optional[List[str]] = None
    ) -> List[QualityIssue]:
        """
        Check for stale data.

        Args:
            data: DataFrame to check
            symbol: Trading symbol
            columns: Columns to check

        Returns:
            List of quality issues
        """
        issues = []

        if data.empty:
            return issues

        if columns is None:
            columns = ['close'] if 'close' in data.columns else data.columns.tolist()

        for col in columns:
            if col not in data.columns:
                continue

            series = data[col]

            # Find consecutive unchanged values
            unchanged = (series == series.shift(1))
            unchanged_runs = unchanged.groupby((unchanged != unchanged.shift()).cumsum()).sum()

            long_runs = unchanged_runs[unchanged_runs > self.max_unchanged_periods]

            if len(long_runs) > 0:
                issues.append(QualityIssue(
                    symbol=symbol,
                    issue_type=QualityIssueType.STALE_DATA,
                    severity=Severity.WARNING,
                    timestamp=datetime.now(),
                    description=f"Column '{col}' has {len(long_runs)} periods with >{self.max_unchanged_periods} consecutive unchanged values",
                    affected_fields=[col],
                    suggested_action="Check data feed for staleness"
                ))

        return issues


class ComprehensiveDataQualityChecker:
    """
    Comprehensive data quality checker.

    Runs all quality checks and generates detailed report.
    """

    def __init__(self):
        """Initialize comprehensive checker."""
        self.missing_detector = MissingDataDetector()
        self.outlier_detector = OutlierDetector()
        self.price_jump_detector = PriceJumpDetector()
        self.volume_detector = VolumeAnomalyDetector()
        self.ohlc_checker = OHLCConsistencyChecker()
        self.stale_detector = StaleDataDetector()

        logger.info("Initialized ComprehensiveDataQualityChecker")

    def check(
        self,
        data: pd.DataFrame,
        symbol: str,
        checks: Optional[List[str]] = None
    ) -> QualityReport:
        """
        Run comprehensive quality checks.

        Args:
            data: DataFrame to check
            symbol: Trading symbol
            checks: Specific checks to run (None = all)

        Returns:
            QualityReport object
        """
        if data.empty:
            report = QualityReport(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_records=0,
                quality_score=0.0,
                passed=False
            )
            report.add_issue(QualityIssue(
                symbol=symbol,
                issue_type=QualityIssueType.MISSING_DATA,
                severity=Severity.CRITICAL,
                timestamp=datetime.now(),
                description="No data available",
                suggested_action="Check data source"
            ))
            return report

        start_date = data.index.min() if isinstance(data.index, pd.DatetimeIndex) else datetime.now()
        end_date = data.index.max() if isinstance(data.index, pd.DatetimeIndex) else datetime.now()

        report = QualityReport(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_records=len(data)
        )

        # Run checks
        all_checks = {
            'missing': self.missing_detector,
            'outliers': self.outlier_detector,
            'price_jumps': self.price_jump_detector,
            'volume': self.volume_detector,
            'ohlc': self.ohlc_checker,
            'stale': self.stale_detector
        }

        checks_to_run = checks if checks else all_checks.keys()

        for check_name in checks_to_run:
            if check_name in all_checks:
                detector = all_checks[check_name]
                try:
                    issues = detector.check(data, symbol)
                    for issue in issues:
                        report.add_issue(issue)
                except Exception as e:
                    logger.error(f"Error running {check_name} check: {e}")

        # Log summary
        critical_count = sum(1 for i in report.issues if i.severity == Severity.CRITICAL)
        error_count = sum(1 for i in report.issues if i.severity == Severity.ERROR)
        warning_count = sum(1 for i in report.issues if i.severity == Severity.WARNING)

        logger.info(
            f"Quality check for {symbol}: "
            f"score={report.quality_score:.1f}, "
            f"critical={critical_count}, errors={error_count}, warnings={warning_count}"
        )

        return report

    def check_multiple(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, QualityReport]:
        """
        Check multiple symbols.

        Args:
            data_dict: Dictionary of DataFrames

        Returns:
            Dictionary of QualityReports
        """
        logger.info(f"Running quality checks on {len(data_dict)} symbols")

        reports = {}
        for symbol, data in data_dict.items():
            reports[symbol] = self.check(data, symbol)

        # Summary statistics
        passed_count = sum(1 for r in reports.values() if r.passed)
        avg_score = np.mean([r.quality_score for r in reports.values()])

        logger.info(
            f"Quality check summary: {passed_count}/{len(reports)} passed, "
            f"avg score={avg_score:.1f}"
        )

        return reports

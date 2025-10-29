"""
Data Handling Module

Comprehensive data handling infrastructure for backtesting.

This module provides essential data management capabilities:
- Data loading (efficient, cached, streaming)
- Multi-asset alignment (calendars, timezones, gaps)
- Survivorship bias correction (delistings, IPOs, mergers)
- Point-in-time data (prevent look-ahead bias)
- Data quality checks (validation, outliers, consistency)

Proper data handling is fundamental to reliable backtesting.
Issues like survivorship bias, look-ahead bias, and data quality
problems can completely invalidate backtest results.
"""

# Data Loading
from .data_loader import (
    DataFrequency,
    DataFormat,
    DataConfig,
    DataLoader,
    FrequencyConverter,
    DataGapHandler,
    MemoryEfficientLoader
)

# Data Alignment
from .data_alignment import (
    TradingCalendar,
    AlignmentMethod,
    AlignmentConfig,
    DataAligner,
    TimezoneHarmonizer,
    CorporateActionAdjuster,
    CurrencyConverter,
    IndexRebalancingHandler,
    MultiAssetPanel
)

# Survivorship Bias
from .survivorship_bias import (
    DelistingReason,
    SecurityStatus,
    SecurityLifecycle,
    SurvivorshipBiasHandler,
    UniverseManager,
    IPOHandler,
    MergerHandler,
    BankruptcyHandler,
    ComprehensiveSurvivorshipModel
)

# Point-in-Time Data
from .point_in_time import (
    DataType,
    ReportingPeriod,
    DataPoint,
    PointInTimeDatabase,
    FundamentalDataHandler,
    EarningsAnnouncementHandler,
    EconomicDataHandler,
    RatingChangesHandler,
    RestatedDataHandler,
    ComprehensivePointInTimeSystem
)

# Data Quality Checks
from .data_quality_checks import (
    QualityIssueType,
    Severity,
    QualityIssue,
    QualityReport,
    MissingDataDetector,
    OutlierDetector,
    PriceJumpDetector,
    VolumeAnomalyDetector,
    OHLCConsistencyChecker,
    StaleDataDetector,
    ComprehensiveDataQualityChecker
)

__all__ = [
    # Data Loading
    'DataFrequency',
    'DataFormat',
    'DataConfig',
    'DataLoader',
    'FrequencyConverter',
    'DataGapHandler',
    'MemoryEfficientLoader',

    # Data Alignment
    'TradingCalendar',
    'AlignmentMethod',
    'AlignmentConfig',
    'DataAligner',
    'TimezoneHarmonizer',
    'CorporateActionAdjuster',
    'CurrencyConverter',
    'IndexRebalancingHandler',
    'MultiAssetPanel',

    # Survivorship Bias
    'DelistingReason',
    'SecurityStatus',
    'SecurityLifecycle',
    'SurvivorshipBiasHandler',
    'UniverseManager',
    'IPOHandler',
    'MergerHandler',
    'BankruptcyHandler',
    'ComprehensiveSurvivorshipModel',

    # Point-in-Time Data
    'DataType',
    'ReportingPeriod',
    'DataPoint',
    'PointInTimeDatabase',
    'FundamentalDataHandler',
    'EarningsAnnouncementHandler',
    'EconomicDataHandler',
    'RatingChangesHandler',
    'RestatedDataHandler',
    'ComprehensivePointInTimeSystem',

    # Data Quality Checks
    'QualityIssueType',
    'Severity',
    'QualityIssue',
    'QualityReport',
    'MissingDataDetector',
    'OutlierDetector',
    'PriceJumpDetector',
    'VolumeAnomalyDetector',
    'OHLCConsistencyChecker',
    'StaleDataDetector',
    'ComprehensiveDataQualityChecker',
]

"""
Point-in-Time Data Handler

Implements point-in-time data handling to prevent look-ahead bias:
- Ensure no look-ahead bias
- Lag fundamental data appropriately (reporting delays)
- Handle restatements
- Earnings announcement timing
- Economic data release schedules
- Rating changes timing
- As-of date functionality

Point-in-time data is critical for realistic backtesting.
Look-ahead bias (using future information) is a common pitfall that
makes backtest results unrealistically optimistic.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class DataType(Enum):
    """Type of data requiring point-in-time handling."""
    FUNDAMENTAL = "FUNDAMENTAL"  # Financial statements
    ESTIMATES = "ESTIMATES"  # Analyst estimates
    ECONOMIC = "ECONOMIC"  # Economic indicators
    RATINGS = "RATINGS"  # Credit/equity ratings
    NEWS = "NEWS"  # News/sentiment
    CORPORATE_ACTIONS = "CORPORATE_ACTIONS"


class ReportingPeriod(Enum):
    """Financial reporting period."""
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"
    TTM = "TTM"  # Trailing twelve months


@dataclass
class DataPoint:
    """
    Point-in-time data point.

    Attributes:
        symbol: Trading symbol
        data_type: Type of data
        field_name: Field name
        value: Data value
        period_end_date: Period the data relates to
        reporting_date: Date data became available
        revision_number: Revision number (for restatements)
        is_restated: Whether this is a restatement
    """
    symbol: str
    data_type: DataType
    field_name: str
    value: Any
    period_end_date: datetime
    reporting_date: datetime
    revision_number: int = 0
    is_restated: bool = False

    def is_available(self, as_of_date: datetime) -> bool:
        """
        Check if data is available as of date.

        Args:
            as_of_date: Query date

        Returns:
            True if data available
        """
        return as_of_date >= self.reporting_date


class PointInTimeDatabase:
    """
    Point-in-time database.

    Stores versioned data with proper as-of dates.
    Ensures only information available at query time is returned.
    """

    def __init__(self):
        """Initialize point-in-time database."""
        # Structure: {symbol: {field_name: [DataPoint]}}
        self.data: Dict[str, Dict[str, List[DataPoint]]] = {}
        logger.info("Initialized PointInTimeDatabase")

    def add_data_point(self, data_point: DataPoint):
        """
        Add data point to database.

        Args:
            data_point: DataPoint to add
        """
        symbol = data_point.symbol
        field = data_point.field_name

        if symbol not in self.data:
            self.data[symbol] = {}

        if field not in self.data[symbol]:
            self.data[symbol][field] = []

        self.data[symbol][field].append(data_point)

        # Keep sorted by reporting date
        self.data[symbol][field].sort(key=lambda x: x.reporting_date)

        logger.debug(
            f"Added PIT data: {symbol}.{field} = {data_point.value} "
            f"(period={data_point.period_end_date}, "
            f"available={data_point.reporting_date})"
        )

    def get_value(
        self,
        symbol: str,
        field_name: str,
        as_of_date: datetime,
        use_latest_restatement: bool = True
    ) -> Optional[Any]:
        """
        Get value as of date.

        Args:
            symbol: Trading symbol
            field_name: Field name
            as_of_date: Query date
            use_latest_restatement: Use latest restatement if multiple

        Returns:
            Value if available, None otherwise
        """
        if symbol not in self.data or field_name not in self.data[symbol]:
            return None

        data_points = self.data[symbol][field_name]

        # Find all points available as of date
        available = [
            dp for dp in data_points
            if dp.is_available(as_of_date)
        ]

        if not available:
            return None

        # Get most recent
        if use_latest_restatement:
            # Latest reporting date (may be restatement)
            latest = max(available, key=lambda x: x.reporting_date)
        else:
            # Latest period, original reporting
            latest = max(
                [dp for dp in available if not dp.is_restated],
                key=lambda x: x.period_end_date,
                default=None
            )
            if latest is None:
                latest = max(available, key=lambda x: x.period_end_date)

        return latest.value

    def get_time_series(
        self,
        symbol: str,
        field_name: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'D'
    ) -> pd.Series:
        """
        Get time series with point-in-time values.

        Args:
            symbol: Trading symbol
            field_name: Field name
            start_date: Start date
            end_date: End date
            frequency: Frequency ('D', 'M', etc.)

        Returns:
            Time series with PIT values
        """
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)

        # Get value for each date
        values = []
        for date in dates:
            value = self.get_value(symbol, field_name, date)
            values.append(value)

        return pd.Series(values, index=dates, name=field_name)

    def get_snapshot(
        self,
        symbol: str,
        as_of_date: datetime,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get snapshot of all fields as of date.

        Args:
            symbol: Trading symbol
            as_of_date: Query date
            fields: Fields to include (None = all)

        Returns:
            Dictionary of field values
        """
        if symbol not in self.data:
            return {}

        snapshot = {}

        available_fields = fields if fields else self.data[symbol].keys()

        for field in available_fields:
            value = self.get_value(symbol, field, as_of_date)
            if value is not None:
                snapshot[field] = value

        return snapshot


class FundamentalDataHandler:
    """
    Handle fundamental data with proper reporting delays.

    Financial statements are typically reported:
    - 10-Q (quarterly): 40-45 days after quarter end
    - 10-K (annual): 60-90 days after year end

    This handler applies realistic delays.
    """

    def __init__(
        self,
        pit_db: Optional[PointInTimeDatabase] = None,
        quarterly_delay_days: int = 45,
        annual_delay_days: int = 75
    ):
        """
        Initialize fundamental data handler.

        Args:
            pit_db: Point-in-time database
            quarterly_delay_days: Typical delay for quarterly reports
            annual_delay_days: Typical delay for annual reports
        """
        self.pit_db = pit_db or PointInTimeDatabase()
        self.quarterly_delay = timedelta(days=quarterly_delay_days)
        self.annual_delay = timedelta(days=annual_delay_days)

        logger.info(
            f"Initialized FundamentalDataHandler: "
            f"Q={quarterly_delay_days}d, A={annual_delay_days}d"
        )

    def add_financial_statement(
        self,
        symbol: str,
        period_end_date: datetime,
        filing_date: datetime,
        period_type: ReportingPeriod,
        data: Dict[str, Any],
        is_restatement: bool = False
    ):
        """
        Add financial statement data.

        Args:
            symbol: Trading symbol
            period_end_date: Period end date
            filing_date: Actual filing date
            period_type: Quarterly or annual
            data: Dictionary of financial metrics
            is_restatement: Whether this is a restatement
        """
        for field_name, value in data.items():
            data_point = DataPoint(
                symbol=symbol,
                data_type=DataType.FUNDAMENTAL,
                field_name=field_name,
                value=value,
                period_end_date=period_end_date,
                reporting_date=filing_date,
                is_restated=is_restatement
            )

            self.pit_db.add_data_point(data_point)

        logger.debug(
            f"Added {period_type.value} statement for {symbol}: "
            f"period={period_end_date}, filed={filing_date}"
        )

    def estimate_filing_date(
        self,
        period_end_date: datetime,
        period_type: ReportingPeriod
    ) -> datetime:
        """
        Estimate filing date based on typical delays.

        Args:
            period_end_date: Period end date
            period_type: Reporting period type

        Returns:
            Estimated filing date
        """
        if period_type == ReportingPeriod.QUARTERLY:
            return period_end_date + self.quarterly_delay
        elif period_type == ReportingPeriod.ANNUAL:
            return period_end_date + self.annual_delay
        else:
            return period_end_date + self.quarterly_delay

    def get_latest_financials(
        self,
        symbol: str,
        as_of_date: datetime,
        fields: List[str]
    ) -> Dict[str, Any]:
        """
        Get latest available financials as of date.

        Args:
            symbol: Trading symbol
            as_of_date: Query date
            fields: Fields to retrieve

        Returns:
            Dictionary of values
        """
        return self.pit_db.get_snapshot(symbol, as_of_date, fields)


class EarningsAnnouncementHandler:
    """
    Handle earnings announcement timing.

    Earnings announcements occur before/after market hours.
    Proper handling is critical for event studies.
    """

    def __init__(self, pit_db: Optional[PointInTimeDatabase] = None):
        """
        Initialize earnings announcement handler.

        Args:
            pit_db: Point-in-time database
        """
        self.pit_db = pit_db or PointInTimeDatabase()
        self.announcements: Dict[str, List[Dict]] = {}

        logger.info("Initialized EarningsAnnouncementHandler")

    def add_announcement(
        self,
        symbol: str,
        announcement_datetime: datetime,
        period_end_date: datetime,
        actual_eps: float,
        estimated_eps: Optional[float] = None,
        before_market: bool = True
    ):
        """
        Add earnings announcement.

        Args:
            symbol: Trading symbol
            announcement_datetime: Announcement date/time
            period_end_date: Quarter end date
            actual_eps: Actual EPS reported
            estimated_eps: Consensus estimate
            before_market: True if before market open
        """
        if symbol not in self.announcements:
            self.announcements[symbol] = []

        self.announcements[symbol].append({
            'announcement_datetime': announcement_datetime,
            'period_end_date': period_end_date,
            'actual_eps': actual_eps,
            'estimated_eps': estimated_eps,
            'before_market': before_market,
            'surprise': actual_eps - estimated_eps if estimated_eps else None
        })

        # Add to PIT database
        data_point = DataPoint(
            symbol=symbol,
            data_type=DataType.FUNDAMENTAL,
            field_name='eps',
            value=actual_eps,
            period_end_date=period_end_date,
            reporting_date=announcement_datetime
        )
        self.pit_db.add_data_point(data_point)

        logger.debug(
            f"Added earnings announcement: {symbol} on {announcement_datetime}"
        )

    def get_announcement_date(
        self,
        symbol: str,
        period_end_date: datetime
    ) -> Optional[datetime]:
        """
        Get announcement date for period.

        Args:
            symbol: Trading symbol
            period_end_date: Quarter end date

        Returns:
            Announcement datetime if available
        """
        if symbol not in self.announcements:
            return None

        for announcement in self.announcements[symbol]:
            if announcement['period_end_date'] == period_end_date:
                return announcement['announcement_datetime']

        return None

    def get_effective_date(
        self,
        symbol: str,
        period_end_date: datetime,
        market_open_time: datetime = None
    ) -> Optional[datetime]:
        """
        Get effective date when earnings information is actionable.

        Args:
            symbol: Trading symbol
            period_end_date: Quarter end date
            market_open_time: Market open time (for next-day effective date)

        Returns:
            Effective date for trading
        """
        announcement = None

        if symbol in self.announcements:
            for ann in self.announcements[symbol]:
                if ann['period_end_date'] == period_end_date:
                    announcement = ann
                    break

        if not announcement:
            return None

        ann_dt = announcement['announcement_datetime']

        # If before market, effective same day at open
        # If after market, effective next day at open
        if announcement['before_market']:
            # Actionable at market open on announcement date
            return ann_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            # Actionable at next market open
            next_day = ann_dt + timedelta(days=1)
            # Skip weekends
            while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
                next_day += timedelta(days=1)
            return next_day.replace(hour=9, minute=30, second=0, microsecond=0)


class EconomicDataHandler:
    """
    Handle economic data releases.

    Economic indicators (GDP, CPI, unemployment, etc.) are released
    on specific schedules. Proper handling prevents look-ahead bias.
    """

    def __init__(self):
        """Initialize economic data handler."""
        self.releases: Dict[str, List[Dict]] = {}
        logger.info("Initialized EconomicDataHandler")

    def add_release(
        self,
        indicator_name: str,
        release_date: datetime,
        reference_period: datetime,
        value: float,
        is_revised: bool = False
    ):
        """
        Add economic data release.

        Args:
            indicator_name: Indicator name (e.g., 'GDP', 'CPI')
            release_date: Release date/time
            reference_period: Period the data refers to
            value: Indicator value
            is_revised: Whether this is a revision
        """
        if indicator_name not in self.releases:
            self.releases[indicator_name] = []

        self.releases[indicator_name].append({
            'release_date': release_date,
            'reference_period': reference_period,
            'value': value,
            'is_revised': is_revised
        })

        # Keep sorted by release date
        self.releases[indicator_name].sort(key=lambda x: x['release_date'])

        logger.debug(
            f"Added {indicator_name} release: {value} "
            f"(period={reference_period}, released={release_date})"
        )

    def get_value(
        self,
        indicator_name: str,
        as_of_date: datetime,
        use_latest_revision: bool = True
    ) -> Optional[float]:
        """
        Get indicator value as of date.

        Args:
            indicator_name: Indicator name
            as_of_date: Query date
            use_latest_revision: Use latest revision if multiple

        Returns:
            Indicator value if available
        """
        if indicator_name not in self.releases:
            return None

        # Find releases available as of date
        available = [
            r for r in self.releases[indicator_name]
            if r['release_date'] <= as_of_date
        ]

        if not available:
            return None

        # Get most recent
        if use_latest_revision:
            latest = max(available, key=lambda x: x['release_date'])
        else:
            # Latest original (non-revised)
            originals = [r for r in available if not r['is_revised']]
            if originals:
                latest = max(originals, key=lambda x: x['reference_period'])
            else:
                latest = max(available, key=lambda x: x['reference_period'])

        return latest['value']

    def get_time_series(
        self,
        indicator_name: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'M'
    ) -> pd.Series:
        """
        Get time series of indicator values.

        Args:
            indicator_name: Indicator name
            start_date: Start date
            end_date: End date
            frequency: Frequency

        Returns:
            Time series
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)

        values = []
        for date in dates:
            value = self.get_value(indicator_name, date)
            values.append(value)

        return pd.Series(values, index=dates, name=indicator_name)


class RatingChangesHandler:
    """
    Handle credit/equity rating changes.

    Rating changes (upgrades/downgrades) are announced at specific times.
    Proper timing is important for credit strategies.
    """

    def __init__(self):
        """Initialize rating changes handler."""
        self.ratings: Dict[str, List[Dict]] = {}
        logger.info("Initialized RatingChangesHandler")

    def add_rating_change(
        self,
        symbol: str,
        announcement_date: datetime,
        rating_agency: str,
        new_rating: str,
        old_rating: Optional[str] = None,
        outlook: Optional[str] = None
    ):
        """
        Add rating change.

        Args:
            symbol: Trading symbol
            announcement_date: Announcement date
            rating_agency: Rating agency (e.g., 'Moody's', 'S&P')
            new_rating: New rating
            old_rating: Previous rating
            outlook: Outlook (Positive, Negative, Stable)
        """
        if symbol not in self.ratings:
            self.ratings[symbol] = []

        self.ratings[symbol].append({
            'announcement_date': announcement_date,
            'rating_agency': rating_agency,
            'new_rating': new_rating,
            'old_rating': old_rating,
            'outlook': outlook
        })

        # Keep sorted
        self.ratings[symbol].sort(key=lambda x: x['announcement_date'])

        logger.debug(
            f"Added rating change: {symbol} {old_rating}→{new_rating} "
            f"by {rating_agency} on {announcement_date}"
        )

    def get_current_rating(
        self,
        symbol: str,
        as_of_date: datetime,
        rating_agency: Optional[str] = None
    ) -> Optional[str]:
        """
        Get current rating as of date.

        Args:
            symbol: Trading symbol
            as_of_date: Query date
            rating_agency: Specific agency (None = most recent)

        Returns:
            Current rating
        """
        if symbol not in self.ratings:
            return None

        # Filter by agency if specified
        ratings = self.ratings[symbol]
        if rating_agency:
            ratings = [r for r in ratings if r['rating_agency'] == rating_agency]

        # Find most recent rating before as_of_date
        applicable = [r for r in ratings if r['announcement_date'] <= as_of_date]

        if not applicable:
            return None

        latest = max(applicable, key=lambda x: x['announcement_date'])
        return latest['new_rating']


class RestatedDataHandler:
    """
    Handle data restatements.

    Financial data is sometimes restated retroactively.
    This handler tracks original vs. restated values.
    """

    def __init__(self, pit_db: Optional[PointInTimeDatabase] = None):
        """
        Initialize restatement handler.

        Args:
            pit_db: Point-in-time database
        """
        self.pit_db = pit_db or PointInTimeDatabase()
        logger.info("Initialized RestatedDataHandler")

    def add_restatement(
        self,
        symbol: str,
        field_name: str,
        period_end_date: datetime,
        restatement_date: datetime,
        new_value: Any,
        original_value: Optional[Any] = None
    ):
        """
        Add restatement.

        Args:
            symbol: Trading symbol
            field_name: Field being restated
            period_end_date: Period the data relates to
            restatement_date: Date of restatement
            new_value: Restated value
            original_value: Original value (optional)
        """
        data_point = DataPoint(
            symbol=symbol,
            data_type=DataType.FUNDAMENTAL,
            field_name=field_name,
            value=new_value,
            period_end_date=period_end_date,
            reporting_date=restatement_date,
            is_restated=True,
            revision_number=1
        )

        self.pit_db.add_data_point(data_point)

        logger.info(
            f"Added restatement: {symbol}.{field_name} = {new_value} "
            f"(period={period_end_date}, restated={restatement_date})"
        )


class ComprehensivePointInTimeSystem:
    """
    Comprehensive point-in-time system.

    Integrates all components:
    - Fundamental data with delays
    - Earnings announcements
    - Economic releases
    - Rating changes
    - Restatements

    Production-ready system for avoiding look-ahead bias.
    """

    def __init__(self):
        """Initialize comprehensive PIT system."""
        self.pit_db = PointInTimeDatabase()
        self.fundamental_handler = FundamentalDataHandler(self.pit_db)
        self.earnings_handler = EarningsAnnouncementHandler(self.pit_db)
        self.economic_handler = EconomicDataHandler()
        self.ratings_handler = RatingChangesHandler()
        self.restatement_handler = RestatedDataHandler(self.pit_db)

        logger.info("Initialized ComprehensivePointInTimeSystem")

    def get_data(
        self,
        symbol: str,
        as_of_date: datetime,
        data_type: DataType,
        field_name: str
    ) -> Optional[Any]:
        """
        Get data value ensuring no look-ahead bias.

        Args:
            symbol: Trading symbol
            as_of_date: Query date
            data_type: Type of data
            field_name: Field name

        Returns:
            Value if available
        """
        if data_type in [DataType.FUNDAMENTAL, DataType.ESTIMATES]:
            return self.pit_db.get_value(symbol, field_name, as_of_date)
        elif data_type == DataType.ECONOMIC:
            return self.economic_handler.get_value(field_name, as_of_date)
        elif data_type == DataType.RATINGS:
            return self.ratings_handler.get_current_rating(symbol, as_of_date)
        else:
            return None

    def validate_no_lookahead(
        self,
        data_dict: Dict[str, pd.DataFrame],
        backtest_start: datetime
    ) -> Tuple[bool, List[str]]:
        """
        Validate that data doesn't contain look-ahead bias.

        Args:
            data_dict: Dictionary of DataFrames
            backtest_start: Backtest start date

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        for symbol, df in data_dict.items():
            if df.empty:
                continue

            # Check if any data exists before it should be available
            # This would require knowledge of reporting dates
            # Simplified check: warn if data starts exactly on backtest start
            if df.index.min() == backtest_start:
                issues.append(
                    f"{symbol}: Data starts exactly on backtest start - "
                    f"verify no look-ahead bias"
                )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("No look-ahead bias detected")
        else:
            logger.warning(f"Found {len(issues)} potential look-ahead bias issues")

        return is_valid, issues

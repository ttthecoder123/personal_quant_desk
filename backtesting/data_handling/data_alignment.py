"""
Data Alignment Module

Implements multi-asset data alignment for backtesting:
- Align multi-asset data properly
- Handle different trading calendars (US, Australia, etc.)
- Forward-fill for missing data
- Corporate action adjustments
- Currency conversions
- Index rebalancing handling
- Time zone harmonization

Proper data alignment is critical for multi-asset strategies.
Misaligned data can cause look-ahead bias and incorrect signals.
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Set, Union
import pandas as pd
import numpy as np
from loguru import logger


class TradingCalendar(Enum):
    """Trading calendar types."""
    NYSE = "NYSE"  # New York Stock Exchange
    NASDAQ = "NASDAQ"
    LSE = "LSE"  # London Stock Exchange
    TSE = "TSE"  # Tokyo Stock Exchange
    ASX = "ASX"  # Australian Securities Exchange
    HKEX = "HKEX"  # Hong Kong Exchange
    SSE = "SSE"  # Shanghai Stock Exchange
    FOREX = "FOREX"  # Forex (24/5)
    CRYPTO = "CRYPTO"  # Crypto (24/7)


class AlignmentMethod(Enum):
    """Data alignment method."""
    OUTER = "OUTER"  # Union of all timestamps
    INNER = "INNER"  # Intersection of all timestamps
    LEFT = "LEFT"  # Use timestamps from first asset
    BUSINESS_DAYS = "BUSINESS_DAYS"  # Align to business days
    TRADING_DAYS = "TRADING_DAYS"  # Align to specific trading calendar


@dataclass
class AlignmentConfig:
    """
    Data alignment configuration.

    Attributes:
        method: Alignment method
        fill_method: How to fill missing values
        trading_calendar: Trading calendar to use
        max_fill_limit: Maximum consecutive fills
        drop_weekends: Whether to drop weekends
        drop_holidays: Whether to drop holidays
    """
    method: AlignmentMethod = AlignmentMethod.OUTER
    fill_method: str = 'ffill'
    trading_calendar: Optional[TradingCalendar] = None
    max_fill_limit: int = 5
    drop_weekends: bool = True
    drop_holidays: bool = True


class DataAligner:
    """
    Primary data alignment engine.

    Aligns multiple time series to common timestamps
    with proper handling of:
    - Different frequencies
    - Different trading hours
    - Missing data
    - Timezone differences
    """

    def __init__(self, config: AlignmentConfig):
        """
        Initialize data aligner.

        Args:
            config: Alignment configuration
        """
        self.config = config
        logger.info(f"Initialized DataAligner: method={config.method.value}")

    def align(
        self,
        data_dict: Dict[str, pd.DataFrame],
        timestamp_col: str = 'timestamp'
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple DataFrames to common timestamps.

        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            timestamp_col: Name of timestamp column (or use index)

        Returns:
            Dictionary of aligned DataFrames
        """
        if not data_dict:
            return {}

        if len(data_dict) == 1:
            return data_dict

        logger.info(f"Aligning {len(data_dict)} time series")

        # Get common index
        common_index = self._get_common_index(data_dict, timestamp_col)

        if common_index is None or len(common_index) == 0:
            logger.warning("No common timestamps found")
            return data_dict

        # Reindex each DataFrame
        aligned_dict = {}
        for symbol, df in data_dict.items():
            # Ensure datetime index
            if df.index.name != timestamp_col and timestamp_col in df.columns:
                df = df.set_index(timestamp_col)

            # Reindex to common index
            aligned = df.reindex(common_index)

            # Fill missing values
            if self.config.fill_method:
                aligned = self._fill_missing(aligned)

            aligned_dict[symbol] = aligned

        logger.info(f"Aligned to {len(common_index)} timestamps")

        return aligned_dict

    def align_to_calendar(
        self,
        data_dict: Dict[str, pd.DataFrame],
        calendar: TradingCalendar
    ) -> Dict[str, pd.DataFrame]:
        """
        Align data to specific trading calendar.

        Args:
            data_dict: Dictionary of DataFrames
            calendar: Trading calendar to align to

        Returns:
            Dictionary of aligned DataFrames
        """
        logger.info(f"Aligning to {calendar.value} calendar")

        # Get trading calendar
        trading_dates = self._get_trading_calendar(
            calendar,
            start_date=self._get_earliest_date(data_dict),
            end_date=self._get_latest_date(data_dict)
        )

        # Reindex to trading dates
        aligned_dict = {}
        for symbol, df in data_dict.items():
            aligned = df.reindex(trading_dates)
            aligned = self._fill_missing(aligned)
            aligned_dict[symbol] = aligned

        return aligned_dict

    def _get_common_index(
        self,
        data_dict: Dict[str, pd.DataFrame],
        timestamp_col: str
    ) -> pd.DatetimeIndex:
        """Get common timestamp index based on alignment method."""
        # Collect all indices
        indices = []
        for symbol, df in data_dict.items():
            if df.index.name == timestamp_col or timestamp_col not in df.columns:
                indices.append(df.index)
            else:
                indices.append(pd.DatetimeIndex(df[timestamp_col]))

        if not indices:
            return None

        # Combine based on method
        if self.config.method == AlignmentMethod.OUTER:
            # Union of all timestamps
            common_index = indices[0]
            for idx in indices[1:]:
                common_index = common_index.union(idx)

        elif self.config.method == AlignmentMethod.INNER:
            # Intersection of all timestamps
            common_index = indices[0]
            for idx in indices[1:]:
                common_index = common_index.intersection(idx)

        elif self.config.method == AlignmentMethod.LEFT:
            # Use first asset's timestamps
            common_index = indices[0]

        elif self.config.method == AlignmentMethod.BUSINESS_DAYS:
            # Business days only
            start_date = min(idx.min() for idx in indices)
            end_date = max(idx.max() for idx in indices)
            common_index = pd.bdate_range(start=start_date, end=end_date)

        elif self.config.method == AlignmentMethod.TRADING_DAYS:
            # Use trading calendar
            if self.config.trading_calendar:
                start_date = min(idx.min() for idx in indices)
                end_date = max(idx.max() for idx in indices)
                common_index = self._get_trading_calendar(
                    self.config.trading_calendar,
                    start_date,
                    end_date
                )
            else:
                common_index = indices[0]

        else:
            common_index = indices[0]

        # Remove weekends if configured
        if self.config.drop_weekends:
            common_index = common_index[common_index.weekday < 5]

        return common_index.sort_values().unique()

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values according to configuration."""
        if df.empty:
            return df

        if self.config.fill_method == 'ffill':
            return df.fillna(method='ffill', limit=self.config.max_fill_limit)
        elif self.config.fill_method == 'bfill':
            return df.fillna(method='bfill', limit=self.config.max_fill_limit)
        elif self.config.fill_method == 'interpolate':
            return df.interpolate(method='linear', limit=self.config.max_fill_limit)
        elif self.config.fill_method == 'zero':
            return df.fillna(0)
        else:
            return df

    def _get_trading_calendar(
        self,
        calendar: TradingCalendar,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DatetimeIndex:
        """Get trading dates for calendar."""
        # Generate business days as baseline
        dates = pd.bdate_range(start=start_date, end=end_date)

        # Apply calendar-specific rules
        if calendar == TradingCalendar.FOREX:
            # Forex: Monday - Friday, 24 hours
            # Exclude weekends only
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = all_dates[all_dates.weekday < 5]

        elif calendar == TradingCalendar.CRYPTO:
            # Crypto: 24/7
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Note: Real implementation would use actual holiday calendars
        # (e.g., pandas_market_calendars library)

        return dates

    def _get_earliest_date(self, data_dict: Dict[str, pd.DataFrame]) -> datetime:
        """Get earliest date across all DataFrames."""
        dates = [df.index.min() for df in data_dict.values() if not df.empty]
        return min(dates) if dates else datetime.now()

    def _get_latest_date(self, data_dict: Dict[str, pd.DataFrame]) -> datetime:
        """Get latest date across all DataFrames."""
        dates = [df.index.max() for df in data_dict.values() if not df.empty]
        return max(dates) if dates else datetime.now()


class TimezoneHarmonizer:
    """
    Harmonize time zones across multiple data sources.

    Critical for:
    - Multi-region trading
    - 24-hour markets
    - Coordinating events across time zones
    """

    def __init__(self, target_timezone: str = 'UTC'):
        """
        Initialize timezone harmonizer.

        Args:
            target_timezone: Target timezone for all data
        """
        self.target_timezone = target_timezone
        logger.info(f"Initialized TimezoneHarmonizer: target={target_timezone}")

    def harmonize(
        self,
        data_dict: Dict[str, pd.DataFrame],
        source_timezones: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Convert all DataFrames to target timezone.

        Args:
            data_dict: Dictionary of DataFrames
            source_timezones: Source timezone for each symbol

        Returns:
            Dictionary of harmonized DataFrames
        """
        harmonized = {}

        for symbol, df in data_dict.items():
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                harmonized[symbol] = df
                continue

            # Get source timezone
            if source_timezones and symbol in source_timezones:
                source_tz = source_timezones[symbol]
            else:
                source_tz = df.index.tz.zone if df.index.tz else 'UTC'

            # Convert to target timezone
            if df.index.tz is None:
                # Localize first
                df.index = df.index.tz_localize(source_tz)

            if df.index.tz.zone != self.target_timezone:
                df.index = df.index.tz_convert(self.target_timezone)

            harmonized[symbol] = df

        logger.info(f"Harmonized {len(harmonized)} time series to {self.target_timezone}")

        return harmonized

    def convert_market_hours(
        self,
        data: pd.DataFrame,
        market_open: time,
        market_close: time,
        market_timezone: str
    ) -> pd.DataFrame:
        """
        Filter data to market hours in target timezone.

        Args:
            data: Input DataFrame
            market_open: Market open time
            market_close: Market close time
            market_timezone: Market's timezone

        Returns:
            Filtered DataFrame
        """
        if data.empty:
            return data

        # Ensure data is in market timezone
        if data.index.tz.zone != market_timezone:
            data_mkt_tz = data.copy()
            data_mkt_tz.index = data_mkt_tz.index.tz_convert(market_timezone)
        else:
            data_mkt_tz = data

        # Filter to market hours
        mask = (
            (data_mkt_tz.index.time >= market_open) &
            (data_mkt_tz.index.time <= market_close)
        )

        filtered = data[mask]

        logger.debug(
            f"Filtered {len(data)} → {len(filtered)} rows "
            f"for market hours {market_open}-{market_close}"
        )

        return filtered


class CorporateActionAdjuster:
    """
    Adjust prices for corporate actions.

    Handles:
    - Stock splits
    - Reverse splits
    - Dividends
    - Rights offerings
    - Spinoffs

    Essential for accurate historical analysis.
    """

    def __init__(self):
        """Initialize corporate action adjuster."""
        logger.info("Initialized CorporateActionAdjuster")

    def adjust_for_splits(
        self,
        data: pd.DataFrame,
        split_events: List[Dict],
        price_columns: List[str] = ['open', 'high', 'low', 'close']
    ) -> pd.DataFrame:
        """
        Adjust prices for stock splits.

        Args:
            data: Price data
            split_events: List of split events
                          [{date, ratio}, ...]
            price_columns: Columns to adjust

        Returns:
            Adjusted DataFrame
        """
        if data.empty or not split_events:
            return data

        adjusted = data.copy()

        for event in sorted(split_events, key=lambda x: x['date']):
            split_date = event['date']
            split_ratio = event['ratio']  # e.g., 2.0 for 2-for-1 split

            # Adjust prices before split date
            mask = adjusted.index < split_date

            for col in price_columns:
                if col in adjusted.columns:
                    adjusted.loc[mask, col] = adjusted.loc[mask, col] / split_ratio

            # Adjust volume (multiply by ratio)
            if 'volume' in adjusted.columns:
                adjusted.loc[mask, 'volume'] = adjusted.loc[mask, 'volume'] * split_ratio

            logger.debug(
                f"Applied {split_ratio}-for-1 split on {split_date}"
            )

        return adjusted

    def adjust_for_dividends(
        self,
        data: pd.DataFrame,
        dividend_events: List[Dict],
        price_columns: List[str] = ['open', 'high', 'low', 'close']
    ) -> pd.DataFrame:
        """
        Adjust prices for dividends (for total return calculation).

        Args:
            data: Price data
            dividend_events: List of dividend events
                            [{ex_date, amount}, ...]
            price_columns: Columns to adjust

        Returns:
            Adjusted DataFrame
        """
        if data.empty or not dividend_events:
            return data

        adjusted = data.copy()

        for event in sorted(dividend_events, key=lambda x: x['ex_date']):
            ex_date = event['ex_date']
            dividend_amount = event['amount']

            # Get close price on day before ex-date
            pre_ex_dates = adjusted.index[adjusted.index < ex_date]
            if len(pre_ex_dates) == 0:
                continue

            last_pre_ex_date = pre_ex_dates[-1]
            close_price = adjusted.loc[last_pre_ex_date, 'close']

            if close_price <= 0:
                continue

            # Adjustment factor
            adjustment_factor = (close_price - dividend_amount) / close_price

            # Adjust prices before ex-date
            mask = adjusted.index < ex_date

            for col in price_columns:
                if col in adjusted.columns:
                    adjusted.loc[mask, col] = adjusted.loc[mask, col] * adjustment_factor

            logger.debug(
                f"Applied dividend adjustment of ${dividend_amount} on {ex_date}"
            )

        return adjusted


class CurrencyConverter:
    """
    Convert prices between currencies.

    Essential for multi-currency portfolios.
    """

    def __init__(self, exchange_rates: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize currency converter.

        Args:
            exchange_rates: Dictionary of exchange rate time series
                           {currency_pair: DataFrame}
        """
        self.exchange_rates = exchange_rates or {}
        logger.info("Initialized CurrencyConverter")

    def convert(
        self,
        data: pd.DataFrame,
        from_currency: str,
        to_currency: str,
        price_columns: List[str] = ['open', 'high', 'low', 'close']
    ) -> pd.DataFrame:
        """
        Convert prices from one currency to another.

        Args:
            data: Price data
            from_currency: Source currency
            to_currency: Target currency
            price_columns: Columns to convert

        Returns:
            Converted DataFrame
        """
        if from_currency == to_currency:
            return data

        if data.empty:
            return data

        # Get exchange rate pair
        pair = f"{from_currency}{to_currency}"
        if pair not in self.exchange_rates:
            logger.warning(f"No exchange rate data for {pair}")
            return data

        fx_data = self.exchange_rates[pair]

        # Align exchange rates to data dates
        fx_aligned = fx_data.reindex(data.index, method='ffill')

        if 'rate' not in fx_aligned.columns:
            logger.error(f"No 'rate' column in {pair} exchange rate data")
            return data

        # Convert prices
        converted = data.copy()
        for col in price_columns:
            if col in converted.columns:
                converted[col] = converted[col] * fx_aligned['rate']

        logger.debug(f"Converted {len(data)} rows from {from_currency} to {to_currency}")

        return converted

    def add_exchange_rate(
        self,
        currency_pair: str,
        rates: pd.DataFrame
    ):
        """
        Add exchange rate time series.

        Args:
            currency_pair: Currency pair (e.g., 'USDAUD')
            rates: DataFrame with 'rate' column
        """
        self.exchange_rates[currency_pair] = rates
        logger.info(f"Added exchange rate data for {currency_pair}")


class IndexRebalancingHandler:
    """
    Handle index rebalancing events.

    Track changes in index composition over time.
    Critical for index-based strategies to avoid look-ahead bias.
    """

    def __init__(self):
        """Initialize index rebalancing handler."""
        self.rebalancing_history: List[Dict] = []
        logger.info("Initialized IndexRebalancingHandler")

    def add_rebalancing_event(
        self,
        date: datetime,
        index_name: str,
        additions: List[str],
        deletions: List[str]
    ):
        """
        Record index rebalancing event.

        Args:
            date: Rebalancing date
            index_name: Index name
            additions: Symbols added to index
            deletions: Symbols removed from index
        """
        self.rebalancing_history.append({
            'date': date,
            'index': index_name,
            'additions': additions,
            'deletions': deletions
        })

        logger.debug(
            f"{index_name} rebalancing on {date}: "
            f"+{len(additions)}, -{len(deletions)}"
        )

    def get_index_constituents(
        self,
        index_name: str,
        as_of_date: datetime,
        initial_constituents: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Get index constituents as of a specific date.

        Args:
            index_name: Index name
            as_of_date: Date to query
            initial_constituents: Initial set of constituents

        Returns:
            Set of symbols in index as of date
        """
        constituents = initial_constituents or set()

        # Apply all rebalancing events up to as_of_date
        for event in sorted(self.rebalancing_history, key=lambda x: x['date']):
            if event['index'] != index_name:
                continue

            if event['date'] <= as_of_date:
                # Add new constituents
                constituents.update(event['additions'])

                # Remove deleted constituents
                constituents.difference_update(event['deletions'])
            else:
                break

        return constituents

    def filter_by_index_membership(
        self,
        data_dict: Dict[str, pd.DataFrame],
        index_name: str,
        initial_constituents: Set[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter data to only include index members at each point in time.

        Args:
            data_dict: Dictionary of DataFrames
            index_name: Index to track
            initial_constituents: Initial index members

        Returns:
            Filtered dictionary
        """
        filtered = {}

        for symbol, df in data_dict.items():
            if df.empty:
                continue

            # For each date, check if symbol was in index
            mask = pd.Series(False, index=df.index)

            for date in df.index:
                constituents = self.get_index_constituents(
                    index_name,
                    date,
                    initial_constituents
                )
                mask[date] = symbol in constituents

            if mask.any():
                filtered[symbol] = df[mask]

        logger.info(
            f"Filtered to {len(filtered)} symbols based on {index_name} membership"
        )

        return filtered


class MultiAssetPanel:
    """
    Create panel data structure for multi-asset analysis.

    Combines multiple time series into a single DataFrame
    with MultiIndex or wide format.
    """

    def __init__(self):
        """Initialize multi-asset panel."""
        logger.info("Initialized MultiAssetPanel")

    def create_panel(
        self,
        data_dict: Dict[str, pd.DataFrame],
        format: str = 'wide'
    ) -> pd.DataFrame:
        """
        Create panel DataFrame from dictionary of series.

        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            format: 'wide' or 'long'

        Returns:
            Panel DataFrame
        """
        if not data_dict:
            return pd.DataFrame()

        if format == 'wide':
            # Wide format: timestamps × (symbol, column)
            panel = pd.concat(data_dict, axis=1, keys=data_dict.keys())

        elif format == 'long':
            # Long format: (timestamp, symbol) × columns
            dfs = []
            for symbol, df in data_dict.items():
                df = df.copy()
                df['symbol'] = symbol
                dfs.append(df)

            panel = pd.concat(dfs, axis=0)
            panel = panel.set_index(['symbol'], append=True)

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Created panel with shape {panel.shape}")

        return panel

    def to_dict(self, panel: pd.DataFrame, format: str = 'wide') -> Dict[str, pd.DataFrame]:
        """
        Convert panel back to dictionary.

        Args:
            panel: Panel DataFrame
            format: Panel format ('wide' or 'long')

        Returns:
            Dictionary of DataFrames
        """
        data_dict = {}

        if format == 'wide':
            # Extract each symbol
            if isinstance(panel.columns, pd.MultiIndex):
                for symbol in panel.columns.levels[0]:
                    data_dict[symbol] = panel[symbol]

        elif format == 'long':
            # Group by symbol
            if 'symbol' in panel.index.names:
                for symbol in panel.index.get_level_values('symbol').unique():
                    data_dict[symbol] = panel.xs(symbol, level='symbol')

        return data_dict

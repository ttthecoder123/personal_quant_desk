"""
Data Loader

Comprehensive data loading module for backtesting:
- Load from Step 2's Parquet storage
- Handle multiple data frequencies (daily, hourly, minute)
- Manage timezone conversions
- Cache frequently used data
- Streaming data simulation
- Handle data gaps properly
- Memory-efficient chunking for large datasets

Efficient data loading is critical for backtest performance.
This module provides flexible, high-performance data access.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from functools import lru_cache
from loguru import logger


class DataFrequency(Enum):
    """Data frequency options."""
    TICK = "TICK"
    SECOND = "1S"
    MINUTE = "1T"
    MINUTE_5 = "5T"
    MINUTE_15 = "15T"
    MINUTE_30 = "30T"
    HOURLY = "1H"
    DAILY = "1D"
    WEEKLY = "1W"
    MONTHLY = "1M"


class DataFormat(Enum):
    """Data storage format."""
    PARQUET = "PARQUET"
    CSV = "CSV"
    HDF5 = "HDF5"
    FEATHER = "FEATHER"


@dataclass
class DataConfig:
    """
    Data loading configuration.

    Attributes:
        data_dir: Base directory for data storage
        frequency: Data frequency
        format: Storage format
        timezone: Target timezone
        cache_size: Number of datasets to cache
        chunk_size: Rows per chunk for streaming
        preload: Whether to preload data
    """
    data_dir: Union[str, Path]
    frequency: DataFrequency = DataFrequency.DAILY
    format: DataFormat = DataFormat.PARQUET
    timezone: str = "UTC"
    cache_size: int = 100
    chunk_size: int = 10000
    preload: bool = False


class DataLoader:
    """
    Primary data loader with caching and optimization.

    Loads data from Step 2's storage with:
    - Efficient parquet reading
    - Intelligent caching
    - Timezone handling
    - Memory management
    """

    def __init__(self, config: DataConfig):
        """
        Initialize data loader.

        Args:
            config: Data loading configuration
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Verify data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            logger.info(f"Creating data directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized DataLoader: dir={self.data_dir}, "
            f"freq={config.frequency.value}, format={config.format.value}"
        )

    def load(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data for a symbol.

        Args:
            symbol: Trading symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            columns: Columns to load (None = all)

        Returns:
            DataFrame with requested data
        """
        # Check cache
        cache_key = self._get_cache_key(symbol, start_date, end_date, columns)
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for {symbol}")
            return self.cache[cache_key].copy()

        self.cache_misses += 1
        logger.debug(f"Cache miss for {symbol}, loading from disk")

        # Load from storage
        data = self._load_from_storage(symbol, columns)

        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # Filter by date range
        if start_date or end_date:
            data = self._filter_by_date(data, start_date, end_date)

        # Handle timezone
        data = self._handle_timezone(data)

        # Cache if small enough
        if len(data) < self.config.chunk_size * 10:
            self._add_to_cache(cache_key, data)

        return data

    def load_multiple(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            columns: Columns to load

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Loading data for {len(symbols)} symbols")

        data_dict = {}
        for symbol in symbols:
            try:
                data = self.load(symbol, start_date, end_date, columns)
                if not data.empty:
                    data_dict[symbol] = data
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        logger.info(f"Successfully loaded {len(data_dict)} symbols")
        return data_dict

    def stream(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        chunk_size: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Stream data in chunks for memory efficiency.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            chunk_size: Rows per chunk

        Yields:
            DataFrame chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        logger.info(f"Streaming {symbol} in chunks of {chunk_size}")

        file_path = self._get_file_path(symbol)

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return

        # Stream from parquet
        if self.config.format == DataFormat.PARQUET:
            parquet_file = pq.ParquetFile(file_path)

            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                df = self._handle_timezone(df)

                # Filter by date
                if start_date or end_date:
                    df = self._filter_by_date(df, start_date, end_date)

                if not df.empty:
                    yield df
        else:
            # For other formats, load and chunk
            data = self._load_from_storage(symbol)
            if start_date or end_date:
                data = self._filter_by_date(data, start_date, end_date)

            for i in range(0, len(data), chunk_size):
                yield data.iloc[i:i + chunk_size]

    def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """
        Get available date range for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (start_date, end_date)
        """
        try:
            # Load minimal data to check dates
            data = self.load(symbol, columns=['timestamp'])
            if data.empty:
                return None, None

            return data.index.min(), data.index.max()
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None, None

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols.

        Returns:
            List of symbol strings
        """
        symbols = []

        if not self.data_dir.exists():
            return symbols

        # Find all data files
        pattern = f"*.{self.config.format.value.lower()}"
        for file_path in self.data_dir.glob(pattern):
            symbol = file_path.stem
            symbols.append(symbol)

        logger.info(f"Found {len(symbols)} available symbols")
        return sorted(symbols)

    def preload_symbols(self, symbols: List[str]):
        """
        Preload symbols into cache.

        Args:
            symbols: List of symbols to preload
        """
        logger.info(f"Preloading {len(symbols)} symbols")

        for symbol in symbols:
            try:
                self.load(symbol)
            except Exception as e:
                logger.error(f"Error preloading {symbol}: {e}")

        logger.info(f"Preloaded {len(self.cache)} datasets")

    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (
            self.cache_hits / total_requests if total_requests > 0 else 0
        )

        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def _load_from_storage(
        self,
        symbol: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load data from storage backend."""
        file_path = self._get_file_path(symbol)

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()

        try:
            if self.config.format == DataFormat.PARQUET:
                df = pd.read_parquet(file_path, columns=columns)
            elif self.config.format == DataFormat.CSV:
                df = pd.read_csv(file_path, usecols=columns)
            elif self.config.format == DataFormat.HDF5:
                df = pd.read_hdf(file_path, key='data', columns=columns)
            elif self.config.format == DataFormat.FEATHER:
                df = pd.read_feather(file_path, columns=columns)
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")

            # Set index if timestamp column exists
            if 'timestamp' in df.columns and df.index.name != 'timestamp':
                df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error loading {symbol} from {file_path}: {e}")
            return pd.DataFrame()

    def _get_file_path(self, symbol: str) -> Path:
        """Get file path for symbol."""
        extension = self.config.format.value.lower()
        freq_dir = self.config.frequency.value.lower().replace('/', '_')

        # Organize by frequency: data/daily/AAPL.parquet
        file_path = self.data_dir / freq_dir / f"{symbol}.{extension}"

        # Fallback to flat structure: data/AAPL.parquet
        if not file_path.exists():
            file_path = self.data_dir / f"{symbol}.{extension}"

        return file_path

    def _filter_by_date(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter data by date range."""
        if data.empty:
            return data

        if start_date:
            data = data[data.index >= start_date]

        if end_date:
            data = data[data.index <= end_date]

        return data

    def _handle_timezone(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle timezone conversion."""
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            return data

        # Localize or convert to target timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')

        if data.index.tz.zone != self.config.timezone:
            data.index = data.index.tz_convert(self.config.timezone)

        return data

    def _get_cache_key(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        columns: Optional[List[str]]
    ) -> str:
        """Generate cache key."""
        key_parts = [symbol]

        if start_date:
            key_parts.append(start_date.strftime('%Y%m%d'))
        if end_date:
            key_parts.append(end_date.strftime('%Y%m%d'))
        if columns:
            key_parts.append('_'.join(sorted(columns)))

        return '_'.join(key_parts)

    def _add_to_cache(self, key: str, data: pd.DataFrame):
        """Add data to cache with size management."""
        # Evict oldest if cache full
        if len(self.cache) >= self.config.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = data.copy()


class FrequencyConverter:
    """
    Convert between different data frequencies.

    Resample data to different timeframes:
    - Minute → Hourly
    - Hourly → Daily
    - Daily → Weekly
    etc.
    """

    def __init__(self):
        """Initialize frequency converter."""
        logger.info("Initialized FrequencyConverter")

    def resample(
        self,
        data: pd.DataFrame,
        target_frequency: DataFrequency,
        aggregation: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample data to target frequency.

        Args:
            data: Input DataFrame
            target_frequency: Target frequency
            aggregation: Column aggregation methods

        Returns:
            Resampled DataFrame
        """
        if data.empty:
            return data

        # Default aggregation rules
        if aggregation is None:
            aggregation = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'vwap': 'mean'
            }

        # Filter to existing columns
        agg_dict = {
            col: method for col, method in aggregation.items()
            if col in data.columns
        }

        if not agg_dict:
            logger.warning("No aggregatable columns found")
            return data

        # Resample
        resampled = data.resample(target_frequency.value).agg(agg_dict)

        # Drop rows with no data
        resampled = resampled.dropna(how='all')

        logger.debug(
            f"Resampled from {len(data)} to {len(resampled)} rows "
            f"at {target_frequency.value}"
        )

        return resampled

    def align_to_frequency(
        self,
        data: pd.DataFrame,
        frequency: DataFrequency
    ) -> pd.DataFrame:
        """
        Align timestamps to frequency boundaries.

        Args:
            data: Input DataFrame
            frequency: Target frequency

        Returns:
            Aligned DataFrame
        """
        if data.empty:
            return data

        # Round timestamps to frequency
        data.index = data.index.round(frequency.value)

        # Remove duplicates (keep last)
        data = data[~data.index.duplicated(keep='last')]

        return data


class DataGapHandler:
    """
    Handle gaps in time series data.

    Provides strategies for dealing with missing data:
    - Forward fill
    - Backward fill
    - Interpolation
    - Drop gaps
    - Fill with value
    """

    def __init__(self):
        """Initialize data gap handler."""
        logger.info("Initialized DataGapHandler")

    def fill_gaps(
        self,
        data: pd.DataFrame,
        method: str = 'ffill',
        limit: Optional[int] = None,
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Fill gaps in data.

        Args:
            data: Input DataFrame
            method: Fill method ('ffill', 'bfill', 'interpolate', 'value')
            limit: Maximum number of consecutive NaNs to fill
            fill_value: Value to use for 'value' method

        Returns:
            DataFrame with gaps filled
        """
        if data.empty:
            return data

        filled = data.copy()

        if method == 'ffill':
            filled = filled.fillna(method='ffill', limit=limit)
        elif method == 'bfill':
            filled = filled.fillna(method='bfill', limit=limit)
        elif method == 'interpolate':
            filled = filled.interpolate(method='linear', limit=limit)
        elif method == 'value':
            if fill_value is not None:
                filled = filled.fillna(fill_value)
        else:
            raise ValueError(f"Unknown fill method: {method}")

        gaps_filled = data.isna().sum().sum() - filled.isna().sum().sum()
        if gaps_filled > 0:
            logger.debug(f"Filled {gaps_filled} gaps using {method}")

        return filled

    def detect_gaps(
        self,
        data: pd.DataFrame,
        frequency: DataFrequency
    ) -> pd.DataFrame:
        """
        Detect gaps in time series.

        Args:
            data: Input DataFrame
            frequency: Expected frequency

        Returns:
            DataFrame with gap information
        """
        if data.empty or len(data) < 2:
            return pd.DataFrame()

        # Expected time delta
        freq_delta = pd.Timedelta(frequency.value)

        # Calculate actual deltas
        time_diffs = pd.Series(data.index).diff()

        # Find gaps (deltas > expected)
        gaps = time_diffs[time_diffs > freq_delta * 1.5]

        if gaps.empty:
            logger.debug("No gaps detected")
            return pd.DataFrame()

        gap_info = pd.DataFrame({
            'start': data.index[gaps.index - 1],
            'end': data.index[gaps.index],
            'duration': gaps.values
        })

        logger.info(f"Detected {len(gap_info)} gaps in data")

        return gap_info

    def create_continuous_index(
        self,
        start: datetime,
        end: datetime,
        frequency: DataFrequency,
        trading_calendar: Optional[pd.DatetimeIndex] = None
    ) -> pd.DatetimeIndex:
        """
        Create continuous datetime index.

        Args:
            start: Start datetime
            end: End datetime
            frequency: Frequency
            trading_calendar: Optional trading calendar (business days only)

        Returns:
            Complete DatetimeIndex
        """
        if trading_calendar is not None:
            # Use trading calendar
            index = trading_calendar[(trading_calendar >= start) & (trading_calendar <= end)]
        else:
            # Generate full index
            index = pd.date_range(
                start=start,
                end=end,
                freq=frequency.value
            )

        return index


class MemoryEfficientLoader:
    """
    Memory-efficient data loader for large datasets.

    Uses chunking and streaming to handle datasets
    larger than available RAM.
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize memory-efficient loader.

        Args:
            data_loader: Base DataLoader instance
        """
        self.data_loader = data_loader
        logger.info("Initialized MemoryEfficientLoader")

    def process_in_chunks(
        self,
        symbol: str,
        processor_func,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        chunk_size: Optional[int] = None
    ) -> List:
        """
        Process data in chunks with custom function.

        Args:
            symbol: Trading symbol
            processor_func: Function to process each chunk
            start_date: Start date
            end_date: End date
            chunk_size: Rows per chunk

        Returns:
            List of processing results
        """
        results = []

        for chunk in self.data_loader.stream(symbol, start_date, end_date, chunk_size):
            try:
                result = processor_func(chunk)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

        return results

    def aggregate_statistics(
        self,
        symbol: str,
        columns: List[str],
        stats: List[str] = ['mean', 'std', 'min', 'max'],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics without loading full dataset.

        Args:
            symbol: Trading symbol
            columns: Columns to analyze
            stats: Statistics to calculate
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary of statistics by column
        """
        logger.info(f"Calculating statistics for {symbol}")

        # Initialize accumulators
        count = 0
        sums = {col: 0.0 for col in columns}
        sum_squares = {col: 0.0 for col in columns}
        mins = {col: float('inf') for col in columns}
        maxs = {col: float('-inf') for col in columns}

        # Process in chunks
        for chunk in self.data_loader.stream(symbol, start_date, end_date):
            for col in columns:
                if col in chunk.columns:
                    values = chunk[col].dropna()
                    if len(values) > 0:
                        count += len(values)
                        sums[col] += values.sum()
                        sum_squares[col] += (values ** 2).sum()
                        mins[col] = min(mins[col], values.min())
                        maxs[col] = max(maxs[col], values.max())

        # Calculate final statistics
        results = {}
        for col in columns:
            if count > 0:
                mean = sums[col] / count
                variance = (sum_squares[col] / count) - (mean ** 2)
                std = np.sqrt(max(0, variance))

                results[col] = {
                    'mean': mean,
                    'std': std,
                    'min': mins[col],
                    'max': maxs[col],
                    'count': count
                }

        return results

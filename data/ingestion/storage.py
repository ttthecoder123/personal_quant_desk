"""
Parquet Storage Handler implementing McKinney's efficient I/O best practices.
Optimized for financial time series with hierarchical indexing and partitioning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
import json
import shutil
from dataclasses import dataclass, asdict
import warnings

# Suppress pyarrow warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pyarrow')


@dataclass
class StorageMetadata:
    """Metadata for stored datasets following McKinney's structured approach."""
    symbol: str
    file_path: str
    rows_stored: int
    columns_stored: List[str]
    date_range: Tuple[str, str]
    storage_timestamp: datetime
    file_size_mb: float
    compression_ratio: float
    partition_info: Dict[str, Any]
    data_types: Dict[str, str]


class ParquetStorage:
    """
    Professional parquet storage system implementing McKinney's I/O best practices.

    Features:
    - Hierarchical MultiIndex for efficient queries (McKinney Ch. 8)
    - Year-based partitioning for optimal performance
    - Snappy compression for speed/size balance
    - Timezone-aware datetime handling (McKinney Ch. 11)
    - Metadata tracking and schema enforcement
    - Memory-efficient operations with chunking
    """

    def __init__(self, base_path: str = "data/processed"):
        """Initialize storage with base path and configuration."""
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / "metadata"

        # Create directory structure
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        # Storage configuration
        self.compression = 'snappy'  # Balance between speed and compression
        self.engine = 'pyarrow'     # Fast, feature-rich parquet engine

        # Schema definitions for validation
        self.ohlcv_schema = self._define_ohlcv_schema()

        logger.info("ParquetStorage initialized with base path: {}", self.base_path)

    def _define_ohlcv_schema(self) -> pa.Schema:
        """
        Define standardized OHLCV schema for consistency.

        Returns:
            PyArrow schema for OHLCV data
        """
        return pa.schema([
            ('Date', pa.timestamp('ns', tz='UTC')),
            ('symbol', pa.string()),
            ('Open', pa.float64()),
            ('High', pa.float64()),
            ('Low', pa.float64()),
            ('Close', pa.float64()),
            ('Volume', pa.int64()),
            ('Adj Close', pa.float64())
        ])

    def save_timeseries(
        self,
        df: pd.DataFrame,
        symbol: str,
        asset_class: str = "equity",
        overwrite: bool = False
    ) -> StorageMetadata:
        """
        Save time series data with McKinney's best practices for hierarchical storage.

        Args:
            df: DataFrame with datetime index and OHLCV columns
            symbol: Instrument symbol
            asset_class: Asset classification for organization
            overwrite: Whether to overwrite existing data

        Returns:
            StorageMetadata: Information about stored data
        """
        logger.info("Saving time series for {} ({} rows)", symbol, len(df))

        if df.empty:
            logger.warning("Empty DataFrame provided for {}", symbol)
            return self._create_empty_metadata(symbol)

        try:
            # Prepare data for storage
            df_prepared = self._prepare_dataframe(df, symbol)

            # Create hierarchical structure (McKinney Ch. 8)
            df_hierarchical = self._create_hierarchical_index(df_prepared, symbol)

            # Partition by year for efficient queries (McKinney recommendation)
            storage_metadata = self._save_partitioned_data(
                df_hierarchical, symbol, asset_class, overwrite
            )

            # Save metadata
            self._save_metadata(storage_metadata)

            logger.success("Successfully saved {} rows for {}", len(df), symbol)
            return storage_metadata

        except Exception as e:
            logger.error("Failed to save data for {}: {}", symbol, str(e))
            raise

    def _prepare_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Prepare DataFrame for storage using McKinney's best practices.

        Args:
            df: Input DataFrame
            symbol: Instrument symbol

        Returns:
            Prepared DataFrame
        """
        df_copy = df.copy()

        # Ensure timezone-aware datetime index (McKinney Ch. 11)
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index)

        if df_copy.index.tz is None:
            logger.warning("Index is not timezone-aware for {}, assuming UTC", symbol)
            df_copy.index = df_copy.index.tz_localize('UTC')
        elif str(df_copy.index.tz) != 'UTC':
            df_copy.index = df_copy.index.tz_convert('UTC')

        # Standardize column names
        column_mapping = {
            'Adj Close': 'AdjClose'  # Remove spaces for parquet compatibility
        }
        df_copy = df_copy.rename(columns=column_mapping)

        # Ensure required columns exist with proper types
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df_copy.columns:
                logger.warning("Missing required column {} for {}", col, symbol)
                df_copy[col] = np.nan

        # Optimize data types for storage efficiency (McKinney Ch. 12)
        df_copy = self._optimize_dtypes(df_copy)

        # Sort by date for optimal parquet performance
        df_copy = df_copy.sort_index()

        return df_copy

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for memory efficiency (McKinney Ch. 12).

        Args:
            df: DataFrame to optimize

        Returns:
            DataFrame with optimized dtypes
        """
        # Convert price columns to float64 (standard for financial data)
        price_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

        # Convert volume to int64 (can handle large volumes)
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype('int64')

        return df

    def _create_hierarchical_index(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create hierarchical MultiIndex structure (McKinney Ch. 8).

        Args:
            df: DataFrame with datetime index
            symbol: Instrument symbol

        Returns:
            DataFrame with MultiIndex [Date, symbol]
        """
        # Reset index to make Date a column
        df_reset = df.reset_index()
        df_reset['symbol'] = symbol

        # Create MultiIndex with Date and symbol
        df_multi = df_reset.set_index(['Date', 'symbol'])

        return df_multi

    def _save_partitioned_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        asset_class: str,
        overwrite: bool
    ) -> StorageMetadata:
        """
        Save data with year-based partitioning for efficient queries.

        Args:
            df: DataFrame with MultiIndex
            symbol: Instrument symbol
            asset_class: Asset classification
            overwrite: Whether to overwrite existing data

        Returns:
            StorageMetadata: Storage information
        """
        date_range = (
            str(df.index.get_level_values('Date').min().date()),
            str(df.index.get_level_values('Date').max().date())
        )

        storage_metadata_list = []
        total_file_size = 0.0

        # Group by year for partitioning
        for year in df.index.get_level_values('Date').year.unique():
            year_data = df.loc[df.index.get_level_values('Date').year == year]

            if year_data.empty:
                continue

            # Create year directory
            year_path = self.base_path / str(year)
            year_path.mkdir(parents=True, exist_ok=True)

            # Define file path
            file_path = year_path / f"{symbol}.parquet"

            # Check if file exists and handle accordingly
            if file_path.exists() and not overwrite:
                # Append mode: merge with existing data
                year_data = self._merge_with_existing(year_data, file_path)

            # Save to parquet with optimal settings
            year_data.to_parquet(
                file_path,
                compression=self.compression,
                engine=self.engine,
                index=True  # Preserve MultiIndex
            )

            # Calculate file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            total_file_size += file_size_mb

            logger.debug("Saved {} rows to {} ({:.2f} MB)", len(year_data), file_path, file_size_mb)

        # Calculate compression ratio (estimate)
        uncompressed_size_estimate = len(df) * len(df.columns) * 8 / (1024 * 1024)  # 8 bytes per float64
        compression_ratio = uncompressed_size_estimate / total_file_size if total_file_size > 0 else 1.0

        return StorageMetadata(
            symbol=symbol,
            file_path=str(self.base_path / "{year}" / f"{symbol}.parquet"),
            rows_stored=len(df),
            columns_stored=list(df.columns),
            date_range=date_range,
            storage_timestamp=datetime.now(pytz.UTC),
            file_size_mb=total_file_size,
            compression_ratio=compression_ratio,
            partition_info={"partition_by": "year", "asset_class": asset_class},
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()}
        )

    def _merge_with_existing(self, new_data: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """
        Merge new data with existing parquet file (append mode).

        Args:
            new_data: New data to merge
            file_path: Path to existing parquet file

        Returns:
            Merged DataFrame
        """
        try:
            existing_data = pd.read_parquet(file_path, engine=self.engine)

            # Combine and remove duplicates (keep latest)
            combined = pd.concat([existing_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()

            logger.debug("Merged new data with existing file: {}", file_path)
            return combined

        except Exception as e:
            logger.warning("Failed to merge with existing file {}: {}", file_path, str(e))
            return new_data

    def load_timeseries(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load time series data with efficient date filtering.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD) or None for all data
            end_date: End date (YYYY-MM-DD) or None for all data
            columns: Specific columns to load or None for all

        Returns:
            DataFrame with datetime index
        """
        logger.info("Loading time series for {} from {} to {}", symbol, start_date, end_date)

        try:
            # Determine which year files to load
            year_files = self._get_relevant_year_files(symbol, start_date, end_date)

            if not year_files:
                logger.warning("No data files found for {}", symbol)
                return pd.DataFrame()

            # Load and combine data from relevant years
            dataframes = []
            for year_file in year_files:
                if year_file.exists():
                    year_df = pd.read_parquet(year_file, engine=self.engine, columns=columns)
                    dataframes.append(year_df)

            if not dataframes:
                return pd.DataFrame()

            # Combine all years
            df = pd.concat(dataframes, axis=0)

            # Filter by symbol (in case multiple symbols in same file)
            if 'symbol' in df.index.names:
                df = df.xs(symbol, level='symbol')

            # Apply date filtering if specified
            if start_date or end_date:
                df = self._filter_by_date_range(df, start_date, end_date)

            # Sort by date
            df = df.sort_index()

            logger.success("Loaded {} rows for {}", len(df), symbol)
            return df

        except Exception as e:
            logger.error("Failed to load data for {}: {}", symbol, str(e))
            raise

    def _get_relevant_year_files(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[Path]:
        """
        Determine which year files are relevant for the query.

        Args:
            symbol: Instrument symbol
            start_date: Start date string
            end_date: End date string

        Returns:
            List of relevant file paths
        """
        # Get all available year directories
        year_dirs = [d for d in self.base_path.iterdir() if d.is_dir() and d.name.isdigit()]

        if not start_date and not end_date:
            # Load all available years
            return [year_dir / f"{symbol}.parquet" for year_dir in year_dirs]

        # Parse date range
        start_year = int(start_date[:4]) if start_date else 1900
        end_year = int(end_date[:4]) if end_date else 3000

        # Filter relevant years
        relevant_files = []
        for year_dir in year_dirs:
            year = int(year_dir.name)
            if start_year <= year <= end_year:
                file_path = year_dir / f"{symbol}.parquet"
                relevant_files.append(file_path)

        return sorted(relevant_files)

    def _filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range using efficient pandas operations.

        Args:
            df: DataFrame with datetime index
            start_date: Start date string
            end_date: End date string

        Returns:
            Filtered DataFrame
        """
        if start_date:
            df = df[df.index >= start_date]

        if end_date:
            df = df[df.index <= end_date]

        return df

    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple symbols efficiently.

        Args:
            symbols: List of instrument symbols
            start_date: Start date string
            end_date: End date string
            columns: Columns to load

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info("Loading {} symbols from {} to {}", len(symbols), start_date, end_date)

        results = {}
        for symbol in symbols:
            try:
                df = self.load_timeseries(symbol, start_date, end_date, columns)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error("Failed to load {}: {}", symbol, str(e))

        logger.info("Successfully loaded {} out of {} symbols", len(results), len(symbols))
        return results

    def _save_metadata(self, metadata: StorageMetadata) -> None:
        """Save storage metadata to JSON file."""
        metadata_file = self.metadata_path / f"{metadata.symbol}_metadata.json"

        # Convert to dictionary and handle datetime serialization
        metadata_dict = asdict(metadata)
        metadata_dict['storage_timestamp'] = metadata.storage_timestamp.isoformat()

        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def get_metadata(self, symbol: str) -> Optional[StorageMetadata]:
        """Load metadata for a symbol."""
        metadata_file = self.metadata_path / f"{symbol}_metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)

            # Convert timestamp back to datetime
            metadata_dict['storage_timestamp'] = datetime.fromisoformat(
                metadata_dict['storage_timestamp']
            )

            return StorageMetadata(**metadata_dict)

        except Exception as e:
            logger.error("Failed to load metadata for {}: {}", symbol, str(e))
            return None

    def list_available_symbols(self) -> List[str]:
        """List all available symbols in storage."""
        symbols = set()

        for year_dir in self.base_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                for file in year_dir.glob("*.parquet"):
                    symbol = file.stem
                    symbols.add(symbol)

        return sorted(list(symbols))

    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of storage usage and statistics."""
        symbols = self.list_available_symbols()
        total_size = 0
        file_count = 0

        for year_dir in self.base_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                for file in year_dir.glob("*.parquet"):
                    total_size += file.stat().st_size
                    file_count += 1

        return {
            'total_symbols': len(symbols),
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'available_symbols': symbols,
            'storage_path': str(self.base_path)
        }

    def _create_empty_metadata(self, symbol: str) -> StorageMetadata:
        """Create metadata for empty dataset."""
        return StorageMetadata(
            symbol=symbol,
            file_path="",
            rows_stored=0,
            columns_stored=[],
            date_range=("", ""),
            storage_timestamp=datetime.now(pytz.UTC),
            file_size_mb=0.0,
            compression_ratio=1.0,
            partition_info={},
            data_types={}
        )

    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """Clean up old data files beyond retention period."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_year = cutoff_date.year

        cleaned_files = 0
        for year_dir in self.base_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                if year < cutoff_year:
                    shutil.rmtree(year_dir)
                    cleaned_files += len(list(year_dir.glob("*.parquet")))
                    logger.info("Cleaned up data for year {}", year)

        if cleaned_files > 0:
            logger.info("Cleanup complete: removed {} old files", cleaned_files)


# Convenience functions
def save_dataframe(df: pd.DataFrame, symbol: str, base_path: str = "data/processed") -> StorageMetadata:
    """Convenience function to save a single DataFrame."""
    storage = ParquetStorage(base_path)
    return storage.save_timeseries(df, symbol)


def load_dataframe(symbol: str, base_path: str = "data/processed") -> pd.DataFrame:
    """Convenience function to load a single DataFrame."""
    storage = ParquetStorage(base_path)
    return storage.load_timeseries(symbol)
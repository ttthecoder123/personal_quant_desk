"""
Market Data Downloader with robust error handling and McKinney's best practices.
Implements efficient batch downloading, retry logic, and timezone handling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import pytz
from retry import retry
from loguru import logger
import warnings
import time
import yaml
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')


@dataclass
class DownloadResult:
    """Container for download results following McKinney's structured approach."""
    symbol: str
    data: Optional[pd.DataFrame]
    success: bool
    error_message: Optional[str]
    download_time: datetime
    rows_downloaded: int
    start_date: Optional[str]
    end_date: Optional[str]
    source: Optional[str] = None
    processing_time: float = 0.0
    metadata: Optional[Dict] = None


class MarketDataDownloader:
    """
    Professional market data downloader implementing McKinney's pandas best practices.

    Features:
    - Vectorized operations for efficient data processing
    - Robust timezone handling (McKinney Ch. 11)
    - Hierarchical MultiIndex for data organization
    - Memory-efficient operations with chunking
    - Comprehensive error handling and retry logic
    """

    def __init__(self, config_path: str = "data/config/instruments.yaml"):
        """Initialize downloader with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Configure yfinance cache location
        cache_dir = Path('.cache/yfinance')
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))

        # Initialize session for connection pooling
        self.session = requests.Session()

        # Performance tracking
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_rows': 0,
            'start_time': None
        }

        logger.info("MarketDataDownloader initialized with cache location: {}", cache_dir)

    def _load_config(self) -> Dict:
        """Load instrument configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("Loaded configuration for {} instruments",
                   sum(len(category) for category in config['instruments'].values()))
        return config

    @retry(tries=3, delay=2, backoff=2, logger=logger)
    def download_single_instrument(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> DownloadResult:
        """
        Download single instrument with retry logic and comprehensive error handling.

        Args:
            symbol: Yahoo Finance symbol (e.g., 'SPY', 'CL=F')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            DownloadResult: Structured result with data and metadata
        """
        download_start = datetime.now()

        try:
            logger.info("Downloading {} from {} to {}", symbol, start_date, end_date)

            # Create ticker object - let yfinance handle session management
            ticker = yf.Ticker(symbol)

            # Download data with specified parameters
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
                prepost=False      # Regular trading hours only
            )

            if data.empty:
                error_msg = f"No data returned for {symbol}"
                logger.warning(error_msg)
                return DownloadResult(
                    symbol=symbol,
                    data=None,
                    success=False,
                    error_message=error_msg,
                    download_time=download_start,
                    rows_downloaded=0,
                    start_date=start_date,
                    end_date=end_date
                )

            # Apply McKinney's best practices for data cleaning
            data = self._clean_ohlcv_data(data, symbol)

            # Handle timezone (McKinney Ch. 11)
            data = self._handle_timezone(data, symbol)

            # Validate data integrity
            self._validate_ohlcv_structure(data, symbol)

            logger.success("Successfully downloaded {} rows for {}", len(data), symbol)

            return DownloadResult(
                symbol=symbol,
                data=data,
                success=True,
                error_message=None,
                download_time=download_start,
                rows_downloaded=len(data),
                start_date=str(data.index.min().date()) if not data.empty else None,
                end_date=str(data.index.max().date()) if not data.empty else None
            )

        except Exception as e:
            error_msg = f"Failed to download {symbol}: {str(e)}"
            logger.error(error_msg)

            return DownloadResult(
                symbol=symbol,
                data=None,
                success=False,
                error_message=error_msg,
                download_time=download_start,
                rows_downloaded=0,
                start_date=start_date,
                end_date=end_date
            )

    def _clean_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean OHLCV data using McKinney's vectorized operations.

        Args:
            df: Raw OHLCV DataFrame
            symbol: Instrument symbol for logging

        Returns:
            Cleaned DataFrame
        """
        original_rows = len(df)

        # Remove rows with all NaN values (McKinney Ch. 7)
        df = df.dropna(how='all')

        # Remove rows where Close price is NaN or <= 0
        df = df[df['Close'].notna() & (df['Close'] > 0)]

        # Ensure OHLC consistency (vectorized operations)
        # High should be >= max(Open, Close)
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))

        # Low should be <= min(Open, Close)
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))

        # Volume should be non-negative
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0).clip(lower=0)

        # Log cleaning results
        cleaned_rows = len(df)
        if original_rows != cleaned_rows:
            logger.info("Cleaned {} data: {} -> {} rows", symbol, original_rows, cleaned_rows)

        return df

    def _handle_timezone(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Handle timezone conversion following McKinney Ch. 11 practices.

        Args:
            df: DataFrame with datetime index
            symbol: Instrument symbol

        Returns:
            DataFrame with UTC timezone
        """
        if df.index.tz is not None:
            # Already timezone-aware, convert to UTC
            df.index = df.index.tz_convert('UTC')
            return df

        # Determine timezone from symbol/config
        timezone_map = {
            'AXJO': 'Australia/Sydney',
            '^AXJO': 'Australia/Sydney',
            'ASX': 'Australia/Sydney',
            '=X': 'UTC',  # Forex pairs
        }

        # Default timezone
        source_tz = 'America/New_York'

        # Check for specific timezone mappings
        for key, tz in timezone_map.items():
            if key in symbol:
                source_tz = tz
                break

        # Localize and convert to UTC
        try:
            df.index = df.index.tz_localize(source_tz, ambiguous='infer').tz_convert('UTC')
            logger.debug("Converted {} timezone: {} -> UTC", symbol, source_tz)
        except Exception as e:
            logger.warning("Timezone conversion failed for {}: {}", symbol, str(e))
            # Fallback to naive UTC
            df.index = df.index.tz_localize('UTC')

        return df

    def _validate_ohlcv_structure(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Validate OHLCV data structure and raise errors for critical issues.

        Args:
            df: DataFrame to validate
            symbol: Instrument symbol

        Raises:
            ValueError: If critical validation fails
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")

        # Check for reasonable price ranges (no negative prices)
        if (df[required_columns] <= 0).any().any():
            raise ValueError(f"Invalid price data detected for {symbol}: negative or zero prices")

        # Check OHLC consistency
        if ((df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])).any():
            raise ValueError(f"OHLC consistency violation detected for {symbol}")

    def download_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        max_workers: int = 4
    ) -> Dict[str, DownloadResult]:
        """
        Download multiple symbols with progress tracking and parallel processing.

        Args:
            symbols: List of Yahoo Finance symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval
            max_workers: Maximum number of concurrent downloads

        Returns:
            Dictionary mapping symbols to DownloadResult objects
        """
        self.download_stats['start_time'] = datetime.now()
        results = {}

        logger.info("Starting batch download of {} symbols", len(symbols))

        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(
                    self.download_single_instrument,
                    symbol,
                    start_date,
                    end_date,
                    interval
                ): symbol for symbol in symbols
            }

            # Process completed downloads
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]

                try:
                    result = future.result()
                    results[symbol] = result

                    # Update statistics
                    self.download_stats['total_downloads'] += 1
                    if result.success:
                        self.download_stats['successful_downloads'] += 1
                        self.download_stats['total_rows'] += result.rows_downloaded
                    else:
                        self.download_stats['failed_downloads'] += 1

                    # Rate limiting
                    rate_limit = self.config.get('pipeline_config', {}).get('download', {}).get('rate_limit_delay', 1)
                    if rate_limit > 0:
                        time.sleep(rate_limit)

                except Exception as e:
                    logger.error("Unexpected error downloading {}: {}", symbol, str(e))
                    results[symbol] = DownloadResult(
                        symbol=symbol,
                        data=None,
                        success=False,
                        error_message=str(e),
                        download_time=datetime.now(),
                        rows_downloaded=0,
                        start_date=start_date,
                        end_date=end_date
                    )
                    self.download_stats['total_downloads'] += 1
                    self.download_stats['failed_downloads'] += 1

        self._log_batch_summary(results)
        return results

    def download_batch_efficient(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, DownloadResult]:
        """
        Efficient batch download using yfinance's group download feature.

        Args:
            symbols: List of symbols to download
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary of DownloadResult objects
        """
        logger.info("Starting efficient batch download for {} symbols", len(symbols))

        try:
            # Use yfinance's efficient batch download
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=False,
                progress=False  # Disable progress bar for cleaner logs
            )

            results = {}

            # Process each symbol's data
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        # Single symbol case
                        symbol_data = data
                    else:
                        # Multi-symbol case - extract data for this symbol
                        symbol_data = data[symbol] if symbol in data.columns.levels[0] else pd.DataFrame()

                    if symbol_data.empty:
                        results[symbol] = DownloadResult(
                            symbol=symbol,
                            data=None,
                            success=False,
                            error_message="No data returned",
                            download_time=datetime.now(),
                            rows_downloaded=0,
                            start_date=start_date,
                            end_date=end_date
                        )
                        continue

                    # Clean and process data
                    symbol_data = self._clean_ohlcv_data(symbol_data, symbol)
                    symbol_data = self._handle_timezone(symbol_data, symbol)

                    results[symbol] = DownloadResult(
                        symbol=symbol,
                        data=symbol_data,
                        success=True,
                        error_message=None,
                        download_time=datetime.now(),
                        rows_downloaded=len(symbol_data),
                        start_date=str(symbol_data.index.min().date()) if not symbol_data.empty else None,
                        end_date=str(symbol_data.index.max().date()) if not symbol_data.empty else None
                    )

                except Exception as e:
                    logger.error("Error processing {} in batch download: {}", symbol, str(e))
                    results[symbol] = DownloadResult(
                        symbol=symbol,
                        data=None,
                        success=False,
                        error_message=str(e),
                        download_time=datetime.now(),
                        rows_downloaded=0,
                        start_date=start_date,
                        end_date=end_date
                    )

            self._log_batch_summary(results)
            return results

        except Exception as e:
            logger.error("Batch download failed: {}", str(e))
            # Return failed results for all symbols
            return {
                symbol: DownloadResult(
                    symbol=symbol,
                    data=None,
                    success=False,
                    error_message=f"Batch download failed: {str(e)}",
                    download_time=datetime.now(),
                    rows_downloaded=0,
                    start_date=start_date,
                    end_date=end_date
                ) for symbol in symbols
            }

    def get_all_symbols(self) -> List[str]:
        """
        Extract all symbols from configuration.

        Returns:
            List of all configured symbols
        """
        symbols = []
        for category in self.config['instruments'].values():
            for instrument_config in category.values():
                symbols.append(instrument_config['symbol'])

        return symbols

    def _log_batch_summary(self, results: Dict[str, DownloadResult]) -> None:
        """Log summary of batch download results."""
        total = len(results)
        successful = sum(1 for r in results.values() if r.success)
        failed = total - successful
        total_rows = sum(r.rows_downloaded for r in results.values() if r.success)

        duration = datetime.now() - self.download_stats['start_time']

        logger.info(
            "Batch download complete: {}/{} successful, {} failed, {} total rows, duration: {}",
            successful, total, failed, total_rows, duration
        )

        # Log failed downloads
        if failed > 0:
            failed_symbols = [symbol for symbol, result in results.items() if not result.success]
            logger.warning("Failed downloads: {}", failed_symbols)

    def get_download_stats(self) -> Dict:
        """Get download statistics."""
        return self.download_stats.copy()


class HybridDataManager:
    """
    Intelligent data manager that combines multiple sources for optimal data quality.

    Features:
    - Smart source selection based on instrument type and date range
    - Alpha Vantage for recent forex data (better quality)
    - yfinance for historical data and indices (broader coverage)
    - Automatic fallback between sources
    - Quality-based data acceptance
    """

    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """Initialize hybrid data manager."""
        # Import Alpha Vantage adapter (optional dependency)
        try:
            from .alpha_vantage import AlphaVantageAdapter
            self.av_adapter = AlphaVantageAdapter(config_path)
            logger.info("Alpha Vantage adapter loaded successfully")
        except ImportError as e:
            logger.warning("Alpha Vantage adapter not available: {}", str(e))
            self.av_adapter = None
        except Exception as e:
            logger.warning("Failed to initialize Alpha Vantage: {}", str(e))
            self.av_adapter = None

        # Initialize yfinance downloader
        self.yf_downloader = MarketDataDownloader()

        # Load configuration
        self.config = self._load_hybrid_config(config_path)

        # Initialize validator for quality checks
        try:
            from .validator import DataValidator
            self.validator = DataValidator()
        except ImportError:
            logger.warning("DataValidator not available - quality checks disabled")
            self.validator = None

        logger.info("HybridDataManager initialized with intelligent source selection")

    def _load_hybrid_config(self, config_path: str) -> Dict:
        """Load hybrid strategy configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning("Hybrid config not found, using defaults")
            return self._default_hybrid_config()

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('hybrid_strategy', self._default_hybrid_config())
        except Exception as e:
            logger.warning("Failed to load hybrid config: {}", str(e))
            return self._default_hybrid_config()

    def _default_hybrid_config(self) -> Dict:
        """Default hybrid configuration."""
        return {
            'recent_data_cutoff_days': 100,
            'overlap_resolution': 'keep_most_recent',
            'min_quality_score': 70,
            'require_both_sources': False
        }

    def download_instrument(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        force_source: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Download instrument data with intelligent source selection.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_source: Force specific source ('alpha_vantage', 'yfinance')

        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        logger.info("Starting hybrid download for {} ({} to {})", symbol, start_date, end_date)

        metadata = {
            'symbol': symbol,
            'sources_used': [],
            'download_strategy': 'hybrid',
            'start_date': start_date,
            'end_date': end_date,
            'quality_metrics': None,
            'source_selection_reason': None
        }

        try:
            # Determine optimal strategy
            strategy = self._select_download_strategy(symbol, start_date, end_date, force_source)
            metadata['source_selection_reason'] = strategy['reason']

            if strategy['method'] == 'alpha_vantage_only':
                return self._download_alpha_vantage_only(symbol, start_date, end_date, metadata)

            elif strategy['method'] == 'yfinance_only':
                return self._download_yfinance_only(symbol, start_date, end_date, metadata)

            elif strategy['method'] == 'hybrid_split':
                return self._download_hybrid_split(symbol, start_date, end_date, metadata)

            else:
                logger.warning("Unknown download strategy: {}", strategy['method'])
                return self._download_yfinance_only(symbol, start_date, end_date, metadata)

        except Exception as e:
            logger.error("Hybrid download failed for {}: {}", symbol, str(e))
            metadata['error'] = str(e)
            return None, metadata

    def _select_download_strategy(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        force_source: Optional[str]
    ) -> Dict[str, str]:
        """Select optimal download strategy based on symbol and date range."""

        # Handle forced source selection
        if force_source == 'alpha_vantage':
            if self.av_adapter and self.av_adapter.is_symbol_supported(symbol):
                return {
                    'method': 'alpha_vantage_only',
                    'reason': 'User forced Alpha Vantage'
                }
            else:
                return {
                    'method': 'yfinance_only',
                    'reason': 'Alpha Vantage not available or symbol not supported'
                }
        elif force_source == 'yfinance':
            return {
                'method': 'yfinance_only',
                'reason': 'User forced yfinance'
            }

        # Calculate date range characteristics
        end = pd.to_datetime(end_date)
        start = pd.to_datetime(start_date)
        days_requested = (end - start).days
        recent_cutoff_days = self.config['recent_data_cutoff_days']

        # Check Alpha Vantage availability
        av_available = (self.av_adapter and
                       self.av_adapter.enabled and
                       self.av_adapter.is_symbol_supported(symbol))

        # Strategy selection logic
        if not av_available:
            return {
                'method': 'yfinance_only',
                'reason': 'Alpha Vantage not available for this symbol'
            }

        # For forex pairs with recent data requests, prefer Alpha Vantage
        if '=X' in symbol and days_requested <= recent_cutoff_days:
            return {
                'method': 'alpha_vantage_only',
                'reason': 'Forex pair with recent data - Alpha Vantage preferred'
            }

        # For indices and ETFs, use yfinance (better support)
        if symbol in ['SPY', 'QQQ'] or symbol.startswith('^'):
            return {
                'method': 'yfinance_only',
                'reason': 'Index/ETF - yfinance has better support'
            }

        # For long historical requests, use hybrid approach
        if days_requested > recent_cutoff_days:
            return {
                'method': 'hybrid_split',
                'reason': f'Long timeframe (>{recent_cutoff_days} days) - using hybrid approach'
            }

        # For short recent requests on supported symbols, try Alpha Vantage first
        return {
            'method': 'alpha_vantage_only',
            'reason': 'Recent data request on supported symbol'
        }

    def _download_alpha_vantage_only(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        metadata: Dict
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Download using Alpha Vantage only."""

        if not self.av_adapter:
            logger.warning("Alpha Vantage not available, falling back to yfinance")
            return self._download_yfinance_only(symbol, start_date, end_date, metadata)

        # Try Alpha Vantage download
        df = self.av_adapter.download(symbol)

        if df is not None and not df.empty:
            # Filter to requested date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]

            if not df.empty:
                metadata['sources_used'] = ['alpha_vantage']

                # Validate and get quality metrics
                if self.validator:
                    df_clean, quality_metrics = self.validator.validate_ohlcv(df, symbol)
                    metadata['quality_metrics'] = quality_metrics.to_dict()
                    return df_clean, metadata
                else:
                    return df, metadata

        # Fallback to yfinance
        logger.info("Alpha Vantage failed for {}, falling back to yfinance", symbol)
        return self._download_yfinance_only(symbol, start_date, end_date, metadata)

    def _download_yfinance_only(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        metadata: Dict
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Download using yfinance only."""

        result = self.yf_downloader.download_single_instrument(symbol, start_date, end_date)

        if result.success and result.data is not None:
            metadata['sources_used'] = ['yfinance']

            # Validate and get quality metrics
            if self.validator:
                df_clean, quality_metrics = self.validator.validate_ohlcv(result.data, symbol)
                metadata['quality_metrics'] = quality_metrics.to_dict()
                return df_clean, metadata
            else:
                return result.data, metadata
        else:
            metadata['error'] = result.error_message
            return None, metadata

    def _download_hybrid_split(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        metadata: Dict
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Download using hybrid approach - historical from yfinance, recent from Alpha Vantage."""

        end = pd.to_datetime(end_date)
        start = pd.to_datetime(start_date)
        recent_cutoff_days = self.config['recent_data_cutoff_days']
        recent_cutoff = end - pd.Timedelta(days=recent_cutoff_days)

        df_historical = None
        df_recent = None
        sources_used = []

        # Get historical data from yfinance (if needed)
        if start < recent_cutoff:
            logger.info("Fetching historical data from yfinance for {}", symbol)
            historical_end = min(recent_cutoff, end)

            result = self.yf_downloader.download_single_instrument(
                symbol,
                start_date,
                historical_end.strftime('%Y-%m-%d')
            )

            if result.success and result.data is not None:
                df_historical = result.data
                sources_used.append('yfinance_historical')
                logger.info("Got {} historical rows from yfinance", len(df_historical))

        # Get recent data from Alpha Vantage (if available and needed)
        if recent_cutoff < end and self.av_adapter:
            logger.info("Attempting Alpha Vantage for recent data for {}", symbol)
            df_av = self.av_adapter.download(symbol)

            if df_av is not None and not df_av.empty:
                # Filter to recent period
                df_recent = df_av[df_av.index >= recent_cutoff]
                df_recent = df_recent[df_recent.index <= end]

                if not df_recent.empty:
                    sources_used.append('alpha_vantage_recent')
                    logger.info("Got {} recent rows from Alpha Vantage", len(df_recent))

        # If Alpha Vantage failed for recent data, get from yfinance
        if df_recent is None or df_recent.empty:
            if recent_cutoff < end:
                logger.info("Alpha Vantage failed, getting recent data from yfinance for {}", symbol)
                result = self.yf_downloader.download_single_instrument(
                    symbol,
                    recent_cutoff.strftime('%Y-%m-%d'),
                    end_date
                )

                if result.success and result.data is not None:
                    df_recent = result.data
                    sources_used.append('yfinance_recent')
                    logger.info("Got {} recent rows from yfinance fallback", len(df_recent))

        # Combine data sources
        combined_df = self._combine_dataframes(df_historical, df_recent, symbol)

        if combined_df is not None and not combined_df.empty:
            metadata['sources_used'] = sources_used

            # Validate combined data
            if self.validator:
                df_clean, quality_metrics = self.validator.validate_ohlcv(combined_df, symbol)
                metadata['quality_metrics'] = quality_metrics.to_dict()
                logger.success("Hybrid download successful for {} ({} rows from {})",
                             symbol, len(df_clean), sources_used)
                return df_clean, metadata
            else:
                logger.success("Hybrid download successful for {} ({} rows from {})",
                             symbol, len(combined_df), sources_used)
                return combined_df, metadata
        else:
            metadata['error'] = 'Failed to get data from any source'
            return None, metadata

    def _combine_dataframes(
        self,
        df_historical: Optional[pd.DataFrame],
        df_recent: Optional[pd.DataFrame],
        symbol: str
    ) -> Optional[pd.DataFrame]:
        """Combine historical and recent dataframes intelligently."""

        if df_historical is not None and df_recent is not None:
            logger.debug("Combining historical ({} rows) and recent ({} rows) data for {}",
                        len(df_historical), len(df_recent), symbol)

            # Concatenate dataframes
            combined = pd.concat([df_historical, df_recent])

            # Handle overlapping dates based on configuration
            overlap_resolution = self.config.get('overlap_resolution', 'keep_most_recent')

            if overlap_resolution == 'keep_most_recent':
                # Keep latest values for duplicate dates
                combined = combined[~combined.index.duplicated(keep='last')]
            elif overlap_resolution == 'alpha_vantage_priority':
                # More complex logic could prioritize Alpha Vantage data
                combined = combined[~combined.index.duplicated(keep='last')]

            # Sort by date
            combined = combined.sort_index()

            logger.debug("Combined data: {} total rows after deduplication", len(combined))
            return combined

        elif df_historical is not None:
            logger.debug("Using historical data only for {} ({} rows)", symbol, len(df_historical))
            return df_historical

        elif df_recent is not None:
            logger.debug("Using recent data only for {} ({} rows)", symbol, len(df_recent))
            return df_recent

        else:
            logger.warning("No data available from any source for {}", symbol)
            return None

    def get_source_capabilities(self) -> Dict[str, Any]:
        """Get information about available data sources and their capabilities."""
        capabilities = {
            'yfinance': {
                'available': True,
                'supported_assets': ['equities', 'etfs', 'indices', 'commodities', 'forex'],
                'strengths': ['Historical data', 'Volume data', 'Broad coverage'],
                'limitations': ['Rate limits', 'Data quality varies']
            },
            'alpha_vantage': {
                'available': self.av_adapter is not None and self.av_adapter.enabled,
                'supported_assets': ['forex', 'some_commodities'],
                'strengths': ['High quality forex data', 'Real-time updates'],
                'limitations': ['Limited free tier', 'Limited instrument coverage']
            }
        }

        if self.av_adapter:
            capabilities['alpha_vantage']['usage_stats'] = self.av_adapter.get_usage_stats()
            capabilities['alpha_vantage']['supported_symbols'] = self.av_adapter.get_supported_symbols()

        return capabilities

    def batch_download(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_workers: int = 3
    ) -> Dict[str, Tuple[Optional[pd.DataFrame], Dict]]:
        """
        Download multiple symbols using hybrid approach.

        Args:
            symbols: List of symbols to download
            start_date: Start date
            end_date: End date
            max_workers: Maximum concurrent downloads

        Returns:
            Dictionary mapping symbols to (DataFrame, metadata) tuples
        """
        logger.info("Starting hybrid batch download for {} symbols", len(symbols))

        results = {}

        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            future_to_symbol = {
                executor.submit(self.download_instrument, symbol, start_date, end_date): symbol
                for symbol in symbols
            }

            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df, metadata = future.result()
                    results[symbol] = (df, metadata)

                    if df is not None:
                        logger.info("✓ {} ({} rows from {})", symbol, len(df), metadata['sources_used'])
                    else:
                        logger.warning("✗ {} failed: {}", symbol, metadata.get('error', 'Unknown error'))

                except Exception as e:
                    logger.error("Error downloading {}: {}", symbol, str(e))
                    results[symbol] = (None, {'error': str(e), 'symbol': symbol})

        # Log batch summary
        successful = sum(1 for df, _ in results.values() if df is not None)
        total = len(results)
        logger.info("Hybrid batch download complete: {}/{} successful", successful, total)

        return results


# Convenience functions for common operations
def download_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Convenience function to download a single symbol.

    Args:
        symbol: Yahoo Finance symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    downloader = MarketDataDownloader()
    result = downloader.download_single_instrument(symbol, start_date, end_date)

    if result.success:
        return result.data
    else:
        raise ValueError(f"Download failed for {symbol}: {result.error_message}")


def download_symbols(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to download multiple symbols.

    Args:
        symbols: List of Yahoo Finance symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    downloader = MarketDataDownloader()
    results = downloader.download_batch_efficient(symbols, start_date, end_date)

    # Return only successful downloads
    return {
        symbol: result.data
        for symbol, result in results.items()
        if result.success and result.data is not None
    }

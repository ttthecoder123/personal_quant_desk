"""
Alpha Vantage API adapter with intelligent caching and rate limiting.
Implements professional-grade API management for financial data.
"""

import os
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any
import pandas as pd
import requests
from loguru import logger
import yaml
from pathlib import Path
from dataclasses import dataclass
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APICallRecord:
    """Record of API call for rate limiting tracking."""
    timestamp: datetime
    symbol: str
    function: str
    success: bool
    response_size: int


class RateLimiter:
    """Professional rate limiter for Alpha Vantage API."""

    def __init__(self, calls_per_minute: int = 5, daily_quota: int = 25):
        """Initialize rate limiter with API constraints."""
        self.calls_per_minute = calls_per_minute
        self.daily_quota = daily_quota
        self.call_history: List[APICallRecord] = []

    def can_make_call(self) -> Tuple[bool, Optional[int]]:
        """
        Check if we can make an API call within rate limits.

        Returns:
            Tuple of (can_call, wait_seconds)
        """
        now = datetime.now()

        # Clean old call records (older than 24 hours)
        cutoff = now - timedelta(hours=24)
        self.call_history = [call for call in self.call_history if call.timestamp > cutoff]

        # Check daily quota
        daily_calls = len(self.call_history)
        if daily_calls >= self.daily_quota:
            logger.warning(f"Daily quota reached: {daily_calls}/{self.daily_quota}")
            return False, None

        # Check per-minute limit
        minute_ago = now - timedelta(minutes=1)
        recent_calls = [call for call in self.call_history if call.timestamp > minute_ago]

        if len(recent_calls) >= self.calls_per_minute:
            # Calculate wait time
            oldest_recent = min(call.timestamp for call in recent_calls)
            wait_seconds = int((oldest_recent + timedelta(minutes=1) - now).total_seconds()) + 1
            logger.info(f"Rate limit reached. Need to wait {wait_seconds} seconds")
            return False, wait_seconds

        return True, 0

    def record_call(self, symbol: str, function: str, success: bool, response_size: int = 0):
        """Record an API call for rate limiting."""
        record = APICallRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            function=function,
            success=success,
            response_size=response_size
        )
        self.call_history.append(record)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        now = datetime.now()
        day_ago = now - timedelta(hours=24)
        minute_ago = now - timedelta(minutes=1)

        daily_calls = len([call for call in self.call_history if call.timestamp > day_ago])
        recent_calls = len([call for call in self.call_history if call.timestamp > minute_ago])

        return {
            'daily_calls_used': daily_calls,
            'daily_quota': self.daily_quota,
            'daily_remaining': self.daily_quota - daily_calls,
            'recent_calls': recent_calls,
            'calls_per_minute_limit': self.calls_per_minute,
            'minute_remaining': self.calls_per_minute - recent_calls
        }


class AlphaVantageCache:
    """SQLite-based cache for Alpha Vantage API responses."""

    def __init__(self, cache_path: str = 'cache/av_cache.db'):
        """Initialize cache database."""
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self.cache_path) as conn:
            # Cache table for API responses
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    function TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    expiry_time DATETIME NOT NULL,
                    data_hash TEXT,
                    response_size INTEGER DEFAULT 0
                )
            ''')

            # Call history table for rate limiting
            conn.execute('''
                CREATE TABLE IF NOT EXISTS call_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    function TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    response_size INTEGER DEFAULT 0,
                    cache_hit BOOLEAN DEFAULT FALSE
                )
            ''')

            # Create indexes for efficient queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_symbol ON api_cache (symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_expiry ON api_cache (expiry_time)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_history_timestamp ON call_history (timestamp)')

            conn.commit()

    def _generate_cache_key(self, symbol: str, function: str, params: Dict) -> str:
        """Generate unique cache key for API request."""
        key_data = f"{symbol}_{function}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, symbol: str, function: str, params: Dict) -> Optional[pd.DataFrame]:
        """
        Get cached response if available and not expired.

        Args:
            symbol: Trading symbol
            function: Alpha Vantage function name
            params: API parameters

        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_key = self._generate_cache_key(symbol, function, params)

        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute('''
                SELECT response_data, timestamp, expiry_time
                FROM api_cache
                WHERE cache_key = ? AND expiry_time > datetime('now')
            ''', (cache_key,))

            row = cursor.fetchone()
            if row:
                try:
                    # Deserialize DataFrame
                    data_json = json.loads(row[0])
                    df = pd.DataFrame.from_dict(data_json)

                    # Reconstruct datetime index if present
                    if 'index' in data_json and data_json['index']:
                        df.index = pd.to_datetime(data_json['index'])

                    logger.debug(f"Cache hit for {symbol} ({function})")
                    return df

                except Exception as e:
                    logger.warning(f"Failed to deserialize cache for {symbol}: {e}")
                    # Remove corrupted cache entry
                    conn.execute('DELETE FROM api_cache WHERE cache_key = ?', (cache_key,))
                    conn.commit()

        return None

    def put(
        self,
        symbol: str,
        function: str,
        params: Dict,
        data: pd.DataFrame,
        expiry_hours: int = 24
    ) -> None:
        """
        Store API response in cache.

        Args:
            symbol: Trading symbol
            function: Alpha Vantage function name
            params: API parameters
            data: DataFrame to cache
            expiry_hours: Hours until cache expires
        """
        cache_key = self._generate_cache_key(symbol, function, params)
        now = datetime.now()
        expiry_time = now + timedelta(hours=expiry_hours)

        try:
            # Serialize DataFrame
            data_dict = data.to_dict()
            if hasattr(data.index, 'strftime'):  # DatetimeIndex
                data_dict['index'] = data.index.strftime('%Y-%m-%d %H:%M:%S').tolist()

            response_data = json.dumps(data_dict)
            data_hash = hashlib.md5(response_data.encode()).hexdigest()

            with sqlite3.connect(self.cache_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO api_cache
                    (cache_key, symbol, function, response_data, timestamp, expiry_time, data_hash, response_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key, symbol, function, response_data,
                    now, expiry_time, data_hash, len(response_data)
                ))
                conn.commit()

            logger.debug(f"Cached response for {symbol} ({function}), expires: {expiry_time}")

        except Exception as e:
            logger.error(f"Failed to cache response for {symbol}: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired cache entries and return count removed."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute('''
                DELETE FROM api_cache WHERE expiry_time <= datetime('now')
            ''')
            removed_count = cursor.rowcount
            conn.commit()

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")

        return removed_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        with sqlite3.connect(self.cache_path) as conn:
            # Total entries
            total_entries = conn.execute('SELECT COUNT(*) FROM api_cache').fetchone()[0]

            # Expired entries
            expired_entries = conn.execute('''
                SELECT COUNT(*) FROM api_cache WHERE expiry_time <= datetime('now')
            ''').fetchone()[0]

            # Cache size
            total_size = conn.execute('''
                SELECT SUM(response_size) FROM api_cache
            ''').fetchone()[0] or 0

            # Recent cache hits
            recent_hits = conn.execute('''
                SELECT COUNT(*) FROM call_history
                WHERE cache_hit = TRUE AND timestamp > datetime('now', '-24 hours')
            ''').fetchone()[0]

            # Recent API calls
            recent_calls = conn.execute('''
                SELECT COUNT(*) FROM call_history
                WHERE timestamp > datetime('now', '-24 hours')
            ''').fetchone()[0]

        cache_hit_rate = (recent_hits / recent_calls * 100) if recent_calls > 0 else 0

        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_hit_rate_24h': round(cache_hit_rate, 2),
            'recent_hits': recent_hits,
            'recent_calls': recent_calls
        }


class AlphaVantageAdapter:
    """
    Professional Alpha Vantage API adapter with caching and rate limiting.

    Features:
    - Intelligent rate limiting with burst handling
    - SQLite-based response caching
    - Comprehensive error handling and retry logic
    - Usage analytics and monitoring
    - Support for multiple data types (FX, commodities, equities)
    """

    def __init__(self, config_path: str = 'config/data_sources.yaml'):
        """Initialize Alpha Vantage adapter."""
        self.config = self._load_config(config_path)
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY not found. Alpha Vantage features disabled.")
            self.enabled = False
            return

        self.enabled = True
        self.base_url = self.config['base_url']

        # Initialize rate limiter
        rate_limits = self.config['rate_limits']
        self.rate_limiter = RateLimiter(
            calls_per_minute=rate_limits['calls_per_minute'],
            daily_quota=rate_limits['daily_quota']
        )

        # Initialize cache
        cache_path = self.config['cache_location']
        self.cache = AlphaVantageCache(cache_path)

        # Load symbol mappings
        self.symbol_mappings = self.config['symbol_mappings']

        logger.info("Alpha Vantage adapter initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load Alpha Vantage configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        return config['alpha_vantage']

    def is_symbol_supported(self, symbol: str) -> bool:
        """Check if symbol is supported by Alpha Vantage."""
        mapping = self.symbol_mappings.get(symbol, {})
        return mapping.get('supported', False)

    def download(
        self,
        symbol: str,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Download data with intelligent caching and rate limiting.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data
            force_refresh: Force API call even if cached data exists

        Returns:
            DataFrame with OHLCV data or None if unavailable
        """
        if not self.enabled:
            logger.debug("Alpha Vantage disabled - no API key")
            return None

        # Check if symbol is supported
        if not self.is_symbol_supported(symbol):
            logger.debug(f"{symbol} not supported by Alpha Vantage")
            return None

        mapping = self.symbol_mappings[symbol]
        function = mapping['function']

        # Check cache first (unless force refresh)
        if use_cache and not force_refresh:
            cached_data = self.cache.get(symbol, function, mapping)
            if cached_data is not None:
                logger.info(f"Using cached Alpha Vantage data for {symbol}")
                self._record_cache_hit(symbol, function)
                return cached_data

        # Check rate limits
        can_call, wait_seconds = self.rate_limiter.can_make_call()
        if not can_call:
            if wait_seconds:
                logger.warning(f"Rate limit hit, need to wait {wait_seconds}s for {symbol}")
            else:
                logger.warning(f"Daily quota exceeded for Alpha Vantage")
            return None

        # Make API call
        try:
            logger.info(f"Fetching {symbol} from Alpha Vantage ({function})")
            df = self._fetch_data(symbol, mapping)

            if df is not None and not df.empty:
                # Cache the response
                expiry_hours = self.config['cache_expiry_hours']
                self.cache.put(symbol, function, mapping, df, expiry_hours)

                # Record successful call
                self.rate_limiter.record_call(symbol, function, True, len(df))
                self._record_api_call(symbol, function, True, len(df))

                logger.success(f"Successfully fetched {len(df)} rows for {symbol} from Alpha Vantage")
                return df
            else:
                # Record failed call
                self.rate_limiter.record_call(symbol, function, False, 0)
                self._record_api_call(symbol, function, False, 0)
                logger.warning(f"No data returned from Alpha Vantage for {symbol}")
                return None

        except Exception as e:
            # Record failed call
            self.rate_limiter.record_call(symbol, function, False, 0)
            self._record_api_call(symbol, function, False, 0)
            logger.error(f"Alpha Vantage API error for {symbol}: {str(e)}")
            return None

    def _fetch_data(self, symbol: str, mapping: Dict) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API based on function type."""
        function = mapping['function']

        if function == 'FX_DAILY':
            return self._fetch_forex(symbol, mapping)
        elif function == 'WTI':
            return self._fetch_commodity_wti(symbol, mapping)
        elif function == 'TIME_SERIES_DAILY':
            return self._fetch_equity(symbol, mapping)
        else:
            logger.warning(f"Unsupported function: {function} for {symbol}")
            return None

    def _fetch_forex(self, symbol: str, mapping: Dict) -> Optional[pd.DataFrame]:
        """Fetch forex data from Alpha Vantage."""
        params = {
            'function': 'FX_DAILY',
            'from_symbol': mapping['from_currency'],
            'to_symbol': mapping['to_currency'],
            'outputsize': mapping.get('outputsize', 'compact'),  # Last 100 days
            'apikey': self.api_key
        }

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError(f"API Limit: {data['Note']}")

        # Extract time series data
        ts_key = 'Time Series FX (Daily)'
        if ts_key not in data:
            logger.warning(f"Expected key '{ts_key}' not found in response for {symbol}")
            return None

        time_series = data[ts_key]
        if not time_series:
            return None

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)

        # Add missing columns for compatibility
        df['Volume'] = 0  # FX doesn't have volume
        df['Adj Close'] = df['Close']  # No adjustments for FX

        # Sort by date
        df = df.sort_index()

        # Add timezone (UTC for FX)
        df.index = df.index.tz_localize('UTC')

        return df

    def _fetch_commodity_wti(self, symbol: str, mapping: Dict) -> Optional[pd.DataFrame]:
        """Fetch WTI crude oil data from Alpha Vantage."""
        params = {
            'function': 'WTI',
            'interval': mapping.get('interval', 'daily'),
            'apikey': self.api_key
        }

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError(f"API Limit: {data['Note']}")

        # Extract data
        if 'data' not in data:
            logger.warning(f"No data found in WTI response for {symbol}")
            return None

        records = data['data']
        if not records:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Rename columns to standard OHLCV format
        df = df.rename(columns={'value': 'Close'})

        # For WTI, we only get close prices, so duplicate for OHLC
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = 0
        df['Adj Close'] = df['Close']

        # Convert to float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Sort by date
        df = df.sort_index()

        # Add timezone
        df.index = df.index.tz_localize('UTC')

        return df

    def _record_cache_hit(self, symbol: str, function: str):
        """Record cache hit for analytics."""
        with sqlite3.connect(self.cache.cache_path) as conn:
            conn.execute('''
                INSERT INTO call_history
                (timestamp, symbol, function, success, response_size, cache_hit)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), symbol, function, True, 0, True))
            conn.commit()

    def _record_api_call(self, symbol: str, function: str, success: bool, response_size: int):
        """Record API call for analytics."""
        with sqlite3.connect(self.cache.cache_path) as conn:
            conn.execute('''
                INSERT INTO call_history
                (timestamp, symbol, function, success, response_size, cache_hit)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), symbol, function, success, response_size, False))
            conn.commit()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        rate_limit_stats = self.rate_limiter.get_usage_stats()
        cache_stats = self.cache.get_cache_stats()

        return {
            'api_enabled': self.enabled,
            'rate_limits': rate_limit_stats,
            'cache': cache_stats,
            'supported_symbols': len([s for s, m in self.symbol_mappings.items() if m.get('supported', False)])
        }

    def cleanup_cache(self) -> Dict[str, int]:
        """Clean up expired cache entries."""
        expired_count = self.cache.cleanup_expired()
        return {'expired_entries_removed': expired_count}

    def get_supported_symbols(self) -> List[str]:
        """Get list of symbols supported by Alpha Vantage."""
        return [
            symbol for symbol, mapping in self.symbol_mappings.items()
            if mapping.get('supported', False)
        ]


# Convenience functions
def create_alpha_vantage_adapter() -> AlphaVantageAdapter:
    """Create Alpha Vantage adapter with default configuration."""
    return AlphaVantageAdapter()


def check_alpha_vantage_support(symbol: str) -> bool:
    """Quick check if symbol is supported by Alpha Vantage."""
    adapter = create_alpha_vantage_adapter()
    return adapter.is_symbol_supported(symbol)
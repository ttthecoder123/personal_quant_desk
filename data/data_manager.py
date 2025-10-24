"""
Data Manager for handling market data feeds, storage, and retrieval.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict
import aiohttp
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
import ccxt
from sqlalchemy import create_engine
import redis
import os

from utils.logger import get_data_logger

log = get_data_logger()


class DataManager:
    """Manages all data operations for the trading system."""

    def __init__(self, config: dict):
        """Initialize the data manager."""
        self.config = config
        self.data_cache = {}
        self.active_feeds = {}
        self.db_engine = None
        self.redis_client = None
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize database and cache connections."""
        try:
            # Initialize PostgreSQL connection
            db_config = self.config['storage']['database']
            db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            self.db_engine = create_engine(db_url)
            log.info("Database connection established")

            # Initialize Redis connection
            cache_config = self.config['storage']['cache']
            self.redis_client = redis.Redis(
                host=cache_config['host'],
                port=cache_config['port'],
                decode_responses=True
            )
            log.info("Redis cache connection established")

        except Exception as e:
            log.error(f"Failed to initialize connections: {str(e)}")

    async def start_feeds(self):
        """Start all configured data feeds."""
        log.info("Starting data feeds...")

        # Initialize Alpha Vantage
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            self.active_feeds['alpha_vantage'] = {
                'timeseries': TimeSeries(
                    key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                    output_format='pandas'
                ),
                'forex': ForeignExchange(
                    key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                    output_format='pandas'
                )
            }

        # Initialize Yahoo Finance (no key required)
        self.active_feeds['yahoo_finance'] = yf

        log.success("Data feeds started successfully")

    async def get_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch latest market data for all configured instruments."""
        market_data = {}

        try:
            # Fetch commodity data
            commodities = await self._fetch_commodity_data()
            market_data.update(commodities)

            # Fetch index data
            indices = await self._fetch_index_data()
            market_data.update(indices)

            # Fetch forex data
            forex = await self._fetch_forex_data()
            market_data.update(forex)

            log.info(f"Fetched data for {len(market_data)} instruments")
            return market_data

        except Exception as e:
            log.error(f"Error fetching market data: {str(e)}")
            return market_data

    async def _fetch_commodity_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch commodity futures data."""
        commodity_data = {}

        for commodity, info in self.config.get('instruments', {}).get('commodities', {}).items():
            try:
                # Check cache first
                cached = self._get_cached_data(commodity)
                if cached is not None:
                    commodity_data[commodity] = cached
                    continue

                # Fetch from Yahoo Finance
                symbol = info['symbol']  # Symbol already includes =F suffix
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")

                if not data.empty:
                    commodity_data[commodity] = data
                    self._cache_data(commodity, data)
                    log.debug(f"Fetched data for {commodity}")

            except Exception as e:
                log.error(f"Error fetching {commodity} data: {str(e)}")

        return commodity_data

    async def _fetch_index_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch index/ETF data."""
        index_data = {}

        for index, info in self.config.get('instruments', {}).get('indices', {}).items():
            try:
                # Check cache first
                cached = self._get_cached_data(index)
                if cached is not None:
                    index_data[index] = cached
                    continue

                # Fetch from Yahoo Finance
                symbol = info['symbol']
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")

                if not data.empty:
                    index_data[index] = data
                    self._cache_data(index, data)
                    log.debug(f"Fetched data for {index}")

            except Exception as e:
                log.error(f"Error fetching {index} data: {str(e)}")

        return index_data

    async def _fetch_forex_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch forex pair data."""
        forex_data = {}

        for pair, info in self.config.get('instruments', {}).get('forex', {}).items():
            try:
                # Check cache first
                cached = self._get_cached_data(pair)
                if cached is not None:
                    forex_data[pair] = cached
                    continue

                # Fetch from Yahoo Finance (forex symbols)
                symbol = info['symbol']  # Symbol already includes =X suffix
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")

                if not data.empty:
                    forex_data[pair] = data
                    self._cache_data(pair, data)
                    log.debug(f"Fetched data for {pair}")

            except Exception as e:
                log.error(f"Error fetching {pair} data: {str(e)}")

        return forex_data

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a specific symbol.

        Args:
            symbol: The trading symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (1m, 5m, 1h, 1d, etc.)

        Returns:
            DataFrame with historical data
        """
        try:
            # Try to get from database first
            data = self._get_from_database(symbol, start_date, end_date)

            if data is not None and not data.empty:
                log.debug(f"Retrieved {symbol} data from database")
                return data

            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if not data.empty:
                # Store in database for future use
                self._store_in_database(symbol, data)
                log.info(f"Fetched historical data for {symbol}")

            return data

        except Exception as e:
            log.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from Redis cache."""
        try:
            if self.redis_client:
                cached = self.redis_client.get(f"market_data:{symbol}")
                if cached:
                    # Convert JSON back to DataFrame
                    import json
                    data_dict = json.loads(cached)
                    return pd.DataFrame(data_dict)
        except Exception as e:
            log.debug(f"Cache miss for {symbol}: {str(e)}")

        return None

    def _cache_data(self, symbol: str, data: pd.DataFrame):
        """Store data in Redis cache."""
        try:
            if self.redis_client and not data.empty:
                # Convert DataFrame to JSON
                import json
                data_json = data.to_json()
                ttl = self.config['storage']['cache'].get('ttl', 3600)
                self.redis_client.setex(
                    f"market_data:{symbol}",
                    ttl,
                    data_json
                )
                log.debug(f"Cached data for {symbol}")
        except Exception as e:
            log.debug(f"Failed to cache data for {symbol}: {str(e)}")

    def _get_from_database(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Retrieve historical data from database."""
        try:
            if self.db_engine:
                query = f"""
                SELECT * FROM market_data
                WHERE symbol = '{symbol}'
                AND timestamp >= '{start_date}'
                AND timestamp <= '{end_date}'
                ORDER BY timestamp
                """
                return pd.read_sql(query, self.db_engine)
        except Exception as e:
            log.debug(f"Database query failed: {str(e)}")

        return None

    def _store_in_database(self, symbol: str, data: pd.DataFrame):
        """Store historical data in database."""
        try:
            if self.db_engine and not data.empty:
                data['symbol'] = symbol
                data.to_sql(
                    'market_data',
                    self.db_engine,
                    if_exists='append',
                    index=True
                )
                log.debug(f"Stored {len(data)} records for {symbol} in database")
        except Exception as e:
            log.error(f"Failed to store data in database: {str(e)}")

    async def stop(self):
        """Stop data feeds and close connections."""
        log.info("Stopping data feeds...")

        # Close database connection
        if self.db_engine:
            self.db_engine.dispose()

        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()

        log.info("Data feeds stopped")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given data.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Simple Moving Averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()

            # Exponential Moving Averages
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()

            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (2 * bb_std)
            data['BB_Lower'] = data['BB_Middle'] - (2 * bb_std)

            # ATR (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['ATR'] = true_range.rolling(window=14).mean()

            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

            log.debug("Technical indicators calculated")
            return data

        except Exception as e:
            log.error(f"Error calculating indicators: {str(e)}")
            return data
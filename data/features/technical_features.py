"""
Technical Indicators using TA-Lib (Chan's approach).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, technical features will be limited")


class TechnicalFeatures:
    """
    Technical indicators using TA-Lib (Chan's approach).
    """
    
    def __init__(self):
        self.feature_names = []
        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not installed. Install with: pip install TA-Lib")
    
    def compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum and trend indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)
        
        if not TALIB_AVAILABLE:
            return self._compute_momentum_fallback(df)
        
        for period in [14, 30]:
            features[f'rsi_{period}'] = talib.RSI(df['Close'].values, timeperiod=period)
        
        macd, signal, hist = talib.MACD(df['Close'].values)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = hist
        
        slowk, slowd = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        
        features['adx'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
        features['plus_di'] = talib.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values)
        features['minus_di'] = talib.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values)
        
        for short, long in [(10, 20), (20, 50), (50, 200)]:
            ma_short = talib.SMA(df['Close'].values, timeperiod=short)
            ma_long = talib.SMA(df['Close'].values, timeperiod=long)
            features[f'ma_cross_{short}_{long}'] = ma_short - ma_long
            features[f'ma_cross_pct_{short}_{long}'] = (ma_short - ma_long) / ma_long
        
        logger.debug("Computed {} momentum features", features.shape[1])
        return features
    
    def _compute_momentum_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback momentum indicators without TA-Lib."""
        features = pd.DataFrame(index=df.index)
        
        for period in [14, 30]:
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=period).mean()
            loss = (-delta).clip(lower=0).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        for short, long in [(10, 20), (20, 50), (50, 200)]:
            ma_short = df['Close'].rolling(short).mean()
            ma_long = df['Close'].rolling(long).mean()
            features[f'ma_cross_{short}_{long}'] = ma_short - ma_long
            features[f'ma_cross_pct_{short}_{long}'] = (ma_short - ma_long) / ma_long
        
        return features
    
    def compute_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean reversion indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with mean reversion features
        """
        features = pd.DataFrame(index=df.index)
        
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20)
            features['bb_upper'] = upper
            features['bb_lower'] = lower
            features['bb_middle'] = middle
            features['bb_bandwidth'] = (upper - lower) / middle
            features['bb_position'] = (df['Close'] - lower) / (upper - lower)
        else:
            middle = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            features['bb_upper'] = middle + 2 * std
            features['bb_lower'] = middle - 2 * std
            features['bb_middle'] = middle
            features['bb_bandwidth'] = (features['bb_upper'] - features['bb_lower']) / middle
            features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        for period in [20, 60]:
            ma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['Close'] - ma) / std
        
        log_price = df['Close'].apply(np.log)
        for period in [20, 60]:
            cov = log_price.diff().rolling(period).cov(log_price.shift())
            var = log_price.rolling(period).var()
            features[f'mean_reversion_speed_{period}'] = -cov / var
        
        logger.debug("Computed {} mean reversion features", features.shape[1])
        return features
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical features
        """
        logger.info("Computing all technical features for {} rows", len(df))
        
        features = pd.concat([
            self.compute_momentum_indicators(df),
            self.compute_mean_reversion_indicators(df)
        ], axis=1)
        
        self.feature_names = list(features.columns)
        logger.success("Computed {} technical features", len(self.feature_names))
        
        return features

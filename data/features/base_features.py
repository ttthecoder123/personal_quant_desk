"""
Base Feature Engineering - Core price and volume transformations.
Following Jansen's ML for Trading Ch. 4 recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger


class BaseFeatures:
    """
    Core price and volume transformations following Jansen Ch. 4.
    
    All features are point-in-time to prevent look-ahead bias.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-horizon returns with proper handling of gaps.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with return features
        """
        features = pd.DataFrame(index=df.index)
        
        for period in [1, 2, 3, 5, 10, 20, 30, 60, 90, 120, 180, 252]:
            features[f'return_{period}d'] = df['Close'].pct_change(period)
        
        features['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        features['log_return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = features['return_1d'].shift(lag)
        
        for period in [5, 20, 60]:
            returns = df['Close'].pct_change()
            features[f'return_mean_{period}d'] = returns.rolling(period).mean()
            features[f'return_std_{period}d'] = returns.rolling(period).std()
            features[f'return_skew_{period}d'] = returns.rolling(period).skew()
            features[f'return_kurt_{period}d'] = returns.rolling(period).kurt()
        
        features['return_autocorr_1'] = df['Close'].pct_change().rolling(20).apply(lambda x: x.autocorr(lag=1))
        features['return_autocorr_5'] = df['Close'].pct_change().rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        for period in [1, 5, 20]:
            features[f'forward_return_{period}d'] = df['Close'].shift(-period).pct_change(period)
        
        logger.debug("Computed {} return features", features.shape[1])
        return features
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features (Jansen Ch. 4.2).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)
        
        features['dollar_volume'] = df['Close'] * df['Volume']
        features['log_volume'] = np.log(df['Volume'] + 1)
        
        for period in [5, 10, 20, 40, 60]:
            features[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['Volume'] / features[f'volume_ma_{period}']
            features[f'volume_std_{period}'] = df['Volume'].rolling(period).std()
            features[f'volume_std_ratio_{period}'] = features[f'volume_std_{period}'] / features[f'volume_ma_{period}']
        
        for lag in [1, 2, 3, 5]:
            features[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        for period in [5, 20]:
            features[f'vwap_{period}'] = (df['Close'] * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
            features[f'price_to_vwap_{period}'] = df['Close'] / features[f'vwap_{period}']
        
        features['amihud_illiquidity'] = (df['Close'].pct_change().abs() / features['dollar_volume']).rolling(20).mean()
        
        price_change = df['Close'].diff()
        features['volume_price_corr_20'] = df['Volume'].rolling(20).corr(price_change.abs())
        
        logger.debug("Computed {} volume features", features.shape[1])
        return features
    
    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility estimates using multiple methods.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)
        
        returns = df['Close'].pct_change()
        for period in [5, 10, 20, 30, 60, 90]:
            features[f'realized_vol_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
        
        for period in [10, 20, 60]:
            log_hl = (df['High'] / df['Low']).apply(np.log)
            features[f'parkinson_vol_{period}'] = np.sqrt(252 * (log_hl ** 2 / (4 * np.log(2))).rolling(period).mean())
        
        for period in [10, 20, 60]:
            features[f'garman_klass_vol_{period}'] = np.sqrt(252 * (
                0.5 * ((df['High'] / df['Low']).apply(np.log) ** 2) -
                (2 * np.log(2) - 1) * ((df['Close'] / df['Open']).apply(np.log) ** 2)
            ).rolling(period).mean())
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        for period in [10, 20, 40]:
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_ratio_{period}'] = true_range / features[f'atr_{period}']
        
        features['close_to_high'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10)
        features['close_to_low'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        for period in [20, 60]:
            features[f'vol_of_vol_{period}'] = features[f'realized_vol_{period}d'].rolling(20).std()
        
        logger.debug("Computed {} volatility features", features.shape[1])
        return features
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all base features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all base features
        """
        logger.info("Computing all base features for {} rows", len(df))
        
        features = pd.concat([
            self.compute_returns(df),
            self.compute_volume_features(df),
            self.compute_volatility_features(df)
        ], axis=1)
        
        self.feature_names = list(features.columns)
        logger.success("Computed {} base features", len(self.feature_names))
        
        return features

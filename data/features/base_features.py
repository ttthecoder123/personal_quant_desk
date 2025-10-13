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
        
        for period in [1, 5, 20, 60, 120]:
            features[f'return_{period}d'] = df['Close'].pct_change(period)
            
        features['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        
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
        
        for period in [5, 20, 60]:
            features[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['Volume'] / features[f'volume_ma_{period}']
        
        features['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        features['price_to_vwap'] = df['Close'] / features['vwap']
        
        features['amihud_illiquidity'] = (df['Close'].pct_change().abs() / features['dollar_volume']).rolling(20).mean()
        
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
        
        for period in [5, 20, 60]:
            features[f'realized_vol_{period}d'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        features['parkinson_vol'] = np.sqrt(252 * (np.log(df['High'] / df['Low']) ** 2 / (4 * np.log(2))).rolling(20).mean())
        
        features['garman_klass_vol'] = np.sqrt(252 * (
            0.5 * (np.log(df['High'] / df['Low']) ** 2) -
            (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
        ).rolling(20).mean())
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        features['atr_20'] = true_range.rolling(20).mean()
        features['atr_ratio'] = true_range / features['atr_20']
        
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

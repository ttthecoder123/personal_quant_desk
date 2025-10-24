"""
Market Regime Detection Features.
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class RegimeFeatures:
    """
    Market regime detection features.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility regime classification.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility regime features
        """
        features = pd.DataFrame(index=df.index)
        
        returns = df['Close'].pct_change()
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        
        features['vol_percentile'] = vol_20.rolling(252).rank(pct=True)
        
        features['vol_regime'] = pd.cut(
            features['vol_percentile'], 
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2]
        ).astype(float)
        
        features['vol_of_vol'] = vol_20.rolling(20).std()
        
        logger.debug("Computed {} volatility regime features", features.shape[1])
        return features
    
    def compute_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend regime detection.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend regime features
        """
        features = pd.DataFrame(index=df.index)
        
        for period in [20, 60, 120]:
            sma = df['Close'].rolling(period).mean()
            features[f'trend_{period}'] = (df['Close'] - sma) / sma
            features[f'trend_strength_{period}'] = df['Close'].rolling(period).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean() if len(x) == period else np.nan,
                raw=False
            )
        
        if TALIB_AVAILABLE:
            features['trend_quality'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
        else:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.DataFrame({
                'hl': high_low,
                'hc': high_close,
                'lc': low_close
            }).max(axis=1)
            features['trend_quality'] = true_range.rolling(14).mean()
        
        logger.debug("Computed {} trend regime features", features.shape[1])
        return features
    
    def detect_structural_breaks(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Detect structural breaks using CUSUM (López de Prado).
        
        Args:
            df: DataFrame with OHLCV data
            window: Window for CUSUM calculation
            
        Returns:
            DataFrame with structural break features
        """
        features = pd.DataFrame(index=df.index)
        
        returns = df['Close'].pct_change()
        
        mean_return = returns.rolling(window).mean()
        std_return = returns.rolling(window).std()
        standardized_returns = (returns - mean_return) / std_return
        
        cusum = standardized_returns.fillna(0).cumsum()
        features['cusum'] = cusum
        features['cusum_signal'] = (np.abs(cusum) > 3).astype(int)
        
        logger.debug("Computed {} structural break features", features.shape[1])
        return features
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all regime features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all regime features
        """
        logger.info("Computing all regime features for {} rows", len(df))
        
        features = pd.concat([
            self.compute_volatility_regime(df),
            self.compute_trend_regime(df),
            self.detect_structural_breaks(df)
        ], axis=1)
        
        self.feature_names = list(features.columns)
        logger.success("Computed {} regime features", len(self.feature_names))
        
        return features

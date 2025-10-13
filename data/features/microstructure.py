"""
Market Microstructure Features (López de Prado Ch. 19).
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class MicrostructureFeatures:
    """
    Market microstructure features (López de Prado Ch. 19).
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute_roll_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Roll (1984) bid-ask spread estimator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with spread features
        """
        features = pd.DataFrame(index=df.index)
        
        price_diff = df['Close'].diff()
        cov = price_diff.rolling(20).cov(price_diff.shift())
        features['roll_spread'] = 2 * np.sqrt(np.abs(cov))
        
        features['high_low_spread'] = 2 * (df['High'] - df['Low']) / (df['High'] + df['Low'])
        
        logger.debug("Computed {} spread features", features.shape[1])
        return features
    
    def compute_kyle_lambda(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kyle's lambda - price impact coefficient.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Kyle's lambda features
        """
        features = pd.DataFrame(index=df.index)
        
        returns = df['Close'].pct_change()
        signed_volume = df['Volume'] * np.sign(returns)
        
        for period in [20, 60]:
            lambda_values = []
            for i in range(period, len(df)):
                window_returns = returns.iloc[i-period:i].abs()
                window_volume = signed_volume.iloc[i-period:i]
                
                vol_var = window_volume.var()
                if pd.notna(vol_var) and vol_var > 0:
                    cov_val = window_returns.cov(window_volume)
                    if pd.notna(cov_val):
                        lambda_val = float(cov_val) / float(vol_var)
                    else:
                        lambda_val = np.nan
                else:
                    lambda_val = np.nan
                lambda_values.append(lambda_val)
            
            features[f'kyle_lambda_{period}'] = pd.Series(
                [np.nan] * period + lambda_values, 
                index=df.index
            )
        
        logger.debug("Computed {} Kyle's lambda features", features.shape[1])
        return features
    
    def compute_tick_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lee-Ready tick rule for trade classification.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with tick rule features
        """
        features = pd.DataFrame(index=df.index)
        
        price_change = df['Close'].diff()
        
        features['tick_direction'] = np.sign(price_change)
        features['tick_direction'] = features['tick_direction'].ffill()
        
        features['cumulative_signed_volume'] = (df['Volume'] * features['tick_direction']).rolling(20).sum()
        
        buy_volume = df['Volume'].where(features['tick_direction'] > 0, 0)
        sell_volume = df['Volume'].where(features['tick_direction'] < 0, 0)
        total_volume = df['Volume'].rolling(20).sum()
        features['order_flow_imbalance'] = (buy_volume.rolling(20).sum() - sell_volume.rolling(20).sum()) / total_volume
        
        logger.debug("Computed {} tick rule features", features.shape[1])
        return features
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all microstructure features
        """
        logger.info("Computing all microstructure features for {} rows", len(df))
        
        features = pd.concat([
            self.compute_roll_spread(df),
            self.compute_kyle_lambda(df),
            self.compute_tick_rule(df)
        ], axis=1)
        
        self.feature_names = list(features.columns)
        logger.success("Computed {} microstructure features", len(self.feature_names))
        
        return features

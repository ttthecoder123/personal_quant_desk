"""
Commodity-Specific Features (Schofield's Commodity Derivatives).
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class CommodityFeatures:
    """
    Features specific to commodity futures (Schofield).
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute_roll_yield(self, front_month: pd.Series, next_month: pd.Series) -> pd.Series:
        """
        Calculate roll yield from futures curve.
        
        Args:
            front_month: Front month futures prices
            next_month: Next month futures prices
            
        Returns:
            Series with roll yield
        """
        roll_yield = (front_month - next_month) / front_month
        return roll_yield
    
    def compute_convenience_yield(
        self, 
        spot: pd.Series, 
        futures: pd.Series, 
        risk_free_rate: float = 0.03, 
        time_to_maturity: float = 1/12
    ) -> pd.Series:
        """
        Estimate convenience yield.
        
        Args:
            spot: Spot prices
            futures: Futures prices
            risk_free_rate: Risk-free rate
            time_to_maturity: Time to maturity in years
            
        Returns:
            Series with convenience yield
        """
        storage_cost = 0.02
        log_ratio = (spot / futures).apply(np.log)
        convenience_yield = (
            log_ratio + 
            (risk_free_rate + storage_cost) * time_to_maturity
        ) / time_to_maturity
        return convenience_yield
    
    def detect_contango_backwardation(self, front: pd.Series, next: pd.Series) -> pd.Series:
        """
        Detect market structure.
        
        Args:
            front: Front month prices
            next: Next month prices
            
        Returns:
            Series with market structure (1=backwardation, -1=contango, 0=flat)
        """
        structure = pd.Series(index=front.index, dtype=float)
        structure[front > next] = 1
        structure[front < next] = -1
        structure[front == next] = 0
        return structure
    
    def compute_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Seasonal patterns in commodities.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with seasonality features
        """
        features = pd.DataFrame(index=df.index)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping seasonality features")
            return features
        
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        returns = df['Close'].pct_change()
        monthly_avg = returns.groupby(df.index.month).transform('mean')
        features['seasonal_strength'] = monthly_avg
        
        def days_to_summer(date):
            year = date.year
            summer_start = pd.Timestamp(year, 5, 25)
            if date.month <= 5:
                return (summer_start - date).days
            else:
                next_summer = pd.Timestamp(year + 1, 5, 25)
                return (next_summer - date).days
        
        features['days_to_summer'] = df.index.to_series().apply(days_to_summer)
        
        logger.debug("Computed {} seasonality features", features.shape[1])
        return features
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all commodity-specific features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all commodity features
        """
        logger.info("Computing all commodity features for {} rows", len(df))
        
        features = self.compute_seasonality_features(df)
        
        self.feature_names = list(features.columns)
        logger.success("Computed {} commodity features", len(self.feature_names))
        
        return features

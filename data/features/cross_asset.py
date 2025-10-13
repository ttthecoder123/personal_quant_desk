"""
Cross-Asset Relationships and Correlations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class CrossAssetFeatures:
    """
    Cross-asset relationships and correlations.
    """
    
    def __init__(self, asset_data: Dict[str, pd.DataFrame]):
        """
        Initialize with data for all assets.
        
        Args:
            asset_data: Dictionary mapping symbols to DataFrames with OHLCV data
        """
        self.asset_data = asset_data
        self.feature_names = []
    
    def compute_correlations(self, window: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Rolling correlations between assets.
        
        Args:
            window: Rolling window for correlation calculation
            
        Returns:
            Dictionary mapping symbols to correlation features
        """
        features = {}
        
        returns = {symbol: data['Close'].pct_change() for symbol, data in self.asset_data.items()}
        
        for symbol1 in returns:
            df_features = pd.DataFrame(index=returns[symbol1].index)
            
            for symbol2 in returns:
                if symbol1 != symbol2:
                    correlation = returns[symbol1].rolling(window).corr(returns[symbol2])
                    df_features[f'corr_{symbol2}_{window}d'] = correlation
            
            if 'SPY' in returns and symbol1 != 'SPY':
                cov = returns[symbol1].rolling(window).cov(returns['SPY'])
                var_spy = returns['SPY'].rolling(window).var()
                df_features[f'beta_spy_{window}d'] = cov / var_spy
            
            features[symbol1] = df_features
            logger.debug("Computed {} correlation features for {}", df_features.shape[1], symbol1)
        
        return features
    
    def compute_spread_features(self) -> Dict[str, pd.DataFrame]:
        """
        Spread features for pairs trading.
        
        Returns:
            Dictionary mapping symbols to spread features
        """
        features = {}
        
        pairs = [
            ('GC=F', 'HG=F'),
            ('CL=F', 'GC=F'),
            ('SPY', 'QQQ'),
            ('AUDUSD=X', 'GC=F')
        ]
        
        for symbol1, symbol2 in pairs:
            if symbol1 in self.asset_data and symbol2 in self.asset_data:
                price1 = self.asset_data[symbol1]['Close']
                price2 = self.asset_data[symbol2]['Close']
                
                log_spread = (price1 / price2).apply(np.log)
                
                for window in [20, 60]:
                    mean = log_spread.rolling(window).mean()
                    std = log_spread.rolling(window).std()
                    zscore = (log_spread - mean) / std
                    
                    if symbol1 not in features:
                        features[symbol1] = pd.DataFrame(index=price1.index)
                    
                    features[symbol1][f'spread_zscore_{symbol2}_{window}d'] = zscore
                
                logger.debug("Computed spread features for {} vs {}", symbol1, symbol2)
        
        return features
    
    def compute_all(self, window: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Compute all cross-asset features.
        
        Args:
            window: Rolling window for calculations
            
        Returns:
            Dictionary mapping symbols to all cross-asset features
        """
        logger.info("Computing all cross-asset features for {} symbols", len(self.asset_data))
        
        correlations = self.compute_correlations(window)
        spreads = self.compute_spread_features()
        
        all_features = {}
        for symbol in self.asset_data.keys():
            feature_list = []
            
            if symbol in correlations:
                feature_list.append(correlations[symbol])
            
            if symbol in spreads:
                feature_list.append(spreads[symbol])
            
            if feature_list:
                all_features[symbol] = pd.concat(feature_list, axis=1)
                logger.success("Computed {} cross-asset features for {}", 
                             all_features[symbol].shape[1], symbol)
        
        return all_features

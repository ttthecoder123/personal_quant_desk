"""
Feature Storage Manager with versioning and metadata tracking.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
import json
from datetime import datetime
import pytz


class FeatureStore:
    """
    Manage feature storage and retrieval with versioning.
    """
    
    def __init__(self, base_path: str = 'data/features/computed/'):
        """
        Initialize feature store.
        
        Args:
            base_path: Base path for feature storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_path / 'metadata'
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("FeatureStore initialized at {}", self.base_path)
    
    def save_features(
        self, 
        features: pd.DataFrame, 
        symbol: str, 
        feature_set: str, 
        version: str = 'v1'
    ) -> None:
        """
        Save features with versioning.
        
        Args:
            features: DataFrame with features
            symbol: Instrument symbol
            feature_set: Name of feature set (e.g., 'base', 'technical')
            version: Version string
        """
        path = self.base_path / f'{symbol}_{feature_set}_{version}.parquet'
        
        metadata = {
            'symbol': symbol,
            'feature_set': feature_set,
            'version': version,
            'created_at': datetime.now(pytz.UTC).isoformat(),
            'n_features': features.shape[1],
            'n_observations': features.shape[0],
            'feature_names': list(features.columns),
            'date_range': [
                str(features.index.min()),
                str(features.index.max())
            ]
        }
        
        features.to_parquet(path, compression='snappy')
        
        metadata_path = self.metadata_path / f'{symbol}_{feature_set}_{version}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Saved {} features for {}/{} to {}", 
                   features.shape[1], symbol, feature_set, path)
    
    def load_features(
        self, 
        symbol: str, 
        feature_sets: Optional[List[str]] = None, 
        version: str = 'v1'
    ) -> pd.DataFrame:
        """
        Load and combine multiple feature sets.
        
        Args:
            symbol: Instrument symbol
            feature_sets: List of feature sets to load (None for all available)
            version: Version string
            
        Returns:
            Combined DataFrame with all requested features
        """
        if feature_sets is None:
            feature_sets = ['base', 'technical', 'microstructure', 'regime']
        
        features_list = []
        for feature_set in feature_sets:
            path = self.base_path / f'{symbol}_{feature_set}_{version}.parquet'
            if path.exists():
                df = pd.read_parquet(path)
                features_list.append(df)
                logger.debug("Loaded {} features from {}", df.shape[1], feature_set)
            else:
                logger.warning("Feature set {} not found for {}", feature_set, symbol)
        
        if not features_list:
            raise FileNotFoundError(f'No features found for {symbol}')
        
        combined = pd.concat(features_list, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        logger.success("Loaded {} total features for {}", combined.shape[1], symbol)
        return combined
    
    def get_metadata(self, symbol: str, feature_set: str, version: str = 'v1') -> Optional[Dict[str, Any]]:
        """
        Get metadata for a feature set.
        
        Args:
            symbol: Instrument symbol
            feature_set: Feature set name
            version: Version string
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.metadata_path / f'{symbol}_{feature_set}_{version}.json'
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_available_features(self, symbol: str) -> List[str]:
        """
        List all available feature sets for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            List of available feature set names
        """
        pattern = f'{symbol}_*_*.parquet'
        files = list(self.base_path.glob(pattern))
        
        feature_sets = set()
        for file in files:
            parts = file.stem.split('_')
            if len(parts) >= 3:
                feature_set = '_'.join(parts[1:-1])
                feature_sets.add(feature_set)
        
        return sorted(list(feature_sets))
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get summary of stored features.
        
        Returns:
            Dictionary with storage statistics
        """
        parquet_files = list(self.base_path.glob('*.parquet'))
        
        total_size = sum(f.stat().st_size for f in parquet_files)
        
        symbols = set()
        for file in parquet_files:
            symbol = file.stem.split('_')[0]
            symbols.add(symbol)
        
        return {
            'total_files': len(parquet_files),
            'total_size_mb': total_size / (1024 * 1024),
            'unique_symbols': len(symbols),
            'symbols': sorted(list(symbols)),
            'storage_path': str(self.base_path)
        }

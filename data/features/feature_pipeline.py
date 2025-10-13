"""
Main Feature Engineering Pipeline Orchestration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from loguru import logger
import asyncio
from datetime import datetime

from .base_features import BaseFeatures
from .technical_features import TechnicalFeatures
from .microstructure import MicrostructureFeatures
from .regime_features import RegimeFeatures
from .cross_asset import CrossAssetFeatures
from .commodity_specific import CommodityFeatures
from .feature_store import FeatureStore

import sys
sys.path.append(str(Path(__file__).parent.parent))
from ingestion.storage import ParquetStorage


class FeaturePipeline:
    """
    Main orchestration for feature engineering.
    """
    
    def __init__(self, config_path: str = 'data/features/config/feature_config.yaml'):
        """
        Initialize feature pipeline.
        
        Args:
            config_path: Path to feature configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.storage = ParquetStorage()
        self.feature_store = FeatureStore()
        
        self.base_calc = BaseFeatures()
        self.tech_calc = TechnicalFeatures()
        self.micro_calc = MicrostructureFeatures()
        self.regime_calc = RegimeFeatures()
        self.commodity_calc = CommodityFeatures()
        
        logger.info("FeaturePipeline initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning("Config file not found: {}, using defaults", self.config_path)
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'symbols': ['CL=F', 'GC=F', 'HG=F', 'SPY', 'QQQ', '^AXJO', 
                       'AUDUSD=X', 'USDJPY=X', 'EURUSD=X'],
            'feature_sets': {
                'base': {'returns': [1, 5, 20, 60, 120], 'volume_windows': [5, 20, 60]},
                'technical': {'rsi_periods': [14, 30], 'ma_pairs': [[10, 20], [20, 50]]},
                'microstructure': {'spread_windows': [20, 60]},
                'regime': {'vol_lookback': 252, 'trend_periods': [20, 60, 120]}
            },
            'quality_checks': {
                'max_missing_pct': 0.05,
                'min_observations': 252
            }
        }
    
    def process_symbol(
        self, 
        symbol: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process all features for a single symbol.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with all computed features
        """
        logger.info("Processing features for {}", symbol)
        
        df = self.storage.load_timeseries(symbol, start_date, end_date)
        
        if df.empty:
            logger.error("No data found for {}", symbol)
            return pd.DataFrame()
        
        logger.info("Loaded {} rows for {}", len(df), symbol)
        
        features = {}
        
        logger.info("Computing base features...")
        base = self.base_calc.compute_all(df)
        features['base'] = base
        self.feature_store.save_features(base, symbol, 'base')
        
        logger.info("Computing technical features...")
        technical = self.tech_calc.compute_all(df)
        features['technical'] = technical
        self.feature_store.save_features(technical, symbol, 'technical')
        
        logger.info("Computing microstructure features...")
        microstructure = self.micro_calc.compute_all(df)
        features['microstructure'] = microstructure
        self.feature_store.save_features(microstructure, symbol, 'microstructure')
        
        logger.info("Computing regime features...")
        regime = self.regime_calc.compute_all(df)
        features['regime'] = regime
        self.feature_store.save_features(regime, symbol, 'regime')
        
        if symbol in ['CL=F', 'GC=F', 'HG=F']:
            logger.info("Computing commodity-specific features...")
            commodity = self.commodity_calc.compute_all(df)
            features['commodity'] = commodity
            self.feature_store.save_features(commodity, symbol, 'commodity')
        
        combined = pd.concat(features.values(), axis=1)
        logger.success("Generated {} total features for {}", combined.shape[1], symbol)
        
        return combined
    
    def process_all_symbols(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Process features for all configured symbols.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        symbols = self.config.get('symbols', [])
        
        if not symbols:
            logger.warning("No symbols configured")
            return {}
        
        logger.info("Processing {} symbols", len(symbols))
        
        results = {}
        for symbol in symbols:
            try:
                features = self.process_symbol(symbol, start_date, end_date)
                if not features.empty:
                    results[symbol] = features
                logger.success("Successfully processed {}", symbol)
            except Exception as e:
                logger.error("Failed to process {}: {}", symbol, str(e))
        
        logger.info("Computing cross-asset features...")
        try:
            asset_data = {
                s: self.storage.load_timeseries(s, start_date, end_date) 
                for s in symbols
            }
            asset_data = {k: v for k, v in asset_data.items() if not v.empty}
            
            if asset_data:
                cross_asset = CrossAssetFeatures(asset_data)
                cross_features = cross_asset.compute_all()
                
                for symbol, features in cross_features.items():
                    self.feature_store.save_features(features, symbol, 'cross_asset')
                    logger.success("Saved cross-asset features for {}", symbol)
        
        except Exception as e:
            logger.error("Failed to compute cross-asset features: {}", str(e))
        
        return results
    
    def generate_feature_report(self, output_path: str = 'data/features/reports') -> str:
        """
        Generate feature quality and importance report.
        
        Args:
            output_path: Directory for report output
            
        Returns:
            Path to generated report
        """
        report_dir = Path(output_path)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'feature_report_{timestamp}.txt'
        
        summary = self.feature_store.get_storage_summary()
        
        report_content = f"""
Feature Engineering Report
Generated: {timestamp}

Storage Summary:
- Total Files: {summary['total_files']}
- Total Size: {summary['total_size_mb']:.2f} MB
- Unique Symbols: {summary['unique_symbols']}
- Symbols: {', '.join(summary['symbols'])}

Storage Path: {summary['storage_path']}
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.success("Feature report generated: {}", report_file)
        return str(report_file)
    
    def validate_features(self, symbol: str) -> Dict[str, Any]:
        """
        Validate computed features for quality checks.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating features for {}", symbol)
        
        try:
            features = self.feature_store.load_features(symbol)
            
            missing_pct = features.isnull().sum() / len(features)
            max_missing = self.config.get('quality_checks', {}).get('max_missing_pct', 0.05)
            
            validation = {
                'symbol': symbol,
                'total_features': features.shape[1],
                'total_observations': features.shape[0],
                'missing_percentage': missing_pct.to_dict(),
                'features_exceeding_missing_threshold': list(missing_pct[missing_pct > max_missing].index),
                'valid': len(missing_pct[missing_pct > max_missing]) == 0
            }
            
            if validation['valid']:
                logger.success("Feature validation passed for {}", symbol)
            else:
                logger.warning("Feature validation failed for {}: {} features exceed missing threshold", 
                             symbol, len(validation['features_exceeding_missing_threshold']))
            
            return validation
        
        except Exception as e:
            logger.error("Feature validation failed for {}: {}", symbol, str(e))
            return {'symbol': symbol, 'valid': False, 'error': str(e)}

"""
Feature Engineering Module for Personal Quant Desk.

This module provides comprehensive feature engineering capabilities for quantitative trading,
implementing techniques from:
- Jansen's Machine Learning for Algorithmic Trading
- López de Prado's Advances in Financial Machine Learning
- Chan's Algorithmic Trading

Features:
- Base price/volume transformations
- Technical indicators
- Market microstructure features
- Regime detection
- Cross-asset relationships
- Commodity-specific features
"""

from .base_features import BaseFeatures
from .technical_features import TechnicalFeatures
from .microstructure import MicrostructureFeatures
from .regime_features import RegimeFeatures
from .cross_asset import CrossAssetFeatures
from .commodity_specific import CommodityFeatures
from .feature_store import FeatureStore
from .feature_pipeline import FeaturePipeline

__version__ = '1.0.0'

__all__ = [
    'BaseFeatures',
    'TechnicalFeatures',
    'MicrostructureFeatures',
    'RegimeFeatures',
    'CrossAssetFeatures',
    'CommodityFeatures',
    'FeatureStore',
    'FeaturePipeline',
]

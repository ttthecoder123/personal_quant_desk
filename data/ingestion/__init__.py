"""
Data Ingestion Package

This package provides comprehensive data ingestion capabilities with:
- Alpha Vantage integration for high-quality recent data
- Yahoo Finance fallback for historical data
- Hybrid data management with intelligent source selection
- Corporate action detection
- Quality scoring and validation
- Efficient caching and rate limiting
"""

from .downloader import MarketDataDownloader, HybridDataManager, DownloadResult
from .validator import DataValidator, QualityMetrics, CorporateAction
from .storage import ParquetStorage, StorageMetadata
from .catalog import DataCatalog, DatasetEntry
from .quality_scorer import QualityScorer, QualityResult
from .alpha_vantage import AlphaVantageAdapter, RateLimiter, AlphaVantageCache

__all__ = [
    'MarketDataDownloader',
    'HybridDataManager',
    'DownloadResult',
    'DataValidator',
    'QualityMetrics',
    'CorporateAction',
    'ParquetStorage',
    'StorageMetadata',
    'DataCatalog',
    'DatasetEntry',
    'QualityScorer',
    'QualityResult',
    'AlphaVantageAdapter',
    'RateLimiter',
    'AlphaVantageCache'
]

__version__ = "1.0.0"
"""
Main Data Pipeline orchestrator implementing McKinney's best practices.
Coordinates download, validation, storage, and cataloging of financial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import asyncio
import yaml
import click
from loguru import logger
import warnings
from dataclasses import dataclass, asdict
import time

# Import our modules
from ingestion.downloader import MarketDataDownloader, HybridDataManager, DownloadResult
from ingestion.validator import DataValidator, QualityMetrics, validate_single_instrument
from ingestion.storage import ParquetStorage, StorageMetadata
from ingestion.catalog import DataCatalog, register_symbol_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    symbol: str
    success: bool
    download_result: Optional[DownloadResult]
    quality_metrics: Optional[QualityMetrics]
    storage_metadata: Optional[StorageMetadata]
    catalog_id: Optional[str]
    processing_time: float
    error_message: Optional[str]


class DataPipeline:
    """
    Professional data pipeline implementing McKinney's best practices for data processing.

    Features:
    - Batch processing with parallel execution
    - Comprehensive error handling and recovery
    - Quality monitoring and alerting
    - Incremental updates and historical backlogs
    - Memory-efficient operations with chunking
    - Complete data lineage tracking
    """

    def __init__(self, config_path: str = "config/config.yaml", use_hybrid: bool = True):
        """Initialize data pipeline with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.use_hybrid = use_hybrid

        # Initialize components
        if self.use_hybrid:
            # Use HybridDataManager for intelligent source selection
            self.downloader = HybridDataManager(str(self.config_path))
        else:
            # Fallback to basic MarketDataDownloader
            self.downloader = MarketDataDownloader(str(self.config_path))

        self.validator = DataValidator()
        self.storage = ParquetStorage()
        self.catalog = DataCatalog()

        # Pipeline statistics
        self.stats = {
            'total_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'total_runtime': 0,
            'last_run': None,
            'quality_scores': []
        }

        # Get instrument list
        self.instruments = self._extract_instruments()

        logger.info("DataPipeline initialized with {} instruments", len(self.instruments))

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _extract_instruments(self) -> List[str]:
        """Extract all instrument symbols from configuration."""
        instruments = []
        for category in self.config['instruments'].values():
            for instrument_config in category.values():
                instruments.append(instrument_config['symbol'])

        return instruments

    async def run_historical_download(
        self,
        lookback_years: int = 5,
        symbols: Optional[List[str]] = None,
        force_update: bool = False
    ) -> Dict[str, PipelineResult]:
        """
        Run historical data download for specified lookback period.

        Args:
            lookback_years: Number of years to download
            symbols: Specific symbols to process (None for all)
            force_update: Force re-download even if data exists

        Returns:
            Dictionary mapping symbols to PipelineResult objects
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')

        logger.info("Starting historical download: {} years ({} to {})", lookback_years, start_date, end_date)

        symbols_to_process = symbols or self.instruments

        return await self._run_pipeline_batch(
            symbols_to_process,
            start_date,
            end_date,
            force_update=force_update,
            pipeline_type="historical"
        )

    async def run_daily_update(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 5
    ) -> Dict[str, PipelineResult]:
        """
        Run incremental daily update for recent data.

        Args:
            symbols: Specific symbols to update (None for all)
            days_back: Number of days to look back for updates

        Returns:
            Dictionary mapping symbols to PipelineResult objects
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        logger.info("Starting daily update: {} days back ({} to {})", days_back, start_date, end_date)

        symbols_to_process = symbols or self.instruments

        return await self._run_pipeline_batch(
            symbols_to_process,
            start_date,
            end_date,
            force_update=False,
            pipeline_type="daily_update"
        )

    async def _run_pipeline_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_update: bool = False,
        pipeline_type: str = "batch"
    ) -> Dict[str, PipelineResult]:
        """
        Run pipeline for a batch of symbols with parallel processing.

        Args:
            symbols: List of symbols to process
            start_date: Start date string
            end_date: End date string
            force_update: Force processing even if data exists
            pipeline_type: Type of pipeline run for logging

        Returns:
            Dictionary of results
        """
        pipeline_start = time.time()
        self.stats['last_run'] = datetime.now(pytz.UTC)

        logger.info("Processing {} symbols in {} mode", len(symbols), pipeline_type)

        # Process symbols in smaller batches to manage memory
        batch_size = self.config.get('pipeline_config', {}).get('download', {}).get('batch_size', 5)
        results = {}

        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            logger.info("Processing batch {}/{}: {}", i // batch_size + 1,
                       (len(symbols) + batch_size - 1) // batch_size, batch_symbols)

            # Process batch
            batch_results = await self._process_symbol_batch(
                batch_symbols, start_date, end_date, force_update
            )
            results.update(batch_results)

            # Brief pause between batches to respect rate limits
            if i + batch_size < len(symbols):
                await asyncio.sleep(2)

        # Update pipeline statistics
        self._update_pipeline_stats(results, time.time() - pipeline_start)

        # Generate summary
        self._log_pipeline_summary(results, pipeline_type)

        return results

    async def _process_symbol_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_update: bool
    ) -> Dict[str, PipelineResult]:
        """Process a batch of symbols through the complete pipeline."""
        batch_results = {}

        # Step 1: Download data for all symbols in batch
        logger.info("Downloading data for batch: {}", symbols)
        if self.use_hybrid:
            # Use HybridDataManager batch download
            download_results = {}
            for symbol in symbols:
                try:
                    data, metadata = self.downloader.download_instrument(symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        download_results[symbol] = DownloadResult(
                            symbol=symbol,
                            success=True,
                            data=data,
                            rows_downloaded=len(data),
                            start_date=start_date,
                            end_date=end_date,
                            source='hybrid',
                            processing_time=metadata.get('processing_time', 0),
                            error_message=None,
                            metadata=metadata
                        )
                    else:
                        download_results[symbol] = DownloadResult(
                            symbol=symbol,
                            success=False,
                            data=None,
                            rows_downloaded=0,
                            start_date=start_date,
                            end_date=end_date,
                            source='hybrid',
                            processing_time=0,
                            error_message="No data returned",
                            metadata=metadata
                        )
                except Exception as e:
                    download_results[symbol] = DownloadResult(
                        symbol=symbol,
                        success=False,
                        data=None,
                        rows_downloaded=0,
                        start_date=start_date,
                        end_date=end_date,
                        source='hybrid',
                        processing_time=0,
                        error_message=str(e),
                        metadata={}
                    )
        else:
            # Use traditional batch download
            download_results = self.downloader.download_batch_efficient(symbols, start_date, end_date)

        # Step 2: Process each symbol through validation, storage, and cataloging
        for symbol in symbols:
            symbol_start = time.time()

            try:
                download_result = download_results.get(symbol)
                if not download_result or not download_result.success:
                    batch_results[symbol] = PipelineResult(
                        symbol=symbol,
                        success=False,
                        download_result=download_result,
                        quality_metrics=None,
                        storage_metadata=None,
                        catalog_id=None,
                        processing_time=time.time() - symbol_start,
                        error_message=download_result.error_message if download_result else "Download failed"
                    )
                    continue

                # Validate data
                cleaned_data, quality_metrics = self.validator.validate_ohlcv(
                    download_result.data, symbol
                )

                # Check if quality meets minimum threshold
                min_quality = 0.7  # 70% minimum quality
                if quality_metrics.quality_score < min_quality:
                    logger.warning("Quality too low for {}: {:.2f}", symbol, quality_metrics.quality_score)

                # Store data
                storage_metadata = self.storage.save_timeseries(
                    cleaned_data,
                    symbol,
                    asset_class=self._get_asset_class(symbol),
                    overwrite=force_update
                )

                # Update catalog
                catalog_metadata = self._prepare_catalog_metadata(
                    symbol, storage_metadata, quality_metrics, download_result
                )
                catalog_id = self.catalog.register_dataset(symbol, catalog_metadata, asdict(quality_metrics))

                # Log access
                self.catalog.log_access(symbol, "write", f"pipeline_{datetime.now().strftime('%Y%m%d')}")

                # Create successful result
                batch_results[symbol] = PipelineResult(
                    symbol=symbol,
                    success=True,
                    download_result=download_result,
                    quality_metrics=quality_metrics,
                    storage_metadata=storage_metadata,
                    catalog_id=catalog_id,
                    processing_time=time.time() - symbol_start,
                    error_message=None
                )

                logger.success("Successfully processed {} (quality: {:.2f})",
                              symbol, quality_metrics.quality_score)

            except Exception as e:
                error_msg = f"Pipeline error for {symbol}: {str(e)}"
                logger.error(error_msg)

                batch_results[symbol] = PipelineResult(
                    symbol=symbol,
                    success=False,
                    download_result=download_results.get(symbol),
                    quality_metrics=None,
                    storage_metadata=None,
                    catalog_id=None,
                    processing_time=time.time() - symbol_start,
                    error_message=error_msg
                )

        return batch_results

    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol."""
        for category, instruments in self.config['instruments'].items():
            for instrument_config in instruments.values():
                if instrument_config['symbol'] == symbol:
                    return instrument_config.get('asset_class', category)
        return 'unknown'

    def _prepare_catalog_metadata(
        self,
        symbol: str,
        storage_metadata: StorageMetadata,
        quality_metrics: QualityMetrics,
        download_result: DownloadResult
    ) -> Dict[str, Any]:
        """Prepare metadata for catalog registration."""
        # Get instrument configuration
        instrument_config = self._get_instrument_config(symbol)

        # Determine data source and lineage from download result
        data_source = download_result.source if hasattr(download_result, 'source') else 'yfinance'
        data_lineage = []

        if data_source == 'hybrid':
            # Extract actual sources from metadata
            if download_result.metadata and 'sources_used' in download_result.metadata:
                sources = download_result.metadata['sources_used']
                data_lineage = [f"{source}_api" for source in sources]
                # Use primary source for display
                if sources:
                    data_source = sources[0]
            else:
                data_lineage = ['hybrid_strategy']
        else:
            data_lineage = [f"{data_source}_api"]

        # Get corporate actions info if available
        processing_steps = ['download', 'validate', 'clean', 'store']
        if hasattr(quality_metrics, 'corporate_actions') and quality_metrics.corporate_actions:
            processing_steps.append('corporate_action_detection')

        return {
            'asset_class': instrument_config.get('asset_class', 'unknown'),
            'exchange': instrument_config.get('exchange', 'unknown'),
            'currency': instrument_config.get('currency', 'USD'),
            'data_source': data_source,
            'start_date': storage_metadata.date_range[0],
            'end_date': storage_metadata.date_range[1],
            'total_rows': storage_metadata.rows_stored,
            'columns': storage_metadata.columns_stored,
            'frequency': 'daily',
            'storage_path': storage_metadata.file_path,
            'file_size_mb': storage_metadata.file_size_mb,
            'compression_type': 'snappy',
            'partition_strategy': 'year',
            'update_frequency': 'daily',
            'data_lineage': data_lineage,
            'processing_steps': processing_steps,
            'tags': [instrument_config.get('asset_class', 'unknown'), 'daily', 'ohlcv'],
            'description': f"Daily OHLCV data for {symbol}",
            'contact_info': 'data_pipeline',
            'checksum': '',  # Could calculate MD5 hash
            'schema_version': '1.0',
            'hybrid_metadata': download_result.metadata if hasattr(download_result, 'metadata') else {},
            'corporate_actions_detected': len(quality_metrics.corporate_actions) if hasattr(quality_metrics, 'corporate_actions') else 0
        }

    def _get_instrument_config(self, symbol: str) -> Dict[str, Any]:
        """Get configuration for specific instrument."""
        for category in self.config['instruments'].values():
            for instrument_config in category.values():
                if instrument_config['symbol'] == symbol:
                    return instrument_config
        return {}

    def _update_pipeline_stats(self, results: Dict[str, PipelineResult], runtime: float) -> None:
        """Update pipeline statistics."""
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful

        self.stats['total_processed'] += len(results)
        self.stats['successful_processing'] += successful
        self.stats['failed_processing'] += failed
        self.stats['total_runtime'] += runtime

        # Collect quality scores
        quality_scores = [r.quality_metrics.quality_score for r in results.values()
                         if r.success and r.quality_metrics]
        self.stats['quality_scores'].extend(quality_scores)

    def _log_pipeline_summary(self, results: Dict[str, PipelineResult], pipeline_type: str) -> None:
        """Log comprehensive pipeline summary."""
        total = len(results)
        successful = sum(1 for r in results.values() if r.success)
        failed = total - successful

        # Calculate average quality score
        quality_scores = [r.quality_metrics.quality_score for r in results.values()
                         if r.success and r.quality_metrics]
        avg_quality = np.mean(quality_scores) if quality_scores else 0

        # Calculate average processing time
        avg_time = np.mean([r.processing_time for r in results.values()])

        logger.info(
            "{} Pipeline Summary: {}/{} successful, {} failed, {:.2f} avg quality, {:.2f}s avg time",
            pipeline_type.title(), successful, total, failed, avg_quality, avg_time
        )

        # Log failed symbols
        if failed > 0:
            failed_symbols = [symbol for symbol, result in results.items() if not result.success]
            logger.warning("Failed symbols: {}", failed_symbols)

        # Log low quality symbols
        low_quality_symbols = [
            symbol for symbol, result in results.items()
            if result.success and result.quality_metrics and result.quality_metrics.quality_score < 0.8
        ]
        if low_quality_symbols:
            logger.warning("Low quality symbols (<80%): {}", low_quality_symbols)

    def generate_quality_report(self, output_path: str = "data/quality_reports") -> str:
        """
        Generate comprehensive data quality report.

        Args:
            output_path: Directory for report output

        Returns:
            Path to generated report
        """
        report_dir = Path(output_path)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"quality_report_{timestamp}.html"

        logger.info("Generating quality report: {}", report_file)

        # Collect quality data for all symbols
        quality_data = []
        for symbol in self.instruments:
            catalog_entry = self.catalog.get_dataset_info(symbol)
            if catalog_entry:
                quality_data.append({
                    'symbol': symbol,
                    'asset_class': catalog_entry.asset_class,
                    'quality_score': catalog_entry.quality_score,
                    'quality_level': catalog_entry.quality_level,
                    'completeness_pct': catalog_entry.completeness_pct,
                    'total_rows': catalog_entry.total_rows,
                    'last_updated': catalog_entry.last_updated.strftime('%Y-%m-%d'),
                    'file_size_mb': catalog_entry.file_size_mb
                })

        if not quality_data:
            logger.warning("No quality data available for report")
            return ""

        # Create DataFrame for analysis
        df = pd.DataFrame(quality_data)

        # Generate HTML report
        html_content = self._generate_html_report(df, timestamp)

        # Save report
        with open(report_file, 'w') as f:
            f.write(html_content)

        logger.success("Quality report generated: {}", report_file)
        return str(report_file)

    def _generate_html_report(self, df: pd.DataFrame, timestamp: str) -> str:
        """Generate HTML quality report."""
        # Calculate summary statistics
        avg_quality = df['quality_score'].mean()
        total_symbols = len(df)
        excellent_count = len(df[df['quality_score'] >= 0.95])
        poor_count = len(df[df['quality_score'] < 0.7])

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 15px; background-color: #e9ecef; border-radius: 5px; }}
                .metric h3 {{ margin: 0; color: #495057; }}
                .metric .value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
                .excellent {{ background-color: #d4edda; }}
                .good {{ background-color: #d1ecf1; }}
                .fair {{ background-color: #fff3cd; }}
                .poor {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <div class="metric">
                    <h3>Total Instruments</h3>
                    <div class="value">{total_symbols}</div>
                </div>
                <div class="metric">
                    <h3>Average Quality</h3>
                    <div class="value">{avg_quality:.1%}</div>
                </div>
                <div class="metric">
                    <h3>Excellent Quality</h3>
                    <div class="value">{excellent_count}</div>
                </div>
                <div class="metric">
                    <h3>Poor Quality</h3>
                    <div class="value">{poor_count}</div>
                </div>
            </div>

            <h2>Detailed Quality Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Asset Class</th>
                        <th>Quality Score</th>
                        <th>Quality Level</th>
                        <th>Completeness</th>
                        <th>Total Rows</th>
                        <th>Last Updated</th>
                        <th>File Size (MB)</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Add table rows
        for _, row in df.iterrows():
            quality_class = ""
            if row['quality_score'] >= 0.95:
                quality_class = "excellent"
            elif row['quality_score'] >= 0.85:
                quality_class = "good"
            elif row['quality_score'] >= 0.7:
                quality_class = "fair"
            else:
                quality_class = "poor"

            html += f"""
                    <tr class="{quality_class}">
                        <td>{row['symbol']}</td>
                        <td>{row['asset_class']}</td>
                        <td>{row['quality_score']:.1%}</td>
                        <td>{row['quality_level']}</td>
                        <td>{row['completeness_pct']:.1%}</td>
                        <td>{row['total_rows']:,}</td>
                        <td>{row['last_updated']}</td>
                        <td>{row['file_size_mb']:.2f}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </body>
        </html>
        """

        return html

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        avg_quality = np.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0

        return {
            'total_processed': self.stats['total_processed'],
            'successful_processing': self.stats['successful_processing'],
            'failed_processing': self.stats['failed_processing'],
            'success_rate': self.stats['successful_processing'] / max(1, self.stats['total_processed']),
            'average_quality_score': avg_quality,
            'total_runtime': self.stats['total_runtime'],
            'avg_processing_time': self.stats['total_runtime'] / max(1, self.stats['total_processed']),
            'last_run': self.stats['last_run'].isoformat() if self.stats['last_run'] else None,
            'configured_instruments': len(self.instruments)
        }


# CLI Interface
@click.group()
def cli():
    """Data Pipeline CLI for financial market data."""
    pass


@cli.command()
@click.option('--years', default=5, help='Number of years to download')
@click.option('--symbols', help='Comma-separated list of symbols (optional)')
@click.option('--force', is_flag=True, help='Force re-download existing data')
@click.option('--no-hybrid', is_flag=True, help='Disable hybrid data strategy (use yfinance only)')
def historical(years, symbols, force, no_hybrid):
    """Download historical data."""
    pipeline = DataPipeline(use_hybrid=not no_hybrid)

    symbol_list = symbols.split(',') if symbols else None

    async def run():
        return await pipeline.run_historical_download(
            lookback_years=years,
            symbols=symbol_list,
            force_update=force
        )

    results = asyncio.run(run())

    # Print summary
    successful = sum(1 for r in results.values() if r.success)
    total = len(results)
    print(f"Historical download complete: {successful}/{total} successful")


@cli.command()
@click.option('--symbols', help='Comma-separated list of symbols (optional)')
@click.option('--days', default=5, help='Number of days to look back')
@click.option('--no-hybrid', is_flag=True, help='Disable hybrid data strategy (use yfinance only)')
def update(symbols, days, no_hybrid):
    """Run daily data update."""
    pipeline = DataPipeline(use_hybrid=not no_hybrid)

    symbol_list = symbols.split(',') if symbols else None

    async def run():
        return await pipeline.run_daily_update(
            symbols=symbol_list,
            days_back=days
        )

    results = asyncio.run(run())

    # Print summary
    successful = sum(1 for r in results.values() if r.success)
    total = len(results)
    print(f"Daily update complete: {successful}/{total} successful")


@cli.command()
@click.option('--output', default='data/quality_reports', help='Output directory for report')
def report(output):
    """Generate data quality report."""
    pipeline = DataPipeline()
    report_path = pipeline.generate_quality_report(output)
    print(f"Quality report generated: {report_path}")


@cli.command()
def stats():
    """Show pipeline statistics."""
    pipeline = DataPipeline()
    stats = pipeline.get_pipeline_stats()

    print("\nPipeline Statistics:")
    print(f"Total Processed: {stats['total_processed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Average Quality: {stats['average_quality_score']:.1%}")
    print(f"Total Runtime: {stats['total_runtime']:.1f}s")
    print(f"Last Run: {stats['last_run']}")


@cli.command()
@click.option('--symbols', default='all', help='Comma-separated symbols or "all"')
@click.option('--feature-sets', default='all', help='Feature sets to compute (all/base/technical/microstructure/regime/cross_asset)')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--validate', is_flag=True, help='Validate features after computation')
def compute_features(symbols, feature_sets, start_date, end_date, validate):
    """Compute features for specified symbols."""
    from features.feature_pipeline import FeaturePipeline
    
    print("\n" + "="*60)
    print("Feature Engineering Pipeline")
    print("="*60)
    
    pipeline = FeaturePipeline()
    
    try:
        if symbols == 'all':
            print("\nComputing features for all configured symbols...")
            results = pipeline.process_all_symbols(start_date, end_date)
        else:
            symbol_list = [s.strip() for s in symbols.split(',')]
            print(f"\nComputing features for: {', '.join(symbol_list)}")
            results = {}
            for symbol in symbol_list:
                features = pipeline.process_symbol(symbol, start_date, end_date)
                if not features.empty:
                    results[symbol] = features
            
            print("\nComputing cross-asset features...")
            from features.cross_asset import CrossAssetFeatures
            try:
                asset_data = {
                    s: pipeline.storage.load_timeseries(s, start_date, end_date) 
                    for s in symbol_list
                }
                asset_data = {k: v for k, v in asset_data.items() if not v.empty}
                
                if asset_data:
                    cross_asset = CrossAssetFeatures(asset_data)
                    cross_features = cross_asset.compute_all()
                    
                    for symbol, features in cross_features.items():
                        pipeline.feature_store.save_features(features, symbol, 'cross_asset')
                        print(f"  Saved {features.shape[1]} cross-asset features for {symbol}")
            except Exception as e:
                print(f"  Warning: Failed to compute cross-asset features: {str(e)}")
        
        print("\n" + "-"*60)
        print("Feature Generation Summary:")
        print("-"*60)
        for symbol, features in results.items():
            print(f"{symbol:12} {features.shape[1]:4d} features, {features.shape[0]:6d} observations")
        
        print(f"\nTotal: {len(results)} symbols processed")
        
        if validate:
            print("\n" + "-"*60)
            print("Feature Validation:")
            print("-"*60)
            for symbol in results.keys():
                validation = pipeline.validate_features(symbol)
                status = "✓ PASS" if validation['valid'] else "✗ FAIL"
                print(f"{symbol:12} {status}")
                if not validation['valid'] and 'features_exceeding_missing_threshold' in validation:
                    print(f"  Warning: {len(validation['features_exceeding_missing_threshold'])} features exceed missing threshold")
        
        report_path = pipeline.generate_feature_report()
        print(f"\nFeature report generated: {report_path}")
        
        print("\n" + "="*60)
        print("Feature engineering complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during feature computation: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.add(
        "data/logs/pipeline_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    cli()

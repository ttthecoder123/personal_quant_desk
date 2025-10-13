"""
Data Catalog for metadata tracking and data lineage management.
Implements McKinney's best practices for data organization and discovery.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import json
from dataclasses import dataclass, asdict, field
from loguru import logger
import sqlite3
from contextlib import contextmanager
import hashlib
from enum import Enum


class DataStatus(Enum):
    """Data status enumeration."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    QUARANTINE = "quarantine"
    ARCHIVED = "archived"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"  # 95-100% quality score
    GOOD = "good"           # 85-94% quality score
    FAIR = "fair"           # 70-84% quality score
    POOR = "poor"           # <70% quality score


@dataclass
class DatasetEntry:
    """Comprehensive dataset catalog entry."""
    symbol: str
    dataset_id: str
    asset_class: str
    exchange: str
    currency: str
    data_source: str

    # Data characteristics
    start_date: str
    end_date: str
    total_rows: int
    columns: List[str]
    frequency: str  # daily, hourly, minute, etc.

    # Quality metrics
    quality_score: float
    quality_level: str
    completeness_pct: float
    outlier_count: int
    last_validation: datetime

    # Storage information
    storage_path: str
    file_size_mb: float
    compression_type: str
    partition_strategy: str

    # Lineage and provenance
    created_timestamp: datetime
    last_updated: datetime
    update_frequency: str
    data_lineage: List[str]
    processing_steps: List[str]

    # Usage tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    contact_info: str = ""
    status: str = DataStatus.ACTIVE.value

    # Data integrity
    checksum: str = ""
    schema_version: str = "1.0"


class DataCatalog:
    """
    Professional data catalog implementing McKinney's best practices for metadata management.

    Features:
    - SQLite backend for efficient metadata queries
    - Comprehensive data lineage tracking
    - Quality monitoring and alerting
    - Usage analytics and access patterns
    - Schema evolution tracking
    - Automated data discovery
    """

    def __init__(self, catalog_path: str = "data/catalog"):
        """Initialize data catalog with SQLite backend."""
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)

        # Database file
        self.db_path = self.catalog_path / "data_catalog.db"

        # Initialize database
        self._initialize_database()

        # Catalog statistics
        self.stats = {
            'total_datasets': 0,
            'total_queries': 0,
            'last_update': None
        }

        logger.info("DataCatalog initialized with database: {}", self.db_path)

    def _initialize_database(self) -> None:
        """Initialize SQLite database with proper schema."""
        with self._get_connection() as conn:
            # Main catalog table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS catalog_entries (
                    dataset_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    asset_class TEXT,
                    exchange TEXT,
                    currency TEXT,
                    data_source TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    total_rows INTEGER,
                    columns TEXT,  -- JSON array
                    frequency TEXT,
                    quality_score REAL,
                    quality_level TEXT,
                    completeness_pct REAL,
                    outlier_count INTEGER,
                    last_validation TEXT,
                    storage_path TEXT,
                    file_size_mb REAL,
                    compression_type TEXT,
                    partition_strategy TEXT,
                    created_timestamp TEXT,
                    last_updated TEXT,
                    update_frequency TEXT,
                    data_lineage TEXT,  -- JSON array
                    processing_steps TEXT,  -- JSON array
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    tags TEXT,  -- JSON array
                    description TEXT,
                    contact_info TEXT,
                    status TEXT DEFAULT 'active',
                    checksum TEXT,
                    schema_version TEXT DEFAULT '1.0'
                )
            """)

            # Quality history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT,
                    quality_score REAL,
                    validation_timestamp TEXT,
                    issues_found TEXT,  -- JSON array
                    FOREIGN KEY (dataset_id) REFERENCES catalog_entries (dataset_id)
                )
            """)

            # Access log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT,
                    access_timestamp TEXT,
                    access_type TEXT,  -- read, write, update
                    user_context TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES catalog_entries (dataset_id)
                )
            """)

            # Create indexes for efficient queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON catalog_entries (symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_class ON catalog_entries (asset_class)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON catalog_entries (quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_updated ON catalog_entries (last_updated)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    def register_dataset(
        self,
        symbol: str,
        metadata: Dict[str, Any],
        quality_metrics: Optional[Dict] = None
    ) -> str:
        """
        Register a new dataset in the catalog.

        Args:
            symbol: Instrument symbol
            metadata: Dataset metadata
            quality_metrics: Optional quality metrics

        Returns:
            Dataset ID
        """
        # Generate unique dataset ID
        dataset_id = self._generate_dataset_id(symbol, metadata)

        # Create catalog entry
        entry = self._create_catalog_entry(symbol, dataset_id, metadata, quality_metrics)

        # Save to database
        self._save_catalog_entry(entry)

        # Log registration
        logger.info("Registered dataset {} for symbol {}", dataset_id, symbol)

        return dataset_id

    def _generate_dataset_id(self, symbol: str, metadata: Dict) -> str:
        """Generate unique dataset ID based on symbol and metadata."""
        # Create hash from symbol, source, and current timestamp
        hash_input = f"{symbol}_{metadata.get('data_source', 'unknown')}_{datetime.now().isoformat()}"
        hash_object = hashlib.md5(hash_input.encode())
        return f"{symbol}_{hash_object.hexdigest()[:8]}"

    def _create_catalog_entry(
        self,
        symbol: str,
        dataset_id: str,
        metadata: Dict,
        quality_metrics: Optional[Dict]
    ) -> DatasetEntry:
        """Create DatasetEntry from metadata."""
        now = datetime.now(pytz.UTC)

        # Extract quality information
        quality_score = quality_metrics.get('quality_score', 0.0) if quality_metrics else 0.0
        quality_level = self._determine_quality_level(quality_score)

        return DatasetEntry(
            symbol=symbol,
            dataset_id=dataset_id,
            asset_class=metadata.get('asset_class', 'unknown'),
            exchange=metadata.get('exchange', 'unknown'),
            currency=metadata.get('currency', 'USD'),
            data_source=metadata.get('data_source', 'unknown'),
            start_date=metadata.get('start_date', ''),
            end_date=metadata.get('end_date', ''),
            total_rows=metadata.get('total_rows', 0),
            columns=metadata.get('columns', []),
            frequency=metadata.get('frequency', 'daily'),
            quality_score=quality_score,
            quality_level=quality_level,
            completeness_pct=quality_metrics.get('completeness_pct', 0.0) if quality_metrics else 0.0,
            outlier_count=quality_metrics.get('outlier_count', 0) if quality_metrics else 0,
            last_validation=now,
            storage_path=metadata.get('storage_path', ''),
            file_size_mb=metadata.get('file_size_mb', 0.0),
            compression_type=metadata.get('compression_type', 'snappy'),
            partition_strategy=metadata.get('partition_strategy', 'year'),
            created_timestamp=now,
            last_updated=now,
            update_frequency=metadata.get('update_frequency', 'daily'),
            data_lineage=metadata.get('data_lineage', []),
            processing_steps=metadata.get('processing_steps', []),
            tags=metadata.get('tags', []),
            description=metadata.get('description', ''),
            contact_info=metadata.get('contact_info', ''),
            checksum=metadata.get('checksum', ''),
            schema_version=metadata.get('schema_version', '1.0')
        )

    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine quality level from score."""
        if quality_score >= 0.95:
            return DataQuality.EXCELLENT.value
        elif quality_score >= 0.85:
            return DataQuality.GOOD.value
        elif quality_score >= 0.70:
            return DataQuality.FAIR.value
        else:
            return DataQuality.POOR.value

    def _save_catalog_entry(self, entry: DatasetEntry) -> None:
        """Save catalog entry to database."""
        with self._get_connection() as conn:
            # Convert entry to dict for database storage
            entry_dict = asdict(entry)

            # Convert lists and datetime objects to JSON/strings
            entry_dict['columns'] = json.dumps(entry_dict['columns'])
            entry_dict['data_lineage'] = json.dumps(entry_dict['data_lineage'])
            entry_dict['processing_steps'] = json.dumps(entry_dict['processing_steps'])
            entry_dict['tags'] = json.dumps(entry_dict['tags'])
            entry_dict['created_timestamp'] = entry_dict['created_timestamp'].isoformat()
            entry_dict['last_updated'] = entry_dict['last_updated'].isoformat()
            entry_dict['last_validation'] = entry_dict['last_validation'].isoformat()

            if entry_dict['last_accessed']:
                entry_dict['last_accessed'] = entry_dict['last_accessed'].isoformat()

            # Insert or replace entry
            columns = ', '.join(entry_dict.keys())
            placeholders = ', '.join(['?' for _ in entry_dict])

            conn.execute(
                f"INSERT OR REPLACE INTO catalog_entries ({columns}) VALUES ({placeholders})",
                list(entry_dict.values())
            )
            conn.commit()

    def update_catalog(self, symbol: str, metrics: Dict[str, Any]) -> None:
        """
        Update catalog entry with new metrics and metadata.

        Args:
            symbol: Instrument symbol
            metrics: Updated metrics dictionary
        """
        logger.info("Updating catalog for {}", symbol)

        # Find existing entry
        entry = self.get_dataset_info(symbol)
        if not entry:
            logger.warning("No existing catalog entry found for {}", symbol)
            return

        # Update quality metrics
        if 'quality_score' in metrics:
            quality_score = metrics['quality_score']
            self._log_quality_history(entry.dataset_id, quality_score, metrics.get('issues_found', []))

        # Update entry with new information
        now = datetime.now(pytz.UTC)
        updated_metadata = {
            'end_date': metrics.get('end_date', entry.end_date),
            'total_rows': metrics.get('total_rows', entry.total_rows),
            'quality_score': metrics.get('quality_score', entry.quality_score),
            'completeness_pct': metrics.get('completeness_pct', entry.completeness_pct),
            'outlier_count': metrics.get('outlier_count', entry.outlier_count),
            'file_size_mb': metrics.get('file_size_mb', entry.file_size_mb),
            'last_updated': now.isoformat(),
            'last_validation': now.isoformat(),
            'quality_level': self._determine_quality_level(metrics.get('quality_score', entry.quality_score))
        }

        # Update database
        with self._get_connection() as conn:
            update_clause = ', '.join([f"{key} = ?" for key in updated_metadata.keys()])
            values = list(updated_metadata.values()) + [entry.dataset_id]

            conn.execute(
                f"UPDATE catalog_entries SET {update_clause} WHERE dataset_id = ?",
                values
            )
            conn.commit()

        logger.success("Updated catalog entry for {}", symbol)

    def _log_quality_history(self, dataset_id: str, quality_score: float, issues_found: List[str]) -> None:
        """Log quality metrics to history table."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO quality_history (dataset_id, quality_score, validation_timestamp, issues_found)
                VALUES (?, ?, ?, ?)
            """, (dataset_id, quality_score, datetime.now(pytz.UTC).isoformat(), json.dumps(issues_found)))
            conn.commit()

    def get_dataset_info(self, symbol: str) -> Optional[DatasetEntry]:
        """
        Get dataset information for a symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            DatasetEntry or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM catalog_entries WHERE symbol = ? AND status = 'active' ORDER BY last_updated DESC LIMIT 1",
                (symbol,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Convert row back to DatasetEntry
            return self._row_to_dataset_entry(dict(row))

    def _row_to_dataset_entry(self, row_dict: Dict) -> DatasetEntry:
        """Convert database row to DatasetEntry."""
        # Convert JSON strings back to lists
        row_dict['columns'] = json.loads(row_dict['columns'])
        row_dict['data_lineage'] = json.loads(row_dict['data_lineage'])
        row_dict['processing_steps'] = json.loads(row_dict['processing_steps'])
        row_dict['tags'] = json.loads(row_dict['tags'])

        # Convert timestamp strings back to datetime
        row_dict['created_timestamp'] = datetime.fromisoformat(row_dict['created_timestamp'])
        row_dict['last_updated'] = datetime.fromisoformat(row_dict['last_updated'])
        row_dict['last_validation'] = datetime.fromisoformat(row_dict['last_validation'])

        if row_dict['last_accessed']:
            row_dict['last_accessed'] = datetime.fromisoformat(row_dict['last_accessed'])

        return DatasetEntry(**row_dict)

    def search_datasets(
        self,
        asset_class: Optional[str] = None,
        quality_threshold: Optional[float] = None,
        tags: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None
    ) -> List[DatasetEntry]:
        """
        Search datasets based on criteria.

        Args:
            asset_class: Filter by asset class
            quality_threshold: Minimum quality score
            tags: Required tags
            date_range: Date range tuple (start, end)

        Returns:
            List of matching DatasetEntry objects
        """
        query = "SELECT * FROM catalog_entries WHERE status = 'active'"
        params = []

        # Add filters
        if asset_class:
            query += " AND asset_class = ?"
            params.append(asset_class)

        if quality_threshold:
            query += " AND quality_score >= ?"
            params.append(quality_threshold)

        if date_range:
            start_date, end_date = date_range
            query += " AND start_date <= ? AND end_date >= ?"
            params.extend([end_date, start_date])

        query += " ORDER BY quality_score DESC, last_updated DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                entry = self._row_to_dataset_entry(dict(row))

                # Filter by tags if specified
                if tags and not all(tag in entry.tags for tag in tags):
                    continue

                results.append(entry)

            return results

    def log_access(self, symbol: str, access_type: str = "read", user_context: str = "") -> None:
        """
        Log dataset access for usage tracking.

        Args:
            symbol: Instrument symbol
            access_type: Type of access (read, write, update)
            user_context: User or process context
        """
        entry = self.get_dataset_info(symbol)
        if not entry:
            return

        now = datetime.now(pytz.UTC)

        with self._get_connection() as conn:
            # Log access
            conn.execute("""
                INSERT INTO access_log (dataset_id, access_timestamp, access_type, user_context)
                VALUES (?, ?, ?, ?)
            """, (entry.dataset_id, now.isoformat(), access_type, user_context))

            # Update access count and last accessed
            conn.execute("""
                UPDATE catalog_entries
                SET access_count = access_count + 1, last_accessed = ?
                WHERE dataset_id = ?
            """, (now.isoformat(), entry.dataset_id))

            conn.commit()

    def get_quality_trends(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get quality trends for a symbol over time.

        Args:
            symbol: Instrument symbol
            days: Number of days to look back

        Returns:
            DataFrame with quality trends
        """
        entry = self.get_dataset_info(symbol)
        if not entry:
            return pd.DataFrame()

        cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days)

        with self._get_connection() as conn:
            query = """
                SELECT quality_score, validation_timestamp, issues_found
                FROM quality_history
                WHERE dataset_id = ? AND validation_timestamp >= ?
                ORDER BY validation_timestamp
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=(entry.dataset_id, cutoff_date.isoformat())
            )

            if not df.empty:
                df['validation_timestamp'] = pd.to_datetime(df['validation_timestamp'])
                df = df.set_index('validation_timestamp')

            return df

    def generate_catalog_report(self) -> Dict[str, Any]:
        """Generate comprehensive catalog report."""
        with self._get_connection() as conn:
            # Basic statistics
            cursor = conn.execute("SELECT COUNT(*) as total FROM catalog_entries WHERE status = 'active'")
            total_datasets = cursor.fetchone()[0]

            # Quality distribution
            cursor = conn.execute("""
                SELECT quality_level, COUNT(*) as count
                FROM catalog_entries
                WHERE status = 'active'
                GROUP BY quality_level
            """)
            quality_dist = dict(cursor.fetchall())

            # Asset class distribution
            cursor = conn.execute("""
                SELECT asset_class, COUNT(*) as count
                FROM catalog_entries
                WHERE status = 'active'
                GROUP BY asset_class
            """)
            asset_dist = dict(cursor.fetchall())

            # Recent activity
            cursor = conn.execute("""
                SELECT symbol, last_updated, quality_score
                FROM catalog_entries
                WHERE status = 'active'
                ORDER BY last_updated DESC
                LIMIT 10
            """)
            recent_activity = [dict(row) for row in cursor.fetchall()]

            # Storage usage
            cursor = conn.execute("""
                SELECT SUM(file_size_mb) as total_size_mb, AVG(file_size_mb) as avg_size_mb
                FROM catalog_entries
                WHERE status = 'active'
            """)
            storage_stats = dict(cursor.fetchone())

            return {
                'summary': {
                    'total_datasets': total_datasets,
                    'quality_distribution': quality_dist,
                    'asset_class_distribution': asset_dist,
                    'storage_usage_mb': storage_stats['total_size_mb'] or 0,
                    'avg_dataset_size_mb': storage_stats['avg_size_mb'] or 0
                },
                'recent_activity': recent_activity,
                'generated_at': datetime.now(pytz.UTC).isoformat()
            }

    def export_catalog(self, output_path: str, format: str = "parquet") -> None:
        """
        Export catalog to file for backup or analysis.

        Args:
            output_path: Output file path
            format: Export format (parquet, csv, json)
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM catalog_entries", conn)

            if format.lower() == "parquet":
                df.to_parquet(output_path)
            elif format.lower() == "csv":
                df.to_csv(output_path, index=False)
            elif format.lower() == "json":
                df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

        logger.info("Catalog exported to {} in {} format", output_path, format)


# Convenience functions
def get_catalog() -> DataCatalog:
    """Get default catalog instance."""
    return DataCatalog()


def register_symbol_data(symbol: str, metadata: Dict, quality_metrics: Dict) -> str:
    """Convenience function to register symbol data."""
    catalog = get_catalog()
    return catalog.register_dataset(symbol, metadata, quality_metrics)
"""
Data Validator implementing McKinney's best practices for data quality and integrity.
Comprehensive validation, cleaning, and quality scoring for financial time series.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import pytz
from loguru import logger
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


@dataclass
class CorporateAction:
    """Container for detected corporate actions."""
    date: datetime
    action_type: str  # 'split', 'dividend', 'unknown'
    factor: float
    confidence: float  # 0.0 to 1.0
    price_impact: float
    volume_impact: Optional[float] = None


@dataclass
class QualityMetrics:
    """Container for data quality metrics following McKinney's structured approach."""
    symbol: str
    total_rows: int
    missing_pct: Dict[str, float]
    outlier_count: Dict[str, int]
    suspicious_jumps: int
    data_completeness: float
    price_consistency_score: float
    volume_quality_score: float
    timezone_issues: int
    quality_score: float
    validation_timestamp: datetime
    issues_found: List[str]
    recommendations: List[str]
    corporate_actions: List[CorporateAction]


class DataValidator:
    """
    Professional data validator implementing McKinney's pandas best practices.

    Features:
    - Vectorized operations for efficient validation (McKinney Ch. 4-5)
    - Statistical outlier detection using IQR and Z-score methods (McKinney Ch. 7)
    - Time series specific validation for financial data
    - Memory-efficient operations with chunking for large datasets
    - Comprehensive quality scoring and reporting
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with configuration."""
        self.config = config or self._default_config()

        # Validation thresholds
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.max_missing_pct = self.config.get('max_missing_pct', 0.05)
        self.max_daily_return = self.config.get('max_daily_return', 0.15)
        self.interpolation_limit = self.config.get('interpolation_limit', 2)

    def _default_config(self) -> Dict:
        """Default validation configuration."""
        return {
            'outlier_threshold': 3.0,
            'max_missing_pct': 0.05,
            'max_daily_return': 0.15,
            'interpolation_limit': 2,
            'min_data_points': 100,
            'outlier_detection_method': 'iqr',
            'price_jump_threshold': 0.10,
            'volume_spike_threshold': 5.0,
            # Corporate action detection parameters
            'corporate_actions': {
                'adjustment_threshold': 0.02,  # 2% change in adj_factor
                'split_threshold': 0.5,        # Factor changes >50% likely splits
                'dividend_threshold': 0.10,    # Factor changes <10% likely dividends
                'min_confidence': 0.7,         # Minimum confidence for action detection
                'lookback_window': 5           # Days to look back for confirmation
            }
        }

    def validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, QualityMetrics]:
        """
        Comprehensive OHLCV validation following McKinney's best practices.

        Args:
            df: OHLCV DataFrame with datetime index
            symbol: Instrument symbol for logging and metadata

        Returns:
            Tuple of (cleaned_df, quality_metrics)
        """
        logger.info("Validating OHLCV data for {} ({} rows)", symbol, len(df))

        if df.empty:
            return df, self._create_empty_metrics(symbol)

        original_df = df.copy()
        issues_found = []
        recommendations = []

        # 1. Basic structure validation
        df, structure_issues = self._validate_structure(df, symbol)
        issues_found.extend(structure_issues)

        # 2. Missing value analysis and handling (McKinney Ch. 7)
        df, missing_metrics = self._handle_missing_values(df, symbol)

        # 3. Outlier detection and treatment (McKinney Ch. 7)
        df, outlier_metrics = self._detect_and_handle_outliers(df, symbol)

        # 4. Price consistency validation
        df, consistency_issues = self._validate_price_consistency(df, symbol)
        issues_found.extend(consistency_issues)

        # 5. Time series specific validation
        df, ts_issues = self._validate_time_series(df, symbol)
        issues_found.extend(ts_issues)

        # 6. Volume validation (if available)
        volume_score = self._validate_volume_data(df, symbol)

        # 7. Corporate action detection
        corporate_actions = self._detect_corporate_actions(df, symbol)
        if corporate_actions:
            issues_found.append(f"Detected {len(corporate_actions)} corporate actions")
            logger.info("Detected {} corporate actions for {}", len(corporate_actions), symbol)

        # 8. Calculate overall quality metrics
        quality_metrics = self._calculate_quality_metrics(
            original_df, df, symbol, missing_metrics, outlier_metrics,
            volume_score, issues_found, recommendations, corporate_actions
        )

        logger.info("Validation complete for {}: Quality score {:.2f}",
                   symbol, quality_metrics.quality_score)

        return df, quality_metrics

    def _validate_structure(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """Validate basic DataFrame structure."""
        issues = []

        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            logger.error("Missing columns for {}: {}", symbol, missing_columns)

        # Check index type
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                issues.append("Converted index to datetime")
            except Exception as e:
                issues.append(f"Invalid datetime index: {str(e)}")

        # Check for duplicate dates
        if df.index.duplicated().any():
            duplicate_count = df.index.duplicated().sum()
            df = df[~df.index.duplicated(keep='last')]
            issues.append(f"Removed {duplicate_count} duplicate dates")
            logger.warning("Removed {} duplicate dates for {}", duplicate_count, symbol)

        # Sort by date (McKinney best practice for time series)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            issues.append("Sorted data by date")

        return df, issues

    def _handle_missing_values(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Handle missing values using McKinney's recommended approaches.

        Reference: McKinney Ch. 7 - Data Cleaning and Preparation
        """
        # Calculate missing value percentages (vectorized operation)
        missing_pct = (df.isna().sum() / len(df) * 100).to_dict()

        # Log missing value summary
        if any(pct > 0 for pct in missing_pct.values()):
            logger.info("Missing values for {}: {}", symbol,
                       {k: f"{v:.2f}%" for k, v in missing_pct.items() if v > 0})

        # Forward fill for small gaps (McKinney recommendation)
        df = df.fillna(method='ffill', limit=self.interpolation_limit)

        # Interpolate remaining gaps using time-based interpolation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate(
            method='time',
            limit=self.interpolation_limit * 2
        )

        # For remaining NaN values, use backward fill as last resort
        df = df.fillna(method='bfill', limit=1)

        # Log if significant missing data remains
        remaining_missing = df.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning("Unable to fill {} missing values for {}", remaining_missing, symbol)

        return df, missing_pct

    def _detect_and_handle_outliers(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Detect and handle outliers using statistical methods (McKinney Ch. 7).

        Implements multiple outlier detection methods:
        - IQR method (robust to extreme values)
        - Z-score method (assumes normal distribution)
        - Isolation Forest (for multivariate outliers)
        """
        outlier_counts = {}

        for column in ['Open', 'High', 'Low', 'Close']:
            if column not in df.columns:
                continue

            original_outliers = 0

            if self.config['outlier_detection_method'] == 'iqr':
                # IQR method (McKinney's recommended robust approach)
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1

                # Define outlier bounds
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR

                # Identify outliers (vectorized operation)
                outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

            elif self.config['outlier_detection_method'] == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outliers_mask = pd.Series(False, index=df.index)
                outliers_mask.loc[df[column].notna()] = z_scores > self.outlier_threshold

            else:
                # Isolation Forest for multivariate outlier detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_scores = iso_forest.fit_predict(df[[column]].fillna(0))
                outliers_mask = outlier_scores == -1

            outlier_count = outliers_mask.sum()
            outlier_counts[column] = outlier_count

            if outlier_count > 0:
                logger.warning("Found {} outliers in {} for {}", outlier_count, column, symbol)

                # Handle outliers using winsorization (cap extreme values)
                if self.config['outlier_detection_method'] == 'iqr':
                    df.loc[outliers_mask, column] = np.where(
                        df.loc[outliers_mask, column] < lower_bound,
                        lower_bound,
                        upper_bound
                    )

        return df, outlier_counts

    def _validate_price_consistency(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate OHLC price consistency using vectorized operations.

        Ensures: High >= max(Open, Close) and Low <= min(Open, Close)
        """
        issues = []

        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return df, ['Missing OHLC columns for consistency check']

        # Check High >= max(Open, Close) - vectorized operation
        high_violations = df['High'] < np.maximum(df['Open'], df['Close'])
        if high_violations.any():
            violation_count = high_violations.sum()
            issues.append(f"Fixed {violation_count} High price violations")
            df.loc[high_violations, 'High'] = np.maximum(
                df.loc[high_violations, 'Open'],
                df.loc[high_violations, 'Close']
            )

        # Check Low <= min(Open, Close) - vectorized operation
        low_violations = df['Low'] > np.minimum(df['Open'], df['Close'])
        if low_violations.any():
            violation_count = low_violations.sum()
            issues.append(f"Fixed {violation_count} Low price violations")
            df.loc[low_violations, 'Low'] = np.minimum(
                df.loc[low_violations, 'Open'],
                df.loc[low_violations, 'Close']
            )

        # Calculate price consistency score
        total_violations = high_violations.sum() + low_violations.sum()
        consistency_score = 1.0 - (total_violations / len(df))

        if total_violations > 0:
            logger.warning("Fixed {} price consistency violations for {}", total_violations, symbol)

        return df, issues

    def _validate_time_series(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """Validate time series specific properties."""
        issues = []

        # Check for reasonable data frequency
        if len(df) > 1:
            # Calculate typical interval between observations
            intervals = df.index.to_series().diff().dropna()
            median_interval = intervals.median()

            # Detect irregular intervals (more than 2x median interval)
            irregular_intervals = intervals > (2 * median_interval)
            if irregular_intervals.any():
                gap_count = irregular_intervals.sum()
                issues.append(f"Found {gap_count} irregular time intervals")

        # Check for suspicious price jumps (McKinney approach using pct_change)
        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change().dropna()
            jump_threshold = self.config.get('price_jump_threshold', 0.10)

            # Identify suspicious jumps (vectorized operation)
            suspicious_jumps = np.abs(returns) > jump_threshold
            jump_count = suspicious_jumps.sum()

            if jump_count > 0:
                issues.append(f"Found {jump_count} suspicious price jumps (>{jump_threshold:.1%})")
                logger.warning("Found {} suspicious price jumps for {}", jump_count, symbol)

        # Validate timezone consistency
        if hasattr(df.index, 'tz'):
            if df.index.tz is None:
                issues.append("Index is not timezone-aware")
            elif str(df.index.tz) != 'UTC':
                issues.append(f"Index timezone is {df.index.tz}, expected UTC")

        return df, issues

    def _validate_volume_data(self, df: pd.DataFrame, symbol: str) -> float:
        """Validate volume data quality."""
        if 'Volume' not in df.columns:
            return 1.0  # Perfect score if volume not available

        volume_series = df['Volume']

        # Check for negative volumes
        negative_volumes = (volume_series < 0).sum()
        if negative_volumes > 0:
            logger.warning("Found {} negative volume values for {}", negative_volumes, symbol)

        # Check for volume spikes (potential data errors)
        if len(volume_series) > 20:
            rolling_median = volume_series.rolling(window=20, center=True).median()
            spike_threshold = self.config.get('volume_spike_threshold', 5.0)

            # Identify volume spikes (vectorized operation)
            volume_spikes = volume_series > (spike_threshold * rolling_median)
            spike_count = volume_spikes.sum()

            if spike_count > 0:
                logger.info("Found {} volume spikes for {}", spike_count, symbol)

        # Calculate volume quality score
        zero_volume_pct = (volume_series == 0).mean()
        missing_volume_pct = volume_series.isna().mean()

        volume_score = 1.0 - (zero_volume_pct * 0.5) - (missing_volume_pct * 0.8)
        return max(0.0, min(1.0, volume_score))

    def _detect_corporate_actions(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """
        Detect corporate actions using price analysis and adjustment factor detection.

        Based on configuration parameters from data_sources.yaml:
        - adjustment_threshold: 2% change triggers detection
        - split_threshold: >50% change likely stock split
        - dividend_threshold: <10% change likely dividend
        """
        corporate_actions = []

        if len(df) < 10 or 'Close' not in df.columns:
            return corporate_actions

        ca_config = self.config.get('corporate_actions', {})
        adj_threshold = ca_config.get('adjustment_threshold', 0.02)
        split_threshold = ca_config.get('split_threshold', 0.5)
        dividend_threshold = ca_config.get('dividend_threshold', 0.10)
        min_confidence = ca_config.get('min_confidence', 0.7)
        lookback_window = ca_config.get('lookback_window', 5)

        # Calculate price returns and adjustment factors if available
        df_sorted = df.sort_index()
        returns = df_sorted['Close'].pct_change().dropna()

        # Method 1: Detect via large price movements
        # Look for overnight price gaps that suggest corporate actions
        for i in range(1, len(df_sorted)):
            current_date = df_sorted.index[i]
            prev_date = df_sorted.index[i-1]

            current_close = df_sorted.iloc[i]['Close']
            prev_close = df_sorted.iloc[i-1]['Close']

            # Skip if insufficient data
            if pd.isna(current_close) or pd.isna(prev_close):
                continue

            # Calculate overnight return
            overnight_return = (current_close / prev_close) - 1

            # Check if this looks like a corporate action
            if abs(overnight_return) > adj_threshold:
                action_factor = abs(overnight_return)
                confidence = min(1.0, action_factor / adj_threshold)

                # Classify action type based on price change magnitude
                if action_factor > split_threshold:
                    # Large change suggests stock split
                    action_type = 'split'

                    # For splits, check if price dropped by ~50%, 67%, 75% etc
                    if overnight_return < -0.4:  # Price dropped significantly
                        split_ratio = 1 / (1 + overnight_return)
                        if abs(split_ratio - round(split_ratio)) < 0.1:
                            confidence = min(1.0, confidence + 0.2)

                elif action_factor < dividend_threshold:
                    # Small change suggests dividend
                    action_type = 'dividend'

                    # For dividends, expect small price drop on ex-date
                    if overnight_return < 0:
                        confidence = min(1.0, confidence + 0.1)
                else:
                    action_type = 'unknown'

                # Validate using volume data if available
                volume_impact = None
                if 'Volume' in df_sorted.columns:
                    current_volume = df_sorted.iloc[i]['Volume']

                    # Get average volume from lookback window
                    start_idx = max(0, i - lookback_window)
                    end_idx = i
                    historical_volumes = df_sorted.iloc[start_idx:end_idx]['Volume'].dropna()

                    if len(historical_volumes) > 0 and not pd.isna(current_volume):
                        avg_volume = historical_volumes.mean()
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                        volume_impact = volume_ratio

                        # High volume increases confidence in corporate action
                        if volume_ratio > 2.0:
                            confidence = min(1.0, confidence + 0.1)

                # Only report actions above minimum confidence
                if confidence >= min_confidence:
                    corporate_action = CorporateAction(
                        date=current_date,
                        action_type=action_type,
                        factor=action_factor,
                        confidence=confidence,
                        price_impact=overnight_return,
                        volume_impact=volume_impact
                    )
                    corporate_actions.append(corporate_action)

        # Method 2: Detect via adjustment factor analysis (if Adj Close available)
        if 'Adj Close' in df_sorted.columns:
            adj_factors = df_sorted['Close'] / df_sorted['Adj Close']
            adj_factor_changes = adj_factors.pct_change().dropna()

            # Look for significant changes in adjustment factor
            for date, change in adj_factor_changes.items():
                if abs(change) > adj_threshold:
                    # Check if we already detected this action
                    existing_action = any(
                        abs((action.date - date).days) <= 1
                        for action in corporate_actions
                    )

                    if not existing_action:
                        action_factor = abs(change)
                        confidence = min(1.0, action_factor / adj_threshold)

                        # Classify based on adjustment factor change
                        if action_factor > split_threshold:
                            action_type = 'split'
                        elif action_factor < dividend_threshold:
                            action_type = 'dividend'
                        else:
                            action_type = 'unknown'

                        if confidence >= min_confidence:
                            corporate_action = CorporateAction(
                                date=date,
                                action_type=action_type,
                                factor=action_factor,
                                confidence=confidence,
                                price_impact=change,
                                volume_impact=None
                            )
                            corporate_actions.append(corporate_action)

        # Sort by date and log findings
        corporate_actions.sort(key=lambda x: x.date)

        if corporate_actions:
            logger.info(
                "Detected {} corporate actions for {}: {}",
                len(corporate_actions),
                symbol,
                [f"{ca.action_type} on {ca.date.date()} (conf={ca.confidence:.2f})"
                 for ca in corporate_actions]
            )

        return corporate_actions

    def _calculate_quality_metrics(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        symbol: str,
        missing_metrics: Dict[str, float],
        outlier_metrics: Dict[str, int],
        volume_score: float,
        issues_found: List[str],
        recommendations: List[str],
        corporate_actions: List[CorporateAction]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""

        # Data completeness score
        max_missing_pct = max(missing_metrics.values()) if missing_metrics else 0
        completeness_score = max(0, 1 - (max_missing_pct / 100))

        # Price consistency score
        total_outliers = sum(outlier_metrics.values()) if outlier_metrics else 0
        consistency_score = max(0, 1 - (total_outliers / len(original_df)) * 2)

        # Calculate overall quality score (weighted average)
        quality_score = (
            completeness_score * 0.4 +
            consistency_score * 0.3 +
            volume_score * 0.2 +
            (1.0 if len(issues_found) == 0 else max(0, 1 - len(issues_found) * 0.1)) * 0.1
        )

        # Generate recommendations based on issues
        if max_missing_pct > 5:
            recommendations.append("Consider alternative data sources for better completeness")
        if total_outliers > len(original_df) * 0.05:
            recommendations.append("Review outlier handling strategy")
        if volume_score < 0.7:
            recommendations.append("Volume data quality needs improvement")

        return QualityMetrics(
            symbol=symbol,
            total_rows=len(cleaned_df),
            missing_pct=missing_metrics,
            outlier_count=outlier_metrics,
            suspicious_jumps=len([i for i in issues_found if 'jumps' in i]),
            data_completeness=completeness_score,
            price_consistency_score=consistency_score,
            volume_quality_score=volume_score,
            timezone_issues=len([i for i in issues_found if 'timezone' in i.lower()]),
            quality_score=quality_score,
            validation_timestamp=datetime.now(pytz.UTC),
            issues_found=issues_found,
            recommendations=recommendations,
            corporate_actions=corporate_actions
        )

    def _create_empty_metrics(self, symbol: str) -> QualityMetrics:
        """Create metrics for empty dataset."""
        return QualityMetrics(
            symbol=symbol,
            total_rows=0,
            missing_pct={},
            outlier_count={},
            suspicious_jumps=0,
            data_completeness=0.0,
            price_consistency_score=0.0,
            volume_quality_score=0.0,
            timezone_issues=0,
            quality_score=0.0,
            validation_timestamp=datetime.now(pytz.UTC),
            issues_found=['Empty dataset'],
            recommendations=['Verify data source and date range'],
            corporate_actions=[]
        )

    def validate_batch(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[pd.DataFrame, QualityMetrics]]:
        """
        Validate multiple instruments in batch.

        Args:
            data_dict: Dictionary mapping symbols to DataFrames

        Returns:
            Dictionary mapping symbols to (cleaned_df, metrics) tuples
        """
        logger.info("Starting batch validation for {} instruments", len(data_dict))

        results = {}
        total_issues = 0

        for symbol, df in data_dict.items():
            try:
                cleaned_df, metrics = self.validate_ohlcv(df, symbol)
                results[symbol] = (cleaned_df, metrics)
                total_issues += len(metrics.issues_found)

            except Exception as e:
                logger.error("Validation failed for {}: {}", symbol, str(e))
                # Create error metrics
                error_metrics = self._create_empty_metrics(symbol)
                error_metrics.issues_found = [f"Validation error: {str(e)}"]
                error_metrics.quality_score = 0.0
                results[symbol] = (df, error_metrics)

        # Log batch summary
        avg_quality = np.mean([metrics.quality_score for _, metrics in results.values()])
        logger.info(
            "Batch validation complete: {} instruments, {:.2f} avg quality, {} total issues",
            len(results), avg_quality, total_issues
        )

        return results

    def generate_quality_report(
        self,
        metrics_list: List[QualityMetrics]
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not metrics_list:
            return {'error': 'No metrics provided'}

        # Calculate summary statistics
        quality_scores = [m.quality_score for m in metrics_list]

        report = {
            'summary': {
                'total_instruments': len(metrics_list),
                'avg_quality_score': np.mean(quality_scores),
                'min_quality_score': np.min(quality_scores),
                'max_quality_score': np.max(quality_scores),
                'instruments_below_threshold': sum(1 for s in quality_scores if s < 0.8)
            },
            'detailed_metrics': [asdict(m) for m in metrics_list],
            'recommendations': {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': []
            }
        }

        # Analyze common issues and generate recommendations
        all_issues = [issue for m in metrics_list for issue in m.issues_found]
        issue_counts = pd.Series(all_issues).value_counts()

        if len(issue_counts) > 0:
            most_common_issues = issue_counts.head(5).index.tolist()
            report['common_issues'] = most_common_issues

        return report


# Convenience functions
def validate_single_instrument(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, QualityMetrics]:
    """Convenience function for single instrument validation."""
    validator = DataValidator()
    return validator.validate_ohlcv(df, symbol)


def quick_quality_check(df: pd.DataFrame, symbol: str) -> float:
    """Quick quality score calculation."""
    validator = DataValidator()
    _, metrics = validator.validate_ohlcv(df, symbol)
    return metrics.quality_score
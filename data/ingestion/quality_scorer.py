"""
Composite Quality Scorer for financial data assessment.
Implements production-ready quality metrics following industry best practices.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import pytz
from scipy import stats
from sklearn.ensemble import IsolationForest
from loguru import logger
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class QualityComponents:
    """Individual quality component scores."""
    completeness: float
    consistency: float
    timeliness: float
    accuracy: float
    volume_quality: float
    corporate_actions: float


@dataclass
class QualityResult:
    """Comprehensive quality assessment result."""
    symbol: str
    composite_score: float
    quality_level: str
    is_production_ready: bool
    components: QualityComponents
    issues_found: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]
    assessment_timestamp: datetime


class QualityScorer:
    """
    Professional quality scorer for financial time series data.

    Implements a weighted composite scoring system that evaluates:
    - Data completeness (missing values, gaps)
    - OHLCV consistency (logical relationships)
    - Data timeliness (freshness, update frequency)
    - Statistical accuracy (outliers, anomalies)
    - Volume data quality (if available)
    - Corporate action detection and handling
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize quality scorer with configuration."""
        self.config = config or self._default_config()

        # Quality component weights (must sum to 1.0)
        self.weights = self.config.get('quality_weights', {
            'completeness': 0.30,
            'consistency': 0.30,
            'timeliness': 0.20,
            'accuracy': 0.15,
            'volume_quality': 0.05
        })

        # Quality thresholds
        self.thresholds = self.config.get('thresholds', {
            'production_ready': 85,
            'excellent': 95,
            'good': 85,
            'fair': 70,
            'poor': 0
        })

        # Validation parameters
        self.outlier_config = self.config.get('outlier_detection', {
            'method': 'iqr',
            'iqr_multiplier': 3.0,
            'zscore_threshold': 3.0,
            'max_outlier_pct': 0.05
        })

        logger.info("QualityScorer initialized with weighted composite scoring")

    def _default_config(self) -> Dict:
        """Default quality scoring configuration."""
        return {
            'quality_weights': {
                'completeness': 0.30,
                'consistency': 0.30,
                'timeliness': 0.20,
                'accuracy': 0.15,
                'volume_quality': 0.05
            },
            'thresholds': {
                'production_ready': 85,
                'excellent': 95,
                'good': 85,
                'fair': 70,
                'poor': 0
            },
            'outlier_detection': {
                'method': 'iqr',
                'iqr_multiplier': 3.0,
                'zscore_threshold': 3.0,
                'max_outlier_pct': 0.05
            }
        }

    def calculate_composite_score(
        self,
        df: pd.DataFrame,
        symbol: str,
        metadata: Optional[Dict] = None
    ) -> QualityResult:
        """
        Calculate comprehensive quality score for financial data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            metadata: Additional metadata for scoring

        Returns:
            QualityResult with detailed assessment
        """
        logger.info(f"Calculating quality score for {symbol} ({len(df)} rows)")

        if df.empty:
            return self._create_empty_result(symbol)

        metadata = metadata or {}
        issues_found = []
        recommendations = []

        try:
            # Calculate individual component scores
            completeness = self._score_completeness(df, symbol, issues_found)
            consistency = self._score_consistency(df, symbol, issues_found)
            timeliness = self._score_timeliness(df, symbol, metadata, issues_found)
            accuracy = self._score_accuracy(df, symbol, issues_found)
            volume_quality = self._score_volume_quality(df, symbol, issues_found)

            # Corporate action assessment (informational, not scored)
            corporate_actions = self._assess_corporate_actions(df, symbol, issues_found)

            # Create components object
            components = QualityComponents(
                completeness=completeness,
                consistency=consistency,
                timeliness=timeliness,
                accuracy=accuracy,
                volume_quality=volume_quality,
                corporate_actions=corporate_actions
            )

            # Calculate weighted composite score
            composite_score = (
                completeness * self.weights['completeness'] +
                consistency * self.weights['consistency'] +
                timeliness * self.weights['timeliness'] +
                accuracy * self.weights['accuracy'] +
                volume_quality * self.weights['volume_quality']
            )

            # Determine quality level and production readiness
            quality_level = self._determine_quality_level(composite_score)
            is_production_ready = composite_score >= self.thresholds['production_ready']

            # Generate recommendations
            recommendations = self._generate_recommendations(components, composite_score)

            # Create comprehensive metadata
            result_metadata = {
                'total_rows': len(df),
                'date_range': {
                    'start': str(df.index.min().date()) if not df.empty else None,
                    'end': str(df.index.max().date()) if not df.empty else None
                },
                'data_span_days': (df.index.max() - df.index.min()).days if len(df) > 1 else 0,
                'source_metadata': metadata,
                'scoring_weights': self.weights,
                'outlier_config': self.outlier_config
            }

            result = QualityResult(
                symbol=symbol,
                composite_score=round(composite_score, 2),
                quality_level=quality_level,
                is_production_ready=is_production_ready,
                components=components,
                issues_found=issues_found,
                recommendations=recommendations,
                metadata=result_metadata,
                assessment_timestamp=datetime.now(pytz.UTC)
            )

            logger.info(
                f"Quality assessment complete for {symbol}: {composite_score:.1f}% ({quality_level})"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating quality score for {symbol}: {str(e)}")
            return self._create_error_result(symbol, str(e))

    def _score_completeness(self, df: pd.DataFrame, symbol: str, issues: List[str]) -> float:
        """
        Score data completeness (0-100).

        Evaluates:
        - Missing value percentage
        - Data gaps and frequency consistency
        - Required columns presence
        """
        score = 100.0

        # Check missing values
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 100

        if missing_pct > 0:
            score -= missing_pct * 2  # -2 points per 1% missing
            issues.append(f"Missing data: {missing_pct:.1f}% of values")

        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            score -= len(missing_columns) * 10  # -10 points per missing column
            issues.append(f"Missing required columns: {missing_columns}")

        # Check for data gaps (weekdays only for daily data)
        if len(df) > 10:  # Need sufficient data to assess gaps
            expected_frequency = self._infer_data_frequency(df)
            if expected_frequency == 'daily':
                gaps = self._detect_data_gaps(df)
                if gaps > 5:  # Allow some gaps for holidays
                    gap_penalty = min(20, gaps)  # Max 20 point penalty
                    score -= gap_penalty
                    issues.append(f"Data gaps detected: {gaps} missing periods")

        return max(0, min(100, score))

    def _score_consistency(self, df: pd.DataFrame, symbol: str, issues: List[str]) -> float:
        """
        Score OHLCV data consistency (0-100).

        Evaluates:
        - OHLC price relationships
        - Logical constraints (prices > 0)
        - Volume consistency
        """
        score = 100.0

        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return 0.0

        # Check OHLC relationships
        ohlc_violations = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        ).sum()

        if ohlc_violations > 0:
            violation_pct = (ohlc_violations / len(df)) * 100
            score -= violation_pct * 10  # -10 points per 1% violations
            issues.append(f"OHLC violations: {ohlc_violations} records ({violation_pct:.1f}%)")

        # Check for negative or zero prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                invalid_prices = (df[col] <= 0).sum()
                if invalid_prices > 0:
                    score -= invalid_prices / len(df) * 20  # Severe penalty
                    issues.append(f"Invalid prices in {col}: {invalid_prices} records")

        # Check price change reasonableness (extreme moves)
        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change().dropna()
            extreme_moves = (np.abs(returns) > 0.5).sum()  # >50% daily moves
            if extreme_moves > 0:
                score -= extreme_moves * 2  # -2 points per extreme move
                issues.append(f"Extreme price moves: {extreme_moves} occurrences")

        # Volume consistency (if available)
        if 'Volume' in df.columns:
            negative_volume = (df['Volume'] < 0).sum()
            if negative_volume > 0:
                score -= negative_volume / len(df) * 10
                issues.append(f"Negative volume: {negative_volume} records")

        return max(0, min(100, score))

    def _score_timeliness(
        self,
        df: pd.DataFrame,
        symbol: str,
        metadata: Dict,
        issues: List[str]
    ) -> float:
        """
        Score data timeliness (0-100).

        Evaluates:
        - How recent the data is
        - Update frequency consistency
        - Data lag compared to expected updates
        """
        score = 100.0

        if df.empty:
            return 0.0

        # Check data recency
        latest_date = df.index.max()
        if pd.isna(latest_date):
            return 0.0

        # Convert to timezone-aware if needed
        if latest_date.tz is None:
            latest_date = latest_date.tz_localize('UTC')

        now = datetime.now(pytz.UTC)
        days_old = (now - latest_date).days

        # Penalize old data
        if days_old > 7:
            age_penalty = min(50, days_old * 2)  # Max 50 point penalty
            score -= age_penalty
            issues.append(f"Data is {days_old} days old")
        elif days_old > 2:
            score -= days_old * 5  # Lighter penalty for moderately old data

        # Check update frequency consistency
        if len(df) > 10:
            intervals = df.index.to_series().diff().dropna()
            if len(intervals) > 0:
                # Detect irregular intervals
                median_interval = intervals.median()
                irregular_count = (intervals > median_interval * 2).sum()

                if irregular_count > len(intervals) * 0.1:  # >10% irregular
                    score -= 15
                    issues.append(f"Irregular update frequency: {irregular_count} gaps")

        # Bonus for very recent data
        if days_old == 0:
            score = min(100, score + 5)  # Small bonus for same-day data

        return max(0, min(100, score))

    def _score_accuracy(self, df: pd.DataFrame, symbol: str, issues: List[str]) -> float:
        """
        Score statistical accuracy (0-100).

        Evaluates:
        - Outlier presence and severity
        - Statistical distribution reasonableness
        - Data smoothness and continuity
        """
        score = 100.0

        if 'Close' not in df.columns or len(df) < 10:
            return score

        try:
            # Calculate returns for outlier detection
            returns = df['Close'].pct_change().dropna()
            if len(returns) == 0:
                return score

            # Detect outliers using configured method
            outliers = self._detect_outliers(returns)
            outlier_pct = (outliers.sum() / len(returns)) * 100

            # Penalize excessive outliers
            max_outlier_pct = self.outlier_config['max_outlier_pct'] * 100
            if outlier_pct > max_outlier_pct:
                outlier_penalty = min(30, (outlier_pct - max_outlier_pct) * 3)
                score -= outlier_penalty
                issues.append(f"Outliers: {outlier_pct:.1f}% of returns (>{max_outlier_pct:.1f}% threshold)")

            # Check for data spikes (sudden jumps followed by reversals)
            spikes = self._detect_price_spikes(df['Close'])
            if spikes > 0:
                spike_penalty = min(20, spikes * 2)
                score -= spike_penalty
                issues.append(f"Price spikes detected: {spikes} occurrences")

            # Statistical distribution check
            if len(returns) > 30:
                # Check for excessive skewness or kurtosis
                skewness = abs(returns.skew())
                kurtosis = returns.kurtosis()

                if skewness > 5:  # Very skewed
                    score -= 10
                    issues.append(f"High return skewness: {skewness:.2f}")

                if kurtosis > 10:  # Very fat tails
                    score -= 10
                    issues.append(f"High return kurtosis: {kurtosis:.2f}")

        except Exception as e:
            logger.warning(f"Error in accuracy scoring for {symbol}: {str(e)}")
            score -= 20  # Penalty for calculation errors

        return max(0, min(100, score))

    def _score_volume_quality(self, df: pd.DataFrame, symbol: str, issues: List[str]) -> float:
        """
        Score volume data quality (0-100).

        Evaluates:
        - Volume data availability
        - Volume pattern reasonableness
        - Volume-price relationship consistency
        """
        if 'Volume' not in df.columns:
            return 100.0  # Perfect score if volume not expected

        score = 100.0
        volume = df['Volume']

        # Check for missing volume data
        missing_volume = volume.isna().sum()
        if missing_volume > 0:
            missing_pct = (missing_volume / len(volume)) * 100
            score -= missing_pct  # -1 point per 1% missing
            issues.append(f"Missing volume data: {missing_pct:.1f}%")

        # Check for zero volume (suspicious for many instruments)
        zero_volume = (volume == 0).sum()
        if zero_volume > len(volume) * 0.1:  # >10% zero volume suspicious
            zero_pct = (zero_volume / len(volume)) * 100
            score -= zero_pct * 0.5  # Lighter penalty
            issues.append(f"Zero volume periods: {zero_pct:.1f}%")

        # Check for volume spikes (potential data errors)
        if len(volume) > 20:
            volume_clean = volume[volume > 0]  # Exclude zeros
            if len(volume_clean) > 10:
                median_volume = volume_clean.median()
                volume_spikes = (volume > median_volume * 10).sum()

                if volume_spikes > 0:
                    spike_pct = (volume_spikes / len(volume)) * 100
                    score -= min(15, spike_pct * 2)
                    issues.append(f"Volume spikes: {volume_spikes} occurrences")

        return max(0, min(100, score))

    def _assess_corporate_actions(self, df: pd.DataFrame, symbol: str, issues: List[str]) -> float:
        """
        Assess corporate action handling (informational score).

        Evaluates:
        - Stock split detection
        - Dividend adjustment consistency
        - Price adjustment factor analysis
        """
        score = 100.0

        if 'Adj Close' not in df.columns or 'Close' not in df.columns:
            return score

        try:
            # Calculate adjustment factor
            adj_factor = df['Adj Close'] / df['Close']
            adj_factor = adj_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # Detect significant adjustment factor changes
            factor_changes = adj_factor.pct_change().abs()
            significant_changes = factor_changes[factor_changes > 0.02]  # >2% change

            if len(significant_changes) > 0:
                for date, change in significant_changes.items():
                    if change > 0.5:  # Likely stock split
                        issues.append(f"Potential stock split detected on {date.date()}")
                    else:  # Likely dividend
                        issues.append(f"Potential dividend adjustment on {date.date()}")

            # Check for gradual drift in adjustment factor (data quality issue)
            if len(adj_factor) > 100:
                factor_trend = np.polyfit(range(len(adj_factor)), adj_factor, 1)[0]
                if abs(factor_trend) > 0.001:  # Significant drift
                    score -= 20
                    issues.append("Adjustment factor drift detected (data quality issue)")

        except Exception as e:
            logger.warning(f"Error in corporate action assessment for {symbol}: {str(e)}")

        return max(0, min(100, score))

    def _detect_outliers(self, data: pd.Series) -> pd.Series:
        """Detect outliers using configured method."""
        method = self.outlier_config['method']

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = self.outlier_config['iqr_multiplier']
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            return (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            threshold = self.outlier_config['zscore_threshold']
            outliers = pd.Series(False, index=data.index)
            outliers.loc[data.notna()] = z_scores > threshold
            return outliers

        else:  # Default to IQR
            return self._detect_outliers_iqr(data)

    def _detect_price_spikes(self, prices: pd.Series, window: int = 3) -> int:
        """Detect price spikes (sudden jumps followed by reversals)."""
        if len(prices) < window * 2:
            return 0

        returns = prices.pct_change()
        spikes = 0

        for i in range(window, len(returns) - window):
            # Check for large move followed by reversal
            current_move = abs(returns.iloc[i])
            next_move = abs(returns.iloc[i + 1])

            if current_move > 0.1 and next_move > 0.05:  # 10% move + 5% reversal
                # Check if it's a reversal
                if returns.iloc[i] * returns.iloc[i + 1] < 0:  # Opposite signs
                    spikes += 1

        return spikes

    def _infer_data_frequency(self, df: pd.DataFrame) -> str:
        """Infer the expected data frequency."""
        if len(df) < 2:
            return 'unknown'

        intervals = df.index.to_series().diff().dropna()
        median_interval = intervals.median()

        if median_interval <= pd.Timedelta(minutes=5):
            return 'intraday'
        elif median_interval <= pd.Timedelta(days=1):
            return 'daily'
        elif median_interval <= pd.Timedelta(days=7):
            return 'weekly'
        else:
            return 'monthly'

    def _detect_data_gaps(self, df: pd.DataFrame) -> int:
        """Detect data gaps for daily frequency data."""
        if len(df) < 5:
            return 0

        # Create expected business day range
        start_date = df.index.min()
        end_date = df.index.max()
        expected_dates = pd.bdate_range(start=start_date, end=end_date)

        # Count missing business days
        actual_dates = df.index.normalize()
        missing_dates = expected_dates.difference(actual_dates)

        return len(missing_dates)

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level from composite score."""
        if score >= self.thresholds['excellent']:
            return 'EXCELLENT'
        elif score >= self.thresholds['good']:
            return 'GOOD'
        elif score >= self.thresholds['fair']:
            return 'FAIR'
        else:
            return 'POOR'

    def _generate_recommendations(self, components: QualityComponents, score: float) -> List[str]:
        """Generate actionable recommendations based on component scores."""
        recommendations = []

        if components.completeness < 90:
            recommendations.append("Consider alternative data sources for better completeness")

        if components.consistency < 80:
            recommendations.append("Review data validation and cleaning procedures")

        if components.timeliness < 70:
            recommendations.append("Implement more frequent data updates")

        if components.accuracy < 80:
            recommendations.append("Enhance outlier detection and handling")

        if components.volume_quality < 70:
            recommendations.append("Verify volume data source and quality")

        if score < self.thresholds['production_ready']:
            recommendations.append("Data quality below production threshold - manual review required")

        if not recommendations:
            recommendations.append("Data quality is excellent - suitable for production use")

        return recommendations

    def _create_empty_result(self, symbol: str) -> QualityResult:
        """Create result for empty dataset."""
        return QualityResult(
            symbol=symbol,
            composite_score=0.0,
            quality_level='POOR',
            is_production_ready=False,
            components=QualityComponents(0, 0, 0, 0, 0, 0),
            issues_found=['Empty dataset'],
            recommendations=['Verify data source and availability'],
            metadata={'total_rows': 0},
            assessment_timestamp=datetime.now(pytz.UTC)
        )

    def _create_error_result(self, symbol: str, error_message: str) -> QualityResult:
        """Create result for scoring error."""
        return QualityResult(
            symbol=symbol,
            composite_score=0.0,
            quality_level='POOR',
            is_production_ready=False,
            components=QualityComponents(0, 0, 0, 0, 0, 0),
            issues_found=[f'Scoring error: {error_message}'],
            recommendations=['Manual review required'],
            metadata={'error': error_message},
            assessment_timestamp=datetime.now(pytz.UTC)
        )


# Convenience functions
def quick_quality_score(df: pd.DataFrame, symbol: str) -> float:
    """Quick quality score calculation."""
    scorer = QualityScorer()
    result = scorer.calculate_composite_score(df, symbol)
    return result.composite_score


def assess_production_readiness(df: pd.DataFrame, symbol: str) -> bool:
    """Quick check if data is production ready."""
    scorer = QualityScorer()
    result = scorer.calculate_composite_score(df, symbol)
    return result.is_production_ready


def generate_quality_report(results: List[QualityResult]) -> Dict[str, Any]:
    """Generate summary report from multiple quality results."""
    if not results:
        return {'error': 'No results provided'}

    scores = [r.composite_score for r in results]

    return {
        'summary': {
            'total_symbols': len(results),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'production_ready_count': sum(1 for r in results if r.is_production_ready),
            'excellent_count': sum(1 for r in results if r.quality_level == 'EXCELLENT'),
            'good_count': sum(1 for r in results if r.quality_level == 'GOOD'),
            'fair_count': sum(1 for r in results if r.quality_level == 'FAIR'),
            'poor_count': sum(1 for r in results if r.quality_level == 'POOR')
        },
        'details': [asdict(result) for result in results],
        'generated_at': datetime.now(pytz.UTC).isoformat()
    }
"""
Model Risk Monitoring

Tracks and analyzes model performance and degradation:
- Model performance degradation over time
- Concept drift detection using statistical tests
- Prediction accuracy metrics (MAE, RMSE, hit rate)
- Feature importance stability tracking
- Model confidence scoring
- Model disagreement (ensemble variance)
- Comprehensive model risk reports with alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from scipy import stats
from loguru import logger


class ModelStatus(Enum):
    """Model status types"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class DriftType(Enum):
    """Types of concept drift"""
    NO_DRIFT = "no_drift"
    GRADUAL_DRIFT = "gradual_drift"
    SUDDEN_DRIFT = "sudden_drift"
    INCREMENTAL_DRIFT = "incremental_drift"


@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    timestamp: datetime
    model_name: str
    instrument: str
    prediction: float
    actual: Optional[float] = None
    confidence: Optional[float] = None
    features: Optional[Dict[str, float]] = None


@dataclass
class AccuracyMetrics:
    """Prediction accuracy metrics"""
    timestamp: datetime
    model_name: str
    sample_size: int
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    hit_rate: float  # Direction accuracy
    correlation: float  # Correlation with actuals
    r_squared: float
    bias: float  # Systematic over/under prediction


@dataclass
class DriftDetection:
    """Concept drift detection result"""
    timestamp: datetime
    model_name: str
    drift_type: DriftType
    drift_score: float
    p_value: float
    statistical_test: str
    is_significant: bool
    window_size: int
    description: str


@dataclass
class FeatureImportance:
    """Feature importance tracking"""
    timestamp: datetime
    model_name: str
    feature_name: str
    importance: float
    importance_change: float  # Change from previous
    rank: int
    rank_change: int


@dataclass
class ModelConfidence:
    """Model confidence metrics"""
    timestamp: datetime
    model_name: str
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    std_confidence: float
    low_confidence_pct: float  # % of predictions with confidence < threshold


@dataclass
class EnsembleDisagreement:
    """Ensemble model disagreement metrics"""
    timestamp: datetime
    instruments: List[str]
    avg_disagreement: float  # Std dev of predictions
    max_disagreement: float
    disagreement_trend: str  # 'increasing', 'stable', 'decreasing'
    high_disagreement_count: int


@dataclass
class ModelDegradation:
    """Model performance degradation tracking"""
    timestamp: datetime
    model_name: str
    current_accuracy: float
    baseline_accuracy: float
    degradation_pct: float
    degradation_rate: float  # Per day
    is_degraded: bool
    days_since_baseline: int


@dataclass
class ModelAlert:
    """Model risk alert"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    alert_type: str
    model_name: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ModelRiskReport:
    """Comprehensive model risk report"""
    timestamp: datetime
    model_name: str
    status: ModelStatus
    accuracy_metrics: AccuracyMetrics
    degradation: ModelDegradation
    drift_detection: Optional[DriftDetection]
    confidence_metrics: ModelConfidence
    feature_stability: List[FeatureImportance]
    alerts: List[ModelAlert]
    recommendations: List[str]


@dataclass
class ModelRiskConfig:
    """Configuration for model risk monitoring"""
    # Accuracy thresholds
    min_acceptable_mae: float = 0.05
    min_hit_rate: float = 0.52  # Direction accuracy
    min_correlation: float = 0.3

    # Degradation thresholds
    max_degradation_pct: float = 10.0  # 10% degradation
    warning_degradation_pct: float = 5.0

    # Drift detection
    drift_window_size: int = 100
    drift_significance_level: float = 0.05
    ks_test_threshold: float = 0.1

    # Confidence thresholds
    min_confidence: float = 0.6
    low_confidence_threshold: float = 0.5
    max_low_confidence_pct: float = 0.2  # Max 20% low confidence predictions

    # Ensemble disagreement
    max_disagreement: float = 0.15  # 15% std dev
    warning_disagreement: float = 0.10

    # Monitoring windows
    accuracy_window_days: int = 30
    drift_check_interval_days: int = 7
    feature_stability_window: int = 50


class ModelRisk:
    """
    Monitor and analyze model risk

    Features:
    - Performance degradation tracking
    - Concept drift detection
    - Accuracy monitoring
    - Feature importance stability
    - Confidence scoring
    - Ensemble disagreement
    """

    def __init__(self, config: Optional[ModelRiskConfig] = None):
        """
        Initialize model risk monitor

        Args:
            config: Model risk configuration (optional)
        """
        self.config = config if config is not None else ModelRiskConfig()

        # Prediction history per model
        self.predictions: Dict[str, deque] = {}

        # Baseline performance metrics
        self.baseline_metrics: Dict[str, AccuracyMetrics] = {}

        # Feature importance history
        self.feature_importance_history: Dict[str, List[FeatureImportance]] = {}

        # Alert tracking
        self.alerts: List[ModelAlert] = []

        # Drift detection history
        self.drift_history: Dict[str, List[DriftDetection]] = {}

        logger.info("ModelRisk monitor initialized")

    def record_prediction(
        self,
        timestamp: datetime,
        model_name: str,
        instrument: str,
        prediction: float,
        actual: Optional[float] = None,
        confidence: Optional[float] = None,
        features: Optional[Dict[str, float]] = None
    ) -> PredictionRecord:
        """
        Record a model prediction

        Args:
            timestamp: Prediction timestamp
            model_name: Name of model
            instrument: Instrument symbol
            prediction: Predicted value
            actual: Actual value (if known)
            confidence: Prediction confidence (0-1)
            features: Feature values used

        Returns:
            PredictionRecord
        """
        # Initialize model history if needed
        if model_name not in self.predictions:
            self.predictions[model_name] = deque(maxlen=1000)

        record = PredictionRecord(
            timestamp=timestamp,
            model_name=model_name,
            instrument=instrument,
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            features=features
        )

        self.predictions[model_name].append(record)

        logger.debug(f"Recorded prediction for {model_name} on {instrument}")

        return record

    def update_actual(
        self,
        model_name: str,
        timestamp: datetime,
        instrument: str,
        actual: float
    ):
        """
        Update prediction record with actual value

        Args:
            model_name: Name of model
            timestamp: Prediction timestamp
            instrument: Instrument symbol
            actual: Actual observed value
        """
        if model_name not in self.predictions:
            logger.warning(f"No predictions found for model {model_name}")
            return

        # Find matching prediction
        for record in self.predictions[model_name]:
            if (record.timestamp == timestamp and
                record.instrument == instrument and
                record.actual is None):
                record.actual = actual
                logger.debug(f"Updated actual value for {model_name} on {instrument}")
                break

    def calculate_accuracy_metrics(
        self,
        model_name: str,
        lookback_days: Optional[int] = None
    ) -> Optional[AccuracyMetrics]:
        """
        Calculate accuracy metrics for a model

        Args:
            model_name: Name of model
            lookback_days: Days to look back (optional)

        Returns:
            AccuracyMetrics or None if insufficient data
        """
        if model_name not in self.predictions:
            return None

        # Get predictions with actuals
        predictions = [
            p for p in self.predictions[model_name]
            if p.actual is not None
        ]

        # Filter by time window
        if lookback_days:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            predictions = [p for p in predictions if p.timestamp >= cutoff]

        if len(predictions) < 10:
            logger.warning(f"Insufficient data for {model_name} accuracy metrics")
            return None

        # Extract values
        y_pred = np.array([p.prediction for p in predictions])
        y_actual = np.array([p.actual for p in predictions])

        # Calculate metrics
        errors = y_actual - y_pred
        abs_errors = np.abs(errors)

        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(np.abs(errors / (y_actual + 1e-10))) * 100

        # Hit rate (direction accuracy)
        pred_direction = np.sign(y_pred)
        actual_direction = np.sign(y_actual)
        hit_rate = np.mean(pred_direction == actual_direction)

        # Correlation and R-squared
        correlation = np.corrcoef(y_pred, y_actual)[0, 1]
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        # Bias (systematic error)
        bias = np.mean(errors)

        metrics = AccuracyMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            sample_size=len(predictions),
            mae=mae,
            rmse=rmse,
            mape=mape,
            hit_rate=hit_rate,
            correlation=correlation,
            r_squared=r_squared,
            bias=bias
        )

        # Check for alerts
        self._check_accuracy_alerts(metrics)

        return metrics

    def detect_concept_drift(
        self,
        model_name: str,
        test_type: str = 'ks'
    ) -> Optional[DriftDetection]:
        """
        Detect concept drift using statistical tests

        Args:
            model_name: Name of model
            test_type: 'ks' (Kolmogorov-Smirnov) or 'anderson'

        Returns:
            DriftDetection result or None
        """
        if model_name not in self.predictions:
            return None

        predictions = [
            p for p in self.predictions[model_name]
            if p.actual is not None
        ]

        if len(predictions) < self.config.drift_window_size * 2:
            return None

        # Split into reference (old) and current (new) windows
        reference = predictions[:-self.config.drift_window_size]
        current = predictions[-self.config.drift_window_size:]

        # Extract errors
        ref_errors = np.array([p.actual - p.prediction for p in reference])
        curr_errors = np.array([p.actual - p.prediction for p in current])

        # Perform statistical test
        if test_type == 'ks':
            statistic, p_value = stats.ks_2samp(ref_errors, curr_errors)
            test_name = "Kolmogorov-Smirnov"
        else:
            # Anderson-Darling test (simplified)
            statistic, p_value = stats.ks_2samp(ref_errors, curr_errors)
            test_name = "Anderson-Darling"

        # Determine drift type
        is_significant = p_value < self.config.drift_significance_level

        if not is_significant:
            drift_type = DriftType.NO_DRIFT
        elif statistic > 0.3:
            drift_type = DriftType.SUDDEN_DRIFT
        elif statistic > 0.15:
            drift_type = DriftType.GRADUAL_DRIFT
        else:
            drift_type = DriftType.INCREMENTAL_DRIFT

        drift_detection = DriftDetection(
            timestamp=datetime.now(),
            model_name=model_name,
            drift_type=drift_type,
            drift_score=statistic,
            p_value=p_value,
            statistical_test=test_name,
            is_significant=is_significant,
            window_size=self.config.drift_window_size,
            description=f"{test_name} test: statistic={statistic:.4f}, p={p_value:.4f}"
        )

        # Store in history
        if model_name not in self.drift_history:
            self.drift_history[model_name] = []
        self.drift_history[model_name].append(drift_detection)

        # Generate alert if significant drift
        if is_significant:
            self.alerts.append(ModelAlert(
                timestamp=datetime.now(),
                severity='warning' if drift_type != DriftType.SUDDEN_DRIFT else 'critical',
                alert_type='concept_drift',
                model_name=model_name,
                message=f"Concept drift detected: {drift_type.value}",
                value=statistic,
                threshold=self.config.drift_significance_level
            ))

        return drift_detection

    def calculate_model_degradation(
        self,
        model_name: str
    ) -> Optional[ModelDegradation]:
        """
        Calculate model performance degradation

        Args:
            model_name: Name of model

        Returns:
            ModelDegradation metrics
        """
        # Get current accuracy
        current_metrics = self.calculate_accuracy_metrics(
            model_name,
            lookback_days=self.config.accuracy_window_days
        )

        if current_metrics is None:
            return None

        # Get or set baseline
        if model_name not in self.baseline_metrics:
            self.baseline_metrics[model_name] = current_metrics
            logger.info(f"Set baseline metrics for {model_name}")

        baseline = self.baseline_metrics[model_name]

        # Calculate degradation (using MAE as primary metric)
        current_accuracy = 1.0 - current_metrics.mae
        baseline_accuracy = 1.0 - baseline.mae

        degradation_pct = ((baseline_accuracy - current_accuracy) / baseline_accuracy) * 100

        # Calculate time-based degradation rate
        days_since_baseline = (current_metrics.timestamp - baseline.timestamp).days
        degradation_rate = degradation_pct / max(days_since_baseline, 1)

        # Determine if degraded
        is_degraded = degradation_pct > self.config.warning_degradation_pct

        degradation = ModelDegradation(
            timestamp=datetime.now(),
            model_name=model_name,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            degradation_pct=degradation_pct,
            degradation_rate=degradation_rate,
            is_degraded=is_degraded,
            days_since_baseline=days_since_baseline
        )

        # Check for alerts
        self._check_degradation_alerts(degradation)

        return degradation

    def calculate_confidence_metrics(
        self,
        model_name: str,
        lookback_predictions: int = 100
    ) -> Optional[ModelConfidence]:
        """
        Calculate model confidence metrics

        Args:
            model_name: Name of model
            lookback_predictions: Number of recent predictions

        Returns:
            ModelConfidence metrics
        """
        if model_name not in self.predictions:
            return None

        # Get recent predictions with confidence
        recent = list(self.predictions[model_name])[-lookback_predictions:]
        confidences = [p.confidence for p in recent if p.confidence is not None]

        if len(confidences) < 10:
            return None

        # Calculate metrics
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        std_confidence = np.std(confidences)

        # Low confidence percentage
        low_conf_count = sum(1 for c in confidences if c < self.config.low_confidence_threshold)
        low_confidence_pct = low_conf_count / len(confidences)

        metrics = ModelConfidence(
            timestamp=datetime.now(),
            model_name=model_name,
            avg_confidence=avg_confidence,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            std_confidence=std_confidence,
            low_confidence_pct=low_confidence_pct
        )

        # Check for alerts
        if low_confidence_pct > self.config.max_low_confidence_pct:
            self.alerts.append(ModelAlert(
                timestamp=datetime.now(),
                severity='warning',
                alert_type='low_confidence',
                model_name=model_name,
                message=f"High percentage of low-confidence predictions: {low_confidence_pct:.1%}",
                value=low_confidence_pct,
                threshold=self.config.max_low_confidence_pct
            ))

        return metrics

    def calculate_ensemble_disagreement(
        self,
        model_predictions: Dict[str, List[Tuple[str, float]]],
        timestamp: datetime
    ) -> EnsembleDisagreement:
        """
        Calculate disagreement between ensemble models

        Args:
            model_predictions: Dict mapping model_name to list of (instrument, prediction)
            timestamp: Timestamp

        Returns:
            EnsembleDisagreement metrics
        """
        # Organize predictions by instrument
        instrument_predictions = {}

        for model_name, predictions in model_predictions.items():
            for instrument, pred in predictions:
                if instrument not in instrument_predictions:
                    instrument_predictions[instrument] = []
                instrument_predictions[instrument].append(pred)

        # Calculate disagreement (std dev) per instrument
        disagreements = []
        high_disagreement_count = 0

        for instrument, preds in instrument_predictions.items():
            if len(preds) >= 2:
                disagreement = np.std(preds) / (np.mean(np.abs(preds)) + 1e-10)
                disagreements.append(disagreement)

                if disagreement > self.config.warning_disagreement:
                    high_disagreement_count += 1

        if not disagreements:
            return EnsembleDisagreement(
                timestamp=timestamp,
                instruments=list(instrument_predictions.keys()),
                avg_disagreement=0.0,
                max_disagreement=0.0,
                disagreement_trend='stable',
                high_disagreement_count=0
            )

        avg_disagreement = np.mean(disagreements)
        max_disagreement = np.max(disagreements)

        # Determine trend (simplified)
        disagreement_trend = 'stable'
        if avg_disagreement > self.config.warning_disagreement:
            disagreement_trend = 'increasing'

        result = EnsembleDisagreement(
            timestamp=timestamp,
            instruments=list(instrument_predictions.keys()),
            avg_disagreement=avg_disagreement,
            max_disagreement=max_disagreement,
            disagreement_trend=disagreement_trend,
            high_disagreement_count=high_disagreement_count
        )

        # Alert on high disagreement
        if avg_disagreement > self.config.max_disagreement:
            self.alerts.append(ModelAlert(
                timestamp=timestamp,
                severity='warning',
                alert_type='high_disagreement',
                model_name='ensemble',
                message=f"High ensemble disagreement: {avg_disagreement:.1%}",
                value=avg_disagreement,
                threshold=self.config.max_disagreement
            ))

        return result

    def update_feature_importance(
        self,
        model_name: str,
        feature_importances: Dict[str, float]
    ):
        """
        Update feature importance tracking

        Args:
            model_name: Name of model
            feature_importances: Dict mapping feature name to importance
        """
        if model_name not in self.feature_importance_history:
            self.feature_importance_history[model_name] = []

        timestamp = datetime.now()

        # Get previous importances
        prev_importances = {}
        if self.feature_importance_history[model_name]:
            for fi in self.feature_importance_history[model_name][-1:]:
                prev_importances[fi.feature_name] = fi

        # Sort by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Create importance records
        for rank, (feature_name, importance) in enumerate(sorted_features, 1):
            prev = prev_importances.get(feature_name)

            importance_change = 0.0
            rank_change = 0

            if prev:
                importance_change = importance - prev.importance
                rank_change = prev.rank - rank

            fi = FeatureImportance(
                timestamp=timestamp,
                model_name=model_name,
                feature_name=feature_name,
                importance=importance,
                importance_change=importance_change,
                rank=rank,
                rank_change=rank_change
            )

            self.feature_importance_history[model_name].append(fi)

    def _check_accuracy_alerts(self, metrics: AccuracyMetrics):
        """Check accuracy metrics for alerts"""
        # Low MAE alert
        if metrics.mae > self.config.min_acceptable_mae:
            self.alerts.append(ModelAlert(
                timestamp=metrics.timestamp,
                severity='warning',
                alert_type='low_accuracy',
                model_name=metrics.model_name,
                message=f"High MAE: {metrics.mae:.4f}",
                value=metrics.mae,
                threshold=self.config.min_acceptable_mae
            ))

        # Low hit rate
        if metrics.hit_rate < self.config.min_hit_rate:
            self.alerts.append(ModelAlert(
                timestamp=metrics.timestamp,
                severity='warning',
                alert_type='low_hit_rate',
                model_name=metrics.model_name,
                message=f"Low hit rate: {metrics.hit_rate:.2%}",
                value=metrics.hit_rate,
                threshold=self.config.min_hit_rate
            ))

    def _check_degradation_alerts(self, degradation: ModelDegradation):
        """Check degradation for alerts"""
        if degradation.degradation_pct > self.config.max_degradation_pct:
            self.alerts.append(ModelAlert(
                timestamp=degradation.timestamp,
                severity='critical',
                alert_type='severe_degradation',
                model_name=degradation.model_name,
                message=f"Severe model degradation: {degradation.degradation_pct:.1f}%",
                value=degradation.degradation_pct,
                threshold=self.config.max_degradation_pct
            ))
        elif degradation.degradation_pct > self.config.warning_degradation_pct:
            self.alerts.append(ModelAlert(
                timestamp=degradation.timestamp,
                severity='warning',
                alert_type='model_degradation',
                model_name=degradation.model_name,
                message=f"Model degradation: {degradation.degradation_pct:.1f}%",
                value=degradation.degradation_pct,
                threshold=self.config.warning_degradation_pct
            ))

    def generate_model_risk_report(
        self,
        model_name: str
    ) -> Optional[ModelRiskReport]:
        """
        Generate comprehensive model risk report

        Args:
            model_name: Name of model

        Returns:
            ModelRiskReport with all metrics
        """
        if model_name not in self.predictions:
            return None

        # Calculate all metrics
        accuracy = self.calculate_accuracy_metrics(model_name)
        if accuracy is None:
            return None

        degradation = self.calculate_model_degradation(model_name)
        drift = self.detect_concept_drift(model_name)
        confidence = self.calculate_confidence_metrics(model_name)

        # Get recent feature importance
        feature_stability = []
        if model_name in self.feature_importance_history:
            feature_stability = self.feature_importance_history[model_name][-10:]

        # Get recent alerts
        recent_alerts = [
            a for a in self.alerts
            if a.model_name == model_name and
            a.timestamp >= datetime.now() - timedelta(days=7)
        ]

        # Determine status
        status = ModelStatus.HEALTHY
        if degradation and degradation.degradation_pct > self.config.max_degradation_pct:
            status = ModelStatus.CRITICAL
        elif degradation and degradation.degradation_pct > self.config.warning_degradation_pct:
            status = ModelStatus.DEGRADED

        # Generate recommendations
        recommendations = self._generate_recommendations(
            accuracy, degradation, drift, confidence
        )

        return ModelRiskReport(
            timestamp=datetime.now(),
            model_name=model_name,
            status=status,
            accuracy_metrics=accuracy,
            degradation=degradation,
            drift_detection=drift,
            confidence_metrics=confidence,
            feature_stability=feature_stability,
            alerts=recent_alerts,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        accuracy: AccuracyMetrics,
        degradation: Optional[ModelDegradation],
        drift: Optional[DriftDetection],
        confidence: Optional[ModelConfidence]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if accuracy.mae > self.config.min_acceptable_mae:
            recommendations.append("Consider retraining model with recent data")

        if degradation and degradation.degradation_pct > self.config.warning_degradation_pct:
            recommendations.append("Model performance has degraded - investigate data quality and feature stability")

        if drift and drift.is_significant:
            recommendations.append(f"Concept drift detected ({drift.drift_type.value}) - retrain or adapt model")

        if confidence and confidence.low_confidence_pct > self.config.max_low_confidence_pct:
            recommendations.append("High proportion of low-confidence predictions - review model uncertainty")

        if accuracy.hit_rate < self.config.min_hit_rate:
            recommendations.append("Poor directional accuracy - review feature engineering")

        if not recommendations:
            recommendations.append("Model performance is healthy")

        return recommendations

    def get_recent_alerts(
        self,
        model_name: Optional[str] = None,
        hours: int = 24
    ) -> List[ModelAlert]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]

        return alerts

    def reset_baseline(self, model_name: str):
        """Reset baseline metrics for a model"""
        if model_name in self.baseline_metrics:
            del self.baseline_metrics[model_name]
            logger.info(f"Reset baseline for {model_name}")

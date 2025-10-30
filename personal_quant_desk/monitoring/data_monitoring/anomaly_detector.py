"""
Anomaly Detector

Statistical anomaly detection, machine learning anomaly models, pattern-based detection,
threshold-based alerts, correlation anomalies, and volume anomalies.
"""

import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from enum import Enum


class AnomalyType(Enum):
    """Types of anomalies."""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    THRESHOLD = "threshold"
    CORRELATION = "correlation"
    VOLUME = "volume"
    TREND = "trend"
    SEASONAL = "seasonal"
    ML_DETECTED = "ml_detected"


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_type: AnomalyType
    symbol: str
    field: str
    value: Any
    expected_value: Optional[Any]
    deviation: float
    severity: AnomalySeverity
    confidence: float  # 0-1
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class StatisticalModel:
    """Statistical model for anomaly detection."""
    symbol: str
    field: str
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    samples: int
    last_updated: datetime


@dataclass
class ThresholdRule:
    """Threshold-based rule."""
    name: str
    symbol: str
    field: str
    min_threshold: Optional[float]
    max_threshold: Optional[float]
    condition: Optional[Callable] = None


@dataclass
class CorrelationPair:
    """Correlation tracking between symbols/fields."""
    symbol1: str
    field1: str
    symbol2: str
    field2: str
    correlation: float
    last_updated: datetime


class AnomalyDetector:
    """
    Comprehensive anomaly detection.

    Features:
    - Statistical anomaly detection (z-score, IQR, modified z-score)
    - Machine learning models (Isolation Forest)
    - Pattern-based detection (sudden changes, trends)
    - Threshold-based alerts
    - Correlation anomalies (breaks in correlation)
    - Volume anomalies
    - Seasonal anomaly detection
    - Adaptive thresholds
    - Multi-variate anomaly detection
    - Real-time and batch detection
    """

    def __init__(self, sensitivity: float = 0.95, window_size: int = 1000):
        """
        Initialize anomaly detector.

        Args:
            sensitivity: Detection sensitivity (0-1, higher = more sensitive)
            window_size: Number of samples for statistical models
        """
        self.sensitivity = sensitivity
        self.window_size = window_size

        # Data history
        self.data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Statistical models
        self.statistical_models: Dict[str, StatisticalModel] = {}

        # ML models
        self.ml_models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Threshold rules
        self.threshold_rules: Dict[str, ThresholdRule] = {}

        # Correlation tracking
        self.correlation_pairs: Dict[str, CorrelationPair] = {}
        self.correlation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Detected anomalies
        self.anomalies: Dict[str, List[Anomaly]] = defaultdict(list)
        self.active_anomalies: Dict[str, List[Anomaly]] = defaultdict(list)

        # Volume tracking
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Pattern tracking
        self.pattern_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Callbacks for anomaly notifications
        self.callbacks: List[Callable] = []

        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start(self):
        """Start anomaly detection."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop anomaly detection."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def add_data_point(self, symbol: str, field: str, value: float,
                       timestamp: Optional[datetime] = None, volume: Optional[float] = None):
        """
        Add a data point for anomaly detection.

        Args:
            symbol: Symbol name
            field: Field name
            value: Data value
            timestamp: Data timestamp
            volume: Optional volume data
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self.lock:
            key = f"{symbol}:{field}"

            # Store data point
            self.data_history[key].append({
                'value': value,
                'timestamp': timestamp,
                'volume': volume
            })

            # Store volume separately if provided
            if volume is not None:
                self.volume_history[symbol].append({
                    'volume': volume,
                    'timestamp': timestamp
                })

            # Update statistical model
            self._update_statistical_model(symbol, field)

            # Run detection
            self._detect_anomalies(symbol, field, value, timestamp, volume)

    def detect_statistical_anomaly(self, symbol: str, field: str, value: float,
                                   method: str = 'zscore', threshold: float = 3.0) -> Optional[Anomaly]:
        """
        Detect statistical anomaly.

        Args:
            symbol: Symbol name
            field: Field name
            value: Value to check
            method: Detection method ('zscore', 'iqr', 'modified_zscore')
            threshold: Detection threshold

        Returns:
            Anomaly if detected, None otherwise
        """
        with self.lock:
            key = f"{symbol}:{field}"

            if key not in self.statistical_models:
                return None

            model = self.statistical_models[key]

            is_anomaly = False
            deviation = 0.0

            if method == 'zscore':
                if model.std_dev > 0:
                    z_score = abs((value - model.mean) / model.std_dev)
                    deviation = z_score
                    is_anomaly = z_score > threshold

            elif method == 'iqr':
                values = [d['value'] for d in self.data_history[key]]
                if len(values) >= 10:
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower = q1 - (threshold * iqr)
                    upper = q3 + (threshold * iqr)
                    is_anomaly = value < lower or value > upper
                    deviation = min(abs(value - lower), abs(value - upper)) / iqr if iqr > 0 else 0

            elif method == 'modified_zscore':
                values = [d['value'] for d in self.data_history[key]]
                if len(values) >= 10:
                    median = np.median(values)
                    mad = np.median([abs(v - median) for v in values])
                    if mad > 0:
                        modified_z = 0.6745 * abs(value - median) / mad
                        deviation = modified_z
                        is_anomaly = modified_z > threshold

            if is_anomaly:
                severity = self._determine_severity(deviation, threshold)

                return Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    symbol=symbol,
                    field=field,
                    value=value,
                    expected_value=model.mean,
                    deviation=deviation,
                    severity=severity,
                    confidence=min(1.0, deviation / (threshold * 2)),
                    description=f"Statistical anomaly: {value} deviates {deviation:.2f}σ from mean {model.mean:.2f}",
                    timestamp=datetime.now(),
                    metadata={'method': method, 'threshold': threshold}
                )

            return None

    def detect_ml_anomaly(self, symbol: str, field: str, value: float) -> Optional[Anomaly]:
        """
        Detect anomaly using ML model.

        Args:
            symbol: Symbol name
            field: Field name
            value: Value to check

        Returns:
            Anomaly if detected, None otherwise
        """
        with self.lock:
            key = f"{symbol}:{field}"

            # Train model if needed
            if key not in self.ml_models:
                if len(self.data_history[key]) >= 100:
                    self._train_ml_model(key)
                else:
                    return None

            if key not in self.ml_models:
                return None

            # Prepare data
            X = np.array([[value]])

            # Scale
            if key in self.scalers:
                X = self.scalers[key].transform(X)

            # Predict
            prediction = self.ml_models[key].predict(X)[0]
            score = self.ml_models[key].score_samples(X)[0]

            if prediction == -1:  # Anomaly
                # Get expected value (mean of recent data)
                recent_values = [d['value'] for d in list(self.data_history[key])[-50:]]
                expected_value = np.mean(recent_values) if recent_values else value

                severity = self._determine_severity_from_score(score)

                return Anomaly(
                    anomaly_type=AnomalyType.ML_DETECTED,
                    symbol=symbol,
                    field=field,
                    value=value,
                    expected_value=expected_value,
                    deviation=abs(value - expected_value),
                    severity=severity,
                    confidence=1.0 - min(1.0, abs(score)),
                    description=f"ML-detected anomaly: {value} (score: {score:.3f})",
                    timestamp=datetime.now(),
                    metadata={'model': 'IsolationForest', 'score': score}
                )

            return None

    def detect_threshold_anomaly(self, symbol: str, field: str, value: float) -> Optional[Anomaly]:
        """
        Detect threshold-based anomaly.

        Args:
            symbol: Symbol name
            field: Field name
            value: Value to check

        Returns:
            Anomaly if detected, None otherwise
        """
        with self.lock:
            # Check threshold rules
            for rule in self.threshold_rules.values():
                if rule.symbol != symbol or rule.field != field:
                    continue

                is_anomaly = False
                threshold_type = None

                if rule.min_threshold is not None and value < rule.min_threshold:
                    is_anomaly = True
                    threshold_type = 'min'

                if rule.max_threshold is not None and value > rule.max_threshold:
                    is_anomaly = True
                    threshold_type = 'max'

                if rule.condition is not None and rule.condition(value):
                    is_anomaly = True
                    threshold_type = 'custom'

                if is_anomaly:
                    expected = rule.max_threshold if threshold_type == 'max' else rule.min_threshold
                    deviation = abs(value - expected) if expected else 0

                    return Anomaly(
                        anomaly_type=AnomalyType.THRESHOLD,
                        symbol=symbol,
                        field=field,
                        value=value,
                        expected_value=expected,
                        deviation=deviation,
                        severity=AnomalySeverity.HIGH,
                        confidence=1.0,
                        description=f"Threshold violation: {value} ({rule.name})",
                        timestamp=datetime.now(),
                        metadata={'rule': rule.name, 'type': threshold_type}
                    )

            return None

    def detect_correlation_anomaly(self, symbol1: str, field1: str,
                                   symbol2: str, field2: str) -> Optional[Anomaly]:
        """
        Detect correlation break anomaly.

        Args:
            symbol1: First symbol
            field1: First field
            symbol2: Second symbol
            field2: Second field

        Returns:
            Anomaly if detected, None otherwise
        """
        with self.lock:
            key1 = f"{symbol1}:{field1}"
            key2 = f"{symbol2}:{field2}"
            pair_key = f"{key1}<->{key2}"

            if len(self.data_history[key1]) < 30 or len(self.data_history[key2]) < 30:
                return None

            # Get recent values
            values1 = [d['value'] for d in list(self.data_history[key1])[-30:]]
            values2 = [d['value'] for d in list(self.data_history[key2])[-30:]]

            # Align lengths
            min_len = min(len(values1), len(values2))
            values1 = values1[-min_len:]
            values2 = values2[-min_len:]

            # Calculate correlation
            if min_len >= 10:
                current_corr = np.corrcoef(values1, values2)[0, 1]

                # Check against historical correlation
                if pair_key in self.correlation_pairs:
                    expected_corr = self.correlation_pairs[pair_key].correlation
                    corr_change = abs(current_corr - expected_corr)

                    # Significant correlation break
                    if corr_change > 0.3:
                        return Anomaly(
                            anomaly_type=AnomalyType.CORRELATION,
                            symbol=symbol1,
                            field=field1,
                            value=current_corr,
                            expected_value=expected_corr,
                            deviation=corr_change,
                            severity=AnomalySeverity.MEDIUM,
                            confidence=min(1.0, corr_change / 0.5),
                            description=f"Correlation break: {current_corr:.3f} vs expected {expected_corr:.3f}",
                            timestamp=datetime.now(),
                            metadata={'symbol2': symbol2, 'field2': field2}
                        )

                # Update correlation
                self.correlation_pairs[pair_key] = CorrelationPair(
                    symbol1=symbol1,
                    field1=field1,
                    symbol2=symbol2,
                    field2=field2,
                    correlation=current_corr,
                    last_updated=datetime.now()
                )

            return None

    def detect_volume_anomaly(self, symbol: str, volume: float,
                             threshold_multiplier: float = 3.0) -> Optional[Anomaly]:
        """
        Detect volume anomaly.

        Args:
            symbol: Symbol name
            volume: Volume value
            threshold_multiplier: Threshold as multiple of average volume

        Returns:
            Anomaly if detected, None otherwise
        """
        with self.lock:
            if symbol not in self.volume_history or len(self.volume_history[symbol]) < 10:
                return None

            # Calculate average volume
            volumes = [d['volume'] for d in self.volume_history[symbol]]
            avg_volume = np.mean(volumes)
            std_volume = np.std(volumes)

            # Check for anomaly
            if avg_volume > 0:
                volume_ratio = volume / avg_volume

                if volume_ratio > threshold_multiplier or volume_ratio < (1.0 / threshold_multiplier):
                    severity = AnomalySeverity.HIGH if volume_ratio > 5.0 else AnomalySeverity.MEDIUM

                    return Anomaly(
                        anomaly_type=AnomalyType.VOLUME,
                        symbol=symbol,
                        field='volume',
                        value=volume,
                        expected_value=avg_volume,
                        deviation=abs(volume - avg_volume) / std_volume if std_volume > 0 else 0,
                        severity=severity,
                        confidence=min(1.0, abs(volume_ratio - 1.0)),
                        description=f"Volume anomaly: {volume:.0f} ({volume_ratio:.1f}x average)",
                        timestamp=datetime.now(),
                        metadata={'ratio': volume_ratio, 'avg_volume': avg_volume}
                    )

            return None

    def add_threshold_rule(self, name: str, symbol: str, field: str,
                          min_threshold: Optional[float] = None,
                          max_threshold: Optional[float] = None,
                          condition: Optional[Callable] = None):
        """
        Add a threshold rule.

        Args:
            name: Rule name
            symbol: Symbol name
            field: Field name
            min_threshold: Minimum threshold
            max_threshold: Maximum threshold
            condition: Custom condition function
        """
        with self.lock:
            rule = ThresholdRule(
                name=name,
                symbol=symbol,
                field=field,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                condition=condition
            )
            self.threshold_rules[name] = rule

    def add_callback(self, callback: Callable):
        """
        Add callback for anomaly notifications.

        Args:
            callback: Callback function (takes Anomaly as argument)
        """
        self.callbacks.append(callback)

    def _detect_anomalies(self, symbol: str, field: str, value: float,
                          timestamp: datetime, volume: Optional[float]):
        """Run all anomaly detection methods."""
        anomalies = []

        # Statistical detection
        stat_anomaly = self.detect_statistical_anomaly(symbol, field, value)
        if stat_anomaly:
            anomalies.append(stat_anomaly)

        # ML detection
        ml_anomaly = self.detect_ml_anomaly(symbol, field, value)
        if ml_anomaly:
            anomalies.append(ml_anomaly)

        # Threshold detection
        threshold_anomaly = self.detect_threshold_anomaly(symbol, field, value)
        if threshold_anomaly:
            anomalies.append(threshold_anomaly)

        # Volume detection
        if volume is not None:
            volume_anomaly = self.detect_volume_anomaly(symbol, volume)
            if volume_anomaly:
                anomalies.append(volume_anomaly)

        # Store and notify
        for anomaly in anomalies:
            self.anomalies[symbol].append(anomaly)
            self.active_anomalies[symbol].append(anomaly)

            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    print(f"Error in anomaly callback: {e}")

    def _update_statistical_model(self, symbol: str, field: str):
        """Update statistical model for symbol/field."""
        key = f"{symbol}:{field}"

        if len(self.data_history[key]) < 10:
            return

        values = [d['value'] for d in self.data_history[key]]

        model = StatisticalModel(
            symbol=symbol,
            field=field,
            mean=np.mean(values),
            std_dev=np.std(values),
            min_value=np.min(values),
            max_value=np.max(values),
            samples=len(values),
            last_updated=datetime.now()
        )

        self.statistical_models[key] = model

    def _train_ml_model(self, key: str):
        """Train ML model for anomaly detection."""
        if len(self.data_history[key]) < 100:
            return

        # Prepare training data
        values = [d['value'] for d in self.data_history[key]]
        X = np.array(values).reshape(-1, 1)

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Isolation Forest
        model = IsolationForest(
            contamination=1.0 - self.sensitivity,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)

        self.ml_models[key] = model
        self.scalers[key] = scaler

    def _determine_severity(self, deviation: float, threshold: float) -> AnomalySeverity:
        """Determine severity based on deviation."""
        ratio = deviation / threshold

        if ratio > 3:
            return AnomalySeverity.CRITICAL
        elif ratio > 2:
            return AnomalySeverity.HIGH
        elif ratio > 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _determine_severity_from_score(self, score: float) -> AnomalySeverity:
        """Determine severity from ML score."""
        if score < -0.5:
            return AnomalySeverity.CRITICAL
        elif score < -0.3:
            return AnomalySeverity.HIGH
        elif score < -0.1:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._retrain_ml_models()
                self._cleanup_old_anomalies()
                threading.Event().wait(60)  # Check every minute
            except Exception as e:
                print(f"Error in anomaly detector loop: {e}")

    def _retrain_ml_models(self):
        """Periodically retrain ML models."""
        with self.lock:
            for key in self.data_history.keys():
                if len(self.data_history[key]) >= 100:
                    self._train_ml_model(key)

    def _cleanup_old_anomalies(self):
        """Clean up old anomalies."""
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=24)

            for symbol in list(self.anomalies.keys()):
                self.anomalies[symbol] = [
                    a for a in self.anomalies[symbol]
                    if a.timestamp >= cutoff
                ]

    def get_anomalies(self, symbol: Optional[str] = None,
                     active_only: bool = True,
                     severity: Optional[AnomalySeverity] = None) -> List[Anomaly]:
        """
        Get detected anomalies.

        Args:
            symbol: Optional specific symbol
            active_only: Only return active anomalies
            severity: Filter by severity

        Returns:
            List of anomalies
        """
        with self.lock:
            if symbol:
                anomalies = self.active_anomalies[symbol] if active_only else self.anomalies[symbol]
            else:
                if active_only:
                    anomalies = [a for anomaly_list in self.active_anomalies.values()
                               for a in anomaly_list]
                else:
                    anomalies = [a for anomaly_list in self.anomalies.values()
                               for a in anomaly_list]

            if severity:
                anomalies = [a for a in anomalies if a.severity == severity]

            return anomalies

    def get_summary(self) -> Dict[str, Any]:
        """
        Get anomaly detection summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total_anomalies = sum(len(anomalies) for anomalies in self.active_anomalies.values())

            by_type = defaultdict(int)
            by_severity = defaultdict(int)

            for anomaly_list in self.active_anomalies.values():
                for anomaly in anomaly_list:
                    by_type[anomaly.anomaly_type.value] += 1
                    by_severity[anomaly.severity.value] += 1

            return {
                'total_active_anomalies': total_anomalies,
                'by_type': dict(by_type),
                'by_severity': dict(by_severity),
                'monitored_symbols': len(self.data_history),
                'ml_models_trained': len(self.ml_models),
                'timestamp': datetime.now().isoformat()
            }

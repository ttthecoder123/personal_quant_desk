"""
Signal Monitor

Signal generation rate, signal quality metrics, signal latency,
feature drift detection, model performance tracking, signal conflicts,
and signal staleness monitoring.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Trading signal information."""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # Signal strength (0-1)
    strategy: str
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    execution_timestamp: Optional[datetime] = None
    outcome: Optional[str] = None  # 'success', 'failure', 'timeout'
    pnl: Optional[float] = None

    @property
    def latency(self) -> Optional[timedelta]:
        """Calculate signal-to-execution latency."""
        if self.execution_timestamp:
            return self.execution_timestamp - self.timestamp
        return None

    @property
    def age(self) -> timedelta:
        """Calculate signal age."""
        return datetime.now() - self.timestamp


@dataclass
class FeatureStats:
    """Feature statistics for drift detection."""
    feature_name: str
    mean: float
    std: float
    min_value: float
    max_value: float
    timestamp: datetime
    sample_size: int


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    total_signals: int
    executed_signals: int
    successful_signals: int
    failed_signals: int
    avg_signal_strength: float
    avg_pnl: float
    win_rate: float
    sharpe_ratio: float
    timestamp: datetime


class SignalMonitor:
    """
    Comprehensive signal monitoring.

    Features:
    - Signal generation rate monitoring
    - Signal quality metrics
    - Signal latency tracking
    - Feature drift detection
    - Model performance tracking
    - Signal conflicts detection
    - Signal staleness monitoring
    - Signal-to-execution correlation
    - Signal strength distribution
    """

    def __init__(self, update_interval: int = 5, staleness_threshold: int = 300):
        """
        Initialize signal monitor.

        Args:
            update_interval: Seconds between metric updates
            staleness_threshold: Seconds before signal is considered stale
        """
        self.update_interval = update_interval
        self.staleness_threshold = staleness_threshold
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Signal tracking
        self.signals: deque = deque(maxlen=100000)
        self.active_signals: Dict[str, Signal] = {}

        # Feature tracking for drift detection
        self.feature_baselines: Dict[str, FeatureStats] = {}
        self.feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.drift_events: deque = deque(maxlen=1000)

        # Performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.performance_history: deque = deque(maxlen=10000)

        # Metrics
        self.generation_rate_history: deque = deque(maxlen=1000)
        self.quality_metrics_history: deque = deque(maxlen=1000)
        self.latency_history: deque = deque(maxlen=1000)
        self.conflict_history: deque = deque(maxlen=1000)

        # Alerts
        self.alerts: deque = deque(maxlen=1000)

        # Thresholds
        self.min_signal_strength = 0.5
        self.max_signal_latency_seconds = 60
        self.drift_threshold = 2.0  # Standard deviations

    def start(self):
        """Start signal monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop signal monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def record_signal(self, signal: Signal):
        """
        Record a trading signal.

        Args:
            signal: Signal object
        """
        with self.lock:
            self.signals.append(signal)
            self.active_signals[signal.signal_id] = signal

            # Track features
            for feature_name, feature_value in signal.features.items():
                self.feature_history[feature_name].append({
                    'timestamp': signal.timestamp,
                    'value': feature_value,
                    'signal_id': signal.signal_id
                })

            # Check for feature drift
            self._check_feature_drift(signal)

    def update_signal_execution(self, signal_id: str, execution_timestamp: datetime,
                                outcome: str, pnl: Optional[float] = None):
        """
        Update signal with execution information.

        Args:
            signal_id: Signal ID
            execution_timestamp: When signal was executed
            outcome: Execution outcome
            pnl: P&L from signal
        """
        with self.lock:
            if signal_id in self.active_signals:
                signal = self.active_signals[signal_id]
                signal.executed = True
                signal.execution_timestamp = execution_timestamp
                signal.outcome = outcome
                signal.pnl = pnl

    def set_feature_baseline(self, feature_name: str, mean: float, std: float,
                           min_value: float, max_value: float):
        """
        Set baseline statistics for a feature.

        Args:
            feature_name: Feature name
            mean: Baseline mean
            std: Baseline standard deviation
            min_value: Baseline minimum
            max_value: Baseline maximum
        """
        with self.lock:
            self.feature_baselines[feature_name] = FeatureStats(
                feature_name=feature_name,
                mean=mean,
                std=std,
                min_value=min_value,
                max_value=max_value,
                timestamp=datetime.now(),
                sample_size=0
            )

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._calculate_generation_rate()
                self._calculate_quality_metrics()
                self._calculate_latency_metrics()
                self._detect_signal_conflicts()
                self._check_signal_staleness()
                self._calculate_model_performance()

                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in signal monitor loop: {e}")

    def _calculate_generation_rate(self):
        """Calculate signal generation rate."""
        with self.lock:
            if not self.signals:
                return

            # Count signals in last minute
            cutoff_1m = datetime.now() - timedelta(minutes=1)
            cutoff_5m = datetime.now() - timedelta(minutes=5)
            cutoff_1h = datetime.now() - timedelta(hours=1)

            signals_1m = sum(1 for s in self.signals if s.timestamp >= cutoff_1m)
            signals_5m = sum(1 for s in self.signals if s.timestamp >= cutoff_5m)
            signals_1h = sum(1 for s in self.signals if s.timestamp >= cutoff_1h)

            # By strategy
            strategy_rates = defaultdict(int)
            for signal in self.signals:
                if signal.timestamp >= cutoff_1m:
                    strategy_rates[signal.strategy] += 1

            self.generation_rate_history.append({
                'timestamp': datetime.now(),
                'signals_per_minute': signals_1m,
                'signals_per_5min': signals_5m / 5,
                'signals_per_hour': signals_1h / 60,
                'by_strategy': dict(strategy_rates)
            })

    def _calculate_quality_metrics(self):
        """Calculate signal quality metrics."""
        with self.lock:
            if not self.signals:
                return

            # Get recent signals (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_signals = [s for s in self.signals if s.timestamp >= cutoff]

            if not recent_signals:
                return

            # Signal strength distribution
            strengths = [s.strength for s in recent_signals]
            avg_strength = np.mean(strengths)
            std_strength = np.std(strengths)

            # Execution rate
            executed = sum(1 for s in recent_signals if s.executed)
            execution_rate = executed / len(recent_signals) if recent_signals else 0

            # Success rate (for executed signals)
            executed_signals = [s for s in recent_signals if s.executed]
            successful = sum(1 for s in executed_signals if s.outcome == 'success')
            success_rate = successful / len(executed_signals) if executed_signals else 0

            # Average P&L
            pnls = [s.pnl for s in executed_signals if s.pnl is not None]
            avg_pnl = np.mean(pnls) if pnls else 0

            # Signal type distribution
            type_distribution = defaultdict(int)
            for signal in recent_signals:
                type_distribution[signal.signal_type] += 1

            self.quality_metrics_history.append({
                'timestamp': datetime.now(),
                'avg_signal_strength': avg_strength,
                'std_signal_strength': std_strength,
                'execution_rate': execution_rate,
                'success_rate': success_rate,
                'avg_pnl': avg_pnl,
                'total_signals': len(recent_signals),
                'type_distribution': dict(type_distribution)
            })

    def _calculate_latency_metrics(self):
        """Calculate signal latency metrics."""
        with self.lock:
            if not self.signals:
                return

            # Get executed signals from last hour
            cutoff = datetime.now() - timedelta(hours=1)
            executed_signals = [
                s for s in self.signals
                if s.timestamp >= cutoff and s.executed and s.latency
            ]

            if not executed_signals:
                return

            latencies = [s.latency.total_seconds() for s in executed_signals]
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            # By strategy
            strategy_latencies = defaultdict(list)
            for signal in executed_signals:
                strategy_latencies[signal.strategy].append(signal.latency.total_seconds())

            avg_by_strategy = {
                strategy: np.mean(latencies)
                for strategy, latencies in strategy_latencies.items()
            }

            self.latency_history.append({
                'timestamp': datetime.now(),
                'avg_latency_seconds': avg_latency,
                'p95_latency_seconds': p95_latency,
                'p99_latency_seconds': p99_latency,
                'by_strategy': avg_by_strategy
            })

            # Check for high latency
            if p95_latency > self.max_signal_latency_seconds:
                self._create_alert(
                    alert_type='high_latency',
                    severity='warning',
                    message=f"High signal latency detected: P95 = {p95_latency:.2f}s"
                )

    def _check_feature_drift(self, signal: Signal):
        """
        Check for feature drift.

        Args:
            signal: Signal to check
        """
        for feature_name, feature_value in signal.features.items():
            if feature_name not in self.feature_baselines:
                continue

            baseline = self.feature_baselines[feature_name]

            # Z-score calculation
            if baseline.std > 0:
                z_score = abs((feature_value - baseline.mean) / baseline.std)

                if z_score > self.drift_threshold:
                    self.drift_events.append({
                        'timestamp': datetime.now(),
                        'feature_name': feature_name,
                        'signal_id': signal.signal_id,
                        'current_value': feature_value,
                        'baseline_mean': baseline.mean,
                        'baseline_std': baseline.std,
                        'z_score': z_score
                    })

                    self._create_alert(
                        alert_type='feature_drift',
                        severity='warning',
                        message=f"Feature drift detected for {feature_name}: z-score = {z_score:.2f}",
                        metadata={
                            'feature': feature_name,
                            'value': feature_value,
                            'z_score': z_score
                        }
                    )

    def _detect_signal_conflicts(self):
        """Detect conflicting signals."""
        with self.lock:
            # Get active signals (recent, not yet stale)
            cutoff = datetime.now() - timedelta(seconds=self.staleness_threshold)
            active = [s for s in self.active_signals.values() if s.timestamp >= cutoff]

            # Group by symbol
            symbol_signals = defaultdict(list)
            for signal in active:
                symbol_signals[signal.symbol].append(signal)

            # Detect conflicts
            conflicts = []
            for symbol, signals in symbol_signals.items():
                if len(signals) > 1:
                    # Check for opposing signals
                    signal_types = set(s.signal_type for s in signals)
                    if 'buy' in signal_types and 'sell' in signal_types:
                        conflicts.append({
                            'symbol': symbol,
                            'conflicting_signals': [s.signal_id for s in signals],
                            'signal_types': list(signal_types)
                        })

            if conflicts:
                self.conflict_history.append({
                    'timestamp': datetime.now(),
                    'conflicts': conflicts,
                    'total_conflicts': len(conflicts)
                })

                for conflict in conflicts:
                    self._create_alert(
                        alert_type='signal_conflict',
                        severity='warning',
                        message=f"Conflicting signals for {conflict['symbol']}: {conflict['signal_types']}",
                        metadata=conflict
                    )

    def _check_signal_staleness(self):
        """Check for stale signals."""
        with self.lock:
            stale_threshold = datetime.now() - timedelta(seconds=self.staleness_threshold)
            stale_signals = []

            for signal_id, signal in list(self.active_signals.items()):
                if not signal.executed and signal.timestamp < stale_threshold:
                    stale_signals.append(signal)
                    # Remove from active signals
                    del self.active_signals[signal_id]

            if stale_signals:
                self._create_alert(
                    alert_type='stale_signals',
                    severity='info',
                    message=f"{len(stale_signals)} signals became stale without execution",
                    metadata={'stale_count': len(stale_signals)}
                )

    def _calculate_model_performance(self):
        """Calculate model/strategy performance."""
        with self.lock:
            if not self.signals:
                return

            # Get recent signals (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_signals = [s for s in self.signals if s.timestamp >= cutoff]

            # Group by strategy
            strategy_signals = defaultdict(list)
            for signal in recent_signals:
                strategy_signals[signal.strategy].append(signal)

            # Calculate performance for each strategy
            for strategy, signals in strategy_signals.items():
                executed = [s for s in signals if s.executed]
                successful = [s for s in executed if s.outcome == 'success']
                failed = [s for s in executed if s.outcome == 'failure']

                strengths = [s.strength for s in signals]
                avg_strength = np.mean(strengths) if strengths else 0

                pnls = [s.pnl for s in executed if s.pnl is not None]
                avg_pnl = np.mean(pnls) if pnls else 0

                win_rate = len(successful) / len(executed) if executed else 0

                # Calculate Sharpe ratio
                sharpe = 0.0
                if len(pnls) > 1:
                    returns = np.array(pnls)
                    if np.std(returns) > 0:
                        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

                self.model_performance[strategy] = ModelPerformance(
                    model_name=strategy,
                    total_signals=len(signals),
                    executed_signals=len(executed),
                    successful_signals=len(successful),
                    failed_signals=len(failed),
                    avg_signal_strength=avg_strength,
                    avg_pnl=avg_pnl,
                    win_rate=win_rate,
                    sharpe_ratio=sharpe,
                    timestamp=datetime.now()
                )

    def _create_alert(self, alert_type: str, severity: str, message: str,
                     metadata: Optional[Dict] = None):
        """
        Create a signal alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            metadata: Additional metadata
        """
        self.alerts.append({
            'timestamp': datetime.now(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'metadata': metadata or {}
        })

    def get_generation_rate(self) -> Dict[str, Any]:
        """
        Get signal generation rate metrics.

        Returns:
            Dictionary of generation rate metrics
        """
        with self.lock:
            if not self.generation_rate_history:
                return {}
            return self.generation_rate_history[-1]

    def get_quality_metrics(self) -> Dict[str, Any]:
        """
        Get signal quality metrics.

        Returns:
            Dictionary of quality metrics
        """
        with self.lock:
            if not self.quality_metrics_history:
                return {}
            return self.quality_metrics_history[-1]

    def get_latency_metrics(self) -> Dict[str, Any]:
        """
        Get signal latency metrics.

        Returns:
            Dictionary of latency metrics
        """
        with self.lock:
            if not self.latency_history:
                return {}
            return self.latency_history[-1]

    def get_drift_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent drift events.

        Args:
            limit: Maximum number of events

        Returns:
            List of drift events
        """
        with self.lock:
            return list(self.drift_events)[-limit:]

    def get_conflicts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent signal conflicts.

        Args:
            limit: Maximum number of conflicts

        Returns:
            List of conflicts
        """
        with self.lock:
            return list(self.conflict_history)[-limit:]

    def get_model_performance(self, strategy: Optional[str] = None) -> Dict[str, ModelPerformance]:
        """
        Get model performance metrics.

        Args:
            strategy: Optional specific strategy

        Returns:
            Dictionary of model performance
        """
        with self.lock:
            if strategy:
                return {strategy: self.model_performance.get(strategy)}
            return self.model_performance.copy()

    def get_alerts(self, alert_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            alert_type: Filter by alert type
            limit: Maximum number of alerts

        Returns:
            List of alerts
        """
        with self.lock:
            alerts = list(self.alerts)
            if alert_type:
                alerts = [a for a in alerts if a['type'] == alert_type]
            return alerts[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get signal monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            generation = self.get_generation_rate()
            quality = self.get_quality_metrics()
            latency = self.get_latency_metrics()

            return {
                'total_signals': len(self.signals),
                'active_signals': len(self.active_signals),
                'signals_per_minute': generation.get('signals_per_minute', 0),
                'avg_signal_strength': quality.get('avg_signal_strength', 0),
                'execution_rate': quality.get('execution_rate', 0),
                'success_rate': quality.get('success_rate', 0),
                'avg_latency_seconds': latency.get('avg_latency_seconds', 0),
                'drift_events': len(self.drift_events),
                'conflicts': len(self.conflict_history),
                'alerts_count': len(self.alerts),
                'total_strategies': len(self.model_performance),
                'timestamp': datetime.now().isoformat()
            }

    def get_signal_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """
        Get signals as DataFrame.

        Args:
            hours: Hours of history to include

        Returns:
            DataFrame of signals
        """
        with self.lock:
            if not self.signals:
                return pd.DataFrame()

            cutoff = datetime.now() - timedelta(hours=hours)
            recent_signals = [s for s in self.signals if s.timestamp >= cutoff]

            data = []
            for signal in recent_signals:
                data.append({
                    'timestamp': signal.timestamp,
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'strategy': signal.strategy,
                    'executed': signal.executed,
                    'outcome': signal.outcome,
                    'pnl': signal.pnl,
                    'latency_seconds': signal.latency.total_seconds() if signal.latency else None,
                    'age_seconds': signal.age.total_seconds()
                })

            return pd.DataFrame(data)

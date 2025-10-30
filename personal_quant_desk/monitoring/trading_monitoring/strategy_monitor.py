"""
Strategy Monitor

Strategy performance metrics, strategy health scores, parameter drift,
strategy correlation, regime performance, strategy capacity,
and strategy turnover monitoring.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    strategy_name: str
    timestamp: datetime
    pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    volatility: float
    return_pct: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyHealth:
    """Strategy health assessment."""
    strategy_name: str
    health_score: float  # 0-100
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: datetime
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyParameters:
    """Strategy parameter configuration."""
    strategy_name: str
    timestamp: datetime
    parameters: Dict[str, Any]
    version: str = "1.0"


@dataclass
class RegimeMetrics:
    """Regime-specific performance metrics."""
    regime: str
    strategy_name: str
    pnl: float
    sharpe_ratio: float
    win_rate: float
    trade_count: int
    timestamp: datetime


class StrategyMonitor:
    """
    Comprehensive strategy monitoring.

    Features:
    - Strategy performance metrics (Sharpe, drawdown, win rate, etc.)
    - Strategy health scores
    - Parameter drift detection
    - Strategy correlation monitoring
    - Regime-based performance tracking
    - Strategy capacity monitoring
    - Strategy turnover analysis
    - Strategy diversification metrics
    - Strategy risk contribution
    """

    def __init__(self, update_interval: int = 60):
        """
        Initialize strategy monitor.

        Args:
            update_interval: Seconds between metric updates
        """
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Strategy tracking
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.strategy_history: deque = deque(maxlen=100000)
        self.strategy_health: Dict[str, StrategyHealth] = {}

        # Parameters tracking
        self.current_parameters: Dict[str, StrategyParameters] = {}
        self.parameter_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.drift_events: deque = deque(maxlen=1000)

        # Performance tracking
        self.pnl_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Correlation tracking
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.correlation_history: deque = deque(maxlen=1000)

        # Regime tracking
        self.current_regime: Optional[str] = None
        self.regime_performance: Dict[Tuple[str, str], RegimeMetrics] = {}  # (strategy, regime) -> metrics
        self.regime_history: deque = deque(maxlen=1000)

        # Capacity tracking
        self.capacity_metrics: Dict[str, Dict[str, Any]] = {}

        # Turnover tracking
        self.turnover_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Alerts
        self.alerts: deque = deque(maxlen=1000)

        # Thresholds
        self.min_health_score = 50.0
        self.max_correlation = 0.8
        self.max_drawdown_threshold = 0.15

    def start(self):
        """Start strategy monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop strategy monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def update_strategy_metrics(self, metrics: StrategyMetrics):
        """
        Update strategy performance metrics.

        Args:
            metrics: StrategyMetrics object
        """
        with self.lock:
            self.strategies[metrics.strategy_name] = metrics
            self.strategy_history.append(metrics)

            # Update P&L history
            self.pnl_history[metrics.strategy_name].append({
                'timestamp': metrics.timestamp,
                'pnl': metrics.pnl
            })

    def update_strategy_parameters(self, parameters: StrategyParameters):
        """
        Update strategy parameters.

        Args:
            parameters: StrategyParameters object
        """
        with self.lock:
            old_params = self.current_parameters.get(parameters.strategy_name)

            # Check for drift
            if old_params:
                self._check_parameter_drift(old_params, parameters)

            self.current_parameters[parameters.strategy_name] = parameters
            self.parameter_history[parameters.strategy_name].append(parameters)

    def record_trade(self, strategy_name: str, pnl: float, timestamp: Optional[datetime] = None):
        """
        Record a trade for a strategy.

        Args:
            strategy_name: Strategy name
            pnl: Trade P&L
            timestamp: Trade timestamp
        """
        with self.lock:
            self.trade_history[strategy_name].append({
                'timestamp': timestamp or datetime.now(),
                'pnl': pnl
            })

    def set_regime(self, regime: str):
        """
        Set current market regime.

        Args:
            regime: Regime name (e.g., 'trending', 'mean_reverting', 'volatile')
        """
        with self.lock:
            self.current_regime = regime
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime
            })

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._calculate_health_scores()
                self._calculate_correlations()
                self._calculate_regime_performance()
                self._calculate_capacity_metrics()
                self._calculate_turnover()
                self._check_strategy_issues()

                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in strategy monitor loop: {e}")

    def _calculate_health_scores(self):
        """Calculate health scores for all strategies."""
        with self.lock:
            for strategy_name, metrics in self.strategies.items():
                # Calculate health score based on multiple factors
                score = 100.0
                issues = []
                health_metrics = {}

                # Factor 1: Drawdown (30 points)
                if metrics.max_drawdown > self.max_drawdown_threshold:
                    drawdown_penalty = (metrics.max_drawdown - self.max_drawdown_threshold) * 100
                    score -= min(30, drawdown_penalty)
                    issues.append(f"High drawdown: {metrics.max_drawdown:.2%}")
                health_metrics['drawdown_score'] = min(30, 30 * (1 - metrics.max_drawdown / 0.5))

                # Factor 2: Sharpe ratio (25 points)
                sharpe_score = min(25, max(0, metrics.sharpe_ratio * 12.5))
                score = score - 25 + sharpe_score
                if metrics.sharpe_ratio < 1.0:
                    issues.append(f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f}")
                health_metrics['sharpe_score'] = sharpe_score

                # Factor 3: Win rate (20 points)
                win_rate_score = metrics.win_rate * 20
                score = score - 20 + win_rate_score
                if metrics.win_rate < 0.4:
                    issues.append(f"Low win rate: {metrics.win_rate:.2%}")
                health_metrics['win_rate_score'] = win_rate_score

                # Factor 4: Recent performance (15 points)
                recent_pnl = list(self.pnl_history[strategy_name])[-10:] if self.pnl_history[strategy_name] else []
                if len(recent_pnl) >= 2:
                    recent_returns = [p['pnl'] for p in recent_pnl]
                    positive_returns = sum(1 for r in recent_returns if r > 0)
                    recent_score = (positive_returns / len(recent_returns)) * 15
                    score = score - 15 + recent_score
                    if positive_returns < len(recent_returns) * 0.4:
                        issues.append("Poor recent performance")
                    health_metrics['recent_performance_score'] = recent_score

                # Factor 5: Trade frequency (10 points)
                if metrics.total_trades < 10:
                    score -= 10
                    issues.append("Insufficient trading activity")
                    health_metrics['frequency_score'] = 0
                else:
                    health_metrics['frequency_score'] = 10

                # Determine status
                if score >= 70:
                    status = 'healthy'
                elif score >= 50:
                    status = 'degraded'
                else:
                    status = 'unhealthy'

                self.strategy_health[strategy_name] = StrategyHealth(
                    strategy_name=strategy_name,
                    health_score=score,
                    status=status,
                    timestamp=datetime.now(),
                    issues=issues,
                    metrics=health_metrics
                )

                # Alert if unhealthy
                if score < self.min_health_score:
                    self._create_alert(
                        alert_type='low_health',
                        severity='warning',
                        message=f"Strategy {strategy_name} health score: {score:.1f}",
                        metadata={'strategy': strategy_name, 'score': score, 'issues': issues}
                    )

    def _calculate_correlations(self):
        """Calculate correlation matrix between strategies."""
        with self.lock:
            if len(self.strategies) < 2:
                return

            # Get aligned P&L series for all strategies
            strategy_returns = {}
            min_length = float('inf')

            for strategy_name in self.strategies.keys():
                pnl_data = list(self.pnl_history[strategy_name])
                if len(pnl_data) >= 30:  # Need minimum data points
                    returns = [p['pnl'] for p in pnl_data[-252:]]  # Last year
                    strategy_returns[strategy_name] = returns
                    min_length = min(min_length, len(returns))

            if len(strategy_returns) < 2 or min_length < 30:
                return

            # Align series to same length
            aligned_returns = {
                name: returns[-min_length:]
                for name, returns in strategy_returns.items()
            }

            # Calculate correlation matrix
            df = pd.DataFrame(aligned_returns)
            self.correlation_matrix = df.corr()

            # Store in history
            self.correlation_history.append({
                'timestamp': datetime.now(),
                'correlation_matrix': self.correlation_matrix.copy()
            })

            # Check for high correlations
            for i, strategy1 in enumerate(self.correlation_matrix.columns):
                for j, strategy2 in enumerate(self.correlation_matrix.columns):
                    if i < j:  # Upper triangle only
                        corr = self.correlation_matrix.loc[strategy1, strategy2]
                        if abs(corr) > self.max_correlation:
                            self._create_alert(
                                alert_type='high_correlation',
                                severity='info',
                                message=f"High correlation between {strategy1} and {strategy2}: {corr:.2f}",
                                metadata={'strategy1': strategy1, 'strategy2': strategy2, 'correlation': corr}
                            )

    def _calculate_regime_performance(self):
        """Calculate strategy performance by regime."""
        with self.lock:
            if not self.current_regime:
                return

            for strategy_name in self.strategies.keys():
                # Get trades in current regime
                regime_start = None
                for entry in reversed(list(self.regime_history)):
                    if entry['regime'] == self.current_regime:
                        regime_start = entry['timestamp']
                    else:
                        break

                if not regime_start:
                    continue

                # Calculate metrics for this regime
                regime_trades = [
                    t for t in self.trade_history[strategy_name]
                    if t['timestamp'] >= regime_start
                ]

                if len(regime_trades) < 5:
                    continue

                pnls = [t['pnl'] for t in regime_trades]
                total_pnl = sum(pnls)
                wins = sum(1 for p in pnls if p > 0)
                win_rate = wins / len(pnls)

                # Sharpe ratio
                sharpe = 0.0
                if len(pnls) > 1:
                    returns = np.array(pnls)
                    if np.std(returns) > 0:
                        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

                key = (strategy_name, self.current_regime)
                self.regime_performance[key] = RegimeMetrics(
                    regime=self.current_regime,
                    strategy_name=strategy_name,
                    pnl=total_pnl,
                    sharpe_ratio=sharpe,
                    win_rate=win_rate,
                    trade_count=len(regime_trades),
                    timestamp=datetime.now()
                )

    def _calculate_capacity_metrics(self):
        """Calculate strategy capacity metrics."""
        with self.lock:
            for strategy_name, metrics in self.strategies.items():
                # Volume-based capacity estimation
                recent_trades = list(self.trade_history[strategy_name])[-100:]
                if not recent_trades:
                    continue

                # Calculate average trade size and frequency
                trade_count = len(recent_trades)
                if trade_count > 0:
                    time_span = (recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']).total_seconds() / 86400
                    avg_trades_per_day = trade_count / time_span if time_span > 0 else 0

                    # Estimate capacity (simplified)
                    avg_pnl = np.mean([t['pnl'] for t in recent_trades])
                    std_pnl = np.std([t['pnl'] for t in recent_trades])

                    self.capacity_metrics[strategy_name] = {
                        'timestamp': datetime.now(),
                        'avg_trades_per_day': avg_trades_per_day,
                        'avg_trade_pnl': avg_pnl,
                        'std_trade_pnl': std_pnl,
                        'estimated_daily_capacity': avg_trades_per_day * abs(avg_pnl) * 10  # Simplified
                    }

    def _calculate_turnover(self):
        """Calculate strategy turnover metrics."""
        with self.lock:
            for strategy_name in self.strategies.keys():
                recent_trades = list(self.trade_history[strategy_name])[-100:]
                if len(recent_trades) < 2:
                    continue

                # Calculate turnover over different periods
                time_span_days = (recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']).total_seconds() / 86400
                if time_span_days > 0:
                    daily_turnover = len(recent_trades) / time_span_days
                    weekly_turnover = daily_turnover * 5  # Trading days
                    monthly_turnover = daily_turnover * 21

                    self.turnover_history[strategy_name].append({
                        'timestamp': datetime.now(),
                        'daily_turnover': daily_turnover,
                        'weekly_turnover': weekly_turnover,
                        'monthly_turnover': monthly_turnover
                    })

    def _check_parameter_drift(self, old_params: StrategyParameters, new_params: StrategyParameters):
        """
        Check for parameter drift.

        Args:
            old_params: Previous parameters
            new_params: New parameters
        """
        drifted_params = []

        for param_name, new_value in new_params.parameters.items():
            old_value = old_params.parameters.get(param_name)

            if old_value is not None and old_value != new_value:
                # Calculate relative change for numeric parameters
                if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    if old_value != 0:
                        change_pct = abs((new_value - old_value) / old_value) * 100
                        if change_pct > 10:  # More than 10% change
                            drifted_params.append({
                                'parameter': param_name,
                                'old_value': old_value,
                                'new_value': new_value,
                                'change_pct': change_pct
                            })
                else:
                    drifted_params.append({
                        'parameter': param_name,
                        'old_value': old_value,
                        'new_value': new_value
                    })

        if drifted_params:
            self.drift_events.append({
                'timestamp': datetime.now(),
                'strategy': new_params.strategy_name,
                'drifted_parameters': drifted_params
            })

            self._create_alert(
                alert_type='parameter_drift',
                severity='info',
                message=f"Parameter drift detected for {new_params.strategy_name}",
                metadata={'strategy': new_params.strategy_name, 'changes': drifted_params}
            )

    def _check_strategy_issues(self):
        """Check for strategy-level issues."""
        with self.lock:
            for strategy_name, health in self.strategy_health.items():
                if health.status == 'unhealthy':
                    self._create_alert(
                        alert_type='strategy_unhealthy',
                        severity='critical',
                        message=f"Strategy {strategy_name} is unhealthy: {', '.join(health.issues)}",
                        metadata={'strategy': strategy_name, 'health_score': health.health_score}
                    )

    def _create_alert(self, alert_type: str, severity: str, message: str,
                     metadata: Optional[Dict] = None):
        """
        Create a strategy alert.

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

    def get_strategy_metrics(self, strategy_name: Optional[str] = None) -> Dict[str, StrategyMetrics]:
        """
        Get strategy metrics.

        Args:
            strategy_name: Optional specific strategy

        Returns:
            Dictionary of strategy metrics
        """
        with self.lock:
            if strategy_name:
                return {strategy_name: self.strategies.get(strategy_name)}
            return self.strategies.copy()

    def get_strategy_health(self, strategy_name: Optional[str] = None) -> Dict[str, StrategyHealth]:
        """
        Get strategy health scores.

        Args:
            strategy_name: Optional specific strategy

        Returns:
            Dictionary of strategy health
        """
        with self.lock:
            if strategy_name:
                return {strategy_name: self.strategy_health.get(strategy_name)}
            return self.strategy_health.copy()

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """
        Get current correlation matrix.

        Returns:
            Correlation matrix DataFrame
        """
        with self.lock:
            return self.correlation_matrix.copy() if self.correlation_matrix is not None else None

    def get_regime_performance(self, regime: Optional[str] = None) -> Dict[Tuple[str, str], RegimeMetrics]:
        """
        Get regime performance metrics.

        Args:
            regime: Optional specific regime

        Returns:
            Dictionary of regime metrics
        """
        with self.lock:
            if regime:
                return {
                    k: v for k, v in self.regime_performance.items()
                    if k[1] == regime
                }
            return self.regime_performance.copy()

    def get_capacity_metrics(self, strategy_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get capacity metrics.

        Args:
            strategy_name: Optional specific strategy

        Returns:
            Dictionary of capacity metrics
        """
        with self.lock:
            if strategy_name:
                return {strategy_name: self.capacity_metrics.get(strategy_name)}
            return self.capacity_metrics.copy()

    def get_turnover_metrics(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get turnover metrics for a strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Turnover metrics or None
        """
        with self.lock:
            if not self.turnover_history[strategy_name]:
                return None
            return self.turnover_history[strategy_name][-1]

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
        Get strategy monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            total_strategies = len(self.strategies)
            healthy = sum(1 for h in self.strategy_health.values() if h.status == 'healthy')
            degraded = sum(1 for h in self.strategy_health.values() if h.status == 'degraded')
            unhealthy = sum(1 for h in self.strategy_health.values() if h.status == 'unhealthy')

            avg_health = np.mean([h.health_score for h in self.strategy_health.values()]) if self.strategy_health else 0

            total_pnl = sum(m.pnl for m in self.strategies.values())
            avg_sharpe = np.mean([m.sharpe_ratio for m in self.strategies.values()]) if self.strategies else 0

            return {
                'total_strategies': total_strategies,
                'healthy_strategies': healthy,
                'degraded_strategies': degraded,
                'unhealthy_strategies': unhealthy,
                'avg_health_score': avg_health,
                'total_pnl': total_pnl,
                'avg_sharpe_ratio': avg_sharpe,
                'current_regime': self.current_regime,
                'drift_events': len(self.drift_events),
                'alerts_count': len(self.alerts),
                'timestamp': datetime.now().isoformat()
            }

    def get_strategy_dataframe(self) -> pd.DataFrame:
        """
        Get strategy metrics as DataFrame.

        Returns:
            DataFrame of strategy metrics
        """
        with self.lock:
            if not self.strategies:
                return pd.DataFrame()

            data = []
            for strategy_name, metrics in self.strategies.items():
                health = self.strategy_health.get(strategy_name)

                data.append({
                    'strategy': strategy_name,
                    'pnl': metrics.pnl,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'total_trades': metrics.total_trades,
                    'avg_trade_pnl': metrics.avg_trade_pnl,
                    'volatility': metrics.volatility,
                    'return_pct': metrics.return_pct,
                    'health_score': health.health_score if health else None,
                    'status': health.status if health else None,
                    'timestamp': metrics.timestamp
                })

            return pd.DataFrame(data)

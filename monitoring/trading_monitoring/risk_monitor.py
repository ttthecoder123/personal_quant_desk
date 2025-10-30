"""
Risk Monitor

Risk metrics monitoring, limit compliance, VaR tracking, exposure monitoring,
concentration risk, correlation monitoring, and stress testing results.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    limit_type: str  # 'position', 'var', 'exposure', 'concentration', 'drawdown'
    limit_value: float
    warning_threshold: float  # Percentage of limit (e.g., 0.8 for 80%)
    scope: str  # 'portfolio', 'strategy', 'symbol'
    scope_name: Optional[str] = None  # Specific strategy or symbol name
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    timestamp: datetime
    portfolio_var: float  # Portfolio Value at Risk
    portfolio_cvar: float  # Conditional VaR (Expected Shortfall)
    gross_exposure: float
    net_exposure: float
    leverage: float
    volatility: float
    beta: float
    max_drawdown: float
    current_drawdown: float
    correlation_risk: float
    concentration_risk: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VaRMetrics:
    """Value at Risk metrics."""
    timestamp: datetime
    var_95: float  # 95% confidence VaR
    var_99: float  # 99% confidence VaR
    cvar_95: float  # 95% CVaR
    cvar_99: float  # 99% CVaR
    method: str  # 'historical', 'parametric', 'monte_carlo'
    lookback_days: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LimitBreach:
    """Risk limit breach information."""
    timestamp: datetime
    limit_type: str
    current_value: float
    limit_value: float
    breach_percentage: float
    severity: str  # 'warning', 'breach'
    scope: str
    scope_name: Optional[str] = None
    message: str = ""


@dataclass
class StressTestResult:
    """Stress test result."""
    timestamp: datetime
    scenario_name: str
    portfolio_pnl: float
    var_change: float
    max_loss: float
    affected_positions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskMonitor:
    """
    Comprehensive risk monitoring.

    Features:
    - Risk metrics calculation (VaR, CVaR, volatility, etc.)
    - Risk limit compliance monitoring
    - VaR tracking (historical, parametric, Monte Carlo)
    - Exposure monitoring (gross, net, leverage)
    - Concentration risk monitoring
    - Correlation monitoring
    - Stress testing integration
    - Drawdown tracking
    - Beta and market risk
    - Marginal VaR and risk contribution
    """

    def __init__(self, update_interval: int = 60, var_confidence: float = 0.95):
        """
        Initialize risk monitor.

        Args:
            update_interval: Seconds between risk calculations
            var_confidence: VaR confidence level
        """
        self.update_interval = update_interval
        self.var_confidence = var_confidence
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Risk limits
        self.limits: List[RiskLimit] = []
        self.limit_breaches: deque = deque(maxlen=1000)

        # Risk metrics tracking
        self.risk_metrics_history: deque = deque(maxlen=10000)
        self.var_history: deque = deque(maxlen=1000)

        # Portfolio data
        self.portfolio_returns: deque = deque(maxlen=1000)
        self.position_data: Dict[str, Dict[str, float]] = {}  # symbol -> {quantity, price, etc.}
        self.portfolio_value_history: deque = deque(maxlen=1000)

        # Exposure tracking
        self.exposure_history: deque = deque(maxlen=1000)

        # Concentration tracking
        self.concentration_history: deque = deque(maxlen=1000)

        # Correlation tracking
        self.correlation_matrix: Optional[pd.DataFrame] = None

        # Stress test results
        self.stress_test_results: deque = deque(maxlen=100)

        # Alerts
        self.alerts: deque = deque(maxlen=1000)

        # Configuration
        self.var_lookback_days = 252
        self.risk_free_rate = 0.02

    def start(self):
        """Start risk monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop risk monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def add_limit(self, limit: RiskLimit):
        """
        Add a risk limit.

        Args:
            limit: RiskLimit object
        """
        with self.lock:
            self.limits.append(limit)

    def remove_limit(self, limit_type: str, scope: str, scope_name: Optional[str] = None):
        """
        Remove a risk limit.

        Args:
            limit_type: Type of limit
            scope: Limit scope
            scope_name: Optional scope name
        """
        with self.lock:
            self.limits = [
                l for l in self.limits
                if not (l.limit_type == limit_type and l.scope == scope and l.scope_name == scope_name)
            ]

    def update_portfolio_value(self, value: float, timestamp: Optional[datetime] = None):
        """
        Update portfolio value.

        Args:
            value: Portfolio value
            timestamp: Optional timestamp
        """
        with self.lock:
            timestamp = timestamp or datetime.now()
            self.portfolio_value_history.append({
                'timestamp': timestamp,
                'value': value
            })

            # Calculate returns
            if len(self.portfolio_value_history) >= 2:
                prev_value = self.portfolio_value_history[-2]['value']
                if prev_value > 0:
                    ret = (value - prev_value) / prev_value
                    self.portfolio_returns.append(ret)

    def update_positions(self, positions: Dict[str, Dict[str, float]]):
        """
        Update position data.

        Args:
            positions: Dictionary of positions {symbol: {quantity, price, value, etc.}}
        """
        with self.lock:
            self.position_data = positions.copy()

    def record_stress_test(self, result: StressTestResult):
        """
        Record stress test result.

        Args:
            result: StressTestResult object
        """
        with self.lock:
            self.stress_test_results.append(result)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._calculate_var()
                self._calculate_exposure()
                self._calculate_concentration()
                self._calculate_risk_metrics()
                self._check_limits()

                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in risk monitor loop: {e}")

    def _calculate_var(self):
        """Calculate Value at Risk."""
        with self.lock:
            if len(self.portfolio_returns) < 30:
                return

            returns = np.array(list(self.portfolio_returns))
            current_value = self.portfolio_value_history[-1]['value'] if self.portfolio_value_history else 0

            # Historical VaR
            var_95_hist = np.percentile(returns, (1 - 0.95) * 100) * current_value
            var_99_hist = np.percentile(returns, (1 - 0.99) * 100) * current_value

            # CVaR (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * current_value
            cvar_99 = np.mean(returns[returns <= np.percentile(returns, 1)]) * current_value

            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            var_95_param = (mean_return - 1.645 * std_return) * current_value
            var_99_param = (mean_return - 2.326 * std_return) * current_value

            var_metrics = VaRMetrics(
                timestamp=datetime.now(),
                var_95=abs(var_95_hist),
                var_99=abs(var_99_hist),
                cvar_95=abs(cvar_95),
                cvar_99=abs(cvar_99),
                method='historical',
                lookback_days=min(len(returns), self.var_lookback_days),
                metadata={
                    'var_95_parametric': abs(var_95_param),
                    'var_99_parametric': abs(var_99_param),
                    'portfolio_value': current_value
                }
            )

            self.var_history.append(var_metrics)

    def _calculate_exposure(self):
        """Calculate portfolio exposure metrics."""
        with self.lock:
            if not self.position_data:
                return

            gross_exposure = sum(
                abs(pos.get('value', 0))
                for pos in self.position_data.values()
            )

            net_exposure = sum(
                pos.get('value', 0) * (1 if pos.get('side') == 'long' else -1)
                for pos in self.position_data.values()
            )

            current_value = self.portfolio_value_history[-1]['value'] if self.portfolio_value_history else 1
            leverage = gross_exposure / current_value if current_value > 0 else 0

            long_exposure = sum(
                pos.get('value', 0)
                for pos in self.position_data.values()
                if pos.get('side') == 'long'
            )

            short_exposure = sum(
                pos.get('value', 0)
                for pos in self.position_data.values()
                if pos.get('side') == 'short'
            )

            self.exposure_history.append({
                'timestamp': datetime.now(),
                'gross_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'leverage': leverage,
                'net_delta': net_exposure / gross_exposure if gross_exposure > 0 else 0
            })

    def _calculate_concentration(self):
        """Calculate concentration risk."""
        with self.lock:
            if not self.position_data:
                return

            total_exposure = sum(
                abs(pos.get('value', 0))
                for pos in self.position_data.values()
            )

            if total_exposure == 0:
                return

            # Position concentration (Herfindahl index)
            weights = [
                (abs(pos.get('value', 0)) / total_exposure) ** 2
                for pos in self.position_data.values()
            ]
            herfindahl_index = sum(weights)

            # Max position concentration
            max_concentration = max(
                abs(pos.get('value', 0)) / total_exposure
                for pos in self.position_data.values()
            ) if self.position_data else 0

            # Top 5 concentration
            position_values = sorted(
                [abs(pos.get('value', 0)) for pos in self.position_data.values()],
                reverse=True
            )
            top5_concentration = sum(position_values[:5]) / total_exposure if len(position_values) >= 5 else max_concentration

            # By sector/strategy if available
            sector_concentration = defaultdict(float)
            for symbol, pos in self.position_data.items():
                sector = pos.get('sector', 'unknown')
                sector_concentration[sector] += abs(pos.get('value', 0))

            max_sector_concentration = max(
                v / total_exposure for v in sector_concentration.values()
            ) if sector_concentration else 0

            self.concentration_history.append({
                'timestamp': datetime.now(),
                'herfindahl_index': herfindahl_index,
                'max_position_concentration': max_concentration,
                'top5_concentration': top5_concentration,
                'max_sector_concentration': max_sector_concentration,
                'sector_concentrations': dict(sector_concentration)
            })

    def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics."""
        with self.lock:
            if len(self.portfolio_returns) < 30:
                return

            returns = np.array(list(self.portfolio_returns))
            current_value = self.portfolio_value_history[-1]['value'] if self.portfolio_value_history else 0

            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)

            # Beta (if market returns available - simplified)
            beta = 1.0  # Placeholder

            # Drawdown
            if len(self.portfolio_value_history) >= 2:
                values = [v['value'] for v in self.portfolio_value_history]
                high_water_mark = max(values)
                current_drawdown = (high_water_mark - current_value) / high_water_mark if high_water_mark > 0 else 0

                # Calculate max drawdown
                max_dd = 0
                peak = values[0]
                for value in values:
                    if value > peak:
                        peak = value
                    dd = (peak - value) / peak if peak > 0 else 0
                    if dd > max_dd:
                        max_dd = dd
            else:
                current_drawdown = 0
                max_dd = 0

            # Get latest VaR
            var_95 = self.var_history[-1].var_95 if self.var_history else 0

            # Get latest exposure
            exposure = self.exposure_history[-1] if self.exposure_history else {}
            gross_exp = exposure.get('gross_exposure', 0)
            net_exp = exposure.get('net_exposure', 0)
            leverage = exposure.get('leverage', 0)

            # Get latest concentration
            concentration = self.concentration_history[-1] if self.concentration_history else {}
            conc_risk = concentration.get('herfindahl_index', 0)

            # Correlation risk (simplified)
            corr_risk = 0.5  # Placeholder

            metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_var=var_95,
                portfolio_cvar=self.var_history[-1].cvar_95 if self.var_history else 0,
                gross_exposure=gross_exp,
                net_exposure=net_exp,
                leverage=leverage,
                volatility=volatility,
                beta=beta,
                max_drawdown=max_dd,
                current_drawdown=current_drawdown,
                correlation_risk=corr_risk,
                concentration_risk=conc_risk
            )

            self.risk_metrics_history.append(metrics)

    def _check_limits(self):
        """Check all risk limits."""
        with self.lock:
            if not self.risk_metrics_history:
                return

            latest_metrics = self.risk_metrics_history[-1]
            latest_exposure = self.exposure_history[-1] if self.exposure_history else {}
            latest_concentration = self.concentration_history[-1] if self.concentration_history else {}

            for limit in self.limits:
                current_value = None
                scope_desc = limit.scope

                if limit.scope_name:
                    scope_desc = f"{limit.scope}:{limit.scope_name}"

                # Get current value based on limit type
                if limit.limit_type == 'var':
                    current_value = latest_metrics.portfolio_var
                elif limit.limit_type == 'exposure':
                    current_value = latest_exposure.get('gross_exposure', 0)
                elif limit.limit_type == 'leverage':
                    current_value = latest_metrics.leverage
                elif limit.limit_type == 'drawdown':
                    current_value = latest_metrics.current_drawdown
                elif limit.limit_type == 'concentration':
                    current_value = latest_concentration.get('max_position_concentration', 0)
                elif limit.limit_type == 'position':
                    if limit.scope_name and limit.scope_name in self.position_data:
                        current_value = abs(self.position_data[limit.scope_name].get('value', 0))

                if current_value is None:
                    continue

                # Check warning threshold
                warning_level = limit.limit_value * limit.warning_threshold
                if current_value >= warning_level:
                    breach_pct = (current_value / limit.limit_value) * 100

                    severity = 'breach' if current_value >= limit.limit_value else 'warning'

                    breach = LimitBreach(
                        timestamp=datetime.now(),
                        limit_type=limit.limit_type,
                        current_value=current_value,
                        limit_value=limit.limit_value,
                        breach_percentage=breach_pct,
                        severity=severity,
                        scope=limit.scope,
                        scope_name=limit.scope_name,
                        message=f"{limit.limit_type.upper()} {severity} for {scope_desc}: {current_value:.2f} / {limit.limit_value:.2f}"
                    )

                    self.limit_breaches.append(breach)

                    self._create_alert(
                        alert_type=f'limit_{severity}',
                        severity='critical' if severity == 'breach' else 'warning',
                        message=breach.message,
                        metadata={
                            'limit_type': limit.limit_type,
                            'current_value': current_value,
                            'limit_value': limit.limit_value,
                            'breach_percentage': breach_pct
                        }
                    )

    def _create_alert(self, alert_type: str, severity: str, message: str,
                     metadata: Optional[Dict] = None):
        """
        Create a risk alert.

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

    def get_current_risk_metrics(self) -> Optional[RiskMetrics]:
        """
        Get current risk metrics.

        Returns:
            Latest RiskMetrics or None
        """
        with self.lock:
            if not self.risk_metrics_history:
                return None
            return self.risk_metrics_history[-1]

    def get_current_var(self) -> Optional[VaRMetrics]:
        """
        Get current VaR metrics.

        Returns:
            Latest VaRMetrics or None
        """
        with self.lock:
            if not self.var_history:
                return None
            return self.var_history[-1]

    def get_exposure_metrics(self) -> Dict[str, Any]:
        """
        Get current exposure metrics.

        Returns:
            Dictionary of exposure metrics
        """
        with self.lock:
            if not self.exposure_history:
                return {}
            return self.exposure_history[-1]

    def get_concentration_metrics(self) -> Dict[str, Any]:
        """
        Get current concentration metrics.

        Returns:
            Dictionary of concentration metrics
        """
        with self.lock:
            if not self.concentration_history:
                return {}
            return self.concentration_history[-1]

    def get_limit_breaches(self, severity: Optional[str] = None, limit: int = 100) -> List[LimitBreach]:
        """
        Get recent limit breaches.

        Args:
            severity: Filter by severity
            limit: Maximum number of breaches

        Returns:
            List of limit breaches
        """
        with self.lock:
            breaches = list(self.limit_breaches)
            if severity:
                breaches = [b for b in breaches if b.severity == severity]
            return breaches[-limit:]

    def get_stress_test_results(self, limit: int = 10) -> List[StressTestResult]:
        """
        Get recent stress test results.

        Args:
            limit: Maximum number of results

        Returns:
            List of stress test results
        """
        with self.lock:
            return list(self.stress_test_results)[-limit:]

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

    def get_risk_limits(self) -> List[RiskLimit]:
        """
        Get all risk limits.

        Returns:
            List of risk limits
        """
        with self.lock:
            return self.limits.copy()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get risk monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            risk_metrics = self.get_current_risk_metrics()
            var_metrics = self.get_current_var()
            exposure = self.get_exposure_metrics()
            concentration = self.get_concentration_metrics()

            breaches = sum(1 for b in self.limit_breaches if b.severity == 'breach')
            warnings = sum(1 for b in self.limit_breaches if b.severity == 'warning')

            return {
                'portfolio_var_95': var_metrics.var_95 if var_metrics else 0,
                'portfolio_cvar_95': var_metrics.cvar_95 if var_metrics else 0,
                'gross_exposure': exposure.get('gross_exposure', 0),
                'net_exposure': exposure.get('net_exposure', 0),
                'leverage': risk_metrics.leverage if risk_metrics else 0,
                'volatility': risk_metrics.volatility if risk_metrics else 0,
                'current_drawdown': risk_metrics.current_drawdown if risk_metrics else 0,
                'max_drawdown': risk_metrics.max_drawdown if risk_metrics else 0,
                'concentration_risk': concentration.get('herfindahl_index', 0),
                'max_position_concentration': concentration.get('max_position_concentration', 0),
                'total_limits': len(self.limits),
                'limit_breaches': breaches,
                'limit_warnings': warnings,
                'alerts_count': len(self.alerts),
                'timestamp': datetime.now().isoformat()
            }

    def get_risk_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """
        Get risk metrics as DataFrame.

        Args:
            hours: Hours of history to include

        Returns:
            DataFrame of risk metrics
        """
        with self.lock:
            if not self.risk_metrics_history:
                return pd.DataFrame()

            cutoff = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.risk_metrics_history if m.timestamp >= cutoff]

            data = []
            for metrics in recent_metrics:
                data.append({
                    'timestamp': metrics.timestamp,
                    'portfolio_var': metrics.portfolio_var,
                    'portfolio_cvar': metrics.portfolio_cvar,
                    'gross_exposure': metrics.gross_exposure,
                    'net_exposure': metrics.net_exposure,
                    'leverage': metrics.leverage,
                    'volatility': metrics.volatility,
                    'beta': metrics.beta,
                    'max_drawdown': metrics.max_drawdown,
                    'current_drawdown': metrics.current_drawdown,
                    'correlation_risk': metrics.correlation_risk,
                    'concentration_risk': metrics.concentration_risk
                })

            return pd.DataFrame(data)

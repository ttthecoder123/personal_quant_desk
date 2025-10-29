"""
Real-Time Risk Dashboard

Provides live monitoring and visualization of all risk metrics:
- Real-time risk metric display
- Position monitoring
- Alert visualization
- Historical trend tracking
- Risk heatmaps
- Portfolio stress indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import threading
import time

from ..core.risk_metrics import RiskMetrics, RiskMetricsResult
from ..core.risk_engine import RiskState, RiskLevel, RiskEngine


class DashboardTheme(Enum):
    """Dashboard color themes"""
    DARK = "dark"
    LIGHT = "light"
    PROFESSIONAL = "professional"


class RefreshRate(Enum):
    """Dashboard refresh rates"""
    REALTIME = 1  # 1 second
    FAST = 5  # 5 seconds
    NORMAL = 30  # 30 seconds
    SLOW = 60  # 1 minute


@dataclass
class DashboardConfig:
    """Configuration for risk dashboard"""
    theme: DashboardTheme = DashboardTheme.PROFESSIONAL
    refresh_rate: RefreshRate = RefreshRate.NORMAL
    history_length: int = 1000  # Number of data points to keep
    show_charts: bool = True
    show_alerts: bool = True
    show_positions: bool = True
    auto_scroll: bool = True
    alert_sound: bool = False


@dataclass
class PositionSnapshot:
    """Snapshot of position data for dashboard"""
    symbol: str
    quantity: float
    market_value: float
    weight_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    var_contribution: float
    risk_contribution: float
    beta: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DashboardData:
    """Container for dashboard data"""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    daily_return_pct: float
    risk_metrics: RiskMetricsResult
    risk_state: RiskState
    positions: List[PositionSnapshot]
    alerts: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_return_pct': self.daily_return_pct,
            'risk_metrics': self.risk_metrics.to_dict(),
            'risk_state': {
                'timestamp': self.risk_state.timestamp.isoformat(),
                'portfolio_value': self.risk_state.portfolio_value,
                'leverage': self.risk_state.leverage,
                'volatility': self.risk_state.volatility,
                'var_95': self.risk_state.var_95,
                'drawdown': self.risk_state.drawdown,
                'max_drawdown': self.risk_state.max_drawdown,
                'position_count': self.risk_state.position_count,
                'effective_bets': self.risk_state.effective_bets,
                'largest_position_pct': self.risk_state.largest_position_pct,
                'violations': self.risk_state.violations,
                'risk_level': self.risk_state.risk_level.value
            },
            'positions': [p.to_dict() for p in self.positions],
            'alerts': self.alerts
        }


class RiskDashboard:
    """
    Real-time risk monitoring dashboard

    Provides comprehensive real-time display of:
    - Portfolio risk metrics
    - Position exposures
    - Risk alerts and violations
    - Historical trends
    - Stress indicators
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        risk_engine: Optional[RiskEngine] = None
    ):
        """
        Initialize risk dashboard

        Args:
            config: Dashboard configuration
            risk_engine: Optional risk engine for integration
        """
        self.config = config if config is not None else DashboardConfig()
        self.risk_engine = risk_engine

        # Data storage
        self.current_data: Optional[DashboardData] = None
        self.history: deque = deque(maxlen=self.config.history_length)

        # Time series data for charts
        self.metrics_history: Dict[str, deque] = {
            'timestamp': deque(maxlen=self.config.history_length),
            'portfolio_value': deque(maxlen=self.config.history_length),
            'volatility': deque(maxlen=self.config.history_length),
            'var_95': deque(maxlen=self.config.history_length),
            'drawdown': deque(maxlen=self.config.history_length),
            'leverage': deque(maxlen=self.config.history_length),
            'sharpe_ratio': deque(maxlen=self.config.history_length),
            'position_count': deque(maxlen=self.config.history_length)
        }

        # Alert tracking
        self.active_alerts: List[Dict] = []
        self.alert_history: deque = deque(maxlen=100)

        # Update tracking
        self.last_update: Optional[datetime] = None
        self.update_count: int = 0

        # Threading for real-time updates
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def update(
        self,
        portfolio_value: float,
        positions: Dict[str, float],
        prices: Dict[str, float],
        returns: pd.Series,
        equity_curve: pd.Series,
        risk_metrics: Optional[RiskMetricsResult] = None,
        risk_state: Optional[RiskState] = None,
        alerts: Optional[List[Dict]] = None
    ) -> DashboardData:
        """
        Update dashboard with latest data

        Args:
            portfolio_value: Current portfolio value
            positions: Current positions {symbol: quantity}
            prices: Current prices {symbol: price}
            returns: Return series
            equity_curve: Equity curve
            risk_metrics: Pre-calculated risk metrics
            risk_state: Pre-calculated risk state
            alerts: Active alerts

        Returns:
            Updated dashboard data
        """
        with self._lock:
            timestamp = datetime.now()

            # Calculate risk metrics if not provided
            if risk_metrics is None:
                metrics_calc = RiskMetrics()
                risk_metrics = metrics_calc.calculate_all_metrics(
                    returns=returns,
                    equity_curve=equity_curve
                )

            # Use risk engine state if available
            if risk_state is None and self.risk_engine is not None:
                risk_state = self.risk_engine.current_state

            # Create dummy risk state if needed
            if risk_state is None:
                risk_state = RiskState(
                    timestamp=timestamp,
                    portfolio_value=portfolio_value,
                    leverage=0.0,
                    volatility=risk_metrics.portfolio_volatility,
                    var_95=risk_metrics.var_95,
                    drawdown=risk_metrics.current_drawdown,
                    max_drawdown=risk_metrics.max_drawdown,
                    position_count=len(positions),
                    effective_bets=float(len(positions)),
                    largest_position_pct=0.0,
                    violations=[],
                    risk_level=RiskLevel.LOW
                )

            # Calculate daily P&L
            daily_pnl = 0.0
            daily_return_pct = 0.0
            if len(equity_curve) > 1:
                daily_pnl = equity_curve.iloc[-1] - equity_curve.iloc[-2]
                daily_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[-2] - 1) * 100

            # Create position snapshots
            position_snapshots = self._create_position_snapshots(
                positions, prices, portfolio_value
            )

            # Create dashboard data
            dashboard_data = DashboardData(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return_pct,
                risk_metrics=risk_metrics,
                risk_state=risk_state,
                positions=position_snapshots,
                alerts=alerts if alerts is not None else []
            )

            # Update current data and history
            self.current_data = dashboard_data
            self.history.append(dashboard_data)

            # Update time series for charts
            self._update_time_series(dashboard_data)

            # Update alerts
            if alerts:
                self.active_alerts = alerts
                for alert in alerts:
                    self.alert_history.append({
                        **alert,
                        'timestamp': timestamp.isoformat()
                    })

            # Update tracking
            self.last_update = timestamp
            self.update_count += 1

            return dashboard_data

    def _create_position_snapshots(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> List[PositionSnapshot]:
        """Create position snapshots for dashboard"""
        snapshots = []

        for symbol, quantity in positions.items():
            if quantity == 0:
                continue

            price = prices.get(symbol, 0.0)
            market_value = quantity * price
            weight_pct = (market_value / portfolio_value * 100) if portfolio_value > 0 else 0

            # Create snapshot (simplified - would need more data for full implementation)
            snapshot = PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                weight_pct=weight_pct,
                unrealized_pnl=0.0,  # Would need entry prices
                unrealized_pnl_pct=0.0,
                var_contribution=0.0,  # Would need covariance matrix
                risk_contribution=0.0,
                beta=None
            )
            snapshots.append(snapshot)

        # Sort by absolute market value
        snapshots.sort(key=lambda x: abs(x.market_value), reverse=True)

        return snapshots

    def _update_time_series(self, data: DashboardData):
        """Update time series data for charts"""
        self.metrics_history['timestamp'].append(data.timestamp)
        self.metrics_history['portfolio_value'].append(data.portfolio_value)
        self.metrics_history['volatility'].append(data.risk_metrics.portfolio_volatility)
        self.metrics_history['var_95'].append(data.risk_metrics.var_95)
        self.metrics_history['drawdown'].append(data.risk_metrics.current_drawdown)
        self.metrics_history['leverage'].append(data.risk_state.leverage)
        self.metrics_history['sharpe_ratio'].append(data.risk_metrics.sharpe_ratio)
        self.metrics_history['position_count'].append(data.risk_state.position_count)

    def get_current_snapshot(self) -> Optional[Dict]:
        """
        Get current dashboard snapshot

        Returns:
            Current dashboard data as dictionary
        """
        if self.current_data is None:
            return None

        return self.current_data.to_dict()

    def get_time_series(
        self,
        metric: str,
        lookback_periods: Optional[int] = None
    ) -> Tuple[List[datetime], List[float]]:
        """
        Get time series data for charting

        Args:
            metric: Metric name (portfolio_value, volatility, etc.)
            lookback_periods: Number of periods to return (None = all)

        Returns:
            Tuple of (timestamps, values)
        """
        if metric not in self.metrics_history:
            return [], []

        timestamps = list(self.metrics_history['timestamp'])
        values = list(self.metrics_history[metric])

        if lookback_periods is not None and lookback_periods < len(timestamps):
            timestamps = timestamps[-lookback_periods:]
            values = values[-lookback_periods:]

        return timestamps, values

    def get_risk_heatmap(self) -> Dict[str, Dict[str, float]]:
        """
        Generate risk heatmap data

        Returns:
            Dictionary with risk category scores
        """
        if self.current_data is None:
            return {}

        metrics = self.current_data.risk_metrics
        state = self.current_data.risk_state

        # Calculate normalized risk scores (0-100)
        heatmap = {
            'Volatility Risk': {
                'score': min(100, (state.volatility / 0.40) * 100),  # 40% = max
                'status': self._get_status_color(state.volatility, 0.15, 0.25, 0.35)
            },
            'VaR Risk': {
                'score': min(100, (metrics.var_95 / 0.05) * 100),  # 5% = max
                'status': self._get_status_color(metrics.var_95, 0.015, 0.025, 0.04)
            },
            'Drawdown Risk': {
                'score': min(100, abs(metrics.current_drawdown / 0.30) * 100),  # 30% = max
                'status': self._get_status_color(abs(metrics.current_drawdown), 0.05, 0.15, 0.25)
            },
            'Leverage Risk': {
                'score': min(100, (state.leverage / 3.0) * 100),  # 3x = max
                'status': self._get_status_color(state.leverage, 1.0, 1.5, 2.5)
            },
            'Concentration Risk': {
                'score': min(100, (state.largest_position_pct / 0.50) * 100),  # 50% = max
                'status': self._get_status_color(state.largest_position_pct, 0.10, 0.20, 0.35)
            },
            'Diversification': {
                'score': min(100, (state.position_count / 20) * 100),  # 20 positions = optimal
                'status': self._get_status_color_inverse(state.position_count, 3, 5, 10)
            }
        }

        return heatmap

    def _get_status_color(
        self,
        value: float,
        green_threshold: float,
        yellow_threshold: float,
        red_threshold: float
    ) -> str:
        """Get status color based on thresholds"""
        if value < green_threshold:
            return 'green'
        elif value < yellow_threshold:
            return 'yellow'
        elif value < red_threshold:
            return 'orange'
        else:
            return 'red'

    def _get_status_color_inverse(
        self,
        value: float,
        red_threshold: float,
        yellow_threshold: float,
        green_threshold: float
    ) -> str:
        """Get status color based on inverse thresholds (higher is better)"""
        if value >= green_threshold:
            return 'green'
        elif value >= yellow_threshold:
            return 'yellow'
        elif value >= red_threshold:
            return 'orange'
        else:
            return 'red'

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from dashboard data

        Returns:
            Dictionary with summary statistics
        """
        if self.current_data is None:
            return {}

        # Calculate statistics from history
        portfolio_values = list(self.metrics_history['portfolio_value'])

        statistics = {
            'current_portfolio_value': self.current_data.portfolio_value,
            'daily_pnl': self.current_data.daily_pnl,
            'daily_return_pct': self.current_data.daily_return_pct,
            'peak_portfolio_value': max(portfolio_values) if portfolio_values else 0,
            'risk_level': self.current_data.risk_state.risk_level.value,
            'position_count': self.current_data.risk_state.position_count,
            'active_violations': len(self.current_data.risk_state.violations),
            'active_alerts': len(self.active_alerts),
            'sharpe_ratio': self.current_data.risk_metrics.sharpe_ratio,
            'sortino_ratio': self.current_data.risk_metrics.sortino_ratio,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_count': self.update_count
        }

        return statistics

    def get_top_positions(self, limit: int = 10) -> List[Dict]:
        """
        Get top positions by absolute market value

        Args:
            limit: Maximum number of positions to return

        Returns:
            List of position dictionaries
        """
        if self.current_data is None:
            return []

        positions = [p.to_dict() for p in self.current_data.positions[:limit]]
        return positions

    def render_text_dashboard(self) -> str:
        """
        Render text-based dashboard for terminal display

        Returns:
            Formatted dashboard string
        """
        if self.current_data is None:
            return "No data available"

        data = self.current_data
        metrics = data.risk_metrics
        state = data.risk_state

        # Build dashboard text
        lines = []
        lines.append("=" * 80)
        lines.append(f"RISK DASHBOARD - {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # Portfolio summary
        lines.append("PORTFOLIO SUMMARY:")
        lines.append(f"  Value:        ${data.portfolio_value:>15,.2f}")
        lines.append(f"  Daily P&L:    ${data.daily_pnl:>15,.2f} ({data.daily_return_pct:>6.2f}%)")
        lines.append(f"  Risk Level:   {state.risk_level.value.upper():>15}")
        lines.append(f"  Positions:    {state.position_count:>15,}")
        lines.append("")

        # Risk metrics
        lines.append("RISK METRICS:")
        lines.append(f"  Volatility:   {metrics.portfolio_volatility:>15.2%}")
        lines.append(f"  VaR (95%):    {metrics.var_95:>15.2%}")
        lines.append(f"  CVaR (95%):   {metrics.cvar_95:>15.2%}")
        lines.append(f"  Drawdown:     {metrics.current_drawdown:>15.2%}")
        lines.append(f"  Max DD:       {metrics.max_drawdown:>15.2%}")
        lines.append(f"  Sharpe:       {metrics.sharpe_ratio:>15.2f}")
        lines.append(f"  Sortino:      {metrics.sortino_ratio:>15.2f}")
        lines.append(f"  Leverage:     {state.leverage:>15.2f}x")
        lines.append("")

        # Top positions
        if data.positions:
            lines.append("TOP POSITIONS:")
            lines.append(f"  {'Symbol':<10} {'Quantity':>12} {'Value':>15} {'Weight':>8}")
            lines.append("  " + "-" * 50)
            for pos in data.positions[:5]:
                lines.append(
                    f"  {pos.symbol:<10} {pos.quantity:>12,.2f} "
                    f"${pos.market_value:>14,.2f} {pos.weight_pct:>7.1f}%"
                )
            lines.append("")

        # Alerts and violations
        if state.violations:
            lines.append("ACTIVE VIOLATIONS:")
            for violation in state.violations:
                lines.append(f"  - {violation}")
            lines.append("")

        if data.alerts:
            lines.append("ACTIVE ALERTS:")
            for alert in data.alerts[:5]:
                lines.append(f"  - [{alert.get('priority', 'INFO')}] {alert.get('message', '')}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_to_json(self, filepath: str):
        """
        Export current dashboard data to JSON file

        Args:
            filepath: Output file path
        """
        if self.current_data is None:
            raise ValueError("No data to export")

        data = self.current_data.to_dict()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def start_auto_refresh(self, callback=None):
        """
        Start automatic dashboard refresh in background thread

        Args:
            callback: Optional callback function to call after each refresh
        """
        if self._running:
            return

        self._running = True

        def refresh_loop():
            while self._running:
                if callback:
                    try:
                        callback(self)
                    except Exception as e:
                        print(f"Callback error: {e}")

                time.sleep(self.config.refresh_rate.value)

        self._update_thread = threading.Thread(target=refresh_loop, daemon=True)
        self._update_thread.start()

    def stop_auto_refresh(self):
        """Stop automatic dashboard refresh"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
            self._update_thread = None

    def clear_history(self):
        """Clear all historical data"""
        with self._lock:
            self.history.clear()
            for key in self.metrics_history:
                self.metrics_history[key].clear()
            self.alert_history.clear()
            self.update_count = 0

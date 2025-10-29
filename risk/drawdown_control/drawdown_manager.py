"""
Drawdown Manager

Implements comprehensive drawdown monitoring and control:
- Real-time drawdown calculation and tracking
- Drawdown duration and depth monitoring
- Graduated response system (5%, 10%, 15%, 20% triggers)
- Recovery metrics and analysis
- Underwater equity curve tracking
- Detailed drawdown reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class DrawdownLevel(Enum):
    """Drawdown severity levels"""
    NORMAL = "normal"          # < 5% drawdown
    CAUTION = "caution"        # 5-10% drawdown
    WARNING = "warning"        # 10-15% drawdown
    CRITICAL = "critical"      # 15-20% drawdown
    EMERGENCY = "emergency"    # > 20% drawdown


class DrawdownResponse(Enum):
    """Graduated response actions"""
    NONE = "none"
    MONITOR = "monitor"              # Watch closely
    REDUCE_SIZE = "reduce_size"      # Reduce position sizes by 25%
    REDUCE_RISK = "reduce_risk"      # Reduce position sizes by 50%
    DEFENSIVE = "defensive"          # Reduce to 25% of normal size
    HALT_TRADING = "halt_trading"    # Stop new positions


@dataclass
class DrawdownPeriod:
    """Container for a drawdown period"""
    start_date: datetime
    end_date: Optional[datetime]
    start_value: float
    trough_value: float
    trough_date: datetime
    recovery_value: Optional[float]
    max_depth: float
    duration_days: int
    underwater_days: int
    is_recovered: bool
    recovery_factor: Optional[float] = None  # (recovery_value - trough_value) / (start_value - trough_value)


@dataclass
class DrawdownMetrics:
    """Current drawdown metrics"""
    timestamp: datetime
    current_value: float
    peak_value: float
    peak_date: datetime
    current_drawdown: float
    max_drawdown: float
    drawdown_level: DrawdownLevel
    recommended_response: DrawdownResponse
    underwater_days: int
    time_to_recovery_estimate: Optional[int]  # days
    recovery_ratio: float  # current_value / peak_value


@dataclass
class DrawdownReport:
    """Comprehensive drawdown report"""
    timestamp: datetime
    current_metrics: DrawdownMetrics
    historical_periods: List[DrawdownPeriod]
    average_drawdown_depth: float
    average_drawdown_duration: float
    average_recovery_time: float
    max_historical_drawdown: float
    total_drawdown_periods: int
    current_period_start: Optional[datetime]
    days_since_peak: int
    risk_adjusted_return: float  # return / max_drawdown


class DrawdownManager:
    """
    Comprehensive drawdown monitoring and control system

    Features:
    - Real-time drawdown tracking
    - Graduated response triggers
    - Recovery analysis
    - Historical drawdown periods
    - Risk-adjusted metrics
    """

    def __init__(
        self,
        caution_threshold: float = 0.05,
        warning_threshold: float = 0.10,
        critical_threshold: float = 0.15,
        emergency_threshold: float = 0.20,
        recovery_confirmation_days: int = 5
    ):
        """
        Initialize drawdown manager

        Args:
            caution_threshold: Caution level drawdown (default 5%)
            warning_threshold: Warning level drawdown (default 10%)
            critical_threshold: Critical level drawdown (default 15%)
            emergency_threshold: Emergency level drawdown (default 20%)
            recovery_confirmation_days: Days to confirm recovery (default 5)
        """
        self.caution_threshold = caution_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.recovery_confirmation_days = recovery_confirmation_days

        # State tracking
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.drawdown_curve: pd.Series = pd.Series(dtype=float)
        self.peak_curve: pd.Series = pd.Series(dtype=float)
        self.historical_periods: List[DrawdownPeriod] = []
        self.current_period: Optional[DrawdownPeriod] = None

    def update(
        self,
        timestamp: datetime,
        portfolio_value: float
    ) -> DrawdownMetrics:
        """
        Update drawdown tracking with new portfolio value

        Args:
            timestamp: Current timestamp
            portfolio_value: Current portfolio value

        Returns:
            Current drawdown metrics
        """
        # Update equity curve
        self.equity_curve[timestamp] = portfolio_value

        # Calculate running peak
        if len(self.peak_curve) == 0:
            peak_value = portfolio_value
        else:
            peak_value = max(self.peak_curve.iloc[-1], portfolio_value)
        self.peak_curve[timestamp] = peak_value

        # Calculate drawdown
        if peak_value > 0:
            drawdown = (portfolio_value - peak_value) / peak_value
        else:
            drawdown = 0.0
        self.drawdown_curve[timestamp] = drawdown

        # Update drawdown periods
        self._update_drawdown_periods(timestamp, portfolio_value, peak_value, drawdown)

        # Get peak date
        peak_date = self.peak_curve[self.peak_curve == peak_value].index[-1]

        # Calculate metrics
        metrics = self._calculate_current_metrics(
            timestamp,
            portfolio_value,
            peak_value,
            peak_date,
            drawdown
        )

        return metrics

    def _update_drawdown_periods(
        self,
        timestamp: datetime,
        value: float,
        peak_value: float,
        drawdown: float
    ):
        """Update historical drawdown periods"""
        # Check if in drawdown
        in_drawdown = drawdown < -0.001  # Small threshold to avoid noise

        if in_drawdown:
            if self.current_period is None:
                # Start new drawdown period
                self.current_period = DrawdownPeriod(
                    start_date=timestamp,
                    end_date=None,
                    start_value=peak_value,
                    trough_value=value,
                    trough_date=timestamp,
                    recovery_value=None,
                    max_depth=drawdown,
                    duration_days=0,
                    underwater_days=0,
                    is_recovered=False
                )
            else:
                # Update existing period
                if value < self.current_period.trough_value:
                    self.current_period.trough_value = value
                    self.current_period.trough_date = timestamp
                    self.current_period.max_depth = drawdown

                self.current_period.underwater_days = (timestamp - self.current_period.start_date).days
                self.current_period.duration_days = self.current_period.underwater_days
        else:
            if self.current_period is not None and not self.current_period.is_recovered:
                # Recovery detected
                recovery_days = (timestamp - self.current_period.trough_date).days

                # Confirm recovery (no drawdown for N days)
                if recovery_days >= self.recovery_confirmation_days:
                    self.current_period.end_date = timestamp
                    self.current_period.recovery_value = value
                    self.current_period.is_recovered = True

                    # Calculate recovery factor
                    if self.current_period.start_value > self.current_period.trough_value:
                        recovered_amount = value - self.current_period.trough_value
                        total_drawdown = self.current_period.start_value - self.current_period.trough_value
                        self.current_period.recovery_factor = recovered_amount / total_drawdown

                    # Add to historical periods
                    self.historical_periods.append(self.current_period)
                    self.current_period = None

    def _calculate_current_metrics(
        self,
        timestamp: datetime,
        current_value: float,
        peak_value: float,
        peak_date: datetime,
        current_drawdown: float
    ) -> DrawdownMetrics:
        """Calculate current drawdown metrics"""
        # Determine drawdown level
        drawdown_abs = abs(current_drawdown)

        if drawdown_abs >= self.emergency_threshold:
            level = DrawdownLevel.EMERGENCY
            response = DrawdownResponse.HALT_TRADING
        elif drawdown_abs >= self.critical_threshold:
            level = DrawdownLevel.CRITICAL
            response = DrawdownResponse.DEFENSIVE
        elif drawdown_abs >= self.warning_threshold:
            level = DrawdownLevel.WARNING
            response = DrawdownResponse.REDUCE_RISK
        elif drawdown_abs >= self.caution_threshold:
            level = DrawdownLevel.CAUTION
            response = DrawdownResponse.REDUCE_SIZE
        else:
            level = DrawdownLevel.NORMAL
            response = DrawdownResponse.NONE

        # Calculate underwater days
        underwater_days = (timestamp - peak_date).days

        # Estimate time to recovery
        time_to_recovery = self._estimate_recovery_time(current_drawdown)

        # Calculate recovery ratio
        recovery_ratio = current_value / peak_value if peak_value > 0 else 1.0

        # Calculate max drawdown
        max_dd = self.drawdown_curve.min() if len(self.drawdown_curve) > 0 else 0.0

        return DrawdownMetrics(
            timestamp=timestamp,
            current_value=current_value,
            peak_value=peak_value,
            peak_date=peak_date,
            current_drawdown=current_drawdown,
            max_drawdown=max_dd,
            drawdown_level=level,
            recommended_response=response,
            underwater_days=underwater_days,
            time_to_recovery_estimate=time_to_recovery,
            recovery_ratio=recovery_ratio
        )

    def _estimate_recovery_time(self, current_drawdown: float) -> Optional[int]:
        """
        Estimate time to recovery based on historical patterns

        Args:
            current_drawdown: Current drawdown percentage

        Returns:
            Estimated days to recovery, or None if insufficient data
        """
        if len(self.historical_periods) < 3:
            return None

        # Find similar historical drawdowns
        similar_periods = [
            p for p in self.historical_periods
            if abs(p.max_depth - current_drawdown) < 0.05  # Within 5%
        ]

        if len(similar_periods) == 0:
            # Use average recovery time
            recovery_times = [p.duration_days for p in self.historical_periods]
            return int(np.mean(recovery_times))

        # Use average of similar periods
        recovery_times = [p.duration_days for p in similar_periods]
        return int(np.mean(recovery_times))

    def get_underwater_curve(self) -> pd.Series:
        """
        Get underwater equity curve (periods below peak)

        Returns:
            Series of underwater periods (1 = underwater, 0 = at peak)
        """
        underwater = (self.drawdown_curve < 0).astype(int)
        return underwater

    def calculate_recovery_metrics(self) -> Dict[str, float]:
        """
        Calculate recovery-related metrics

        Returns:
            Dictionary of recovery metrics
        """
        if len(self.historical_periods) == 0:
            return {
                'avg_recovery_time': 0.0,
                'avg_recovery_factor': 0.0,
                'max_recovery_time': 0.0,
                'min_recovery_time': 0.0,
                'recovery_success_rate': 0.0
            }

        # Filter recovered periods
        recovered = [p for p in self.historical_periods if p.is_recovered]

        if len(recovered) == 0:
            return {
                'avg_recovery_time': 0.0,
                'avg_recovery_factor': 0.0,
                'max_recovery_time': 0.0,
                'min_recovery_time': 0.0,
                'recovery_success_rate': 0.0
            }

        recovery_times = [p.duration_days for p in recovered]
        recovery_factors = [p.recovery_factor for p in recovered if p.recovery_factor is not None]

        return {
            'avg_recovery_time': np.mean(recovery_times),
            'avg_recovery_factor': np.mean(recovery_factors) if recovery_factors else 0.0,
            'max_recovery_time': np.max(recovery_times),
            'min_recovery_time': np.min(recovery_times),
            'recovery_success_rate': len(recovered) / len(self.historical_periods)
        }

    def get_drawdown_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive drawdown statistics

        Returns:
            Dictionary of drawdown statistics
        """
        if len(self.historical_periods) == 0:
            return {
                'avg_drawdown_depth': 0.0,
                'avg_drawdown_duration': 0.0,
                'max_drawdown_depth': 0.0,
                'max_drawdown_duration': 0.0,
                'total_periods': 0,
                'current_underwater_days': 0
            }

        depths = [abs(p.max_depth) for p in self.historical_periods]
        durations = [p.duration_days for p in self.historical_periods]

        current_underwater = 0
        if self.current_period is not None:
            current_underwater = self.current_period.underwater_days

        return {
            'avg_drawdown_depth': np.mean(depths),
            'avg_drawdown_duration': np.mean(durations),
            'max_drawdown_depth': np.max(depths),
            'max_drawdown_duration': np.max(durations),
            'total_periods': len(self.historical_periods),
            'current_underwater_days': current_underwater
        }

    def generate_report(self, timestamp: datetime) -> DrawdownReport:
        """
        Generate comprehensive drawdown report

        Args:
            timestamp: Report timestamp

        Returns:
            Detailed drawdown report
        """
        # Get current metrics
        if len(self.equity_curve) == 0:
            raise ValueError("No equity data available")

        current_value = self.equity_curve.iloc[-1]
        peak_value = self.peak_curve.iloc[-1]
        current_drawdown = self.drawdown_curve.iloc[-1]
        peak_date = self.peak_curve[self.peak_curve == peak_value].index[-1]

        current_metrics = self._calculate_current_metrics(
            timestamp,
            current_value,
            peak_value,
            peak_date,
            current_drawdown
        )

        # Calculate statistics
        dd_stats = self.get_drawdown_statistics()
        recovery_metrics = self.calculate_recovery_metrics()

        # Calculate risk-adjusted return
        if len(self.equity_curve) > 1:
            total_return = (current_value - self.equity_curve.iloc[0]) / self.equity_curve.iloc[0]
            max_dd = abs(self.drawdown_curve.min())
            risk_adjusted_return = total_return / max_dd if max_dd > 0 else 0.0
        else:
            risk_adjusted_return = 0.0

        # Current period start
        current_period_start = None
        if self.current_period is not None:
            current_period_start = self.current_period.start_date

        return DrawdownReport(
            timestamp=timestamp,
            current_metrics=current_metrics,
            historical_periods=self.historical_periods.copy(),
            average_drawdown_depth=dd_stats['avg_drawdown_depth'],
            average_drawdown_duration=dd_stats['avg_drawdown_duration'],
            average_recovery_time=recovery_metrics['avg_recovery_time'],
            max_historical_drawdown=dd_stats['max_drawdown_depth'],
            total_drawdown_periods=dd_stats['total_periods'],
            current_period_start=current_period_start,
            days_since_peak=(timestamp - peak_date).days,
            risk_adjusted_return=risk_adjusted_return
        )

    def get_graduated_position_scalar(self, drawdown_level: DrawdownLevel) -> float:
        """
        Get position size scalar based on drawdown level

        Args:
            drawdown_level: Current drawdown level

        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        scalars = {
            DrawdownLevel.NORMAL: 1.0,      # Full size
            DrawdownLevel.CAUTION: 0.75,    # 25% reduction
            DrawdownLevel.WARNING: 0.50,    # 50% reduction
            DrawdownLevel.CRITICAL: 0.25,   # 75% reduction
            DrawdownLevel.EMERGENCY: 0.0    # No new positions
        }

        return scalars.get(drawdown_level, 1.0)

    def should_halt_trading(self, drawdown_level: DrawdownLevel) -> bool:
        """
        Determine if trading should be halted

        Args:
            drawdown_level: Current drawdown level

        Returns:
            True if trading should be halted
        """
        return drawdown_level == DrawdownLevel.EMERGENCY

    def reset(self):
        """Reset drawdown tracking (for new strategy or time period)"""
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)
        self.peak_curve = pd.Series(dtype=float)
        self.historical_periods = []
        self.current_period = None

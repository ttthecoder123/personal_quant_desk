"""
Recovery Rules

Implements post-drawdown recovery protocols:
- Graduated position sizing after drawdowns
- Multi-stage re-entry protocol
- Performance validation before scaling up
- Risk budget restoration
- Confidence rebuilding metrics
- Strategy rotation after drawdowns
- Recovery monitoring and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class RecoveryPhase(Enum):
    """Recovery phase after drawdown"""
    NORMAL = "normal"              # No recent drawdown
    PHASE_1 = "phase_1"            # Initial recovery - 25% size
    PHASE_2 = "phase_2"            # Partial recovery - 50% size
    PHASE_3 = "phase_3"            # Advanced recovery - 75% size
    PHASE_4 = "phase_4"            # Near full recovery - 90% size
    FULL_RECOVERY = "full_recovery"  # Full size restored


class ValidationStatus(Enum):
    """Performance validation status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass
class RecoveryPhaseConfig:
    """Configuration for a recovery phase"""
    phase: RecoveryPhase
    position_size_multiplier: float  # Multiplier for normal position size
    min_days_in_phase: int          # Minimum days before advancing
    win_rate_threshold: float       # Minimum win rate to advance
    profit_threshold: float         # Minimum profit to advance (as % of max loss)
    max_drawdown_in_phase: float    # Maximum allowed drawdown in phase
    min_trades_required: int        # Minimum trades to validate


@dataclass
class ValidationCriteria:
    """Criteria for advancing to next phase"""
    min_winning_days: int
    min_sharpe_ratio: float
    max_volatility_increase: float
    min_profit_factor: float
    max_consecutive_losses: int


@dataclass
class RecoveryMetrics:
    """Metrics tracking recovery progress"""
    timestamp: datetime
    current_phase: RecoveryPhase
    days_in_phase: int
    trades_in_phase: int
    win_rate_in_phase: float
    profit_in_phase: float
    drawdown_in_phase: float
    recovery_percentage: float  # % of drawdown recovered
    validation_status: ValidationStatus
    can_advance: bool
    next_phase: Optional[RecoveryPhase]
    recommended_position_scalar: float


@dataclass
class ConfidenceMetrics:
    """Metrics for strategy confidence"""
    timestamp: datetime
    strategy_name: str
    confidence_score: float  # 0-1 scale
    recent_win_rate: float
    recent_sharpe: float
    correlation_stability: float
    volatility_ratio: float  # current/expected
    days_since_failure: int
    recommendation: str  # "increase", "maintain", "decrease", "halt"


@dataclass
class RecoveryReport:
    """Comprehensive recovery report"""
    timestamp: datetime
    recovery_metrics: RecoveryMetrics
    phase_history: List[Tuple[RecoveryPhase, datetime, datetime]]
    total_recovery_time: int  # days
    current_position_scalar: float
    strategies_rotated: List[str]
    confidence_by_strategy: Dict[str, ConfidenceMetrics]
    ready_for_full_size: bool
    recommendations: List[str]


class RecoveryRules:
    """
    Post-drawdown recovery management system

    Features:
    - Graduated re-entry protocol
    - Performance validation
    - Position sizing controls
    - Strategy confidence tracking
    - Recovery monitoring
    """

    def __init__(
        self,
        drawdown_trigger: float = 0.10,
        recovery_confirmation_days: int = 5
    ):
        """
        Initialize recovery rules

        Args:
            drawdown_trigger: Drawdown level that triggers recovery mode (default 10%)
            recovery_confirmation_days: Days to confirm phase advancement (default 5)
        """
        self.drawdown_trigger = drawdown_trigger
        self.recovery_confirmation_days = recovery_confirmation_days

        # Define recovery phases
        self.phase_configs = {
            RecoveryPhase.PHASE_1: RecoveryPhaseConfig(
                phase=RecoveryPhase.PHASE_1,
                position_size_multiplier=0.25,
                min_days_in_phase=5,
                win_rate_threshold=0.55,
                profit_threshold=0.10,  # 10% of max loss recovered
                max_drawdown_in_phase=0.03,
                min_trades_required=10
            ),
            RecoveryPhase.PHASE_2: RecoveryPhaseConfig(
                phase=RecoveryPhase.PHASE_2,
                position_size_multiplier=0.50,
                min_days_in_phase=7,
                win_rate_threshold=0.55,
                profit_threshold=0.30,  # 30% of max loss recovered
                max_drawdown_in_phase=0.05,
                min_trades_required=15
            ),
            RecoveryPhase.PHASE_3: RecoveryPhaseConfig(
                phase=RecoveryPhase.PHASE_3,
                position_size_multiplier=0.75,
                min_days_in_phase=10,
                win_rate_threshold=0.55,
                profit_threshold=0.60,  # 60% of max loss recovered
                max_drawdown_in_phase=0.07,
                min_trades_required=20
            ),
            RecoveryPhase.PHASE_4: RecoveryPhaseConfig(
                phase=RecoveryPhase.PHASE_4,
                position_size_multiplier=0.90,
                min_days_in_phase=10,
                win_rate_threshold=0.55,
                profit_threshold=0.90,  # 90% of max loss recovered
                max_drawdown_in_phase=0.08,
                min_trades_required=25
            )
        }

        # State tracking
        self.current_phase = RecoveryPhase.NORMAL
        self.phase_start_time: Optional[datetime] = None
        self.drawdown_start_time: Optional[datetime] = None
        self.max_drawdown_value: float = 0.0
        self.drawdown_amount: float = 0.0

        # Phase performance tracking
        self.phase_trades: List[Dict] = []
        self.phase_history: List[Tuple[RecoveryPhase, datetime, datetime]] = []

        # Strategy confidence tracking
        self.strategy_confidence: Dict[str, ConfidenceMetrics] = {}
        self.rotated_strategies: List[str] = []

    def enter_recovery_mode(
        self,
        timestamp: datetime,
        drawdown_amount: float,
        current_value: float
    ):
        """
        Enter recovery mode after significant drawdown

        Args:
            timestamp: Timestamp of drawdown
            drawdown_amount: Amount of drawdown
            current_value: Current portfolio value
        """
        self.current_phase = RecoveryPhase.PHASE_1
        self.phase_start_time = timestamp
        self.drawdown_start_time = timestamp
        self.max_drawdown_value = abs(drawdown_amount)
        self.drawdown_amount = drawdown_amount
        self.phase_trades = []

    def update(
        self,
        timestamp: datetime,
        portfolio_value: float,
        peak_value: float,
        trades: Optional[List[Dict]] = None
    ) -> RecoveryMetrics:
        """
        Update recovery tracking

        Args:
            timestamp: Current timestamp
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            trades: List of recent trades (optional)

        Returns:
            Current recovery metrics
        """
        # Check if we should enter recovery mode
        current_drawdown = (portfolio_value - peak_value) / peak_value

        if self.current_phase == RecoveryPhase.NORMAL:
            if current_drawdown <= -self.drawdown_trigger:
                self.enter_recovery_mode(timestamp, current_drawdown * peak_value, portfolio_value)

        # Update phase trades if provided
        if trades is not None:
            self.phase_trades.extend(trades)

        # Calculate metrics
        metrics = self._calculate_metrics(timestamp, portfolio_value, peak_value)

        # Check if can advance to next phase
        if metrics.can_advance and metrics.next_phase is not None:
            self._advance_phase(timestamp, metrics.next_phase)

        return metrics

    def _calculate_metrics(
        self,
        timestamp: datetime,
        portfolio_value: float,
        peak_value: float
    ) -> RecoveryMetrics:
        """Calculate current recovery metrics"""
        if self.current_phase == RecoveryPhase.NORMAL:
            return RecoveryMetrics(
                timestamp=timestamp,
                current_phase=RecoveryPhase.NORMAL,
                days_in_phase=0,
                trades_in_phase=0,
                win_rate_in_phase=0.0,
                profit_in_phase=0.0,
                drawdown_in_phase=0.0,
                recovery_percentage=1.0,
                validation_status=ValidationStatus.PASSED,
                can_advance=False,
                next_phase=None,
                recommended_position_scalar=1.0
            )

        # Calculate days in phase
        days_in_phase = 0
        if self.phase_start_time is not None:
            days_in_phase = (timestamp - self.phase_start_time).days

        # Calculate trade metrics
        trades_count = len(self.phase_trades)
        win_rate = 0.0
        profit = 0.0

        if trades_count > 0:
            winning_trades = [t for t in self.phase_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / trades_count
            profit = sum(t.get('pnl', 0) for t in self.phase_trades)

        # Calculate recovery percentage
        if self.max_drawdown_value > 0:
            recovered_amount = portfolio_value - (peak_value + self.drawdown_amount)
            recovery_pct = recovered_amount / self.max_drawdown_value
            recovery_pct = max(0.0, min(1.0, recovery_pct))
        else:
            recovery_pct = 1.0

        # Calculate drawdown in current phase
        phase_values = [t.get('portfolio_value', portfolio_value) for t in self.phase_trades]
        if len(phase_values) > 0:
            phase_peak = max(phase_values)
            phase_drawdown = (portfolio_value - phase_peak) / phase_peak
        else:
            phase_drawdown = 0.0

        # Check if can advance
        can_advance, next_phase, validation_status = self._check_advancement_criteria(
            days_in_phase,
            trades_count,
            win_rate,
            profit,
            phase_drawdown,
            recovery_pct
        )

        # Get position scalar
        config = self.phase_configs.get(self.current_phase)
        position_scalar = config.position_size_multiplier if config else 1.0

        return RecoveryMetrics(
            timestamp=timestamp,
            current_phase=self.current_phase,
            days_in_phase=days_in_phase,
            trades_in_phase=trades_count,
            win_rate_in_phase=win_rate,
            profit_in_phase=profit,
            drawdown_in_phase=phase_drawdown,
            recovery_percentage=recovery_pct,
            validation_status=validation_status,
            can_advance=can_advance,
            next_phase=next_phase,
            recommended_position_scalar=position_scalar
        )

    def _check_advancement_criteria(
        self,
        days_in_phase: int,
        trades_count: int,
        win_rate: float,
        profit: float,
        phase_drawdown: float,
        recovery_pct: float
    ) -> Tuple[bool, Optional[RecoveryPhase], ValidationStatus]:
        """
        Check if criteria met to advance to next phase

        Returns:
            Tuple of (can_advance, next_phase, validation_status)
        """
        if self.current_phase == RecoveryPhase.NORMAL:
            return False, None, ValidationStatus.PASSED

        if self.current_phase == RecoveryPhase.FULL_RECOVERY:
            return False, None, ValidationStatus.PASSED

        # Get current phase config
        config = self.phase_configs.get(self.current_phase)
        if config is None:
            return False, None, ValidationStatus.NOT_STARTED

        # Check minimum requirements
        if days_in_phase < config.min_days_in_phase:
            return False, None, ValidationStatus.IN_PROGRESS

        if trades_count < config.min_trades_required:
            return False, None, ValidationStatus.IN_PROGRESS

        # Check performance criteria
        criteria_met = []

        # Win rate
        if win_rate >= config.win_rate_threshold:
            criteria_met.append(True)
        else:
            criteria_met.append(False)

        # Profit recovery
        if self.max_drawdown_value > 0:
            profit_pct = profit / self.max_drawdown_value
            if profit_pct >= config.profit_threshold:
                criteria_met.append(True)
            else:
                criteria_met.append(False)
        else:
            criteria_met.append(True)

        # Drawdown control
        if abs(phase_drawdown) <= config.max_drawdown_in_phase:
            criteria_met.append(True)
        else:
            criteria_met.append(False)

        # Determine next phase
        phase_order = [
            RecoveryPhase.PHASE_1,
            RecoveryPhase.PHASE_2,
            RecoveryPhase.PHASE_3,
            RecoveryPhase.PHASE_4,
            RecoveryPhase.FULL_RECOVERY
        ]

        current_idx = phase_order.index(self.current_phase)
        next_phase = phase_order[current_idx + 1] if current_idx < len(phase_order) - 1 else None

        # All criteria must be met
        if all(criteria_met):
            return True, next_phase, ValidationStatus.PASSED
        else:
            return False, None, ValidationStatus.IN_PROGRESS

    def _advance_phase(self, timestamp: datetime, next_phase: RecoveryPhase):
        """Advance to next recovery phase"""
        # Record phase history
        if self.phase_start_time is not None:
            self.phase_history.append((self.current_phase, self.phase_start_time, timestamp))

        # Update phase
        self.current_phase = next_phase
        self.phase_start_time = timestamp

        # Reset phase tracking
        self.phase_trades = []

    def calculate_position_scalar(
        self,
        base_position_size: float,
        strategy_name: Optional[str] = None
    ) -> float:
        """
        Calculate position size scalar based on recovery phase

        Args:
            base_position_size: Normal position size
            strategy_name: Strategy name for confidence adjustment (optional)

        Returns:
            Adjusted position size
        """
        # Get phase multiplier
        if self.current_phase == RecoveryPhase.NORMAL or self.current_phase == RecoveryPhase.FULL_RECOVERY:
            phase_scalar = 1.0
        else:
            config = self.phase_configs.get(self.current_phase)
            phase_scalar = config.position_size_multiplier if config else 1.0

        # Apply strategy confidence if provided
        if strategy_name is not None and strategy_name in self.strategy_confidence:
            confidence = self.strategy_confidence[strategy_name]
            confidence_scalar = confidence.confidence_score
            final_scalar = phase_scalar * confidence_scalar
        else:
            final_scalar = phase_scalar

        return base_position_size * final_scalar

    def update_strategy_confidence(
        self,
        strategy_name: str,
        recent_trades: List[Dict],
        expected_sharpe: float = 1.5,
        expected_win_rate: float = 0.55
    ) -> ConfidenceMetrics:
        """
        Update confidence metrics for a strategy

        Args:
            strategy_name: Name of strategy
            recent_trades: Recent trade history
            expected_sharpe: Expected Sharpe ratio
            expected_win_rate: Expected win rate

        Returns:
            Updated confidence metrics
        """
        if len(recent_trades) == 0:
            confidence_score = 0.5  # Neutral
        else:
            # Calculate recent performance
            winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(recent_trades)

            # Calculate recent Sharpe
            returns = [t.get('return', 0) for t in recent_trades]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0.0

            # Calculate confidence components (0-1 scale)
            win_rate_score = min(win_rate / expected_win_rate, 1.0)
            sharpe_score = min(sharpe / expected_sharpe, 1.0) if expected_sharpe > 0 else 0.0

            # Weighted confidence score
            confidence_score = 0.6 * win_rate_score + 0.4 * sharpe_score
            confidence_score = max(0.0, min(1.0, confidence_score))

        # Determine recommendation
        if confidence_score >= 0.8:
            recommendation = "increase"
        elif confidence_score >= 0.6:
            recommendation = "maintain"
        elif confidence_score >= 0.4:
            recommendation = "decrease"
        else:
            recommendation = "halt"

        metrics = ConfidenceMetrics(
            timestamp=datetime.now(),
            strategy_name=strategy_name,
            confidence_score=confidence_score,
            recent_win_rate=win_rate if recent_trades else 0.0,
            recent_sharpe=sharpe if recent_trades else 0.0,
            correlation_stability=1.0,  # Would calculate from actual data
            volatility_ratio=1.0,       # Would calculate from actual data
            days_since_failure=0,       # Would track from actual data
            recommendation=recommendation
        )

        self.strategy_confidence[strategy_name] = metrics
        return metrics

    def rotate_strategy(
        self,
        strategy_name: str,
        reason: str
    ):
        """
        Rotate out a strategy during recovery

        Args:
            strategy_name: Strategy to rotate out
            reason: Reason for rotation
        """
        if strategy_name not in self.rotated_strategies:
            self.rotated_strategies.append(strategy_name)

        # Set confidence to zero
        if strategy_name in self.strategy_confidence:
            self.strategy_confidence[strategy_name].confidence_score = 0.0
            self.strategy_confidence[strategy_name].recommendation = "halt"

    def restore_strategy(
        self,
        strategy_name: str
    ):
        """
        Restore a previously rotated strategy

        Args:
            strategy_name: Strategy to restore
        """
        if strategy_name in self.rotated_strategies:
            self.rotated_strategies.remove(strategy_name)

    def generate_report(self, timestamp: datetime) -> RecoveryReport:
        """
        Generate comprehensive recovery report

        Args:
            timestamp: Report timestamp

        Returns:
            Detailed recovery report
        """
        # Calculate current metrics (with dummy values for required params)
        current_metrics = RecoveryMetrics(
            timestamp=timestamp,
            current_phase=self.current_phase,
            days_in_phase=0,
            trades_in_phase=len(self.phase_trades),
            win_rate_in_phase=0.0,
            profit_in_phase=0.0,
            drawdown_in_phase=0.0,
            recovery_percentage=0.0,
            validation_status=ValidationStatus.IN_PROGRESS,
            can_advance=False,
            next_phase=None,
            recommended_position_scalar=1.0
        )

        # Calculate total recovery time
        total_recovery_time = 0
        if self.drawdown_start_time is not None:
            total_recovery_time = (timestamp - self.drawdown_start_time).days

        # Get current position scalar
        if self.current_phase in self.phase_configs:
            position_scalar = self.phase_configs[self.current_phase].position_size_multiplier
        else:
            position_scalar = 1.0

        # Check if ready for full size
        ready_for_full_size = (self.current_phase == RecoveryPhase.FULL_RECOVERY or
                               self.current_phase == RecoveryPhase.NORMAL)

        # Generate recommendations
        recommendations = self._generate_recommendations(current_metrics)

        return RecoveryReport(
            timestamp=timestamp,
            recovery_metrics=current_metrics,
            phase_history=self.phase_history.copy(),
            total_recovery_time=total_recovery_time,
            current_position_scalar=position_scalar,
            strategies_rotated=self.rotated_strategies.copy(),
            confidence_by_strategy=self.strategy_confidence.copy(),
            ready_for_full_size=ready_for_full_size,
            recommendations=recommendations
        )

    def _generate_recommendations(self, metrics: RecoveryMetrics) -> List[str]:
        """Generate recovery recommendations"""
        recommendations = []

        if metrics.current_phase == RecoveryPhase.NORMAL:
            recommendations.append("Operating at normal position sizing")
            return recommendations

        # Phase-specific recommendations
        if metrics.days_in_phase < 5:
            recommendations.append(f"Continue in {metrics.current_phase.value} - early in phase")

        if metrics.trades_in_phase < 10:
            recommendations.append("Need more trades to validate performance")

        if metrics.win_rate_in_phase < 0.55:
            recommendations.append("Win rate below threshold - focus on quality setups")

        if abs(metrics.drawdown_in_phase) > 0.05:
            recommendations.append("Drawdown in phase elevated - reduce risk further")

        if metrics.can_advance:
            recommendations.append(f"Ready to advance to {metrics.next_phase.value}")

        # Strategy confidence recommendations
        low_confidence_strategies = [
            name for name, conf in self.strategy_confidence.items()
            if conf.confidence_score < 0.5
        ]

        if low_confidence_strategies:
            recommendations.append(f"Consider rotating out: {', '.join(low_confidence_strategies)}")

        return recommendations

    def is_in_recovery_mode(self) -> bool:
        """Check if currently in recovery mode"""
        return self.current_phase not in [RecoveryPhase.NORMAL, RecoveryPhase.FULL_RECOVERY]

    def get_current_phase(self) -> RecoveryPhase:
        """Get current recovery phase"""
        return self.current_phase

    def force_phase_advance(self, timestamp: datetime):
        """Manually advance to next phase (for testing or override)"""
        phase_order = [
            RecoveryPhase.PHASE_1,
            RecoveryPhase.PHASE_2,
            RecoveryPhase.PHASE_3,
            RecoveryPhase.PHASE_4,
            RecoveryPhase.FULL_RECOVERY
        ]

        if self.current_phase in phase_order:
            current_idx = phase_order.index(self.current_phase)
            if current_idx < len(phase_order) - 1:
                next_phase = phase_order[current_idx + 1]
                self._advance_phase(timestamp, next_phase)

    def reset(self):
        """Reset recovery system"""
        self.current_phase = RecoveryPhase.NORMAL
        self.phase_start_time = None
        self.drawdown_start_time = None
        self.max_drawdown_value = 0.0
        self.drawdown_amount = 0.0
        self.phase_trades = []
        self.phase_history = []
        self.strategy_confidence = {}
        self.rotated_strategies = []

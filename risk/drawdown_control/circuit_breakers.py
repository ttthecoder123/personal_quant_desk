"""
Circuit Breakers

Implements automated trading halts and risk controls:
- Multi-level circuit breaker system (Level 1, 2, 3)
- Daily loss limits
- Velocity-based triggers (rapid losses)
- Correlation breakdown detection
- System failure triggers
- Graduated halt protocols
- Automatic recovery conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class CircuitBreakerLevel(Enum):
    """Circuit breaker severity levels"""
    NONE = "none"
    LEVEL_1 = "level_1"  # Minor halt - 15 minute pause
    LEVEL_2 = "level_2"  # Moderate halt - 1 hour pause
    LEVEL_3 = "level_3"  # Severe halt - trading suspended for day


class TriggerType(Enum):
    """Types of circuit breaker triggers"""
    DAILY_LOSS = "daily_loss"
    VELOCITY = "velocity"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    SYSTEM_FAILURE = "system_failure"
    MANUAL = "manual"
    DRAWDOWN = "drawdown"


class TradingStatus(Enum):
    """Trading status"""
    ACTIVE = "active"
    PAUSED = "paused"
    HALTED = "halted"
    SUSPENDED = "suspended"


@dataclass
class CircuitBreakerTrigger:
    """Record of circuit breaker trigger event"""
    timestamp: datetime
    level: CircuitBreakerLevel
    trigger_type: TriggerType
    trigger_value: float
    threshold: float
    description: str
    halt_duration: timedelta
    resume_time: datetime
    auto_resume: bool


@dataclass
class LossVelocity:
    """Track rate of losses"""
    timestamp: datetime
    loss_amount: float
    time_window_minutes: int
    velocity: float  # Loss per minute
    threshold: float


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker system"""
    # Level 1 triggers (minor halt - 15 min)
    level_1_daily_loss: float = 0.03      # 3% daily loss
    level_1_velocity: float = 0.01        # 1% loss in 5 minutes
    level_1_halt_minutes: int = 15

    # Level 2 triggers (moderate halt - 1 hour)
    level_2_daily_loss: float = 0.05      # 5% daily loss
    level_2_velocity: float = 0.02        # 2% loss in 5 minutes
    level_2_halt_minutes: int = 60

    # Level 3 triggers (severe halt - rest of day)
    level_3_daily_loss: float = 0.07      # 7% daily loss
    level_3_velocity: float = 0.03        # 3% loss in 5 minutes
    level_3_halt_hours: int = 24

    # Additional triggers
    correlation_breakdown: float = 0.30   # Correlation drops below 0.3
    volatility_spike: float = 3.0         # Volatility > 3x normal
    max_consecutive_losses: int = 5       # Max consecutive losing trades

    # Velocity settings
    velocity_window_minutes: int = 5      # Time window for velocity calculation


@dataclass
class CircuitBreakerStatus:
    """Current circuit breaker status"""
    timestamp: datetime
    trading_status: TradingStatus
    active_level: CircuitBreakerLevel
    triggers_today: int
    total_triggers: int
    last_trigger: Optional[CircuitBreakerTrigger]
    resume_time: Optional[datetime]
    can_trade: bool
    time_until_resume: Optional[timedelta]


class CircuitBreakers:
    """
    Comprehensive circuit breaker system for risk management

    Features:
    - Three-tier graduated halt system
    - Multiple trigger types
    - Automatic and manual triggers
    - Recovery protocols
    - Historical tracking
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker system

        Args:
            config: Circuit breaker configuration (optional)
        """
        self.config = config if config is not None else CircuitBreakerConfig()

        # State tracking
        self.trading_status = TradingStatus.ACTIVE
        self.active_level = CircuitBreakerLevel.NONE
        self.trigger_history: List[CircuitBreakerTrigger] = []
        self.current_halt: Optional[CircuitBreakerTrigger] = None

        # Daily tracking
        self.daily_start_value: Optional[float] = None
        self.daily_peak_value: Optional[float] = None
        self.daily_triggers: int = 0
        self.current_date: Optional[datetime] = None

        # Loss tracking for velocity
        self.recent_losses: List[Tuple[datetime, float]] = []
        self.consecutive_losses: int = 0

    def update(
        self,
        timestamp: datetime,
        portfolio_value: float,
        returns: Optional[pd.Series] = None
    ) -> CircuitBreakerStatus:
        """
        Update circuit breaker system and check for triggers

        Args:
            timestamp: Current timestamp
            portfolio_value: Current portfolio value
            returns: Recent return series (optional, for volatility checks)

        Returns:
            Current circuit breaker status
        """
        # Reset daily counters if new day
        self._check_new_day(timestamp, portfolio_value)

        # Check if currently halted
        if self._is_halted(timestamp):
            return self._get_status(timestamp)

        # Check various triggers
        triggered = self._check_all_triggers(timestamp, portfolio_value, returns)

        if triggered:
            self.trigger_history.append(triggered)
            self.current_halt = triggered
            self.daily_triggers += 1
            self.active_level = triggered.level
            self._set_trading_status(triggered.level)

        return self._get_status(timestamp)

    def _check_new_day(self, timestamp: datetime, portfolio_value: float):
        """Check if it's a new trading day and reset counters"""
        current_date = timestamp.date()

        if self.current_date is None or current_date != self.current_date:
            # New day - reset counters
            self.current_date = current_date
            self.daily_start_value = portfolio_value
            self.daily_peak_value = portfolio_value
            self.daily_triggers = 0
            self.consecutive_losses = 0

            # Check if suspended halt should be lifted
            if self.trading_status == TradingStatus.SUSPENDED:
                self.trading_status = TradingStatus.ACTIVE
                self.active_level = CircuitBreakerLevel.NONE
                self.current_halt = None
        else:
            # Update daily peak
            if self.daily_peak_value is None or portfolio_value > self.daily_peak_value:
                self.daily_peak_value = portfolio_value

    def _is_halted(self, timestamp: datetime) -> bool:
        """Check if trading is currently halted"""
        if self.current_halt is None:
            return False

        if timestamp < self.current_halt.resume_time:
            return True

        # Halt period expired
        if self.current_halt.auto_resume:
            self._resume_trading(timestamp)
            return False

        return True

    def _check_all_triggers(
        self,
        timestamp: datetime,
        portfolio_value: float,
        returns: Optional[pd.Series]
    ) -> Optional[CircuitBreakerTrigger]:
        """
        Check all circuit breaker triggers

        Returns:
            CircuitBreakerTrigger if triggered, None otherwise
        """
        # Check daily loss limits
        trigger = self._check_daily_loss(timestamp, portfolio_value)
        if trigger:
            return trigger

        # Check loss velocity
        trigger = self._check_velocity(timestamp, portfolio_value)
        if trigger:
            return trigger

        # Check volatility spike (if returns provided)
        if returns is not None:
            trigger = self._check_volatility_spike(timestamp, returns)
            if trigger:
                return trigger

        return None

    def _check_daily_loss(
        self,
        timestamp: datetime,
        portfolio_value: float
    ) -> Optional[CircuitBreakerTrigger]:
        """Check daily loss limits"""
        if self.daily_start_value is None:
            return None

        daily_return = (portfolio_value - self.daily_start_value) / self.daily_start_value

        # Check Level 3 (most severe)
        if daily_return <= -self.config.level_3_daily_loss:
            return CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_3,
                trigger_type=TriggerType.DAILY_LOSS,
                trigger_value=daily_return,
                threshold=-self.config.level_3_daily_loss,
                description=f"Level 3: Daily loss {daily_return:.2%} exceeded {self.config.level_3_daily_loss:.2%}",
                halt_duration=timedelta(hours=self.config.level_3_halt_hours),
                resume_time=timestamp + timedelta(hours=self.config.level_3_halt_hours),
                auto_resume=True
            )

        # Check Level 2
        elif daily_return <= -self.config.level_2_daily_loss:
            return CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_2,
                trigger_type=TriggerType.DAILY_LOSS,
                trigger_value=daily_return,
                threshold=-self.config.level_2_daily_loss,
                description=f"Level 2: Daily loss {daily_return:.2%} exceeded {self.config.level_2_daily_loss:.2%}",
                halt_duration=timedelta(minutes=self.config.level_2_halt_minutes),
                resume_time=timestamp + timedelta(minutes=self.config.level_2_halt_minutes),
                auto_resume=True
            )

        # Check Level 1
        elif daily_return <= -self.config.level_1_daily_loss:
            return CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_1,
                trigger_type=TriggerType.DAILY_LOSS,
                trigger_value=daily_return,
                threshold=-self.config.level_1_daily_loss,
                description=f"Level 1: Daily loss {daily_return:.2%} exceeded {self.config.level_1_daily_loss:.2%}",
                halt_duration=timedelta(minutes=self.config.level_1_halt_minutes),
                resume_time=timestamp + timedelta(minutes=self.config.level_1_halt_minutes),
                auto_resume=True
            )

        return None

    def _check_velocity(
        self,
        timestamp: datetime,
        portfolio_value: float
    ) -> Optional[CircuitBreakerTrigger]:
        """Check loss velocity (rapid losses)"""
        if self.daily_peak_value is None:
            return None

        # Track recent loss
        current_loss = (portfolio_value - self.daily_peak_value) / self.daily_peak_value
        if current_loss < 0:
            self.recent_losses.append((timestamp, abs(current_loss)))

        # Clean old losses outside window
        window_start = timestamp - timedelta(minutes=self.config.velocity_window_minutes)
        self.recent_losses = [
            (t, loss) for t, loss in self.recent_losses
            if t >= window_start
        ]

        if len(self.recent_losses) == 0:
            return None

        # Calculate velocity (loss per minute)
        total_loss = sum(loss for _, loss in self.recent_losses)
        velocity = total_loss / self.config.velocity_window_minutes

        # Check velocity triggers
        if velocity >= self.config.level_3_velocity:
            return CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_3,
                trigger_type=TriggerType.VELOCITY,
                trigger_value=velocity,
                threshold=self.config.level_3_velocity,
                description=f"Level 3: Loss velocity {velocity:.4%}/min exceeded {self.config.level_3_velocity:.4%}/min",
                halt_duration=timedelta(hours=self.config.level_3_halt_hours),
                resume_time=timestamp + timedelta(hours=self.config.level_3_halt_hours),
                auto_resume=True
            )

        elif velocity >= self.config.level_2_velocity:
            return CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_2,
                trigger_type=TriggerType.VELOCITY,
                trigger_value=velocity,
                threshold=self.config.level_2_velocity,
                description=f"Level 2: Loss velocity {velocity:.4%}/min exceeded {self.config.level_2_velocity:.4%}/min",
                halt_duration=timedelta(minutes=self.config.level_2_halt_minutes),
                resume_time=timestamp + timedelta(minutes=self.config.level_2_halt_minutes),
                auto_resume=True
            )

        elif velocity >= self.config.level_1_velocity:
            return CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_1,
                trigger_type=TriggerType.VELOCITY,
                trigger_value=velocity,
                threshold=self.config.level_1_velocity,
                description=f"Level 1: Loss velocity {velocity:.4%}/min exceeded {self.config.level_1_velocity:.4%}/min",
                halt_duration=timedelta(minutes=self.config.level_1_halt_minutes),
                resume_time=timestamp + timedelta(minutes=self.config.level_1_halt_minutes),
                auto_resume=True
            )

        return None

    def _check_volatility_spike(
        self,
        timestamp: datetime,
        returns: pd.Series
    ) -> Optional[CircuitBreakerTrigger]:
        """Check for volatility spikes"""
        if len(returns) < 20:
            return None

        # Calculate recent volatility
        recent_vol = returns.tail(5).std()
        baseline_vol = returns.tail(20).std()

        if baseline_vol == 0:
            return None

        vol_ratio = recent_vol / baseline_vol

        if vol_ratio >= self.config.volatility_spike:
            return CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_2,
                trigger_type=TriggerType.VOLATILITY,
                trigger_value=vol_ratio,
                threshold=self.config.volatility_spike,
                description=f"Volatility spike: {vol_ratio:.2f}x normal volatility",
                halt_duration=timedelta(minutes=self.config.level_2_halt_minutes),
                resume_time=timestamp + timedelta(minutes=self.config.level_2_halt_minutes),
                auto_resume=True
            )

        return None

    def trigger_correlation_breaker(
        self,
        timestamp: datetime,
        expected_correlation: float,
        actual_correlation: float
    ) -> Optional[CircuitBreakerTrigger]:
        """
        Manually trigger correlation breakdown circuit breaker

        Args:
            timestamp: Current timestamp
            expected_correlation: Expected correlation
            actual_correlation: Actual correlation

        Returns:
            CircuitBreakerTrigger if triggered
        """
        if actual_correlation < self.config.correlation_breakdown:
            trigger = CircuitBreakerTrigger(
                timestamp=timestamp,
                level=CircuitBreakerLevel.LEVEL_2,
                trigger_type=TriggerType.CORRELATION,
                trigger_value=actual_correlation,
                threshold=self.config.correlation_breakdown,
                description=f"Correlation breakdown: {actual_correlation:.2f} < {self.config.correlation_breakdown:.2f}",
                halt_duration=timedelta(minutes=self.config.level_2_halt_minutes),
                resume_time=timestamp + timedelta(minutes=self.config.level_2_halt_minutes),
                auto_resume=False  # Requires manual review
            )

            self.trigger_history.append(trigger)
            self.current_halt = trigger
            self.daily_triggers += 1
            self.active_level = trigger.level
            self._set_trading_status(trigger.level)

            return trigger

        return None

    def trigger_system_failure(
        self,
        timestamp: datetime,
        description: str
    ) -> CircuitBreakerTrigger:
        """
        Trigger circuit breaker due to system failure

        Args:
            timestamp: Current timestamp
            description: Description of failure

        Returns:
            CircuitBreakerTrigger
        """
        trigger = CircuitBreakerTrigger(
            timestamp=timestamp,
            level=CircuitBreakerLevel.LEVEL_3,
            trigger_type=TriggerType.SYSTEM_FAILURE,
            trigger_value=0.0,
            threshold=0.0,
            description=f"System failure: {description}",
            halt_duration=timedelta(hours=24),
            resume_time=timestamp + timedelta(hours=24),
            auto_resume=False  # Requires manual intervention
        )

        self.trigger_history.append(trigger)
        self.current_halt = trigger
        self.daily_triggers += 1
        self.active_level = CircuitBreakerLevel.LEVEL_3
        self.trading_status = TradingStatus.SUSPENDED

        return trigger

    def trigger_manual_halt(
        self,
        timestamp: datetime,
        level: CircuitBreakerLevel,
        description: str,
        duration_minutes: int = 60
    ) -> CircuitBreakerTrigger:
        """
        Manually trigger circuit breaker

        Args:
            timestamp: Current timestamp
            level: Circuit breaker level
            description: Reason for manual halt
            duration_minutes: Halt duration in minutes

        Returns:
            CircuitBreakerTrigger
        """
        trigger = CircuitBreakerTrigger(
            timestamp=timestamp,
            level=level,
            trigger_type=TriggerType.MANUAL,
            trigger_value=0.0,
            threshold=0.0,
            description=description,
            halt_duration=timedelta(minutes=duration_minutes),
            resume_time=timestamp + timedelta(minutes=duration_minutes),
            auto_resume=False  # Manual halts require manual resume
        )

        self.trigger_history.append(trigger)
        self.current_halt = trigger
        self.daily_triggers += 1
        self.active_level = level
        self._set_trading_status(level)

        return trigger

    def _set_trading_status(self, level: CircuitBreakerLevel):
        """Set trading status based on circuit breaker level"""
        if level == CircuitBreakerLevel.LEVEL_1:
            self.trading_status = TradingStatus.PAUSED
        elif level == CircuitBreakerLevel.LEVEL_2:
            self.trading_status = TradingStatus.HALTED
        elif level == CircuitBreakerLevel.LEVEL_3:
            self.trading_status = TradingStatus.SUSPENDED

    def _resume_trading(self, timestamp: datetime):
        """Resume trading after halt period"""
        self.trading_status = TradingStatus.ACTIVE
        self.active_level = CircuitBreakerLevel.NONE
        self.current_halt = None

    def manual_resume(self, timestamp: datetime) -> bool:
        """
        Manually resume trading

        Args:
            timestamp: Resume timestamp

        Returns:
            True if resumed, False if not halted
        """
        if self.current_halt is None:
            return False

        self._resume_trading(timestamp)
        return True

    def can_trade(self, timestamp: datetime) -> bool:
        """
        Check if trading is allowed

        Args:
            timestamp: Current timestamp

        Returns:
            True if trading allowed
        """
        return not self._is_halted(timestamp)

    def _get_status(self, timestamp: datetime) -> CircuitBreakerStatus:
        """Get current circuit breaker status"""
        time_until_resume = None
        resume_time = None

        if self.current_halt is not None:
            resume_time = self.current_halt.resume_time
            if timestamp < resume_time:
                time_until_resume = resume_time - timestamp

        return CircuitBreakerStatus(
            timestamp=timestamp,
            trading_status=self.trading_status,
            active_level=self.active_level,
            triggers_today=self.daily_triggers,
            total_triggers=len(self.trigger_history),
            last_trigger=self.current_halt,
            resume_time=resume_time,
            can_trade=self.can_trade(timestamp),
            time_until_resume=time_until_resume
        )

    def get_trigger_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trigger_type: Optional[TriggerType] = None
    ) -> List[CircuitBreakerTrigger]:
        """
        Get historical circuit breaker triggers

        Args:
            start_date: Filter start date (optional)
            end_date: Filter end date (optional)
            trigger_type: Filter by trigger type (optional)

        Returns:
            List of triggers matching filters
        """
        triggers = self.trigger_history

        if start_date is not None:
            triggers = [t for t in triggers if t.timestamp >= start_date]

        if end_date is not None:
            triggers = [t for t in triggers if t.timestamp <= end_date]

        if trigger_type is not None:
            triggers = [t for t in triggers if t.trigger_type == trigger_type]

        return triggers

    def get_statistics(self) -> Dict[str, any]:
        """
        Get circuit breaker statistics

        Returns:
            Dictionary of statistics
        """
        if len(self.trigger_history) == 0:
            return {
                'total_triggers': 0,
                'level_1_triggers': 0,
                'level_2_triggers': 0,
                'level_3_triggers': 0,
                'avg_halt_duration_minutes': 0.0,
                'triggers_by_type': {}
            }

        level_counts = {
            CircuitBreakerLevel.LEVEL_1: sum(1 for t in self.trigger_history if t.level == CircuitBreakerLevel.LEVEL_1),
            CircuitBreakerLevel.LEVEL_2: sum(1 for t in self.trigger_history if t.level == CircuitBreakerLevel.LEVEL_2),
            CircuitBreakerLevel.LEVEL_3: sum(1 for t in self.trigger_history if t.level == CircuitBreakerLevel.LEVEL_3),
        }

        type_counts = {}
        for trigger in self.trigger_history:
            trigger_type = trigger.trigger_type.value
            type_counts[trigger_type] = type_counts.get(trigger_type, 0) + 1

        avg_duration = np.mean([t.halt_duration.total_seconds() / 60 for t in self.trigger_history])

        return {
            'total_triggers': len(self.trigger_history),
            'level_1_triggers': level_counts[CircuitBreakerLevel.LEVEL_1],
            'level_2_triggers': level_counts[CircuitBreakerLevel.LEVEL_2],
            'level_3_triggers': level_counts[CircuitBreakerLevel.LEVEL_3],
            'avg_halt_duration_minutes': avg_duration,
            'triggers_by_type': type_counts
        }

    def reset(self):
        """Reset circuit breaker system"""
        self.trading_status = TradingStatus.ACTIVE
        self.active_level = CircuitBreakerLevel.NONE
        self.trigger_history = []
        self.current_halt = None
        self.daily_triggers = 0
        self.recent_losses = []
        self.consecutive_losses = 0

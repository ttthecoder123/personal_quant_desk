"""
Execution Risk Monitoring

Tracks and analyzes execution quality:
- Slippage statistics and distribution
- Market impact models (linear and square-root)
- Implementation shortfall calculation
- Failed execution tracking with retry logic
- Partial fill monitoring and completion rates
- Execution cost analysis (total, per trade, per instrument)
- Detailed execution reports with cost breakdown
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from loguru import logger


class ExecutionStatus(Enum):
    """Execution status types"""
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"
    CANCELLED = "cancelled"


class SlippageType(Enum):
    """Types of slippage"""
    FAVORABLE = "favorable"
    ADVERSE = "adverse"
    NEUTRAL = "neutral"


@dataclass
class ExecutionRecord:
    """Record of a single execution"""
    timestamp: datetime
    instrument: str
    side: str  # 'buy' or 'sell'
    intended_price: float
    executed_price: float
    intended_quantity: float
    executed_quantity: float
    slippage: float  # In price units
    slippage_bps: float  # In basis points
    market_impact_bps: float
    execution_cost: float
    status: ExecutionStatus
    retry_count: int = 0
    latency_ms: Optional[float] = None


@dataclass
class SlippageStats:
    """Statistical summary of slippage"""
    timestamp: datetime
    total_trades: int
    avg_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    min_slippage_bps: float
    std_slippage_bps: float
    favorable_count: int
    adverse_count: int
    neutral_count: int
    percentile_25: float
    percentile_75: float
    percentile_95: float


@dataclass
class MarketImpact:
    """Market impact analysis"""
    timestamp: datetime
    instrument: str
    trade_size: float
    avg_daily_volume: float
    participation_rate: float
    linear_impact_bps: float
    sqrt_impact_bps: float
    actual_impact_bps: float
    impact_ratio: float  # actual / predicted


@dataclass
class ImplementationShortfall:
    """Implementation shortfall metrics"""
    timestamp: datetime
    instrument: str
    decision_price: float
    executed_price: float
    quantity: float
    side: str
    shortfall_bps: float
    shortfall_cost: float
    timing_cost: float
    market_impact_cost: float
    opportunity_cost: float


@dataclass
class ExecutionStats:
    """Comprehensive execution statistics"""
    timestamp: datetime
    total_executions: int
    successful_executions: int
    failed_executions: int
    partial_executions: int
    success_rate: float
    avg_completion_rate: float
    total_execution_cost: float
    avg_cost_per_trade: float
    total_slippage_bps: float
    avg_slippage_bps: float
    total_market_impact_bps: float
    avg_latency_ms: Optional[float] = None


@dataclass
class ExecutionAlert:
    """Execution risk alert"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    alert_type: str
    message: str
    instrument: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ExecutionReport:
    """Comprehensive execution report"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    stats: ExecutionStats
    slippage_stats: SlippageStats
    recent_failures: List[ExecutionRecord]
    alerts: List[ExecutionAlert]
    cost_breakdown: Dict[str, float]
    instrument_stats: Dict[str, Dict]


@dataclass
class ExecutionRiskConfig:
    """Configuration for execution risk monitoring"""
    # Slippage thresholds (in bps)
    max_acceptable_slippage_bps: float = 10.0
    critical_slippage_bps: float = 25.0

    # Market impact parameters
    linear_impact_coeff: float = 0.1  # bps per 1% of ADV
    sqrt_impact_coeff: float = 0.5  # bps coefficient for sqrt model

    # Execution quality thresholds
    min_success_rate: float = 0.95
    min_completion_rate: float = 0.90
    max_retry_count: int = 3

    # Cost thresholds
    max_cost_per_trade_bps: float = 15.0
    max_daily_cost_pct: float = 0.001  # 0.1% of portfolio

    # Monitoring window
    monitoring_window_trades: int = 100
    alert_lookback_hours: int = 24


class ExecutionRisk:
    """
    Monitor and analyze execution risk

    Features:
    - Real-time slippage tracking
    - Market impact estimation
    - Implementation shortfall calculation
    - Execution failure analysis
    - Cost attribution
    - Alert generation
    """

    def __init__(self, config: Optional[ExecutionRiskConfig] = None):
        """
        Initialize execution risk monitor

        Args:
            config: Execution risk configuration (optional)
        """
        self.config = config if config is not None else ExecutionRiskConfig()

        # Execution history
        self.executions: deque = deque(maxlen=self.config.monitoring_window_trades)
        self.failed_executions: List[ExecutionRecord] = []

        # Market data cache (for impact estimation)
        self.avg_daily_volumes: Dict[str, float] = {}

        # Alert tracking
        self.alerts: List[ExecutionAlert] = []

        logger.info("ExecutionRisk monitor initialized")

    def record_execution(
        self,
        timestamp: datetime,
        instrument: str,
        side: str,
        intended_price: float,
        executed_price: float,
        intended_quantity: float,
        executed_quantity: float,
        status: ExecutionStatus = ExecutionStatus.COMPLETED,
        retry_count: int = 0,
        latency_ms: Optional[float] = None
    ) -> ExecutionRecord:
        """
        Record an execution and calculate metrics

        Args:
            timestamp: Execution timestamp
            instrument: Instrument symbol
            side: 'buy' or 'sell'
            intended_price: Price at order decision
            executed_price: Actual execution price
            intended_quantity: Intended quantity
            executed_quantity: Actual executed quantity
            status: Execution status
            retry_count: Number of retry attempts
            latency_ms: Execution latency in milliseconds

        Returns:
            ExecutionRecord with calculated metrics
        """
        # Calculate slippage
        slippage = self._calculate_slippage(
            side, intended_price, executed_price
        )
        slippage_bps = (slippage / intended_price) * 10000

        # Estimate market impact
        market_impact_bps = self._estimate_market_impact(
            instrument, executed_quantity, side
        )

        # Calculate execution cost
        execution_cost = abs(slippage * executed_quantity)

        # Create execution record
        record = ExecutionRecord(
            timestamp=timestamp,
            instrument=instrument,
            side=side,
            intended_price=intended_price,
            executed_price=executed_price,
            intended_quantity=intended_quantity,
            executed_quantity=executed_quantity,
            slippage=slippage,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            execution_cost=execution_cost,
            status=status,
            retry_count=retry_count,
            latency_ms=latency_ms
        )

        # Store execution
        self.executions.append(record)

        # Track failures
        if status == ExecutionStatus.FAILED:
            self.failed_executions.append(record)

        # Check for alerts
        self._check_execution_alerts(record)

        logger.info(
            f"Recorded execution: {instrument} {side} "
            f"slippage={slippage_bps:.2f}bps status={status.value}"
        )

        return record

    def _calculate_slippage(
        self,
        side: str,
        intended_price: float,
        executed_price: float
    ) -> float:
        """
        Calculate slippage (positive = adverse, negative = favorable)

        For buys: slippage = executed - intended (positive = paid more)
        For sells: slippage = intended - executed (positive = received less)
        """
        if side.lower() == 'buy':
            return executed_price - intended_price
        else:  # sell
            return intended_price - executed_price

    def _estimate_market_impact(
        self,
        instrument: str,
        quantity: float,
        side: str
    ) -> float:
        """
        Estimate market impact in basis points

        Uses both linear and square-root models
        """
        # Get average daily volume (use default if not available)
        adv = self.avg_daily_volumes.get(instrument, 1000000)

        # Calculate participation rate
        participation_rate = abs(quantity) / adv

        # Linear model: impact = coeff * participation_rate
        linear_impact = self.config.linear_impact_coeff * participation_rate * 10000

        # Square-root model: impact = coeff * sqrt(participation_rate)
        sqrt_impact = self.config.sqrt_impact_coeff * np.sqrt(participation_rate) * 10000

        # Use average of both models
        estimated_impact = (linear_impact + sqrt_impact) / 2

        return estimated_impact

    def calculate_market_impact_detailed(
        self,
        instrument: str,
        trade_size: float,
        avg_daily_volume: float,
        actual_slippage_bps: float
    ) -> MarketImpact:
        """
        Calculate detailed market impact analysis

        Args:
            instrument: Instrument symbol
            trade_size: Trade size (quantity)
            avg_daily_volume: Average daily volume
            actual_slippage_bps: Actual observed slippage

        Returns:
            MarketImpact analysis
        """
        participation_rate = abs(trade_size) / avg_daily_volume

        # Linear model
        linear_impact = self.config.linear_impact_coeff * participation_rate * 10000

        # Square-root model
        sqrt_impact = self.config.sqrt_impact_coeff * np.sqrt(participation_rate) * 10000

        # Impact ratio (actual / predicted)
        predicted_impact = (linear_impact + sqrt_impact) / 2
        impact_ratio = actual_slippage_bps / predicted_impact if predicted_impact > 0 else 1.0

        return MarketImpact(
            timestamp=datetime.now(),
            instrument=instrument,
            trade_size=trade_size,
            avg_daily_volume=avg_daily_volume,
            participation_rate=participation_rate,
            linear_impact_bps=linear_impact,
            sqrt_impact_bps=sqrt_impact,
            actual_impact_bps=actual_slippage_bps,
            impact_ratio=impact_ratio
        )

    def calculate_implementation_shortfall(
        self,
        instrument: str,
        decision_price: float,
        executed_price: float,
        quantity: float,
        side: str,
        decision_time: datetime,
        execution_time: datetime
    ) -> ImplementationShortfall:
        """
        Calculate implementation shortfall

        Args:
            instrument: Instrument symbol
            decision_price: Price at decision time
            executed_price: Actual execution price
            quantity: Quantity executed
            side: 'buy' or 'sell'
            decision_time: Time of trading decision
            execution_time: Time of execution

        Returns:
            ImplementationShortfall metrics
        """
        # Calculate shortfall
        if side.lower() == 'buy':
            shortfall = executed_price - decision_price
        else:  # sell
            shortfall = decision_price - executed_price

        shortfall_bps = (shortfall / decision_price) * 10000
        shortfall_cost = shortfall * quantity

        # Estimate components (simplified)
        # In practice, these would require more market data
        timing_cost = shortfall_cost * 0.3  # 30% attributed to timing
        market_impact_cost = shortfall_cost * 0.5  # 50% to market impact
        opportunity_cost = shortfall_cost * 0.2  # 20% to opportunity cost

        return ImplementationShortfall(
            timestamp=execution_time,
            instrument=instrument,
            decision_price=decision_price,
            executed_price=executed_price,
            quantity=quantity,
            side=side,
            shortfall_bps=shortfall_bps,
            shortfall_cost=shortfall_cost,
            timing_cost=timing_cost,
            market_impact_cost=market_impact_cost,
            opportunity_cost=opportunity_cost
        )

    def get_slippage_statistics(
        self,
        lookback_trades: Optional[int] = None
    ) -> SlippageStats:
        """
        Calculate slippage statistics

        Args:
            lookback_trades: Number of recent trades to analyze

        Returns:
            SlippageStats summary
        """
        if not self.executions:
            return SlippageStats(
                timestamp=datetime.now(),
                total_trades=0,
                avg_slippage_bps=0.0,
                median_slippage_bps=0.0,
                max_slippage_bps=0.0,
                min_slippage_bps=0.0,
                std_slippage_bps=0.0,
                favorable_count=0,
                adverse_count=0,
                neutral_count=0,
                percentile_25=0.0,
                percentile_75=0.0,
                percentile_95=0.0
            )

        # Get recent executions
        recent_executions = list(self.executions)
        if lookback_trades:
            recent_executions = recent_executions[-lookback_trades:]

        # Extract slippage values
        slippage_bps = [e.slippage_bps for e in recent_executions]

        # Count slippage types
        favorable = sum(1 for s in slippage_bps if s < -1)
        adverse = sum(1 for s in slippage_bps if s > 1)
        neutral = len(slippage_bps) - favorable - adverse

        return SlippageStats(
            timestamp=datetime.now(),
            total_trades=len(slippage_bps),
            avg_slippage_bps=np.mean(slippage_bps),
            median_slippage_bps=np.median(slippage_bps),
            max_slippage_bps=np.max(slippage_bps),
            min_slippage_bps=np.min(slippage_bps),
            std_slippage_bps=np.std(slippage_bps),
            favorable_count=favorable,
            adverse_count=adverse,
            neutral_count=neutral,
            percentile_25=np.percentile(slippage_bps, 25),
            percentile_75=np.percentile(slippage_bps, 75),
            percentile_95=np.percentile(slippage_bps, 95)
        )

    def get_execution_statistics(self) -> ExecutionStats:
        """
        Get comprehensive execution statistics

        Returns:
            ExecutionStats summary
        """
        if not self.executions:
            return ExecutionStats(
                timestamp=datetime.now(),
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                partial_executions=0,
                success_rate=0.0,
                avg_completion_rate=0.0,
                total_execution_cost=0.0,
                avg_cost_per_trade=0.0,
                total_slippage_bps=0.0,
                avg_slippage_bps=0.0,
                total_market_impact_bps=0.0
            )

        executions = list(self.executions)

        # Count statuses
        successful = sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for e in executions if e.status == ExecutionStatus.FAILED)
        partial = sum(1 for e in executions if e.status == ExecutionStatus.PARTIAL)

        # Calculate rates
        success_rate = successful / len(executions) if executions else 0.0

        # Completion rates
        completion_rates = [
            e.executed_quantity / e.intended_quantity
            for e in executions
            if e.intended_quantity > 0
        ]
        avg_completion_rate = np.mean(completion_rates) if completion_rates else 0.0

        # Costs
        total_cost = sum(e.execution_cost for e in executions)
        avg_cost = total_cost / len(executions)

        # Slippage
        total_slippage_bps = sum(abs(e.slippage_bps) for e in executions)
        avg_slippage_bps = total_slippage_bps / len(executions)

        # Market impact
        total_impact_bps = sum(e.market_impact_bps for e in executions)

        # Latency
        latencies = [e.latency_ms for e in executions if e.latency_ms is not None]
        avg_latency = np.mean(latencies) if latencies else None

        return ExecutionStats(
            timestamp=datetime.now(),
            total_executions=len(executions),
            successful_executions=successful,
            failed_executions=failed,
            partial_executions=partial,
            success_rate=success_rate,
            avg_completion_rate=avg_completion_rate,
            total_execution_cost=total_cost,
            avg_cost_per_trade=avg_cost,
            total_slippage_bps=total_slippage_bps,
            avg_slippage_bps=avg_slippage_bps,
            total_market_impact_bps=total_impact_bps,
            avg_latency_ms=avg_latency
        )

    def get_instrument_statistics(self) -> Dict[str, Dict]:
        """
        Get per-instrument execution statistics

        Returns:
            Dictionary mapping instrument to statistics
        """
        if not self.executions:
            return {}

        instrument_data = {}

        for execution in self.executions:
            instrument = execution.instrument

            if instrument not in instrument_data:
                instrument_data[instrument] = []

            instrument_data[instrument].append(execution)

        # Calculate stats per instrument
        instrument_stats = {}

        for instrument, executions in instrument_data.items():
            slippage_bps = [e.slippage_bps for e in executions]
            costs = [e.execution_cost for e in executions]

            instrument_stats[instrument] = {
                'total_trades': len(executions),
                'avg_slippage_bps': np.mean(slippage_bps),
                'total_cost': sum(costs),
                'success_rate': sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED) / len(executions)
            }

        return instrument_stats

    def _check_execution_alerts(self, execution: ExecutionRecord):
        """Check execution for alert conditions"""
        # Critical slippage
        if abs(execution.slippage_bps) > self.config.critical_slippage_bps:
            self.alerts.append(ExecutionAlert(
                timestamp=execution.timestamp,
                severity='critical',
                alert_type='high_slippage',
                message=f"Critical slippage on {execution.instrument}: {execution.slippage_bps:.2f} bps",
                instrument=execution.instrument,
                value=execution.slippage_bps,
                threshold=self.config.critical_slippage_bps
            ))

        # High slippage warning
        elif abs(execution.slippage_bps) > self.config.max_acceptable_slippage_bps:
            self.alerts.append(ExecutionAlert(
                timestamp=execution.timestamp,
                severity='warning',
                alert_type='high_slippage',
                message=f"High slippage on {execution.instrument}: {execution.slippage_bps:.2f} bps",
                instrument=execution.instrument,
                value=execution.slippage_bps,
                threshold=self.config.max_acceptable_slippage_bps
            ))

        # Execution failure
        if execution.status == ExecutionStatus.FAILED:
            self.alerts.append(ExecutionAlert(
                timestamp=execution.timestamp,
                severity='critical',
                alert_type='execution_failure',
                message=f"Execution failed for {execution.instrument} after {execution.retry_count} retries",
                instrument=execution.instrument,
                value=float(execution.retry_count),
                threshold=float(self.config.max_retry_count)
            ))

        # Partial fill
        if execution.status == ExecutionStatus.PARTIAL:
            completion_rate = execution.executed_quantity / execution.intended_quantity
            if completion_rate < self.config.min_completion_rate:
                self.alerts.append(ExecutionAlert(
                    timestamp=execution.timestamp,
                    severity='warning',
                    alert_type='low_completion',
                    message=f"Low completion rate on {execution.instrument}: {completion_rate:.1%}",
                    instrument=execution.instrument,
                    value=completion_rate,
                    threshold=self.config.min_completion_rate
                ))

    def generate_execution_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ExecutionReport:
        """
        Generate comprehensive execution report

        Args:
            period_start: Report period start (optional)
            period_end: Report period end (optional)

        Returns:
            ExecutionReport with detailed metrics
        """
        # Default to last 24 hours if not specified
        if period_end is None:
            period_end = datetime.now()
        if period_start is None:
            period_start = period_end - timedelta(hours=24)

        # Filter executions by period
        period_executions = [
            e for e in self.executions
            if period_start <= e.timestamp <= period_end
        ]

        # Get statistics
        stats = self.get_execution_statistics()
        slippage_stats = self.get_slippage_statistics()
        instrument_stats = self.get_instrument_statistics()

        # Recent failures
        recent_failures = [
            e for e in self.failed_executions[-10:]
            if period_start <= e.timestamp <= period_end
        ]

        # Filter recent alerts
        recent_alerts = [
            a for a in self.alerts
            if period_start <= a.timestamp <= period_end
        ]

        # Cost breakdown
        cost_breakdown = {
            'total_execution_cost': stats.total_execution_cost,
            'slippage_cost': sum(e.execution_cost for e in period_executions),
            'market_impact_cost': sum(
                e.market_impact_bps * e.executed_price * e.executed_quantity / 10000
                for e in period_executions
            ),
            'avg_cost_per_trade': stats.avg_cost_per_trade
        }

        return ExecutionReport(
            timestamp=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            stats=stats,
            slippage_stats=slippage_stats,
            recent_failures=recent_failures,
            alerts=recent_alerts,
            cost_breakdown=cost_breakdown,
            instrument_stats=instrument_stats
        )

    def update_avg_daily_volume(self, instrument: str, volume: float):
        """Update average daily volume for an instrument"""
        self.avg_daily_volumes[instrument] = volume

    def get_recent_alerts(self, hours: int = 24) -> List[ExecutionAlert]:
        """Get recent alerts within specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a.timestamp >= cutoff]

    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        self.alerts = [a for a in self.alerts if a.timestamp >= cutoff]

    def get_failed_executions(
        self,
        instrument: Optional[str] = None,
        hours: Optional[int] = None
    ) -> List[ExecutionRecord]:
        """
        Get failed executions with optional filters

        Args:
            instrument: Filter by instrument (optional)
            hours: Lookback hours (optional)

        Returns:
            List of failed executions
        """
        failures = self.failed_executions

        if instrument:
            failures = [f for f in failures if f.instrument == instrument]

        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            failures = [f for f in failures if f.timestamp >= cutoff]

        return failures

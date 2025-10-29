"""
Main Risk Management Engine

Orchestrates all risk subsystems:
- Pre-trade and post-trade risk checks
- Real-time risk monitoring
- Risk limit enforcement
- Position adjustments
- Emergency protocols
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .risk_metrics import RiskMetrics, RiskMetricsResult
from .var_models import VaRModels, VaRResult
from .stress_testing import StressTester, StressTestResult


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """Risk management actions"""
    ALLOW = "allow"
    WARN = "warn"
    REDUCE = "reduce"
    BLOCK = "block"
    FLATTEN = "flatten"


@dataclass
class RiskLimits:
    """Portfolio risk limits configuration"""
    max_var_95: float = 0.02  # 2% daily VaR limit
    max_drawdown: float = 0.20  # 20% maximum drawdown
    target_volatility: float = 0.20  # 20% annual volatility
    max_leverage: float = 2.0  # 2x maximum leverage
    max_position_size: float = 0.02  # 2% risk per position
    max_concentration: float = 0.20  # 20% max per instrument
    max_correlation: float = 0.85  # Maximum position correlation
    min_effective_bets: int = 3  # Minimum diversification
    max_slippage_bps: int = 10  # Maximum slippage in basis points
    max_daily_turnover: float = 0.50  # 50% max daily turnover


@dataclass
class RiskState:
    """Current portfolio risk state"""
    timestamp: datetime
    portfolio_value: float
    leverage: float
    volatility: float
    var_95: float
    drawdown: float
    max_drawdown: float
    position_count: int
    effective_bets: float
    largest_position_pct: float
    violations: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class PreTradeCheckResult:
    """Result of pre-trade risk check"""
    allowed: bool
    action: RiskAction
    violations: List[str]
    recommended_size: Optional[float] = None
    message: str = ""


class RiskEngine:
    """
    Main risk management engine

    Coordinates all risk subsystems and enforces limits
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize risk engine

        Args:
            limits: Risk limits configuration
            initial_capital: Initial portfolio capital
        """
        self.limits = limits if limits is not None else RiskLimits()
        self.initial_capital = initial_capital

        # Initialize subsystems
        self.risk_metrics = RiskMetrics()
        self.var_models = VaRModels()
        self.stress_tester = StressTester()

        # Risk state
        self.current_state: Optional[RiskState] = None
        self.risk_history: List[RiskState] = []

        # Emergency overrides
        self.emergency_mode = False
        self.trading_halted = False

        # Logging
        self.logger = logging.getLogger(__name__)

    def check_pre_trade_risk(
        self,
        symbol: str,
        order_size: float,
        order_price: float,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> PreTradeCheckResult:
        """
        Pre-trade risk check before order execution

        Args:
            symbol: Symbol to trade
            order_size: Size of order (signed: positive=buy, negative=sell)
            order_price: Expected execution price
            current_positions: Current positions
            current_prices: Current market prices
            portfolio_value: Current portfolio value

        Returns:
            PreTradeCheckResult
        """
        violations = []

        # Check if trading is halted
        if self.trading_halted:
            return PreTradeCheckResult(
                allowed=False,
                action=RiskAction.BLOCK,
                violations=["Trading halted due to emergency"],
                message="Trading is currently halted"
            )

        # Calculate order value
        order_value = abs(order_size * order_price)

        # Check position size limit
        position_size_pct = order_value / portfolio_value
        if position_size_pct > self.limits.max_position_size:
            violations.append(
                f"Position size {position_size_pct:.2%} exceeds limit {self.limits.max_position_size:.2%}"
            )

        # Calculate new position
        new_positions = current_positions.copy()
        new_positions[symbol] = new_positions.get(symbol, 0) + order_size

        # Check concentration limit
        new_position_value = abs(new_positions[symbol] * current_prices.get(symbol, order_price))
        concentration = new_position_value / portfolio_value
        if concentration > self.limits.max_concentration:
            violations.append(
                f"Concentration {concentration:.2%} exceeds limit {self.limits.max_concentration:.2%}"
            )

        # Calculate new leverage
        total_exposure = sum(
            abs(pos * current_prices.get(sym, 0))
            for sym, pos in new_positions.items()
        )
        new_leverage = total_exposure / portfolio_value
        if new_leverage > self.limits.max_leverage:
            violations.append(
                f"Leverage {new_leverage:.2f}x exceeds limit {self.limits.max_leverage:.2f}x"
            )

        # Determine action
        if len(violations) == 0:
            return PreTradeCheckResult(
                allowed=True,
                action=RiskAction.ALLOW,
                violations=[],
                message="Order approved"
            )
        elif len(violations) <= 1 and position_size_pct < self.limits.max_position_size * 1.5:
            # Minor violation - reduce size
            recommended_size = order_size * (self.limits.max_position_size / position_size_pct)
            return PreTradeCheckResult(
                allowed=False,
                action=RiskAction.REDUCE,
                violations=violations,
                recommended_size=recommended_size,
                message=f"Order too large, recommend size: {recommended_size:.2f}"
            )
        else:
            # Major violation - block
            return PreTradeCheckResult(
                allowed=False,
                action=RiskAction.BLOCK,
                violations=violations,
                message="Order blocked due to risk violations"
            )

    def update_risk_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> RiskState:
        """
        Update all risk metrics and return current risk state

        Args:
            returns: Return series
            equity_curve: Equity curve
            current_positions: Current positions
            current_prices: Current prices
            portfolio_value: Current portfolio value

        Returns:
            Updated RiskState
        """
        # Calculate core metrics
        metrics = self.risk_metrics.calculate_all_metrics(
            returns=returns,
            equity_curve=equity_curve
        )

        # Calculate leverage
        total_exposure = sum(
            abs(pos * current_prices.get(sym, 0))
            for sym, pos in current_positions.items()
        )
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Calculate effective number of bets (simplified)
        position_count = sum(1 for pos in current_positions.values() if abs(pos) > 0)

        # Find largest position
        position_values = [
            abs(pos * current_prices.get(sym, 0)) / portfolio_value
            for sym, pos in current_positions.items()
        ]
        largest_position_pct = max(position_values) if position_values else 0

        # Check for violations
        violations = []

        if metrics.var_95 > self.limits.max_var_95:
            violations.append(f"VaR {metrics.var_95:.2%} exceeds limit {self.limits.max_var_95:.2%}")

        if abs(metrics.current_drawdown) > self.limits.max_drawdown:
            violations.append(f"Drawdown {abs(metrics.current_drawdown):.2%} exceeds limit {self.limits.max_drawdown:.2%}")

        if metrics.portfolio_volatility > self.limits.target_volatility * 1.5:
            violations.append(f"Volatility {metrics.portfolio_volatility:.2%} too high (target {self.limits.target_volatility:.2%})")

        if leverage > self.limits.max_leverage:
            violations.append(f"Leverage {leverage:.2f}x exceeds limit {self.limits.max_leverage:.2f}x")

        if largest_position_pct > self.limits.max_concentration:
            violations.append(f"Largest position {largest_position_pct:.2%} exceeds concentration limit {self.limits.max_concentration:.2%}")

        # Determine risk level
        if len(violations) == 0:
            risk_level = RiskLevel.LOW
        elif len(violations) <= 1 or abs(metrics.current_drawdown) < self.limits.max_drawdown * 0.5:
            risk_level = RiskLevel.MEDIUM
        elif abs(metrics.current_drawdown) < self.limits.max_drawdown * 0.75:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        # Create risk state
        state = RiskState(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            leverage=leverage,
            volatility=metrics.portfolio_volatility,
            var_95=metrics.var_95,
            drawdown=metrics.current_drawdown,
            max_drawdown=metrics.max_drawdown,
            position_count=position_count,
            effective_bets=float(position_count),  # Simplified
            largest_position_pct=largest_position_pct,
            violations=violations,
            risk_level=risk_level
        )

        # Update current state and history
        self.current_state = state
        self.risk_history.append(state)

        return state

    def adjust_positions(
        self,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        target_volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted position sizes

        Args:
            current_positions: Current positions
            current_prices: Current prices
            portfolio_value: Current portfolio value
            target_volatility: Override target volatility

        Returns:
            Dictionary of recommended position adjustments
        """
        target_vol = target_volatility if target_volatility else self.limits.target_volatility
        adjustments = {}

        if self.current_state is None:
            return adjustments

        # If in emergency mode or critical risk, recommend flattening
        if self.emergency_mode or self.current_state.risk_level == RiskLevel.CRITICAL:
            for symbol in current_positions:
                adjustments[symbol] = -current_positions[symbol]  # Close all
            return adjustments

        # If volatility too high, scale down proportionally
        current_vol = self.current_state.volatility
        if current_vol > target_vol * 1.2:
            scale_factor = target_vol / current_vol

            for symbol, pos in current_positions.items():
                adjustment = pos * (scale_factor - 1)
                if abs(adjustment) > 0.01:  # Only significant adjustments
                    adjustments[symbol] = adjustment

        # If drawdown too high, reduce positions
        elif abs(self.current_state.drawdown) > self.limits.max_drawdown * 0.5:
            # Graduated response
            if abs(self.current_state.drawdown) > self.limits.max_drawdown * 0.75:
                # 75%+ drawdown: reduce by 50%
                scale_factor = 0.5
            else:
                # 50-75% drawdown: reduce by 25%
                scale_factor = 0.75

            for symbol, pos in current_positions.items():
                adjustment = pos * (scale_factor - 1)
                if abs(adjustment) > 0.01:
                    adjustments[symbol] = adjustment

        return adjustments

    def trigger_risk_action(
        self,
        risk_state: RiskState
    ) -> RiskAction:
        """
        Determine and trigger appropriate risk action

        Args:
            risk_state: Current risk state

        Returns:
            RiskAction taken
        """
        if risk_state.risk_level == RiskLevel.LOW:
            return RiskAction.ALLOW

        elif risk_state.risk_level == RiskLevel.MEDIUM:
            self.logger.warning(f"Medium risk level: {risk_state.violations}")
            return RiskAction.WARN

        elif risk_state.risk_level == RiskLevel.HIGH:
            self.logger.error(f"High risk level: {risk_state.violations}")
            return RiskAction.REDUCE

        else:  # CRITICAL
            self.logger.critical(f"Critical risk level: {risk_state.violations}")
            self.trading_halted = True
            return RiskAction.FLATTEN

    def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report

        Returns:
            Dictionary with risk report data
        """
        if self.current_state is None:
            return {"error": "No risk state available"}

        report = {
            'timestamp': self.current_state.timestamp,
            'risk_level': self.current_state.risk_level.value,
            'portfolio_value': self.current_state.portfolio_value,
            'leverage': self.current_state.leverage,
            'volatility': self.current_state.volatility,
            'var_95': self.current_state.var_95,
            'drawdown': self.current_state.drawdown,
            'max_drawdown': self.current_state.max_drawdown,
            'position_count': self.current_state.position_count,
            'effective_bets': self.current_state.effective_bets,
            'largest_position_pct': self.current_state.largest_position_pct,
            'violations': self.current_state.violations,
            'trading_halted': self.trading_halted,
            'emergency_mode': self.emergency_mode,
            'limits': {
                'max_var_95': self.limits.max_var_95,
                'max_drawdown': self.limits.max_drawdown,
                'target_volatility': self.limits.target_volatility,
                'max_leverage': self.limits.max_leverage,
                'max_position_size': self.limits.max_position_size,
                'max_concentration': self.limits.max_concentration
            }
        }

        return report

    def enable_emergency_mode(self):
        """Enable emergency risk mode"""
        self.emergency_mode = True
        self.trading_halted = True
        self.logger.critical("EMERGENCY MODE ENABLED - Trading halted")

    def disable_emergency_mode(self):
        """Disable emergency mode (requires manual intervention)"""
        self.emergency_mode = False
        self.trading_halted = False
        self.logger.info("Emergency mode disabled")

    def get_risk_summary(self) -> str:
        """
        Get human-readable risk summary

        Returns:
            Formatted risk summary string
        """
        if self.current_state is None:
            return "No risk data available"

        summary = f"""
Risk Summary - {self.current_state.timestamp}
{'=' * 50}
Risk Level: {self.current_state.risk_level.value.upper()}
Portfolio Value: ${self.current_state.portfolio_value:,.2f}
Leverage: {self.current_state.leverage:.2f}x
Volatility: {self.current_state.volatility:.2%} (Target: {self.limits.target_volatility:.2%})
VaR (95%): {self.current_state.var_95:.2%} (Limit: {self.limits.max_var_95:.2%})
Drawdown: {self.current_state.drawdown:.2%} (Max: {self.limits.max_drawdown:.2%})
Position Count: {self.current_state.position_count}
Largest Position: {self.current_state.largest_position_pct:.2%}

Status:
- Trading Halted: {self.trading_halted}
- Emergency Mode: {self.emergency_mode}

Violations ({len(self.current_state.violations)}):
"""
        for v in self.current_state.violations:
            summary += f"  - {v}\n"

        return summary

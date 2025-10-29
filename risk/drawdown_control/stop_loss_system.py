"""
Stop Loss System

Implements comprehensive stop loss mechanisms:
- Trailing stop losses with dynamic adjustment
- Volatility-based stop positioning
- Time-based stops (triple barrier method)
- Correlated asset stop triggers
- Portfolio-level stop loss
- Strategy-specific stops
- Stop loss optimization and backtesting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class StopType(Enum):
    """Types of stop losses"""
    FIXED = "fixed"                    # Fixed percentage stop
    TRAILING = "trailing"              # Trailing stop
    VOLATILITY = "volatility"          # Volatility-based stop
    TIME_BASED = "time_based"          # Time-based stop (triple barrier)
    CORRELATION = "correlation"        # Correlation breakdown stop
    PORTFOLIO = "portfolio"            # Portfolio-level stop


class StopStatus(Enum):
    """Stop loss status"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class StopLossOrder:
    """Stop loss order details"""
    symbol: str
    stop_type: StopType
    entry_price: float
    stop_price: float
    current_price: float
    stop_distance: float  # As percentage
    creation_time: datetime
    last_update_time: datetime
    status: StopStatus
    position_size: float
    max_loss_amount: float
    trailing_high: Optional[float] = None  # For trailing stops
    volatility_multiplier: Optional[float] = None  # For volatility stops
    time_limit: Optional[datetime] = None  # For time-based stops
    correlation_threshold: Optional[float] = None  # For correlation stops


@dataclass
class TripleBarrier:
    """Triple barrier parameters (de Prado)"""
    upper_barrier: float  # Profit target
    lower_barrier: float  # Stop loss
    time_barrier: datetime  # Maximum holding period
    entry_price: float
    entry_time: datetime


@dataclass
class StopLossResult:
    """Result of stop loss check"""
    symbol: str
    should_exit: bool
    exit_reason: str
    stop_price: float
    current_price: float
    loss_amount: float
    loss_percentage: float
    recommended_exit_size: float  # Can be partial


class StopLossSystem:
    """
    Comprehensive stop loss management system

    Features:
    - Multiple stop loss types
    - Dynamic adjustment based on market conditions
    - Portfolio-level risk control
    - Correlation-based stops
    - Optimization and backtesting support
    """

    def __init__(
        self,
        default_stop_pct: float = 0.02,
        trailing_activation_pct: float = 0.01,
        volatility_multiplier: float = 2.0,
        max_holding_days: int = 30,
        correlation_threshold: float = 0.5,
        portfolio_stop_pct: float = 0.10
    ):
        """
        Initialize stop loss system

        Args:
            default_stop_pct: Default stop loss percentage (default 2%)
            trailing_activation_pct: Profit level to activate trailing stop (default 1%)
            volatility_multiplier: Multiplier for volatility-based stops (default 2.0 ATR)
            max_holding_days: Maximum holding period for time-based stops (default 30)
            correlation_threshold: Correlation breakdown threshold (default 0.5)
            portfolio_stop_pct: Portfolio-level stop loss (default 10%)
        """
        self.default_stop_pct = default_stop_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.volatility_multiplier = volatility_multiplier
        self.max_holding_days = max_holding_days
        self.correlation_threshold = correlation_threshold
        self.portfolio_stop_pct = portfolio_stop_pct

        # Active stop orders
        self.active_stops: Dict[str, StopLossOrder] = {}
        self.triggered_stops: List[StopLossOrder] = []

    def create_fixed_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        stop_pct: Optional[float] = None
    ) -> StopLossOrder:
        """
        Create fixed percentage stop loss

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            current_price: Current price
            position_size: Position size
            stop_pct: Stop loss percentage (optional, uses default)

        Returns:
            StopLossOrder
        """
        stop_percentage = stop_pct if stop_pct is not None else self.default_stop_pct
        stop_price = entry_price * (1 - stop_percentage)
        max_loss = position_size * (entry_price - stop_price)

        stop = StopLossOrder(
            symbol=symbol,
            stop_type=StopType.FIXED,
            entry_price=entry_price,
            stop_price=stop_price,
            current_price=current_price,
            stop_distance=stop_percentage,
            creation_time=datetime.now(),
            last_update_time=datetime.now(),
            status=StopStatus.ACTIVE,
            position_size=position_size,
            max_loss_amount=max_loss
        )

        self.active_stops[symbol] = stop
        return stop

    def create_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        trailing_pct: Optional[float] = None
    ) -> StopLossOrder:
        """
        Create trailing stop loss

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            current_price: Current price
            position_size: Position size
            trailing_pct: Trailing distance percentage (optional)

        Returns:
            StopLossOrder
        """
        trail_pct = trailing_pct if trailing_pct is not None else self.default_stop_pct
        stop_price = current_price * (1 - trail_pct)
        max_loss = position_size * (entry_price - stop_price)

        stop = StopLossOrder(
            symbol=symbol,
            stop_type=StopType.TRAILING,
            entry_price=entry_price,
            stop_price=stop_price,
            current_price=current_price,
            stop_distance=trail_pct,
            creation_time=datetime.now(),
            last_update_time=datetime.now(),
            status=StopStatus.ACTIVE,
            position_size=position_size,
            max_loss_amount=max_loss,
            trailing_high=current_price
        )

        self.active_stops[symbol] = stop
        return stop

    def create_volatility_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        atr: float,
        multiplier: Optional[float] = None
    ) -> StopLossOrder:
        """
        Create volatility-based stop loss (ATR-based)

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            current_price: Current price
            position_size: Position size
            atr: Average True Range
            multiplier: ATR multiplier (optional)

        Returns:
            StopLossOrder
        """
        mult = multiplier if multiplier is not None else self.volatility_multiplier
        stop_distance = mult * atr
        stop_price = entry_price - stop_distance
        stop_pct = stop_distance / entry_price
        max_loss = position_size * stop_distance

        stop = StopLossOrder(
            symbol=symbol,
            stop_type=StopType.VOLATILITY,
            entry_price=entry_price,
            stop_price=stop_price,
            current_price=current_price,
            stop_distance=stop_pct,
            creation_time=datetime.now(),
            last_update_time=datetime.now(),
            status=StopStatus.ACTIVE,
            position_size=position_size,
            max_loss_amount=max_loss,
            volatility_multiplier=mult
        )

        self.active_stops[symbol] = stop
        return stop

    def create_triple_barrier_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        profit_target_pct: float = 0.05,
        stop_loss_pct: Optional[float] = None,
        max_days: Optional[int] = None
    ) -> Tuple[StopLossOrder, TripleBarrier]:
        """
        Create triple barrier stop (de Prado method)

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            current_price: Current price
            position_size: Position size
            profit_target_pct: Upper barrier (profit target)
            stop_loss_pct: Lower barrier (stop loss)
            max_days: Time barrier (max holding period)

        Returns:
            Tuple of (StopLossOrder, TripleBarrier)
        """
        stop_pct = stop_loss_pct if stop_loss_pct is not None else self.default_stop_pct
        days = max_days if max_days is not None else self.max_holding_days

        upper_barrier = entry_price * (1 + profit_target_pct)
        lower_barrier = entry_price * (1 - stop_pct)
        time_barrier = datetime.now() + timedelta(days=days)

        max_loss = position_size * (entry_price - lower_barrier)

        stop = StopLossOrder(
            symbol=symbol,
            stop_type=StopType.TIME_BASED,
            entry_price=entry_price,
            stop_price=lower_barrier,
            current_price=current_price,
            stop_distance=stop_pct,
            creation_time=datetime.now(),
            last_update_time=datetime.now(),
            status=StopStatus.ACTIVE,
            position_size=position_size,
            max_loss_amount=max_loss,
            time_limit=time_barrier
        )

        triple_barrier = TripleBarrier(
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            time_barrier=time_barrier,
            entry_price=entry_price,
            entry_time=datetime.now()
        )

        self.active_stops[symbol] = stop
        return stop, triple_barrier

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[StopLossOrder]:
        """
        Update trailing stop based on new price

        Args:
            symbol: Asset symbol
            current_price: Current price

        Returns:
            Updated StopLossOrder or None if not found
        """
        if symbol not in self.active_stops:
            return None

        stop = self.active_stops[symbol]

        if stop.stop_type != StopType.TRAILING:
            return None

        # Update trailing high
        if stop.trailing_high is None or current_price > stop.trailing_high:
            stop.trailing_high = current_price

            # Recalculate stop price
            new_stop_price = stop.trailing_high * (1 - stop.stop_distance)

            # Only move stop up, never down
            if new_stop_price > stop.stop_price:
                stop.stop_price = new_stop_price
                stop.last_update_time = datetime.now()

        stop.current_price = current_price
        return stop

    def update_volatility_stop(
        self,
        symbol: str,
        current_price: float,
        new_atr: float
    ) -> Optional[StopLossOrder]:
        """
        Update volatility-based stop with new ATR

        Args:
            symbol: Asset symbol
            current_price: Current price
            new_atr: Updated ATR value

        Returns:
            Updated StopLossOrder or None if not found
        """
        if symbol not in self.active_stops:
            return None

        stop = self.active_stops[symbol]

        if stop.stop_type != StopType.VOLATILITY:
            return None

        # Recalculate stop price with new ATR
        if stop.volatility_multiplier is not None:
            stop_distance = stop.volatility_multiplier * new_atr
            new_stop_price = stop.entry_price - stop_distance

            # Update stop price (can move up or down based on volatility)
            stop.stop_price = new_stop_price
            stop.stop_distance = stop_distance / stop.entry_price
            stop.current_price = current_price
            stop.last_update_time = datetime.now()

        return stop

    def check_stop(
        self,
        symbol: str,
        current_price: float,
        current_time: Optional[datetime] = None
    ) -> StopLossResult:
        """
        Check if stop loss should be triggered

        Args:
            symbol: Asset symbol
            current_price: Current price
            current_time: Current timestamp (optional)

        Returns:
            StopLossResult with exit recommendation
        """
        if symbol not in self.active_stops:
            return StopLossResult(
                symbol=symbol,
                should_exit=False,
                exit_reason="",
                stop_price=0.0,
                current_price=current_price,
                loss_amount=0.0,
                loss_percentage=0.0,
                recommended_exit_size=0.0
            )

        stop = self.active_stops[symbol]
        timestamp = current_time if current_time is not None else datetime.now()

        # Update current price
        stop.current_price = current_price

        should_exit = False
        exit_reason = ""

        # Check price-based stop
        if current_price <= stop.stop_price:
            should_exit = True
            exit_reason = f"Price stop triggered: {current_price:.2f} <= {stop.stop_price:.2f}"

        # Check time-based stop
        if stop.time_limit is not None and timestamp >= stop.time_limit:
            should_exit = True
            exit_reason = f"Time stop triggered: holding period exceeded {self.max_holding_days} days"

        # Calculate loss
        loss_amount = stop.position_size * (stop.entry_price - current_price)
        loss_pct = (current_price - stop.entry_price) / stop.entry_price

        # Mark as triggered if should exit
        if should_exit:
            stop.status = StopStatus.TRIGGERED
            self.triggered_stops.append(stop)
            del self.active_stops[symbol]

        return StopLossResult(
            symbol=symbol,
            should_exit=should_exit,
            exit_reason=exit_reason,
            stop_price=stop.stop_price,
            current_price=current_price,
            loss_amount=loss_amount,
            loss_percentage=loss_pct,
            recommended_exit_size=stop.position_size
        )

    def check_correlation_stop(
        self,
        symbol: str,
        current_price: float,
        expected_correlation: float,
        actual_correlation: float
    ) -> StopLossResult:
        """
        Check for correlation breakdown stop

        Args:
            symbol: Asset symbol
            current_price: Current price
            expected_correlation: Expected correlation with strategy
            actual_correlation: Actual recent correlation

        Returns:
            StopLossResult with exit recommendation
        """
        should_exit = False
        exit_reason = ""

        # Check if correlation has broken down
        correlation_diff = abs(expected_correlation - actual_correlation)

        if correlation_diff > self.correlation_threshold:
            should_exit = True
            exit_reason = f"Correlation breakdown: expected={expected_correlation:.2f}, actual={actual_correlation:.2f}"

        if symbol in self.active_stops:
            stop = self.active_stops[symbol]
            loss_amount = stop.position_size * (stop.entry_price - current_price)
            loss_pct = (current_price - stop.entry_price) / stop.entry_price
            position_size = stop.position_size

            if should_exit:
                stop.status = StopStatus.TRIGGERED
                self.triggered_stops.append(stop)
                del self.active_stops[symbol]
        else:
            loss_amount = 0.0
            loss_pct = 0.0
            position_size = 0.0

        return StopLossResult(
            symbol=symbol,
            should_exit=should_exit,
            exit_reason=exit_reason,
            stop_price=current_price,
            current_price=current_price,
            loss_amount=loss_amount,
            loss_percentage=loss_pct,
            recommended_exit_size=position_size
        )

    def check_portfolio_stop(
        self,
        portfolio_value: float,
        peak_value: float
    ) -> bool:
        """
        Check if portfolio-level stop should be triggered

        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value

        Returns:
            True if portfolio stop triggered
        """
        drawdown = (portfolio_value - peak_value) / peak_value
        return drawdown <= -self.portfolio_stop_pct

    def optimize_stop_distance(
        self,
        returns: pd.Series,
        prices: pd.Series,
        test_range: np.ndarray = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize stop loss distance using historical data

        Args:
            returns: Historical returns
            prices: Historical prices
            test_range: Array of stop percentages to test

        Returns:
            Tuple of (optimal_stop_pct, performance_metrics)
        """
        if test_range is None:
            test_range = np.arange(0.01, 0.10, 0.005)  # 1% to 10% in 0.5% steps

        best_sharpe = -np.inf
        best_stop = 0.02
        results = {}

        for stop_pct in test_range:
            # Simulate strategy with this stop
            stopped_returns = self._simulate_stops(returns, prices, stop_pct)

            # Calculate metrics
            sharpe = self._calculate_sharpe(stopped_returns)
            max_dd = self._calculate_max_drawdown(stopped_returns)
            win_rate = (stopped_returns > 0).sum() / len(stopped_returns)

            results[stop_pct] = {
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'win_rate': win_rate
            }

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_stop = stop_pct

        return best_stop, results[best_stop]

    def _simulate_stops(
        self,
        returns: pd.Series,
        prices: pd.Series,
        stop_pct: float
    ) -> pd.Series:
        """Simulate trading with stop losses"""
        stopped_returns = returns.copy()

        for i in range(1, len(prices)):
            # Calculate drawdown from entry
            entry_price = prices.iloc[i-1]
            current_price = prices.iloc[i]
            drawdown = (current_price - entry_price) / entry_price

            # If stop triggered, return is the stop loss
            if drawdown <= -stop_pct:
                stopped_returns.iloc[i] = -stop_pct

        return stopped_returns

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def get_active_stops(self) -> Dict[str, StopLossOrder]:
        """Get all active stop orders"""
        return self.active_stops.copy()

    def get_triggered_stops(self) -> List[StopLossOrder]:
        """Get history of triggered stops"""
        return self.triggered_stops.copy()

    def cancel_stop(self, symbol: str) -> bool:
        """
        Cancel active stop for symbol

        Args:
            symbol: Asset symbol

        Returns:
            True if cancelled, False if not found
        """
        if symbol in self.active_stops:
            stop = self.active_stops[symbol]
            stop.status = StopStatus.CANCELLED
            del self.active_stops[symbol]
            return True
        return False

    def get_stop_statistics(self) -> Dict[str, float]:
        """
        Get stop loss statistics

        Returns:
            Dictionary of stop statistics
        """
        if len(self.triggered_stops) == 0:
            return {
                'total_stops': 0,
                'avg_loss': 0.0,
                'max_loss': 0.0,
                'stop_rate': 0.0
            }

        losses = [stop.loss_amount for stop in self.triggered_stops if hasattr(stop, 'loss_amount')]

        return {
            'total_stops': len(self.triggered_stops),
            'avg_loss': np.mean(losses) if losses else 0.0,
            'max_loss': np.max(losses) if losses else 0.0,
            'stop_rate': len(self.triggered_stops) / (len(self.triggered_stops) + len(self.active_stops))
        }

    def reset(self):
        """Reset stop loss system"""
        self.active_stops = {}
        self.triggered_stops = []

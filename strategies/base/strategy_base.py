"""
Abstract Base Class for All Trading Strategies

Defines the interface and common functionality for all strategies in the system.
Integrates with Step 4 signals (triple-barrier labels and meta-labels).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger


class SignalType(Enum):
    """Signal types for strategy actions."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"


class PositionSide(Enum):
    """Position sides."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class StrategySignal:
    """
    Trading signal from a strategy.

    Attributes:
        timestamp: Signal generation timestamp
        symbol: Instrument symbol
        signal_type: Type of signal (LONG/SHORT/EXIT)
        confidence: Signal confidence (0-1) from ML models
        size: Suggested position size (in units)
        entry_price: Expected entry price
        stop_loss: Stop loss price (from triple-barrier)
        take_profit: Take profit price (from triple-barrier)
        metadata: Additional signal metadata
    """
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float  # From meta-labels
    size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RiskMetrics:
    """Risk metrics for position sizing and validation."""
    max_position_size: float
    max_portfolio_allocation: float  # Maximum % of portfolio
    max_correlation: float  # Maximum correlation with existing positions
    volatility_target: float  # Annualized volatility target
    max_leverage: float
    position_risk: float  # Maximum loss per position as % of capital


class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface that all strategies must implement,
    providing hooks for signal generation, position sizing, risk management,
    and integration with the broader trading system.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize strategy base.

        Args:
            name: Strategy name
            config: Strategy configuration parameters
            risk_metrics: Risk limits and targets
        """
        self.name = name
        self.config = config
        self.risk_metrics = risk_metrics or self._default_risk_metrics()
        self.positions: Dict[str, Any] = {}
        self.signals_history: List[StrategySignal] = []
        self.performance: Dict[str, float] = {}
        self.enabled = config.get('enabled', True)

        logger.info(f"Initialized strategy: {name}")

    def _default_risk_metrics(self) -> RiskMetrics:
        """Default risk metrics if none provided."""
        return RiskMetrics(
            max_position_size=100000.0,
            max_portfolio_allocation=0.20,  # 20% max per position
            max_correlation=0.85,
            volatility_target=0.20,  # 20% annualized
            max_leverage=2.0,
            position_risk=0.02  # 2% max loss per position
        )

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate trading signal for a given instrument.

        Args:
            symbol: Instrument symbol
            market_data: OHLCV price data
            features: Engineered features from Step 3
            ml_signals: ML signals from Step 4 (includes meta-labels and confidence)

        Returns:
            StrategySignal or None if no signal
        """
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size for a signal.

        Implements volatility-based position sizing with risk limits.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Current instrument volatility
            existing_positions: Current portfolio positions

        Returns:
            Position size in units
        """
        pass

    def manage_risk(
        self,
        signal: StrategySignal,
        portfolio_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate signal against risk constraints.

        Args:
            signal: Proposed signal
            portfolio_state: Current portfolio state

        Returns:
            (is_valid, rejection_reason)
        """
        # Check if signal confidence meets minimum threshold
        min_confidence = self.config.get('min_confidence', 0.5)
        if signal.confidence < min_confidence:
            return False, f"Confidence {signal.confidence:.2f} below threshold {min_confidence}"

        # Check portfolio allocation limit
        position_value = signal.size * signal.entry_price
        portfolio_value = portfolio_state.get('total_value', 0)
        if portfolio_value > 0:
            allocation = position_value / portfolio_value
            if allocation > self.risk_metrics.max_portfolio_allocation:
                return False, f"Allocation {allocation:.2%} exceeds limit {self.risk_metrics.max_portfolio_allocation:.2%}"

        # Check correlation with existing positions
        if 'positions' in portfolio_state:
            correlations = self._check_correlations(signal.symbol, portfolio_state['positions'])
            if any(abs(corr) > self.risk_metrics.max_correlation for corr in correlations):
                return False, f"Correlation exceeds limit {self.risk_metrics.max_correlation}"

        return True, None

    def _check_correlations(
        self,
        symbol: str,
        existing_positions: Dict[str, Any]
    ) -> List[float]:
        """
        Check correlation between proposed position and existing positions.

        Args:
            symbol: New position symbol
            existing_positions: Current positions

        Returns:
            List of correlation coefficients
        """
        # Placeholder - should integrate with correlation_manager in portfolio/
        # For now, return empty list
        return []

    def update_position(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        entry_price: float,
        timestamp: datetime
    ):
        """
        Update position tracking.

        Args:
            symbol: Instrument symbol
            side: Position side (LONG/SHORT)
            size: Position size
            entry_price: Entry price
            timestamp: Entry timestamp
        """
        self.positions[symbol] = {
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'unrealized_pnl': 0.0
        }

    def close_position(self, symbol: str, exit_price: float, timestamp: datetime) -> float:
        """
        Close position and calculate P&L.

        Args:
            symbol: Instrument symbol
            exit_price: Exit price
            timestamp: Exit timestamp

        Returns:
            Realized P&L
        """
        if symbol not in self.positions:
            logger.warning(f"Attempting to close non-existent position: {symbol}")
            return 0.0

        position = self.positions[symbol]
        pnl = (exit_price - position['entry_price']) * position['size'] * position['side'].value

        logger.info(
            f"Closed {symbol} position: "
            f"Entry={position['entry_price']:.2f}, "
            f"Exit={exit_price:.2f}, "
            f"P&L={pnl:.2f}"
        )

        del self.positions[symbol]
        return pnl

    def update_performance(self, metric: str, value: float):
        """
        Update strategy performance metrics.

        Args:
            metric: Metric name
            value: Metric value
        """
        self.performance[metric] = value

    def get_state(self) -> Dict[str, Any]:
        """
        Get current strategy state for persistence.

        Returns:
            Strategy state dictionary
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'positions': self.positions,
            'performance': self.performance,
            'signals_count': len(self.signals_history)
        }

    def load_state(self, state: Dict[str, Any]):
        """
        Load strategy state from persistence.

        Args:
            state: Strategy state dictionary
        """
        self.positions = state.get('positions', {})
        self.performance = state.get('performance', {})
        self.enabled = state.get('enabled', True)
        logger.info(f"Loaded state for strategy: {self.name}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"positions={len(self.positions)}, "
            f"enabled={self.enabled})"
        )

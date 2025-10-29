"""
Market Impact Model

Implements various market impact models for realistic trade cost estimation:
- Linear impact model
- Square-root impact model
- Almgren-Chriss temporary and permanent impact
- Intraday impact patterns
- Impact decay modeling

Based on research by:
- Almgren & Chriss (2001)
- Grinold & Kahn (2000)
- Kissell & Glantz (2003)
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional, Callable
import numpy as np
from loguru import logger


class ImpactType(Enum):
    """Market impact model type."""
    LINEAR = "LINEAR"
    SQRT = "SQRT"
    ALMGREN_CHRISS = "ALMGREN_CHRISS"
    CUSTOM = "CUSTOM"


@dataclass
class ImpactParameters:
    """
    Parameters for market impact calculation.

    Attributes:
        lambda_linear: Linear impact coefficient
        lambda_sqrt: Square-root impact coefficient
        eta: Temporary impact coefficient (Almgren-Chriss)
        gamma: Permanent impact coefficient (Almgren-Chriss)
        decay_rate: Impact decay rate (per unit time)
        volume_scaling: Whether to scale by daily volume
        volatility_scaling: Whether to scale by volatility
    """
    lambda_linear: float = 0.0001  # 1 bp per unit of trade size
    lambda_sqrt: float = 0.01  # Square-root impact coefficient
    eta: float = 0.0005  # Temporary impact coefficient
    gamma: float = 0.0001  # Permanent impact coefficient
    decay_rate: float = 0.1  # Impact decay rate
    volume_scaling: bool = True
    volatility_scaling: bool = True


class MarketImpactModel:
    """
    Market impact model for estimating price impact of trades.

    Supports multiple impact models and realistic market microstructure effects.
    """

    def __init__(
        self,
        impact_type: ImpactType = ImpactType.SQRT,
        params: Optional[ImpactParameters] = None
    ):
        """
        Initialize market impact model.

        Args:
            impact_type: Type of impact model to use
            params: Model parameters
        """
        self.impact_type = impact_type
        self.params = params or ImpactParameters()
        self._custom_impact_func: Optional[Callable] = None

        # Historical impact data for decay modeling
        self.impact_history: Dict[str, list] = {}

        logger.info(f"Initialized MarketImpactModel with type={impact_type.value}")

    def set_custom_impact_function(self, func: Callable):
        """
        Set custom impact function.

        Args:
            func: Custom impact function with signature:
                  func(trade_size, adv, volatility, **kwargs) -> float
        """
        self._custom_impact_func = func
        self.impact_type = ImpactType.CUSTOM
        logger.info("Set custom impact function")

    def calculate_impact(
        self,
        symbol: str,
        trade_size: float,
        side: str,
        market_price: float,
        adv: Optional[float] = None,  # Average Daily Volume
        volatility: Optional[float] = None,
        spread: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate market impact for a trade.

        Args:
            symbol: Trading symbol
            trade_size: Trade size (positive)
            side: 'BUY' or 'SELL'
            market_price: Current market price
            adv: Average daily volume
            volatility: Historical volatility
            spread: Current bid-ask spread
            timestamp: Trade timestamp
            **kwargs: Additional parameters

        Returns:
            Dictionary with impact metrics
        """
        if trade_size <= 0:
            return self._empty_impact()

        # Calculate base impact
        if self.impact_type == ImpactType.LINEAR:
            impact = self._linear_impact(trade_size, adv, volatility)
        elif self.impact_type == ImpactType.SQRT:
            impact = self._sqrt_impact(trade_size, adv, volatility)
        elif self.impact_type == ImpactType.ALMGREN_CHRISS:
            impact = self._almgren_chriss_impact(trade_size, adv, volatility)
        elif self.impact_type == ImpactType.CUSTOM and self._custom_impact_func:
            impact = self._custom_impact_func(trade_size, adv, volatility, **kwargs)
        else:
            impact = self._sqrt_impact(trade_size, adv, volatility)

        # Apply intraday pattern
        if timestamp:
            impact *= self._get_intraday_multiplier(timestamp)

        # Convert to price impact
        impact_price = impact * market_price

        # Apply spread crossing for aggressive orders
        if spread:
            impact_price += spread / 2.0

        # Calculate total cost
        total_impact_bps = (impact_price / market_price) * 10000

        # Record impact for decay modeling
        self._record_impact(symbol, timestamp or datetime.now(), impact, trade_size)

        # Calculate decayed impact from recent trades
        residual_impact = self._calculate_residual_impact(symbol, timestamp)

        result = {
            'impact_bps': total_impact_bps,
            'impact_price': impact_price,
            'temporary_impact': impact_price * 0.7,  # Assume 70% temporary
            'permanent_impact': impact_price * 0.3,  # Assume 30% permanent
            'residual_impact': residual_impact,
            'total_cost': impact_price + residual_impact * market_price
        }

        logger.debug(f"Impact for {symbol}: {total_impact_bps:.2f} bps on {trade_size} shares")

        return result

    def _linear_impact(self, trade_size: float, adv: Optional[float], volatility: Optional[float]) -> float:
        """
        Linear impact model: MI = λ × trade_size

        Args:
            trade_size: Trade size
            adv: Average daily volume
            volatility: Volatility

        Returns:
            Impact as fraction of price
        """
        impact = self.params.lambda_linear * trade_size

        # Scale by volume if available
        if adv and self.params.volume_scaling:
            participation_rate = trade_size / adv
            impact *= (1 + participation_rate)

        # Scale by volatility if available
        if volatility and self.params.volatility_scaling:
            impact *= (1 + volatility)

        return impact

    def _sqrt_impact(self, trade_size: float, adv: Optional[float], volatility: Optional[float]) -> float:
        """
        Square-root impact model: MI = λ × √trade_size

        More realistic for large trades. Based on empirical research.

        Args:
            trade_size: Trade size
            adv: Average daily volume
            volatility: Volatility

        Returns:
            Impact as fraction of price
        """
        base_impact = self.params.lambda_sqrt * np.sqrt(trade_size)

        # Scale by participation rate
        if adv and adv > 0 and self.params.volume_scaling:
            participation_rate = trade_size / adv
            # Impact increases non-linearly with participation
            base_impact *= np.sqrt(1 + participation_rate * 10)

        # Scale by volatility
        if volatility and self.params.volatility_scaling:
            base_impact *= (1 + volatility)

        return base_impact

    def _almgren_chriss_impact(self, trade_size: float, adv: Optional[float], volatility: Optional[float]) -> float:
        """
        Simplified Almgren-Chriss impact model.

        Total impact = permanent + temporary
        Permanent: γ × (trade_size / adv)
        Temporary: η × (trade_size / adv)

        Args:
            trade_size: Trade size
            adv: Average daily volume
            volatility: Volatility

        Returns:
            Impact as fraction of price
        """
        if adv is None or adv <= 0:
            adv = trade_size * 10  # Assume 10% participation if unknown

        participation = trade_size / adv

        # Permanent impact (linear in participation)
        permanent = self.params.gamma * participation

        # Temporary impact (linear in participation)
        temporary = self.params.eta * participation

        total_impact = permanent + temporary

        # Scale by volatility
        if volatility and self.params.volatility_scaling:
            total_impact *= (1 + volatility)

        return total_impact

    def _get_intraday_multiplier(self, timestamp: datetime) -> float:
        """
        Get intraday impact multiplier based on time of day.

        Impact is typically higher at:
        - Market open (9:30-10:00)
        - Market close (15:30-16:00)
        - Lunch time (12:00-13:00)

        Args:
            timestamp: Trade timestamp

        Returns:
            Multiplier (typically between 0.8 and 1.5)
        """
        trade_time = timestamp.time()

        # Market open (high volatility and impact)
        if time(9, 30) <= trade_time < time(10, 0):
            return 1.5
        # Early morning (elevated)
        elif time(10, 0) <= trade_time < time(10, 30):
            return 1.2
        # Lunch time (reduced liquidity)
        elif time(12, 0) <= trade_time < time(13, 0):
            return 1.15
        # Market close (high volatility)
        elif time(15, 30) <= trade_time <= time(16, 0):
            return 1.4
        # Late afternoon
        elif time(15, 0) <= trade_time < time(15, 30):
            return 1.2
        # Normal trading hours
        else:
            return 1.0

    def _record_impact(self, symbol: str, timestamp: datetime, impact: float, trade_size: float):
        """Record impact for decay modeling."""
        if symbol not in self.impact_history:
            self.impact_history[symbol] = []

        self.impact_history[symbol].append({
            'timestamp': timestamp,
            'impact': impact,
            'trade_size': trade_size
        })

        # Keep only recent history (last 100 trades)
        if len(self.impact_history[symbol]) > 100:
            self.impact_history[symbol] = self.impact_history[symbol][-100:]

    def _calculate_residual_impact(self, symbol: str, current_time: Optional[datetime]) -> float:
        """
        Calculate residual impact from recent trades with exponential decay.

        Args:
            symbol: Symbol
            current_time: Current timestamp

        Returns:
            Residual impact as fraction of price
        """
        if symbol not in self.impact_history or not self.impact_history[symbol]:
            return 0.0

        if current_time is None:
            return 0.0

        residual = 0.0

        for trade in self.impact_history[symbol]:
            time_diff = (current_time - trade['timestamp']).total_seconds() / 3600.0  # Hours
            # Exponential decay
            decay_factor = np.exp(-self.params.decay_rate * time_diff)
            residual += trade['impact'] * decay_factor * 0.3  # Only permanent part persists

        return residual

    def _empty_impact(self) -> Dict[str, float]:
        """Return empty impact result."""
        return {
            'impact_bps': 0.0,
            'impact_price': 0.0,
            'temporary_impact': 0.0,
            'permanent_impact': 0.0,
            'residual_impact': 0.0,
            'total_cost': 0.0
        }

    def reset_history(self, symbol: Optional[str] = None):
        """
        Reset impact history.

        Args:
            symbol: Symbol to reset (None = reset all)
        """
        if symbol:
            self.impact_history.pop(symbol, None)
        else:
            self.impact_history.clear()

        logger.info(f"Reset impact history for {symbol or 'all symbols'}")


class AlmgrenChrissModel:
    """
    Full Almgren-Chriss market impact model with optimal execution.

    Implements the complete model from "Optimal Execution of Portfolio Transactions"
    (Almgren & Chriss, 2001).
    """

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.5,
        volatility: float = 0.02,
        risk_aversion: float = 1e-6
    ):
        """
        Initialize Almgren-Chriss model.

        Args:
            permanent_impact_coef: Permanent impact coefficient (γ)
            temporary_impact_coef: Temporary impact coefficient (η)
            volatility: Daily volatility (σ)
            risk_aversion: Risk aversion parameter (λ)
        """
        self.gamma = permanent_impact_coef
        self.eta = temporary_impact_coef
        self.sigma = volatility
        self.lambda_risk = risk_aversion

        logger.info("Initialized AlmgrenChrissModel")

    def calculate_temporary_impact(self, trade_rate: float, volatility: float, adv: float) -> float:
        """
        Calculate temporary impact.

        Temporary impact = η × σ × (trade_rate / adv)

        Args:
            trade_rate: Trading rate (shares per unit time)
            volatility: Volatility
            adv: Average daily volume

        Returns:
            Temporary impact in price units
        """
        if adv <= 0:
            return 0.0

        return self.eta * volatility * (trade_rate / adv)

    def calculate_permanent_impact(self, trade_size: float, adv: float) -> float:
        """
        Calculate permanent impact.

        Permanent impact = γ × (trade_size / adv)

        Args:
            trade_size: Total trade size
            adv: Average daily volume

        Returns:
            Permanent impact in price units
        """
        if adv <= 0:
            return 0.0

        return self.gamma * (trade_size / adv)

    def optimal_trajectory(
        self,
        total_shares: float,
        time_horizon: float,
        num_periods: int,
        adv: float
    ) -> np.ndarray:
        """
        Calculate optimal trading trajectory.

        Args:
            total_shares: Total shares to trade
            time_horizon: Time horizon (in days)
            num_periods: Number of trading periods
            adv: Average daily volume

        Returns:
            Array of trade sizes for each period
        """
        if num_periods <= 0:
            return np.array([total_shares])

        tau = time_horizon / num_periods

        # Calculate kappa (measures trade-off between impact and risk)
        kappa = np.sqrt(self.lambda_risk * self.sigma**2 / (self.eta * tau))

        # Calculate sinh and cosh terms
        sinh_term = np.sinh(kappa * tau)
        cosh_term = np.cosh(kappa * tau)

        # Trading trajectory
        trajectory = np.zeros(num_periods)

        for k in range(num_periods):
            remaining_time = time_horizon - k * tau
            sinh_kt = np.sinh(kappa * remaining_time)
            sinh_kt_minus_tau = np.sinh(kappa * (remaining_time - tau))

            trajectory[k] = total_shares * (sinh_kt - sinh_kt_minus_tau) / np.sinh(kappa * time_horizon)

        return trajectory

    def calculate_total_cost(
        self,
        total_shares: float,
        time_horizon: float,
        num_periods: int,
        adv: float,
        initial_price: float
    ) -> Dict[str, float]:
        """
        Calculate total expected cost and variance for optimal execution.

        Args:
            total_shares: Total shares to trade
            time_horizon: Time horizon (in days)
            num_periods: Number of trading periods
            adv: Average daily volume
            initial_price: Initial stock price

        Returns:
            Dictionary with cost metrics
        """
        trajectory = self.optimal_trajectory(total_shares, time_horizon, num_periods, adv)

        tau = time_horizon / num_periods

        # Expected cost from permanent impact
        permanent_cost = self.gamma * (total_shares / adv) * initial_price * total_shares / 2

        # Expected cost from temporary impact
        temporary_cost = 0.0
        for trade in trajectory:
            trade_rate = trade / tau
            temp_impact = self.calculate_temporary_impact(trade_rate, self.sigma, adv)
            temporary_cost += temp_impact * trade

        # Variance of execution cost
        variance = self.sigma**2 * sum(trajectory**2)

        # Total expected cost
        total_cost = permanent_cost + temporary_cost

        # Cost in basis points
        cost_bps = (total_cost / (initial_price * total_shares)) * 10000

        return {
            'total_cost': total_cost,
            'permanent_cost': permanent_cost,
            'temporary_cost': temporary_cost,
            'variance': variance,
            'std_dev': np.sqrt(variance),
            'cost_bps': cost_bps,
            'optimal_trajectory': trajectory
        }

    def calculate_implementation_shortfall(
        self,
        total_shares: float,
        actual_fills: np.ndarray,
        fill_prices: np.ndarray,
        initial_price: float
    ) -> Dict[str, float]:
        """
        Calculate implementation shortfall (actual cost vs. benchmark).

        Args:
            total_shares: Total shares to trade
            actual_fills: Array of actual fill quantities
            fill_prices: Array of actual fill prices
            initial_price: Initial (decision) price

        Returns:
            Implementation shortfall metrics
        """
        if len(actual_fills) == 0:
            return {'implementation_shortfall': 0.0, 'shortfall_bps': 0.0}

        # Calculate average execution price
        avg_price = np.average(fill_prices, weights=actual_fills)

        # Implementation shortfall
        shortfall = (avg_price - initial_price) * total_shares

        # In basis points
        shortfall_bps = ((avg_price - initial_price) / initial_price) * 10000

        return {
            'implementation_shortfall': shortfall,
            'shortfall_bps': shortfall_bps,
            'avg_execution_price': avg_price,
            'benchmark_price': initial_price,
            'slippage': avg_price - initial_price
        }

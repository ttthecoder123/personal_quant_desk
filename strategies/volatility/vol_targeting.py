"""
Volatility Targeting Strategy

Implements Rob Carver's volatility targeting approach with:
- EWMA volatility estimation
- Target portfolio volatility (20% annual)
- Dynamic leverage adjustment
- GARCH(1,1) volatility forecasting
- Risk budget allocation across assets
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from strategies.base.strategy_base import (
    StrategyBase,
    StrategySignal,
    SignalType,
    PositionSide,
    RiskMetrics
)


class VolatilityTargetingStrategy(StrategyBase):
    """
    Volatility targeting strategy based on Rob Carver's approach.

    Key principles:
    1. Target constant volatility across all instruments
    2. Adjust position sizes inversely to volatility
    3. Use exponential weighting for recent volatility
    4. Scale positions to maintain portfolio-level volatility target
    """

    def __init__(
        self,
        name: str = "VolatilityTargeting",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize volatility targeting strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - target_vol: Target annual volatility (default: 0.20)
                - ewma_span: EWMA span for volatility (default: 36)
                - garch_enabled: Use GARCH forecasting (default: True)
                - risk_budget: Per-asset risk budget (default: 0.05)
                - min_observations: Minimum data points needed (default: 30)
                - rebalance_threshold: Trigger rebalance when vol deviates by this % (default: 0.10)
            risk_metrics: Risk limits
        """
        default_config = {
            'target_vol': 0.20,  # 20% annualized
            'ewma_span': 36,  # ~2 months for daily data
            'garch_enabled': True,
            'risk_budget': 0.05,  # 5% risk per position
            'min_observations': 30,
            'rebalance_threshold': 0.10,  # 10% deviation triggers rebalance
            'lookback_days': 252,  # ~1 year for volatility estimation
            'min_confidence': 0.6,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Volatility tracking
        self.vol_forecasts: Dict[str, float] = {}
        self.realized_vols: Dict[str, float] = {}
        self.leverage_ratios: Dict[str, float] = {}

        logger.info(
            f"Initialized {name} with target_vol={self.config['target_vol']:.1%}, "
            f"ewma_span={self.config['ewma_span']}"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate volatility-weighted signal.

        Signal logic:
        1. Calculate current volatility (EWMA + GARCH)
        2. Determine leverage needed to hit target volatility
        3. If ML signal exists, scale position by confidence
        4. Generate entry/exit signals based on position rebalancing needs

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            features: Engineered features (optional)
            ml_signals: ML predictions with confidence scores

        Returns:
            StrategySignal or None
        """
        try:
            if len(market_data) < self.config['min_observations']:
                logger.warning(f"{symbol}: Insufficient data ({len(market_data)} rows)")
                return None

            # Calculate current volatility
            current_vol = self._calculate_volatility(symbol, market_data)
            if current_vol is None or current_vol <= 0:
                logger.warning(f"{symbol}: Invalid volatility calculation")
                return None

            # Forecast future volatility
            forecast_vol = self._forecast_volatility(symbol, market_data, current_vol)
            self.vol_forecasts[symbol] = forecast_vol
            self.realized_vols[symbol] = current_vol

            # Calculate target leverage
            leverage = self._calculate_leverage(forecast_vol)
            self.leverage_ratios[symbol] = leverage

            # Get ML signal direction and confidence
            signal_type, confidence = self._get_ml_direction(ml_signals)

            # Check if rebalancing is needed
            current_position = self.positions.get(symbol, {})
            needs_rebalance = self._needs_rebalance(
                symbol,
                current_position,
                leverage,
                signal_type
            )

            if not needs_rebalance:
                return None

            # Generate signal
            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()
            current_price = float(market_data['close'].iloc[-1])

            # Calculate stop loss and take profit based on volatility
            stop_distance = forecast_vol * current_price * np.sqrt(1/252)  # 1-day vol
            stop_loss = current_price - stop_distance if signal_type == SignalType.LONG else current_price + stop_distance
            take_profit = current_price + 2 * stop_distance if signal_type == SignalType.LONG else current_price - 2 * stop_distance

            signal = StrategySignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=0.0,  # Will be calculated in calculate_position_size
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'strategy': self.name,
                    'current_vol': current_vol,
                    'forecast_vol': forecast_vol,
                    'leverage': leverage,
                    'target_vol': self.config['target_vol']
                }
            )

            self.signals_history.append(signal)
            logger.info(
                f"{symbol} signal: {signal_type.value}, "
                f"vol={forecast_vol:.2%}, leverage={leverage:.2f}x, "
                f"confidence={confidence:.2%}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size using volatility targeting.

        Position size formula:
        size = (portfolio_value * risk_budget) / (price * volatility * leverage)

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_volatility: Current instrument volatility (annualized)
            existing_positions: Current positions

        Returns:
            Position size in units
        """
        try:
            if portfolio_value <= 0 or current_volatility <= 0:
                logger.warning("Invalid inputs for position sizing")
                return 0.0

            # Get leverage from signal metadata
            leverage = signal.metadata.get('leverage', 1.0)

            # Risk budget for this position
            risk_budget = self.config['risk_budget']

            # Calculate base position size (volatility-scaled)
            # Target: risk_budget of portfolio at current volatility
            volatility_scalar = self.config['target_vol'] / current_volatility
            base_size = (portfolio_value * risk_budget * volatility_scalar) / signal.entry_price

            # Apply leverage adjustment
            position_size = base_size * leverage

            # Scale by ML confidence
            confidence_scalar = signal.confidence  # Linear scaling by confidence
            position_size *= confidence_scalar

            # Apply risk limits
            max_position_value = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_size = max_position_value / signal.entry_price
            position_size = min(position_size, max_size)

            # Apply leverage limit
            total_exposure = sum(
                abs(pos.get('size', 0) * pos.get('entry_price', 0))
                for pos in existing_positions.values()
            )
            total_exposure += position_size * signal.entry_price

            if total_exposure > portfolio_value * self.risk_metrics.max_leverage:
                # Scale down to respect leverage limit
                allowed_exposure = portfolio_value * self.risk_metrics.max_leverage - (total_exposure - position_size * signal.entry_price)
                position_size = max(0, allowed_exposure / signal.entry_price)

            logger.debug(
                f"{signal.symbol} position size: {position_size:.2f} units "
                f"(value: ${position_size * signal.entry_price:,.0f}, "
                f"leverage: {leverage:.2f}x)"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0

    def _calculate_volatility(
        self,
        symbol: str,
        market_data: pd.DataFrame
    ) -> Optional[float]:
        """
        Calculate EWMA volatility.

        Args:
            symbol: Instrument symbol
            market_data: Price data

        Returns:
            Annualized volatility or None
        """
        try:
            # Calculate log returns
            returns = np.log(market_data['close'] / market_data['close'].shift(1))
            returns = returns.dropna()

            if len(returns) < self.config['min_observations']:
                return None

            # EWMA volatility
            ewma_var = returns.ewm(span=self.config['ewma_span'], adjust=False).var()
            current_vol = np.sqrt(ewma_var.iloc[-1] * 252)  # Annualize

            return float(current_vol)

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None

    def _forecast_volatility(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        current_vol: float
    ) -> float:
        """
        Forecast future volatility using GARCH(1,1) if enabled, else use EWMA.

        Args:
            symbol: Instrument symbol
            market_data: Price data
            current_vol: Current EWMA volatility

        Returns:
            Forecasted annualized volatility
        """
        if not self.config['garch_enabled']:
            return current_vol

        try:
            # Simple GARCH(1,1) approximation
            # Full GARCH would require arch library
            returns = np.log(market_data['close'] / market_data['close'].shift(1)).dropna()

            if len(returns) < 100:
                return current_vol

            # GARCH parameters (simplified)
            omega = 0.000001  # Long-run variance
            alpha = 0.10  # ARCH term
            beta = 0.85   # GARCH term

            # Calculate conditional variance
            returns_squared = returns ** 2
            long_run_var = returns.var()

            # GARCH forecast: sigma^2_t+1 = omega + alpha*return^2_t + beta*sigma^2_t
            last_return_sq = returns_squared.iloc[-1]
            current_var = current_vol ** 2 / 252  # De-annualize

            forecast_var = omega + alpha * last_return_sq + beta * current_var
            forecast_vol = np.sqrt(forecast_var * 252)  # Annualize

            # Sanity check: don't let forecast deviate too much from current
            min_vol = current_vol * 0.5
            max_vol = current_vol * 2.0
            forecast_vol = np.clip(forecast_vol, min_vol, max_vol)

            return float(forecast_vol)

        except Exception as e:
            logger.warning(f"GARCH forecast failed for {symbol}, using EWMA: {e}")
            return current_vol

    def _calculate_leverage(self, forecast_vol: float) -> float:
        """
        Calculate leverage needed to hit target volatility.

        Args:
            forecast_vol: Forecasted volatility

        Returns:
            Leverage ratio
        """
        if forecast_vol <= 0:
            return 1.0

        # Leverage = target_vol / forecast_vol
        leverage = self.config['target_vol'] / forecast_vol

        # Apply leverage limits
        leverage = np.clip(leverage, 1/self.risk_metrics.max_leverage, self.risk_metrics.max_leverage)

        return float(leverage)

    def _get_ml_direction(
        self,
        ml_signals: Optional[pd.DataFrame]
    ) -> tuple[SignalType, float]:
        """
        Extract direction and confidence from ML signals.

        Args:
            ml_signals: ML predictions with columns ['prediction', 'confidence']

        Returns:
            (signal_type, confidence)
        """
        if ml_signals is None or len(ml_signals) == 0:
            # No ML signal, default to LONG with neutral confidence
            return SignalType.LONG, 0.6

        try:
            latest_signal = ml_signals.iloc[-1]
            prediction = latest_signal.get('prediction', 1)
            confidence = latest_signal.get('confidence', 0.6)

            # Map prediction to signal type
            if prediction > 0:
                signal_type = SignalType.LONG
            elif prediction < 0:
                signal_type = SignalType.SHORT
            else:
                signal_type = SignalType.HOLD

            return signal_type, float(confidence)

        except Exception as e:
            logger.warning(f"Error parsing ML signals: {e}")
            return SignalType.LONG, 0.6

    def _needs_rebalance(
        self,
        symbol: str,
        current_position: Dict[str, Any],
        target_leverage: float,
        signal_type: SignalType
    ) -> bool:
        """
        Check if position needs rebalancing.

        Args:
            symbol: Instrument symbol
            current_position: Current position details
            target_leverage: Target leverage for this instrument
            signal_type: Desired signal type

        Returns:
            True if rebalancing needed
        """
        # No position exists - enter new position
        if not current_position:
            return signal_type in [SignalType.LONG, SignalType.SHORT]

        # Check if leverage has drifted
        current_leverage = self.leverage_ratios.get(symbol, 1.0)
        leverage_drift = abs(current_leverage - target_leverage) / target_leverage

        if leverage_drift > self.config['rebalance_threshold']:
            logger.info(
                f"{symbol}: Rebalancing due to leverage drift "
                f"({leverage_drift:.1%} > {self.config['rebalance_threshold']:.1%})"
            )
            return True

        # Check if direction changed
        position_side = current_position.get('side', PositionSide.FLAT)
        if signal_type == SignalType.LONG and position_side != PositionSide.LONG:
            return True
        if signal_type == SignalType.SHORT and position_side != PositionSide.SHORT:
            return True

        return False

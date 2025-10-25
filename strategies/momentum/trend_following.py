"""
Trend Following Strategy - Carver's Approach

Implements Rob Carver's trend following methodology with:
- Multiple timeframe trend detection (20, 60, 120 days)
- Forecast scaling to standardized range (-20 to +20)
- Volatility-adjusted position sizing
- ADX trend strength filtering
- Diversification multiplier for multi-instrument portfolios

References:
- "Leveraged Trading" by Rob Carver
- "Systematic Trading" by Rob Carver
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from strategies.base.strategy_base import (
    StrategyBase,
    StrategySignal,
    SignalType,
    RiskMetrics
)


class TrendFollowingStrategy(StrategyBase):
    """
    Carver's trend following strategy with multiple timeframes and forecast combination.

    The strategy generates forecasts for multiple lookback periods (fast, medium, slow)
    and combines them into a single forecast scaled to -20 to +20 range.

    Key Features:
    - EWMA-based trend signals across 20, 60, 120 day periods
    - Forecast scaling based on historical volatility of raw forecasts
    - ADX filter to avoid choppy markets
    - Volatility targeting for position sizing
    - ML signal integration for confidence adjustment
    """

    def __init__(
        self,
        name: str = "TrendFollowing",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize trend following strategy.

        Args:
            name: Strategy name
            config: Configuration parameters
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'enabled': True,
            'lookback_periods': [20, 60, 120],  # Fast, medium, slow
            'ewma_spans': [16, 32, 64],  # EWMA spans for each period
            'forecast_scalar': 10.0,  # Target forecast std dev
            'forecast_cap': 20.0,  # Maximum forecast value
            'min_adx': 25.0,  # Minimum ADX for trend validity
            'min_confidence': 0.55,  # Minimum ML confidence
            'vol_lookback': 25,  # Days for volatility calculation
            'vol_target': 0.20,  # Annual volatility target (20%)
            'diversification_multiplier': 1.0,  # For multi-instrument
            'position_inertia': 0.10,  # Minimum change to rebalance (10%)
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Initialize forecast tracking
        self.forecast_history: Dict[str, list] = {}
        self.forecast_scalars: Dict[str, float] = {}

        logger.info(
            f"Initialized {name} with periods {self.config['lookback_periods']}, "
            f"ADX threshold {self.config['min_adx']}"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate trend following signal using Carver's methodology.

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            features: Technical features including ADX, ATR
            ml_signals: ML predictions and probabilities

        Returns:
            StrategySignal or None
        """
        try:
            # Validate data
            if len(market_data) < max(self.config['lookback_periods']) + 50:
                logger.warning(
                    f"{symbol}: Insufficient data ({len(market_data)} bars), "
                    f"need at least {max(self.config['lookback_periods']) + 50}"
                )
                return None

            # Get latest data
            close_prices = market_data['Close'].values
            current_price = close_prices[-1]
            timestamp = market_data.index[-1]

            # Calculate trends for each timeframe
            forecasts = []
            for lookback, ewma_span in zip(
                self.config['lookback_periods'],
                self.config['ewma_spans']
            ):
                forecast = self._calculate_ewma_forecast(
                    close_prices,
                    lookback=lookback,
                    ewma_span=ewma_span
                )
                forecasts.append(forecast)

            # Combine forecasts (equal weighting)
            raw_forecast = np.mean(forecasts)

            # Scale forecast to target range
            scaled_forecast = self._scale_forecast(symbol, raw_forecast)

            # Cap forecast
            capped_forecast = np.clip(
                scaled_forecast,
                -self.config['forecast_cap'],
                self.config['forecast_cap']
            )

            # Check ADX trend strength filter
            if features is not None and 'adx' in features.columns:
                adx = features['adx'].iloc[-1]
                if adx < self.config['min_adx']:
                    logger.debug(
                        f"{symbol}: ADX {adx:.1f} below threshold "
                        f"{self.config['min_adx']}, no signal"
                    )
                    return None

            # Determine signal type from forecast
            signal_type = self._forecast_to_signal(capped_forecast)

            if signal_type == SignalType.HOLD:
                return None

            # Get ML confidence if available
            confidence = self._get_ml_confidence(
                signal_type,
                ml_signals,
                timestamp
            )

            # Check minimum confidence
            if confidence < self.config['min_confidence']:
                logger.debug(
                    f"{symbol}: Confidence {confidence:.2f} below threshold "
                    f"{self.config['min_confidence']}"
                )
                return None

            # Calculate stop loss and take profit from ATR
            stop_loss, take_profit = None, None
            if features is not None and 'atr' in features.columns:
                atr = features['atr'].iloc[-1] if 'atr' in features.columns else None
                if atr and not np.isnan(atr):
                    # 2 ATR stop, 3 ATR target
                    if signal_type == SignalType.LONG:
                        stop_loss = current_price - 2 * atr
                        take_profit = current_price + 3 * atr
                    elif signal_type == SignalType.SHORT:
                        stop_loss = current_price + 2 * atr
                        take_profit = current_price - 3 * atr

            # Create signal
            signal = StrategySignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=0.0,  # Will be calculated by calculate_position_size
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'raw_forecast': raw_forecast,
                    'scaled_forecast': scaled_forecast,
                    'capped_forecast': capped_forecast,
                    'individual_forecasts': forecasts,
                    'adx': features['adx'].iloc[-1] if features is not None and 'adx' in features.columns else None,
                    'strategy': self.name
                }
            )

            logger.info(
                f"{symbol}: {signal_type.value} signal, "
                f"forecast={capped_forecast:.2f}, "
                f"confidence={confidence:.2f}, "
                f"price={current_price:.2f}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def _calculate_ewma_forecast(
        self,
        prices: np.ndarray,
        lookback: int,
        ewma_span: int
    ) -> float:
        """
        Calculate EWMA-based trend forecast.

        Compares fast EWMA to slow EWMA to determine trend direction.

        Args:
            prices: Price array
            lookback: Lookback period for slow EWMA
            ewma_span: Span for fast EWMA

        Returns:
            Raw forecast value (unstandardized)
        """
        # Calculate fast and slow EMAs
        fast_ema = pd.Series(prices).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
        slow_ema = pd.Series(prices).ewm(span=lookback, adjust=False).mean().iloc[-1]

        # Calculate forecast as percentage difference
        # Multiply by 100 to get better scaling
        forecast = ((fast_ema - slow_ema) / slow_ema) * 100

        return forecast

    def _scale_forecast(self, symbol: str, raw_forecast: float) -> float:
        """
        Scale forecast to target standard deviation.

        Uses historical forecast volatility to normalize to target range.

        Args:
            symbol: Instrument symbol
            raw_forecast: Raw forecast value

        Returns:
            Scaled forecast
        """
        # Track forecast history
        if symbol not in self.forecast_history:
            self.forecast_history[symbol] = []

        self.forecast_history[symbol].append(raw_forecast)

        # Keep only recent history (500 forecasts)
        if len(self.forecast_history[symbol]) > 500:
            self.forecast_history[symbol] = self.forecast_history[symbol][-500:]

        # Need at least 30 forecasts to calculate scalar
        if len(self.forecast_history[symbol]) < 30:
            return raw_forecast  # No scaling yet

        # Calculate forecast volatility
        forecast_std = np.std(self.forecast_history[symbol])

        if forecast_std < 1e-6:  # Avoid division by zero
            return raw_forecast

        # Calculate scalar to hit target volatility
        scalar = self.config['forecast_scalar'] / forecast_std

        # Cache scalar
        self.forecast_scalars[symbol] = scalar

        # Apply scaling
        scaled = raw_forecast * scalar

        return scaled

    def _forecast_to_signal(self, forecast: float) -> SignalType:
        """
        Convert forecast to signal type.

        Args:
            forecast: Capped forecast value (-20 to +20)

        Returns:
            SignalType
        """
        # Use inertia threshold to avoid overtrading
        threshold = self.config['position_inertia'] * self.config['forecast_cap']

        if forecast > threshold:
            return SignalType.LONG
        elif forecast < -threshold:
            return SignalType.SHORT
        else:
            return SignalType.HOLD

    def _get_ml_confidence(
        self,
        signal_type: SignalType,
        ml_signals: Optional[pd.DataFrame],
        timestamp: datetime
    ) -> float:
        """
        Get ML confidence for signal.

        Args:
            signal_type: Signal type
            ml_signals: ML predictions with probabilities
            timestamp: Signal timestamp

        Returns:
            Confidence score (0-1)
        """
        if ml_signals is None or ml_signals.empty:
            return 0.6  # Default confidence

        try:
            # Get latest ML prediction
            if timestamp in ml_signals.index:
                ml_row = ml_signals.loc[timestamp]
            else:
                ml_row = ml_signals.iloc[-1]

            # Get probability for predicted direction
            # Assumes ml_signals has columns like 'prob_-1', 'prob_0', 'prob_1'
            if signal_type == SignalType.LONG:
                # Look for long probability
                if 'prob_1' in ml_signals.columns:
                    confidence = ml_row['prob_1']
                elif 1 in ml_signals.columns:
                    confidence = ml_row[1]
                else:
                    confidence = 0.6
            elif signal_type == SignalType.SHORT:
                # Look for short probability
                if 'prob_-1' in ml_signals.columns:
                    confidence = ml_row['prob_-1']
                elif -1 in ml_signals.columns:
                    confidence = ml_row[-1]
                else:
                    confidence = 0.6
            else:
                confidence = 0.5

            return float(confidence)

        except Exception as e:
            logger.debug(f"Could not extract ML confidence: {e}")
            return 0.6  # Default

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate volatility-adjusted position size using Carver's methodology.

        Position size = (Portfolio Value * Target Vol * IDM * Forecast) / (Instrument Vol * Price)

        Where:
        - IDM = Instrument Diversification Multiplier
        - Forecast = Scaled forecast (-20 to +20)

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Instrument volatility (annualized)
            existing_positions: Current positions

        Returns:
            Position size in units (can be negative for short)
        """
        try:
            # Get forecast from metadata
            forecast = signal.metadata.get('capped_forecast', 10.0)

            # Normalize forecast to 0-1 range for sizing
            # Forecast ranges from -20 to +20, so normalize
            forecast_weight = abs(forecast) / self.config['forecast_cap']
            forecast_weight = np.clip(forecast_weight, 0.1, 1.0)  # Min 10% size

            # Calculate volatility-adjusted size
            if current_volatility < 1e-6:
                logger.warning(
                    f"{signal.symbol}: Volatility too low ({current_volatility}), "
                    "using default"
                )
                current_volatility = 0.20  # Default 20%

            # Target risk per trade
            target_vol = self.risk_metrics.volatility_target
            idm = self.config['diversification_multiplier']

            # Carver's formula
            # Capital * Target Vol * IDM * Forecast Weight / (Instrument Vol * Price)
            notional_size = (
                portfolio_value *
                target_vol *
                idm *
                forecast_weight
            ) / (current_volatility * signal.entry_price)

            # Apply position limits
            max_position_value = (
                portfolio_value *
                self.risk_metrics.max_portfolio_allocation
            )
            max_size = max_position_value / signal.entry_price

            # Cap size
            position_size = min(abs(notional_size), max_size)

            # Apply sign based on direction
            if signal.signal_type == SignalType.SHORT:
                position_size = -position_size

            # Check minimum size
            min_size = 0.01  # Minimum fractional size
            if abs(position_size) < min_size:
                logger.debug(
                    f"{signal.symbol}: Position size {position_size:.4f} below minimum"
                )
                return 0.0

            logger.info(
                f"{signal.symbol}: Position size {position_size:.2f} units, "
                f"forecast_weight={forecast_weight:.2f}, "
                f"vol={current_volatility:.2%}"
            )

            return position_size

        except Exception as e:
            logger.error(
                f"Error calculating position size for {signal.symbol}: {e}",
                exc_info=True
            )
            return 0.0

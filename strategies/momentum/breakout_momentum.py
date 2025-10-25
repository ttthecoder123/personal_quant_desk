"""
Breakout Momentum Strategy

Implements a breakout strategy using:
- Donchian channel breakouts (20/55 day highs/lows)
- ATR-based stops and position sizing
- Volume surge confirmation
- Volatility squeeze detection (Bollinger Bands contraction)
- False breakout filtering using ML signals

The strategy identifies momentum breakouts while filtering false signals
through volatility regime detection and ML confirmation.

References:
- "Following the Trend" by Andreas Clenow
- "Turtle Trading" methodology
- Bollinger Bands squeeze (John Bollinger)
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


class BreakoutMomentumStrategy(StrategyBase):
    """
    Breakout momentum strategy using Donchian channels with filters.

    Key Features:
    - Long on breakout above N-day high
    - Short on breakdown below N-day low
    - Volume confirmation (surge above average)
    - Volatility squeeze detection (low volatility precedes breakout)
    - ATR-based stops and targets
    - ML signal filtering to reduce false breakouts
    """

    def __init__(
        self,
        name: str = "BreakoutMomentum",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize breakout momentum strategy.

        Args:
            name: Strategy name
            config: Configuration parameters
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'enabled': True,
            'entry_lookback': 55,  # Breakout period (55 days = Turtle)
            'exit_lookback': 20,  # Exit on opposite 20-day breakout
            'volume_lookback': 20,  # Volume moving average period
            'volume_surge_threshold': 1.5,  # 1.5x average volume
            'min_confidence': 0.60,  # Minimum ML confidence
            'atr_multiplier_stop': 2.0,  # ATR multiplier for stop loss
            'atr_multiplier_target': 3.0,  # ATR multiplier for take profit
            'atr_period': 14,  # ATR calculation period
            'bb_period': 20,  # Bollinger Band period for squeeze
            'bb_std': 2.0,  # Bollinger Band standard deviations
            'squeeze_threshold': 0.02,  # BB bandwidth threshold for squeeze
            'require_volume_confirmation': True,
            'require_squeeze': False,  # Optional squeeze requirement
            'min_bars_since_breakout': 0,  # Avoid chasing
            'max_bars_since_breakout': 5,  # Enter within N bars
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        logger.info(
            f"Initialized {name} with entry lookback {self.config['entry_lookback']}, "
            f"volume threshold {self.config['volume_surge_threshold']}x"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate breakout signal with confirmation filters.

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data
            features: Technical features including ATR, Bollinger Bands
            ml_signals: ML predictions and probabilities

        Returns:
            StrategySignal or None
        """
        try:
            # Validate data
            min_bars = max(
                self.config['entry_lookback'],
                self.config['volume_lookback'],
                self.config['bb_period']
            ) + 10

            if len(market_data) < min_bars:
                logger.warning(
                    f"{symbol}: Insufficient data ({len(market_data)} bars), "
                    f"need at least {min_bars}"
                )
                return None

            # Get latest data
            high_prices = market_data['High'].values
            low_prices = market_data['Low'].values
            close_prices = market_data['Close'].values
            volume = market_data['Volume'].values if 'Volume' in market_data.columns else None

            current_price = close_prices[-1]
            current_high = high_prices[-1]
            current_low = low_prices[-1]
            timestamp = market_data.index[-1]

            # Calculate Donchian channels
            entry_lookback = self.config['entry_lookback']
            upper_channel = np.max(high_prices[-entry_lookback-1:-1])
            lower_channel = np.min(low_prices[-entry_lookback-1:-1])

            # Check for breakout
            is_upper_breakout = current_high > upper_channel
            is_lower_breakout = current_low < lower_channel

            if not (is_upper_breakout or is_lower_breakout):
                return None

            # Determine signal direction
            if is_upper_breakout:
                signal_type = SignalType.LONG
                breakout_level = upper_channel
            else:
                signal_type = SignalType.SHORT
                breakout_level = lower_channel

            logger.debug(
                f"{symbol}: {signal_type.value} breakout detected, "
                f"price={current_price:.2f}, level={breakout_level:.2f}"
            )

            # Volume confirmation
            if self.config['require_volume_confirmation'] and volume is not None:
                if not self._check_volume_surge(volume):
                    logger.debug(f"{symbol}: Volume confirmation failed")
                    return None

            # Volatility squeeze detection
            if self.config['require_squeeze']:
                if features is not None:
                    if not self._check_volatility_squeeze(features):
                        logger.debug(f"{symbol}: No volatility squeeze detected")
                        return None
                else:
                    # Calculate squeeze on the fly if features not provided
                    if not self._calculate_squeeze(close_prices):
                        logger.debug(f"{symbol}: No volatility squeeze detected")
                        return None

            # Check for false breakout using ML signals
            confidence = self._get_ml_confidence(
                signal_type,
                ml_signals,
                timestamp
            )

            if confidence < self.config['min_confidence']:
                logger.debug(
                    f"{symbol}: ML confidence {confidence:.2f} below threshold "
                    f"{self.config['min_confidence']}, likely false breakout"
                )
                return None

            # Calculate ATR for stops and targets
            atr = self._get_atr(features, high_prices, low_prices, close_prices)

            if atr is None or atr < 1e-6:
                logger.warning(f"{symbol}: Invalid ATR, cannot set stops")
                return None

            # Calculate stop loss and take profit
            if signal_type == SignalType.LONG:
                stop_loss = current_price - self.config['atr_multiplier_stop'] * atr
                take_profit = current_price + self.config['atr_multiplier_target'] * atr
            else:  # SHORT
                stop_loss = current_price + self.config['atr_multiplier_stop'] * atr
                take_profit = current_price - self.config['atr_multiplier_target'] * atr

            # Create signal
            signal = StrategySignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=0.0,  # Calculated in calculate_position_size
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'breakout_level': breakout_level,
                    'upper_channel': upper_channel,
                    'lower_channel': lower_channel,
                    'atr': atr,
                    'volume_confirmed': True if volume is not None else False,
                    'strategy': self.name
                }
            )

            logger.info(
                f"{symbol}: {signal_type.value} breakout signal, "
                f"price={current_price:.2f}, "
                f"stop={stop_loss:.2f}, "
                f"target={take_profit:.2f}, "
                f"confidence={confidence:.2f}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def _check_volume_surge(self, volume: np.ndarray) -> bool:
        """
        Check if current volume is above threshold.

        Args:
            volume: Volume array

        Returns:
            True if volume surge detected
        """
        lookback = self.config['volume_lookback']
        threshold = self.config['volume_surge_threshold']

        # Calculate average volume
        avg_volume = np.mean(volume[-lookback-1:-1])

        if avg_volume < 1e-6:
            return False

        # Current volume
        current_volume = volume[-1]

        # Check surge
        return current_volume > (threshold * avg_volume)

    def _check_volatility_squeeze(self, features: pd.DataFrame) -> bool:
        """
        Check for Bollinger Band squeeze (low volatility).

        A squeeze occurs when BB bandwidth is below threshold,
        indicating low volatility that often precedes a breakout.

        Args:
            features: Features DataFrame with Bollinger Bands

        Returns:
            True if squeeze detected
        """
        if 'bb_bandwidth' not in features.columns:
            return False

        bandwidth = features['bb_bandwidth'].iloc[-1]

        if np.isnan(bandwidth):
            return False

        return bandwidth < self.config['squeeze_threshold']

    def _calculate_squeeze(self, close_prices: np.ndarray) -> bool:
        """
        Calculate Bollinger Band squeeze on the fly.

        Args:
            close_prices: Close price array

        Returns:
            True if squeeze detected
        """
        period = self.config['bb_period']

        if len(close_prices) < period:
            return False

        # Calculate Bollinger Bands
        ma = np.mean(close_prices[-period:])
        std = np.std(close_prices[-period:])

        if ma < 1e-6:
            return False

        # Bandwidth as percentage
        bandwidth = (std * self.config['bb_std'] * 2) / ma

        return bandwidth < self.config['squeeze_threshold']

    def _get_atr(
        self,
        features: Optional[pd.DataFrame],
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray
    ) -> Optional[float]:
        """
        Get ATR from features or calculate it.

        Args:
            features: Features DataFrame
            high_prices: High price array
            low_prices: Low price array
            close_prices: Close price array

        Returns:
            ATR value or None
        """
        # Try to get from features first
        if features is not None and 'atr' in features.columns:
            atr = features['atr'].iloc[-1]
            if not np.isnan(atr):
                return float(atr)

        # Calculate ATR manually
        period = self.config['atr_period']

        if len(close_prices) < period + 1:
            return None

        # Calculate True Range
        high_low = high_prices[-period:] - low_prices[-period:]
        high_close = np.abs(high_prices[-period:] - close_prices[-period-1:-1])
        low_close = np.abs(low_prices[-period:] - close_prices[-period-1:-1])

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))

        # ATR is simple average of TR
        atr = np.mean(true_range)

        return float(atr)

    def _get_ml_confidence(
        self,
        signal_type: SignalType,
        ml_signals: Optional[pd.DataFrame],
        timestamp: datetime
    ) -> float:
        """
        Get ML confidence to filter false breakouts.

        Args:
            signal_type: Signal direction
            ml_signals: ML predictions
            timestamp: Signal timestamp

        Returns:
            Confidence score (0-1)
        """
        if ml_signals is None or ml_signals.empty:
            return 0.65  # Default confidence

        try:
            # Get latest ML prediction
            if timestamp in ml_signals.index:
                ml_row = ml_signals.loc[timestamp]
            else:
                ml_row = ml_signals.iloc[-1]

            # Get probability for signal direction
            if signal_type == SignalType.LONG:
                if 'prob_1' in ml_signals.columns:
                    confidence = ml_row['prob_1']
                elif 1 in ml_signals.columns:
                    confidence = ml_row[1]
                else:
                    confidence = 0.65
            elif signal_type == SignalType.SHORT:
                if 'prob_-1' in ml_signals.columns:
                    confidence = ml_row['prob_-1']
                elif -1 in ml_signals.columns:
                    confidence = ml_row[-1]
                else:
                    confidence = 0.65
            else:
                confidence = 0.5

            return float(confidence)

        except Exception as e:
            logger.debug(f"Could not extract ML confidence: {e}")
            return 0.65

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size based on ATR risk.

        Uses fixed fractional position sizing based on ATR stop distance.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Instrument volatility
            existing_positions: Current positions

        Returns:
            Position size in units
        """
        try:
            # Get ATR from metadata
            atr = signal.metadata.get('atr')

            if atr is None or atr < 1e-6:
                logger.warning(f"{signal.symbol}: Invalid ATR for position sizing")
                return 0.0

            # Calculate risk per trade as % of portfolio
            risk_per_trade = self.risk_metrics.position_risk  # e.g., 2%

            # Risk amount in dollars
            risk_amount = portfolio_value * risk_per_trade

            # ATR-based stop distance
            stop_distance = self.config['atr_multiplier_stop'] * atr

            # Position size = Risk Amount / Stop Distance
            position_size = risk_amount / stop_distance

            # Apply portfolio allocation limit
            max_position_value = (
                portfolio_value *
                self.risk_metrics.max_portfolio_allocation
            )
            max_size = max_position_value / signal.entry_price

            # Cap size
            position_size = min(abs(position_size), max_size)

            # Apply confidence scaling
            position_size *= signal.confidence

            # Apply sign for direction
            if signal.signal_type == SignalType.SHORT:
                position_size = -position_size

            # Check minimum size
            min_size = 0.01
            if abs(position_size) < min_size:
                logger.debug(
                    f"{signal.symbol}: Position size {position_size:.4f} below minimum"
                )
                return 0.0

            logger.info(
                f"{signal.symbol}: Position size {position_size:.2f} units, "
                f"risk=${risk_amount:.2f}, "
                f"stop_distance=${stop_distance:.2f}"
            )

            return position_size

        except Exception as e:
            logger.error(
                f"Error calculating position size for {signal.symbol}: {e}",
                exc_info=True
            )
            return 0.0

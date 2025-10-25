"""
Bollinger Band Mean Reversion Strategy

Implements Bollinger Band mean reversion with RSI divergence and volume confirmation.
Uses dynamic band calculation with multiple timeframes and confirmation signals.
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
    PositionSide,
    RiskMetrics
)


class BollingerReversionStrategy(StrategyBase):
    """
    Bollinger Band mean reversion strategy with confirmations.

    Methodology:
    1. Calculate dynamic Bollinger Bands (SMA ± N*std)
    2. Detect RSI divergence for confirmation
    3. Check volume spike for entry confirmation
    4. Scale position based on distance from mean
    5. Integrate with Step 4 meta-labels for bet sizing

    References:
        Bollinger, J. (2002). Bollinger on Bollinger Bands
        Chan, E. (2009). Quantitative Trading
    """

    def __init__(
        self,
        name: str = "BollingerReversion",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize Bollinger Band reversion strategy.

        Args:
            name: Strategy name
            config: Configuration dict with parameters:
                - bb_period: Bollinger Band period (default: 20)
                - bb_std: Number of standard deviations (default: 2.0)
                - rsi_period: RSI calculation period (default: 14)
                - rsi_oversold: RSI oversold level (default: 30)
                - rsi_overbought: RSI overbought level (default: 70)
                - volume_threshold: Volume spike threshold (default: 1.5x average)
                - min_confidence: Minimum meta-label confidence (default: 0.55)
                - use_rsi_divergence: Enable RSI divergence detection (default: True)
                - use_volume_confirmation: Enable volume confirmation (default: True)
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5,
            'min_confidence': 0.55,
            'use_rsi_divergence': True,
            'use_volume_confirmation': True,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        logger.info(f"Initialized {name} with BB period={self.config['bb_period']}, "
                   f"std={self.config['bb_std']}")

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int,
        num_std: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Price series
            period: Moving average period
            num_std: Number of standard deviations

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()

        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)

        return upper_band, middle_band, lower_band

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI series
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate exponential moving averages
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def detect_rsi_divergence(
        self,
        prices: pd.Series,
        rsi: pd.Series,
        lookback: int = 5
    ) -> tuple[bool, bool]:
        """
        Detect bullish or bearish RSI divergence.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high

        Args:
            prices: Price series
            rsi: RSI series
            lookback: Lookback period for divergence detection

        Returns:
            Tuple of (bullish_divergence, bearish_divergence)
        """
        if len(prices) < lookback * 2:
            return False, False

        try:
            recent_prices = prices.iloc[-lookback:]
            recent_rsi = rsi.iloc[-lookback:]

            # Find local extrema
            price_min_idx = recent_prices.idxmin()
            price_max_idx = recent_prices.idxmax()
            rsi_min_idx = recent_rsi.idxmin()
            rsi_max_idx = recent_rsi.idxmax()

            # Check for bullish divergence
            bullish_div = False
            if len(prices) > lookback * 2:
                prev_prices = prices.iloc[-(lookback*2):-lookback]
                prev_rsi = rsi.iloc[-(lookback*2):-lookback]

                if len(prev_prices) > 0 and len(prev_rsi) > 0:
                    prev_price_min = prev_prices.min()
                    prev_rsi_min = prev_rsi.min()
                    curr_price_min = recent_prices.min()
                    curr_rsi_min = recent_rsi.min()

                    # Price lower low, RSI higher low
                    if curr_price_min < prev_price_min and curr_rsi_min > prev_rsi_min:
                        bullish_div = True

            # Check for bearish divergence
            bearish_div = False
            if len(prices) > lookback * 2:
                prev_prices = prices.iloc[-(lookback*2):-lookback]
                prev_rsi = rsi.iloc[-(lookback*2):-lookback]

                if len(prev_prices) > 0 and len(prev_rsi) > 0:
                    prev_price_max = prev_prices.max()
                    prev_rsi_max = prev_rsi.max()
                    curr_price_max = recent_prices.max()
                    curr_rsi_max = recent_rsi.max()

                    # Price higher high, RSI lower high
                    if curr_price_max > prev_price_max and curr_rsi_max < prev_rsi_max:
                        bearish_div = True

            return bullish_div, bearish_div

        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            return False, False

    def check_volume_confirmation(
        self,
        volumes: pd.Series,
        period: int = 20,
        threshold: float = 1.5
    ) -> bool:
        """
        Check if current volume exceeds threshold of average volume.

        Args:
            volumes: Volume series
            period: Period for average calculation
            threshold: Threshold multiplier (e.g., 1.5 = 150% of average)

        Returns:
            True if volume spike detected
        """
        if len(volumes) < period + 1:
            return False

        avg_volume = volumes.iloc[-period:-1].mean()
        current_volume = volumes.iloc[-1]

        return current_volume > (avg_volume * threshold)

    def calculate_band_width(
        self,
        upper_band: pd.Series,
        lower_band: pd.Series,
        middle_band: pd.Series
    ) -> pd.Series:
        """
        Calculate Bollinger Band width (volatility indicator).

        Args:
            upper_band: Upper band series
            lower_band: Lower band series
            middle_band: Middle band series

        Returns:
            Band width series
        """
        return (upper_band - lower_band) / middle_band

    def calculate_percent_b(
        self,
        price: float,
        upper_band: float,
        lower_band: float
    ) -> float:
        """
        Calculate %B indicator (position within bands).

        %B = (Price - Lower Band) / (Upper Band - Lower Band)

        Args:
            price: Current price
            upper_band: Upper band value
            lower_band: Lower band value

        Returns:
            %B value (0 = lower band, 1 = upper band)
        """
        if upper_band == lower_band:
            return 0.5

        return (price - lower_band) / (upper_band - lower_band)

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate Bollinger Band reversion signal with confirmations.

        Args:
            symbol: Instrument symbol
            market_data: OHLCV DataFrame with columns: open, high, low, close, volume
            features: Engineered features (optional)
            ml_signals: ML signals with meta-labels for confidence

        Returns:
            StrategySignal or None
        """
        try:
            required_length = max(self.config['bb_period'], self.config['rsi_period']) + 20

            if market_data is None or len(market_data) < required_length:
                logger.debug(f"Insufficient data for {symbol}")
                return None

            # Extract price and volume data
            if 'close' not in market_data.columns:
                logger.warning(f"Missing 'close' column for {symbol}")
                return None

            close_prices = market_data['close']
            volumes = market_data.get('volume', pd.Series([0] * len(market_data)))

            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
                close_prices,
                self.config['bb_period'],
                self.config['bb_std']
            )

            # Calculate RSI
            rsi = self.calculate_rsi(close_prices, self.config['rsi_period'])

            # Get current values
            current_price = close_prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_middle = middle_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_rsi = rsi.iloc[-1]

            # Calculate indicators
            percent_b = self.calculate_percent_b(current_price, current_upper, current_lower)
            band_width = self.calculate_band_width(upper_band, lower_band, middle_band).iloc[-1]

            # Detect RSI divergence if enabled
            bullish_div = False
            bearish_div = False
            if self.config['use_rsi_divergence']:
                bullish_div, bearish_div = self.detect_rsi_divergence(
                    close_prices, rsi, lookback=5
                )

            # Check volume confirmation if enabled
            volume_confirmed = True
            if self.config['use_volume_confirmation']:
                volume_confirmed = self.check_volume_confirmation(
                    volumes,
                    period=20,
                    threshold=self.config['volume_threshold']
                )

            # Generate signals
            signal_type = None
            confidence = 0.5

            # Check existing position
            existing_position = self.positions.get(symbol)

            if existing_position is None:
                # Entry signals
                # Long signal: Price below lower band + oversold RSI + confirmations
                if (current_price < current_lower and
                    current_rsi < self.config['rsi_oversold']):

                    signal_type = SignalType.LONG

                    # Base confidence from band penetration
                    penetration = abs(percent_b)  # How far below 0
                    confidence = min(0.6 + penetration * 0.4, 1.0)

                    # Boost confidence with confirmations
                    if bullish_div:
                        confidence = min(confidence + 0.1, 1.0)
                    if volume_confirmed:
                        confidence = min(confidence + 0.05, 1.0)

                # Short signal: Price above upper band + overbought RSI + confirmations
                elif (current_price > current_upper and
                      current_rsi > self.config['rsi_overbought']):

                    signal_type = SignalType.SHORT

                    # Base confidence from band penetration
                    penetration = abs(percent_b - 1.0)  # How far above 1
                    confidence = min(0.6 + penetration * 0.4, 1.0)

                    # Boost confidence with confirmations
                    if bearish_div:
                        confidence = min(confidence + 0.1, 1.0)
                    if volume_confirmed:
                        confidence = min(confidence + 0.05, 1.0)

            else:
                # Exit signals - mean reversion to middle band
                position_side = existing_position['side']

                if position_side == PositionSide.LONG:
                    # Exit long when price crosses above middle band
                    if current_price > current_middle or current_rsi > 60:
                        signal_type = SignalType.EXIT_LONG
                        confidence = 0.7

                elif position_side == PositionSide.SHORT:
                    # Exit short when price crosses below middle band
                    if current_price < current_middle or current_rsi < 40:
                        signal_type = SignalType.EXIT_SHORT
                        confidence = 0.7

            if signal_type is None:
                return None

            # Integrate meta-labels from Step 4 if available
            if ml_signals is not None and len(ml_signals) > 0:
                latest_ml = ml_signals.iloc[-1]
                if 'meta_label_prob' in latest_ml:
                    ml_confidence = latest_ml['meta_label_prob']
                    # Blend strategy confidence with ML confidence
                    confidence = 0.5 * confidence + 0.5 * ml_confidence

            # Calculate stop loss and take profit
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                # Stop loss at opposite band
                if signal_type == SignalType.LONG:
                    stop_loss = current_lower * 0.98  # 2% below lower band
                    take_profit = current_middle * 1.01  # 1% above middle
                else:  # SHORT
                    stop_loss = current_upper * 1.02  # 2% above upper band
                    take_profit = current_middle * 0.99  # 1% below middle
            else:
                stop_loss = None
                take_profit = None

            # Calculate position scaling based on distance from mean
            distance_from_mean = abs(current_price - current_middle) / current_middle
            position_scale = min(distance_from_mean / 0.05, 1.5)  # Max 1.5x for 5% deviation

            signal = StrategySignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=position_scale,  # Scaled size
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'upper_band': current_upper,
                    'middle_band': current_middle,
                    'lower_band': current_lower,
                    'percent_b': percent_b,
                    'band_width': band_width,
                    'rsi': current_rsi,
                    'bullish_divergence': bullish_div,
                    'bearish_divergence': bearish_div,
                    'volume_confirmed': volume_confirmed,
                    'distance_from_mean': distance_from_mean,
                    'position_scale': position_scale
                }
            )

            self.signals_history.append(signal)
            logger.info(f"Generated {signal_type.value} signal for {symbol}: "
                       f"price={current_price:.2f}, %B={percent_b:.2f}, "
                       f"RSI={current_rsi:.1f}, conf={confidence:.2f}")

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size with volatility targeting and distance-based scaling.

        Position size is scaled based on:
        1. Portfolio volatility target
        2. Current market volatility (band width)
        3. Distance from mean (confidence in mean reversion)
        4. Meta-label confidence from Step 4

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Current volatility
            existing_positions: Current positions

        Returns:
            Position size in dollars
        """
        try:
            # Base position size from volatility targeting
            vol_target = self.risk_metrics.volatility_target

            if current_volatility < 1e-8:
                current_volatility = 0.15

            base_size = (portfolio_value * vol_target) / current_volatility

            # Scale by signal confidence
            confidence_scale = signal.confidence ** 2  # Quadratic scaling
            adjusted_size = base_size * confidence_scale

            # Scale by position_scale from signal (distance from mean)
            if signal.size > 0:
                adjusted_size *= signal.size

            # Apply maximum position limits
            max_position = portfolio_value * self.risk_metrics.max_portfolio_allocation
            adjusted_size = min(adjusted_size, max_position)

            # Ensure minimum viable size
            min_size = portfolio_value * 0.01
            adjusted_size = max(adjusted_size, min_size)

            logger.debug(f"Position size for {signal.symbol}: ${adjusted_size:,.0f} "
                        f"(conf={signal.confidence:.2f}, scale={signal.size:.2f})")

            return adjusted_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return portfolio_value * 0.05

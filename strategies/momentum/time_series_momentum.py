"""
Time Series Momentum Strategy

Implements sign-based momentum strategy based on Moskowitz, Ooi, and Pedersen (2012):
"Time Series Momentum"

Key insights:
- Absolute (time series) momentum is distinct from relative (cross-sectional) momentum
- Sign of past returns predicts future returns across asset classes
- Works best with multiple lookback periods
- Volatility scaling improves Sharpe ratios
- Trend persistence measures help filter false signals

References:
- Moskowitz, Ooi, Pedersen (2012) - "Time Series Momentum"
- Antonacci - "Dual Momentum Investing"
"""

from datetime import datetime
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
from loguru import logger

from strategies.base.strategy_base import (
    StrategyBase,
    StrategySignal,
    SignalType,
    RiskMetrics
)


class TimeSeriesMomentumStrategy(StrategyBase):
    """
    Time series momentum strategy with sign-based signals.

    Key Features:
    - Multiple lookback periods (1m, 3m, 6m, 12m)
    - Sign-based momentum (positive/negative returns)
    - Volatility scaling for position sizing
    - Trend persistence measures
    - ML confirmation for signal quality
    - Ex-ante volatility targeting
    """

    def __init__(
        self,
        name: str = "TimeSeriesMomentum",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize time series momentum strategy.

        Args:
            name: Strategy name
            config: Configuration parameters
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'enabled': True,
            'lookback_periods': [21, 63, 126, 252],  # 1m, 3m, 6m, 12m
            'lookback_weights': [0.25, 0.25, 0.25, 0.25],  # Equal weight
            'sign_threshold': 0.0,  # Return must be > 0 for long signal
            'min_confidence': 0.55,  # Minimum ML confidence
            'vol_lookback': 60,  # Volatility estimation period
            'vol_target': 0.15,  # Target volatility (15% annualized)
            'min_return_threshold': 0.01,  # Minimum 1% return to trigger
            'trend_persistence_lookback': 10,  # Days to check persistence
            'trend_persistence_threshold': 0.6,  # 60% days in same direction
            'use_skip_month': True,  # Skip most recent month (Moskowitz)
            'skip_days': 21,  # Days to skip from most recent
            'combine_signals': 'vote',  # 'vote', 'average', or 'unanimous'
            'require_ml_confirmation': True,
            'risk_scaling': 'volatility',  # 'volatility' or 'fixed'
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Tracking
        self.signal_history: Dict[str, List[int]] = {}  # symbol -> list of signals

        logger.info(
            f"Initialized {name} with periods {self.config['lookback_periods']}, "
            f"vol target {self.config['vol_target']:.1%}"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate time series momentum signal.

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data
            features: Technical features
            ml_signals: ML predictions and probabilities

        Returns:
            StrategySignal or None
        """
        try:
            # Validate data
            max_lookback = max(self.config['lookback_periods'])
            if self.config['use_skip_month']:
                max_lookback += self.config['skip_days']

            min_bars = max_lookback + self.config['vol_lookback'] + 10

            if len(market_data) < min_bars:
                logger.warning(
                    f"{symbol}: Insufficient data ({len(market_data)} bars), "
                    f"need at least {min_bars}"
                )
                return None

            # Get price data
            close_prices = market_data['Close'].values
            current_price = close_prices[-1]
            timestamp = market_data.index[-1]

            # Calculate momentum signals for each period
            momentum_signals = []
            momentum_returns = []

            for lookback in self.config['lookback_periods']:
                signal, ret = self._calculate_period_momentum(
                    close_prices,
                    lookback,
                    self.config['skip_days'] if self.config['use_skip_month'] else 0
                )
                momentum_signals.append(signal)
                momentum_returns.append(ret)

            # Combine signals
            combined_signal = self._combine_signals(momentum_signals)

            if combined_signal == 0:  # No signal
                return None

            # Determine signal type
            signal_type = SignalType.LONG if combined_signal > 0 else SignalType.SHORT

            # Check trend persistence
            if not self._check_trend_persistence(close_prices):
                logger.debug(f"{symbol}: Trend persistence check failed")
                return None

            # Get ML confidence
            confidence = self._get_ml_confidence(
                signal_type,
                ml_signals,
                timestamp
            )

            # Check minimum confidence
            if self.config['require_ml_confirmation']:
                if confidence < self.config['min_confidence']:
                    logger.debug(
                        f"{symbol}: ML confidence {confidence:.2f} below threshold "
                        f"{self.config['min_confidence']}"
                    )
                    return None

            # Calculate current volatility for position sizing
            current_volatility = self._calculate_volatility(close_prices)

            # Calculate stop loss and take profit
            stop_loss, take_profit = None, None
            if features is not None and 'atr' in features.columns:
                atr = features['atr'].iloc[-1]
                if not np.isnan(atr):
                    # 2 ATR stop, 3 ATR target
                    if signal_type == SignalType.LONG:
                        stop_loss = current_price - 2 * atr
                        take_profit = current_price + 3 * atr
                    else:
                        stop_loss = current_price + 2 * atr
                        take_profit = current_price - 3 * atr

            # Track signal history
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            self.signal_history[symbol].append(combined_signal)
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]

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
                    'momentum_signals': momentum_signals,
                    'momentum_returns': momentum_returns,
                    'combined_signal': combined_signal,
                    'current_volatility': current_volatility,
                    'strategy': self.name
                }
            )

            logger.info(
                f"{symbol}: {signal_type.value} signal, "
                f"signals={momentum_signals}, "
                f"confidence={confidence:.2f}, "
                f"vol={current_volatility:.2%}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def _calculate_period_momentum(
        self,
        prices: np.ndarray,
        lookback: int,
        skip_days: int
    ) -> tuple:
        """
        Calculate momentum signal for a single period.

        Args:
            prices: Price array
            lookback: Lookback period in days
            skip_days: Days to skip from most recent

        Returns:
            Tuple of (signal, return) where signal is -1, 0, or 1
        """
        # Calculate return over lookback period, skipping recent days
        if len(prices) < lookback + skip_days:
            return 0, 0.0

        # End price (skip recent if configured)
        end_idx = -1 - skip_days if skip_days > 0 else -1
        start_idx = end_idx - lookback

        end_price = prices[end_idx]
        start_price = prices[start_idx]

        # Calculate return
        ret = (end_price / start_price) - 1.0

        # Generate signal based on sign
        threshold = self.config['sign_threshold']
        min_return = self.config['min_return_threshold']

        if ret > threshold and abs(ret) > min_return:
            signal = 1  # Long
        elif ret < -threshold and abs(ret) > min_return:
            signal = -1  # Short
        else:
            signal = 0  # No signal

        return signal, ret

    def _combine_signals(self, signals: List[int]) -> int:
        """
        Combine multiple momentum signals into one.

        Args:
            signals: List of signals (-1, 0, 1)

        Returns:
            Combined signal (-1, 0, 1)
        """
        method = self.config['combine_signals']

        if method == 'vote':
            # Majority vote
            long_votes = sum(1 for s in signals if s > 0)
            short_votes = sum(1 for s in signals if s < 0)

            if long_votes > short_votes:
                return 1
            elif short_votes > long_votes:
                return -1
            else:
                return 0

        elif method == 'average':
            # Average and threshold
            avg = np.mean(signals)
            if avg > 0.5:
                return 1
            elif avg < -0.5:
                return -1
            else:
                return 0

        elif method == 'unanimous':
            # All signals must agree
            if all(s > 0 for s in signals if s != 0):
                return 1
            elif all(s < 0 for s in signals if s != 0):
                return -1
            else:
                return 0

        else:
            # Default to vote
            return self._combine_signals(signals)

    def _check_trend_persistence(self, prices: np.ndarray) -> bool:
        """
        Check if trend is persistent over recent period.

        Args:
            prices: Price array

        Returns:
            True if trend is persistent
        """
        lookback = self.config['trend_persistence_lookback']
        threshold = self.config['trend_persistence_threshold']

        if len(prices) < lookback + 1:
            return True  # Not enough data, pass check

        # Calculate daily returns
        recent_prices = prices[-lookback-1:]
        daily_returns = np.diff(recent_prices) / recent_prices[:-1]

        # Count positive days
        positive_days = np.sum(daily_returns > 0)
        negative_days = np.sum(daily_returns < 0)

        # Calculate persistence as fraction of days in dominant direction
        total_days = len(daily_returns)
        max_days = max(positive_days, negative_days)
        persistence = max_days / total_days

        return persistence >= threshold

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Calculate annualized volatility.

        Args:
            prices: Price array

        Returns:
            Annualized volatility
        """
        lookback = self.config['vol_lookback']

        if len(prices) < lookback + 1:
            return 0.20  # Default 20%

        # Calculate returns
        recent_prices = prices[-lookback-1:]
        returns = np.diff(recent_prices) / recent_prices[:-1]

        # Calculate volatility
        vol = np.std(returns)

        # Annualize (assuming daily data)
        annual_vol = vol * np.sqrt(252)

        return float(annual_vol)

    def _get_ml_confidence(
        self,
        signal_type: SignalType,
        ml_signals: Optional[pd.DataFrame],
        timestamp: datetime
    ) -> float:
        """
        Get ML confidence for signal.

        Args:
            signal_type: Signal direction
            ml_signals: ML predictions
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

            # Get probability for signal direction
            if signal_type == SignalType.LONG:
                if 'prob_1' in ml_signals.columns:
                    confidence = ml_row['prob_1']
                elif 1 in ml_signals.columns:
                    confidence = ml_row[1]
                else:
                    confidence = 0.6
            elif signal_type == SignalType.SHORT:
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
            return 0.6

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate volatility-scaled position size.

        Uses inverse volatility scaling: higher volatility = smaller position.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Instrument volatility (annualized)
            existing_positions: Current positions

        Returns:
            Position size in units
        """
        try:
            # Get volatility from signal metadata if available
            if 'current_volatility' in signal.metadata:
                current_volatility = signal.metadata['current_volatility']

            if current_volatility < 1e-6:
                logger.warning(
                    f"{signal.symbol}: Volatility too low ({current_volatility:.4f}), "
                    "using default"
                )
                current_volatility = 0.20

            # Target volatility
            target_vol = self.config['vol_target']

            # Volatility scaling factor
            vol_scalar = target_vol / current_volatility

            # Base position size as % of portfolio
            base_allocation = self.risk_metrics.max_portfolio_allocation

            # Calculate position value
            position_value = portfolio_value * base_allocation * vol_scalar

            # Apply confidence scaling
            position_value *= signal.confidence

            # Convert to units
            position_size = position_value / signal.entry_price

            # Apply portfolio limit
            max_position_value = (
                portfolio_value *
                self.risk_metrics.max_portfolio_allocation
            )
            max_size = max_position_value / signal.entry_price

            # Cap size
            position_size = min(abs(position_size), max_size)

            # Apply signal direction
            if signal.signal_type == SignalType.SHORT:
                position_size = -position_size

            # Check minimum
            min_size = 0.01
            if abs(position_size) < min_size:
                logger.debug(
                    f"{signal.symbol}: Position size {position_size:.4f} below minimum"
                )
                return 0.0

            logger.info(
                f"{signal.symbol}: Position size {position_size:.2f} units, "
                f"vol_scalar={vol_scalar:.2f}, "
                f"vol={current_volatility:.2%}"
            )

            return position_size

        except Exception as e:
            logger.error(
                f"Error calculating position size for {signal.symbol}: {e}",
                exc_info=True
            )
            return 0.0

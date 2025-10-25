"""
Regime Switching Strategy

Adapts trading approach based on detected market regime:
- Market regime detection using HMM or volatility-based methods
- Different sub-strategies for different regimes
- Smooth transitions to avoid whipsaw
- Regime confidence scoring
"""

from datetime import datetime
from enum import Enum
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


class MarketRegime(Enum):
    """Market regime types."""
    BULL_LOW_VOL = "BULL_LOW_VOL"  # Trending up, low volatility
    BULL_HIGH_VOL = "BULL_HIGH_VOL"  # Trending up, high volatility
    BEAR_LOW_VOL = "BEAR_LOW_VOL"  # Trending down, low volatility
    BEAR_HIGH_VOL = "BEAR_HIGH_VOL"  # Trending down, high volatility
    SIDEWAYS_LOW_VOL = "SIDEWAYS_LOW_VOL"  # Range-bound, low volatility
    SIDEWAYS_HIGH_VOL = "SIDEWAYS_HIGH_VOL"  # Range-bound, high volatility
    UNKNOWN = "UNKNOWN"


class RegimeSwitchingStrategy(StrategyBase):
    """
    Regime-adaptive trading strategy.

    Approach:
    1. Detect current market regime
    2. Select appropriate sub-strategy for regime
    3. Apply smooth transition when regime changes
    4. Track regime confidence
    """

    def __init__(
        self,
        name: str = "RegimeSwitching",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize regime switching strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - regime_detection_method: 'volatility' or 'hmm' (default: 'volatility')
                - vol_lookback: Lookback for volatility calculation (default: 30)
                - trend_lookback: Lookback for trend detection (default: 50)
                - vol_threshold_low: Low vol threshold percentile (default: 0.33)
                - vol_threshold_high: High vol threshold percentile (default: 0.67)
                - trend_threshold: Minimum trend strength (default: 0.02)
                - regime_confirmation_periods: Periods to confirm regime (default: 3)
                - transition_smoothing: Smooth position changes on regime switch (default: True)
            risk_metrics: Risk limits
        """
        default_config = {
            'regime_detection_method': 'volatility',
            'vol_lookback': 30,
            'trend_lookback': 50,
            'vol_threshold_low': 0.33,
            'vol_threshold_high': 0.67,
            'trend_threshold': 0.02,
            'regime_confirmation_periods': 3,
            'transition_smoothing': True,
            'min_observations': 60,
            'min_confidence': 0.60,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Regime tracking
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN
        self.regime_confidence: float = 0.0
        self.regime_history: List[MarketRegime] = []
        self.regime_probabilities: Dict[MarketRegime, float] = {}

        # Sub-strategy performance
        self.regime_performance: Dict[MarketRegime, List[float]] = {
            regime: [] for regime in MarketRegime
        }

        logger.info(
            f"Initialized {name} with detection={self.config['regime_detection_method']}, "
            f"vol_lookback={self.config['vol_lookback']}d"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate regime-adaptive signal.

        Process:
        1. Detect current market regime
        2. Select appropriate strategy for regime
        3. Generate signal using regime-specific logic
        4. Apply transition smoothing if regime changed

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data
            features: Engineered features
            ml_signals: ML predictions

        Returns:
            StrategySignal or None
        """
        try:
            if len(market_data) < self.config['min_observations']:
                logger.warning(f"{symbol}: Insufficient data")
                return None

            # Detect regime
            regime, confidence = self._detect_regime(market_data, features)

            # Update regime tracking
            regime_changed = regime != self.current_regime
            self.current_regime = regime
            self.regime_confidence = confidence
            self.regime_history.append(regime)

            # Keep history manageable
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]

            logger.debug(
                f"{symbol}: Regime={regime.value}, confidence={confidence:.2%}, "
                f"changed={regime_changed}"
            )

            # Generate regime-specific signal
            signal_type, regime_confidence = self._generate_regime_signal(
                regime,
                symbol,
                market_data,
                features,
                ml_signals
            )

            if signal_type == SignalType.HOLD:
                return None

            # Get ML confidence if available
            ml_confidence = self._get_ml_confidence(ml_signals)

            # Combine regime confidence with ML confidence
            combined_confidence = 0.5 * confidence + 0.3 * regime_confidence + 0.2 * ml_confidence

            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()
            current_price = float(market_data['close'].iloc[-1])

            # Calculate stops based on regime
            stop_loss, take_profit = self._calculate_regime_stops(
                current_price,
                signal_type,
                regime,
                market_data
            )

            signal = StrategySignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                confidence=combined_confidence,
                size=0.0,  # Calculated in calculate_position_size
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'strategy': self.name,
                    'regime': regime.value,
                    'regime_confidence': confidence,
                    'regime_changed': regime_changed,
                    'regime_probabilities': self.regime_probabilities.copy()
                }
            )

            self.signals_history.append(signal)
            logger.info(
                f"{symbol} regime signal: {signal_type.value} in {regime.value}, "
                f"confidence={combined_confidence:.2%}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating regime signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size adapted to regime.

        Different regimes warrant different position sizing:
        - High vol regimes: smaller positions
        - Bull regimes: potentially larger positions
        - Transition periods: reduced positions

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_volatility: Current instrument volatility
            existing_positions: Current positions

        Returns:
            Position size in units
        """
        try:
            if portfolio_value <= 0:
                return 0.0

            # Base position size
            risk_budget = self.config.get('risk_budget', 0.05)
            base_position_value = portfolio_value * risk_budget

            # Regime adjustment
            regime = MarketRegime(signal.metadata['regime'])
            regime_scalar = self._get_regime_size_scalar(regime)
            position_value = base_position_value * regime_scalar

            # Volatility scaling
            vol_scalar = 0.20 / max(current_volatility, 0.01)
            position_value *= vol_scalar

            # Confidence scaling
            position_value *= signal.confidence

            # Transition smoothing
            if signal.metadata.get('regime_changed', False) and self.config['transition_smoothing']:
                transition_scalar = 0.5  # 50% size during transition
                position_value *= transition_scalar
                logger.debug(f"{signal.symbol}: Applying transition smoothing (50% size)")

            # Convert to shares
            position_size = position_value / signal.entry_price

            # Apply portfolio allocation limit
            max_position_value = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_size = max_position_value / signal.entry_price
            position_size = min(position_size, max_size)

            logger.debug(
                f"{signal.symbol} regime size: {position_size:.2f} units "
                f"(regime={regime.value}, scalar={regime_scalar:.2f})"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating regime position size: {e}", exc_info=True)
            return 0.0

    def _detect_regime(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> tuple[MarketRegime, float]:
        """
        Detect current market regime.

        Args:
            market_data: Price data
            features: Features

        Returns:
            (regime, confidence)
        """
        method = self.config['regime_detection_method']

        if method == 'hmm':
            return self._detect_regime_hmm(market_data, features)
        else:
            return self._detect_regime_volatility(market_data, features)

    def _detect_regime_volatility(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> tuple[MarketRegime, float]:
        """
        Detect regime using volatility and trend analysis.

        Args:
            market_data: Price data
            features: Features

        Returns:
            (regime, confidence)
        """
        try:
            # Calculate volatility
            vol_lookback = self.config['vol_lookback']
            returns = np.log(market_data['close'] / market_data['close'].shift(1))
            returns = returns.dropna().tail(vol_lookback * 3)  # Need history for percentiles

            if len(returns) < vol_lookback:
                return MarketRegime.UNKNOWN, 0.0

            current_vol = returns.tail(vol_lookback).std() * np.sqrt(252)
            historical_vols = returns.rolling(vol_lookback).std() * np.sqrt(252)
            historical_vols = historical_vols.dropna()

            # Determine volatility regime
            vol_percentile = (historical_vols < current_vol).sum() / len(historical_vols)

            if vol_percentile < self.config['vol_threshold_low']:
                vol_regime = "LOW"
            elif vol_percentile > self.config['vol_threshold_high']:
                vol_regime = "HIGH"
            else:
                vol_regime = "MEDIUM"

            # Calculate trend
            trend_lookback = self.config['trend_lookback']
            recent_prices = market_data['close'].tail(trend_lookback)

            if len(recent_prices) < trend_lookback:
                return MarketRegime.UNKNOWN, 0.0

            # Linear regression trend
            x = np.arange(len(recent_prices))
            y = recent_prices.values
            trend_slope = np.polyfit(x, y, 1)[0]
            trend_strength = abs(trend_slope) / recent_prices.mean()

            if trend_strength > self.config['trend_threshold']:
                if trend_slope > 0:
                    trend_regime = "BULL"
                else:
                    trend_regime = "BEAR"
            else:
                trend_regime = "SIDEWAYS"

            # Combine trend and volatility
            if trend_regime == "BULL" and vol_regime == "LOW":
                regime = MarketRegime.BULL_LOW_VOL
            elif trend_regime == "BULL" and vol_regime == "HIGH":
                regime = MarketRegime.BULL_HIGH_VOL
            elif trend_regime == "BEAR" and vol_regime == "LOW":
                regime = MarketRegime.BEAR_LOW_VOL
            elif trend_regime == "BEAR" and vol_regime == "HIGH":
                regime = MarketRegime.BEAR_HIGH_VOL
            elif trend_regime == "SIDEWAYS" and vol_regime == "LOW":
                regime = MarketRegime.SIDEWAYS_LOW_VOL
            elif trend_regime == "SIDEWAYS" and vol_regime == "HIGH":
                regime = MarketRegime.SIDEWAYS_HIGH_VOL
            else:
                regime = MarketRegime.SIDEWAYS_LOW_VOL  # Default

            # Calculate confidence based on how clear the regime is
            vol_confidence = abs(vol_percentile - 0.5) * 2  # 0-1 scale
            trend_confidence = min(trend_strength / self.config['trend_threshold'], 1.0)
            confidence = 0.5 * vol_confidence + 0.5 * trend_confidence

            # Store probabilities (simplified)
            self.regime_probabilities = {
                regime: confidence,
                MarketRegime.UNKNOWN: 1 - confidence
            }

            return regime, float(confidence)

        except Exception as e:
            logger.error(f"Error detecting regime: {e}", exc_info=True)
            return MarketRegime.UNKNOWN, 0.0

    def _detect_regime_hmm(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> tuple[MarketRegime, float]:
        """
        Detect regime using Hidden Markov Model.

        Note: Requires hmmlearn library. Falls back to volatility method if not available.

        Args:
            market_data: Price data
            features: Features

        Returns:
            (regime, confidence)
        """
        try:
            from hmmlearn import hmm

            # Calculate returns
            returns = np.log(market_data['close'] / market_data['close'].shift(1))
            returns = returns.dropna().tail(252)  # 1 year

            if len(returns) < 100:
                return self._detect_regime_volatility(market_data, features)

            # Fit Gaussian HMM with 3 states
            model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
            X = returns.values.reshape(-1, 1)
            model.fit(X)

            # Predict current state
            current_state = model.predict(X[-10:].reshape(-1, 1))[-1]
            state_probs = model.predict_proba(X[-1].reshape(-1, 1))[0]

            # Map HMM states to regimes
            # State with lowest mean = bear, highest = bull, middle = sideways
            state_means = model.means_.flatten()
            sorted_states = np.argsort(state_means)

            if current_state == sorted_states[0]:
                regime = MarketRegime.BEAR_LOW_VOL
            elif current_state == sorted_states[2]:
                regime = MarketRegime.BULL_LOW_VOL
            else:
                regime = MarketRegime.SIDEWAYS_LOW_VOL

            confidence = float(state_probs[current_state])

            self.regime_probabilities = {
                MarketRegime.BEAR_LOW_VOL: float(state_probs[sorted_states[0]]),
                MarketRegime.SIDEWAYS_LOW_VOL: float(state_probs[sorted_states[1]]),
                MarketRegime.BULL_LOW_VOL: float(state_probs[sorted_states[2]])
            }

            return regime, confidence

        except ImportError:
            logger.warning("hmmlearn not available, falling back to volatility-based detection")
            return self._detect_regime_volatility(market_data, features)
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}", exc_info=True)
            return self._detect_regime_volatility(market_data, features)

    def _generate_regime_signal(
        self,
        regime: MarketRegime,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame],
        ml_signals: Optional[pd.DataFrame]
    ) -> tuple[SignalType, float]:
        """
        Generate signal appropriate for the current regime.

        Args:
            regime: Current market regime
            symbol: Instrument symbol
            market_data: Price data
            features: Features
            ml_signals: ML predictions

        Returns:
            (signal_type, confidence)
        """
        try:
            # Get ML signal if available
            ml_prediction, ml_conf = self._extract_ml_signal(ml_signals)

            # Regime-specific logic
            if regime == MarketRegime.BULL_LOW_VOL:
                # Favorable for trend following - go LONG
                return SignalType.LONG, 0.80

            elif regime == MarketRegime.BULL_HIGH_VOL:
                # Bullish but volatile - use ML guidance
                if ml_prediction == 1:
                    return SignalType.LONG, 0.65
                else:
                    return SignalType.HOLD, 0.0

            elif regime == MarketRegime.BEAR_LOW_VOL:
                # Clear downtrend - SHORT or stay out
                return SignalType.SHORT, 0.75

            elif regime == MarketRegime.BEAR_HIGH_VOL:
                # Bearish and volatile - reduce exposure
                if ml_prediction == -1:
                    return SignalType.SHORT, 0.60
                else:
                    return SignalType.HOLD, 0.0

            elif regime == MarketRegime.SIDEWAYS_LOW_VOL:
                # Range-bound, low vol - mean reversion
                if features is not None and 'rsi' in features.columns:
                    rsi = features['rsi'].iloc[-1]
                    if rsi < 30:
                        return SignalType.LONG, 0.70
                    elif rsi > 70:
                        return SignalType.SHORT, 0.70
                return SignalType.HOLD, 0.0

            elif regime == MarketRegime.SIDEWAYS_HIGH_VOL:
                # Choppy market - reduce activity
                return SignalType.HOLD, 0.0

            else:
                return SignalType.HOLD, 0.0

        except Exception as e:
            logger.error(f"Error generating regime signal: {e}")
            return SignalType.HOLD, 0.0

    def _get_regime_size_scalar(self, regime: MarketRegime) -> float:
        """
        Get position size scalar for regime.

        Args:
            regime: Market regime

        Returns:
            Size scalar (0.5 - 1.5)
        """
        scalars = {
            MarketRegime.BULL_LOW_VOL: 1.3,  # Favorable - larger positions
            MarketRegime.BULL_HIGH_VOL: 0.9,  # Volatile - smaller
            MarketRegime.BEAR_LOW_VOL: 1.0,  # Normal
            MarketRegime.BEAR_HIGH_VOL: 0.7,  # Very volatile - much smaller
            MarketRegime.SIDEWAYS_LOW_VOL: 1.1,  # Decent for mean reversion
            MarketRegime.SIDEWAYS_HIGH_VOL: 0.6,  # Choppy - avoid large positions
            MarketRegime.UNKNOWN: 0.5  # Unknown - very conservative
        }

        return scalars.get(regime, 1.0)

    def _calculate_regime_stops(
        self,
        current_price: float,
        signal_type: SignalType,
        regime: MarketRegime,
        market_data: pd.DataFrame
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate stops based on regime.

        Args:
            current_price: Current price
            signal_type: Signal type
            regime: Market regime
            market_data: Price data

        Returns:
            (stop_loss, take_profit)
        """
        try:
            # Calculate ATR for dynamic stops
            if len(market_data) >= 14:
                high_low = market_data['high'] - market_data['low']
                high_close = abs(market_data['high'] - market_data['close'].shift(1))
                low_close = abs(market_data['low'] - market_data['close'].shift(1))
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.tail(14).mean()
            else:
                atr = current_price * 0.02

            # Adjust stop distance based on regime
            if "HIGH_VOL" in regime.value:
                atr_multiplier = 3.0  # Wider stops in high vol
            else:
                atr_multiplier = 2.0  # Tighter stops in low vol

            stop_distance = atr * atr_multiplier

            if signal_type == SignalType.LONG:
                stop_loss = current_price - stop_distance
                take_profit = current_price + stop_distance * 2
            elif signal_type == SignalType.SHORT:
                stop_loss = current_price + stop_distance
                take_profit = current_price - stop_distance * 2
            else:
                stop_loss = None
                take_profit = None

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error calculating regime stops: {e}")
            return None, None

    def _get_ml_confidence(self, ml_signals: Optional[pd.DataFrame]) -> float:
        """Get ML confidence."""
        if ml_signals is None or len(ml_signals) == 0:
            return 0.65
        try:
            return float(ml_signals['confidence'].iloc[-1])
        except Exception:
            return 0.65

    def _extract_ml_signal(
        self,
        ml_signals: Optional[pd.DataFrame]
    ) -> tuple[Optional[int], float]:
        """Extract ML prediction and confidence."""
        if ml_signals is None or len(ml_signals) == 0:
            return None, 0.0
        try:
            latest = ml_signals.iloc[-1]
            pred = latest.get('prediction', 0)
            conf = latest.get('confidence', 0.0)
            return int(np.sign(pred)) if pred != 0 else 0, float(conf)
        except Exception:
            return None, 0.0

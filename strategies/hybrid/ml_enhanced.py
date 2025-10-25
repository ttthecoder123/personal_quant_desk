"""
ML-Enhanced Strategy

Combines rule-based trading signals with ML predictions:
- Traditional technical indicators provide base signals
- ML models enhance/filter signals with confidence scores
- Dynamic weight adjustment based on performance
- Feature importance tracking for strategy refinement
- A/B testing framework for strategy comparison
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
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


class MLEnhancedStrategy(StrategyBase):
    """
    ML-enhanced discretionary strategy.

    Combines:
    1. Rule-based signals (technical indicators, patterns)
    2. ML model predictions (from Step 4)
    3. Dynamic weight allocation between rules and ML
    4. Confidence-based position sizing
    5. Feature importance feedback loop
    """

    def __init__(
        self,
        name: str = "MLEnhanced",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None,
        rule_generators: Optional[List[Callable]] = None
    ):
        """
        Initialize ML-enhanced strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - ml_weight: Initial weight for ML signals (default: 0.60)
                - rule_weight: Initial weight for rule signals (default: 0.40)
                - weight_adaptation_rate: Learning rate for weights (default: 0.05)
                - min_ml_confidence: Minimum ML confidence to use ML signal (default: 0.60)
                - signal_agreement_threshold: Threshold for signal agreement (default: 0.70)
                - feature_importance_tracking: Track feature importance (default: True)
                - performance_lookback: Days for performance tracking (default: 30)
            risk_metrics: Risk limits
            rule_generators: List of rule-based signal generators
        """
        default_config = {
            'ml_weight': 0.60,
            'rule_weight': 0.40,
            'weight_adaptation_rate': 0.05,
            'min_ml_confidence': 0.60,
            'signal_agreement_threshold': 0.70,
            'feature_importance_tracking': True,
            'performance_lookback': 30,
            'min_observations': 30,
            'min_confidence': 0.55,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        self.rule_generators = rule_generators or []

        # Performance tracking for weight adaptation
        self.ml_performance: List[float] = []
        self.rule_performance: List[float] = []
        self.combined_performance: List[float] = []

        # Feature importance
        self.feature_importances: Dict[str, float] = {}

        # Signal agreement tracking
        self.agreement_history: List[bool] = []

        logger.info(
            f"Initialized {name} with ml_weight={self.config['ml_weight']:.1%}, "
            f"rule_weight={self.config['rule_weight']:.1%}, "
            f"num_rules={len(self.rule_generators)}"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate ML-enhanced signal.

        Process:
        1. Generate rule-based signals
        2. Extract ML prediction and confidence
        3. Combine signals with dynamic weights
        4. Calculate final confidence
        5. Track feature importance

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data
            features: Engineered features
            ml_signals: ML predictions with confidence

        Returns:
            StrategySignal or None
        """
        try:
            if len(market_data) < self.config['min_observations']:
                logger.warning(f"{symbol}: Insufficient data")
                return None

            # Generate rule-based signal
            rule_signal = self._generate_rule_signal(symbol, market_data, features)

            # Extract ML signal
            ml_prediction, ml_confidence = self._extract_ml_signal(ml_signals)

            # Check if ML confidence is sufficient
            use_ml = ml_confidence >= self.config['min_ml_confidence']

            if not use_ml and rule_signal is None:
                logger.debug(f"{symbol}: No valid signals (ML confidence too low, no rule signal)")
                return None

            # Combine signals
            combined_signal_type, combined_confidence = self._combine_signals(
                rule_signal,
                ml_prediction,
                ml_confidence,
                use_ml
            )

            if combined_signal_type == SignalType.HOLD:
                return None

            # Track signal agreement
            if rule_signal is not None and use_ml:
                signals_agree = self._check_signal_agreement(rule_signal, ml_prediction)
                self.agreement_history.append(signals_agree)

            # Track feature importance if available
            if self.config['feature_importance_tracking'] and ml_signals is not None:
                self._update_feature_importance(ml_signals)

            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()
            current_price = float(market_data['close'].iloc[-1])

            # Calculate stops based on volatility
            volatility = self._calculate_volatility(market_data)
            stop_distance = volatility * current_price * np.sqrt(1/252)

            if combined_signal_type == SignalType.LONG:
                stop_loss = current_price - stop_distance * 2
                take_profit = current_price + stop_distance * 3
            elif combined_signal_type == SignalType.SHORT:
                stop_loss = current_price + stop_distance * 2
                take_profit = current_price - stop_distance * 3
            else:
                stop_loss = None
                take_profit = None

            signal = StrategySignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=combined_signal_type,
                confidence=combined_confidence,
                size=0.0,  # Calculated in calculate_position_size
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'strategy': self.name,
                    'rule_signal': rule_signal,
                    'ml_prediction': ml_prediction,
                    'ml_confidence': ml_confidence,
                    'ml_weight': self.config['ml_weight'],
                    'rule_weight': self.config['rule_weight'],
                    'use_ml': use_ml,
                    'signals_agree': self.agreement_history[-1] if self.agreement_history else None,
                    'feature_importances': self.feature_importances.copy() if self.feature_importances else None
                }
            )

            self.signals_history.append(signal)
            logger.info(
                f"{symbol} ML-enhanced signal: {combined_signal_type.value}, "
                f"confidence={combined_confidence:.2%} "
                f"(ML={ml_confidence:.2%}, rule={rule_signal if rule_signal else 'None'})"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating ML-enhanced signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size with ML confidence scaling.

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

            # Volatility scaling
            vol_scalar = 0.20 / max(current_volatility, 0.01)
            position_value = base_position_value * vol_scalar

            # ML confidence scaling
            # Higher confidence = larger position
            confidence_scalar = signal.confidence
            position_value *= confidence_scalar

            # Agreement bonus: if ML and rules agree, increase size
            signals_agree = signal.metadata.get('signals_agree', False)
            if signals_agree:
                agreement_bonus = 1.20  # 20% increase
                position_value *= agreement_bonus
                logger.debug(f"{signal.symbol}: Signals agree, applying {agreement_bonus}x bonus")

            # Convert to shares
            position_size = position_value / signal.entry_price

            # Apply portfolio allocation limit
            max_position_value = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_size = max_position_value / signal.entry_price
            position_size = min(position_size, max_size)

            logger.debug(
                f"{signal.symbol} ML-enhanced size: {position_size:.2f} units "
                f"(confidence={signal.confidence:.2%}, value=${position_value:,.0f})"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating ML-enhanced position size: {e}", exc_info=True)
            return 0.0

    def _generate_rule_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[int]:
        """
        Generate rule-based signal using configured rule generators.

        Args:
            symbol: Instrument symbol
            market_data: Price data
            features: Engineered features

        Returns:
            Signal direction: 1 (long), -1 (short), 0 (hold), or None
        """
        if not self.rule_generators:
            # Default simple rules if none provided
            return self._default_rule_signal(market_data, features)

        # Collect signals from all rule generators
        rule_signals = []
        for rule_generator in self.rule_generators:
            try:
                signal = rule_generator(symbol, market_data, features)
                if signal is not None:
                    rule_signals.append(signal)
            except Exception as e:
                logger.warning(f"Rule generator failed: {e}")

        if not rule_signals:
            return None

        # Aggregate: majority vote
        avg_signal = np.mean(rule_signals)

        if avg_signal > 0.5:
            return 1  # LONG
        elif avg_signal < -0.5:
            return -1  # SHORT
        else:
            return 0  # HOLD

    def _default_rule_signal(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[int]:
        """
        Default rule-based signal using simple technical indicators.

        Args:
            market_data: Price data
            features: Features (may contain indicators)

        Returns:
            Signal direction: 1, -1, 0, or None
        """
        try:
            # Simple moving average crossover
            if features is not None and 'sma_20' in features.columns and 'sma_50' in features.columns:
                sma_20 = features['sma_20'].iloc[-1]
                sma_50 = features['sma_50'].iloc[-1]

                if pd.notna(sma_20) and pd.notna(sma_50):
                    if sma_20 > sma_50 * 1.01:  # 1% above
                        return 1
                    elif sma_20 < sma_50 * 0.99:  # 1% below
                        return -1

            # Fallback: momentum-based
            if len(market_data) >= 20:
                returns = market_data['close'].pct_change(20).iloc[-1]
                if returns > 0.05:  # 5% gain
                    return 1
                elif returns < -0.05:  # 5% loss
                    return -1

            return 0

        except Exception as e:
            logger.debug(f"Default rule signal failed: {e}")
            return None

    def _extract_ml_signal(
        self,
        ml_signals: Optional[pd.DataFrame]
    ) -> tuple[Optional[int], float]:
        """
        Extract ML prediction and confidence.

        Args:
            ml_signals: ML signal DataFrame

        Returns:
            (prediction, confidence)
        """
        if ml_signals is None or len(ml_signals) == 0:
            return None, 0.0

        try:
            latest = ml_signals.iloc[-1]
            prediction = latest.get('prediction', None)
            confidence = latest.get('confidence', 0.0)

            # Ensure prediction is in correct format
            if pd.notna(prediction):
                if prediction > 0:
                    return 1, float(confidence)
                elif prediction < 0:
                    return -1, float(confidence)
                else:
                    return 0, float(confidence)

            return None, 0.0

        except Exception as e:
            logger.warning(f"Error extracting ML signal: {e}")
            return None, 0.0

    def _combine_signals(
        self,
        rule_signal: Optional[int],
        ml_prediction: Optional[int],
        ml_confidence: float,
        use_ml: bool
    ) -> tuple[SignalType, float]:
        """
        Combine rule and ML signals with dynamic weights.

        Args:
            rule_signal: Rule-based signal
            ml_prediction: ML prediction
            ml_confidence: ML confidence
            use_ml: Whether to use ML signal

        Returns:
            (combined_signal_type, combined_confidence)
        """
        ml_weight = self.config['ml_weight']
        rule_weight = self.config['rule_weight']

        # Normalize weights
        total_weight = ml_weight + rule_weight
        ml_weight /= total_weight
        rule_weight /= total_weight

        # Calculate weighted signal
        if use_ml and ml_prediction is not None:
            if rule_signal is not None:
                # Both signals available
                weighted_signal = ml_weight * ml_prediction + rule_weight * rule_signal
                # Confidence is weighted average
                rule_confidence = 0.70  # Assume moderate confidence for rules
                combined_confidence = ml_weight * ml_confidence + rule_weight * rule_confidence
            else:
                # Only ML signal
                weighted_signal = ml_prediction
                combined_confidence = ml_confidence
        elif rule_signal is not None:
            # Only rule signal
            weighted_signal = rule_signal
            combined_confidence = 0.65  # Default rule confidence
        else:
            # No signals
            return SignalType.HOLD, 0.0

        # Convert to SignalType
        if weighted_signal > 0.3:
            signal_type = SignalType.LONG
        elif weighted_signal < -0.3:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.HOLD

        return signal_type, float(combined_confidence)

    def _check_signal_agreement(
        self,
        rule_signal: int,
        ml_prediction: int
    ) -> bool:
        """
        Check if rule and ML signals agree.

        Args:
            rule_signal: Rule signal (-1, 0, 1)
            ml_prediction: ML prediction (-1, 0, 1)

        Returns:
            True if signals agree
        """
        # Signals agree if they have the same sign
        return (rule_signal * ml_prediction) > 0 or (rule_signal == 0 and ml_prediction == 0)

    def _update_feature_importance(self, ml_signals: pd.DataFrame):
        """
        Update feature importance tracking from ML signals.

        Args:
            ml_signals: ML signal DataFrame with feature_importance column
        """
        try:
            if 'feature_importance' not in ml_signals.columns:
                return

            latest_importance = ml_signals['feature_importance'].iloc[-1]

            if pd.isna(latest_importance) or not isinstance(latest_importance, dict):
                return

            # Update running average of feature importances
            for feature, importance in latest_importance.items():
                if feature in self.feature_importances:
                    # Exponential moving average
                    alpha = 0.1
                    self.feature_importances[feature] = (
                        alpha * importance + (1 - alpha) * self.feature_importances[feature]
                    )
                else:
                    self.feature_importances[feature] = importance

            logger.debug(f"Updated feature importances: {len(self.feature_importances)} features")

        except Exception as e:
            logger.debug(f"Error updating feature importance: {e}")

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate recent volatility."""
        try:
            returns = np.log(market_data['close'] / market_data['close'].shift(1))
            vol = returns.tail(30).std() * np.sqrt(252)
            return float(vol) if vol > 0 else 0.20
        except Exception:
            return 0.20

    def update_weights_from_performance(self):
        """
        Adapt ML vs rule weights based on recent performance.

        Called periodically to update strategy weights based on what's working.
        """
        try:
            if len(self.ml_performance) < 10 or len(self.rule_performance) < 10:
                logger.debug("Insufficient performance data for weight adaptation")
                return

            # Calculate recent performance
            lookback = self.config['performance_lookback']
            ml_perf = np.mean(self.ml_performance[-lookback:])
            rule_perf = np.mean(self.rule_performance[-lookback:])

            # Update weights based on relative performance
            learning_rate = self.config['weight_adaptation_rate']

            if ml_perf > rule_perf:
                # ML performing better, increase its weight
                self.config['ml_weight'] += learning_rate
                self.config['rule_weight'] -= learning_rate
            else:
                # Rules performing better, increase their weight
                self.config['ml_weight'] -= learning_rate
                self.config['rule_weight'] += learning_rate

            # Ensure weights stay in valid range [0.2, 0.8]
            self.config['ml_weight'] = np.clip(self.config['ml_weight'], 0.2, 0.8)
            self.config['rule_weight'] = np.clip(self.config['rule_weight'], 0.2, 0.8)

            # Normalize
            total = self.config['ml_weight'] + self.config['rule_weight']
            self.config['ml_weight'] /= total
            self.config['rule_weight'] /= total

            logger.info(
                f"Weight adaptation: ML={self.config['ml_weight']:.2%}, "
                f"Rule={self.config['rule_weight']:.2%} "
                f"(ML perf={ml_perf:.3f}, Rule perf={rule_perf:.3f})"
            )

        except Exception as e:
            logger.error(f"Error updating weights: {e}", exc_info=True)

    def record_performance(self, signal_metadata: Dict[str, Any], pnl: float):
        """
        Record performance for weight adaptation.

        Args:
            signal_metadata: Metadata from the signal that generated this trade
            pnl: Realized P&L
        """
        try:
            use_ml = signal_metadata.get('use_ml', False)
            had_rule = signal_metadata.get('rule_signal') is not None

            if use_ml:
                self.ml_performance.append(pnl)

            if had_rule:
                self.rule_performance.append(pnl)

            self.combined_performance.append(pnl)

            # Trim to reasonable length
            max_history = self.config['performance_lookback'] * 3
            if len(self.ml_performance) > max_history:
                self.ml_performance = self.ml_performance[-max_history:]
            if len(self.rule_performance) > max_history:
                self.rule_performance = self.rule_performance[-max_history:]
            if len(self.combined_performance) > max_history:
                self.combined_performance = self.combined_performance[-max_history:]

        except Exception as e:
            logger.error(f"Error recording performance: {e}")

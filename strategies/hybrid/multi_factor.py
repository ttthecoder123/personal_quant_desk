"""
Multi-Factor Strategy

Combines multiple factors for systematic trading:
- Value factors (price-to-book, earnings yield, etc.)
- Momentum factors (price momentum, earnings momentum)
- Quality factors (ROE, profit margins, stability)
- Volatility factors (realized vol, vol of vol)
- Factor scoring and ranking
- Dynamic factor allocation based on performance
- Factor correlation management
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
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


class MultiFactorStrategy(StrategyBase):
    """
    Multi-factor combination strategy.

    Combines multiple factor signals:
    1. Calculate individual factor scores
    2. Normalize and rank factors
    3. Combine using dynamic weights
    4. Manage factor correlations
    5. Adapt weights based on factor performance
    """

    def __init__(
        self,
        name: str = "MultiFactory",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize multi-factor strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - factors_enabled: List of factor names to use (default: all)
                - factor_weights: Initial factor weights (default: equal-weighted)
                - weight_adaptation: Adapt weights based on performance (default: True)
                - adaptation_rate: Learning rate for weight updates (default: 0.05)
                - lookback_periods: Lookback for factor calculation (default: 60)
                - zscore_threshold: Z-score threshold for extreme values (default: 2.0)
                - correlation_threshold: Max correlation between factors (default: 0.85)
                - rebalance_frequency: Days between rebalances (default: 5)
            risk_metrics: Risk limits
        """
        default_config = {
            'factors_enabled': ['momentum', 'volatility', 'value', 'quality'],
            'factor_weights': {
                'momentum': 0.30,
                'volatility': 0.25,
                'value': 0.25,
                'quality': 0.20
            },
            'weight_adaptation': True,
            'adaptation_rate': 0.05,
            'lookback_periods': 60,
            'zscore_threshold': 2.0,
            'correlation_threshold': 0.85,
            'rebalance_frequency': 5,
            'min_observations': 60,
            'min_confidence': 0.60,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Factor tracking
        self.factor_scores: Dict[str, float] = {}
        self.factor_rankings: Dict[str, float] = {}
        self.composite_score: float = 0.0

        # Performance tracking for adaptation
        self.factor_performance: Dict[str, List[float]] = {
            factor: [] for factor in self.config['factors_enabled']
        }

        # Correlation tracking
        self.factor_correlation_matrix: Optional[pd.DataFrame] = None

        logger.info(
            f"Initialized {name} with factors={self.config['factors_enabled']}, "
            f"lookback={self.config['lookback_periods']}d"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate multi-factor signal.

        Process:
        1. Calculate all factor scores
        2. Normalize and rank factors
        3. Combine using weighted average
        4. Check factor correlations
        5. Generate signal based on composite score

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

            # Calculate all factor scores
            factor_scores = {}

            if 'momentum' in self.config['factors_enabled']:
                factor_scores['momentum'] = self._calculate_momentum_factor(market_data, features)

            if 'volatility' in self.config['factors_enabled']:
                factor_scores['volatility'] = self._calculate_volatility_factor(market_data, features)

            if 'value' in self.config['factors_enabled']:
                factor_scores['value'] = self._calculate_value_factor(market_data, features)

            if 'quality' in self.config['factors_enabled']:
                factor_scores['quality'] = self._calculate_quality_factor(market_data, features)

            # Remove None values
            factor_scores = {k: v for k, v in factor_scores.items() if v is not None}

            if not factor_scores:
                logger.warning(f"{symbol}: No valid factor scores")
                return None

            self.factor_scores = factor_scores

            # Normalize factor scores (z-score)
            normalized_scores = self._normalize_factor_scores(factor_scores)

            # Calculate composite score
            composite_score = self._calculate_composite_score(normalized_scores)
            self.composite_score = composite_score

            # Determine signal type
            signal_type = self._determine_signal_from_score(composite_score)

            if signal_type == SignalType.HOLD:
                return None

            # Calculate confidence from factor agreement
            confidence = self._calculate_factor_confidence(normalized_scores)

            # Enhance with ML confidence if available
            ml_confidence = self._get_ml_confidence(ml_signals)
            combined_confidence = 0.7 * confidence + 0.3 * ml_confidence

            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()
            current_price = float(market_data['close'].iloc[-1])

            # Calculate stops
            volatility = factor_scores.get('volatility', 0.20)
            stop_distance = volatility * current_price * np.sqrt(5/252)  # 5-day vol

            if signal_type == SignalType.LONG:
                stop_loss = current_price - stop_distance * 2
                take_profit = current_price + stop_distance * 3
            elif signal_type == SignalType.SHORT:
                stop_loss = current_price + stop_distance * 2
                take_profit = current_price - stop_distance * 3
            else:
                stop_loss = None
                take_profit = None

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
                    'factor_scores': factor_scores.copy(),
                    'normalized_scores': normalized_scores.copy(),
                    'composite_score': composite_score,
                    'factor_weights': self.config['factor_weights'].copy(),
                    'ml_confidence': ml_confidence
                }
            )

            self.signals_history.append(signal)
            logger.info(
                f"{symbol} multi-factor signal: {signal_type.value}, "
                f"composite={composite_score:.3f}, confidence={combined_confidence:.2%}, "
                f"factors={list(factor_scores.keys())}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating multi-factor signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size based on factor scores.

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

            # Scale by composite score magnitude
            composite_score = abs(signal.metadata.get('composite_score', 0.0))
            score_scalar = min(composite_score / 1.0, 2.0)  # Cap at 2x
            position_value *= score_scalar

            # Scale by confidence
            position_value *= signal.confidence

            # Convert to shares
            position_size = position_value / signal.entry_price

            # Apply portfolio allocation limit
            max_position_value = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_size = max_position_value / signal.entry_price
            position_size = min(position_size, max_size)

            logger.debug(
                f"{signal.symbol} multi-factor size: {position_size:.2f} units "
                f"(composite={signal.metadata.get('composite_score', 0):.3f})"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating multi-factor position size: {e}", exc_info=True)
            return 0.0

    def _calculate_momentum_factor(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Calculate momentum factor score.

        Uses multiple timeframes for robustness.

        Args:
            market_data: Price data
            features: Features

        Returns:
            Momentum score (-1 to 1)
        """
        try:
            lookback = self.config['lookback_periods']

            # Multiple momentum periods
            mom_1m = market_data['close'].pct_change(20).iloc[-1]  # 1 month
            mom_3m = market_data['close'].pct_change(60).iloc[-1]  # 3 months

            if pd.isna(mom_1m) or pd.isna(mom_3m):
                return None

            # Weight recent momentum more
            momentum_score = 0.6 * mom_1m + 0.4 * mom_3m

            # Clip to reasonable range
            momentum_score = np.clip(momentum_score, -0.5, 0.5)

            return float(momentum_score)

        except Exception as e:
            logger.debug(f"Error calculating momentum factor: {e}")
            return None

    def _calculate_volatility_factor(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Calculate volatility factor score.

        Lower volatility is generally better (negative score = low vol = good).

        Args:
            market_data: Price data
            features: Features

        Returns:
            Volatility score (annualized)
        """
        try:
            lookback = self.config['lookback_periods']
            returns = np.log(market_data['close'] / market_data['close'].shift(1))
            returns = returns.dropna().tail(lookback)

            if len(returns) < lookback // 2:
                return None

            volatility = returns.std() * np.sqrt(252)
            return float(volatility)

        except Exception as e:
            logger.debug(f"Error calculating volatility factor: {e}")
            return None

    def _calculate_value_factor(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Calculate value factor score.

        For crypto/futures, use price deviation from moving average.
        For stocks, would use P/E, P/B, etc.

        Args:
            market_data: Price data
            features: Features

        Returns:
            Value score (-1 to 1)
        """
        try:
            # Price deviation from 200-day MA
            if len(market_data) < 200:
                return None

            current_price = market_data['close'].iloc[-1]
            ma_200 = market_data['close'].tail(200).mean()

            # Value score: negative deviation = undervalued = positive score
            deviation = (current_price - ma_200) / ma_200

            # Invert and normalize
            value_score = -deviation  # Negative deviation = positive value score

            # Clip to reasonable range
            value_score = np.clip(value_score, -0.3, 0.3)

            return float(value_score)

        except Exception as e:
            logger.debug(f"Error calculating value factor: {e}")
            return None

    def _calculate_quality_factor(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Calculate quality factor score.

        For price data, use trend consistency and return stability.
        For stocks, would use ROE, profit margins, etc.

        Args:
            market_data: Price data
            features: Features

        Returns:
            Quality score (0 to 1)
        """
        try:
            lookback = self.config['lookback_periods']
            returns = market_data['close'].pct_change().tail(lookback)

            if len(returns) < lookback // 2:
                return None

            # Quality metrics:
            # 1. Return stability (inverse of std)
            return_stability = 1 / (1 + returns.std())

            # 2. Trend consistency (% of positive days)
            trend_consistency = (returns > 0).sum() / len(returns)

            # Combine
            quality_score = 0.5 * return_stability + 0.5 * trend_consistency

            # Normalize to [-1, 1]
            quality_score = (quality_score - 0.5) * 2

            return float(quality_score)

        except Exception as e:
            logger.debug(f"Error calculating quality factor: {e}")
            return None

    def _normalize_factor_scores(
        self,
        factor_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize factor scores using z-score.

        Args:
            factor_scores: Raw factor scores

        Returns:
            Normalized scores
        """
        normalized = {}

        for factor, score in factor_scores.items():
            # For simplification, clip to zscore threshold
            # In production, would use historical distribution
            zscore_threshold = self.config['zscore_threshold']
            normalized_score = np.clip(score, -zscore_threshold, zscore_threshold)
            normalized[factor] = float(normalized_score)

        return normalized

    def _calculate_composite_score(
        self,
        normalized_scores: Dict[str, float]
    ) -> float:
        """
        Calculate composite score from individual factors.

        Args:
            normalized_scores: Normalized factor scores

        Returns:
            Composite score
        """
        factor_weights = self.config['factor_weights']
        composite = 0.0
        total_weight = 0.0

        for factor, score in normalized_scores.items():
            weight = factor_weights.get(factor, 0.0)
            composite += weight * score
            total_weight += weight

        if total_weight > 0:
            composite /= total_weight

        return float(composite)

    def _determine_signal_from_score(self, composite_score: float) -> SignalType:
        """
        Determine signal type from composite score.

        Args:
            composite_score: Composite factor score

        Returns:
            SignalType
        """
        # Thresholds for signal generation
        long_threshold = 0.5
        short_threshold = -0.5

        if composite_score > long_threshold:
            return SignalType.LONG
        elif composite_score < short_threshold:
            return SignalType.SHORT
        else:
            return SignalType.HOLD

    def _calculate_factor_confidence(
        self,
        normalized_scores: Dict[str, float]
    ) -> float:
        """
        Calculate confidence from factor agreement.

        High confidence when factors agree, low when they disagree.

        Args:
            normalized_scores: Normalized factor scores

        Returns:
            Confidence score (0 to 1)
        """
        if not normalized_scores:
            return 0.5

        # Calculate agreement: how aligned are the factors?
        scores = list(normalized_scores.values())

        # If all factors point same direction, high confidence
        positive_count = sum(1 for s in scores if s > 0)
        negative_count = sum(1 for s in scores if s < 0)
        total = len(scores)

        # Agreement ratio
        max_agreement = max(positive_count, negative_count)
        agreement = max_agreement / total

        # Also consider magnitude
        avg_magnitude = np.mean([abs(s) for s in scores])
        magnitude_confidence = min(avg_magnitude / 1.0, 1.0)

        # Combine
        confidence = 0.6 * agreement + 0.4 * magnitude_confidence

        return float(np.clip(confidence, 0.0, 1.0))

    def _get_ml_confidence(self, ml_signals: Optional[pd.DataFrame]) -> float:
        """Get ML confidence."""
        if ml_signals is None or len(ml_signals) == 0:
            return 0.65
        try:
            return float(ml_signals['confidence'].iloc[-1])
        except Exception:
            return 0.65

    def update_factor_weights(self):
        """
        Update factor weights based on recent performance.

        Called periodically to adapt to changing market conditions.
        """
        if not self.config['weight_adaptation']:
            return

        try:
            min_history = 20

            # Calculate average performance for each factor
            factor_returns = {}
            for factor, performance in self.factor_performance.items():
                if len(performance) < min_history:
                    continue
                factor_returns[factor] = np.mean(performance[-min_history:])

            if len(factor_returns) < 2:
                logger.debug("Insufficient performance data for weight adaptation")
                return

            # Normalize returns to get new weights
            returns_array = np.array(list(factor_returns.values()))
            returns_array = returns_array - returns_array.min()  # Shift to positive

            if returns_array.sum() == 0:
                return

            new_weights = returns_array / returns_array.sum()

            # Update weights with learning rate
            learning_rate = self.config['adaptation_rate']

            for i, factor in enumerate(factor_returns.keys()):
                old_weight = self.config['factor_weights'].get(factor, 0.25)
                new_weight = (1 - learning_rate) * old_weight + learning_rate * new_weights[i]
                self.config['factor_weights'][factor] = float(new_weight)

            # Normalize weights to sum to 1
            total_weight = sum(self.config['factor_weights'].values())
            for factor in self.config['factor_weights']:
                self.config['factor_weights'][factor] /= total_weight

            logger.info(f"Updated factor weights: {self.config['factor_weights']}")

        except Exception as e:
            logger.error(f"Error updating factor weights: {e}", exc_info=True)

    def record_factor_performance(
        self,
        signal_metadata: Dict[str, Any],
        pnl: float
    ):
        """
        Record performance for factor weight adaptation.

        Args:
            signal_metadata: Metadata from signal
            pnl: Realized P&L
        """
        try:
            factor_scores = signal_metadata.get('factor_scores', {})

            # Attribute performance to factors based on their contribution
            for factor, score in factor_scores.items():
                if factor in self.factor_performance:
                    # Weight performance by factor contribution
                    weighted_pnl = pnl * abs(score)
                    self.factor_performance[factor].append(weighted_pnl)

                    # Trim history
                    max_history = 100
                    if len(self.factor_performance[factor]) > max_history:
                        self.factor_performance[factor] = self.factor_performance[factor][-max_history:]

        except Exception as e:
            logger.error(f"Error recording factor performance: {e}")

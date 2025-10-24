"""
Ensemble Strategy

Meta-strategy that combines multiple sub-strategies:
- Dynamic weight allocation based on performance
- Correlation-based diversification
- Online learning for weight updates
- Strategy attribution tracking
- Ensemble signal aggregation
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


class EnsembleStrategy(StrategyBase):
    """
    Ensemble of multiple strategies with dynamic weight allocation.

    Core functionality:
    1. Maintain multiple sub-strategies
    2. Collect signals from each sub-strategy
    3. Combine signals using dynamic weights
    4. Update weights based on performance
    5. Manage correlation between strategies
    6. Track attribution for each strategy
    """

    def __init__(
        self,
        name: str = "Ensemble",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None,
        sub_strategies: Optional[List[StrategyBase]] = None
    ):
        """
        Initialize ensemble strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - combination_method: 'weighted_average', 'voting', 'confidence_weighted' (default: 'confidence_weighted')
                - weight_update_method: 'performance', 'sharpe', 'online_learning' (default: 'sharpe')
                - initial_weights: Initial strategy weights (default: equal-weighted)
                - learning_rate: Learning rate for online updates (default: 0.05)
                - performance_lookback: Days for performance calculation (default: 30)
                - min_agreement_threshold: Min % of strategies that must agree (default: 0.5)
                - correlation_penalty: Penalize correlated strategies (default: True)
                - rebalance_frequency: Days between weight rebalances (default: 7)
            risk_metrics: Risk limits
            sub_strategies: List of sub-strategies to ensemble
        """
        default_config = {
            'combination_method': 'confidence_weighted',
            'weight_update_method': 'sharpe',
            'initial_weights': {},
            'learning_rate': 0.05,
            'performance_lookback': 30,
            'min_agreement_threshold': 0.5,
            'correlation_penalty': True,
            'rebalance_frequency': 7,
            'min_observations': 30,
            'min_confidence': 0.60,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Sub-strategies
        self.sub_strategies = sub_strategies or []

        # Initialize weights
        self.strategy_weights: Dict[str, float] = {}
        self._initialize_weights()

        # Performance tracking
        self.strategy_performance: Dict[str, List[float]] = {
            s.name: [] for s in self.sub_strategies
        }
        self.strategy_sharpe: Dict[str, float] = {}

        # Signal tracking
        self.recent_signals: Dict[str, List[StrategySignal]] = {
            s.name: [] for s in self.sub_strategies
        }

        # Correlation tracking
        self.strategy_correlation_matrix: Optional[pd.DataFrame] = None

        # Attribution
        self.strategy_attribution: Dict[str, float] = {
            s.name: 0.0 for s in self.sub_strategies
        }

        logger.info(
            f"Initialized {name} with {len(self.sub_strategies)} sub-strategies, "
            f"method={self.config['combination_method']}"
        )

    def add_strategy(self, strategy: StrategyBase):
        """
        Add a sub-strategy to the ensemble.

        Args:
            strategy: Strategy to add
        """
        if strategy not in self.sub_strategies:
            self.sub_strategies.append(strategy)
            self.strategy_weights[strategy.name] = 1.0 / len(self.sub_strategies)
            self.strategy_performance[strategy.name] = []
            self.recent_signals[strategy.name] = []
            self.strategy_attribution[strategy.name] = 0.0
            logger.info(f"Added strategy {strategy.name} to ensemble")

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate ensemble signal by combining sub-strategy signals.

        Process:
        1. Collect signals from all sub-strategies
        2. Filter out low-confidence signals
        3. Combine signals using configured method
        4. Check minimum agreement threshold
        5. Generate ensemble signal

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

            if not self.sub_strategies:
                logger.warning("No sub-strategies in ensemble")
                return None

            # Collect signals from all sub-strategies
            sub_signals: Dict[str, Optional[StrategySignal]] = {}

            for strategy in self.sub_strategies:
                try:
                    signal = strategy.generate_signal(symbol, market_data, features, ml_signals)
                    sub_signals[strategy.name] = signal

                    # Track signal
                    if signal is not None:
                        self.recent_signals[strategy.name].append(signal)
                        # Trim history
                        if len(self.recent_signals[strategy.name]) > 100:
                            self.recent_signals[strategy.name] = self.recent_signals[strategy.name][-100:]

                except Exception as e:
                    logger.warning(f"Strategy {strategy.name} failed to generate signal: {e}")
                    sub_signals[strategy.name] = None

            # Filter valid signals
            valid_signals = {k: v for k, v in sub_signals.items() if v is not None}

            if not valid_signals:
                logger.debug(f"{symbol}: No valid signals from sub-strategies")
                return None

            # Check minimum agreement
            if not self._check_agreement(valid_signals):
                logger.debug(f"{symbol}: Insufficient strategy agreement")
                return None

            # Combine signals
            combined_signal_type, combined_confidence = self._combine_signals(valid_signals)

            if combined_signal_type == SignalType.HOLD:
                return None

            # Calculate ensemble entry price (weighted average)
            entry_price = self._calculate_weighted_price(valid_signals, 'entry_price')

            # Calculate ensemble stops (conservative)
            stop_loss, take_profit = self._calculate_ensemble_stops(valid_signals, combined_signal_type)

            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()

            signal = StrategySignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=combined_signal_type,
                confidence=combined_confidence,
                size=0.0,  # Calculated in calculate_position_size
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'strategy': self.name,
                    'num_sub_signals': len(valid_signals),
                    'sub_signals': {k: v.signal_type.value for k, v in valid_signals.items()},
                    'sub_confidences': {k: v.confidence for k, v in valid_signals.items()},
                    'strategy_weights': self.strategy_weights.copy(),
                    'combination_method': self.config['combination_method']
                }
            )

            self.signals_history.append(signal)
            logger.info(
                f"{symbol} ensemble signal: {combined_signal_type.value}, "
                f"confidence={combined_confidence:.2%}, "
                f"sub_signals={len(valid_signals)}/{len(self.sub_strategies)}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating ensemble signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size for ensemble signal.

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

            # Scale by ensemble confidence
            position_value *= signal.confidence

            # Scale by number of agreeing strategies
            num_sub_signals = signal.metadata.get('num_sub_signals', 1)
            agreement_bonus = min(1.0 + 0.1 * (num_sub_signals - 1), 1.5)  # Up to 50% bonus
            position_value *= agreement_bonus

            # Convert to shares
            position_size = position_value / signal.entry_price

            # Apply portfolio allocation limit
            max_position_value = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_size = max_position_value / signal.entry_price
            position_size = min(position_size, max_size)

            logger.debug(
                f"{signal.symbol} ensemble size: {position_size:.2f} units "
                f"(sub_signals={num_sub_signals}, bonus={agreement_bonus:.2f}x)"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating ensemble position size: {e}", exc_info=True)
            return 0.0

    def _initialize_weights(self):
        """Initialize strategy weights."""
        if self.config['initial_weights']:
            self.strategy_weights = self.config['initial_weights'].copy()
        else:
            # Equal-weighted
            if self.sub_strategies:
                weight = 1.0 / len(self.sub_strategies)
                self.strategy_weights = {s.name: weight for s in self.sub_strategies}

    def _check_agreement(self, valid_signals: Dict[str, StrategySignal]) -> bool:
        """
        Check if minimum agreement threshold is met.

        Args:
            valid_signals: Valid sub-strategy signals

        Returns:
            True if agreement threshold met
        """
        if not valid_signals:
            return False

        # Count signal directions
        long_count = sum(1 for sig in valid_signals.values() if sig.signal_type == SignalType.LONG)
        short_count = sum(1 for sig in valid_signals.values() if sig.signal_type == SignalType.SHORT)
        total = len(valid_signals)

        # Agreement is max direction count / total
        max_agreement = max(long_count, short_count) / total

        threshold = self.config['min_agreement_threshold']

        if max_agreement >= threshold:
            return True

        logger.debug(
            f"Agreement {max_agreement:.2%} below threshold {threshold:.2%} "
            f"(LONG={long_count}, SHORT={short_count})"
        )
        return False

    def _combine_signals(
        self,
        valid_signals: Dict[str, StrategySignal]
    ) -> Tuple[SignalType, float]:
        """
        Combine sub-strategy signals.

        Args:
            valid_signals: Valid sub-strategy signals

        Returns:
            (combined_signal_type, combined_confidence)
        """
        method = self.config['combination_method']

        if method == 'weighted_average':
            return self._combine_weighted_average(valid_signals)
        elif method == 'voting':
            return self._combine_voting(valid_signals)
        elif method == 'confidence_weighted':
            return self._combine_confidence_weighted(valid_signals)
        else:
            logger.warning(f"Unknown combination method {method}, using confidence_weighted")
            return self._combine_confidence_weighted(valid_signals)

    def _combine_weighted_average(
        self,
        valid_signals: Dict[str, StrategySignal]
    ) -> Tuple[SignalType, float]:
        """
        Combine using weighted average of signal directions.

        Args:
            valid_signals: Valid signals

        Returns:
            (signal_type, confidence)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        avg_confidence = 0.0

        for strategy_name, signal in valid_signals.items():
            weight = self.strategy_weights.get(strategy_name, 0.0)

            # Convert signal to numeric
            if signal.signal_type == SignalType.LONG:
                signal_value = 1.0
            elif signal.signal_type == SignalType.SHORT:
                signal_value = -1.0
            else:
                signal_value = 0.0

            weighted_sum += weight * signal_value
            total_weight += weight
            avg_confidence += weight * signal.confidence

        if total_weight > 0:
            weighted_sum /= total_weight
            avg_confidence /= total_weight

        # Convert to signal type
        if weighted_sum > 0.3:
            return SignalType.LONG, avg_confidence
        elif weighted_sum < -0.3:
            return SignalType.SHORT, avg_confidence
        else:
            return SignalType.HOLD, 0.0

    def _combine_voting(
        self,
        valid_signals: Dict[str, StrategySignal]
    ) -> Tuple[SignalType, float]:
        """
        Combine using majority voting.

        Args:
            valid_signals: Valid signals

        Returns:
            (signal_type, confidence)
        """
        long_count = sum(1 for sig in valid_signals.values() if sig.signal_type == SignalType.LONG)
        short_count = sum(1 for sig in valid_signals.values() if sig.signal_type == SignalType.SHORT)

        total = len(valid_signals)
        avg_confidence = np.mean([sig.confidence for sig in valid_signals.values()])

        if long_count > short_count:
            confidence = (long_count / total) * avg_confidence
            return SignalType.LONG, confidence
        elif short_count > long_count:
            confidence = (short_count / total) * avg_confidence
            return SignalType.SHORT, confidence
        else:
            return SignalType.HOLD, 0.0

    def _combine_confidence_weighted(
        self,
        valid_signals: Dict[str, StrategySignal]
    ) -> Tuple[SignalType, float]:
        """
        Combine using both strategy weights and signal confidence.

        Args:
            valid_signals: Valid signals

        Returns:
            (signal_type, confidence)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0

        for strategy_name, signal in valid_signals.items():
            strategy_weight = self.strategy_weights.get(strategy_name, 0.0)
            signal_confidence = signal.confidence

            # Combined weight
            combined_weight = strategy_weight * signal_confidence

            # Convert signal to numeric
            if signal.signal_type == SignalType.LONG:
                signal_value = 1.0
            elif signal.signal_type == SignalType.SHORT:
                signal_value = -1.0
            else:
                signal_value = 0.0

            weighted_sum += combined_weight * signal_value
            total_weight += combined_weight
            weighted_confidence += combined_weight * signal_confidence

        if total_weight > 0:
            weighted_sum /= total_weight
            weighted_confidence /= total_weight

        # Convert to signal type
        if weighted_sum > 0.3:
            return SignalType.LONG, weighted_confidence
        elif weighted_sum < -0.3:
            return SignalType.SHORT, weighted_confidence
        else:
            return SignalType.HOLD, 0.0

    def _calculate_weighted_price(
        self,
        valid_signals: Dict[str, StrategySignal],
        price_field: str
    ) -> float:
        """
        Calculate weighted average price from signals.

        Args:
            valid_signals: Valid signals
            price_field: Price field to average ('entry_price', 'stop_loss', etc.)

        Returns:
            Weighted average price
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for strategy_name, signal in valid_signals.items():
            weight = self.strategy_weights.get(strategy_name, 0.0)
            price = getattr(signal, price_field, 0.0)

            if price is not None and price > 0:
                weighted_sum += weight * price
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to simple average
            prices = [getattr(sig, price_field, 0.0) for sig in valid_signals.values()]
            prices = [p for p in prices if p is not None and p > 0]
            return np.mean(prices) if prices else 0.0

    def _calculate_ensemble_stops(
        self,
        valid_signals: Dict[str, StrategySignal],
        signal_type: SignalType
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate conservative ensemble stops.

        Uses tightest stop loss and average take profit.

        Args:
            valid_signals: Valid signals
            signal_type: Ensemble signal type

        Returns:
            (stop_loss, take_profit)
        """
        stop_losses = [sig.stop_loss for sig in valid_signals.values() if sig.stop_loss is not None]
        take_profits = [sig.take_profit for sig in valid_signals.values() if sig.take_profit is not None]

        if signal_type == SignalType.LONG:
            # Use tightest (highest) stop loss for longs
            stop_loss = max(stop_losses) if stop_losses else None
            # Average take profit
            take_profit = np.mean(take_profits) if take_profits else None

        elif signal_type == SignalType.SHORT:
            # Use tightest (lowest) stop loss for shorts
            stop_loss = min(stop_losses) if stop_losses else None
            # Average take profit
            take_profit = np.mean(take_profits) if take_profits else None

        else:
            stop_loss = None
            take_profit = None

        return stop_loss, take_profit

    def update_strategy_weights(self):
        """
        Update strategy weights based on performance.

        Called periodically to adapt ensemble to changing conditions.
        """
        method = self.config['weight_update_method']

        if method == 'performance':
            self._update_weights_by_performance()
        elif method == 'sharpe':
            self._update_weights_by_sharpe()
        elif method == 'online_learning':
            self._update_weights_online_learning()
        else:
            logger.warning(f"Unknown weight update method: {method}")

    def _update_weights_by_performance(self):
        """Update weights based on recent returns."""
        try:
            lookback = self.config['performance_lookback']
            learning_rate = self.config['learning_rate']

            strategy_returns = {}
            for strategy_name, performance in self.strategy_performance.items():
                if len(performance) >= 10:
                    recent_perf = performance[-lookback:]
                    avg_return = np.mean(recent_perf)
                    strategy_returns[strategy_name] = avg_return

            if len(strategy_returns) < 2:
                return

            # Convert to weights (softmax of returns)
            returns_array = np.array(list(strategy_returns.values()))
            # Shift to positive
            returns_array = returns_array - returns_array.min() + 0.01

            # Softmax
            exp_returns = np.exp(returns_array / returns_array.std())
            new_weights = exp_returns / exp_returns.sum()

            # Update with learning rate
            for i, strategy_name in enumerate(strategy_returns.keys()):
                old_weight = self.strategy_weights.get(strategy_name, 0.0)
                self.strategy_weights[strategy_name] = (
                    (1 - learning_rate) * old_weight + learning_rate * new_weights[i]
                )

            # Normalize
            total_weight = sum(self.strategy_weights.values())
            for name in self.strategy_weights:
                self.strategy_weights[name] /= total_weight

            logger.info(f"Updated weights by performance: {self.strategy_weights}")

        except Exception as e:
            logger.error(f"Error updating weights by performance: {e}")

    def _update_weights_by_sharpe(self):
        """Update weights based on Sharpe ratio."""
        try:
            lookback = self.config['performance_lookback']

            sharpe_ratios = {}
            for strategy_name, performance in self.strategy_performance.items():
                if len(performance) >= lookback:
                    recent_perf = performance[-lookback:]
                    avg_return = np.mean(recent_perf)
                    std_return = np.std(recent_perf)

                    if std_return > 0:
                        sharpe = avg_return / std_return
                        sharpe_ratios[strategy_name] = sharpe
                        self.strategy_sharpe[strategy_name] = sharpe

            if len(sharpe_ratios) < 2:
                return

            # Convert Sharpe to weights
            sharpe_array = np.array(list(sharpe_ratios.values()))
            # Shift to positive
            sharpe_array = sharpe_array - sharpe_array.min() + 0.1

            # Normalize to weights
            new_weights = sharpe_array / sharpe_array.sum()

            # Update with learning rate
            learning_rate = self.config['learning_rate']
            for i, strategy_name in enumerate(sharpe_ratios.keys()):
                old_weight = self.strategy_weights.get(strategy_name, 0.0)
                self.strategy_weights[strategy_name] = (
                    (1 - learning_rate) * old_weight + learning_rate * new_weights[i]
                )

            # Normalize
            total_weight = sum(self.strategy_weights.values())
            for name in self.strategy_weights:
                self.strategy_weights[name] /= total_weight

            logger.info(f"Updated weights by Sharpe: {self.strategy_weights}")

        except Exception as e:
            logger.error(f"Error updating weights by Sharpe: {e}")

    def _update_weights_online_learning(self):
        """Update weights using online gradient descent."""
        # Simplified online learning - would use regret minimization in production
        self._update_weights_by_sharpe()

    def record_strategy_performance(
        self,
        signal_metadata: Dict[str, Any],
        pnl: float
    ):
        """
        Record performance and attribute to sub-strategies.

        Args:
            signal_metadata: Metadata from ensemble signal
            pnl: Realized P&L
        """
        try:
            sub_signals = signal_metadata.get('sub_signals', {})

            # Attribute performance to strategies that contributed
            for strategy_name in sub_signals.keys():
                if strategy_name in self.strategy_performance:
                    weight = self.strategy_weights.get(strategy_name, 0.0)
                    attributed_pnl = pnl * weight

                    self.strategy_performance[strategy_name].append(pnl)
                    self.strategy_attribution[strategy_name] += attributed_pnl

                    # Trim history
                    max_history = self.config['performance_lookback'] * 3
                    if len(self.strategy_performance[strategy_name]) > max_history:
                        self.strategy_performance[strategy_name] = \
                            self.strategy_performance[strategy_name][-max_history:]

        except Exception as e:
            logger.error(f"Error recording strategy performance: {e}")

    def get_strategy_attribution(self) -> Dict[str, float]:
        """
        Get P&L attribution by strategy.

        Returns:
            Dictionary mapping strategy names to attributed P&L
        """
        return self.strategy_attribution.copy()

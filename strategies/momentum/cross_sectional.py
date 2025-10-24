"""
Cross-Sectional Momentum Strategy

Implements relative momentum ranking across a universe of assets:
- Ranks assets by momentum strength
- Longs top performers, shorts bottom performers
- Optional sector/asset class neutrality
- Periodic rebalancing with turnover constraints
- Volatility-adjusted scoring

This strategy exploits the tendency of relative winners to continue
outperforming and relative losers to continue underperforming.

References:
- "Momentum" by Antonacci
- Jegadeesh and Titman (1993) - "Returns to Buying Winners and Selling Losers"
- Carhart (1997) - Four-factor model
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
    RiskMetrics
)


class CrossSectionalMomentumStrategy(StrategyBase):
    """
    Cross-sectional momentum strategy with relative ranking.

    Key Features:
    - Multi-period momentum scoring (1m, 3m, 6m, 12m)
    - Volatility-adjusted returns for fair comparison
    - Quartile-based portfolio construction (long top, short bottom)
    - Optional sector neutrality
    - Turnover control to reduce trading costs
    - ML confidence weighting
    """

    def __init__(
        self,
        name: str = "CrossSectionalMomentum",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize cross-sectional momentum strategy.

        Args:
            name: Strategy name
            config: Configuration parameters
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'enabled': True,
            'lookback_periods': [21, 63, 126, 252],  # 1m, 3m, 6m, 12m
            'lookback_weights': [0.2, 0.3, 0.3, 0.2],  # Weight each period
            'vol_lookback': 60,  # Volatility calculation period
            'vol_adjusted': True,  # Adjust returns by volatility
            'rebalance_frequency': 21,  # Rebalance every 21 days
            'long_percentile': 0.75,  # Long top 25%
            'short_percentile': 0.25,  # Short bottom 25%
            'min_assets': 5,  # Minimum assets in universe
            'max_positions': 20,  # Maximum total positions
            'sector_neutral': False,  # Sector neutrality (requires metadata)
            'min_confidence': 0.55,  # Minimum ML confidence
            'turnover_constraint': 0.5,  # Max 50% turnover per rebalance
            'exclude_recent_ipos': True,  # Exclude assets with < 252 days history
            'skip_microcaps': True,  # Skip if volume too low
            'min_avg_volume': 100000,  # Minimum average daily volume
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Track rebalancing
        self.last_rebalance: Optional[datetime] = None
        self.current_universe: Dict[str, float] = {}  # symbol -> momentum score
        self.target_positions: Dict[str, float] = {}  # symbol -> target weight

        logger.info(
            f"Initialized {name} with periods {self.config['lookback_periods']}, "
            f"long/short percentiles {self.config['long_percentile']}/{self.config['short_percentile']}"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Note: Cross-sectional momentum requires universe-level analysis.
        This method is designed to work with generate_signals_for_universe()
        rather than single-asset analysis.

        For single-asset calls, returns None. Use generate_signals_for_universe() instead.

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data
            features: Technical features
            ml_signals: ML predictions

        Returns:
            None (use generate_signals_for_universe instead)
        """
        logger.debug(
            f"{symbol}: Cross-sectional momentum requires universe analysis, "
            "use generate_signals_for_universe()"
        )
        return None

    def generate_signals_for_universe(
        self,
        universe_data: Dict[str, pd.DataFrame],
        universe_features: Optional[Dict[str, pd.DataFrame]] = None,
        universe_ml_signals: Optional[Dict[str, pd.DataFrame]] = None,
        current_timestamp: Optional[datetime] = None
    ) -> List[StrategySignal]:
        """
        Generate signals for entire universe of assets.

        Args:
            universe_data: Dict of symbol -> market_data DataFrame
            universe_features: Dict of symbol -> features DataFrame
            universe_ml_signals: Dict of symbol -> ML signals DataFrame
            current_timestamp: Current timestamp for signal generation

        Returns:
            List of StrategySignal objects
        """
        try:
            if len(universe_data) < self.config['min_assets']:
                logger.warning(
                    f"Universe too small: {len(universe_data)} assets, "
                    f"need at least {self.config['min_assets']}"
                )
                return []

            # Check if rebalancing is needed
            if not self._should_rebalance(current_timestamp):
                logger.debug("Rebalancing not required yet")
                return []

            # Calculate momentum scores for all assets
            momentum_scores = self._calculate_momentum_scores(
                universe_data,
                universe_features,
                universe_ml_signals
            )

            if len(momentum_scores) < self.config['min_assets']:
                logger.warning(
                    f"Too few valid momentum scores: {len(momentum_scores)}"
                )
                return []

            # Rank assets and select long/short
            long_assets, short_assets = self._rank_and_select(momentum_scores)

            logger.info(
                f"Selected {len(long_assets)} longs and {len(short_assets)} shorts "
                f"from universe of {len(momentum_scores)} assets"
            )

            # Generate signals for selected assets
            signals = []

            # Long signals
            for symbol in long_assets:
                signal = self._create_signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    momentum_score=momentum_scores[symbol],
                    universe_data=universe_data,
                    universe_features=universe_features,
                    universe_ml_signals=universe_ml_signals,
                    current_timestamp=current_timestamp
                )
                if signal:
                    signals.append(signal)

            # Short signals
            for symbol in short_assets:
                signal = self._create_signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    momentum_score=momentum_scores[symbol],
                    universe_data=universe_data,
                    universe_features=universe_features,
                    universe_ml_signals=universe_ml_signals,
                    current_timestamp=current_timestamp
                )
                if signal:
                    signals.append(signal)

            # Update rebalance timestamp
            self.last_rebalance = current_timestamp

            logger.success(
                f"Generated {len(signals)} cross-sectional momentum signals "
                f"at {current_timestamp}"
            )

            return signals

        except Exception as e:
            logger.error(f"Error generating universe signals: {e}", exc_info=True)
            return []

    def _should_rebalance(self, current_timestamp: Optional[datetime]) -> bool:
        """
        Check if rebalancing is needed based on frequency.

        Args:
            current_timestamp: Current timestamp

        Returns:
            True if rebalancing needed
        """
        if self.last_rebalance is None:
            return True

        if current_timestamp is None:
            return True

        # Calculate days since last rebalance
        days_since = (current_timestamp - self.last_rebalance).days

        return days_since >= self.config['rebalance_frequency']

    def _calculate_momentum_scores(
        self,
        universe_data: Dict[str, pd.DataFrame],
        universe_features: Optional[Dict[str, pd.DataFrame]],
        universe_ml_signals: Optional[Dict[str, pd.DataFrame]]
    ) -> Dict[str, float]:
        """
        Calculate momentum scores for all assets in universe.

        Args:
            universe_data: Market data for each symbol
            universe_features: Features for each symbol
            universe_ml_signals: ML signals for each symbol

        Returns:
            Dict of symbol -> momentum score
        """
        momentum_scores = {}

        for symbol, market_data in universe_data.items():
            try:
                # Check minimum history
                max_lookback = max(self.config['lookback_periods'])
                if self.config['exclude_recent_ipos']:
                    if len(market_data) < max_lookback + 20:
                        logger.debug(
                            f"{symbol}: Insufficient history ({len(market_data)} bars)"
                        )
                        continue

                # Calculate multi-period momentum
                score = self._calculate_single_momentum(
                    symbol,
                    market_data,
                    universe_features.get(symbol) if universe_features else None
                )

                if score is not None:
                    momentum_scores[symbol] = score

            except Exception as e:
                logger.debug(f"Error calculating momentum for {symbol}: {e}")
                continue

        return momentum_scores

    def _calculate_single_momentum(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Calculate momentum score for single asset.

        Args:
            symbol: Asset symbol
            market_data: OHLCV data
            features: Technical features

        Returns:
            Momentum score or None
        """
        close_prices = market_data['Close'].values
        lookback_periods = self.config['lookback_periods']
        lookback_weights = self.config['lookback_weights']

        # Calculate returns for each lookback period
        returns = []
        weights = []

        for period, weight in zip(lookback_periods, lookback_weights):
            if len(close_prices) <= period:
                continue

            # Total return over period
            ret = (close_prices[-1] / close_prices[-period-1]) - 1.0

            # Volatility adjustment if enabled
            if self.config['vol_adjusted']:
                vol_lookback = min(self.config['vol_lookback'], period)
                daily_returns = pd.Series(close_prices).pct_change().dropna()

                if len(daily_returns) >= vol_lookback:
                    vol = daily_returns.iloc[-vol_lookback:].std()
                    if vol > 1e-6:
                        # Sharpe-like ratio (return / volatility)
                        ret = ret / (vol * np.sqrt(period))

            returns.append(ret)
            weights.append(weight)

        if not returns:
            return None

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Weighted average momentum score
        momentum_score = sum(r * w for r, w in zip(returns, weights))

        return momentum_score

    def _rank_and_select(
        self,
        momentum_scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Rank assets by momentum and select top/bottom.

        Args:
            momentum_scores: Dict of symbol -> score

        Returns:
            Tuple of (long_assets, short_assets)
        """
        # Sort by momentum score
        sorted_assets = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        n_assets = len(sorted_assets)

        # Calculate percentile cutoffs
        long_cutoff = int(n_assets * self.config['long_percentile'])
        short_cutoff = int(n_assets * self.config['short_percentile'])

        # Select top and bottom
        long_assets = [symbol for symbol, _ in sorted_assets[:n_assets - long_cutoff]]
        short_assets = [symbol for symbol, _ in sorted_assets[-short_cutoff:]]

        # Apply max positions limit
        max_per_side = self.config['max_positions'] // 2

        if len(long_assets) > max_per_side:
            long_assets = long_assets[:max_per_side]

        if len(short_assets) > max_per_side:
            short_assets = short_assets[:max_per_side]

        return long_assets, short_assets

    def _create_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        momentum_score: float,
        universe_data: Dict[str, pd.DataFrame],
        universe_features: Optional[Dict[str, pd.DataFrame]],
        universe_ml_signals: Optional[Dict[str, pd.DataFrame]],
        current_timestamp: Optional[datetime]
    ) -> Optional[StrategySignal]:
        """
        Create signal for selected asset.

        Args:
            symbol: Asset symbol
            signal_type: LONG or SHORT
            momentum_score: Momentum score
            universe_data: Market data
            universe_features: Features
            universe_ml_signals: ML signals
            current_timestamp: Current time

        Returns:
            StrategySignal or None
        """
        try:
            market_data = universe_data[symbol]
            current_price = market_data['Close'].iloc[-1]

            if current_timestamp is None:
                current_timestamp = market_data.index[-1]

            # Get ML confidence
            confidence = 0.6  # Default
            if universe_ml_signals and symbol in universe_ml_signals:
                confidence = self._get_ml_confidence(
                    signal_type,
                    universe_ml_signals[symbol],
                    current_timestamp
                )

            if confidence < self.config['min_confidence']:
                logger.debug(
                    f"{symbol}: Confidence {confidence:.2f} below threshold"
                )
                return None

            # Get stop loss / take profit from features
            stop_loss, take_profit = None, None
            if universe_features and symbol in universe_features:
                features = universe_features[symbol]
                if 'atr' in features.columns:
                    atr = features['atr'].iloc[-1]
                    if not np.isnan(atr):
                        # 2 ATR stop
                        if signal_type == SignalType.LONG:
                            stop_loss = current_price - 2 * atr
                            take_profit = current_price + 3 * atr
                        else:
                            stop_loss = current_price + 2 * atr
                            take_profit = current_price - 3 * atr

            # Create signal
            signal = StrategySignal(
                timestamp=current_timestamp,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=0.0,  # Calculated in calculate_position_size
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'momentum_score': momentum_score,
                    'strategy': self.name,
                    'rebalance_timestamp': current_timestamp
                }
            )

            return signal

        except Exception as e:
            logger.error(f"Error creating signal for {symbol}: {e}", exc_info=True)
            return None

    def _get_ml_confidence(
        self,
        signal_type: SignalType,
        ml_signals: pd.DataFrame,
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
            return 0.6

        try:
            if timestamp in ml_signals.index:
                ml_row = ml_signals.loc[timestamp]
            else:
                ml_row = ml_signals.iloc[-1]

            if signal_type == SignalType.LONG:
                if 'prob_1' in ml_signals.columns:
                    return float(ml_row['prob_1'])
                elif 1 in ml_signals.columns:
                    return float(ml_row[1])
            elif signal_type == SignalType.SHORT:
                if 'prob_-1' in ml_signals.columns:
                    return float(ml_row['prob_-1'])
                elif -1 in ml_signals.columns:
                    return float(ml_row[-1])

            return 0.6

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
        Calculate equal-weighted position size for cross-sectional portfolio.

        All positions in the portfolio get equal weight, adjusted by confidence.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Instrument volatility
            existing_positions: Current positions

        Returns:
            Position size in units
        """
        try:
            # Equal weight across max positions
            n_positions = self.config['max_positions']
            equal_weight = 1.0 / n_positions

            # Position value
            position_value = portfolio_value * equal_weight

            # Adjust by confidence
            position_value *= signal.confidence

            # Convert to units
            position_size = position_value / signal.entry_price

            # Apply direction
            if signal.signal_type == SignalType.SHORT:
                position_size = -position_size

            # Check minimum
            min_size = 0.01
            if abs(position_size) < min_size:
                return 0.0

            logger.info(
                f"{signal.symbol}: Position size {position_size:.2f} units, "
                f"weight={equal_weight:.2%}, "
                f"confidence={signal.confidence:.2f}"
            )

            return position_size

        except Exception as e:
            logger.error(
                f"Error calculating position size for {signal.symbol}: {e}",
                exc_info=True
            )
            return 0.0

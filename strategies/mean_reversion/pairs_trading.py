"""
Pairs Trading Strategy Using Cointegration

Implements cointegration-based pairs trading strategy following Chan's methodology.
Uses Johansen test for cointegration, OLS for hedge ratio, and z-score based entry/exit.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from loguru import logger

from strategies.base.strategy_base import (
    StrategyBase,
    StrategySignal,
    SignalType,
    PositionSide,
    RiskMetrics
)


class PairsTradingStrategy(StrategyBase):
    """
    Cointegration-based pairs trading strategy.

    Methodology:
    1. Test for cointegration using Johansen test
    2. Calculate hedge ratio using OLS regression
    3. Compute spread and normalize to z-score
    4. Generate signals based on z-score thresholds
    5. Integrate with Step 4 meta-labels for bet sizing

    References:
        Chan, E. (2009). Quantitative Trading, Chapter 7
    """

    def __init__(
        self,
        name: str = "PairsTrading",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize pairs trading strategy.

        Args:
            name: Strategy name
            config: Configuration dict with parameters:
                - entry_threshold: Z-score threshold for entry (default: 2.0)
                - exit_threshold: Z-score threshold for exit (default: 0.5)
                - lookback_period: Lookback for cointegration test (default: 252)
                - half_life_window: Window for half-life calculation (default: 20)
                - min_confidence: Minimum meta-label confidence (default: 0.55)
                - significance_level: Johansen test significance (default: 0.05)
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'lookback_period': 252,
            'half_life_window': 20,
            'min_confidence': 0.55,
            'significance_level': 0.05,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Store cointegration parameters for each pair
        self.pair_parameters: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized {name} with entry_z={self.config['entry_threshold']}, "
                   f"exit_z={self.config['exit_threshold']}")

    def test_cointegration(
        self,
        price_series_1: pd.Series,
        price_series_2: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        Test for cointegration using Johansen test.

        Args:
            price_series_1: Price series for first asset
            price_series_2: Price series for second asset

        Returns:
            Tuple of (is_cointegrated, test_statistic, critical_value)
        """
        try:
            # Combine series into dataframe
            data = pd.DataFrame({
                'y1': price_series_1,
                'y2': price_series_2
            }).dropna()

            if len(data) < 50:
                logger.warning("Insufficient data for cointegration test")
                return False, 0.0, 0.0

            # Perform Johansen test (test_type: 0=trace, 1=maximal eigenvalue)
            result = coint_johansen(data, det_order=0, k_ar_diff=1)

            # Extract trace statistic and critical value
            trace_stat = result.lr1[0]  # First eigenvalue trace statistic

            # Critical values at 90%, 95%, 99%
            significance_map = {0.10: 0, 0.05: 1, 0.01: 2}
            sig_level = self.config['significance_level']
            crit_idx = significance_map.get(sig_level, 1)
            critical_value = result.cvt[0, crit_idx]

            is_cointegrated = trace_stat > critical_value

            logger.debug(f"Johansen test: trace_stat={trace_stat:.4f}, "
                        f"critical={critical_value:.4f}, "
                        f"cointegrated={is_cointegrated}")

            return is_cointegrated, trace_stat, critical_value

        except Exception as e:
            logger.error(f"Error in cointegration test: {e}")
            return False, 0.0, 0.0

    def calculate_hedge_ratio(
        self,
        price_series_1: pd.Series,
        price_series_2: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate hedge ratio using OLS regression.

        Regresses price_series_1 on price_series_2: y1 = beta * y2 + alpha

        Args:
            price_series_1: Dependent variable (Y)
            price_series_2: Independent variable (X)

        Returns:
            Tuple of (hedge_ratio, intercept)
        """
        try:
            # Prepare data
            data = pd.DataFrame({
                'y': price_series_1,
                'x': price_series_2
            }).dropna()

            if len(data) < 30:
                logger.warning("Insufficient data for hedge ratio calculation")
                return 1.0, 0.0

            # OLS regression: y = beta * x + alpha
            model = OLS(data['y'], data['x']).fit()
            hedge_ratio = model.params[0]
            intercept = 0.0  # No constant term in this formulation

            logger.debug(f"Hedge ratio: {hedge_ratio:.4f}, R²={model.rsquared:.4f}")

            return hedge_ratio, intercept

        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}")
            return 1.0, 0.0

    def calculate_spread(
        self,
        price_1: float,
        price_2: float,
        hedge_ratio: float
    ) -> float:
        """
        Calculate spread between pairs.

        Args:
            price_1: Price of asset 1
            price_2: Price of asset 2
            hedge_ratio: Hedge ratio from OLS

        Returns:
            Spread value
        """
        return price_1 - hedge_ratio * price_2

    def calculate_half_life(self, spread_series: pd.Series) -> float:
        """
        Calculate mean reversion half-life using OU process.

        Half-life = -log(2) / lambda, where lambda is from AR(1) model

        Args:
            spread_series: Historical spread series

        Returns:
            Half-life in periods
        """
        try:
            spread_diff = spread_series.diff().dropna()
            spread_lag = spread_series.shift(1).dropna()

            # Align series
            aligned_data = pd.DataFrame({
                'diff': spread_diff,
                'lag': spread_lag
            }).dropna()

            if len(aligned_data) < 10:
                return 20.0  # Default

            # AR(1) regression: diff = lambda * lag + epsilon
            model = OLS(aligned_data['diff'], aligned_data['lag']).fit()
            lambda_param = model.params[0]

            if lambda_param >= 0:
                return 20.0  # Not mean reverting

            half_life = -np.log(2) / lambda_param

            # Bound half-life to reasonable range
            half_life = np.clip(half_life, 1, 100)

            logger.debug(f"Calculated half-life: {half_life:.2f}")

            return half_life

        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return 20.0

    def calculate_zscore(self, spread_series: pd.Series, window: int = 20) -> float:
        """
        Calculate z-score of current spread.

        Args:
            spread_series: Historical spread series
            window: Rolling window for mean/std calculation

        Returns:
            Current z-score
        """
        if len(spread_series) < window:
            return 0.0

        spread_mean = spread_series.iloc[-window:].mean()
        spread_std = spread_series.iloc[-window:].std()

        if spread_std < 1e-8:
            return 0.0

        current_spread = spread_series.iloc[-1]
        zscore = (current_spread - spread_mean) / spread_std

        return zscore

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate pairs trading signal.

        Args:
            symbol: Pair identifier (e.g., "AAPL_MSFT")
            market_data: DataFrame with 'price_1' and 'price_2' columns
            features: Engineered features (optional)
            ml_signals: ML signals with meta-labels for confidence

        Returns:
            StrategySignal or None
        """
        try:
            if market_data is None or len(market_data) < self.config['lookback_period']:
                logger.debug(f"Insufficient data for {symbol}")
                return None

            # Extract individual price series
            if 'price_1' not in market_data.columns or 'price_2' not in market_data.columns:
                logger.warning(f"Missing price columns for {symbol}")
                return None

            price_series_1 = market_data['price_1']
            price_series_2 = market_data['price_2']

            # Test for cointegration
            is_cointegrated, trace_stat, crit_val = self.test_cointegration(
                price_series_1, price_series_2
            )

            if not is_cointegrated:
                logger.debug(f"Pair {symbol} not cointegrated")
                return None

            # Calculate hedge ratio
            hedge_ratio, intercept = self.calculate_hedge_ratio(
                price_series_1, price_series_2
            )

            # Calculate spread series
            spread_series = price_series_1 - hedge_ratio * price_series_2

            # Calculate z-score
            zscore = self.calculate_zscore(
                spread_series,
                window=self.config['half_life_window']
            )

            # Store pair parameters
            self.pair_parameters[symbol] = {
                'hedge_ratio': hedge_ratio,
                'spread_mean': spread_series.iloc[-self.config['half_life_window']:].mean(),
                'spread_std': spread_series.iloc[-self.config['half_life_window']:].std(),
                'half_life': self.calculate_half_life(spread_series),
                'cointegration_stat': trace_stat
            }

            # Generate signal based on z-score
            signal_type = None
            confidence = 0.5  # Default

            entry_threshold = self.config['entry_threshold']
            exit_threshold = self.config['exit_threshold']

            # Check if we have an existing position
            existing_position = self.positions.get(symbol)

            if existing_position is None:
                # Entry signals
                if zscore > entry_threshold:
                    # Spread too high, short spread (short asset 1, long asset 2)
                    signal_type = SignalType.SHORT
                    confidence = min(abs(zscore) / entry_threshold, 1.0)
                elif zscore < -entry_threshold:
                    # Spread too low, long spread (long asset 1, short asset 2)
                    signal_type = SignalType.LONG
                    confidence = min(abs(zscore) / entry_threshold, 1.0)
            else:
                # Exit signals
                position_side = existing_position['side']
                if position_side == PositionSide.LONG and zscore > -exit_threshold:
                    signal_type = SignalType.EXIT_LONG
                    confidence = 0.7
                elif position_side == PositionSide.SHORT and zscore < exit_threshold:
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
                    confidence = 0.6 * confidence + 0.4 * ml_confidence

            # Current prices
            current_price_1 = price_series_1.iloc[-1]
            current_price_2 = price_series_2.iloc[-1]

            # Entry price is the average of the two assets (normalized)
            entry_price = (current_price_1 + hedge_ratio * current_price_2) / 2

            # Stop loss and take profit based on spread
            spread_std = self.pair_parameters[symbol]['spread_std']
            stop_loss = entry_price * (1 - 3 * spread_std / entry_price)
            take_profit = entry_price * (1 + 2 * spread_std / entry_price)

            signal = StrategySignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=1.0,  # Will be adjusted by position sizing
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'zscore': zscore,
                    'hedge_ratio': hedge_ratio,
                    'spread': spread_series.iloc[-1],
                    'half_life': self.pair_parameters[symbol]['half_life'],
                    'cointegration_stat': trace_stat,
                    'price_1': current_price_1,
                    'price_2': current_price_2
                }
            )

            self.signals_history.append(signal)
            logger.info(f"Generated {signal_type.value} signal for {symbol}: "
                       f"zscore={zscore:.2f}, confidence={confidence:.2f}")

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
        Calculate position size using volatility-based sizing with Kelly criterion.

        For pairs trading, size is based on:
        1. Portfolio volatility target
        2. Spread volatility
        3. Meta-label confidence (from Step 4)
        4. Kelly criterion for optimal sizing

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Current spread volatility
            existing_positions: Current positions

        Returns:
            Position size in dollars
        """
        try:
            # Base position size from volatility targeting
            vol_target = self.risk_metrics.volatility_target

            if current_volatility < 1e-8:
                current_volatility = 0.15  # Default 15% volatility

            # Notional position for volatility matching
            base_size = (portfolio_value * vol_target) / current_volatility

            # Kelly criterion adjustment
            # Kelly fraction = (edge * confidence) / spread_volatility
            edge = signal.confidence - 0.5  # Excess over random
            if edge > 0:
                kelly_fraction = (edge * signal.confidence) / max(current_volatility, 0.01)
                kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Max 25% Kelly
            else:
                kelly_fraction = 0.1  # Conservative default

            adjusted_size = base_size * kelly_fraction

            # Apply maximum position limits
            max_position = portfolio_value * self.risk_metrics.max_portfolio_allocation
            adjusted_size = min(adjusted_size, max_position)

            # Ensure minimum viable size
            min_size = portfolio_value * 0.01  # At least 1%
            adjusted_size = max(adjusted_size, min_size)

            # For pairs, we need to size both legs
            if signal.metadata and 'hedge_ratio' in signal.metadata:
                hedge_ratio = signal.metadata['hedge_ratio']
                # Adjust for hedge ratio
                adjusted_size = adjusted_size / (1 + hedge_ratio)

            logger.debug(f"Position size for {signal.symbol}: ${adjusted_size:,.0f} "
                        f"(kelly={kelly_fraction:.2%}, conf={signal.confidence:.2f})")

            return adjusted_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return portfolio_value * 0.05  # Conservative 5% default

    def get_pair_parameters(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get stored parameters for a trading pair.

        Args:
            symbol: Pair symbol

        Returns:
            Dictionary of pair parameters or None
        """
        return self.pair_parameters.get(symbol)

    def update_pair_parameters(self, symbol: str, params: Dict[str, Any]):
        """
        Update parameters for a trading pair.

        Args:
            symbol: Pair symbol
            params: Parameter dictionary
        """
        self.pair_parameters[symbol] = params
        logger.debug(f"Updated parameters for {symbol}")

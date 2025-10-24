"""
Ornstein-Uhlenbeck Process Mean Reversion Strategy

Implements mean reversion strategy based on OU process parameter estimation.
Calculates optimal entry/exit thresholds using statistical properties of OU process.
"""

from datetime import datetime
from typing import Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
from loguru import logger

from strategies.base.strategy_base import (
    StrategyBase,
    StrategySignal,
    SignalType,
    PositionSide,
    RiskMetrics
)


class OUProcessStrategy(StrategyBase):
    """
    Ornstein-Uhlenbeck process mean reversion strategy.

    The OU process follows: dX_t = θ(μ - X_t)dt + σdW_t
    where:
        θ (theta): Mean reversion speed
        μ (mu): Long-term mean
        σ (sigma): Volatility of the process

    Methodology:
    1. Estimate OU parameters (θ, μ, σ) from price data
    2. Calculate mean reversion half-life: ln(2)/θ
    3. Determine optimal entry/exit thresholds
    4. Generate signals when price deviates significantly from mean
    5. Integrate with Step 4 meta-labels for bet sizing

    References:
        Chan, E. (2013). Algorithmic Trading, Chapter 4
        Vasicek, O. (1977). An equilibrium characterization of the term structure
    """

    def __init__(
        self,
        name: str = "OUProcess",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize OU process strategy.

        Args:
            name: Strategy name
            config: Configuration dict with parameters:
                - estimation_window: Window for parameter estimation (default: 60)
                - entry_threshold: Number of std devs for entry (default: 2.0)
                - exit_threshold: Number of std devs for exit (default: 0.5)
                - min_half_life: Minimum acceptable half-life in days (default: 1)
                - max_half_life: Maximum acceptable half-life in days (default: 30)
                - min_confidence: Minimum meta-label confidence (default: 0.55)
                - use_kalman_filter: Use Kalman filter for parameter updates (default: False)
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'estimation_window': 60,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'min_half_life': 1,
            'max_half_life': 30,
            'min_confidence': 0.55,
            'use_kalman_filter': False,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Store OU parameters for each symbol
        self.ou_parameters: Dict[str, Dict[str, float]] = {}

        logger.info(f"Initialized {name} with entry_threshold={self.config['entry_threshold']}, "
                   f"estimation_window={self.config['estimation_window']}")

    def estimate_ou_parameters(
        self,
        price_series: pd.Series,
        dt: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """
        Estimate OU process parameters using maximum likelihood.

        Uses discrete-time approximation:
        X_{t+1} = X_t + θ(μ - X_t)Δt + σ√Δt * ε_t

        Can be rewritten as AR(1):
        X_{t+1} = a + b*X_t + ε_t
        where: a = θμΔt, b = 1 - θΔt

        Args:
            price_series: Price series (preferably log prices)
            dt: Time step (default: 1 day)

        Returns:
            Tuple of (theta, mu, sigma, r_squared)
        """
        try:
            prices = price_series.dropna()

            if len(prices) < 30:
                logger.warning("Insufficient data for OU parameter estimation")
                return 0.0, prices.mean(), prices.std(), 0.0

            # Prepare data for regression
            y = prices.iloc[1:].values
            x = prices.iloc[:-1].values

            # Add constant term
            X = np.column_stack([np.ones(len(x)), x])

            # OLS estimation: y = a + b*x + ε
            params = np.linalg.lstsq(X, y, rcond=None)[0]
            a, b = params

            # Convert to OU parameters
            theta = -(np.log(b)) / dt if b > 0 and b < 1 else 0.1
            mu = a / (theta * dt) if theta > 0 else prices.mean()

            # Estimate sigma from residuals
            residuals = y - (a + b * x)
            sigma = np.std(residuals) / np.sqrt(dt)

            # Calculate R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Validate parameters
            if theta <= 0 or theta > 10:  # Unreasonable theta
                theta = 0.1
            if sigma <= 0 or sigma > prices.std() * 3:  # Unreasonable sigma
                sigma = prices.std()

            logger.debug(f"Estimated OU params: θ={theta:.4f}, μ={mu:.4f}, "
                        f"σ={sigma:.4f}, R²={r_squared:.4f}")

            return theta, mu, sigma, r_squared

        except Exception as e:
            logger.error(f"Error estimating OU parameters: {e}")
            mean_price = price_series.mean()
            std_price = price_series.std()
            return 0.1, mean_price, std_price, 0.0

    def calculate_half_life(self, theta: float) -> float:
        """
        Calculate mean reversion half-life.

        Half-life = ln(2) / θ

        Args:
            theta: Mean reversion speed

        Returns:
            Half-life in time units (typically days)
        """
        if theta <= 0:
            return float('inf')

        half_life = np.log(2) / theta
        return half_life

    def calculate_equilibrium_distribution(
        self,
        theta: float,
        mu: float,
        sigma: float
    ) -> Tuple[float, float]:
        """
        Calculate equilibrium (stationary) distribution of OU process.

        For OU process, the equilibrium distribution is:
        X ~ N(μ, σ²/(2θ))

        Args:
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Volatility

        Returns:
            Tuple of (equilibrium_mean, equilibrium_std)
        """
        eq_mean = mu
        eq_std = sigma / np.sqrt(2 * theta) if theta > 0 else sigma

        return eq_mean, eq_std

    def calculate_optimal_thresholds(
        self,
        theta: float,
        mu: float,
        sigma: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate optimal entry/exit thresholds based on OU parameters.

        Uses equilibrium distribution to determine thresholds that maximize
        expected profit while controlling risk.

        Args:
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Volatility
            confidence_level: Confidence level for thresholds

        Returns:
            Tuple of (upper_threshold, lower_threshold)
        """
        eq_mean, eq_std = self.calculate_equilibrium_distribution(theta, mu, sigma)

        # Calculate thresholds based on confidence level
        z_score = norm.ppf((1 + confidence_level) / 2)

        upper_threshold = eq_mean + z_score * eq_std
        lower_threshold = eq_mean - z_score * eq_std

        return upper_threshold, lower_threshold

    def calculate_expected_return_time(
        self,
        current_price: float,
        mu: float,
        theta: float
    ) -> float:
        """
        Calculate expected time for price to return to mean.

        For OU process: E[X_t | X_0] = μ + (X_0 - μ)e^(-θt)
        Solving for t when E[X_t] ≈ μ (say, 90% of the way):
        t = -ln(0.1) / θ

        Args:
            current_price: Current price level
            mu: Long-term mean
            theta: Mean reversion speed

        Returns:
            Expected return time in periods
        """
        if theta <= 0:
            return float('inf')

        # Time to reach 90% of the way to mean
        return -np.log(0.1) / theta

    def calculate_ou_zscore(
        self,
        current_price: float,
        mu: float,
        theta: float,
        sigma: float
    ) -> float:
        """
        Calculate standardized deviation from mean using OU parameters.

        Z-score = (X - μ) / (σ/√(2θ))

        Args:
            current_price: Current price
            mu: Long-term mean
            theta: Mean reversion speed
            sigma: Volatility

        Returns:
            OU-based z-score
        """
        eq_mean, eq_std = self.calculate_equilibrium_distribution(theta, mu, sigma)

        if eq_std < 1e-8:
            return 0.0

        zscore = (current_price - eq_mean) / eq_std
        return zscore

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate OU process-based mean reversion signal.

        Args:
            symbol: Instrument symbol
            market_data: OHLCV DataFrame with 'close' column
            features: Engineered features (optional)
            ml_signals: ML signals with meta-labels for confidence

        Returns:
            StrategySignal or None
        """
        try:
            required_length = self.config['estimation_window'] + 10

            if market_data is None or len(market_data) < required_length:
                logger.debug(f"Insufficient data for {symbol}")
                return None

            # Extract close prices (or use log prices for better OU fit)
            if 'close' not in market_data.columns:
                logger.warning(f"Missing 'close' column for {symbol}")
                return None

            close_prices = market_data['close']

            # Use log prices for OU estimation (more stationary)
            log_prices = np.log(close_prices)

            # Estimate OU parameters
            theta, mu, sigma, r_squared = self.estimate_ou_parameters(
                log_prices.iloc[-self.config['estimation_window']:],
                dt=1.0
            )

            # Calculate half-life
            half_life = self.calculate_half_life(theta)

            # Validate half-life
            if (half_life < self.config['min_half_life'] or
                half_life > self.config['max_half_life']):
                logger.debug(f"Half-life {half_life:.1f} outside acceptable range for {symbol}")
                return None

            # Store parameters
            self.ou_parameters[symbol] = {
                'theta': theta,
                'mu': mu,
                'sigma': sigma,
                'half_life': half_life,
                'r_squared': r_squared
            }

            # Calculate current position relative to equilibrium
            current_log_price = log_prices.iloc[-1]
            zscore = self.calculate_ou_zscore(current_log_price, mu, theta, sigma)

            # Calculate optimal thresholds
            upper_threshold, lower_threshold = self.calculate_optimal_thresholds(
                theta, mu, sigma, confidence_level=0.95
            )

            # Generate signals
            signal_type = None
            confidence = 0.5

            entry_threshold = self.config['entry_threshold']
            exit_threshold = self.config['exit_threshold']

            # Check existing position
            existing_position = self.positions.get(symbol)

            if existing_position is None:
                # Entry signals based on z-score
                if zscore > entry_threshold:
                    # Price too high, expect reversion down
                    signal_type = SignalType.SHORT
                    # Confidence increases with deviation and mean reversion speed
                    confidence = min(0.5 + (abs(zscore) - entry_threshold) * 0.2, 1.0)
                    confidence *= min(theta, 1.0)  # Stronger theta = higher confidence

                elif zscore < -entry_threshold:
                    # Price too low, expect reversion up
                    signal_type = SignalType.LONG
                    confidence = min(0.5 + (abs(zscore) - entry_threshold) * 0.2, 1.0)
                    confidence *= min(theta, 1.0)

            else:
                # Exit signals
                position_side = existing_position['side']

                if position_side == PositionSide.LONG and zscore > -exit_threshold:
                    # Exit long when price reverts to mean
                    signal_type = SignalType.EXIT_LONG
                    confidence = 0.7

                elif position_side == PositionSide.SHORT and zscore < exit_threshold:
                    # Exit short when price reverts to mean
                    signal_type = SignalType.EXIT_SHORT
                    confidence = 0.7

            if signal_type is None:
                return None

            # Boost confidence with R-squared (fit quality)
            confidence *= (0.5 + 0.5 * r_squared)

            # Integrate meta-labels from Step 4 if available
            if ml_signals is not None and len(ml_signals) > 0:
                latest_ml = ml_signals.iloc[-1]
                if 'meta_label_prob' in latest_ml:
                    ml_confidence = latest_ml['meta_label_prob']
                    # Blend strategy confidence with ML confidence
                    confidence = 0.6 * confidence + 0.4 * ml_confidence

            # Convert back to price space
            current_price = close_prices.iloc[-1]
            eq_mean_price = np.exp(mu)
            eq_std_price = eq_mean_price * (sigma / np.sqrt(2 * theta))

            # Calculate stop loss and take profit
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                expected_return_time = self.calculate_expected_return_time(
                    current_log_price, mu, theta
                )

                if signal_type == SignalType.LONG:
                    # Stop loss: 2 equilibrium std devs below current
                    stop_loss = current_price * (1 - 2 * eq_std_price / current_price)
                    # Take profit: at mean
                    take_profit = eq_mean_price
                else:  # SHORT
                    # Stop loss: 2 equilibrium std devs above current
                    stop_loss = current_price * (1 + 2 * eq_std_price / current_price)
                    # Take profit: at mean
                    take_profit = eq_mean_price
            else:
                stop_loss = None
                take_profit = None

            signal = StrategySignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=1.0,  # Will be adjusted by position sizing
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'theta': theta,
                    'mu': mu,
                    'sigma': sigma,
                    'half_life': half_life,
                    'zscore': zscore,
                    'r_squared': r_squared,
                    'equilibrium_mean': eq_mean_price,
                    'equilibrium_std': eq_std_price,
                    'upper_threshold': np.exp(upper_threshold),
                    'lower_threshold': np.exp(lower_threshold),
                    'expected_return_time': expected_return_time if signal_type in [SignalType.LONG, SignalType.SHORT] else None
                }
            )

            self.signals_history.append(signal)
            logger.info(f"Generated {signal_type.value} signal for {symbol}: "
                       f"zscore={zscore:.2f}, half_life={half_life:.1f}, "
                       f"conf={confidence:.2f}")

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
        Calculate position size using OU parameters and Kelly criterion.

        Position sizing considers:
        1. Mean reversion speed (theta) - faster reversion = larger size
        2. Half-life - shorter half-life = larger size
        3. Z-score magnitude - larger deviation = larger size
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

            # Extract OU parameters from metadata
            metadata = signal.metadata or {}
            theta = metadata.get('theta', 0.1)
            half_life = metadata.get('half_life', 20)
            zscore = abs(metadata.get('zscore', 1.0))

            # Kelly criterion for OU process
            # Kelly fraction ≈ (edge * theta) / sigma²
            # Simplified: use theta as proxy for edge strength
            edge = signal.confidence - 0.5
            if edge > 0:
                kelly_fraction = (edge * theta * zscore) / (current_volatility ** 2)
                kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Max 25% Kelly
            else:
                kelly_fraction = 0.05

            # Adjust for half-life (shorter = faster returns = larger size)
            half_life_factor = np.clip(20 / max(half_life, 1), 0.5, 2.0)
            kelly_fraction *= half_life_factor

            adjusted_size = base_size * kelly_fraction

            # Apply maximum position limits
            max_position = portfolio_value * self.risk_metrics.max_portfolio_allocation
            adjusted_size = min(adjusted_size, max_position)

            # Ensure minimum viable size
            min_size = portfolio_value * 0.01
            adjusted_size = max(adjusted_size, min_size)

            logger.debug(f"Position size for {signal.symbol}: ${adjusted_size:,.0f} "
                        f"(kelly={kelly_fraction:.2%}, theta={theta:.3f}, "
                        f"half_life={half_life:.1f})")

            return adjusted_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return portfolio_value * 0.05

    def get_ou_parameters(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get stored OU parameters for a symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            Dictionary of OU parameters or None
        """
        return self.ou_parameters.get(symbol)

    def update_ou_parameters(self, symbol: str, params: Dict[str, float]):
        """
        Update OU parameters for a symbol.

        Args:
            symbol: Instrument symbol
            params: Parameter dictionary
        """
        self.ou_parameters[symbol] = params
        logger.debug(f"Updated OU parameters for {symbol}")

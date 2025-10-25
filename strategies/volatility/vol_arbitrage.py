"""
Volatility Arbitrage Strategy

Trades the difference between implied and realized volatility:
- Compare option implied volatility vs historical realized volatility
- Volatility term structure analysis
- Delta-neutral portfolio construction
- Greeks calculation and management
"""

from datetime import datetime
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


class VolArbitrageStrategy(StrategyBase):
    """
    Volatility arbitrage strategy.

    Core idea:
    - When implied vol > realized vol: SELL volatility (short options, delta hedge)
    - When realized vol > implied vol: BUY volatility (long options, delta hedge)
    - Maintain delta-neutral positions
    - Profit from mean reversion in volatility spread
    """

    def __init__(
        self,
        name: str = "VolArbitrage",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize volatility arbitrage strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - iv_lookback: Lookback for implied vol estimation (default: 30)
                - rv_lookback: Lookback for realized vol (default: 30)
                - entry_threshold: Vol spread threshold to enter (default: 0.05, 5%)
                - exit_threshold: Vol spread threshold to exit (default: 0.02, 2%)
                - delta_hedge_threshold: Delta threshold for rehedging (default: 0.10)
                - term_structure_slope_threshold: Min slope for term structure signal (default: 0.02)
                - min_confidence: Minimum ML confidence (default: 0.65)
            risk_metrics: Risk limits
        """
        default_config = {
            'iv_lookback': 30,
            'rv_lookback': 30,
            'entry_threshold': 0.05,  # 5% vol spread
            'exit_threshold': 0.02,   # 2% vol spread
            'delta_hedge_threshold': 0.10,  # Rehedge when delta > 0.10
            'term_structure_slope_threshold': 0.02,
            'min_observations': 30,
            'min_confidence': 0.65,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Volatility tracking
        self.implied_vols: Dict[str, float] = {}
        self.realized_vols: Dict[str, float] = {}
        self.vol_spreads: Dict[str, float] = {}
        self.greeks: Dict[str, Dict[str, float]] = {}

        logger.info(
            f"Initialized {name} with entry_threshold={self.config['entry_threshold']:.1%}, "
            f"rv_lookback={self.config['rv_lookback']}d"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate volatility arbitrage signal.

        Signal logic:
        1. Calculate realized volatility from price data
        2. Extract/estimate implied volatility (from features if available)
        3. Compute volatility spread (IV - RV)
        4. Analyze volatility term structure
        5. Generate signal when spread exceeds threshold
        6. Calculate delta for hedging

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data
            features: Features including implied_vol if available
            ml_signals: ML predictions

        Returns:
            StrategySignal or None
        """
        try:
            if len(market_data) < self.config['min_observations']:
                logger.warning(f"{symbol}: Insufficient data ({len(market_data)} rows)")
                return None

            # Calculate realized volatility
            realized_vol = self._calculate_realized_volatility(market_data)
            if realized_vol is None or realized_vol <= 0:
                logger.warning(f"{symbol}: Invalid realized volatility")
                return None

            self.realized_vols[symbol] = realized_vol

            # Extract/estimate implied volatility
            implied_vol = self._get_implied_volatility(symbol, market_data, features)
            if implied_vol is None or implied_vol <= 0:
                logger.warning(f"{symbol}: Invalid implied volatility")
                return None

            self.implied_vols[symbol] = implied_vol

            # Calculate volatility spread
            vol_spread = implied_vol - realized_vol
            vol_spread_pct = vol_spread / realized_vol
            self.vol_spreads[symbol] = vol_spread_pct

            # Analyze term structure
            term_structure_signal = self._analyze_term_structure(features)

            # Get ML confidence
            confidence = self._get_ml_confidence(ml_signals)

            # Generate signal based on spread
            signal_type = self._determine_signal_type(
                vol_spread_pct,
                term_structure_signal,
                symbol
            )

            if signal_type == SignalType.HOLD:
                return None

            # Calculate Greeks for delta-neutral construction
            greeks = self._calculate_greeks(
                symbol,
                market_data['close'].iloc[-1],
                implied_vol,
                time_to_expiry=30/365  # Assume 30-day options
            )
            self.greeks[symbol] = greeks

            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()
            current_price = float(market_data['close'].iloc[-1])

            # Calculate stop loss and take profit
            # For vol arb, stops based on vol spread reversal
            stop_loss_vol_change = realized_vol * 0.20  # 20% vol change
            if signal_type == SignalType.SHORT:  # Selling vol
                # Stop if realized vol increases significantly
                stop_loss = current_price + stop_loss_vol_change
                take_profit = current_price - realized_vol * 0.10
            else:  # Buying vol
                # Stop if realized vol decreases significantly
                stop_loss = current_price - stop_loss_vol_change
                take_profit = current_price + realized_vol * 0.10

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
                    'strategy': self.name,
                    'implied_vol': implied_vol,
                    'realized_vol': realized_vol,
                    'vol_spread': vol_spread_pct,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'vega': greeks['vega'],
                    'theta': greeks['theta'],
                    'term_structure_signal': term_structure_signal
                }
            )

            self.signals_history.append(signal)
            logger.info(
                f"{symbol} vol arb signal: {signal_type.value}, "
                f"IV={implied_vol:.2%}, RV={realized_vol:.2%}, "
                f"spread={vol_spread_pct:.2%}, confidence={confidence:.2%}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating vol arb signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate delta-neutral position size.

        Position sizing considers:
        1. Portfolio risk budget
        2. Delta neutrality (may need stock hedge)
        3. Vega exposure limits
        4. Gamma exposure

        Args:
            signal: Trading signal with Greeks in metadata
            portfolio_value: Total portfolio value
            current_volatility: Current instrument volatility
            existing_positions: Current positions

        Returns:
            Position size in units
        """
        try:
            if portfolio_value <= 0:
                return 0.0

            # Base position size on portfolio risk budget
            risk_budget = self.config.get('risk_budget', 0.05)
            base_position_value = portfolio_value * risk_budget

            # Adjust for volatility (higher vol = smaller position)
            vol_scalar = 0.20 / max(current_volatility, 0.01)  # Target 20% vol
            position_value = base_position_value * vol_scalar

            # Scale by ML confidence
            position_value *= signal.confidence

            # Calculate position size
            position_size = position_value / signal.entry_price

            # Apply portfolio allocation limits
            max_position_value = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_size = max_position_value / signal.entry_price
            position_size = min(position_size, max_size)

            # Check if delta hedging is needed
            delta = signal.metadata.get('delta', 0.0)
            if abs(delta) > self.config['delta_hedge_threshold']:
                logger.info(
                    f"{signal.symbol}: High delta exposure {delta:.3f}, "
                    f"consider hedging with {abs(delta * position_size):.2f} shares"
                )

            logger.debug(
                f"{signal.symbol} vol arb size: {position_size:.2f} units "
                f"(delta={signal.metadata.get('delta', 0):.3f}, "
                f"vega={signal.metadata.get('vega', 0):.3f})"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating vol arb position size: {e}", exc_info=True)
            return 0.0

    def _calculate_realized_volatility(self, market_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate realized volatility using close-to-close returns.

        Args:
            market_data: Price data

        Returns:
            Annualized realized volatility
        """
        try:
            lookback = self.config['rv_lookback']
            returns = np.log(market_data['close'] / market_data['close'].shift(1))
            returns = returns.dropna().tail(lookback)

            if len(returns) < lookback // 2:
                return None

            realized_vol = returns.std() * np.sqrt(252)  # Annualize
            return float(realized_vol)

        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return None

    def _get_implied_volatility(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Get implied volatility from features or estimate from price action.

        Args:
            symbol: Instrument symbol
            market_data: Price data
            features: Feature DataFrame that may contain 'implied_vol' column

        Returns:
            Implied volatility or None
        """
        try:
            # Try to get from features first
            if features is not None and 'implied_vol' in features.columns:
                iv = features['implied_vol'].iloc[-1]
                if pd.notna(iv) and iv > 0:
                    return float(iv)

            # Estimate using Parkinson's high-low estimator
            # IV ≈ sqrt((1/(4*ln(2))) * (ln(H/L))^2)
            lookback = self.config['iv_lookback']
            recent_data = market_data.tail(lookback)

            hl_ratios = np.log(recent_data['high'] / recent_data['low'])
            parkinson_var = (1 / (4 * np.log(2))) * (hl_ratios ** 2).mean()
            estimated_iv = np.sqrt(parkinson_var * 252)  # Annualize

            logger.debug(f"{symbol}: Estimated IV={estimated_iv:.2%} using Parkinson estimator")
            return float(estimated_iv)

        except Exception as e:
            logger.error(f"Error getting implied volatility: {e}")
            return None

    def _analyze_term_structure(self, features: Optional[pd.DataFrame]) -> float:
        """
        Analyze volatility term structure.

        Upward sloping (contango): IV increases with maturity
        Downward sloping (backwardation): IV decreases with maturity

        Args:
            features: Features that may include term structure data

        Returns:
            Term structure signal (-1 to 1)
        """
        try:
            if features is None:
                return 0.0

            # Look for term structure columns: iv_30d, iv_60d, iv_90d, etc.
            term_cols = [col for col in features.columns if col.startswith('iv_') and col.endswith('d')]

            if len(term_cols) < 2:
                return 0.0

            # Get most recent values
            latest = features[term_cols].iloc[-1]
            latest = latest.dropna()

            if len(latest) < 2:
                return 0.0

            # Calculate slope (simple linear regression)
            maturities = [int(col.replace('iv_', '').replace('d', '')) for col in latest.index]
            vols = latest.values

            # Simple slope calculation
            slope = (vols[-1] - vols[0]) / (maturities[-1] - maturities[0])

            # Normalize to [-1, 1]
            # Positive slope (contango) suggests selling vol
            # Negative slope (backwardation) suggests buying vol
            normalized_slope = np.tanh(slope / 0.01)  # Divide by 0.01 to scale

            return float(normalized_slope)

        except Exception as e:
            logger.debug(f"Error analyzing term structure: {e}")
            return 0.0

    def _calculate_greeks(
        self,
        symbol: str,
        spot_price: float,
        volatility: float,
        time_to_expiry: float,
        strike: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate option Greeks for delta-neutral hedging.

        Using Black-Scholes for simplification.

        Args:
            symbol: Instrument symbol
            spot_price: Current spot price
            volatility: Implied volatility
            time_to_expiry: Time to expiry in years
            strike: Strike price (defaults to ATM)

        Returns:
            Dictionary with delta, gamma, vega, theta
        """
        try:
            from scipy.stats import norm

            # ATM option if strike not specified
            if strike is None:
                strike = spot_price

            risk_free_rate = 0.02  # 2% risk-free rate

            # Black-Scholes Greeks (simplified, call option)
            d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
                 (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)

            # Delta
            delta = norm.cdf(d1)

            # Gamma
            gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))

            # Vega (per 1% change in vol)
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100

            # Theta (per day)
            theta = -(spot_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) / 365

            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'vega': float(vega),
                'theta': float(theta)
            }

        except ImportError:
            logger.warning("scipy not available, using simplified Greeks")
            # Simplified estimates
            return {
                'delta': 0.5,  # ATM delta
                'gamma': 0.05,
                'vega': 0.10,
                'theta': -0.01
            }
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    def _get_ml_confidence(self, ml_signals: Optional[pd.DataFrame]) -> float:
        """
        Extract confidence from ML signals.

        Args:
            ml_signals: ML predictions

        Returns:
            Confidence score [0, 1]
        """
        if ml_signals is None or len(ml_signals) == 0:
            return 0.65  # Neutral confidence

        try:
            confidence = ml_signals['confidence'].iloc[-1]
            return float(confidence)
        except Exception:
            return 0.65

    def _determine_signal_type(
        self,
        vol_spread_pct: float,
        term_structure_signal: float,
        symbol: str
    ) -> SignalType:
        """
        Determine signal type based on volatility spread and term structure.

        Args:
            vol_spread_pct: (IV - RV) / RV
            term_structure_signal: Term structure slope signal
            symbol: Instrument symbol

        Returns:
            SignalType
        """
        entry_threshold = self.config['entry_threshold']
        exit_threshold = self.config['exit_threshold']

        current_position = self.positions.get(symbol, {})
        has_position = bool(current_position)

        # Entry logic
        if not has_position:
            # IV > RV significantly: SELL volatility (SHORT)
            if vol_spread_pct > entry_threshold:
                # Confirm with term structure (contango supports selling)
                if term_structure_signal > 0:
                    return SignalType.SHORT
                elif term_structure_signal > -0.5:  # Not strongly negative
                    return SignalType.SHORT

            # RV > IV significantly: BUY volatility (LONG)
            elif vol_spread_pct < -entry_threshold:
                # Confirm with term structure (backwardation supports buying)
                if term_structure_signal < 0:
                    return SignalType.LONG
                elif term_structure_signal < 0.5:  # Not strongly positive
                    return SignalType.LONG

        # Exit logic
        else:
            position_side = current_position.get('side', PositionSide.FLAT)

            # Exit SHORT position (sold vol) when spread narrows
            if position_side == PositionSide.SHORT and abs(vol_spread_pct) < exit_threshold:
                return SignalType.EXIT_SHORT

            # Exit LONG position (bought vol) when spread narrows
            if position_side == PositionSide.LONG and abs(vol_spread_pct) < exit_threshold:
                return SignalType.EXIT_LONG

        return SignalType.HOLD

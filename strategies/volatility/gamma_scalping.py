"""
Gamma Scalping Strategy

Profit from gamma by:
- Holding long options for positive gamma
- Delta hedging frequently to capture realized volatility
- Rebalancing when delta exceeds threshold
- Optimizing rehedge frequency vs transaction costs
"""

from datetime import datetime, timedelta
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


class GammaScalpingStrategy(StrategyBase):
    """
    Gamma scalping strategy.

    Core mechanics:
    1. Buy ATM straddles/strangles for positive gamma
    2. Delta hedge to remain market neutral
    3. Rehedge when delta exceeds threshold
    4. Profit when realized vol > implied vol paid
    5. Optimize rehedge frequency vs costs
    """

    def __init__(
        self,
        name: str = "GammaScalping",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize gamma scalping strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - delta_threshold: Delta threshold to trigger rehedge (default: 0.15)
                - gamma_threshold: Minimum gamma to enter trade (default: 0.05)
                - rv_iv_threshold: Min RV - IV spread to enter (default: 0.03, 3%)
                - rehedge_interval_hours: Min time between rehedges (default: 4)
                - max_rehedge_per_day: Max rehedges per day (default: 6)
                - transaction_cost_bps: Transaction cost in bps (default: 5)
                - time_decay_tolerance: Max daily theta loss as % of premium (default: 0.02)
            risk_metrics: Risk limits
        """
        default_config = {
            'delta_threshold': 0.15,
            'gamma_threshold': 0.05,
            'rv_iv_threshold': 0.03,  # Enter when RV > IV by 3%
            'rehedge_interval_hours': 4,
            'max_rehedge_per_day': 6,
            'transaction_cost_bps': 5,
            'time_decay_tolerance': 0.02,
            'min_observations': 30,
            'min_confidence': 0.60,
            'lookback_days': 30,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Gamma scalping tracking
        self.current_delta: Dict[str, float] = {}
        self.current_gamma: Dict[str, float] = {}
        self.last_hedge_time: Dict[str, datetime] = {}
        self.rehedge_count: Dict[str, int] = {}
        self.hedge_pnl: Dict[str, float] = {}

        logger.info(
            f"Initialized {name} with delta_threshold={self.config['delta_threshold']}, "
            f"rehedge_interval={self.config['rehedge_interval_hours']}h"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate gamma scalping signal.

        Signal logic:
        1. Calculate realized vs implied volatility
        2. Check if gamma opportunity exists (RV > IV)
        3. Determine if delta hedging is needed
        4. Generate signal for new position or rehedge

        Args:
            symbol: Instrument symbol
            market_data: OHLCV data
            features: Features including volatility data
            ml_signals: ML predictions

        Returns:
            StrategySignal or None
        """
        try:
            if len(market_data) < self.config['min_observations']:
                logger.warning(f"{symbol}: Insufficient data")
                return None

            # Calculate realized volatility
            realized_vol = self._calculate_realized_vol(market_data)
            if realized_vol is None:
                return None

            # Get implied volatility
            implied_vol = self._get_implied_vol(features)
            if implied_vol is None:
                return None

            # Calculate current Greeks
            current_price = float(market_data['close'].iloc[-1])
            greeks = self._calculate_greeks(current_price, implied_vol)

            self.current_delta[symbol] = greeks['delta']
            self.current_gamma[symbol] = greeks['gamma']

            # Check if we have existing position
            current_position = self.positions.get(symbol, {})
            has_position = bool(current_position)

            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()

            # Determine signal type
            if has_position:
                # Check if rehedge is needed
                signal_type = self._check_rehedge_needed(symbol, greeks, timestamp)
                if signal_type == SignalType.HOLD:
                    return None
            else:
                # Check if entry condition is met
                vol_spread = realized_vol - implied_vol
                if vol_spread < self.config['rv_iv_threshold']:
                    logger.debug(f"{symbol}: Vol spread {vol_spread:.2%} below threshold")
                    return None

                if greeks['gamma'] < self.config['gamma_threshold']:
                    logger.debug(f"{symbol}: Gamma {greeks['gamma']:.4f} below threshold")
                    return None

                signal_type = SignalType.LONG  # Buy options for positive gamma

            # Get ML confidence
            confidence = self._get_ml_confidence(ml_signals)

            # Calculate stop loss and take profit
            # For gamma scalping, stops based on realized vs implied vol
            vol_stop_distance = realized_vol * current_price * np.sqrt(1/252)
            stop_loss = current_price - vol_stop_distance
            take_profit = current_price + vol_stop_distance * 1.5

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
                    'realized_vol': realized_vol,
                    'implied_vol': implied_vol,
                    'vol_spread': realized_vol - implied_vol,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'vega': greeks['vega'],
                    'theta': greeks['theta'],
                    'is_rehedge': has_position,
                    'rehedge_count_today': self.rehedge_count.get(symbol, 0)
                }
            )

            self.signals_history.append(signal)
            logger.info(
                f"{symbol} gamma scalp signal: {signal_type.value}, "
                f"RV={realized_vol:.2%}, IV={implied_vol:.2%}, "
                f"gamma={greeks['gamma']:.4f}, delta={greeks['delta']:.3f}"
            )

            # Update last hedge time
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                self.last_hedge_time[symbol] = timestamp
                self.rehedge_count[symbol] = self.rehedge_count.get(symbol, 0) + 1

            return signal

        except Exception as e:
            logger.error(f"Error generating gamma scalp signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size for gamma scalping.

        For gamma scalping, size based on:
        1. Gamma exposure desired
        2. Risk budget for theta decay
        3. Transaction costs vs expected gamma profit
        4. Delta hedging requirements

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_volatility: Current instrument volatility
            existing_positions: Current positions

        Returns:
            Position size in option contracts (or delta-equivalent shares)
        """
        try:
            if portfolio_value <= 0:
                return 0.0

            is_rehedge = signal.metadata.get('is_rehedge', False)

            if is_rehedge:
                # For rehedging, size is determined by delta neutralization
                return self._calculate_hedge_size(signal, existing_positions)

            # For new positions, size based on risk budget
            risk_budget = self.config.get('risk_budget', 0.05)
            position_value = portfolio_value * risk_budget

            # Adjust for theta decay - don't size too large if theta is high
            theta = abs(signal.metadata.get('theta', 0.01))
            theta_tolerance = self.config['time_decay_tolerance']
            max_contracts_theta = (portfolio_value * theta_tolerance) / (theta * 252)  # Daily theta

            # Adjust for volatility
            vol_scalar = 0.20 / max(current_volatility, 0.01)
            position_value *= vol_scalar

            # Scale by ML confidence
            position_value *= signal.confidence

            # Convert to contracts (assuming 100 shares per contract)
            contracts = position_value / (signal.entry_price * 100)

            # Apply theta constraint
            contracts = min(contracts, max_contracts_theta / 100)

            # Apply portfolio allocation limit
            max_allocation = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_contracts = max_allocation / (signal.entry_price * 100)
            contracts = min(contracts, max_contracts)

            # Round to whole contracts
            contracts = np.floor(contracts)

            # Convert back to shares for position tracking
            position_size = contracts * 100

            logger.debug(
                f"{signal.symbol} gamma scalp size: {contracts} contracts "
                f"({position_size:.0f} shares, gamma={signal.metadata.get('gamma', 0):.4f})"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating gamma scalp position size: {e}", exc_info=True)
            return 0.0

    def _calculate_realized_vol(self, market_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate realized volatility for gamma scalping.

        Uses Parkinson (high-low) estimator for better intraday vol capture.

        Args:
            market_data: Price data

        Returns:
            Annualized realized volatility
        """
        try:
            lookback = self.config['lookback_days']
            recent_data = market_data.tail(lookback)

            # Parkinson's high-low estimator
            hl_ratios = np.log(recent_data['high'] / recent_data['low'])
            parkinson_var = (1 / (4 * np.log(2))) * (hl_ratios ** 2).mean()
            realized_vol = np.sqrt(parkinson_var * 252)

            return float(realized_vol)

        except Exception as e:
            logger.error(f"Error calculating realized vol: {e}")
            return None

    def _get_implied_vol(self, features: Optional[pd.DataFrame]) -> Optional[float]:
        """
        Get implied volatility from features.

        Args:
            features: Feature DataFrame

        Returns:
            Implied volatility
        """
        try:
            if features is None or 'implied_vol' not in features.columns:
                logger.warning("Implied vol not available in features")
                return None

            iv = features['implied_vol'].iloc[-1]
            if pd.isna(iv) or iv <= 0:
                return None

            return float(iv)

        except Exception as e:
            logger.error(f"Error getting implied vol: {e}")
            return None

    def _calculate_greeks(
        self,
        spot_price: float,
        volatility: float,
        time_to_expiry: float = 30/365,
        strike: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks for gamma scalping.

        Args:
            spot_price: Current price
            volatility: Implied volatility
            time_to_expiry: Time to expiry in years
            strike: Strike price (ATM if None)

        Returns:
            Greeks dictionary
        """
        try:
            from scipy.stats import norm

            if strike is None:
                strike = spot_price  # ATM

            r = 0.02  # Risk-free rate

            d1 = (np.log(spot_price / strike) + (r + 0.5 * volatility ** 2) * time_to_expiry) / \
                 (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)

            # For straddle (long call + long put)
            call_delta = norm.cdf(d1)
            put_delta = call_delta - 1
            straddle_delta = call_delta + put_delta  # Should be ~0 for ATM

            # Gamma is same for call and put
            gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
            straddle_gamma = 2 * gamma  # Double gamma for straddle

            # Vega
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
            straddle_vega = 2 * vega

            # Theta
            call_theta = -(spot_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))
            put_theta = call_theta  # Approximately same for ATM
            straddle_theta = (call_theta + put_theta) / 365  # Per day

            return {
                'delta': float(straddle_delta),
                'gamma': float(straddle_gamma),
                'vega': float(straddle_vega),
                'theta': float(straddle_theta)
            }

        except ImportError:
            logger.warning("scipy not available, using simplified Greeks")
            return {
                'delta': 0.0,  # ATM straddle
                'gamma': 0.10,
                'vega': 0.20,
                'theta': -0.02
            }
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    def _check_rehedge_needed(
        self,
        symbol: str,
        greeks: Dict[str, float],
        current_time: datetime
    ) -> SignalType:
        """
        Check if delta rehedging is needed.

        Args:
            symbol: Instrument symbol
            greeks: Current Greeks
            current_time: Current timestamp

        Returns:
            SignalType (LONG/SHORT for rehedge, HOLD if no rehedge needed)
        """
        delta = greeks['delta']
        delta_threshold = self.config['delta_threshold']

        # Check if delta exceeds threshold
        if abs(delta) < delta_threshold:
            return SignalType.HOLD

        # Check time since last hedge
        last_hedge = self.last_hedge_time.get(symbol)
        if last_hedge:
            hours_since_hedge = (current_time - last_hedge).total_seconds() / 3600
            if hours_since_hedge < self.config['rehedge_interval_hours']:
                logger.debug(
                    f"{symbol}: Rehedge needed (delta={delta:.3f}) but too soon "
                    f"({hours_since_hedge:.1f}h < {self.config['rehedge_interval_hours']}h)"
                )
                return SignalType.HOLD

        # Check daily rehedge limit
        today_count = self.rehedge_count.get(symbol, 0)
        if today_count >= self.config['max_rehedge_per_day']:
            logger.warning(
                f"{symbol}: Daily rehedge limit reached ({today_count}/{self.config['max_rehedge_per_day']})"
            )
            return SignalType.HOLD

        # Rehedge needed - direction based on delta sign
        if delta > 0:
            return SignalType.SHORT  # Sell shares to neutralize positive delta
        else:
            return SignalType.LONG  # Buy shares to neutralize negative delta

    def _calculate_hedge_size(
        self,
        signal: StrategySignal,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate hedge size to neutralize delta.

        Args:
            signal: Signal with delta in metadata
            existing_positions: Current positions

        Returns:
            Number of shares to trade for delta neutralization
        """
        delta = signal.metadata.get('delta', 0.0)
        current_position = existing_positions.get(signal.symbol, {})
        current_size = current_position.get('size', 0)

        # Delta of option position
        option_delta = delta * current_size

        # Hedge size is negative of option delta
        hedge_size = abs(option_delta)

        logger.debug(
            f"{signal.symbol}: Hedging delta {option_delta:.2f} with {hedge_size:.0f} shares"
        )

        return hedge_size

    def _get_ml_confidence(self, ml_signals: Optional[pd.DataFrame]) -> float:
        """Get ML confidence score."""
        if ml_signals is None or len(ml_signals) == 0:
            return 0.65

        try:
            return float(ml_signals['confidence'].iloc[-1])
        except Exception:
            return 0.65

    def reset_daily_counters(self):
        """Reset daily rehedge counters. Should be called at start of trading day."""
        self.rehedge_count.clear()
        logger.info("Reset daily rehedge counters")

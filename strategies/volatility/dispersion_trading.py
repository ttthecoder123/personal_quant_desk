"""
Dispersion Trading Strategy

Trade the volatility difference between an index and its components:
- Long index volatility, short component volatilities (or vice versa)
- Based on correlation dynamics
- Profit from correlation changes
- Typically: sell index vol when correlation is high, buy when low
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


class DispersionTradingStrategy(StrategyBase):
    """
    Dispersion trading strategy.

    Core concept:
    - Index vol = weighted avg of component vols * correlation
    - When correlation is high: index vol > weighted component vol → sell index, buy components
    - When correlation is low: index vol < weighted component vol → buy index, sell components
    - Profit from mean reversion in correlation
    """

    def __init__(
        self,
        name: str = "DispersionTrading",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize dispersion trading strategy.

        Args:
            name: Strategy name
            config: Configuration with:
                - index_symbol: Index symbol (e.g., 'SPY')
                - component_symbols: List of component symbols
                - correlation_lookback: Days for correlation calculation (default: 60)
                - correlation_threshold_high: High correlation threshold (default: 0.70)
                - correlation_threshold_low: Low correlation threshold (default: 0.40)
                - dispersion_threshold: Min dispersion to enter (default: 0.03, 3%)
                - rebalance_frequency_days: Days between rebalances (default: 5)
                - num_components: Number of components to trade (default: 10)
            risk_metrics: Risk limits
        """
        default_config = {
            'index_symbol': 'SPY',
            'component_symbols': [],
            'correlation_lookback': 60,
            'correlation_threshold_high': 0.70,
            'correlation_threshold_low': 0.40,
            'dispersion_threshold': 0.03,
            'rebalance_frequency_days': 5,
            'num_components': 10,
            'min_observations': 60,
            'min_confidence': 0.65,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Dispersion tracking
        self.avg_correlation: Optional[float] = None
        self.index_vol: Optional[float] = None
        self.component_vols: Dict[str, float] = {}
        self.dispersion_measure: Optional[float] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_rebalance: Optional[datetime] = None

        logger.info(
            f"Initialized {name} with index={self.config['index_symbol']}, "
            f"components={len(self.config['component_symbols'])}, "
            f"corr_lookback={self.config['correlation_lookback']}d"
        )

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate dispersion trading signal.

        Note: For dispersion, we need data for index AND components.
        This method should be called with index symbol, and will generate
        signals for the full dispersion trade.

        Args:
            symbol: Index symbol
            market_data: Index OHLCV data
            features: Features including component data
            ml_signals: ML predictions

        Returns:
            StrategySignal or None
        """
        try:
            if symbol != self.config['index_symbol']:
                logger.warning(f"Dispersion strategy expects index {self.config['index_symbol']}, got {symbol}")
                return None

            if len(market_data) < self.config['min_observations']:
                logger.warning(f"{symbol}: Insufficient data")
                return None

            # Check if rebalance is needed
            if not self._should_rebalance():
                return None

            # Calculate correlations and dispersion
            correlation_data = self._get_correlation_data(features)
            if correlation_data is None:
                logger.warning("Could not calculate correlation data")
                return None

            avg_corr = correlation_data['avg_correlation']
            dispersion = correlation_data['dispersion']

            self.avg_correlation = avg_corr
            self.dispersion_measure = dispersion

            # Calculate index volatility
            index_vol = self._calculate_volatility(market_data)
            if index_vol is None:
                return None

            self.index_vol = index_vol

            # Determine signal based on correlation regime
            signal_type = self._determine_dispersion_signal(avg_corr, dispersion)

            if signal_type == SignalType.HOLD:
                return None

            # Get ML confidence
            confidence = self._get_ml_confidence(ml_signals)

            timestamp = market_data.index[-1] if isinstance(market_data.index[-1], datetime) else datetime.now()
            current_price = float(market_data['close'].iloc[-1])

            # For dispersion, stop loss based on correlation reversal
            stop_distance = index_vol * current_price * 0.10  # 10% of annual vol
            if signal_type == SignalType.SHORT:  # Selling index vol
                stop_loss = current_price + stop_distance
                take_profit = current_price - stop_distance * 1.5
            else:  # Buying index vol
                stop_loss = current_price - stop_distance
                take_profit = current_price + stop_distance * 1.5

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
                    'avg_correlation': avg_corr,
                    'dispersion': dispersion,
                    'index_vol': index_vol,
                    'component_vols': self.component_vols.copy(),
                    'num_components': len(self.component_vols),
                    'trade_type': 'sell_index' if signal_type == SignalType.SHORT else 'buy_index'
                }
            )

            self.signals_history.append(signal)
            self.last_rebalance = timestamp

            logger.info(
                f"{symbol} dispersion signal: {signal_type.value}, "
                f"corr={avg_corr:.2%}, dispersion={dispersion:.2%}, "
                f"index_vol={index_vol:.2%}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating dispersion signal: {e}", exc_info=True)
            return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_volatility: float,
        existing_positions: Dict[str, Any]
    ) -> float:
        """
        Calculate position size for dispersion trade.

        For dispersion:
        - Size index leg
        - Component legs sized to be beta-neutral to index
        - Total position = index leg + sum(component legs)

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_volatility: Index volatility
            existing_positions: Current positions

        Returns:
            Index leg position size
        """
        try:
            if portfolio_value <= 0:
                return 0.0

            # Risk budget for entire dispersion trade
            risk_budget = self.config.get('risk_budget', 0.05)
            total_position_value = portfolio_value * risk_budget

            # Split between index and components
            # Typically: 50% index, 50% components (distributed)
            index_allocation = 0.50
            component_allocation = 0.50

            # Index position value
            index_position_value = total_position_value * index_allocation

            # Adjust for volatility
            vol_scalar = 0.20 / max(current_volatility, 0.01)
            index_position_value *= vol_scalar

            # Scale by ML confidence
            index_position_value *= signal.confidence

            # Scale by dispersion magnitude
            dispersion = abs(signal.metadata.get('dispersion', 0.03))
            dispersion_scalar = min(dispersion / 0.03, 2.0)  # Cap at 2x
            index_position_value *= dispersion_scalar

            # Convert to shares
            position_size = index_position_value / signal.entry_price

            # Apply portfolio allocation limit
            max_position_value = portfolio_value * self.risk_metrics.max_portfolio_allocation
            max_size = max_position_value / signal.entry_price
            position_size = min(position_size, max_size)

            logger.debug(
                f"{signal.symbol} dispersion size: {position_size:.0f} shares "
                f"(index leg, total value: ${index_position_value:,.0f})"
            )

            # Note: Component legs would be calculated separately
            # based on beta-neutral hedging requirements

            return position_size

        except Exception as e:
            logger.error(f"Error calculating dispersion position size: {e}", exc_info=True)
            return 0.0

    def _calculate_volatility(self, market_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate volatility for a symbol.

        Args:
            market_data: Price data

        Returns:
            Annualized volatility
        """
        try:
            lookback = self.config['correlation_lookback']
            returns = np.log(market_data['close'] / market_data['close'].shift(1))
            returns = returns.dropna().tail(lookback)

            if len(returns) < lookback // 2:
                return None

            volatility = returns.std() * np.sqrt(252)
            return float(volatility)

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return None

    def _get_correlation_data(
        self,
        features: Optional[pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate correlation and dispersion metrics.

        Args:
            features: Features containing component returns/prices

        Returns:
            Dictionary with avg_correlation and dispersion
        """
        try:
            if features is None:
                logger.warning("Features required for dispersion calculation")
                return None

            # Look for correlation data in features
            # Expected columns: 'avg_correlation', 'dispersion'
            # Or individual component returns to calculate

            if 'avg_correlation' in features.columns and 'dispersion' in features.columns:
                avg_corr = features['avg_correlation'].iloc[-1]
                dispersion = features['dispersion'].iloc[-1]

                if pd.notna(avg_corr) and pd.notna(dispersion):
                    return {
                        'avg_correlation': float(avg_corr),
                        'dispersion': float(dispersion)
                    }

            # Calculate from component data if available
            component_symbols = self.config['component_symbols'][:self.config['num_components']]

            if not component_symbols:
                logger.warning("No component symbols configured")
                return None

            # Look for component return columns
            return_cols = [f'{sym}_return' for sym in component_symbols if f'{sym}_return' in features.columns]

            if len(return_cols) < 3:
                logger.warning(f"Insufficient component returns found ({len(return_cols)})")
                return None

            # Calculate correlation matrix
            lookback = self.config['correlation_lookback']
            component_returns = features[return_cols].tail(lookback)

            if len(component_returns) < lookback // 2:
                return None

            # Correlation matrix
            corr_matrix = component_returns.corr()
            self.correlation_matrix = corr_matrix

            # Average pairwise correlation
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            avg_corr = corr_matrix.where(mask).stack().mean()

            # Calculate component volatilities
            component_vols = {}
            for col in return_cols:
                symbol = col.replace('_return', '')
                vol = component_returns[col].std() * np.sqrt(252)
                component_vols[symbol] = float(vol)

            self.component_vols = component_vols

            # Dispersion = index_vol - weighted_avg_component_vol
            # Simplified: use average component vol
            avg_component_vol = np.mean(list(component_vols.values()))

            # Estimate index vol from components and correlation
            # index_var ≈ avg_component_var + 2 * avg_corr * avg_component_var
            estimated_index_vol = np.sqrt(avg_component_vol**2 * (1 + 2 * avg_corr * (len(component_vols) - 1) / len(component_vols)))

            # Dispersion measure
            dispersion = (estimated_index_vol - avg_component_vol) / avg_component_vol

            return {
                'avg_correlation': float(avg_corr),
                'dispersion': float(dispersion)
            }

        except Exception as e:
            logger.error(f"Error calculating correlation data: {e}", exc_info=True)
            return None

    def _determine_dispersion_signal(
        self,
        avg_correlation: float,
        dispersion: float
    ) -> SignalType:
        """
        Determine signal based on correlation and dispersion.

        Strategy logic:
        - High correlation (> threshold): Sell index vol, buy component vol
        - Low correlation (< threshold): Buy index vol, sell component vol
        - Require minimum dispersion to trade

        Args:
            avg_correlation: Average pairwise correlation
            dispersion: Dispersion measure

        Returns:
            SignalType
        """
        corr_high = self.config['correlation_threshold_high']
        corr_low = self.config['correlation_threshold_low']
        min_dispersion = self.config['dispersion_threshold']

        # Check minimum dispersion
        if abs(dispersion) < min_dispersion:
            logger.debug(f"Dispersion {dispersion:.2%} below threshold {min_dispersion:.2%}")
            return SignalType.HOLD

        # High correlation regime: sell index vol (short)
        if avg_correlation > corr_high:
            logger.debug(f"High correlation {avg_correlation:.2%} → sell index vol")
            return SignalType.SHORT

        # Low correlation regime: buy index vol (long)
        if avg_correlation < corr_low:
            logger.debug(f"Low correlation {avg_correlation:.2%} → buy index vol")
            return SignalType.LONG

        # Mid-range correlation: no trade
        return SignalType.HOLD

    def _should_rebalance(self) -> bool:
        """
        Check if rebalancing is needed.

        Returns:
            True if rebalance needed
        """
        if self.last_rebalance is None:
            return True

        # Check if enough time has passed
        now = datetime.now()
        days_since_rebalance = (now - self.last_rebalance).days

        if days_since_rebalance >= self.config['rebalance_frequency_days']:
            logger.info(f"Rebalance due: {days_since_rebalance} days since last rebalance")
            return True

        return False

    def _get_ml_confidence(self, ml_signals: Optional[pd.DataFrame]) -> float:
        """Get ML confidence score."""
        if ml_signals is None or len(ml_signals) == 0:
            return 0.70  # Slightly higher default for dispersion

        try:
            return float(ml_signals['confidence'].iloc[-1])
        except Exception:
            return 0.70

    def get_component_positions(
        self,
        index_position_size: float,
        index_price: float
    ) -> Dict[str, float]:
        """
        Calculate component position sizes for beta-neutral hedge.

        Args:
            index_position_size: Index position size
            index_price: Index price

        Returns:
            Dictionary mapping component symbols to position sizes
        """
        try:
            component_positions = {}
            num_components = len(self.component_vols)

            if num_components == 0:
                return component_positions

            # Total component value should match index value
            index_value = index_position_size * index_price
            component_value_each = index_value / num_components

            # For simplicity, equal-weight components
            # In practice, would use beta-weighted
            for symbol, vol in self.component_vols.items():
                # Placeholder price - in practice, would use actual component prices
                component_price = 100.0  # Assume $100 per share
                position_size = component_value_each / component_price
                component_positions[symbol] = position_size

            logger.debug(
                f"Component positions: {len(component_positions)} symbols, "
                f"~{component_value_each:,.0f} each"
            )

            return component_positions

        except Exception as e:
            logger.error(f"Error calculating component positions: {e}")
            return {}

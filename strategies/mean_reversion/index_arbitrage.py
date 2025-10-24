"""
Index Arbitrage Strategy

Implements index arbitrage by tracking baskets of components vs index ETF.
Identifies mispricings between index and constituent stocks for arbitrage opportunities.
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


class IndexArbitrageStrategy(StrategyBase):
    """
    Index arbitrage strategy for exploiting basis spread.

    Methodology:
    1. Track basket of index constituents with proper weights
    2. Calculate theoretical index value from basket
    3. Compute spread between actual index and theoretical value
    4. Generate arbitrage signals when spread exceeds threshold
    5. Integrate with Step 4 meta-labels for bet sizing

    Common arbitrage types:
    - Cash-futures basis arbitrage
    - Index ETF vs constituents
    - Sector ETF vs components

    References:
        Chan, E. (2009). Quantitative Trading
        Harris, L. (2003). Trading and Exchanges, Chapter 23
    """

    def __init__(
        self,
        name: str = "IndexArbitrage",
        config: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[RiskMetrics] = None
    ):
        """
        Initialize index arbitrage strategy.

        Args:
            name: Strategy name
            config: Configuration dict with parameters:
                - entry_threshold: Spread threshold for entry in bps (default: 10)
                - exit_threshold: Spread threshold for exit in bps (default: 2)
                - basket_weights: Dict of {symbol: weight} for basket constituents
                - rebalance_frequency: Days between weight updates (default: 30)
                - transaction_cost_bps: Transaction costs in basis points (default: 5)
                - min_confidence: Minimum meta-label confidence (default: 0.55)
                - execution_window: Max seconds for execution (default: 30)
                - slippage_bps: Expected slippage in basis points (default: 2)
            risk_metrics: Risk limits and targets
        """
        default_config = {
            'entry_threshold': 10.0,  # 10 bps
            'exit_threshold': 2.0,    # 2 bps
            'basket_weights': {},
            'rebalance_frequency': 30,
            'transaction_cost_bps': 5.0,
            'min_confidence': 0.55,
            'execution_window': 30,
            'slippage_bps': 2.0,
            'enabled': True
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, risk_metrics)

        # Validate basket weights
        if not self.config.get('basket_weights'):
            logger.warning("No basket weights provided - strategy may not function correctly")

        # Store spread history for analysis
        self.spread_history: Dict[str, List[float]] = {}
        self.basket_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized {name} with entry_threshold={self.config['entry_threshold']} bps, "
                   f"basket_size={len(self.config.get('basket_weights', {}))}")

    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize basket weights to sum to 1.0.

        Args:
            weights: Dictionary of {symbol: weight}

        Returns:
            Normalized weights dictionary
        """
        total_weight = sum(weights.values())

        if total_weight <= 0:
            logger.error("Total weight is zero or negative")
            return weights

        normalized = {symbol: weight / total_weight for symbol, weight in weights.items()}

        return normalized

    def calculate_basket_value(
        self,
        basket_prices: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate theoretical value of basket.

        Args:
            basket_prices: Dictionary of {symbol: current_price}
            weights: Dictionary of {symbol: weight} (should sum to 1.0)

        Returns:
            Basket value
        """
        basket_value = 0.0

        for symbol, weight in weights.items():
            if symbol in basket_prices:
                basket_value += basket_prices[symbol] * weight
            else:
                logger.warning(f"Missing price for basket component: {symbol}")

        return basket_value

    def calculate_spread(
        self,
        index_price: float,
        basket_value: float
    ) -> float:
        """
        Calculate spread between index and basket in basis points.

        Spread (bps) = ((Index - Basket) / Basket) * 10000

        Args:
            index_price: Current index price
            basket_value: Current basket theoretical value

        Returns:
            Spread in basis points
        """
        if basket_value <= 0:
            logger.warning("Basket value is zero or negative")
            return 0.0

        spread_bps = ((index_price - basket_value) / basket_value) * 10000

        return spread_bps

    def calculate_tracking_error(
        self,
        index_returns: pd.Series,
        basket_returns: pd.Series,
        window: int = 30
    ) -> float:
        """
        Calculate tracking error between index and basket.

        Tracking error = std(index_returns - basket_returns)

        Args:
            index_returns: Index return series
            basket_returns: Basket return series
            window: Window for calculation

        Returns:
            Annualized tracking error
        """
        try:
            # Align series
            aligned = pd.DataFrame({
                'index': index_returns,
                'basket': basket_returns
            }).dropna()

            if len(aligned) < window:
                return 0.0

            # Calculate return differences
            return_diff = aligned['index'] - aligned['basket']

            # Calculate tracking error (annualized)
            tracking_error = return_diff.iloc[-window:].std() * np.sqrt(252)

            return tracking_error

        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.0

    def estimate_execution_cost(
        self,
        basket_value: float,
        num_components: int,
        transaction_cost_bps: float,
        slippage_bps: float
    ) -> float:
        """
        Estimate total execution cost for arbitrage trade.

        Total cost = (transaction_cost + slippage) * num_legs
        where num_legs = 1 (index) + num_components

        Args:
            basket_value: Total basket value
            num_components: Number of basket components
            transaction_cost_bps: Transaction costs in bps
            slippage_bps: Expected slippage in bps

        Returns:
            Total execution cost in dollars
        """
        num_legs = 1 + num_components  # Index + all components
        total_cost_bps = (transaction_cost_bps + slippage_bps) * num_legs

        execution_cost = (basket_value * total_cost_bps) / 10000

        return execution_cost

    def calculate_arbitrage_profit(
        self,
        spread_bps: float,
        basket_value: float,
        execution_cost: float
    ) -> float:
        """
        Calculate expected arbitrage profit after costs.

        Args:
            spread_bps: Current spread in basis points
            basket_value: Basket value
            execution_cost: Estimated execution cost

        Returns:
            Expected profit in dollars
        """
        gross_profit = (abs(spread_bps) / 10000) * basket_value
        net_profit = gross_profit - execution_cost

        return net_profit

    def check_arbitrage_feasibility(
        self,
        spread_bps: float,
        basket_value: float,
        num_components: int
    ) -> Tuple[bool, float, str]:
        """
        Check if arbitrage opportunity is feasible after costs.

        Args:
            spread_bps: Current spread in basis points
            basket_value: Basket value
            num_components: Number of basket components

        Returns:
            Tuple of (is_feasible, expected_profit, rejection_reason)
        """
        # Calculate execution cost
        execution_cost = self.estimate_execution_cost(
            basket_value,
            num_components,
            self.config['transaction_cost_bps'],
            self.config['slippage_bps']
        )

        # Calculate expected profit
        expected_profit = self.calculate_arbitrage_profit(
            spread_bps,
            basket_value,
            execution_cost
        )

        # Check if profitable
        if expected_profit <= 0:
            return False, expected_profit, f"Insufficient profit: ${expected_profit:.2f}"

        # Check if spread exceeds cost threshold
        total_cost_bps = ((execution_cost / basket_value) * 10000)
        if abs(spread_bps) < total_cost_bps:
            return False, expected_profit, f"Spread {abs(spread_bps):.1f} bps < costs {total_cost_bps:.1f} bps"

        return True, expected_profit, ""

    def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        ml_signals: Optional[pd.DataFrame] = None
    ) -> Optional[StrategySignal]:
        """
        Generate index arbitrage signal.

        Args:
            symbol: Index symbol
            market_data: DataFrame with index and basket component prices
                Expected columns: 'index_price' and one column per basket component
            features: Engineered features (optional)
            ml_signals: ML signals with meta-labels for confidence

        Returns:
            StrategySignal or None
        """
        try:
            if market_data is None or len(market_data) < 10:
                logger.debug(f"Insufficient data for {symbol}")
                return None

            # Get basket weights
            basket_weights = self.config.get('basket_weights', {})
            if not basket_weights:
                logger.warning(f"No basket weights configured for {symbol}")
                return None

            # Normalize weights
            basket_weights = self.normalize_weights(basket_weights)

            # Extract current prices
            if 'index_price' not in market_data.columns:
                logger.warning(f"Missing 'index_price' column for {symbol}")
                return None

            current_index_price = market_data['index_price'].iloc[-1]

            # Get basket component prices
            basket_prices = {}
            missing_components = []

            for component in basket_weights.keys():
                if component in market_data.columns:
                    basket_prices[component] = market_data[component].iloc[-1]
                else:
                    missing_components.append(component)

            if missing_components:
                logger.warning(f"Missing prices for components: {missing_components}")
                # Can still proceed if we have most components
                if len(missing_components) / len(basket_weights) > 0.2:  # >20% missing
                    return None

            # Calculate theoretical basket value
            basket_value = self.calculate_basket_value(basket_prices, basket_weights)

            if basket_value <= 0:
                logger.error(f"Invalid basket value: {basket_value}")
                return None

            # Calculate spread
            spread_bps = self.calculate_spread(current_index_price, basket_value)

            # Store spread history
            if symbol not in self.spread_history:
                self.spread_history[symbol] = []
            self.spread_history[symbol].append(spread_bps)

            # Keep only recent history
            if len(self.spread_history[symbol]) > 100:
                self.spread_history[symbol] = self.spread_history[symbol][-100:]

            # Check arbitrage feasibility
            is_feasible, expected_profit, rejection_reason = self.check_arbitrage_feasibility(
                spread_bps,
                basket_value,
                len(basket_prices)
            )

            if not is_feasible:
                logger.debug(f"Arbitrage not feasible for {symbol}: {rejection_reason}")
                return None

            # Generate signals based on spread
            signal_type = None
            confidence = 0.5

            entry_threshold = self.config['entry_threshold']
            exit_threshold = self.config['exit_threshold']

            # Check existing position
            existing_position = self.positions.get(symbol)

            if existing_position is None:
                # Entry signals
                if spread_bps > entry_threshold:
                    # Index overpriced relative to basket
                    # Short index, long basket
                    signal_type = SignalType.SHORT
                    # Confidence based on spread magnitude and expected profit
                    confidence = min(0.5 + (spread_bps / entry_threshold - 1) * 0.2, 0.95)
                    confidence *= min(expected_profit / (basket_value * 0.001), 1.0)  # Cap at 0.1%

                elif spread_bps < -entry_threshold:
                    # Index underpriced relative to basket
                    # Long index, short basket
                    signal_type = SignalType.LONG
                    confidence = min(0.5 + (abs(spread_bps) / entry_threshold - 1) * 0.2, 0.95)
                    confidence *= min(expected_profit / (basket_value * 0.001), 1.0)

            else:
                # Exit signals - spread converges
                position_side = existing_position['side']

                if position_side == PositionSide.LONG and spread_bps > -exit_threshold:
                    # Spread converged, exit long
                    signal_type = SignalType.EXIT_LONG
                    confidence = 0.8

                elif position_side == PositionSide.SHORT and spread_bps < exit_threshold:
                    # Spread converged, exit short
                    signal_type = SignalType.EXIT_SHORT
                    confidence = 0.8

            if signal_type is None:
                return None

            # Calculate tracking error for risk assessment
            if len(market_data) >= 30:
                index_returns = market_data['index_price'].pct_change()

                # Calculate basket returns
                basket_series = pd.Series(index=market_data.index, dtype=float)
                for i in range(len(market_data)):
                    row_prices = {}
                    for component in basket_weights.keys():
                        if component in market_data.columns:
                            row_prices[component] = market_data[component].iloc[i]
                    basket_series.iloc[i] = self.calculate_basket_value(row_prices, basket_weights)

                basket_returns = basket_series.pct_change()
                tracking_error = self.calculate_tracking_error(index_returns, basket_returns)
            else:
                tracking_error = 0.0

            # Integrate meta-labels from Step 4 if available
            if ml_signals is not None and len(ml_signals) > 0:
                latest_ml = ml_signals.iloc[-1]
                if 'meta_label_prob' in latest_ml:
                    ml_confidence = latest_ml['meta_label_prob']
                    # Blend strategy confidence with ML confidence
                    confidence = 0.7 * confidence + 0.3 * ml_confidence

            # Calculate stop loss and take profit based on spread
            spread_std = np.std(self.spread_history[symbol]) if len(self.spread_history[symbol]) > 10 else 5.0

            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                if signal_type == SignalType.LONG:
                    # Stop if spread widens further (more negative)
                    stop_loss = current_index_price * (1 - (entry_threshold + 2 * spread_std) / 10000)
                    # Take profit when spread closes
                    take_profit = basket_value
                else:  # SHORT
                    # Stop if spread widens further (more positive)
                    stop_loss = current_index_price * (1 + (entry_threshold + 2 * spread_std) / 10000)
                    # Take profit when spread closes
                    take_profit = basket_value
            else:
                stop_loss = None
                take_profit = None

            signal = StrategySignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                size=1.0,  # Will be adjusted by position sizing
                entry_price=current_index_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'spread_bps': spread_bps,
                    'basket_value': basket_value,
                    'index_price': current_index_price,
                    'expected_profit': expected_profit,
                    'tracking_error': tracking_error,
                    'num_components': len(basket_prices),
                    'basket_weights': basket_weights,
                    'basket_prices': basket_prices,
                    'spread_std': spread_std
                }
            )

            self.signals_history.append(signal)
            logger.info(f"Generated {signal_type.value} signal for {symbol}: "
                       f"spread={spread_bps:.2f} bps, expected_profit=${expected_profit:.2f}, "
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
        Calculate position size for index arbitrage.

        Position sizing considers:
        1. Expected profit magnitude
        2. Tracking error (lower = more size)
        3. Number of components (fewer = more size)
        4. Portfolio allocation limits

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Current volatility
            existing_positions: Current positions

        Returns:
            Position size in dollars
        """
        try:
            metadata = signal.metadata or {}
            expected_profit = metadata.get('expected_profit', 0.0)
            tracking_error = metadata.get('tracking_error', 0.05)
            num_components = metadata.get('num_components', 10)

            # Base size from expected profit (scale with confidence in profit)
            profit_margin = expected_profit / (signal.entry_price * 100)  # Profit per $100
            base_size = portfolio_value * min(profit_margin * 10, 0.25)  # Max 25% from profit signal

            # Adjust for tracking error (lower tracking error = more reliable = larger size)
            tracking_factor = 1.0 / (1.0 + tracking_error * 10)  # Normalize
            adjusted_size = base_size * tracking_factor

            # Adjust for execution complexity (fewer components = easier execution)
            complexity_factor = np.clip(20 / num_components, 0.5, 2.0)
            adjusted_size *= complexity_factor

            # Scale by confidence
            adjusted_size *= signal.confidence

            # Apply maximum position limits
            max_position = portfolio_value * self.risk_metrics.max_portfolio_allocation
            adjusted_size = min(adjusted_size, max_position)

            # Ensure minimum viable size
            min_size = portfolio_value * 0.02  # At least 2% for arbitrage
            adjusted_size = max(adjusted_size, min_size)

            logger.debug(f"Position size for {signal.symbol}: ${adjusted_size:,.0f} "
                        f"(profit=${expected_profit:.2f}, tracking_err={tracking_error:.2%}, "
                        f"components={num_components})")

            return adjusted_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return portfolio_value * 0.05

    def get_basket_weights(self, symbol: str) -> Dict[str, float]:
        """
        Get basket weights for an index.

        Args:
            symbol: Index symbol

        Returns:
            Dictionary of basket weights
        """
        return self.config.get('basket_weights', {})

    def update_basket_weights(self, symbol: str, weights: Dict[str, float]):
        """
        Update basket weights for an index.

        Args:
            symbol: Index symbol
            weights: New basket weights
        """
        self.config['basket_weights'] = self.normalize_weights(weights)
        logger.info(f"Updated basket weights for {symbol}: {len(weights)} components")

    def get_spread_history(self, symbol: str) -> List[float]:
        """
        Get spread history for analysis.

        Args:
            symbol: Index symbol

        Returns:
            List of historical spreads
        """
        return self.spread_history.get(symbol, [])

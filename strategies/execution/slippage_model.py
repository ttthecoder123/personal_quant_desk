"""
Slippage modeling for realistic backtesting and execution cost estimation.

Models various sources of slippage:
- Bid-ask spread
- Market impact (linear and square-root)
- Adverse selection
- Volatility-based slippage
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger


class SlippageModel:
    """
    Comprehensive slippage model for trading cost estimation.

    Models multiple components of slippage:
    - Bid-ask spread (immediate cost)
    - Market impact (temporary and permanent)
    - Adverse selection (informed trading costs)
    - Volatility-based slippage

    Attributes:
        spread_bps (float): Typical bid-ask spread in basis points
        linear_impact_coeff (float): Linear market impact coefficient
        sqrt_impact_coeff (float): Square-root market impact coefficient
        adverse_selection_factor (float): Adverse selection multiplier
        volatility_factor (float): Volatility-based slippage factor
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        linear_impact_coeff: float = 0.1,
        sqrt_impact_coeff: float = 0.5,
        adverse_selection_factor: float = 0.3,
        volatility_factor: float = 0.2,
    ):
        """
        Initialize slippage model.

        Args:
            spread_bps: Typical bid-ask spread in basis points
            linear_impact_coeff: Coefficient for linear market impact
            sqrt_impact_coeff: Coefficient for square-root market impact
            adverse_selection_factor: Adverse selection multiplier
            volatility_factor: Volatility-based slippage factor
        """
        self.spread_bps = spread_bps
        self.linear_impact_coeff = linear_impact_coeff
        self.sqrt_impact_coeff = sqrt_impact_coeff
        self.adverse_selection_factor = adverse_selection_factor
        self.volatility_factor = volatility_factor

        logger.info(
            f"SlippageModel initialized: spread={spread_bps}bps, "
            f"linear_impact={linear_impact_coeff}, sqrt_impact={sqrt_impact_coeff}"
        )

    def estimate_bid_ask_spread(
        self,
        price: float,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> float:
        """
        Estimate bid-ask spread.

        Spread typically increases with volatility and decreases with volume.

        Args:
            price: Current price
            volatility: Asset volatility (annualized)
            volume: Trading volume

        Returns:
            Estimated spread in dollars
        """
        # Base spread
        spread_pct = self.spread_bps / 10000

        # Adjust for volatility
        if volatility is not None and volatility > 0:
            # Higher volatility → wider spread
            vol_multiplier = 1.0 + (volatility - 0.20) * 0.5  # Normalized around 20% vol
            spread_pct *= max(0.5, vol_multiplier)

        # Adjust for volume (liquidity)
        if volume is not None and volume > 0:
            # Higher volume → tighter spread
            # Assuming average volume of 1M, adjust accordingly
            volume_multiplier = np.sqrt(1000000 / max(volume, 1))
            spread_pct *= np.clip(volume_multiplier, 0.5, 2.0)

        spread = price * spread_pct

        logger.debug(
            f"Estimated spread: ${spread:.4f} ({spread_pct * 10000:.2f}bps) "
            f"at price=${price:.2f}"
        )

        return spread

    def linear_market_impact(
        self,
        quantity: float,
        daily_volume: float,
        price: float,
    ) -> float:
        """
        Calculate linear market impact.

        Impact increases linearly with order size relative to daily volume.
        Suitable for small to medium orders.

        Formula: impact = coefficient * (quantity / daily_volume) * price

        Args:
            quantity: Order quantity
            daily_volume: Average daily trading volume
            price: Current price

        Returns:
            Market impact in dollars per share
        """
        if daily_volume <= 0:
            logger.warning("Invalid daily volume, using default impact")
            return price * 0.001  # 10 bps default

        # Participation rate
        participation = quantity / daily_volume

        # Linear impact
        impact_pct = self.linear_impact_coeff * participation
        impact = price * impact_pct

        logger.debug(
            f"Linear impact: ${impact:.4f} ({impact_pct * 10000:.2f}bps) "
            f"for {participation:.2%} participation"
        )

        return impact

    def sqrt_market_impact(
        self,
        quantity: float,
        daily_volume: float,
        price: float,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate square-root market impact.

        Impact increases with square root of order size. More realistic
        for large orders where impact grows sub-linearly.

        Formula: impact = coefficient * sqrt(quantity / daily_volume) * price * volatility

        Args:
            quantity: Order quantity
            daily_volume: Average daily trading volume
            price: Current price
            volatility: Asset volatility (annualized)

        Returns:
            Market impact in dollars per share
        """
        if daily_volume <= 0:
            logger.warning("Invalid daily volume")
            return price * 0.001

        participation = quantity / daily_volume

        # Square-root impact
        impact_pct = self.sqrt_impact_coeff * np.sqrt(participation)

        # Scale by volatility if provided
        if volatility is not None and volatility > 0:
            # Normalize around 20% volatility
            vol_scalar = volatility / 0.20
            impact_pct *= vol_scalar

        impact = price * impact_pct / 100  # Coefficient is in percentage points

        logger.debug(
            f"Sqrt impact: ${impact:.4f} ({impact / price * 10000:.2f}bps) "
            f"for {participation:.2%} participation"
        )

        return impact

    def adverse_selection_cost(
        self,
        quantity: float,
        is_aggressive: bool,
        spread: float,
    ) -> float:
        """
        Calculate adverse selection cost.

        Cost of trading against informed traders. Higher for aggressive
        orders that demand immediate liquidity.

        Args:
            quantity: Order quantity
            is_aggressive: Whether order is aggressive (market/aggressive limit)
            spread: Current bid-ask spread

        Returns:
            Adverse selection cost per share
        """
        if not is_aggressive:
            # Passive orders have lower adverse selection
            return 0.0

        # Adverse selection as fraction of spread
        adverse_cost = spread * self.adverse_selection_factor

        logger.debug(
            f"Adverse selection cost: ${adverse_cost:.4f} "
            f"({self.adverse_selection_factor:.1%} of spread)"
        )

        return adverse_cost

    def volatility_based_slippage(
        self,
        price: float,
        volatility: float,
        time_horizon_minutes: float = 5.0,
    ) -> float:
        """
        Calculate volatility-based slippage.

        Price may move unfavorably during execution due to normal volatility.

        Args:
            price: Current price
            volatility: Asset volatility (annualized)
            time_horizon_minutes: Execution time horizon in minutes

        Returns:
            Expected slippage from volatility
        """
        if volatility <= 0:
            return 0.0

        # Convert annual volatility to minutes
        # Annual vol → daily vol: vol / sqrt(252)
        # Daily vol → minute vol: daily_vol / sqrt(390) [6.5 hour trading day]
        minute_vol = volatility / np.sqrt(252 * 390)

        # Expected price movement over time horizon
        # Using sqrt(time) scaling
        expected_move = minute_vol * np.sqrt(time_horizon_minutes)

        # Slippage is a fraction of expected move
        slippage = price * expected_move * self.volatility_factor

        logger.debug(
            f"Volatility slippage: ${slippage:.4f} ({slippage / price * 10000:.2f}bps) "
            f"for {time_horizon_minutes:.1f}min horizon, vol={volatility:.2%}"
        )

        return slippage

    def total_slippage(
        self,
        quantity: float,
        price: float,
        daily_volume: float,
        volatility: float,
        is_aggressive: bool = True,
        execution_time_minutes: float = 5.0,
        side: str = 'BUY',
    ) -> Dict[str, float]:
        """
        Calculate total slippage from all components.

        Args:
            quantity: Order quantity
            price: Current price
            daily_volume: Average daily volume
            volatility: Asset volatility (annualized)
            is_aggressive: Whether order is aggressive
            execution_time_minutes: Time to execute order
            side: Order side ('BUY' or 'SELL')

        Returns:
            Dictionary with slippage components and total
        """
        # Bid-ask spread
        spread = self.estimate_bid_ask_spread(price, volatility, daily_volume)
        spread_cost = spread / 2  # Pay half spread on average

        # Market impact (use square-root for realism)
        impact = self.sqrt_market_impact(quantity, daily_volume, price, volatility)

        # Adverse selection
        adverse_selection = self.adverse_selection_cost(quantity, is_aggressive, spread)

        # Volatility slippage
        vol_slippage = self.volatility_based_slippage(price, volatility, execution_time_minutes)

        # Total slippage
        total = spread_cost + impact + adverse_selection + vol_slippage

        # Adjust sign for side
        if side.upper() == 'SELL':
            total = -total

        slippage_bps = (total / price) * 10000

        components = {
            'spread_cost': spread_cost,
            'market_impact': impact,
            'adverse_selection': adverse_selection,
            'volatility_slippage': vol_slippage,
            'total_slippage': total,
            'slippage_bps': slippage_bps,
            'effective_price': price + total if side.upper() == 'BUY' else price - total,
        }

        logger.info(
            f"Total slippage: {slippage_bps:.2f}bps (${total:.4f}) for "
            f"{quantity} shares @ ${price:.2f}"
        )

        return components

    def calibrate_from_trades(
        self,
        trades: pd.DataFrame,
        adjust_model: bool = True,
    ) -> Dict[str, float]:
        """
        Calibrate slippage model from historical trades.

        Args:
            trades: DataFrame with columns: quantity, decision_price, execution_price,
                    daily_volume, volatility, side
            adjust_model: Whether to update model parameters

        Returns:
            Dictionary with calibrated parameters
        """
        if trades.empty:
            logger.warning("No trades for calibration")
            return {}

        # Calculate realized slippage
        trades = trades.copy()
        trades['slippage'] = np.where(
            trades['side'].str.upper() == 'BUY',
            trades['execution_price'] - trades['decision_price'],
            trades['decision_price'] - trades['execution_price']
        )
        trades['slippage_bps'] = (trades['slippage'] / trades['decision_price']) * 10000

        # Calculate participation rate
        trades['participation'] = trades['quantity'] / trades['daily_volume']

        # Fit linear impact coefficient
        # slippage ≈ linear_coeff * participation * price
        trades['predicted_linear'] = (
            self.linear_impact_coeff * trades['participation'] * trades['decision_price']
        )

        # Fit square-root impact coefficient
        trades['predicted_sqrt'] = (
            self.sqrt_impact_coeff * np.sqrt(trades['participation']) *
            trades['decision_price'] * trades['volatility'] / 0.20
        ) / 100

        # Calculate model errors
        linear_error = (trades['slippage'] - trades['predicted_linear']).abs().mean()
        sqrt_error = (trades['slippage'] - trades['predicted_sqrt']).abs().mean()

        stats = {
            'mean_slippage_bps': trades['slippage_bps'].mean(),
            'median_slippage_bps': trades['slippage_bps'].median(),
            'std_slippage_bps': trades['slippage_bps'].std(),
            'linear_model_error': linear_error,
            'sqrt_model_error': sqrt_error,
            'n_trades': len(trades),
        }

        # Simple recalibration if requested
        if adjust_model and sqrt_error > 0:
            # Adjust sqrt coefficient to minimize error
            actual_impact = trades['slippage'].mean()
            predicted_impact = trades['predicted_sqrt'].mean()

            if predicted_impact > 0:
                adjustment_factor = actual_impact / predicted_impact
                self.sqrt_impact_coeff *= np.clip(adjustment_factor, 0.5, 2.0)

                logger.info(
                    f"Model recalibrated: sqrt_impact_coeff={self.sqrt_impact_coeff:.3f} "
                    f"(adjustment={adjustment_factor:.2f}x)"
                )

        logger.info(
            f"Slippage calibration: mean={stats['mean_slippage_bps']:.2f}bps, "
            f"median={stats['median_slippage_bps']:.2f}bps, "
            f"std={stats['std_slippage_bps']:.2f}bps ({stats['n_trades']} trades)"
        )

        return stats

    def simulate_fill_price(
        self,
        quantity: float,
        decision_price: float,
        daily_volume: float,
        volatility: float,
        is_aggressive: bool = True,
        side: str = 'BUY',
        random_seed: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Simulate realistic fill price with slippage.

        Adds randomness to model execution uncertainty.

        Args:
            quantity: Order quantity
            decision_price: Price when order was decided
            daily_volume: Average daily volume
            volatility: Asset volatility
            is_aggressive: Aggressive execution
            side: Order side
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (fill_price, slippage_components)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate expected slippage
        slippage_components = self.total_slippage(
            quantity=quantity,
            price=decision_price,
            daily_volume=daily_volume,
            volatility=volatility,
            is_aggressive=is_aggressive,
            side=side,
        )

        expected_slippage = slippage_components['total_slippage']

        # Add random component (slippage uncertainty)
        # Assume slippage has 50% uncertainty
        random_component = np.random.normal(0, abs(expected_slippage) * 0.5)

        total_slippage = expected_slippage + random_component

        # Calculate fill price
        if side.upper() == 'BUY':
            fill_price = decision_price + total_slippage
        else:
            fill_price = decision_price - total_slippage

        # Update components with actual values
        slippage_components['total_slippage'] = total_slippage
        slippage_components['random_component'] = random_component
        slippage_components['fill_price'] = fill_price

        logger.debug(
            f"Simulated fill: decision=${decision_price:.2f}, "
            f"fill=${fill_price:.2f}, "
            f"slippage=${total_slippage:.4f}"
        )

        return fill_price, slippage_components

    def estimate_total_cost(
        self,
        quantity: float,
        price: float,
        daily_volume: float,
        volatility: float,
        side: str = 'BUY',
    ) -> float:
        """
        Estimate total transaction cost including slippage.

        Args:
            quantity: Order quantity
            price: Current price
            daily_volume: Average daily volume
            volatility: Asset volatility
            side: Order side

        Returns:
            Total transaction cost in dollars
        """
        slippage = self.total_slippage(
            quantity=quantity,
            price=price,
            daily_volume=daily_volume,
            volatility=volatility,
            side=side,
        )

        # Total cost = slippage per share * quantity
        total_cost = abs(slippage['total_slippage']) * quantity

        logger.info(
            f"Estimated total transaction cost: ${total_cost:.2f} "
            f"({slippage['slippage_bps']:.2f}bps) for {quantity} shares"
        )

        return total_cost

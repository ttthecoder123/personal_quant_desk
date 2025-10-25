"""
Portfolio rebalancing with transaction cost optimization.

Implements intelligent rebalancing strategies that balance the benefits
of maintaining target allocations against the costs of trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class Rebalancer:
    """
    Portfolio rebalancing manager.

    Features:
    - Threshold-based rebalancing triggers
    - Transaction cost optimization
    - Portfolio drift monitoring
    - Emergency rebalancing on risk breach
    - Time-based and drift-based triggers

    Attributes:
        drift_threshold (float): Maximum allowed drift from target weights
        min_rebalance_interval (int): Minimum days between rebalances
        max_rebalance_interval (int): Maximum days before forced rebalance
        transaction_cost_bps (float): Transaction costs in basis points
        min_trade_size (float): Minimum trade size (fraction of portfolio)
        risk_breach_threshold (float): Risk level triggering emergency rebalance
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        min_rebalance_interval: int = 5,
        max_rebalance_interval: int = 60,
        transaction_cost_bps: float = 10.0,
        min_trade_size: float = 0.01,
        risk_breach_threshold: float = 1.5,
    ):
        """
        Initialize rebalancer.

        Args:
            drift_threshold: Maximum weight drift (e.g., 0.05 = 5%)
            min_rebalance_interval: Minimum days between rebalances
            max_rebalance_interval: Maximum days before forced rebalance
            transaction_cost_bps: Transaction costs in basis points
            min_trade_size: Minimum trade size as fraction of portfolio
            risk_breach_threshold: Risk multiplier triggering emergency rebalance
        """
        self.drift_threshold = drift_threshold
        self.min_rebalance_interval = min_rebalance_interval
        self.max_rebalance_interval = max_rebalance_interval
        self.transaction_cost_bps = transaction_cost_bps
        self.min_trade_size = min_trade_size
        self.risk_breach_threshold = risk_breach_threshold

        self.last_rebalance_date = None
        self.rebalance_history = []

        logger.info(
            f"Rebalancer initialized: drift_threshold={drift_threshold}, "
            f"min_interval={min_rebalance_interval}d, max_interval={max_rebalance_interval}d"
        )

    def calculate_drift(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> pd.Series:
        """
        Calculate portfolio drift from target weights.

        Drift = |current_weight - target_weight|

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            Drift for each asset
        """
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current_weights = current_weights.reindex(all_assets, fill_value=0)
        target_weights = target_weights.reindex(all_assets, fill_value=0)

        # Calculate drift
        drift = (current_weights - target_weights).abs()

        logger.debug(
            f"Portfolio drift: max={drift.max():.3f}, mean={drift.mean():.3f}"
        )

        return drift

    def calculate_total_drift(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> float:
        """
        Calculate total portfolio drift.

        Total drift = sum of absolute weight differences / 2

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            Total drift as a single number
        """
        drift = self.calculate_drift(current_weights, target_weights)
        total_drift = drift.sum() / 2  # Divide by 2 to avoid double-counting

        return total_drift

    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        current_date: Optional[datetime] = None,
        current_risk: Optional[float] = None,
        target_risk: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if portfolio should be rebalanced.

        Checks multiple conditions:
        1. Drift threshold breach
        2. Minimum interval elapsed
        3. Maximum interval exceeded
        4. Risk breach

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_date: Current date
            current_risk: Current portfolio risk level
            target_risk: Target portfolio risk level

        Returns:
            Tuple of (should_rebalance, reason)
        """
        if current_date is None:
            current_date = datetime.now()

        # Check minimum interval
        if self.last_rebalance_date is not None:
            days_since_rebalance = (current_date - self.last_rebalance_date).days

            if days_since_rebalance < self.min_rebalance_interval:
                return False, f"Too soon (last rebalance {days_since_rebalance}d ago)"

            # Check maximum interval
            if days_since_rebalance >= self.max_rebalance_interval:
                logger.info(f"Max interval reached: {days_since_rebalance}d")
                return True, "Maximum interval exceeded"

        # Check risk breach (emergency rebalancing)
        if current_risk is not None and target_risk is not None:
            risk_ratio = current_risk / target_risk
            if risk_ratio > self.risk_breach_threshold:
                logger.warning(
                    f"Risk breach detected: {risk_ratio:.2f}x target "
                    f"(current={current_risk:.3f}, target={target_risk:.3f})"
                )
                return True, "Risk breach - emergency rebalancing"

        # Check drift threshold
        total_drift = self.calculate_total_drift(current_weights, target_weights)

        if total_drift > self.drift_threshold:
            logger.info(f"Drift threshold breached: {total_drift:.3f} > {self.drift_threshold:.3f}")
            return True, f"Drift threshold breached ({total_drift:.3f})"

        return False, "No rebalancing needed"

    def calculate_rebalancing_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
    ) -> pd.Series:
        """
        Calculate trades needed to rebalance portfolio.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            Trade sizes in dollars (positive = buy, negative = sell)
        """
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current_weights = current_weights.reindex(all_assets, fill_value=0)
        target_weights = target_weights.reindex(all_assets, fill_value=0)

        # Calculate weight changes
        weight_changes = target_weights - current_weights

        # Convert to dollar amounts
        trades = weight_changes * portfolio_value

        # Filter out small trades
        min_trade_value = self.min_trade_size * portfolio_value
        trades[trades.abs() < min_trade_value] = 0

        logger.debug(
            f"Rebalancing trades: {(trades != 0).sum()} non-zero trades, "
            f"total_trade_value=${trades.abs().sum():,.2f}"
        )

        return trades

    def optimize_rebalancing_bands(
        self,
        target_weights: pd.Series,
        volatilities: pd.Series,
        base_band: float = 0.05,
    ) -> pd.DataFrame:
        """
        Calculate optimal rebalancing bands for each asset.

        Higher volatility assets get wider bands to reduce turnover.
        Lower volatility assets get tighter bands.

        Args:
            target_weights: Target portfolio weights
            volatilities: Asset volatilities
            base_band: Base rebalancing band width

        Returns:
            DataFrame with lower_band and upper_band for each asset
        """
        # Normalize volatilities
        avg_vol = volatilities.mean()
        vol_ratios = volatilities / avg_vol

        # Wider bands for high-vol assets
        band_widths = base_band * vol_ratios

        # Calculate bands around target weights
        bands = pd.DataFrame({
            'target': target_weights,
            'lower_band': target_weights - band_widths,
            'upper_band': target_weights + band_widths,
        })

        # Ensure bands are positive
        bands['lower_band'] = bands['lower_band'].clip(lower=0)

        logger.debug(
            f"Rebalancing bands: width range "
            f"[{band_widths.min():.3f}, {band_widths.max():.3f}]"
        )

        return bands

    def check_band_breach(
        self,
        current_weights: pd.Series,
        rebalancing_bands: pd.DataFrame,
    ) -> pd.Series:
        """
        Check which assets have breached their rebalancing bands.

        Args:
            current_weights: Current portfolio weights
            rebalancing_bands: DataFrame with lower_band and upper_band

        Returns:
            Boolean Series indicating which assets breached bands
        """
        breached = pd.Series(False, index=current_weights.index)

        for asset in current_weights.index:
            if asset not in rebalancing_bands.index:
                continue

            current = current_weights[asset]
            lower = rebalancing_bands.loc[asset, 'lower_band']
            upper = rebalancing_bands.loc[asset, 'upper_band']

            if current < lower or current > upper:
                breached[asset] = True

        n_breached = breached.sum()
        logger.debug(f"Band breach check: {n_breached}/{len(breached)} assets breached")

        return breached

    def estimate_transaction_costs(
        self,
        trades: pd.Series,
        portfolio_value: float,
    ) -> float:
        """
        Estimate transaction costs for rebalancing trades.

        Costs include:
        - Proportional costs (commissions, spreads)
        - Market impact (for large trades)

        Args:
            trades: Trade sizes in dollars
            portfolio_value: Total portfolio value

        Returns:
            Total estimated transaction cost
        """
        # Proportional costs
        trade_value = trades.abs().sum()
        proportional_cost = trade_value * (self.transaction_cost_bps / 10000)

        # Market impact (square-root model for simplicity)
        # Impact increases with square root of trade size
        impact_cost = 0
        for trade in trades:
            if trade != 0:
                trade_fraction = abs(trade) / portfolio_value
                # Simple impact model: 5 bps * sqrt(trade_fraction / 0.01)
                impact = 5 * np.sqrt(trade_fraction / 0.01) / 10000
                impact_cost += abs(trade) * impact

        total_cost = proportional_cost + impact_cost

        logger.debug(
            f"Transaction costs: proportional=${proportional_cost:.2f}, "
            f"impact=${impact_cost:.2f}, total=${total_cost:.2f}"
        )

        return total_cost

    def calculate_rebalancing_benefit(
        self,
        drift: pd.Series,
        expected_returns: pd.Series,
        volatilities: pd.Series,
    ) -> float:
        """
        Estimate benefit of rebalancing (reduced risk and improved returns).

        Args:
            drift: Weight drift from target
            expected_returns: Expected returns for each asset
            volatilities: Asset volatilities

        Returns:
            Estimated benefit in dollars (or basis points)
        """
        # Benefit from reducing drift:
        # - Reduced tracking error
        # - Return to optimal risk/return profile

        # Simple model: benefit proportional to drift magnitude
        # weighted by expected returns and volatilities
        risk_reduction_benefit = (drift * volatilities).sum() * 100  # bps

        return_improvement_benefit = (drift * expected_returns.abs()).sum() * 100  # bps

        total_benefit = risk_reduction_benefit + return_improvement_benefit

        logger.debug(
            f"Rebalancing benefit: risk_reduction={risk_reduction_benefit:.2f}bps, "
            f"return_improvement={return_improvement_benefit:.2f}bps"
        )

        return total_benefit

    def execute_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
        current_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Execute portfolio rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            current_date: Date of rebalancing

        Returns:
            Dictionary with rebalancing details
        """
        if current_date is None:
            current_date = datetime.now()

        logger.info(f"Executing rebalancing on {current_date.strftime('%Y-%m-%d')}")

        # Calculate trades
        trades = self.calculate_rebalancing_trades(
            current_weights,
            target_weights,
            portfolio_value
        )

        # Estimate costs
        transaction_costs = self.estimate_transaction_costs(trades, portfolio_value)

        # Calculate drift
        total_drift = self.calculate_total_drift(current_weights, target_weights)

        # Update state
        self.last_rebalance_date = current_date

        # Record in history
        rebalance_record = {
            'date': current_date,
            'drift': total_drift,
            'n_trades': (trades != 0).sum(),
            'trade_value': trades.abs().sum(),
            'transaction_costs': transaction_costs,
            'cost_ratio': transaction_costs / portfolio_value,
        }
        self.rebalance_history.append(rebalance_record)

        logger.info(
            f"Rebalancing executed: {rebalance_record['n_trades']} trades, "
            f"drift={total_drift:.3f}, costs=${transaction_costs:.2f} "
            f"({rebalance_record['cost_ratio']:.4%} of portfolio)"
        )

        return {
            'trades': trades,
            'transaction_costs': transaction_costs,
            'drift': total_drift,
            'date': current_date,
            'new_weights': target_weights,
        }

    def get_rebalancing_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'monthly',
    ) -> List[datetime]:
        """
        Generate rebalancing schedule.

        Args:
            start_date: Start date
            end_date: End date
            frequency: Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')

        Returns:
            List of rebalancing dates
        """
        schedule = []
        current = start_date

        if frequency == 'daily':
            delta = timedelta(days=1)
        elif frequency == 'weekly':
            delta = timedelta(days=7)
        elif frequency == 'monthly':
            delta = timedelta(days=30)
        elif frequency == 'quarterly':
            delta = timedelta(days=90)
        else:
            logger.warning(f"Unknown frequency {frequency}, using monthly")
            delta = timedelta(days=30)

        while current <= end_date:
            schedule.append(current)
            current += delta

        logger.info(
            f"Rebalancing schedule: {len(schedule)} dates from "
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        return schedule

    def analyze_rebalancing_history(self) -> Dict:
        """
        Analyze historical rebalancing performance.

        Returns:
            Dictionary of statistics
        """
        if not self.rebalance_history:
            logger.warning("No rebalancing history available")
            return {}

        df = pd.DataFrame(self.rebalance_history)

        stats = {
            'n_rebalances': len(df),
            'avg_drift': df['drift'].mean(),
            'max_drift': df['drift'].max(),
            'avg_trades_per_rebalance': df['n_trades'].mean(),
            'total_transaction_costs': df['transaction_costs'].sum(),
            'avg_cost_ratio': df['cost_ratio'].mean(),
            'total_trade_value': df['trade_value'].sum(),
        }

        logger.info(
            f"Rebalancing history analysis: {stats['n_rebalances']} rebalances, "
            f"avg_drift={stats['avg_drift']:.3f}, "
            f"total_costs=${stats['total_transaction_costs']:.2f}"
        )

        return stats

    def smart_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
        expected_returns: pd.Series,
        volatilities: pd.Series,
        current_date: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """
        Smart rebalancing with cost-benefit analysis.

        Only rebalances if benefits exceed costs.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            expected_returns: Expected returns for each asset
            volatilities: Asset volatilities
            current_date: Date of rebalancing

        Returns:
            Rebalancing result dict if executed, None if skipped
        """
        # Check if rebalancing is needed
        should_rebalance, reason = self.should_rebalance(
            current_weights,
            target_weights,
            current_date
        )

        if not should_rebalance:
            logger.debug(f"Skipping rebalance: {reason}")
            return None

        # Calculate potential trades and costs
        trades = self.calculate_rebalancing_trades(
            current_weights,
            target_weights,
            portfolio_value
        )
        costs = self.estimate_transaction_costs(trades, portfolio_value)

        # Calculate benefits
        drift = self.calculate_drift(current_weights, target_weights)
        benefits = self.calculate_rebalancing_benefit(drift, expected_returns, volatilities)

        # Cost-benefit analysis
        benefit_to_cost_ratio = benefits / (costs + 1e-8)

        logger.info(
            f"Rebalancing cost-benefit: benefits={benefits:.2f}bps, "
            f"costs=${costs:.2f}, ratio={benefit_to_cost_ratio:.2f}"
        )

        # Only rebalance if benefits > costs (with margin of safety)
        if benefit_to_cost_ratio < 1.5 and reason != "Risk breach - emergency rebalancing":
            logger.info("Benefits don't justify costs, skipping rebalance")
            return None

        # Execute rebalancing
        return self.execute_rebalance(
            current_weights,
            target_weights,
            portfolio_value,
            current_date
        )

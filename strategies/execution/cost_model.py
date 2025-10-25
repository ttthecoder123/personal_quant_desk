"""
Comprehensive transaction cost modeling.

Models all costs associated with trading:
- Broker commissions
- Exchange fees
- Regulatory fees
- Funding costs for leveraged positions
- Currency conversion costs
- Total Cost of Ownership (TCO)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from loguru import logger


class CostModel:
    """
    Transaction cost model for realistic P&L calculation.

    Models multiple cost components:
    - Broker commissions (per-share and per-trade)
    - Exchange fees
    - SEC/regulatory fees
    - Borrowing costs for short positions
    - Funding costs for leveraged positions
    - Currency conversion costs
    - Total Cost of Ownership

    Attributes:
        commission_per_share (float): Commission per share in dollars
        commission_per_trade (float): Fixed commission per trade
        min_commission (float): Minimum commission per trade
        max_commission (float): Maximum commission per trade
        exchange_fee_bps (float): Exchange fees in basis points
        sec_fee_rate (float): SEC fee rate (dollars per million)
        short_borrow_rate (float): Annual rate for borrowing shares to short
        margin_interest_rate (float): Annual interest rate on margin
        fx_conversion_bps (float): Currency conversion cost in basis points
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        commission_per_trade: float = 1.0,
        min_commission: float = 1.0,
        max_commission: float = 0.005,  # As percentage of trade value
        exchange_fee_bps: float = 0.3,
        sec_fee_rate: float = 22.90,  # Per million dollars
        short_borrow_rate: float = 0.03,  # 3% annual
        margin_interest_rate: float = 0.05,  # 5% annual
        fx_conversion_bps: float = 2.0,
    ):
        """
        Initialize cost model.

        Args:
            commission_per_share: Commission per share (e.g., $0.005)
            commission_per_trade: Fixed commission per trade
            min_commission: Minimum commission per trade
            max_commission: Maximum commission as % of trade value
            exchange_fee_bps: Exchange fees in basis points
            sec_fee_rate: SEC fee in dollars per million traded
            short_borrow_rate: Annual short borrowing rate
            margin_interest_rate: Annual margin interest rate
            fx_conversion_bps: FX conversion cost in bps
        """
        self.commission_per_share = commission_per_share
        self.commission_per_trade = commission_per_trade
        self.min_commission = min_commission
        self.max_commission = max_commission
        self.exchange_fee_bps = exchange_fee_bps
        self.sec_fee_rate = sec_fee_rate
        self.short_borrow_rate = short_borrow_rate
        self.margin_interest_rate = margin_interest_rate
        self.fx_conversion_bps = fx_conversion_bps

        logger.info(
            f"CostModel initialized: commission=${commission_per_share:.4f}/share, "
            f"fixed=${commission_per_trade:.2f}, exchange={exchange_fee_bps}bps"
        )

    def calculate_commission(
        self,
        quantity: float,
        price: float,
    ) -> float:
        """
        Calculate broker commission.

        Commission structure:
        - Per-share fee + fixed per-trade fee
        - Subject to minimum and maximum caps

        Args:
            quantity: Number of shares
            price: Price per share

        Returns:
            Total commission in dollars
        """
        # Base commission: per-share * quantity + per-trade
        base_commission = (self.commission_per_share * quantity) + self.commission_per_trade

        # Apply minimum
        commission = max(base_commission, self.min_commission)

        # Apply maximum (as percentage of trade value)
        trade_value = quantity * price
        max_commission_dollars = trade_value * self.max_commission
        commission = min(commission, max_commission_dollars)

        logger.debug(
            f"Commission: ${commission:.2f} for {quantity} shares @ ${price:.2f} "
            f"(${trade_value:,.2f} trade value)"
        )

        return commission

    def calculate_exchange_fees(
        self,
        quantity: float,
        price: float,
        add_liquidity: bool = False,
    ) -> float:
        """
        Calculate exchange fees.

        Exchanges charge fees for removing liquidity and sometimes
        provide rebates for adding liquidity.

        Args:
            quantity: Number of shares
            price: Price per share
            add_liquidity: Whether order adds liquidity (maker vs taker)

        Returns:
            Exchange fee in dollars (negative = rebate)
        """
        trade_value = quantity * price

        if add_liquidity:
            # Maker rebate (negative fee)
            fee = -trade_value * (self.exchange_fee_bps * 0.5) / 10000
        else:
            # Taker fee (positive cost)
            fee = trade_value * self.exchange_fee_bps / 10000

        logger.debug(
            f"Exchange {'rebate' if add_liquidity else 'fee'}: "
            f"${abs(fee):.2f} ({self.exchange_fee_bps * (0.5 if add_liquidity else 1.0):.2f}bps)"
        )

        return fee

    def calculate_sec_fees(
        self,
        quantity: float,
        price: float,
        is_sell: bool = True,
    ) -> float:
        """
        Calculate SEC regulatory fees.

        SEC charges fees on sell orders only (in US markets).

        Formula: (trade_value / 1,000,000) * sec_fee_rate

        Args:
            quantity: Number of shares
            price: Price per share
            is_sell: Whether this is a sell order

        Returns:
            SEC fee in dollars
        """
        if not is_sell:
            return 0.0

        trade_value = quantity * price
        fee = (trade_value / 1_000_000) * self.sec_fee_rate

        logger.debug(
            f"SEC fee: ${fee:.4f} for ${trade_value:,.2f} sell order "
            f"({self.sec_fee_rate:.2f}/million)"
        )

        return fee

    def calculate_short_borrow_cost(
        self,
        quantity: float,
        price: float,
        holding_days: float,
        borrow_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate borrowing cost for short positions.

        Hard-to-borrow stocks have higher rates. This cost is charged
        daily while the short position is held.

        Args:
            quantity: Number of shares borrowed
            price: Price per share
            holding_days: Number of days position is held
            borrow_rate: Annual borrow rate (uses default if None)

        Returns:
            Total borrowing cost
        """
        if borrow_rate is None:
            borrow_rate = self.short_borrow_rate

        position_value = quantity * price

        # Daily borrow cost = position_value * (annual_rate / 360)
        daily_cost = position_value * (borrow_rate / 360)
        total_cost = daily_cost * holding_days

        logger.debug(
            f"Short borrow cost: ${total_cost:.2f} "
            f"({borrow_rate:.2%} annual rate for {holding_days:.1f} days)"
        )

        return total_cost

    def calculate_margin_interest(
        self,
        borrowed_amount: float,
        holding_days: float,
        margin_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate margin interest for leveraged positions.

        Interest charged on borrowed capital for leveraged trades.

        Args:
            borrowed_amount: Amount borrowed on margin
            holding_days: Number of days position is held
            margin_rate: Annual margin rate (uses default if None)

        Returns:
            Total margin interest cost
        """
        if margin_rate is None:
            margin_rate = self.margin_interest_rate

        # Daily interest = borrowed_amount * (annual_rate / 360)
        daily_interest = borrowed_amount * (margin_rate / 360)
        total_interest = daily_interest * holding_days

        logger.debug(
            f"Margin interest: ${total_interest:.2f} "
            f"({margin_rate:.2%} annual rate on ${borrowed_amount:,.2f} for {holding_days:.1f} days)"
        )

        return total_interest

    def calculate_fx_conversion_cost(
        self,
        amount: float,
        conversion_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate currency conversion cost.

        For trading in foreign markets, currency conversion incurs costs.

        Args:
            amount: Amount to convert (in base currency)
            conversion_rate: Conversion cost in bps (uses default if None)

        Returns:
            Conversion cost
        """
        if conversion_rate is None:
            conversion_rate = self.fx_conversion_bps

        cost = amount * (conversion_rate / 10000)

        logger.debug(
            f"FX conversion cost: ${cost:.2f} ({conversion_rate:.2f}bps on ${amount:,.2f})"
        )

        return cost

    def calculate_roundtrip_cost(
        self,
        quantity: float,
        entry_price: float,
        exit_price: float,
        holding_days: float = 1.0,
        is_short: bool = False,
        use_margin: bool = False,
        foreign_currency: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate total round-trip transaction costs.

        Includes all costs for entering and exiting a position:
        - Entry commission and fees
        - Exit commission and fees
        - Holding costs (borrow fees or margin interest)
        - FX costs if applicable

        Args:
            quantity: Position size
            entry_price: Entry price per share
            exit_price: Exit price per share
            holding_days: Days position is held
            is_short: Whether this is a short position
            use_margin: Whether position uses margin
            foreign_currency: Whether position is in foreign currency

        Returns:
            Dictionary with cost breakdown
        """
        costs = {}

        # Entry costs
        costs['entry_commission'] = self.calculate_commission(quantity, entry_price)
        costs['entry_exchange_fee'] = self.calculate_exchange_fees(
            quantity, entry_price, add_liquidity=False
        )
        costs['entry_sec_fee'] = 0.0  # No SEC fee on buys

        # Exit costs
        costs['exit_commission'] = self.calculate_commission(quantity, exit_price)
        costs['exit_exchange_fee'] = self.calculate_exchange_fees(
            quantity, exit_price, add_liquidity=False
        )
        costs['exit_sec_fee'] = self.calculate_sec_fees(
            quantity, exit_price, is_sell=True
        )

        # Holding costs
        if is_short:
            costs['short_borrow_cost'] = self.calculate_short_borrow_cost(
                quantity, entry_price, holding_days
            )
        else:
            costs['short_borrow_cost'] = 0.0

        if use_margin:
            # Assume 50% margin (borrow half the position value)
            borrowed = (quantity * entry_price) * 0.5
            costs['margin_interest'] = self.calculate_margin_interest(
                borrowed, holding_days
            )
        else:
            costs['margin_interest'] = 0.0

        # FX costs
        if foreign_currency:
            entry_value = quantity * entry_price
            exit_value = quantity * exit_price
            costs['fx_cost'] = (
                self.calculate_fx_conversion_cost(entry_value) +
                self.calculate_fx_conversion_cost(exit_value)
            )
        else:
            costs['fx_cost'] = 0.0

        # Total costs
        costs['total_entry_costs'] = (
            costs['entry_commission'] +
            costs['entry_exchange_fee'] +
            costs['entry_sec_fee']
        )

        costs['total_exit_costs'] = (
            costs['exit_commission'] +
            costs['exit_exchange_fee'] +
            costs['exit_sec_fee']
        )

        costs['total_holding_costs'] = (
            costs['short_borrow_cost'] +
            costs['margin_interest']
        )

        costs['total_costs'] = (
            costs['total_entry_costs'] +
            costs['total_exit_costs'] +
            costs['total_holding_costs'] +
            costs['fx_cost']
        )

        # Calculate as percentage of position value
        position_value = quantity * entry_price
        costs['total_cost_pct'] = (costs['total_costs'] / position_value) * 100
        costs['total_cost_bps'] = costs['total_cost_pct'] * 100

        logger.info(
            f"Round-trip costs: ${costs['total_costs']:.2f} "
            f"({costs['total_cost_bps']:.2f}bps) for {quantity} shares "
            f"@ ${entry_price:.2f} held {holding_days:.1f} days"
        )

        return costs

    def calculate_portfolio_tco(
        self,
        trades: pd.DataFrame,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Calculate Total Cost of Ownership (TCO) for portfolio.

        TCO is the comprehensive cost of running a trading strategy,
        expressed as percentage of portfolio value or AUM.

        Args:
            trades: DataFrame with trade history (columns: quantity, price,
                    side, holding_days, etc.)
            portfolio_value: Total portfolio value (AUM)

        Returns:
            Dictionary with TCO metrics
        """
        if trades.empty:
            logger.warning("No trades for TCO calculation")
            return {}

        total_costs = 0.0
        cost_components = {
            'commissions': 0.0,
            'exchange_fees': 0.0,
            'sec_fees': 0.0,
            'borrow_costs': 0.0,
            'margin_interest': 0.0,
            'fx_costs': 0.0,
        }

        # Calculate costs for each trade
        for idx, trade in trades.iterrows():
            quantity = trade['quantity']
            price = trade['price']
            side = trade.get('side', 'BUY')
            holding_days = trade.get('holding_days', 1.0)

            # Commission
            commission = self.calculate_commission(quantity, price)
            cost_components['commissions'] += commission

            # Exchange fees
            exchange_fee = self.calculate_exchange_fees(quantity, price)
            cost_components['exchange_fees'] += exchange_fee

            # SEC fees (sells only)
            if side.upper() == 'SELL':
                sec_fee = self.calculate_sec_fees(quantity, price, is_sell=True)
                cost_components['sec_fees'] += sec_fee

            # Borrow costs (shorts only)
            if side.upper() == 'SELL' and quantity < 0:
                borrow_cost = self.calculate_short_borrow_cost(
                    abs(quantity), price, holding_days
                )
                cost_components['borrow_costs'] += borrow_cost

            total_costs += (commission + exchange_fee +
                           cost_components['sec_fees'] +
                           cost_components['borrow_costs'])

        # Calculate TCO metrics
        tco_metrics = {
            **cost_components,
            'total_costs': sum(cost_components.values()),
            'n_trades': len(trades),
            'avg_cost_per_trade': sum(cost_components.values()) / max(len(trades), 1),
            'tco_pct': (sum(cost_components.values()) / portfolio_value) * 100,
            'tco_bps': (sum(cost_components.values()) / portfolio_value) * 10000,
        }

        logger.info(
            f"Portfolio TCO: ${tco_metrics['total_costs']:,.2f} "
            f"({tco_metrics['tco_bps']:.2f}bps) on ${portfolio_value:,.2f} AUM, "
            f"{tco_metrics['n_trades']} trades"
        )

        return tco_metrics

    def estimate_annual_costs(
        self,
        avg_daily_turnover: float,
        portfolio_value: float,
        trading_days: int = 252,
    ) -> Dict[str, float]:
        """
        Estimate annual trading costs based on turnover.

        Args:
            avg_daily_turnover: Average daily turnover (as fraction of portfolio)
            portfolio_value: Portfolio value
            trading_days: Number of trading days per year

        Returns:
            Dictionary with annual cost estimates
        """
        # Daily traded value
        daily_trade_value = portfolio_value * avg_daily_turnover

        # Estimate costs per dollar traded
        # Assuming average price of $50 and 100 share lots
        avg_shares = daily_trade_value / 50
        daily_commission = self.calculate_commission(avg_shares, 50)
        daily_exchange_fee = self.calculate_exchange_fees(avg_shares, 50)
        daily_sec_fee = self.calculate_sec_fees(avg_shares, 50) * 0.5  # Half are sells

        daily_total = daily_commission + daily_exchange_fee + daily_sec_fee

        # Annualize
        annual_costs = {
            'annual_commissions': daily_commission * trading_days,
            'annual_exchange_fees': daily_exchange_fee * trading_days,
            'annual_sec_fees': daily_sec_fee * trading_days,
            'annual_total_costs': daily_total * trading_days,
            'annual_turnover': avg_daily_turnover * trading_days,
            'cost_per_turnover_bps': (daily_total / daily_trade_value) * 10000,
            'annual_cost_pct': (daily_total * trading_days / portfolio_value) * 100,
        }

        logger.info(
            f"Annual cost estimate: ${annual_costs['annual_total_costs']:,.2f} "
            f"({annual_costs['annual_cost_pct']:.2f}% of AUM) "
            f"for {annual_costs['annual_turnover']:.1f}x turnover"
        )

        return annual_costs

    def compare_brokers(
        self,
        quantity: float,
        price: float,
        broker_configs: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Compare costs across different brokers.

        Args:
            quantity: Trade quantity
            price: Price per share
            broker_configs: Dictionary of broker configurations

        Returns:
            DataFrame comparing costs
        """
        comparisons = []

        for broker_name, config in broker_configs.items():
            # Temporarily update model parameters
            original_params = {
                'commission_per_share': self.commission_per_share,
                'commission_per_trade': self.commission_per_trade,
                'min_commission': self.min_commission,
            }

            self.commission_per_share = config.get('commission_per_share', 0.005)
            self.commission_per_trade = config.get('commission_per_trade', 1.0)
            self.min_commission = config.get('min_commission', 1.0)

            # Calculate costs
            commission = self.calculate_commission(quantity, price)

            comparisons.append({
                'broker': broker_name,
                'commission': commission,
                'commission_pct': (commission / (quantity * price)) * 100,
            })

            # Restore original parameters
            for param, value in original_params.items():
                setattr(self, param, value)

        comparison_df = pd.DataFrame(comparisons).sort_values('commission')

        logger.info(
            f"Broker comparison: {len(comparisons)} brokers for "
            f"{quantity} shares @ ${price:.2f}"
        )

        return comparison_df

    def get_cost_summary(self) -> Dict[str, float]:
        """
        Get summary of current cost model parameters.

        Returns:
            Dictionary with all cost parameters
        """
        return {
            'commission_per_share': self.commission_per_share,
            'commission_per_trade': self.commission_per_trade,
            'min_commission': self.min_commission,
            'max_commission_pct': self.max_commission * 100,
            'exchange_fee_bps': self.exchange_fee_bps,
            'sec_fee_per_million': self.sec_fee_rate,
            'short_borrow_rate_annual': self.short_borrow_rate * 100,
            'margin_rate_annual': self.margin_interest_rate * 100,
            'fx_conversion_bps': self.fx_conversion_bps,
        }

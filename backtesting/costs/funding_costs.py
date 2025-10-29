"""
Funding Cost Models

Implements funding cost models for backtesting:
- Margin interest rates (tiered by balance)
- Overnight funding costs
- Currency carry costs
- Leverage costs
- Capital allocation costs
- Opportunity cost of capital

Funding costs represent the cost of borrowing capital to finance positions.
For leveraged strategies, these costs can be significant and must be accurately modeled.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
import numpy as np
from loguru import logger


class AccountType(Enum):
    """Account type affecting funding rates."""
    RETAIL = "RETAIL"
    PROFESSIONAL = "PROFESSIONAL"
    INSTITUTIONAL = "INSTITUTIONAL"
    HEDGE_FUND = "HEDGE_FUND"


class PositionType(Enum):
    """Position type for funding calculation."""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


@dataclass
class FundingCostResult:
    """
    Result of funding cost calculation.

    Attributes:
        daily_funding_rate: Daily funding rate (annualized)
        daily_funding_cost: Dollar cost per day
        total_funding_cost: Total cost for period
        margin_interest: Margin interest component
        carry_cost: Currency carry component
        opportunity_cost: Opportunity cost component
        cost_breakdown: Detailed breakdown
    """
    daily_funding_rate: float  # Annualized
    daily_funding_cost: float
    total_funding_cost: float
    margin_interest: float = 0.0
    carry_cost: float = 0.0
    opportunity_cost: float = 0.0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Populate breakdown."""
        self.cost_breakdown = {
            'margin_interest': self.margin_interest,
            'carry_cost': self.carry_cost,
            'opportunity_cost': self.opportunity_cost,
            'total': self.total_funding_cost
        }


class MarginInterestModel:
    """
    Margin interest rate model.

    Models interest charged on margin loans (borrowed capital).
    Rates typically:
    - Tiered by account balance (larger balances → lower rates)
    - Based on benchmark rate (e.g., Federal Funds + spread)
    - Vary by account type
    """

    def __init__(
        self,
        base_rate: float = 0.055,  # 5.5% base rate
        account_type: AccountType = AccountType.RETAIL
    ):
        """
        Initialize margin interest model.

        Args:
            base_rate: Base margin rate (annualized)
            account_type: Type of account
        """
        self.base_rate = base_rate
        self.account_type = account_type

        # Tiered rates by account balance (example: Interactive Brokers-style)
        self.retail_tiers = [
            {'min_balance': 0, 'rate': base_rate + 0.030},  # +300 bps
            {'min_balance': 100_000, 'rate': base_rate + 0.020},  # +200 bps
            {'min_balance': 1_000_000, 'rate': base_rate + 0.010},  # +100 bps
            {'min_balance': 3_000_000, 'rate': base_rate + 0.005},  # +50 bps
        ]

        self.professional_tiers = [
            {'min_balance': 0, 'rate': base_rate + 0.015},  # +150 bps
            {'min_balance': 100_000, 'rate': base_rate + 0.010},  # +100 bps
            {'min_balance': 1_000_000, 'rate': base_rate + 0.005},  # +50 bps
            {'min_balance': 3_000_000, 'rate': base_rate},  # Base rate
        ]

        self.institutional_tiers = [
            {'min_balance': 0, 'rate': base_rate + 0.005},  # +50 bps
            {'min_balance': 1_000_000, 'rate': base_rate},  # Base rate
            {'min_balance': 10_000_000, 'rate': base_rate - 0.005},  # -50 bps
        ]

        logger.info(
            f"Initialized MarginInterestModel: "
            f"base={base_rate*100:.2f}%, type={account_type.value}"
        )

    def get_margin_rate(self, account_balance: float) -> float:
        """
        Get margin interest rate based on account balance.

        Args:
            account_balance: Total account balance

        Returns:
            Annualized margin rate
        """
        # Select tier structure
        if self.account_type == AccountType.RETAIL:
            tiers = self.retail_tiers
        elif self.account_type == AccountType.PROFESSIONAL:
            tiers = self.professional_tiers
        else:
            tiers = self.institutional_tiers

        # Find applicable tier
        applicable_rate = tiers[0]['rate']
        for tier in reversed(tiers):
            if account_balance >= tier['min_balance']:
                applicable_rate = tier['rate']
                break

        return applicable_rate

    def calculate_margin_interest(
        self,
        borrowed_amount: float,
        account_balance: float,
        holding_days: int = 1
    ) -> float:
        """
        Calculate margin interest cost.

        Args:
            borrowed_amount: Amount borrowed on margin
            account_balance: Total account balance
            holding_days: Days holding position

        Returns:
            Total margin interest cost
        """
        if borrowed_amount <= 0:
            return 0.0

        rate = self.get_margin_rate(account_balance)
        daily_rate = rate / 365
        daily_cost = borrowed_amount * daily_rate
        total_cost = daily_cost * holding_days

        return total_cost


class OvernightFundingModel:
    """
    Overnight funding cost model.

    Models overnight financing charges, common in:
    - CFD trading
    - Forex
    - Perpetual futures
    - Some equity derivatives

    Rate typically tied to overnight interbank rates (e.g., SOFR, SONIA).
    """

    def __init__(
        self,
        benchmark_rate: float = 0.05,  # Benchmark overnight rate (e.g., SOFR)
        long_spread: float = 0.0025,  # 25 bps spread for longs
        short_spread: float = -0.0015,  # -15 bps for shorts (may receive rebate)
        weekend_multiplier: float = 3.0  # Triple charge over weekends
    ):
        """
        Initialize overnight funding model.

        Args:
            benchmark_rate: Benchmark overnight rate (annualized)
            long_spread: Spread for long positions
            short_spread: Spread for short positions (can be negative)
            weekend_multiplier: Multiplier for weekends
        """
        self.benchmark_rate = benchmark_rate
        self.long_spread = long_spread
        self.short_spread = short_spread
        self.weekend_multiplier = weekend_multiplier

        logger.info(
            f"Initialized OvernightFundingModel: "
            f"benchmark={benchmark_rate*100:.2f}%"
        )

    def calculate_overnight_funding(
        self,
        position_value: float,
        position_type: PositionType,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Calculate overnight funding cost.

        Args:
            position_value: Total position value
            position_type: LONG or SHORT
            timestamp: Current timestamp (for weekend detection)

        Returns:
            Overnight funding cost (positive = cost, negative = rebate)
        """
        # Determine rate
        if position_type == PositionType.LONG:
            rate = self.benchmark_rate + self.long_spread
        else:  # SHORT
            rate = self.benchmark_rate + self.short_spread

        # Daily funding
        daily_rate = rate / 365
        funding = position_value * daily_rate

        # Weekend adjustment
        if timestamp and timestamp.weekday() == 4:  # Friday
            funding *= self.weekend_multiplier

        return funding


class CurrencyCarryModel:
    """
    Currency carry cost model.

    Models the cost/benefit of holding positions in different currencies.
    Important for:
    - Forex trading
    - International equity portfolios
    - Multi-currency accounts

    Carry = interest rate differential between currencies.
    """

    def __init__(self):
        """Initialize currency carry model."""
        # Example interest rates by currency (annualized)
        self.interest_rates = {
            'USD': 0.055,  # 5.5%
            'EUR': 0.040,  # 4.0%
            'GBP': 0.052,  # 5.2%
            'JPY': 0.001,  # 0.1%
            'AUD': 0.045,  # 4.5%
            'CAD': 0.050,  # 5.0%
            'CHF': 0.015,  # 1.5%
            'NZD': 0.055,  # 5.5%
        }

        logger.info("Initialized CurrencyCarryModel")

    def calculate_carry_cost(
        self,
        position_value: float,
        base_currency: str,
        quote_currency: str,
        holding_days: int = 1
    ) -> float:
        """
        Calculate currency carry cost/benefit.

        Args:
            position_value: Position value in quote currency
            base_currency: Base currency (what you're borrowing)
            quote_currency: Quote currency (what you're lending)
            holding_days: Holding period

        Returns:
            Carry cost (positive = cost, negative = benefit)
        """
        base_rate = self.interest_rates.get(base_currency, 0.03)
        quote_rate = self.interest_rates.get(quote_currency, 0.03)

        # Interest rate differential (annualized)
        carry_rate = base_rate - quote_rate

        # Daily carry
        daily_carry = position_value * (carry_rate / 365)
        total_carry = daily_carry * holding_days

        return total_carry

    def update_rates(self, currency: str, rate: float):
        """
        Update interest rate for a currency.

        Args:
            currency: Currency code
            rate: New annualized interest rate
        """
        self.interest_rates[currency] = rate
        logger.info(f"Updated {currency} rate to {rate*100:.2f}%")


class LeverageCostModel:
    """
    Leverage cost model.

    Models the implicit cost of leverage, including:
    - Direct financing costs
    - Risk premium for leverage
    - Margin call risk
    - Forced liquidation risk

    Higher leverage → higher costs (non-linear).
    """

    def __init__(
        self,
        base_funding_rate: float = 0.05,
        leverage_premium_factor: float = 0.5
    ):
        """
        Initialize leverage cost model.

        Args:
            base_funding_rate: Base funding rate without leverage
            leverage_premium_factor: Premium scaling with leverage
        """
        self.base_funding_rate = base_funding_rate
        self.leverage_premium_factor = leverage_premium_factor

        logger.info(
            f"Initialized LeverageCostModel: "
            f"base={base_funding_rate*100:.2f}%"
        )

    def calculate_leverage_ratio(
        self,
        total_exposure: float,
        account_equity: float
    ) -> float:
        """
        Calculate leverage ratio.

        Args:
            total_exposure: Total position exposure (gross)
            account_equity: Account equity

        Returns:
            Leverage ratio (e.g., 2.0 = 2x leverage)
        """
        if account_equity <= 0:
            return 0.0
        return total_exposure / account_equity

    def calculate_leverage_premium(self, leverage_ratio: float) -> float:
        """
        Calculate leverage premium.

        Premium increases non-linearly with leverage.

        Args:
            leverage_ratio: Current leverage ratio

        Returns:
            Additional rate premium (annualized)
        """
        if leverage_ratio <= 1.0:
            return 0.0

        # Non-linear premium: increases with square root of excess leverage
        excess_leverage = leverage_ratio - 1.0
        premium = self.leverage_premium_factor * (excess_leverage ** 0.75) * 0.01

        return premium

    def calculate_leverage_cost(
        self,
        total_exposure: float,
        account_equity: float,
        holding_days: int = 1
    ) -> Dict[str, float]:
        """
        Calculate leverage cost.

        Args:
            total_exposure: Total position exposure
            account_equity: Account equity
            holding_days: Holding period

        Returns:
            Dictionary with leverage metrics and costs
        """
        leverage_ratio = self.calculate_leverage_ratio(total_exposure, account_equity)
        leverage_premium = self.calculate_leverage_premium(leverage_ratio)

        # Total rate
        total_rate = self.base_funding_rate + leverage_premium

        # Calculate cost on borrowed amount
        borrowed_amount = max(0, total_exposure - account_equity)
        daily_cost = borrowed_amount * (total_rate / 365)
        total_cost = daily_cost * holding_days

        return {
            'leverage_ratio': leverage_ratio,
            'leverage_premium': leverage_premium,
            'total_rate': total_rate,
            'daily_cost': daily_cost,
            'total_cost': total_cost
        }


class CapitalAllocationModel:
    """
    Capital allocation cost model.

    Models the opportunity cost of tying up capital in a strategy.
    Important for:
    - Portfolio optimization
    - Strategy comparison
    - Risk-adjusted returns

    Capital has an opportunity cost = could be deployed elsewhere.
    """

    def __init__(
        self,
        hurdle_rate: float = 0.10,  # 10% minimum acceptable return
        risk_free_rate: float = 0.05  # 5% risk-free rate
    ):
        """
        Initialize capital allocation model.

        Args:
            hurdle_rate: Minimum acceptable return (annualized)
            risk_free_rate: Risk-free rate (annualized)
        """
        self.hurdle_rate = hurdle_rate
        self.risk_free_rate = risk_free_rate

        logger.info(
            f"Initialized CapitalAllocationModel: "
            f"hurdle={hurdle_rate*100:.1f}%, rf={risk_free_rate*100:.1f}%"
        )

    def calculate_opportunity_cost(
        self,
        capital_allocated: float,
        holding_days: int = 1,
        strategy_sharpe: Optional[float] = None
    ) -> float:
        """
        Calculate opportunity cost of capital.

        Args:
            capital_allocated: Amount of capital tied up
            holding_days: Holding period
            strategy_sharpe: Strategy Sharpe ratio (affects hurdle rate)

        Returns:
            Opportunity cost
        """
        # Adjust hurdle rate based on strategy risk
        if strategy_sharpe is not None:
            # Higher Sharpe → lower opportunity cost
            if strategy_sharpe > 1.0:
                adjusted_hurdle = self.risk_free_rate
            else:
                adjusted_hurdle = self.hurdle_rate * (1 - strategy_sharpe / 2)
                adjusted_hurdle = max(adjusted_hurdle, self.risk_free_rate)
        else:
            adjusted_hurdle = self.hurdle_rate

        # Calculate opportunity cost
        daily_cost = capital_allocated * (adjusted_hurdle / 365)
        total_cost = daily_cost * holding_days

        return total_cost


class ComprehensiveFundingModel:
    """
    Comprehensive funding cost model combining all components.

    Includes:
    - Margin interest
    - Overnight funding
    - Currency carry
    - Leverage premium
    - Opportunity cost

    Most complete model for production backtesting.
    """

    def __init__(
        self,
        margin_model: Optional[MarginInterestModel] = None,
        overnight_model: Optional[OvernightFundingModel] = None,
        carry_model: Optional[CurrencyCarryModel] = None,
        leverage_model: Optional[LeverageCostModel] = None,
        capital_model: Optional[CapitalAllocationModel] = None
    ):
        """
        Initialize comprehensive funding model.

        Args:
            margin_model: Margin interest model
            overnight_model: Overnight funding model
            carry_model: Currency carry model
            leverage_model: Leverage cost model
            capital_model: Capital allocation model
        """
        self.margin_model = margin_model or MarginInterestModel()
        self.overnight_model = overnight_model or OvernightFundingModel()
        self.carry_model = carry_model or CurrencyCarryModel()
        self.leverage_model = leverage_model or LeverageCostModel()
        self.capital_model = capital_model or CapitalAllocationModel()

        logger.info("Initialized ComprehensiveFundingModel")

    def calculate_funding_cost(
        self,
        position_value: float,
        account_equity: float,
        account_balance: float,
        position_type: PositionType = PositionType.LONG,
        holding_days: int = 1,
        base_currency: str = 'USD',
        quote_currency: str = 'USD',
        timestamp: Optional[datetime] = None,
        include_opportunity_cost: bool = False,
        **kwargs
    ) -> FundingCostResult:
        """
        Calculate comprehensive funding cost.

        Args:
            position_value: Total position value
            account_equity: Account equity
            account_balance: Total account balance
            position_type: LONG or SHORT
            holding_days: Holding period
            base_currency: Base currency
            quote_currency: Quote currency
            timestamp: Current timestamp
            include_opportunity_cost: Whether to include opportunity cost
            **kwargs: Additional parameters

        Returns:
            FundingCostResult object
        """
        # Calculate borrowed amount
        borrowed_amount = max(0, position_value - account_equity)

        # Margin interest (if using margin)
        margin_cost = 0.0
        if borrowed_amount > 0:
            margin_cost = self.margin_model.calculate_margin_interest(
                borrowed_amount=borrowed_amount,
                account_balance=account_balance,
                holding_days=holding_days
            )

        # Overnight funding
        overnight_cost = 0.0
        if kwargs.get('use_overnight_funding', False):
            overnight_cost = self.overnight_model.calculate_overnight_funding(
                position_value=position_value,
                position_type=position_type,
                timestamp=timestamp
            ) * holding_days

        # Currency carry
        carry_cost = 0.0
        if base_currency != quote_currency:
            carry_cost = self.carry_model.calculate_carry_cost(
                position_value=position_value,
                base_currency=base_currency,
                quote_currency=quote_currency,
                holding_days=holding_days
            )

        # Opportunity cost
        opportunity_cost = 0.0
        if include_opportunity_cost:
            opportunity_cost = self.capital_model.calculate_opportunity_cost(
                capital_allocated=account_equity,
                holding_days=holding_days,
                strategy_sharpe=kwargs.get('strategy_sharpe')
            )

        # Total funding cost
        total_cost = margin_cost + overnight_cost + carry_cost + opportunity_cost

        # Calculate effective rate
        if position_value > 0:
            daily_cost = total_cost / holding_days
            daily_rate = (daily_cost / position_value) * 365
        else:
            daily_cost = 0.0
            daily_rate = 0.0

        return FundingCostResult(
            daily_funding_rate=daily_rate,
            daily_funding_cost=daily_cost,
            total_funding_cost=total_cost,
            margin_interest=margin_cost,
            carry_cost=carry_cost,
            opportunity_cost=opportunity_cost
        )

    def calculate_total_leverage_cost(
        self,
        total_exposure: float,
        account_equity: float,
        account_balance: float,
        holding_days: int = 1,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate total leverage cost including premium.

        Args:
            total_exposure: Total position exposure (gross)
            account_equity: Account equity
            account_balance: Total account balance
            holding_days: Holding period
            **kwargs: Additional parameters

        Returns:
            Dictionary with leverage metrics and costs
        """
        # Get leverage cost
        leverage_result = self.leverage_model.calculate_leverage_cost(
            total_exposure=total_exposure,
            account_equity=account_equity,
            holding_days=holding_days
        )

        # Get base funding cost
        borrowed_amount = max(0, total_exposure - account_equity)
        margin_cost = 0.0
        if borrowed_amount > 0:
            margin_cost = self.margin_model.calculate_margin_interest(
                borrowed_amount=borrowed_amount,
                account_balance=account_balance,
                holding_days=holding_days
            )

        # Combine
        total_cost = margin_cost + leverage_result['total_cost']

        return {
            'leverage_ratio': leverage_result['leverage_ratio'],
            'margin_cost': margin_cost,
            'leverage_premium_cost': leverage_result['total_cost'],
            'total_leverage_cost': total_cost,
            'effective_rate': (total_cost / (borrowed_amount * holding_days / 365)) if borrowed_amount > 0 else 0.0
        }

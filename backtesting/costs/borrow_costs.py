"""
Borrow Cost Models

Implements short selling cost models for backtesting:
- Hard-to-borrow fees (based on availability)
- General collateral rates
- Demand-based pricing (supply/demand dynamics)
- Term structure of borrow costs
- Recall risk modeling
- Dividend payment obligations for short sellers

Short selling involves borrowing shares, which incurs costs that can
significantly impact strategy profitability. These costs vary widely
based on stock availability, demand, and market conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class BorrowAvailability(Enum):
    """Stock borrow availability classification."""
    EASY_TO_BORROW = "EASY_TO_BORROW"  # General collateral
    MODERATE = "MODERATE"  # Moderate borrow cost
    HARD_TO_BORROW = "HARD_TO_BORROW"  # High borrow cost
    VERY_HARD = "VERY_HARD"  # Very expensive or limited
    UNBORROW ABLE = "UNBORROWABLE"  # Cannot borrow


class RecallRisk(Enum):
    """Risk of forced share recall."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


@dataclass
class BorrowCostResult:
    """
    Result of borrow cost calculation.

    Attributes:
        daily_borrow_rate: Daily borrow rate (annualized)
        daily_borrow_cost: Dollar cost per day
        total_borrow_cost: Total cost for holding period
        availability: Borrow availability classification
        recall_risk: Risk of forced recall
        dividend_obligations: Dividend payments due on short position
        total_cost: Total short selling cost
        cost_breakdown: Detailed cost breakdown
    """
    daily_borrow_rate: float  # Annualized rate
    daily_borrow_cost: float
    total_borrow_cost: float
    availability: BorrowAvailability
    recall_risk: RecallRisk
    dividend_obligations: float = 0.0
    total_cost: float = 0.0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate totals."""
        self.total_cost = self.total_borrow_cost + self.dividend_obligations
        self.cost_breakdown = {
            'borrow_cost': self.total_borrow_cost,
            'dividend_obligations': self.dividend_obligations,
            'total': self.total_cost
        }


class GeneralCollateralModel:
    """
    General collateral borrow cost model.

    For easy-to-borrow stocks (most liquid large caps).
    Typically charges close to the general collateral (GC) rate,
    which is near the risk-free rate.
    """

    def __init__(
        self,
        base_rate: float = 0.02,  # 2% annualized (GC rate)
        spread: float = 0.005,  # 50 bps spread over GC
        min_rate: float = 0.01,
        max_rate: float = 0.05
    ):
        """
        Initialize general collateral model.

        Args:
            base_rate: Base GC rate (annualized)
            spread: Spread over GC rate
            min_rate: Minimum borrow rate
            max_rate: Maximum borrow rate (for GC)
        """
        self.base_rate = base_rate
        self.spread = spread
        self.min_rate = min_rate
        self.max_rate = max_rate

        logger.info(
            f"Initialized GeneralCollateralModel: "
            f"base={base_rate*100:.2f}%, spread={spread*100:.2f}%"
        )

    def calculate_borrow_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        holding_days: int = 1,
        **kwargs
    ) -> BorrowCostResult:
        """
        Calculate GC borrow cost.

        Args:
            symbol: Trading symbol
            quantity: Number of shares short
            price: Current stock price
            holding_days: Days holding short position
            **kwargs: Additional parameters

        Returns:
            BorrowCostResult object
        """
        # Annualized borrow rate
        borrow_rate = np.clip(
            self.base_rate + self.spread,
            self.min_rate,
            self.max_rate
        )

        # Daily cost
        position_value = quantity * price
        daily_cost = position_value * (borrow_rate / 365)

        # Total cost
        total_cost = daily_cost * holding_days

        return BorrowCostResult(
            daily_borrow_rate=borrow_rate,
            daily_borrow_cost=daily_cost,
            total_borrow_cost=total_cost,
            availability=BorrowAvailability.EASY_TO_BORROW,
            recall_risk=RecallRisk.LOW
        )


class DemandBasedBorrowModel:
    """
    Demand-based borrow pricing model.

    Borrow cost varies with supply/demand dynamics:
    - High short interest → higher borrow cost
    - Limited float → higher cost
    - Recent price moves → higher cost (short squeeze risk)
    - Volatility → higher cost

    Models realistic borrow market behavior.
    """

    def __init__(
        self,
        base_rate: float = 0.02,
        short_interest_factor: float = 2.0,
        float_factor: float = 1.5,
        volatility_factor: float = 1.0,
        max_rate: float = 1.00  # 100% annualized for very hard to borrow
    ):
        """
        Initialize demand-based borrow model.

        Args:
            base_rate: Base borrow rate
            short_interest_factor: Scaling for short interest impact
            float_factor: Scaling for float impact
            volatility_factor: Scaling for volatility impact
            max_rate: Maximum borrow rate
        """
        self.base_rate = base_rate
        self.short_interest_factor = short_interest_factor
        self.float_factor = float_factor
        self.volatility_factor = volatility_factor
        self.max_rate = max_rate

        logger.info(
            f"Initialized DemandBasedBorrowModel: base={base_rate*100:.2f}%"
        )

    def calculate_borrow_rate(
        self,
        symbol: str,
        short_interest_ratio: Optional[float] = None,  # % of float shorted
        float_percentage: Optional[float] = None,  # % of shares outstanding
        volatility: Optional[float] = None,  # Historical volatility
        recent_return: Optional[float] = None,  # Recent price move
        **kwargs
    ) -> float:
        """
        Calculate demand-based borrow rate.

        Args:
            symbol: Trading symbol
            short_interest_ratio: Short interest as % of float
            float_percentage: Float as % of shares outstanding
            volatility: Historical volatility
            recent_return: Recent return (for squeeze risk)
            **kwargs: Additional parameters

        Returns:
            Annualized borrow rate
        """
        rate = self.base_rate

        # Short interest component
        # Higher short interest → harder to borrow
        if short_interest_ratio is not None:
            # Non-linear: rate increases rapidly above 10% short interest
            if short_interest_ratio > 0.10:  # 10%
                si_premium = ((short_interest_ratio - 0.10) ** 1.5) * self.short_interest_factor
                rate += si_premium

        # Float component
        # Lower float → harder to borrow
        if float_percentage is not None and float_percentage < 0.50:  # <50% float
            float_premium = ((0.50 - float_percentage) ** 1.2) * self.float_factor
            rate += float_premium

        # Volatility component
        # Higher vol → higher borrow cost (risk for lender)
        if volatility is not None:
            # Scale: 20% vol → 0.05 = 5% additional cost
            vol_premium = (volatility ** 1.5) * self.volatility_factor
            rate += vol_premium

        # Short squeeze risk
        # Strong positive return + high short interest → much higher cost
        if recent_return is not None and recent_return > 0.10 and \
           short_interest_ratio is not None and short_interest_ratio > 0.15:
            squeeze_premium = recent_return * short_interest_ratio * 2.0
            rate += squeeze_premium

        # Cap at max rate
        rate = min(rate, self.max_rate)

        return rate

    def classify_availability(self, borrow_rate: float) -> BorrowAvailability:
        """
        Classify borrow availability based on rate.

        Args:
            borrow_rate: Annualized borrow rate

        Returns:
            BorrowAvailability classification
        """
        if borrow_rate < 0.05:  # < 5%
            return BorrowAvailability.EASY_TO_BORROW
        elif borrow_rate < 0.15:  # < 15%
            return BorrowAvailability.MODERATE
        elif borrow_rate < 0.40:  # < 40%
            return BorrowAvailability.HARD_TO_BORROW
        elif borrow_rate < 1.00:  # < 100%
            return BorrowAvailability.VERY_HARD
        else:
            return BorrowAvailability.UNBORROWABLE

    def assess_recall_risk(
        self,
        borrow_rate: float,
        short_interest_ratio: Optional[float] = None
    ) -> RecallRisk:
        """
        Assess risk of forced share recall.

        Args:
            borrow_rate: Current borrow rate
            short_interest_ratio: Short interest ratio

        Returns:
            RecallRisk classification
        """
        # High borrow cost indicates tight supply → higher recall risk
        if borrow_rate > 0.50:
            return RecallRisk.VERY_HIGH
        elif borrow_rate > 0.25:
            return RecallRisk.HIGH
        elif borrow_rate > 0.10:
            return RecallRisk.MODERATE

        # Very high short interest also increases recall risk
        if short_interest_ratio is not None and short_interest_ratio > 0.30:
            return RecallRisk.HIGH

        return RecallRisk.LOW

    def calculate_borrow_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        holding_days: int = 1,
        **kwargs
    ) -> BorrowCostResult:
        """
        Calculate demand-based borrow cost.

        Args:
            symbol: Trading symbol
            quantity: Number of shares short
            price: Current stock price
            holding_days: Days holding short position
            **kwargs: Additional parameters (short_interest_ratio, etc.)

        Returns:
            BorrowCostResult object
        """
        # Calculate borrow rate
        borrow_rate = self.calculate_borrow_rate(symbol=symbol, **kwargs)

        # Classify availability
        availability = self.classify_availability(borrow_rate)

        # Assess recall risk
        recall_risk = self.assess_recall_risk(
            borrow_rate=borrow_rate,
            short_interest_ratio=kwargs.get('short_interest_ratio')
        )

        # Calculate costs
        position_value = quantity * price
        daily_cost = position_value * (borrow_rate / 365)
        total_cost = daily_cost * holding_days

        return BorrowCostResult(
            daily_borrow_rate=borrow_rate,
            daily_borrow_cost=daily_cost,
            total_borrow_cost=total_cost,
            availability=availability,
            recall_risk=recall_risk
        )


class TieredBorrowRateModel:
    """
    Tiered borrow rate model based on stock characteristics.

    Assigns stocks to tiers based on market cap, liquidity, etc.
    Simple but effective categorization approach.
    """

    @dataclass
    class BorrowTier:
        """Borrow rate tier."""
        name: str
        min_market_cap: float  # Minimum market cap for tier
        base_rate: float  # Base annualized rate
        recall_risk: RecallRisk

    def __init__(self):
        """Initialize tiered borrow rate model."""
        self.tiers = [
            self.BorrowTier(
                name="Large Cap",
                min_market_cap=10e9,  # $10B+
                base_rate=0.03,  # 3%
                recall_risk=RecallRisk.LOW
            ),
            self.BorrowTier(
                name="Mid Cap",
                min_market_cap=2e9,  # $2B-$10B
                base_rate=0.08,  # 8%
                recall_risk=RecallRisk.MODERATE
            ),
            self.BorrowTier(
                name="Small Cap",
                min_market_cap=500e6,  # $500M-$2B
                base_rate=0.15,  # 15%
                recall_risk=RecallRisk.HIGH
            ),
            self.BorrowTier(
                name="Micro Cap",
                min_market_cap=0,  # <$500M
                base_rate=0.30,  # 30%
                recall_risk=RecallRisk.VERY_HIGH
            ),
        ]

        logger.info(f"Initialized TieredBorrowRateModel with {len(self.tiers)} tiers")

    def get_tier(self, market_cap: float) -> BorrowTier:
        """
        Get tier for stock based on market cap.

        Args:
            market_cap: Market capitalization

        Returns:
            BorrowTier object
        """
        for tier in self.tiers:
            if market_cap >= tier.min_market_cap:
                return tier
        return self.tiers[-1]  # Default to lowest tier

    def calculate_borrow_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        holding_days: int = 1,
        market_cap: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
        **kwargs
    ) -> BorrowCostResult:
        """
        Calculate tiered borrow cost.

        Args:
            symbol: Trading symbol
            quantity: Number of shares short
            price: Current stock price
            holding_days: Days holding short position
            market_cap: Market capitalization
            shares_outstanding: Shares outstanding
            **kwargs: Additional parameters

        Returns:
            BorrowCostResult object
        """
        # Estimate market cap if not provided
        if market_cap is None:
            if shares_outstanding is not None:
                market_cap = shares_outstanding * price
            else:
                # Default to mid-cap
                market_cap = 5e9

        # Get tier
        tier = self.get_tier(market_cap)

        # Calculate cost
        position_value = quantity * price
        daily_cost = position_value * (tier.base_rate / 365)
        total_cost = daily_cost * holding_days

        # Classify availability
        if tier.base_rate < 0.05:
            availability = BorrowAvailability.EASY_TO_BORROW
        elif tier.base_rate < 0.15:
            availability = BorrowAvailability.MODERATE
        elif tier.base_rate < 0.30:
            availability = BorrowAvailability.HARD_TO_BORROW
        else:
            availability = BorrowAvailability.VERY_HARD

        return BorrowCostResult(
            daily_borrow_rate=tier.base_rate,
            daily_borrow_cost=daily_cost,
            total_borrow_cost=total_cost,
            availability=availability,
            recall_risk=tier.recall_risk
        )


class DividendObligationModel:
    """
    Model dividend payment obligations for short sellers.

    Short sellers must pay any dividends declared on shorted shares.
    This can be a significant cost, especially for high-yield stocks.
    """

    def __init__(self):
        """Initialize dividend obligation model."""
        logger.info("Initialized DividendObligationModel")

    def calculate_dividend_obligation(
        self,
        symbol: str,
        quantity: float,
        dividend_per_share: float,
        ex_dividend_dates: List[datetime],
        entry_date: datetime,
        exit_date: datetime
    ) -> float:
        """
        Calculate dividend obligations for short position.

        Args:
            symbol: Trading symbol
            quantity: Number of shares short
            dividend_per_share: Dividend per share
            ex_dividend_dates: List of ex-dividend dates during position
            entry_date: Position entry date
            exit_date: Position exit date

        Returns:
            Total dividend obligation
        """
        total_obligation = 0.0

        for ex_date in ex_dividend_dates:
            # Must pay dividend if short on ex-date
            if entry_date <= ex_date <= exit_date:
                obligation = quantity * dividend_per_share
                total_obligation += obligation

                logger.debug(
                    f"Dividend obligation for {symbol}: "
                    f"{quantity} shares @ ${dividend_per_share:.2f} = ${obligation:.2f}"
                )

        return total_obligation

    def get_estimated_dividends(
        self,
        symbol: str,
        holding_days: int,
        annual_dividend_yield: float,
        price: float
    ) -> float:
        """
        Estimate dividend payments based on yield.

        Args:
            symbol: Trading symbol
            holding_days: Holding period in days
            annual_dividend_yield: Annual dividend yield (%)
            price: Current stock price

        Returns:
            Estimated dividend per share during holding period
        """
        # Annualized dividend
        annual_dividend = price * annual_dividend_yield

        # Prorate for holding period
        estimated_dividend = annual_dividend * (holding_days / 365)

        return estimated_dividend


class TermStructureBorrowModel:
    """
    Term structure of borrow costs.

    Borrow rates can vary by loan term:
    - Overnight: Typically cheapest
    - Term (1 week, 1 month, etc.): May be higher or lower
    - Long-term: Often more expensive (uncertainty premium)

    Models the forward curve of borrow costs.
    """

    def __init__(
        self,
        base_model: Optional[DemandBasedBorrowModel] = None,
        term_premium_curve: Optional[Dict[int, float]] = None
    ):
        """
        Initialize term structure model.

        Args:
            base_model: Base borrow cost model
            term_premium_curve: Premium by days {days: premium_rate}
        """
        self.base_model = base_model or DemandBasedBorrowModel()

        # Default term structure (example premiums)
        if term_premium_curve is None:
            self.term_premium_curve = {
                1: 0.00,      # Overnight: no premium
                7: 0.005,     # 1 week: 50 bps
                30: 0.01,     # 1 month: 100 bps
                90: 0.02,     # 3 months: 200 bps
                180: 0.03,    # 6 months: 300 bps
                365: 0.05,    # 1 year: 500 bps
            }
        else:
            self.term_premium_curve = term_premium_curve

        logger.info("Initialized TermStructureBorrowModel")

    def get_term_premium(self, holding_days: int) -> float:
        """
        Get term premium for holding period.

        Args:
            holding_days: Holding period in days

        Returns:
            Term premium (annualized)
        """
        # Find applicable term
        sorted_terms = sorted(self.term_premium_curve.keys())

        if holding_days <= sorted_terms[0]:
            return self.term_premium_curve[sorted_terms[0]]

        if holding_days >= sorted_terms[-1]:
            return self.term_premium_curve[sorted_terms[-1]]

        # Interpolate
        for i in range(len(sorted_terms) - 1):
            if sorted_terms[i] <= holding_days < sorted_terms[i + 1]:
                days_lower = sorted_terms[i]
                days_upper = sorted_terms[i + 1]
                premium_lower = self.term_premium_curve[days_lower]
                premium_upper = self.term_premium_curve[days_upper]

                # Linear interpolation
                weight = (holding_days - days_lower) / (days_upper - days_lower)
                return premium_lower + weight * (premium_upper - premium_lower)

        return 0.0

    def calculate_borrow_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        holding_days: int = 1,
        **kwargs
    ) -> BorrowCostResult:
        """
        Calculate borrow cost with term structure.

        Args:
            symbol: Trading symbol
            quantity: Number of shares short
            price: Current stock price
            holding_days: Days holding short position
            **kwargs: Additional parameters for base model

        Returns:
            BorrowCostResult object
        """
        # Get base borrow cost
        base_result = self.base_model.calculate_borrow_cost(
            symbol=symbol,
            quantity=quantity,
            price=price,
            holding_days=holding_days,
            **kwargs
        )

        # Add term premium
        term_premium = self.get_term_premium(holding_days)
        adjusted_rate = base_result.daily_borrow_rate + term_premium

        # Recalculate costs with adjusted rate
        position_value = quantity * price
        daily_cost = position_value * (adjusted_rate / 365)
        total_cost = daily_cost * holding_days

        return BorrowCostResult(
            daily_borrow_rate=adjusted_rate,
            daily_borrow_cost=daily_cost,
            total_borrow_cost=total_cost,
            availability=base_result.availability,
            recall_risk=base_result.recall_risk
        )


class ComprehensiveBorrowCostModel:
    """
    Comprehensive borrow cost model combining all components.

    Includes:
    - Demand-based borrow rates
    - Term structure
    - Dividend obligations
    - Recall risk assessment

    Most complete model for production backtesting.
    """

    def __init__(
        self,
        borrow_model: Optional[DemandBasedBorrowModel] = None,
        term_structure_model: Optional[TermStructureBorrowModel] = None,
        dividend_model: Optional[DividendObligationModel] = None
    ):
        """
        Initialize comprehensive borrow cost model.

        Args:
            borrow_model: Base borrow cost model
            term_structure_model: Term structure model
            dividend_model: Dividend obligation model
        """
        self.borrow_model = borrow_model or DemandBasedBorrowModel()
        self.term_structure_model = term_structure_model or TermStructureBorrowModel(self.borrow_model)
        self.dividend_model = dividend_model or DividendObligationModel()

        logger.info("Initialized ComprehensiveBorrowCostModel")

    def calculate_borrow_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        holding_days: int = 1,
        annual_dividend_yield: Optional[float] = None,
        ex_dividend_dates: Optional[List[datetime]] = None,
        entry_date: Optional[datetime] = None,
        exit_date: Optional[datetime] = None,
        **kwargs
    ) -> BorrowCostResult:
        """
        Calculate comprehensive borrow cost.

        Args:
            symbol: Trading symbol
            quantity: Number of shares short
            price: Current stock price
            holding_days: Days holding short position
            annual_dividend_yield: Annual dividend yield
            ex_dividend_dates: Ex-dividend dates during holding
            entry_date: Position entry date
            exit_date: Position exit date
            **kwargs: Additional parameters

        Returns:
            BorrowCostResult object
        """
        # Calculate base borrow cost with term structure
        borrow_result = self.term_structure_model.calculate_borrow_cost(
            symbol=symbol,
            quantity=quantity,
            price=price,
            holding_days=holding_days,
            **kwargs
        )

        # Calculate dividend obligations
        dividend_cost = 0.0

        if ex_dividend_dates and entry_date and exit_date:
            # Exact calculation if dates provided
            dividend_per_share = kwargs.get('dividend_per_share', 0.0)
            dividend_cost = self.dividend_model.calculate_dividend_obligation(
                symbol=symbol,
                quantity=quantity,
                dividend_per_share=dividend_per_share,
                ex_dividend_dates=ex_dividend_dates,
                entry_date=entry_date,
                exit_date=exit_date
            )
        elif annual_dividend_yield:
            # Estimate based on yield
            estimated_dividend = self.dividend_model.get_estimated_dividends(
                symbol=symbol,
                holding_days=holding_days,
                annual_dividend_yield=annual_dividend_yield,
                price=price
            )
            dividend_cost = quantity * estimated_dividend

        # Combine costs
        return BorrowCostResult(
            daily_borrow_rate=borrow_result.daily_borrow_rate,
            daily_borrow_cost=borrow_result.daily_borrow_cost,
            total_borrow_cost=borrow_result.total_borrow_cost,
            availability=borrow_result.availability,
            recall_risk=borrow_result.recall_risk,
            dividend_obligations=dividend_cost
        )

"""
Commission Models

Implements realistic broker commission structures for backtesting:
- Fixed per-trade commissions
- Percentage-based commissions
- Tiered commission structures (volume discounts)
- Maker-taker models (rebates for providing liquidity)
- Currency conversion costs
- Exchange fees (per exchange)
- Regulatory fees (SEC, FINRA)
- Multiple broker models (Interactive Brokers, TD Ameritrade, etc.)

These models ensure realistic transaction cost modeling for accurate backtesting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
import numpy as np
from loguru import logger


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class LiquidityType(Enum):
    """Liquidity type for maker-taker models."""
    MAKER = "MAKER"  # Add liquidity (limit order)
    TAKER = "TAKER"  # Remove liquidity (market order)


class AssetClass(Enum):
    """Asset class for commission calculation."""
    STOCK = "STOCK"
    ETF = "ETF"
    OPTION = "OPTION"
    FUTURES = "FUTURES"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"


@dataclass
class CommissionResult:
    """
    Result of commission calculation.

    Attributes:
        total_commission: Total commission paid
        base_commission: Base broker commission
        exchange_fee: Exchange fees
        regulatory_fee: Regulatory fees (SEC, FINRA, etc.)
        currency_conversion_fee: Currency conversion costs
        liquidity_rebate: Rebate for providing liquidity (negative cost)
        breakdown: Detailed breakdown of all fees
    """
    total_commission: float
    base_commission: float
    exchange_fee: float = 0.0
    regulatory_fee: float = 0.0
    currency_conversion_fee: float = 0.0
    liquidity_rebate: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate total and populate breakdown."""
        self.total_commission = (
            self.base_commission +
            self.exchange_fee +
            self.regulatory_fee +
            self.currency_conversion_fee -
            self.liquidity_rebate
        )

        self.breakdown = {
            'base_commission': self.base_commission,
            'exchange_fee': self.exchange_fee,
            'regulatory_fee': self.regulatory_fee,
            'currency_conversion_fee': self.currency_conversion_fee,
            'liquidity_rebate': -self.liquidity_rebate,
            'total': self.total_commission
        }


class CommissionModel(ABC):
    """Base class for commission models."""

    @abstractmethod
    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        **kwargs
    ) -> CommissionResult:
        """
        Calculate commission for a trade.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Trade quantity
            price: Execution price
            asset_class: Asset class
            **kwargs: Additional parameters

        Returns:
            CommissionResult object
        """
        pass


class FixedCommissionModel(CommissionModel):
    """
    Fixed per-trade commission model.

    Simple model where each trade incurs a fixed cost.
    Common for discount brokers with flat pricing.
    """

    def __init__(
        self,
        commission_per_trade: float = 5.0,
        min_commission: float = 1.0,
        max_commission: float = None
    ):
        """
        Initialize fixed commission model.

        Args:
            commission_per_trade: Fixed cost per trade
            min_commission: Minimum commission
            max_commission: Maximum commission (optional cap)
        """
        self.commission_per_trade = commission_per_trade
        self.min_commission = min_commission
        self.max_commission = max_commission

        logger.info(
            f"Initialized FixedCommissionModel: "
            f"${commission_per_trade} per trade"
        )

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        **kwargs
    ) -> CommissionResult:
        """Calculate fixed commission."""
        commission = self.commission_per_trade

        # Apply bounds
        if self.min_commission:
            commission = max(commission, self.min_commission)
        if self.max_commission:
            commission = min(commission, self.max_commission)

        return CommissionResult(
            total_commission=commission,
            base_commission=commission
        )


class PercentageCommissionModel(CommissionModel):
    """
    Percentage-based commission model.

    Commission as a percentage of trade value.
    Common for mutual funds and some international brokers.
    """

    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% = 10 bps
        min_commission: float = 1.0,
        max_commission: float = None
    ):
        """
        Initialize percentage commission model.

        Args:
            commission_rate: Commission as decimal (0.001 = 0.1%)
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.max_commission = max_commission

        logger.info(
            f"Initialized PercentageCommissionModel: "
            f"{commission_rate*100:.3f}% per trade"
        )

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        **kwargs
    ) -> CommissionResult:
        """Calculate percentage-based commission."""
        trade_value = quantity * price
        commission = trade_value * self.commission_rate

        # Apply bounds
        if self.min_commission:
            commission = max(commission, self.min_commission)
        if self.max_commission:
            commission = min(commission, self.max_commission)

        return CommissionResult(
            total_commission=commission,
            base_commission=commission
        )


class PerShareCommissionModel(CommissionModel):
    """
    Per-share commission model.

    Commission based on number of shares traded.
    Common for active trading accounts.
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        max_commission: float = None
    ):
        """
        Initialize per-share commission model.

        Args:
            commission_per_share: Cost per share
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade
        """
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission = max_commission

        logger.info(
            f"Initialized PerShareCommissionModel: "
            f"${commission_per_share} per share"
        )

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        **kwargs
    ) -> CommissionResult:
        """Calculate per-share commission."""
        commission = quantity * self.commission_per_share

        # Apply bounds
        if self.min_commission:
            commission = max(commission, self.min_commission)
        if self.max_commission:
            commission = min(commission, self.max_commission)

        return CommissionResult(
            total_commission=commission,
            base_commission=commission
        )


class TieredCommissionModel(CommissionModel):
    """
    Tiered commission structure with volume discounts.

    Lower per-share rates for higher monthly volume.
    Common for professional trading accounts.
    """

    @dataclass
    class Tier:
        """Commission tier definition."""
        volume_threshold: float  # Monthly volume threshold
        commission_per_share: float

    def __init__(
        self,
        tiers: Optional[List[Tier]] = None,
        min_commission: float = 0.35
    ):
        """
        Initialize tiered commission model.

        Args:
            tiers: List of commission tiers (sorted by threshold)
            min_commission: Minimum commission per trade
        """
        # Default tiers (example: Interactive Brokers-style)
        if tiers is None:
            self.tiers = [
                self.Tier(volume_threshold=0, commission_per_share=0.005),
                self.Tier(volume_threshold=300000, commission_per_share=0.004),
                self.Tier(volume_threshold=3000000, commission_per_share=0.003),
                self.Tier(volume_threshold=20000000, commission_per_share=0.002),
                self.Tier(volume_threshold=100000000, commission_per_share=0.0015),
            ]
        else:
            self.tiers = sorted(tiers, key=lambda t: t.volume_threshold)

        self.min_commission = min_commission
        self.monthly_volume = 0  # Track monthly volume
        self.month_start = datetime.now().replace(day=1)

        logger.info(f"Initialized TieredCommissionModel with {len(self.tiers)} tiers")

    def update_monthly_volume(self, timestamp: datetime, shares: float):
        """
        Update monthly volume tracking.

        Args:
            timestamp: Trade timestamp
            shares: Number of shares traded
        """
        # Reset if new month
        if timestamp.month != self.month_start.month:
            self.monthly_volume = 0
            self.month_start = timestamp.replace(day=1)

        self.monthly_volume += shares

    def get_current_tier(self) -> Tier:
        """Get current commission tier based on monthly volume."""
        for tier in reversed(self.tiers):
            if self.monthly_volume >= tier.volume_threshold:
                return tier
        return self.tiers[0]

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> CommissionResult:
        """Calculate tiered commission."""
        # Update volume tracking
        if timestamp:
            self.update_monthly_volume(timestamp, quantity)

        # Get applicable tier
        tier = self.get_current_tier()

        # Calculate commission
        commission = quantity * tier.commission_per_share
        commission = max(commission, self.min_commission)

        return CommissionResult(
            total_commission=commission,
            base_commission=commission
        )


class MakerTakerCommissionModel(CommissionModel):
    """
    Maker-taker commission model.

    Different rates (or rebates) for adding vs. removing liquidity.
    Common in electronic trading and exchanges.

    Makers (limit orders) often receive rebates.
    Takers (market orders) pay fees.
    """

    def __init__(
        self,
        maker_rate: float = -0.0002,  # Negative = rebate
        taker_rate: float = 0.0003,
        min_commission: float = 0.0
    ):
        """
        Initialize maker-taker commission model.

        Args:
            maker_rate: Rate for makers (negative = rebate)
            taker_rate: Rate for takers
            min_commission: Minimum commission
        """
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
        self.min_commission = min_commission

        logger.info(
            f"Initialized MakerTakerCommissionModel: "
            f"maker={maker_rate*10000:.1f}bps, taker={taker_rate*10000:.1f}bps"
        )

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        liquidity_type: LiquidityType = LiquidityType.TAKER,
        **kwargs
    ) -> CommissionResult:
        """Calculate maker-taker commission."""
        trade_value = quantity * price

        if liquidity_type == LiquidityType.MAKER:
            rate = self.maker_rate
            rebate = abs(trade_value * rate) if rate < 0 else 0
            commission = trade_value * rate if rate > 0 else 0
        else:
            rate = self.taker_rate
            commission = trade_value * rate
            rebate = 0

        commission = max(commission, self.min_commission)

        return CommissionResult(
            total_commission=commission,
            base_commission=commission,
            liquidity_rebate=rebate
        )


class InteractiveBrokersCommissionModel(CommissionModel):
    """
    Interactive Brokers commission model.

    Implements IB's actual commission structure:
    - Tiered pricing
    - Fixed pricing
    - Asset-specific rates
    - Exchange fees
    - Regulatory fees
    """

    def __init__(self, pricing_plan: str = "tiered"):
        """
        Initialize IB commission model.

        Args:
            pricing_plan: "tiered" or "fixed"
        """
        self.pricing_plan = pricing_plan

        # Tiered pricing (USD stocks)
        self.tiered_rates = [
            {'threshold': 0, 'rate': 0.0035, 'min': 0.35, 'max': 1.0},
            {'threshold': 300000, 'rate': 0.0020, 'min': 0.35, 'max': 1.0},
            {'threshold': 3000000, 'rate': 0.0015, 'min': 0.35, 'max': 1.0},
            {'threshold': 20000000, 'rate': 0.0010, 'min': 0.35, 'max': 1.0},
            {'threshold': 100000000, 'rate': 0.0005, 'min': 0.35, 'max': 1.0},
        ]

        # Fixed pricing
        self.fixed_rate = 0.005  # $0.005 per share
        self.fixed_min = 1.0
        self.fixed_max = 1.0  # Max 1% of trade value

        self.monthly_volume = 0

        logger.info(f"Initialized InteractiveBrokersCommissionModel: {pricing_plan}")

    def calculate_sec_fee(self, trade_value: float, side: OrderSide) -> float:
        """
        Calculate SEC fee (Section 31).

        Only applies to sales of exchange-listed securities.
        Rate: $27.80 per $1,000,000 (as of 2024).

        Args:
            trade_value: Trade value in USD
            side: Order side

        Returns:
            SEC fee amount
        """
        if side == OrderSide.SELL:
            sec_rate = 27.80 / 1_000_000  # Per dollar
            return trade_value * sec_rate
        return 0.0

    def calculate_finra_fee(self, quantity: float, side: OrderSide) -> float:
        """
        Calculate FINRA Trading Activity Fee (TAF).

        Only applies to sales of covered securities.
        Rate: $0.000166 per share (as of 2024), max $8.30.

        Args:
            quantity: Number of shares
            side: Order side

        Returns:
            FINRA TAF amount
        """
        if side == OrderSide.SELL:
            taf_rate = 0.000166
            return min(quantity * taf_rate, 8.30)
        return 0.0

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        **kwargs
    ) -> CommissionResult:
        """Calculate IB commission with all fees."""
        trade_value = quantity * price

        # Base commission
        if self.pricing_plan == "tiered":
            # Find applicable tier
            rate_info = self.tiered_rates[0]
            for tier in reversed(self.tiered_rates):
                if self.monthly_volume >= tier['threshold']:
                    rate_info = tier
                    break

            base_commission = quantity * rate_info['rate']
            base_commission = np.clip(
                base_commission,
                rate_info['min'],
                min(rate_info['max'], trade_value * 0.01)
            )
        else:  # Fixed
            base_commission = quantity * self.fixed_rate
            base_commission = np.clip(
                base_commission,
                self.fixed_min,
                min(self.fixed_max, trade_value * 0.01)
            )

        # Regulatory fees
        sec_fee = self.calculate_sec_fee(trade_value, side)
        finra_fee = self.calculate_finra_fee(quantity, side)

        return CommissionResult(
            total_commission=base_commission + sec_fee + finra_fee,
            base_commission=base_commission,
            regulatory_fee=sec_fee + finra_fee
        )


class TDAmeritrade CommissionModel(CommissionModel):
    """
    TD Ameritrade commission model.

    - $0 commissions for online stock/ETF trades
    - Options: $0 + $0.65 per contract
    """

    def __init__(self):
        """Initialize TD Ameritrade commission model."""
        self.stock_commission = 0.0
        self.option_commission_per_contract = 0.65

        logger.info("Initialized TDAmeritrade CommissionModel")

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        **kwargs
    ) -> CommissionResult:
        """Calculate TD Ameritrade commission."""
        if asset_class == AssetClass.OPTION:
            base_commission = quantity * self.option_commission_per_contract
        else:
            base_commission = self.stock_commission

        return CommissionResult(
            total_commission=base_commission,
            base_commission=base_commission
        )


class CurrencyConversionModel:
    """
    Currency conversion cost model.

    Models the cost of converting between currencies,
    important for international trading.
    """

    def __init__(
        self,
        conversion_spread_bps: float = 10.0,  # 10 bps typical
        min_fee: float = 2.0
    ):
        """
        Initialize currency conversion model.

        Args:
            conversion_spread_bps: Spread in basis points
            min_fee: Minimum conversion fee
        """
        self.conversion_spread_bps = conversion_spread_bps
        self.min_fee = min_fee

        logger.info(
            f"Initialized CurrencyConversionModel: "
            f"{conversion_spread_bps} bps"
        )

    def calculate_conversion_cost(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        exchange_rate: float
    ) -> float:
        """
        Calculate currency conversion cost.

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            exchange_rate: Current exchange rate

        Returns:
            Conversion cost in target currency
        """
        if from_currency == to_currency:
            return 0.0

        spread_pct = self.conversion_spread_bps / 10000.0
        cost = amount * spread_pct

        return max(cost, self.min_fee)


class ExchangeFeeModel:
    """
    Exchange-specific fee model.

    Different exchanges charge different fees.
    Important for international trading and multi-venue execution.
    """

    def __init__(self):
        """Initialize exchange fee model."""
        # Fee structures by exchange (example rates)
        self.exchange_fees = {
            'NYSE': {'maker': -0.0003, 'taker': 0.0005},  # Rebate for makers
            'NASDAQ': {'maker': -0.0002, 'taker': 0.0005},
            'ARCA': {'maker': -0.0002, 'taker': 0.0003},
            'ASX': {'maker': 0.0, 'taker': 0.0001},  # Australia
            'LSE': {'maker': 0.0, 'taker': 0.0005},  # London
            'TSE': {'maker': 0.0, 'taker': 0.0004},  # Tokyo
        }

        logger.info("Initialized ExchangeFeeModel")

    def calculate_exchange_fee(
        self,
        exchange: str,
        liquidity_type: LiquidityType,
        trade_value: float
    ) -> Dict[str, float]:
        """
        Calculate exchange fee.

        Args:
            exchange: Exchange code
            liquidity_type: MAKER or TAKER
            trade_value: Trade value

        Returns:
            Dictionary with fee and rebate
        """
        if exchange not in self.exchange_fees:
            logger.warning(f"Unknown exchange: {exchange}, using default fees")
            exchange = 'NYSE'

        fees = self.exchange_fees[exchange]
        rate_key = 'maker' if liquidity_type == LiquidityType.MAKER else 'taker'
        rate = fees[rate_key]

        fee_amount = trade_value * abs(rate)

        if rate < 0:  # Rebate
            return {'fee': 0.0, 'rebate': fee_amount}
        else:
            return {'fee': fee_amount, 'rebate': 0.0}


class ComprehensiveCommissionModel(CommissionModel):
    """
    Comprehensive commission model combining all cost components.

    Includes:
    - Base commission
    - Exchange fees
    - Regulatory fees
    - Currency conversion
    - Maker-taker rebates
    """

    def __init__(
        self,
        base_model: CommissionModel,
        exchange_fee_model: Optional[ExchangeFeeModel] = None,
        currency_conversion_model: Optional[CurrencyConversionModel] = None,
        include_regulatory_fees: bool = True
    ):
        """
        Initialize comprehensive commission model.

        Args:
            base_model: Base commission model
            exchange_fee_model: Exchange fee model
            currency_conversion_model: Currency conversion model
            include_regulatory_fees: Whether to include SEC/FINRA fees
        """
        self.base_model = base_model
        self.exchange_fee_model = exchange_fee_model or ExchangeFeeModel()
        self.currency_conversion_model = currency_conversion_model or CurrencyConversionModel()
        self.include_regulatory_fees = include_regulatory_fees

        logger.info("Initialized ComprehensiveCommissionModel")

    def calculate_commission(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.STOCK,
        exchange: str = 'NYSE',
        liquidity_type: LiquidityType = LiquidityType.TAKER,
        currency: str = 'USD',
        target_currency: str = 'USD',
        exchange_rate: float = 1.0,
        **kwargs
    ) -> CommissionResult:
        """Calculate comprehensive commission with all fees."""
        # Base commission
        base_result = self.base_model.calculate_commission(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            asset_class=asset_class,
            **kwargs
        )

        trade_value = quantity * price

        # Exchange fees
        exchange_fees = self.exchange_fee_model.calculate_exchange_fee(
            exchange=exchange,
            liquidity_type=liquidity_type,
            trade_value=trade_value
        )

        # Regulatory fees
        regulatory_fee = 0.0
        if self.include_regulatory_fees and asset_class == AssetClass.STOCK:
            # SEC fee (sells only)
            if side == OrderSide.SELL:
                sec_rate = 27.80 / 1_000_000
                sec_fee = trade_value * sec_rate

                # FINRA TAF (sells only)
                taf_rate = 0.000166
                finra_fee = min(quantity * taf_rate, 8.30)

                regulatory_fee = sec_fee + finra_fee

        # Currency conversion
        currency_fee = 0.0
        if currency != target_currency:
            currency_fee = self.currency_conversion_model.calculate_conversion_cost(
                amount=trade_value,
                from_currency=currency,
                to_currency=target_currency,
                exchange_rate=exchange_rate
            )

        return CommissionResult(
            total_commission=(
                base_result.base_commission +
                exchange_fees['fee'] +
                regulatory_fee +
                currency_fee -
                exchange_fees['rebate']
            ),
            base_commission=base_result.base_commission,
            exchange_fee=exchange_fees['fee'],
            regulatory_fee=regulatory_fee,
            currency_conversion_fee=currency_fee,
            liquidity_rebate=exchange_fees['rebate']
        )

"""
Costs Module

Comprehensive transaction cost modeling for backtesting.

This module provides realistic cost models including:
- Commission structures (fixed, percentage, tiered, maker-taker)
- Bid-ask spread costs (fixed, time-varying, volume-dependent, volatility-based)
- Short selling costs (borrow fees, dividend obligations)
- Funding costs (margin interest, overnight financing, currency carry)
- Tax impact (capital gains, wash sales, lot tracking)

Accurate cost modeling is critical for realistic backtest results.
Ignoring or underestimating costs is a common source of strategy failure in live trading.
"""

# Commission Models
from .commission_models import (
    OrderSide,
    LiquidityType,
    AssetClass,
    CommissionResult,
    CommissionModel,
    FixedCommissionModel,
    PercentageCommissionModel,
    PerShareCommissionModel,
    TieredCommissionModel,
    MakerTakerCommissionModel,
    InteractiveBrokersCommissionModel,
    TDAmeritrade CommissionModel,
    CurrencyConversionModel,
    ExchangeFeeModel,
    ComprehensiveCommissionModel
)

# Spread Models
from .spread_models import (
    SpreadType,
    SpreadResult,
    FixedSpreadModel,
    TimeVaryingSpreadModel,
    VolumeDependentSpreadModel,
    VolatilityBasedSpreadModel,
    DynamicSpreadModel,
    ImplementationShortfallModel,
    EffectiveSpreadAnalyzer
)

# Borrow Costs
from .borrow_costs import (
    BorrowAvailability,
    RecallRisk,
    BorrowCostResult,
    GeneralCollateralModel,
    DemandBasedBorrowModel,
    TieredBorrowRateModel,
    DividendObligationModel,
    TermStructureBorrowModel,
    ComprehensiveBorrowCostModel
)

# Funding Costs
from .funding_costs import (
    AccountType,
    PositionType,
    FundingCostResult,
    MarginInterestModel,
    OvernightFundingModel,
    CurrencyCarryModel,
    LeverageCostModel,
    CapitalAllocationModel,
    ComprehensiveFundingModel
)

# Tax Models
from .tax_models import (
    TaxTreatment,
    LotMatchingMethod,
    TaxLot,
    TaxResult,
    CapitalGainsTaxModel,
    WashSaleModel,
    LotMatchingEngine,
    Section475MTMModel,
    TaxLossHarvestingAnalyzer,
    ComprehensiveTaxModel
)

__all__ = [
    # Commission Models
    'OrderSide',
    'LiquidityType',
    'AssetClass',
    'CommissionResult',
    'CommissionModel',
    'FixedCommissionModel',
    'PercentageCommissionModel',
    'PerShareCommissionModel',
    'TieredCommissionModel',
    'MakerTakerCommissionModel',
    'InteractiveBrokersCommissionModel',
    'TDAmeritrade CommissionModel',
    'CurrencyConversionModel',
    'ExchangeFeeModel',
    'ComprehensiveCommissionModel',

    # Spread Models
    'SpreadType',
    'SpreadResult',
    'FixedSpreadModel',
    'TimeVaryingSpreadModel',
    'VolumeDependentSpreadModel',
    'VolatilityBasedSpreadModel',
    'DynamicSpreadModel',
    'ImplementationShortfallModel',
    'EffectiveSpreadAnalyzer',

    # Borrow Costs
    'BorrowAvailability',
    'RecallRisk',
    'BorrowCostResult',
    'GeneralCollateralModel',
    'DemandBasedBorrowModel',
    'TieredBorrowRateModel',
    'DividendObligationModel',
    'TermStructureBorrowModel',
    'ComprehensiveBorrowCostModel',

    # Funding Costs
    'AccountType',
    'PositionType',
    'FundingCostResult',
    'MarginInterestModel',
    'OvernightFundingModel',
    'CurrencyCarryModel',
    'LeverageCostModel',
    'CapitalAllocationModel',
    'ComprehensiveFundingModel',

    # Tax Models
    'TaxTreatment',
    'LotMatchingMethod',
    'TaxLot',
    'TaxResult',
    'CapitalGainsTaxModel',
    'WashSaleModel',
    'LotMatchingEngine',
    'Section475MTMModel',
    'TaxLossHarvestingAnalyzer',
    'ComprehensiveTaxModel',
]

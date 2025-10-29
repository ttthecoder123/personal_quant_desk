"""
Market Simulation Module

Provides realistic market simulation including:
- Order book dynamics
- Market impact modeling
- Slippage simulation
- Fill probability and execution
- Corporate actions handling
"""

from .order_book_simulator import OrderBookSimulator, OrderBook
from .market_impact_model import MarketImpactModel, AlmgrenChrissModel
from .slippage_model import SlippageModel, DynamicSlippageModel
from .fill_simulator import FillSimulator, FillProbability
from .corporate_actions import CorporateActionHandler, Split, Dividend

__all__ = [
    'OrderBookSimulator',
    'OrderBook',
    'MarketImpactModel',
    'AlmgrenChrissModel',
    'SlippageModel',
    'DynamicSlippageModel',
    'FillSimulator',
    'FillProbability',
    'CorporateActionHandler',
    'Split',
    'Dividend',
]

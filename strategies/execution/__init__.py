"""
Execution layer for systematic trading.

This module provides comprehensive order execution capabilities including:
- Order generation from strategy signals
- Execution algorithms (TWAP, VWAP, etc.)
- Slippage modeling
- Transaction cost modeling
"""

from strategies.execution.order_generator import OrderGenerator
from strategies.execution.execution_algo import ExecutionAlgorithm
from strategies.execution.slippage_model import SlippageModel
from strategies.execution.cost_model import CostModel

__all__ = [
    'OrderGenerator',
    'ExecutionAlgorithm',
    'SlippageModel',
    'CostModel',
]

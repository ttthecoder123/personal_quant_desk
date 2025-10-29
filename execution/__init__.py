"""
Execution and Order Management System (OMS) for Personal Quant Desk

This module provides institutional-grade execution infrastructure including:
- Order management and lifecycle tracking
- Smart order routing across venues
- Execution algorithms (TWAP, VWAP, IS, Adaptive)
- Broker connectivity (IB, Alpaca, FIX)
- Pre-trade and post-trade analytics
- Real-time monitoring and alerting
- State management and recovery

References:
- Chan's execution strategies
- Kissell's transaction cost analysis
- Almgren-Chriss optimal execution
"""

from .execution_engine import ExecutionEngine

__all__ = ['ExecutionEngine']

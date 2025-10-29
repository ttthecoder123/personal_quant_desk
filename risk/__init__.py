"""
Risk Management System for Personal Quant Desk

Comprehensive institutional-grade risk management implementing:
- Carver's systematic risk framework
- Lopez de Prado's risk metrics
- VaR/CVaR models
- Position sizing and volatility targeting
- Drawdown control
- Stress testing
"""

# Core risk manager
from .risk_manager import RiskManager

# Portfolio risk modules
from .portfolio_risk import (
    CorrelationRisk,
    ConcentrationRisk,
    LiquidityRisk,
    TailRisk
)

__all__ = [
    'RiskManager',
    'CorrelationRisk',
    'ConcentrationRisk',
    'LiquidityRisk',
    'TailRisk'
]

__version__ = '1.0.0'

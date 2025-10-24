"""
Volatility Trading Strategies

This module contains volatility-based trading strategies including:
- Volatility targeting (Carver's approach)
- Volatility arbitrage (implied vs realized)
- Gamma scalping
- Dispersion trading
"""

from strategies.volatility.vol_targeting import VolatilityTargetingStrategy
from strategies.volatility.vol_arbitrage import VolArbitrageStrategy
from strategies.volatility.gamma_scalping import GammaScalpingStrategy
from strategies.volatility.dispersion_trading import DispersionTradingStrategy

__all__ = [
    'VolatilityTargetingStrategy',
    'VolArbitrageStrategy',
    'GammaScalpingStrategy',
    'DispersionTradingStrategy'
]

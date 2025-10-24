"""
Momentum Strategy Module

This module contains various momentum-based trading strategies:
- TrendFollowingStrategy: Carver's trend following with multi-timeframe analysis
- BreakoutMomentumStrategy: Donchian channel breakouts with volatility filters
- CrossSectionalMomentumStrategy: Relative momentum ranking across assets
- TimeSeriesMomentumStrategy: Sign-based momentum (Moskowitz et al.)
"""

from strategies.momentum.trend_following import TrendFollowingStrategy
from strategies.momentum.breakout_momentum import BreakoutMomentumStrategy
from strategies.momentum.cross_sectional import CrossSectionalMomentumStrategy
from strategies.momentum.time_series_momentum import TimeSeriesMomentumStrategy

__all__ = [
    'TrendFollowingStrategy',
    'BreakoutMomentumStrategy',
    'CrossSectionalMomentumStrategy',
    'TimeSeriesMomentumStrategy'
]

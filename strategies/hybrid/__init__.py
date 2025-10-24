"""
Hybrid Trading Strategies

This module contains hybrid strategies that combine multiple approaches:
- ML-enhanced discretionary trading
- Regime-switching strategies
- Multi-factor combination
- Strategy ensemble (meta-strategy)
"""

from strategies.hybrid.ml_enhanced import MLEnhancedStrategy
from strategies.hybrid.regime_switching import RegimeSwitchingStrategy
from strategies.hybrid.multi_factor import MultiFactorStrategy
from strategies.hybrid.ensemble_strategy import EnsembleStrategy

__all__ = [
    'MLEnhancedStrategy',
    'RegimeSwitchingStrategy',
    'MultiFactorStrategy',
    'EnsembleStrategy'
]

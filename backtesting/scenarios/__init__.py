"""
Scenario testing module for backtesting system.

This module provides comprehensive scenario testing capabilities:
- Historical crisis scenarios
- Synthetic scenario generation
- Extreme stress testing
- Regime-specific scenarios
"""

from .historical_scenarios import (
    HistoricalScenarioTester,
    CrisisScenario,
    CrisisType,
)
from .synthetic_scenarios import (
    SyntheticScenarioGenerator,
    BlockBootstrap,
    GARCHSimulator,
    VineCopulaGenerator,
)
from .stress_scenarios import (
    StressTestEngine,
    CorrelationBreakdown,
    VolatilityExplosion,
    LiquidityEvaporation,
)
from .regime_scenarios import (
    RegimeScenarioGenerator,
    MarketRegime,
    RegimeType,
)

__all__ = [
    # Historical scenarios
    'HistoricalScenarioTester',
    'CrisisScenario',
    'CrisisType',
    # Synthetic scenarios
    'SyntheticScenarioGenerator',
    'BlockBootstrap',
    'GARCHSimulator',
    'VineCopulaGenerator',
    # Stress scenarios
    'StressTestEngine',
    'CorrelationBreakdown',
    'VolatilityExplosion',
    'LiquidityEvaporation',
    # Regime scenarios
    'RegimeScenarioGenerator',
    'MarketRegime',
    'RegimeType',
]

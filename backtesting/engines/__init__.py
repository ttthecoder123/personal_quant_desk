"""
Backtesting Engines Module

Provides different backtesting engine implementations:
- EventEngine: Event-driven simulation (most realistic)
- VectorizedEngine: Fast vectorized backtesting
- SimulationEngine: Monte Carlo simulations
- WalkForwardEngine: Walk-forward analysis
"""

from .event_engine import EventEngine, Event, EventType
from .vectorized_engine import VectorizedEngine
from .simulation_engine import SimulationEngine, MonteCarloSimulator
from .walk_forward_engine import WalkForwardEngine, WalkForwardWindow

__all__ = [
    'EventEngine',
    'Event',
    'EventType',
    'VectorizedEngine',
    'SimulationEngine',
    'MonteCarloSimulator',
    'WalkForwardEngine',
    'WalkForwardWindow',
]

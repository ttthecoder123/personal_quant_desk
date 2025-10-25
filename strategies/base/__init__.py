"""
Base Strategy Framework

Core infrastructure for strategy development including:
- Abstract base classes
- Position management
- Performance tracking
"""

from strategies.base.strategy_base import StrategyBase, SignalType, PositionSide
from strategies.base.position_manager import PositionManager, Position
from strategies.base.performance_tracker import PerformanceTracker, PerformanceMetrics

__all__ = [
    "StrategyBase",
    "SignalType",
    "PositionSide",
    "PositionManager",
    "Position",
    "PerformanceTracker",
    "PerformanceMetrics",
]

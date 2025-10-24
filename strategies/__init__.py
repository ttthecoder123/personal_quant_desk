"""
Personal Quant Desk - Strategy Development System (Step 5)

Comprehensive strategy development and portfolio construction system implementing
strategies from Chan, Carver, and Jansen.

Architecture:
    - base: Core infrastructure (StrategyBase, PositionManager, PerformanceTracker)
    - mean_reversion: Mean reversion strategies (pairs, Bollinger, OU process, index arb)
    - momentum: Momentum strategies (trend following, breakouts, cross-sectional)
    - volatility: Volatility-based strategies (targeting, arbitrage, gamma, dispersion)
    - hybrid: Hybrid strategies (ML-enhanced, regime switching, multi-factor, ensemble)
    - portfolio: Portfolio construction (optimizer, risk parity, Kelly, correlation, rebalancing)
    - execution: Execution layer (order generation, algos, slippage, costs)
    - strategy_engine: Main orchestration system

Integration:
    - Step 2: Data ingestion from ParquetStorage
    - Step 3: Engineered features for signal generation
    - Step 4: ML signals with meta-labels and triple-barrier exits
    - Step 7: Risk management (future)
    - Step 10: Backtesting (future)
"""

__version__ = "1.0.0"
__author__ = "Personal Quant Desk"

# Core components
from strategies.base.strategy_base import StrategyBase, StrategySignal, SignalType, PositionSide
from strategies.base.position_manager import PositionManager, Position, PositionStatus
from strategies.base.performance_tracker import PerformanceTracker, PerformanceMetrics

# Main engine
from strategies.strategy_engine import StrategyEngine

# Mean reversion strategies
from strategies.mean_reversion import (
    PairsTradingStrategy,
    BollingerReversionStrategy,
    OUProcessStrategy,
    IndexArbitrageStrategy
)

# Momentum strategies
from strategies.momentum import (
    TrendFollowingStrategy,
    BreakoutMomentumStrategy,
    CrossSectionalMomentumStrategy,
    TimeSeriesMomentumStrategy
)

# Volatility strategies
from strategies.volatility import (
    VolatilityTargetingStrategy,
    VolArbitrageStrategy,
    GammaScalpingStrategy,
    DispersionTradingStrategy
)

# Hybrid strategies
from strategies.hybrid import (
    MLEnhancedStrategy,
    RegimeSwitchingStrategy,
    MultiFactorStrategy,
    EnsembleStrategy
)

# Portfolio construction
from strategies.portfolio import (
    PortfolioOptimizer,
    RiskParityAllocator,
    KellySizer,
    CorrelationManager,
    Rebalancer
)

# Execution
from strategies.execution import (
    OrderGenerator,
    ExecutionAlgorithm,
    SlippageModel,
    CostModel
)

__all__ = [
    # Core
    "StrategyBase",
    "StrategySignal",
    "SignalType",
    "PositionSide",
    "PositionManager",
    "Position",
    "PositionStatus",
    "PerformanceTracker",
    "PerformanceMetrics",
    "StrategyEngine",
    # Mean Reversion
    "PairsTradingStrategy",
    "BollingerReversionStrategy",
    "OUProcessStrategy",
    "IndexArbitrageStrategy",
    # Momentum
    "TrendFollowingStrategy",
    "BreakoutMomentumStrategy",
    "CrossSectionalMomentumStrategy",
    "TimeSeriesMomentumStrategy",
    # Volatility
    "VolatilityTargetingStrategy",
    "VolArbitrageStrategy",
    "GammaScalpingStrategy",
    "DispersionTradingStrategy",
    # Hybrid
    "MLEnhancedStrategy",
    "RegimeSwitchingStrategy",
    "MultiFactorStrategy",
    "EnsembleStrategy",
    # Portfolio
    "PortfolioOptimizer",
    "RiskParityAllocator",
    "KellySizer",
    "CorrelationManager",
    "Rebalancer",
    # Execution
    "OrderGenerator",
    "ExecutionAlgorithm",
    "SlippageModel",
    "CostModel",
]

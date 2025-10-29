"""
Backtesting Module - Step 7

Comprehensive backtesting engine and strategy validation system implementing:
- Event-driven simulation with realistic market conditions
- López de Prado's backtesting practices and validation frameworks
- Jansen's ML backtesting methodologies
- Carver's performance metrics and optimization

Components:
- Engines: Event-driven, vectorized, simulation, walk-forward
- Market Simulation: Order books, market impact, slippage, fills
- Validation: Statistical tests, overfitting detection, regime analysis
- Performance: Comprehensive metrics, tear sheets, attribution
- Optimization: Parameter optimization, genetic algorithms, CPCV
- Cost Modeling: Commissions, spreads, borrow costs, funding, taxes
- Data Handling: Loading, alignment, survivorship bias, quality checks
- Reporting: HTML/PDF reports, visualizations, comparisons
- Scenarios: Historical crises, synthetic data, stress tests
"""

# Main orchestrator
from .backtest_orchestrator import BacktestOrchestrator, run_quick_backtest, run_full_validation

# Engines
from .engines import (
    EventEngine,
    VectorizedEngine,
    SimulationEngine,
    WalkForwardEngine,
    MonteCarloSimulator
)

# Legacy backtest engine (from Step 7 starter)
from .backtest_engine import BacktestEngine, BacktestResults

__version__ = "1.0.0"

__all__ = [
    # Main API
    'BacktestOrchestrator',
    'run_quick_backtest',
    'run_full_validation',

    # Engines
    'EventEngine',
    'VectorizedEngine',
    'SimulationEngine',
    'WalkForwardEngine',
    'MonteCarloSimulator',

    # Legacy
    'BacktestEngine',
    'BacktestResults',
]

# Module metadata
__author__ = "Personal Quant Desk Team"
__description__ = "Institutional-grade backtesting engine for quantitative trading strategies"
__license__ = "MIT"

"""
Reporting module for backtesting system.

This module provides comprehensive reporting capabilities including:
- Backtest reports (HTML, PDF, JSON)
- Trade analysis
- Strategy comparison
- Optimization reports
- Interactive visualizations
"""

from .backtest_report import (
    BacktestReportGenerator,
    ReportFormat,
    ReportSection,
)
from .trade_analysis import (
    TradeAnalyzer,
    TradeStatistics,
    TradeClustering,
)
from .comparison_reports import (
    StrategyComparator,
    ComparisonReport,
    EfficientFrontier,
)
from .optimization_reports import (
    OptimizationReporter,
    ParameterSpaceVisualizer,
    WalkForwardAnalyzer,
)
from .visual_analytics import (
    InteractiveVisualizer,
    PerformanceDashboard,
    ParameterExplorer,
)

__all__ = [
    # Backtest reports
    'BacktestReportGenerator',
    'ReportFormat',
    'ReportSection',
    # Trade analysis
    'TradeAnalyzer',
    'TradeStatistics',
    'TradeClustering',
    # Strategy comparison
    'StrategyComparator',
    'ComparisonReport',
    'EfficientFrontier',
    # Optimization reports
    'OptimizationReporter',
    'ParameterSpaceVisualizer',
    'WalkForwardAnalyzer',
    # Interactive visualizations
    'InteractiveVisualizer',
    'PerformanceDashboard',
    'ParameterExplorer',
]

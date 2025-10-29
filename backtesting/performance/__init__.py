"""
Performance Analytics Module for Backtesting System.

This module provides comprehensive performance analysis tools including:
- Metrics calculation (returns, risk, risk-adjusted, drawdowns)
- Risk metrics (VaR, CVaR, CDaR, tail risk)
- Tear sheet generation (pyfolio-style reports)
- Performance attribution analysis
- Benchmark comparison and analysis

Usage:
    from backtesting.performance import (
        MetricsCalculator,
        RiskMetrics,
        TearSheetGenerator,
        AttributionAnalysis,
        BenchmarkComparison,
        create_tear_sheet,
        create_attribution_report,
        create_benchmark_report,
    )

    # Calculate comprehensive metrics
    calc = MetricsCalculator(returns, benchmark_returns)
    metrics = calc.calculate_all_metrics()

    # Generate tear sheet
    tear_sheet_path = create_tear_sheet(
        returns=returns,
        benchmark_returns=benchmark_returns,
        trades=trades_df,
        output_path="reports/strategy_tearsheet.pdf"
    )

    # Perform attribution analysis
    attribution = AttributionAnalysis(
        portfolio_returns=returns,
        holdings=holdings_df,
        component_returns=component_returns_df
    )
    attr_results = attribution.calculate_full_attribution()

    # Compare against benchmark
    comparison = BenchmarkComparison(
        portfolio_returns=returns,
        benchmark_returns=benchmark_returns
    )
    comp_metrics = comparison.calculate_all_comparisons()
"""

# Import main classes
from backtesting.performance.metrics_calculator import (
    MetricsCalculator,
    calculate_trade_statistics,
)

from backtesting.performance.risk_metrics import (
    RiskMetrics,
    calculate_portfolio_var,
    calculate_marginal_var,
)

from backtesting.performance.tear_sheet_generator import (
    TearSheetGenerator,
    create_tear_sheet,
)

from backtesting.performance.attribution_analysis import (
    AttributionAnalysis,
    create_attribution_report,
)

from backtesting.performance.benchmark_comparison import (
    BenchmarkComparison,
    compare_multiple_benchmarks,
    create_benchmark_report,
)

# Define public API
__all__ = [
    # Main classes
    'MetricsCalculator',
    'RiskMetrics',
    'TearSheetGenerator',
    'AttributionAnalysis',
    'BenchmarkComparison',

    # Convenience functions
    'calculate_trade_statistics',
    'calculate_portfolio_var',
    'calculate_marginal_var',
    'create_tear_sheet',
    'create_attribution_report',
    'create_benchmark_report',
    'compare_multiple_benchmarks',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Personal Quant Desk'
__description__ = 'Comprehensive performance analytics for backtesting'

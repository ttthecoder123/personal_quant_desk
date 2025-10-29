"""
Example usage of the Performance Analytics Module.

This script demonstrates how to use all components of the performance
analytics system for comprehensive backtest analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import performance analytics modules
from backtesting.performance import (
    MetricsCalculator,
    RiskMetrics,
    TearSheetGenerator,
    AttributionAnalysis,
    BenchmarkComparison,
    create_tear_sheet,
    create_attribution_report,
    create_benchmark_report,
    calculate_trade_statistics,
)


def generate_sample_data():
    """Generate sample data for demonstration."""
    # Generate date range
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

    # Generate sample returns
    np.random.seed(42)
    portfolio_returns = pd.Series(
        np.random.normal(0.0005, 0.015, len(dates)),
        index=dates,
        name='portfolio_returns'
    )

    benchmark_returns = pd.Series(
        np.random.normal(0.0004, 0.012, len(dates)),
        index=dates,
        name='benchmark_returns'
    )

    # Generate sample trades
    n_trades = 100
    trade_dates = pd.to_datetime(np.random.choice(dates, n_trades))
    trades = pd.DataFrame({
        'date': trade_dates,
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], n_trades),
        'action': np.random.choice(['BUY', 'SELL'], n_trades),
        'quantity': np.random.randint(10, 100, n_trades),
        'price': np.random.uniform(100, 300, n_trades),
        'pnl': np.random.normal(500, 2000, n_trades),
    })
    trades.set_index('date', inplace=True)
    trades.sort_index(inplace=True)

    # Generate sample holdings
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    holdings = pd.DataFrame(
        np.random.dirichlet([1, 1, 1, 1], len(dates)),
        index=dates,
        columns=symbols
    )

    # Generate component returns
    component_returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (len(dates), len(symbols))),
        index=dates,
        columns=symbols
    )

    return portfolio_returns, benchmark_returns, trades, holdings, component_returns


def example_1_basic_metrics():
    """Example 1: Calculate basic performance metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Performance Metrics")
    print("="*80)

    portfolio_returns, benchmark_returns, _, _, _ = generate_sample_data()

    # Initialize calculator
    calc = MetricsCalculator(
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    # Calculate all metrics
    metrics = calc.calculate_all_metrics()

    # Print key metrics
    print(f"\nPortfolio Performance:")
    print(f"  Total Return:        {metrics['total_return']:.2%}")
    print(f"  CAGR:                {metrics['cagr']:.2%}")
    print(f"  Volatility:          {metrics['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:.2f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate:            {metrics['positive_periods']:.2%}")

    if 'alpha' in metrics:
        print(f"\nBenchmark Comparison:")
        print(f"  Alpha:               {metrics['alpha']:.2%}")
        print(f"  Beta:                {metrics['beta']:.2f}")
        print(f"  Information Ratio:   {metrics['information_ratio']:.2f}")
        print(f"  Correlation:         {metrics['correlation']:.2f}")


def example_2_risk_analysis():
    """Example 2: Advanced risk analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Advanced Risk Analysis")
    print("="*80)

    portfolio_returns, benchmark_returns, _, _, _ = generate_sample_data()

    # Initialize risk calculator
    risk_calc = RiskMetrics(
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        confidence_level=0.95
    )

    # Calculate risk metrics
    risk_metrics = risk_calc.calculate_all_risk_metrics()

    print(f"\nValue at Risk (95% confidence):")
    print(f"  Historical VaR:      {risk_metrics['var_historical']:.2%}")
    print(f"  Parametric VaR:      {risk_metrics['var_parametric']:.2%}")
    print(f"  Cornish-Fisher VaR:  {risk_metrics['var_cornish_fisher']:.2%}")

    print(f"\nConditional Value at Risk (CVaR):")
    print(f"  Historical CVaR:     {risk_metrics['cvar_historical']:.2%}")
    print(f"  Parametric CVaR:     {risk_metrics['cvar_parametric']:.2%}")

    print(f"\nDrawdown Risk:")
    print(f"  CDaR:                {risk_metrics['cdar']:.2%}")
    print(f"  Max Drawdown:        {risk_metrics['max_drawdown']:.2%}")

    print(f"\nTail Risk:")
    print(f"  Tail Ratio:          {risk_metrics['tail_ratio']:.2f}")
    print(f"  Pain Index:          {risk_metrics['pain_index']:.2%}")
    print(f"  Ulcer Index:         {risk_metrics['ulcer_index']:.2f}")


def example_3_tear_sheet():
    """Example 3: Generate tear sheet."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Tear Sheet Generation")
    print("="*80)

    portfolio_returns, benchmark_returns, trades, _, _ = generate_sample_data()

    print("\nGenerating comprehensive tear sheet...")

    # Note: In actual usage, this would generate a PDF/HTML file
    # For this example, we'll just initialize the generator
    generator = TearSheetGenerator(
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        trades=trades,
        title="Sample Strategy Performance"
    )

    print(f"  Tear sheet generator initialized")
    print(f"  Data period: {portfolio_returns.index[0]} to {portfolio_returns.index[-1]}")
    print(f"  Number of trades: {len(trades)}")
    print(f"  \nTo generate actual tear sheet, call:")
    print(f"    create_tear_sheet(returns, benchmark_returns, trades, 'output.pdf')")


def example_4_attribution():
    """Example 4: Performance attribution."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Performance Attribution")
    print("="*80)

    portfolio_returns, benchmark_returns, _, holdings, component_returns = generate_sample_data()

    # Initialize attribution analyzer
    attribution = AttributionAnalysis(
        portfolio_returns=portfolio_returns,
        holdings=holdings,
        component_returns=component_returns,
        benchmark_returns=benchmark_returns
    )

    # Component attribution
    print("\nComponent Attribution:")
    component_attr = attribution.component_attribution()
    if not component_attr.empty:
        print(component_attr[['contribution', 'avg_weight', 'component_return']].head())

    # Risk attribution
    print("\nRisk Attribution:")
    risk_attr = attribution.risk_attribution()
    if not risk_attr.empty:
        print(risk_attr[['weight', 'component_contrib', 'pct_contrib']].head())

    # Time-based attribution
    print("\nAnnual Attribution:")
    annual_attr = attribution.time_period_attribution('Y')
    if not annual_attr.empty:
        print(annual_attr[['return', 'cumulative_return']].head())


def example_5_benchmark_comparison():
    """Example 5: Benchmark comparison."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Benchmark Comparison")
    print("="*80)

    portfolio_returns, benchmark_returns, _, _, _ = generate_sample_data()

    # Initialize comparison
    comparison = BenchmarkComparison(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        portfolio_name="My Strategy",
        benchmark_name="S&P 500"
    )

    # Get all metrics
    metrics = comparison.calculate_all_comparisons()

    print("\nAbsolute Performance:")
    print(f"  Portfolio Return:    {metrics['portfolio_total_return']:.2%}")
    print(f"  Benchmark Return:    {metrics['benchmark_total_return']:.2%}")
    print(f"  Excess Return:       {metrics['excess_return']:.2%}")

    print("\nRisk-Adjusted Performance:")
    print(f"  Portfolio Sharpe:    {metrics['portfolio_sharpe']:.2f}")
    print(f"  Benchmark Sharpe:    {metrics['benchmark_sharpe']:.2f}")
    print(f"  Sharpe Difference:   {metrics['sharpe_diff']:.2f}")

    print("\nCAPM Analysis:")
    print(f"  Alpha (Annual):      {metrics['alpha_annualized']:.2%}")
    print(f"  Beta:                {metrics['beta']:.2f}")
    print(f"  R-squared:           {metrics['r_squared']:.2f}")

    print("\nCapture Ratios:")
    print(f"  Up Capture:          {metrics['up_capture']:.2%}")
    print(f"  Down Capture:        {metrics['down_capture']:.2%}")
    print(f"  Capture Ratio:       {metrics['capture_ratio']:.2f}")

    print("\nTracking Analysis:")
    print(f"  Tracking Error:      {metrics['tracking_error']:.2%}")
    print(f"  Information Ratio:   {metrics['information_ratio']:.2f}")

    print("\nStatistical Tests:")
    print(f"  T-statistic:         {metrics['t_statistic']:.2f}")
    print(f"  P-value:             {metrics['p_value']:.4f}")
    print(f"  Significant (5%):    {metrics['significant_at_5pct']}")


def example_6_trade_statistics():
    """Example 6: Trade statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Trade Statistics")
    print("="*80)

    _, _, trades, _, _ = generate_sample_data()

    # Calculate trade statistics
    stats = calculate_trade_statistics(trades)

    print("\nTrade Performance:")
    print(f"  Total Trades:        {stats['total_trades']}")
    print(f"  Winning Trades:      {stats['winning_trades']}")
    print(f"  Losing Trades:       {stats['losing_trades']}")
    print(f"  Win Rate:            {stats['win_rate']:.2%}")

    print("\nTrade Metrics:")
    print(f"  Average Win:         ${stats['avg_win']:.2f}")
    print(f"  Average Loss:        ${stats['avg_loss']:.2f}")
    print(f"  Average Trade:       ${stats['avg_trade']:.2f}")
    print(f"  Largest Win:         ${stats['largest_win']:.2f}")
    print(f"  Largest Loss:        ${stats['largest_loss']:.2f}")

    print("\nTrade Ratios:")
    print(f"  Payoff Ratio:        {stats['payoff_ratio']:.2f}")
    print(f"  Profit Factor:       {stats['profit_factor']:.2f}")
    print(f"  Expectancy:          ${stats['expectancy']:.2f}")


def example_7_rolling_analysis():
    """Example 7: Rolling metrics analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Rolling Metrics Analysis")
    print("="*80)

    portfolio_returns, benchmark_returns, _, _, _ = generate_sample_data()

    # Calculate rolling metrics
    calc = MetricsCalculator(portfolio_returns, benchmark_returns)

    window = 252  # 1 year
    print(f"\nCalculating rolling metrics (window={window} days)...")

    rolling_metrics = calc.rolling_metrics(
        window=window,
        metrics=['sharpe', 'sortino', 'volatility', 'max_dd']
    )

    if not rolling_metrics.empty:
        print(f"\nRolling Metrics Summary:")
        print(rolling_metrics.describe())

        print(f"\nLatest Rolling Metrics:")
        latest = rolling_metrics.iloc[-1]
        for metric, value in latest.items():
            print(f"  {metric:20s}: {value:.2f}")


def main():
    """Run all examples."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       PERFORMANCE ANALYTICS MODULE - USAGE EXAMPLES                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    try:
        example_1_basic_metrics()
        example_2_risk_analysis()
        example_3_tear_sheet()
        example_4_attribution()
        example_5_benchmark_comparison()
        example_6_trade_statistics()
        example_7_rolling_analysis()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        print("\nFor more information, see:")
        print("  - README.md in the performance/ directory")
        print("  - Docstrings in each module")
        print("  - Individual module documentation")
        print("\n")

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Note: Some examples may require additional dependencies to be installed.")


if __name__ == "__main__":
    main()

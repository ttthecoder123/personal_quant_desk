# Performance Analytics Module

Comprehensive performance analysis tools for backtesting system.

## Overview

This module provides institutional-grade performance analytics including:

- **Comprehensive Metrics**: Returns, risk, risk-adjusted, drawdowns, win/loss statistics
- **Advanced Risk Metrics**: VaR, CVaR, CDaR, tail risk, beta decomposition
- **Professional Tear Sheets**: PyFolio-style reports with visualizations
- **Performance Attribution**: Component, factor, sector, and risk attribution
- **Benchmark Comparison**: CAPM analysis, capture ratios, statistical tests

## Module Structure

```
performance/
├── __init__.py                    # Module exports
├── metrics_calculator.py          # Comprehensive performance metrics
├── risk_metrics.py                # Advanced risk calculations
├── tear_sheet_generator.py        # Visual performance reports
├── attribution_analysis.py        # Performance attribution
└── benchmark_comparison.py        # Benchmark analysis
```

## Quick Start

### 1. Calculate Performance Metrics

```python
import pandas as pd
from backtesting.performance import MetricsCalculator

# Load your returns data
returns = pd.Series(...)  # Your portfolio returns
benchmark_returns = pd.Series(...)  # Optional benchmark

# Initialize calculator
calc = MetricsCalculator(
    returns=returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02,
    periods_per_year=252
)

# Calculate all metrics
metrics = calc.calculate_all_metrics()

# Access specific metrics
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"CAGR: {metrics['cagr']:.2%}")

# Calculate rolling metrics
rolling_metrics = calc.rolling_metrics(window=252, metrics=['sharpe', 'volatility'])
```

### 2. Advanced Risk Analysis

```python
from backtesting.performance import RiskMetrics

# Initialize risk calculator
risk_calc = RiskMetrics(
    returns=returns,
    benchmark_returns=benchmark_returns,
    confidence_level=0.95
)

# Calculate all risk metrics
risk_metrics = risk_calc.calculate_all_risk_metrics()

# Access specific risk measures
print(f"VaR (95%): {risk_metrics['var_historical']:.2%}")
print(f"CVaR (95%): {risk_metrics['cvar_historical']:.2%}")
print(f"CDaR: {risk_metrics['cdar']:.2%}")
print(f"Tail Ratio: {risk_metrics['tail_ratio']:.2f}")

# Calculate rolling VaR
rolling_var = risk_calc.rolling_var(window=252, method='historical')

# Portfolio-level VaR
from backtesting.performance import calculate_portfolio_var

positions = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}
returns_data = pd.DataFrame(...)  # Returns for each asset

portfolio_var = calculate_portfolio_var(
    positions=positions,
    returns_data=returns_data,
    confidence_level=0.95,
    method='cornish_fisher'
)
```

### 3. Generate Tear Sheets

```python
from backtesting.performance import create_tear_sheet

# Generate comprehensive tear sheet
tear_sheet_path = create_tear_sheet(
    returns=returns,
    benchmark_returns=benchmark_returns,
    trades=trades_df,  # Optional: DataFrame with trade data
    output_path="reports/strategy_tearsheet.pdf",
    title="My Trading Strategy"
)

# Generate HTML tear sheet (interactive)
html_path = create_tear_sheet(
    returns=returns,
    benchmark_returns=benchmark_returns,
    output_path="reports/strategy_tearsheet.html",
    title="My Trading Strategy"
)
```

### 4. Performance Attribution

```python
from backtesting.performance import AttributionAnalysis

# Initialize attribution analyzer
attribution = AttributionAnalysis(
    portfolio_returns=returns,
    holdings=holdings_df,  # DataFrame of positions over time
    component_returns=component_returns_df,  # Returns by asset
    benchmark_returns=benchmark_returns,
    factor_returns=factor_returns_df  # Optional: Fama-French factors
)

# Calculate full attribution
results = attribution.calculate_full_attribution()

# Component attribution
component_attr = results['component_attribution']
print(component_attr.sort_values('contribution', ascending=False))

# Factor attribution (Fama-French style)
factor_attr = results['factor_attribution']
print(f"Alpha: {factor_attr[factor_attr['factor'] == 'Alpha']['contribution'].iloc[0]:.2%}")

# Risk attribution
risk_attr = attribution.risk_attribution()
print(risk_attr[['weight', 'component_contrib', 'pct_contrib']])

# Sector attribution
sector_mapping = {'AAPL': 'Tech', 'GOOGL': 'Tech', 'XOM': 'Energy'}
sector_attr = attribution.sector_attribution(sector_mapping)
```

### 5. Benchmark Comparison

```python
from backtesting.performance import BenchmarkComparison

# Initialize comparison
comparison = BenchmarkComparison(
    portfolio_returns=returns,
    benchmark_returns=benchmark_returns,
    portfolio_name="My Strategy",
    benchmark_name="S&P 500"
)

# Calculate all comparison metrics
metrics = comparison.calculate_all_comparisons()

# CAPM analysis
print(f"Alpha: {metrics['alpha_annualized']:.2%}")
print(f"Beta: {metrics['beta']:.2f}")
print(f"R-squared: {metrics['r_squared']:.2f}")

# Capture ratios
print(f"Up Capture: {metrics['up_capture']:.2%}")
print(f"Down Capture: {metrics['down_capture']:.2%}")

# Statistical significance
print(f"P-value: {metrics['p_value']:.4f}")
print(f"Significant at 5%: {metrics['significant_at_5pct']}")

# Generate comparison summary
summary = comparison.comparison_summary()
print(summary)

# Rolling analysis
rolling_capm = comparison.rolling_capm(window=252)
rolling_capture = comparison.rolling_capture_ratios(window=252)
```

## Detailed Features

### MetricsCalculator

**Return Metrics:**
- Total return, CAGR, annualized return
- Daily/monthly mean and median returns
- Best and worst days
- Positive/negative period ratios

**Risk Metrics:**
- Volatility (standard deviation)
- Downside deviation and semi-variance
- Tracking error and active risk

**Risk-Adjusted Metrics:**
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Omega ratio
- Kappa ratio (generalized Sortino)
- Gain-to-pain ratio
- Information ratio

**Drawdown Metrics:**
- Maximum drawdown
- Average drawdown
- Drawdown duration
- Recovery factor
- Ulcer index

**Distribution Metrics:**
- Skewness and kurtosis
- Tail ratio
- Value at Risk (VaR)
- Conditional VaR (CVaR)

**Benchmark Metrics:**
- Alpha and beta
- Correlation and R-squared
- Up/down capture ratios
- Active return

### RiskMetrics

**Value at Risk (VaR):**
- Historical VaR
- Parametric VaR (normal distribution)
- Cornish-Fisher VaR (adjusted for skewness/kurtosis)
- Monte Carlo VaR

**Conditional VaR (CVaR/Expected Shortfall):**
- Historical CVaR
- Parametric CVaR

**Drawdown Risk:**
- Conditional Drawdown at Risk (CDaR)
- Maximum and average drawdown

**Tail Risk:**
- Tail ratio (right/left tail)
- Left and right tail variance
- Tail index (Hill estimator)

**Pain Metrics:**
- Pain index (average drawdown)
- Ulcer index (RMS of drawdowns)
- Pain ratio

**Beta Analysis:**
- Beta to benchmark
- Bull and bear market betas
- Systematic vs idiosyncratic risk

### TearSheetGenerator

**Visualizations:**
- Equity curve with benchmark
- Underwater plot (drawdowns over time)
- Returns distribution with normal overlay
- Q-Q plot for normality
- Monthly returns heatmap
- Annual returns bar chart
- Rolling volatility and Sharpe ratio
- Rolling beta and correlation
- Best/worst periods analysis

**Output Formats:**
- PDF (publication-quality)
- PNG (high-resolution images)
- HTML (interactive Plotly charts)

**Trade Analysis (if trade data provided):**
- Trade PnL distribution
- Cumulative PnL
- Win/loss analysis
- Trade statistics table

### AttributionAnalysis

**Component Attribution:**
- Return contribution by asset/strategy
- Period-by-period attribution
- Rolling attribution

**Time-Based Attribution:**
- Monthly, quarterly, annual attribution
- Cumulative contributions

**Brinson Attribution:**
- Allocation effect
- Selection effect
- Interaction effect

**Factor Attribution:**
- Factor regression (Fama-French style)
- Factor betas and contributions
- Rolling factor attribution
- Alpha decomposition

**Risk Attribution:**
- Contribution to portfolio variance
- Marginal risk contribution
- Risk decomposition (systematic vs idiosyncratic)
- Diversification benefit

**Sector Attribution:**
- Attribution by sector/asset class
- Sector contribution analysis

### BenchmarkComparison

**Absolute Performance:**
- Total and annualized returns
- Volatility comparison
- Correlation analysis

**Risk-Adjusted Performance:**
- Sharpe, Sortino, Calmar ratios
- Information ratio

**CAPM Analysis:**
- Alpha and beta
- R-squared
- Residual volatility
- Rolling CAPM metrics

**Capture Ratios:**
- Up-capture ratio
- Down-capture ratio
- Capture ratio (up/down)
- Rolling capture ratios

**Tracking Analysis:**
- Tracking error
- Active return
- Information ratio
- Active share proxy

**Outperformance Analysis:**
- Outperformance periods and percentages
- Average magnitude of out/underperformance
- Longest streaks
- Relative strength

**Statistical Tests:**
- T-test for active return significance
- Sharpe ratio comparison
- Win rate binomial test

## Example Workflow

```python
import pandas as pd
from backtesting.performance import (
    MetricsCalculator,
    RiskMetrics,
    TearSheetGenerator,
    AttributionAnalysis,
    BenchmarkComparison,
    create_tear_sheet,
)

# 1. Load data
returns = pd.read_csv('portfolio_returns.csv', index_col=0, parse_dates=True)['returns']
benchmark_returns = pd.read_csv('benchmark_returns.csv', index_col=0, parse_dates=True)['returns']
trades = pd.read_csv('trades.csv', index_col=0, parse_dates=True)

# 2. Calculate comprehensive metrics
metrics_calc = MetricsCalculator(returns, benchmark_returns)
all_metrics = metrics_calc.calculate_all_metrics()

# Print key metrics
print(f"Total Return: {all_metrics['total_return']:.2%}")
print(f"CAGR: {all_metrics['cagr']:.2%}")
print(f"Sharpe Ratio: {all_metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {all_metrics['max_drawdown']:.2%}")
print(f"Alpha: {all_metrics['alpha']:.2%}")
print(f"Beta: {all_metrics['beta']:.2f}")

# 3. Calculate advanced risk metrics
risk_calc = RiskMetrics(returns, benchmark_returns)
risk_metrics = risk_calc.calculate_all_risk_metrics()

print(f"VaR (95%): {risk_metrics['var_historical']:.2%}")
print(f"CVaR (95%): {risk_metrics['cvar_historical']:.2%}")

# 4. Generate tear sheet
tear_sheet_path = create_tear_sheet(
    returns=returns,
    benchmark_returns=benchmark_returns,
    trades=trades,
    output_path="reports/full_tearsheet.pdf",
    title="Strategy Performance Analysis"
)
print(f"Tear sheet saved to: {tear_sheet_path}")

# 5. Benchmark comparison
comparison = BenchmarkComparison(returns, benchmark_returns)
comp_summary = comparison.comparison_summary()
print("\nBenchmark Comparison:")
print(comp_summary)

# 6. Statistical significance
stats = comparison.statistical_tests()
print(f"\nOutperformance is significant: {stats['significant_at_5pct']}")
print(f"P-value: {stats['p_value']:.4f}")
```

## Trade Statistics

If you have trade-level data, you can calculate detailed trade statistics:

```python
from backtesting.performance import calculate_trade_statistics

# Trades DataFrame should have a 'pnl' column
trade_stats = calculate_trade_statistics(trades_df)

print(f"Total Trades: {trade_stats['total_trades']}")
print(f"Win Rate: {trade_stats['win_rate']:.2%}")
print(f"Profit Factor: {trade_stats['profit_factor']:.2f}")
print(f"Payoff Ratio: {trade_stats['payoff_ratio']:.2f}")
print(f"Expectancy: {trade_stats['expectancy']:.4f}")
```

## Tips and Best Practices

1. **Data Requirements:**
   - Ensure returns are properly aligned (same index)
   - Use daily returns for most calculations
   - Convert prices to returns: `returns = prices.pct_change()`

2. **Risk-Free Rate:**
   - Default is 2% annual (0.02)
   - Adjust based on your market and time period
   - Use current treasury rates for accuracy

3. **Periods Per Year:**
   - Daily data: 252 (trading days)
   - Weekly data: 52
   - Monthly data: 12

4. **Rolling Windows:**
   - Use 252 days (1 year) for daily data
   - Use 756 days (3 years) for longer-term trends
   - Ensure window size is appropriate for data frequency

5. **Benchmark Selection:**
   - Choose appropriate benchmark for your strategy
   - Ensure benchmark data covers same period
   - Consider multiple benchmarks for comparison

6. **Attribution Analysis:**
   - Requires holdings/weights data over time
   - Component returns should match portfolio components
   - Factor data can be obtained from Kenneth French's data library

7. **Statistical Significance:**
   - Longer time periods provide more reliable statistics
   - Check p-values for significance tests
   - Be aware of overfitting in optimized strategies

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.15.0 (optional, for HTML reports)
- empyrical >= 0.5.5 (optional)
- quantstats >= 0.0.62 (optional)

## References

- PyFolio: https://github.com/quantopian/pyfolio
- QuantStats: https://github.com/ranaroussi/quantstats
- Empyrical: https://github.com/quantopian/empyrical
- Fama-French Data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

## Support

For issues or questions about the performance analytics module:
1. Check the docstrings in each module
2. Review the example workflows above
3. Consult the individual module documentation

## Version

Version: 1.0.0
Created: 2025-10-29

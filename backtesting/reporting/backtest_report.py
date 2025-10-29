"""
Comprehensive backtest report generator.

This module generates professional backtest reports in multiple formats:
- HTML reports with embedded charts
- PDF reports for formal presentations
- JSON reports for programmatic access

Reports include:
- Executive summary
- Strategy description
- Performance metrics
- Risk analysis
- Trade statistics
- Cost analysis
- Validation results
- Visualizations
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import base64
import io
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# Import from existing modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from performance.metrics_calculator import MetricsCalculator


class ReportFormat(Enum):
    """Report output format."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    MARKDOWN = "markdown"


class ReportSection(Enum):
    """Report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    STRATEGY_DESCRIPTION = "strategy_description"
    PERFORMANCE_METRICS = "performance_metrics"
    RISK_ANALYSIS = "risk_analysis"
    TRADE_STATISTICS = "trade_statistics"
    COST_ANALYSIS = "cost_analysis"
    VALIDATION_RESULTS = "validation_results"
    CHARTS = "charts"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Backtest Report"
    subtitle: Optional[str] = None
    author: Optional[str] = None
    include_sections: Optional[List[ReportSection]] = None
    chart_style: str = "seaborn-v0_8-darkgrid"
    chart_dpi: int = 100
    color_palette: str = "husl"
    logo_path: Optional[Path] = None


class BacktestReportGenerator:
    """
    Generate comprehensive backtest reports in multiple formats.

    This class creates professional, publication-quality reports that include
    all relevant performance metrics, visualizations, and analysis.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(self.config.color_palette)

        logger.info("BacktestReportGenerator initialized")

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.HTML,
    ) -> Path:
        """
        Generate comprehensive backtest report.

        Args:
            results: Backtest results dictionary containing:
                - returns: pd.Series of returns
                - equity_curve: pd.Series of portfolio values
                - trades: pd.DataFrame of trades
                - positions: pd.DataFrame of positions
                - metrics: Dict of calculated metrics
                - strategy_info: Dict with strategy metadata
                - validation_results: Optional validation results
            output_path: Output file path
            format: Report format (HTML, PDF, JSON)

        Returns:
            Path to generated report
        """
        logger.info(f"Generating {format.value} report: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ReportFormat.HTML:
            return self._generate_html_report(results, output_path)
        elif format == ReportFormat.PDF:
            return self._generate_pdf_report(results, output_path)
        elif format == ReportFormat.JSON:
            return self._generate_json_report(results, output_path)
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(results, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    # ===================================================================
    # HTML Report Generation
    # ===================================================================

    def _generate_html_report(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Generate HTML report with embedded charts."""
        logger.debug("Generating HTML report")

        # Build HTML content
        html_parts = [
            self._html_header(),
            self._html_executive_summary(results),
            self._html_strategy_description(results),
            self._html_performance_metrics(results),
            self._html_risk_analysis(results),
            self._html_trade_statistics(results),
            self._html_cost_analysis(results),
            self._html_charts(results),
            self._html_validation_results(results),
            self._html_footer(),
        ]

        html_content = "\n".join(html_parts)

        # Write to file
        output_path.write_text(html_content, encoding='utf-8')
        logger.success(f"HTML report generated: {output_path}")

        return output_path

    def _html_header(self) -> str:
        """Generate HTML header."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .metric-value.positive {{
            color: #28a745;
        }}
        .metric-value.negative {{
            color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            margin-top: 40px;
            border-top: 2px solid #ddd;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .info {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.title}</h1>
        <p>{self.config.subtitle or ""}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {f"<p>Author: {self.config.author}</p>" if self.config.author else ""}
    </div>
"""

    def _html_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        metrics = results.get('metrics', {})

        # Key metrics for executive summary
        total_return = metrics.get('total_return', 0) * 100
        cagr = metrics.get('cagr', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0) * 100

        return_class = "positive" if total_return > 0 else "negative"

        return f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {return_class}">{total_return:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAGR</div>
                <div class="metric-value">{cagr:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{sharpe:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{max_dd:.2f}%</div>
            </div>
        </div>
    </div>
"""

    def _html_strategy_description(self, results: Dict[str, Any]) -> str:
        """Generate strategy description section."""
        strategy_info = results.get('strategy_info', {})

        return f"""
    <div class="section">
        <h2>Strategy Description</h2>
        <p><strong>Strategy Name:</strong> {strategy_info.get('name', 'N/A')}</p>
        <p><strong>Description:</strong> {strategy_info.get('description', 'No description available')}</p>
        <p><strong>Backtest Period:</strong> {strategy_info.get('start_date', 'N/A')} to {strategy_info.get('end_date', 'N/A')}</p>
        <p><strong>Universe:</strong> {strategy_info.get('universe', 'N/A')}</p>
        <p><strong>Initial Capital:</strong> ${strategy_info.get('initial_capital', 0):,.2f}</p>
    </div>
"""

    def _html_performance_metrics(self, results: Dict[str, Any]) -> str:
        """Generate performance metrics table."""
        metrics = results.get('metrics', {})

        # Organize metrics into categories
        return_metrics = [
            ('Total Return', metrics.get('total_return', 0), '%'),
            ('CAGR', metrics.get('cagr', 0), '%'),
            ('Annualized Return', metrics.get('annualized_return', 0), '%'),
            ('Best Day', metrics.get('best_day', 0), '%'),
            ('Worst Day', metrics.get('worst_day', 0), '%'),
        ]

        risk_metrics = [
            ('Annualized Volatility', metrics.get('annualized_volatility', 0), '%'),
            ('Downside Deviation', metrics.get('downside_deviation', 0), '%'),
            ('Max Drawdown', metrics.get('max_drawdown', 0), '%'),
            ('VaR (95%)', metrics.get('value_at_risk_95', 0), '%'),
            ('CVaR (95%)', metrics.get('conditional_var_95', 0), '%'),
        ]

        ratio_metrics = [
            ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), ''),
            ('Sortino Ratio', metrics.get('sortino_ratio', 0), ''),
            ('Calmar Ratio', metrics.get('calmar_ratio', 0), ''),
            ('Omega Ratio', metrics.get('omega_ratio', 0), ''),
        ]

        def format_metric(name, value, unit):
            formatted_value = f"{value * 100:.2f}" if unit == '%' else f"{value:.2f}"
            return f"<tr><td>{name}</td><td>{formatted_value}{unit}</td></tr>"

        return f"""
    <div class="section">
        <h2>Performance Metrics</h2>

        <h3>Return Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {"".join(format_metric(n, v, u) for n, v, u in return_metrics)}
        </table>

        <h3>Risk Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {"".join(format_metric(n, v, u) for n, v, u in risk_metrics)}
        </table>

        <h3>Risk-Adjusted Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {"".join(format_metric(n, v, u) for n, v, u in ratio_metrics)}
        </table>
    </div>
"""

    def _html_risk_analysis(self, results: Dict[str, Any]) -> str:
        """Generate risk analysis section."""
        metrics = results.get('metrics', {})

        return f"""
    <div class="section">
        <h2>Risk Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Skewness</div>
                <div class="metric-value">{metrics.get('skewness', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Excess Kurtosis</div>
                <div class="metric-value">{metrics.get('excess_kurtosis', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Tail Ratio</div>
                <div class="metric-value">{metrics.get('tail_ratio', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Ulcer Index</div>
                <div class="metric-value">{metrics.get('ulcer_index', 0):.2f}</div>
            </div>
        </div>

        {self._risk_warning(metrics)}
    </div>
"""

    def _risk_warning(self, metrics: Dict[str, float]) -> str:
        """Generate risk warnings if applicable."""
        warnings = []

        if metrics.get('max_drawdown', 0) < -0.30:
            warnings.append("Maximum drawdown exceeds 30% - high risk strategy")

        if metrics.get('sharpe_ratio', 0) < 0.5:
            warnings.append("Low Sharpe ratio indicates poor risk-adjusted returns")

        if abs(metrics.get('skewness', 0)) > 1.0:
            warnings.append("High skewness indicates asymmetric return distribution")

        if warnings:
            warning_html = "".join(f"<p>• {w}</p>" for w in warnings)
            return f'<div class="warning"><strong>Risk Warnings:</strong>{warning_html}</div>'

        return ""

    def _html_trade_statistics(self, results: Dict[str, Any]) -> str:
        """Generate trade statistics section."""
        trades = results.get('trades', pd.DataFrame())

        if trades.empty or 'pnl' not in trades.columns:
            return """
    <div class="section">
        <h2>Trade Statistics</h2>
        <p>No trade data available.</p>
    </div>
"""

        # Calculate trade statistics
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]

        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        return f"""
    <div class="section">
        <h2>Trade Statistics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{total_trades}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Win</div>
                <div class="metric-value positive">${avg_win:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Loss</div>
                <div class="metric-value negative">${avg_loss:,.2f}</div>
            </div>
        </div>
    </div>
"""

    def _html_cost_analysis(self, results: Dict[str, Any]) -> str:
        """Generate cost analysis section."""
        trades = results.get('trades', pd.DataFrame())

        if trades.empty:
            return ""

        total_commission = trades.get('commission', pd.Series()).sum()
        total_slippage = trades.get('slippage', pd.Series()).sum()

        return f"""
    <div class="section">
        <h2>Cost Analysis</h2>
        <table>
            <tr><th>Cost Type</th><th>Total</th></tr>
            <tr><td>Commission</td><td>${total_commission:,.2f}</td></tr>
            <tr><td>Slippage</td><td>${total_slippage:,.2f}</td></tr>
            <tr><td><strong>Total Costs</strong></td><td><strong>${total_commission + total_slippage:,.2f}</strong></td></tr>
        </table>
    </div>
"""

    def _html_charts(self, results: Dict[str, Any]) -> str:
        """Generate charts section."""
        charts_html = '<div class="section"><h2>Performance Charts</h2>'

        # Generate equity curve chart
        if 'equity_curve' in results:
            equity_chart = self._generate_equity_curve_chart(results['equity_curve'])
            charts_html += f'<div class="chart-container">{equity_chart}</div>'

        # Generate drawdown chart
        if 'returns' in results:
            dd_chart = self._generate_drawdown_chart(results['returns'])
            charts_html += f'<div class="chart-container">{dd_chart}</div>'

        # Generate returns distribution
        if 'returns' in results:
            dist_chart = self._generate_returns_distribution(results['returns'])
            charts_html += f'<div class="chart-container">{dist_chart}</div>'

        charts_html += '</div>'
        return charts_html

    def _generate_equity_curve_chart(self, equity_curve: pd.Series) -> str:
        """Generate equity curve chart as base64 image."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color='#667eea')
        ax.fill_between(equity_curve.index, equity_curve.values, alpha=0.3, color='#667eea')

        ax.set_title('Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Convert to base64
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)

        return f'<img src="data:image/png;base64,{img_base64}" alt="Equity Curve">'

    def _generate_drawdown_chart(self, returns: pd.Series) -> str:
        """Generate drawdown chart."""
        # Calculate drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                        alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown.values * 100,
               linewidth=1, color='darkred')

        ax.set_title('Drawdown Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        plt.xticks(rotation=45)
        plt.tight_layout()

        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)

        return f'<img src="data:image/png;base64,{img_base64}" alt="Drawdown">'

    def _generate_returns_distribution(self, returns: pd.Series) -> str:
        """Generate returns distribution chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(returns * 100, bins=50, alpha=0.7, color='#667eea', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)

        return f'<img src="data:image/png;base64,{img_base64}" alt="Returns Distribution">'

    def _html_validation_results(self, results: Dict[str, Any]) -> str:
        """Generate validation results section."""
        validation = results.get('validation_results', {})

        if not validation:
            return ""

        return f"""
    <div class="section">
        <h2>Validation Results</h2>
        <div class="info">
            <p>Validation tests help identify potential overfitting and assess strategy robustness.</p>
        </div>
        <table>
            <tr><th>Test</th><th>Result</th><th>Status</th></tr>
            {self._format_validation_table(validation)}
        </table>
    </div>
"""

    def _format_validation_table(self, validation: Dict) -> str:
        """Format validation results as table rows."""
        rows = []
        for test_name, result in validation.items():
            status = "✓ Pass" if result.get('passed', False) else "✗ Fail"
            status_class = "positive" if result.get('passed', False) else "negative"
            value = result.get('value', 'N/A')
            rows.append(f'<tr><td>{test_name}</td><td>{value}</td><td class="{status_class}">{status}</td></tr>')
        return "".join(rows)

    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return """
    <div class="footer">
        <p>This report was generated by the Personal Quant Desk Backtesting System</p>
        <p>Past performance is not indicative of future results. This report is for informational purposes only.</p>
    </div>
</body>
</html>
"""

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.config.chart_dpi, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return img_base64

    # ===================================================================
    # PDF Report Generation
    # ===================================================================

    def _generate_pdf_report(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Generate PDF report."""
        logger.debug("Generating PDF report")

        with PdfPages(output_path) as pdf:
            # Page 1: Title and Executive Summary
            self._pdf_title_page(pdf, results)

            # Page 2: Performance Metrics
            self._pdf_metrics_page(pdf, results)

            # Page 3: Charts
            self._pdf_charts_page(pdf, results)

            # Page 4: Trade Analysis
            self._pdf_trades_page(pdf, results)

            # Add metadata
            d = pdf.infodict()
            d['Title'] = self.config.title
            d['Author'] = self.config.author or 'Backtesting System'
            d['Subject'] = 'Backtest Results'
            d['CreationDate'] = datetime.now()

        logger.success(f"PDF report generated: {output_path}")
        return output_path

    def _pdf_title_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create PDF title page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.8, self.config.title,
                ha='center', va='center', fontsize=28, fontweight='bold')

        # Subtitle
        if self.config.subtitle:
            ax.text(0.5, 0.72, self.config.subtitle,
                   ha='center', va='center', fontsize=16)

        # Date
        ax.text(0.5, 0.65, f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
               ha='center', va='center', fontsize=12)

        # Key metrics
        metrics = results.get('metrics', {})
        summary_text = f"""
        Total Return: {metrics.get('total_return', 0) * 100:.2f}%
        CAGR: {metrics.get('cagr', 0) * 100:.2f}%
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%
        """

        ax.text(0.5, 0.4, summary_text,
               ha='center', va='center', fontsize=14,
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _pdf_metrics_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create PDF metrics page."""
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))

        metrics = results.get('metrics', {})

        # Return metrics table
        ax = axes[0]
        ax.axis('tight')
        ax.axis('off')
        ax.set_title('Return Metrics', fontsize=14, fontweight='bold', pad=20)

        return_data = [
            ['Total Return', f"{metrics.get('total_return', 0) * 100:.2f}%"],
            ['CAGR', f"{metrics.get('cagr', 0) * 100:.2f}%"],
            ['Best Day', f"{metrics.get('best_day', 0) * 100:.2f}%"],
            ['Worst Day', f"{metrics.get('worst_day', 0) * 100:.2f}%"],
        ]

        table = ax.table(cellText=return_data, cellLoc='left',
                        colWidths=[0.6, 0.4], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Risk metrics table
        ax = axes[1]
        ax.axis('tight')
        ax.axis('off')
        ax.set_title('Risk Metrics', fontsize=14, fontweight='bold', pad=20)

        risk_data = [
            ['Volatility', f"{metrics.get('annualized_volatility', 0) * 100:.2f}%"],
            ['Max Drawdown', f"{metrics.get('max_drawdown', 0) * 100:.2f}%"],
            ['VaR (95%)', f"{metrics.get('value_at_risk_95', 0) * 100:.2f}%"],
            ['CVaR (95%)', f"{metrics.get('conditional_var_95', 0) * 100:.2f}%"],
        ]

        table = ax.table(cellText=risk_data, cellLoc='left',
                        colWidths=[0.6, 0.4], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Ratios table
        ax = axes[2]
        ax.axis('tight')
        ax.axis('off')
        ax.set_title('Risk-Adjusted Metrics', fontsize=14, fontweight='bold', pad=20)

        ratio_data = [
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"],
            ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"],
            ['Omega Ratio', f"{metrics.get('omega_ratio', 0):.2f}"],
        ]

        table = ax.table(cellText=ratio_data, cellLoc='left',
                        colWidths=[0.6, 0.4], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _pdf_charts_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create PDF charts page."""
        fig = plt.figure(figsize=(8.5, 11))
        gs = fig.add_gridspec(3, 1, hspace=0.3)

        # Equity curve
        if 'equity_curve' in results:
            ax1 = fig.add_subplot(gs[0])
            equity = results['equity_curve']
            ax1.plot(equity.index, equity.values, linewidth=2)
            ax1.set_title('Equity Curve', fontweight='bold')
            ax1.set_ylabel('Portfolio Value')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Drawdown
        if 'returns' in results:
            ax2 = fig.add_subplot(gs[1])
            returns = results['returns']
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max

            ax2.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.5, color='red')
            ax2.set_title('Drawdown', fontweight='bold')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

        # Returns distribution
        if 'returns' in results:
            ax3 = fig.add_subplot(gs[2])
            returns = results['returns']
            ax3.hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
            ax3.set_title('Returns Distribution', fontweight='bold')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _pdf_trades_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create PDF trades analysis page."""
        trades = results.get('trades', pd.DataFrame())

        if trades.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))

        # P&L distribution
        if 'pnl' in trades.columns:
            ax = axes[0, 0]
            ax.hist(trades['pnl'], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title('P&L Distribution', fontweight='bold')
            ax.set_xlabel('P&L ($)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

        # Cumulative P&L
        if 'pnl' in trades.columns:
            ax = axes[0, 1]
            cum_pnl = trades['pnl'].cumsum()
            ax.plot(cum_pnl.values, linewidth=2)
            ax.set_title('Cumulative P&L', fontweight='bold')
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Cumulative P&L ($)')
            ax.grid(True, alpha=0.3)

        # Win/Loss by symbol
        if 'symbol' in trades.columns and 'pnl' in trades.columns:
            ax = axes[1, 0]
            symbol_pnl = trades.groupby('symbol')['pnl'].sum().sort_values()
            symbol_pnl.plot(kind='barh', ax=ax)
            ax.set_title('P&L by Symbol', fontweight='bold')
            ax.set_xlabel('Total P&L ($)')
            ax.grid(True, alpha=0.3)

        # Trade statistics text
        ax = axes[1, 1]
        ax.axis('off')

        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]

        stats_text = f"""
        Trade Statistics

        Total Trades: {len(trades)}
        Winning Trades: {len(winning_trades)}
        Losing Trades: {len(losing_trades)}

        Win Rate: {len(winning_trades) / len(trades) * 100:.2f}%

        Avg Win: ${winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0:,.2f}
        Avg Loss: ${losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0:,.2f}

        Largest Win: ${trades['pnl'].max():,.2f}
        Largest Loss: ${trades['pnl'].min():,.2f}
        """

        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ===================================================================
    # JSON Report Generation
    # ===================================================================

    def _generate_json_report(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Generate JSON report for programmatic access."""
        logger.debug("Generating JSON report")

        # Convert pandas objects to JSON-serializable format
        json_results = self._prepare_for_json(results)

        # Add metadata
        json_results['report_metadata'] = {
            'title': self.config.title,
            'generated_at': datetime.now().isoformat(),
            'author': self.config.author,
        }

        # Write to file
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, default=str)

        logger.success(f"JSON report generated: {output_path}")
        return output_path

    def _prepare_for_json(self, results: Dict[str, Any]) -> Dict:
        """Prepare results for JSON serialization."""
        json_data = {}

        for key, value in results.items():
            if isinstance(value, pd.Series):
                json_data[key] = {
                    'index': value.index.tolist(),
                    'values': value.tolist(),
                }
            elif isinstance(value, pd.DataFrame):
                json_data[key] = value.to_dict(orient='records')
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                json_data[key] = value
            else:
                json_data[key] = str(value)

        return json_data

    # ===================================================================
    # Markdown Report Generation
    # ===================================================================

    def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Generate Markdown report."""
        logger.debug("Generating Markdown report")

        md_parts = [
            self._md_header(results),
            self._md_executive_summary(results),
            self._md_performance_metrics(results),
            self._md_trade_statistics(results),
        ]

        md_content = "\n\n".join(md_parts)

        output_path.write_text(md_content, encoding='utf-8')
        logger.success(f"Markdown report generated: {output_path}")

        return output_path

    def _md_header(self, results: Dict[str, Any]) -> str:
        """Generate Markdown header."""
        return f"""# {self.config.title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{f"**Author:** {self.config.author}" if self.config.author else ""}

---
"""

    def _md_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate Markdown executive summary."""
        metrics = results.get('metrics', {})

        return f"""## Executive Summary

| Metric | Value |
|--------|-------|
| Total Return | {metrics.get('total_return', 0) * 100:.2f}% |
| CAGR | {metrics.get('cagr', 0) * 100:.2f}% |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {metrics.get('max_drawdown', 0) * 100:.2f}% |
"""

    def _md_performance_metrics(self, results: Dict[str, Any]) -> str:
        """Generate Markdown performance metrics."""
        metrics = results.get('metrics', {})

        return f"""## Performance Metrics

### Return Metrics

| Metric | Value |
|--------|-------|
| Total Return | {metrics.get('total_return', 0) * 100:.2f}% |
| CAGR | {metrics.get('cagr', 0) * 100:.2f}% |
| Annualized Return | {metrics.get('annualized_return', 0) * 100:.2f}% |

### Risk Metrics

| Metric | Value |
|--------|-------|
| Volatility | {metrics.get('annualized_volatility', 0) * 100:.2f}% |
| Max Drawdown | {metrics.get('max_drawdown', 0) * 100:.2f}% |
| VaR (95%) | {metrics.get('value_at_risk_95', 0) * 100:.2f}% |

### Risk-Adjusted Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {metrics.get('calmar_ratio', 0):.2f} |
"""

    def _md_trade_statistics(self, results: Dict[str, Any]) -> str:
        """Generate Markdown trade statistics."""
        trades = results.get('trades', pd.DataFrame())

        if trades.empty or 'pnl' not in trades.columns:
            return "## Trade Statistics\n\nNo trade data available."

        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]

        return f"""## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {len(trades)} |
| Winning Trades | {len(winning_trades)} |
| Losing Trades | {len(losing_trades)} |
| Win Rate | {len(winning_trades) / len(trades) * 100:.2f}% |
| Average Win | ${winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0:,.2f} |
| Average Loss | ${losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0:,.2f} |
"""

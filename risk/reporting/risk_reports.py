"""
Risk Report Generation System

Generates comprehensive risk reports in multiple formats:
- Daily risk summaries
- Weekly performance and risk reports
- Monthly comprehensive reports
- PDF and HTML output formats
- Executive summaries
- Detailed analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
from io import BytesIO
import base64

from ..core.risk_metrics import RiskMetrics, RiskMetricsResult
from ..core.risk_engine import RiskState, RiskLevel


class ReportFrequency(Enum):
    """Report generation frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ADHOC = "adhoc"


class ReportFormat(Enum):
    """Report output format"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"


class ReportType(Enum):
    """Type of risk report"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_RISK = "detailed_risk"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    STRESS_TEST = "stress_test"
    VAR_BACKTEST = "var_backtest"


@dataclass
class ReportSection:
    """Report section configuration"""
    title: str
    content: str
    order: int
    include_charts: bool = True
    charts: List[Dict] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)


@dataclass
class ReportMetadata:
    """Report metadata"""
    report_id: str
    report_type: ReportType
    frequency: ReportFrequency
    format: ReportFormat
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    generated_by: str = "RiskReporter"
    version: str = "1.0.0"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    include_executive_summary: bool = True
    include_risk_metrics: bool = True
    include_performance_metrics: bool = True
    include_position_analysis: bool = True
    include_var_analysis: bool = True
    include_stress_tests: bool = False
    include_charts: bool = True
    include_recommendations: bool = True
    logo_path: Optional[str] = None
    company_name: str = "Personal Quant Desk"
    report_footer: Optional[str] = None


class RiskReporter:
    """
    Comprehensive risk report generation system

    Generates professional risk reports in multiple formats:
    - Daily/Weekly/Monthly risk reports
    - PDF and HTML outputs
    - Executive summaries
    - Detailed analytics
    - Charts and visualizations
    """

    def __init__(
        self,
        output_dir: str = "./reports",
        config: Optional[ReportConfig] = None
    ):
        """
        Initialize risk reporter

        Args:
            output_dir: Directory for report output
            config: Report configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config if config is not None else ReportConfig()

        # Report history
        self.generated_reports: List[ReportMetadata] = []

        # Styling
        self.html_styles = self._get_html_styles()

    def generate_report(
        self,
        report_type: ReportType,
        frequency: ReportFrequency,
        period_start: datetime,
        period_end: datetime,
        portfolio_data: Dict[str, Any],
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState],
        format: ReportFormat = ReportFormat.HTML,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive risk report

        Args:
            report_type: Type of report to generate
            frequency: Report frequency
            period_start: Start of reporting period
            period_end: End of reporting period
            portfolio_data: Portfolio data dictionary
            risk_metrics: Calculated risk metrics
            risk_states: Historical risk states
            format: Output format
            output_filename: Optional custom filename

        Returns:
            Path to generated report file
        """
        # Generate report ID
        report_id = self._generate_report_id(report_type, period_end)

        # Create metadata
        metadata = ReportMetadata(
            report_id=report_id,
            report_type=report_type,
            frequency=frequency,
            format=format,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end
        )

        # Build report sections
        sections = self._build_report_sections(
            report_type=report_type,
            portfolio_data=portfolio_data,
            risk_metrics=risk_metrics,
            risk_states=risk_states,
            period_start=period_start,
            period_end=period_end
        )

        # Generate report based on format
        if format == ReportFormat.HTML:
            content = self._generate_html_report(metadata, sections)
            extension = ".html"
        elif format == ReportFormat.PDF:
            # Generate HTML first, then convert to PDF
            html_content = self._generate_html_report(metadata, sections)
            content = self._convert_html_to_pdf(html_content)
            extension = ".pdf"
        elif format == ReportFormat.JSON:
            content = self._generate_json_report(metadata, sections)
            extension = ".json"
        elif format == ReportFormat.MARKDOWN:
            content = self._generate_markdown_report(metadata, sections)
            extension = ".md"
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Determine output filename
        if output_filename is None:
            output_filename = f"{report_id}{extension}"

        output_path = self.output_dir / output_filename

        # Write report
        if format == ReportFormat.PDF:
            with open(output_path, 'wb') as f:
                f.write(content)
        else:
            with open(output_path, 'w') as f:
                f.write(content)

        # Track generated report
        self.generated_reports.append(metadata)

        return str(output_path)

    def generate_daily_report(
        self,
        date: datetime,
        portfolio_data: Dict[str, Any],
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState],
        format: ReportFormat = ReportFormat.HTML
    ) -> str:
        """
        Generate daily risk report

        Args:
            date: Report date
            portfolio_data: Portfolio data
            risk_metrics: Risk metrics
            risk_states: Risk states for the day
            format: Output format

        Returns:
            Path to generated report
        """
        return self.generate_report(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            frequency=ReportFrequency.DAILY,
            period_start=date.replace(hour=0, minute=0, second=0),
            period_end=date.replace(hour=23, minute=59, second=59),
            portfolio_data=portfolio_data,
            risk_metrics=risk_metrics,
            risk_states=risk_states,
            format=format
        )

    def generate_weekly_report(
        self,
        week_end_date: datetime,
        portfolio_data: Dict[str, Any],
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState],
        format: ReportFormat = ReportFormat.PDF
    ) -> str:
        """
        Generate weekly risk report

        Args:
            week_end_date: End date of week
            portfolio_data: Portfolio data
            risk_metrics: Risk metrics
            risk_states: Risk states for the week
            format: Output format

        Returns:
            Path to generated report
        """
        week_start = week_end_date - timedelta(days=6)

        return self.generate_report(
            report_type=ReportType.DETAILED_RISK,
            frequency=ReportFrequency.WEEKLY,
            period_start=week_start,
            period_end=week_end_date,
            portfolio_data=portfolio_data,
            risk_metrics=risk_metrics,
            risk_states=risk_states,
            format=format
        )

    def generate_monthly_report(
        self,
        month: datetime,
        portfolio_data: Dict[str, Any],
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState],
        format: ReportFormat = ReportFormat.PDF
    ) -> str:
        """
        Generate monthly risk report

        Args:
            month: Month end date
            portfolio_data: Portfolio data
            risk_metrics: Risk metrics
            risk_states: Risk states for the month
            format: Output format

        Returns:
            Path to generated report
        """
        # Get first and last day of month
        period_start = month.replace(day=1, hour=0, minute=0, second=0)
        next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        period_end = next_month - timedelta(seconds=1)

        return self.generate_report(
            report_type=ReportType.DETAILED_RISK,
            frequency=ReportFrequency.MONTHLY,
            period_start=period_start,
            period_end=period_end,
            portfolio_data=portfolio_data,
            risk_metrics=risk_metrics,
            risk_states=risk_states,
            format=format
        )

    def _build_report_sections(
        self,
        report_type: ReportType,
        portfolio_data: Dict[str, Any],
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState],
        period_start: datetime,
        period_end: datetime
    ) -> List[ReportSection]:
        """Build report sections based on type"""
        sections = []

        # Executive Summary
        if self.config.include_executive_summary:
            sections.append(self._build_executive_summary(
                portfolio_data, risk_metrics, risk_states, period_start, period_end
            ))

        # Risk Metrics
        if self.config.include_risk_metrics:
            sections.append(self._build_risk_metrics_section(
                risk_metrics, risk_states
            ))

        # Performance Metrics
        if self.config.include_performance_metrics:
            sections.append(self._build_performance_section(
                portfolio_data, risk_metrics
            ))

        # Position Analysis
        if self.config.include_position_analysis:
            sections.append(self._build_position_analysis(
                portfolio_data
            ))

        # VaR Analysis
        if self.config.include_var_analysis:
            sections.append(self._build_var_analysis(
                risk_metrics, risk_states
            ))

        # Recommendations
        if self.config.include_recommendations:
            sections.append(self._build_recommendations(
                risk_metrics, risk_states
            ))

        return sections

    def _build_executive_summary(
        self,
        portfolio_data: Dict[str, Any],
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState],
        period_start: datetime,
        period_end: datetime
    ) -> ReportSection:
        """Build executive summary section"""
        portfolio_value = portfolio_data.get('portfolio_value', 0)
        initial_value = portfolio_data.get('initial_value', portfolio_value)

        period_return = ((portfolio_value / initial_value) - 1) * 100 if initial_value > 0 else 0

        # Calculate risk level distribution
        risk_level_counts = {}
        for state in risk_states:
            level = state.risk_level.value
            risk_level_counts[level] = risk_level_counts.get(level, 0) + 1

        content = f"""
**Period:** {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}

**Portfolio Performance:**
- Current Value: ${portfolio_value:,.2f}
- Period Return: {period_return:.2f}%
- Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
- Sortino Ratio: {risk_metrics.sortino_ratio:.2f}

**Risk Summary:**
- Portfolio Volatility: {risk_metrics.portfolio_volatility:.2%}
- Value at Risk (95%): {risk_metrics.var_95:.2%}
- Maximum Drawdown: {risk_metrics.max_drawdown:.2%}
- Current Drawdown: {risk_metrics.current_drawdown:.2%}

**Risk Level Distribution:**
"""
        for level, count in sorted(risk_level_counts.items()):
            pct = (count / len(risk_states) * 100) if risk_states else 0
            content += f"- {level.upper()}: {count} periods ({pct:.1f}%)\n"

        return ReportSection(
            title="Executive Summary",
            content=content,
            order=1,
            include_charts=True
        )

    def _build_risk_metrics_section(
        self,
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState]
    ) -> ReportSection:
        """Build risk metrics section"""
        content = f"""
**Volatility Metrics:**
- Annualized Volatility: {risk_metrics.portfolio_volatility:.2%}

**Value at Risk:**
- VaR (95% confidence): {risk_metrics.var_95:.2%}
- VaR (99% confidence): {risk_metrics.var_99:.2%}
- CVaR (95% confidence): {risk_metrics.cvar_95:.2%}
- CVaR (99% confidence): {risk_metrics.cvar_99:.2%}

**Drawdown Analysis:**
- Current Drawdown: {risk_metrics.current_drawdown:.2%}
- Maximum Drawdown: {risk_metrics.max_drawdown:.2%}

**Risk-Adjusted Returns:**
- Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
- Sortino Ratio: {risk_metrics.sortino_ratio:.2f}
"""

        if risk_metrics.beta is not None:
            content += f"\n**Market Exposure:**\n"
            content += f"- Beta: {risk_metrics.beta:.2f}\n"
            content += f"- Correlation to Market: {risk_metrics.correlation_to_market:.2f}\n"

        return ReportSection(
            title="Risk Metrics",
            content=content,
            order=2,
            include_charts=True
        )

    def _build_performance_section(
        self,
        portfolio_data: Dict[str, Any],
        risk_metrics: RiskMetricsResult
    ) -> ReportSection:
        """Build performance metrics section"""
        content = f"""
**Return Metrics:**
- Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
- Sortino Ratio: {risk_metrics.sortino_ratio:.2f}

**Risk Metrics:**
- Volatility: {risk_metrics.portfolio_volatility:.2%}
- Maximum Drawdown: {risk_metrics.max_drawdown:.2%}

**Portfolio Statistics:**
- Number of Positions: {portfolio_data.get('position_count', 0)}
- Leverage: {portfolio_data.get('leverage', 0):.2f}x
"""

        return ReportSection(
            title="Performance Analysis",
            content=content,
            order=3,
            include_charts=True
        )

    def _build_position_analysis(
        self,
        portfolio_data: Dict[str, Any]
    ) -> ReportSection:
        """Build position analysis section"""
        positions = portfolio_data.get('positions', {})

        content = f"**Total Positions:** {len(positions)}\n\n"

        if positions:
            content += "**Top Positions:**\n"
            # Sort by absolute value
            sorted_positions = sorted(
                positions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for i, (symbol, value) in enumerate(sorted_positions[:10], 1):
                content += f"{i}. {symbol}: ${value:,.2f}\n"

        return ReportSection(
            title="Position Analysis",
            content=content,
            order=4,
            include_charts=False
        )

    def _build_var_analysis(
        self,
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState]
    ) -> ReportSection:
        """Build VaR analysis section"""
        content = f"""
**Value at Risk (VaR) Analysis:**

**Confidence Levels:**
- 95% VaR: {risk_metrics.var_95:.2%}
- 99% VaR: {risk_metrics.var_99:.2%}

**Conditional Value at Risk (CVaR):**
- 95% CVaR: {risk_metrics.cvar_95:.2%}
- 99% CVaR: {risk_metrics.cvar_99:.2%}

**Interpretation:**
The VaR represents the maximum expected loss over the reporting period at the given confidence level.
CVaR (Expected Shortfall) represents the average loss when VaR is breached.
"""

        return ReportSection(
            title="VaR Analysis",
            content=content,
            order=5,
            include_charts=True
        )

    def _build_recommendations(
        self,
        risk_metrics: RiskMetricsResult,
        risk_states: List[RiskState]
    ) -> ReportSection:
        """Build recommendations section"""
        recommendations = []

        # Check volatility
        if risk_metrics.portfolio_volatility > 0.30:
            recommendations.append(
                "- HIGH VOLATILITY: Consider reducing position sizes or hedging exposure"
            )

        # Check drawdown
        if abs(risk_metrics.current_drawdown) > 0.15:
            recommendations.append(
                "- SIGNIFICANT DRAWDOWN: Review stop-loss levels and risk limits"
            )

        # Check VaR
        if risk_metrics.var_95 > 0.03:
            recommendations.append(
                "- ELEVATED VAR: Consider diversifying portfolio or reducing leverage"
            )

        # Check Sharpe ratio
        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append(
                "- LOW SHARPE RATIO: Risk-adjusted returns below target, review strategy"
            )

        if not recommendations:
            recommendations.append("- Portfolio risk metrics within acceptable ranges")

        content = "**Risk Management Recommendations:**\n\n"
        content += "\n".join(recommendations)

        return ReportSection(
            title="Recommendations",
            content=content,
            order=6,
            include_charts=False
        )

    def _generate_html_report(
        self,
        metadata: ReportMetadata,
        sections: List[ReportSection]
    ) -> str:
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{metadata.report_type.value.replace('_', ' ').title()} Report</title>
    <style>
        {self.html_styles}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.config.company_name}</h1>
            <h2>{metadata.report_type.value.replace('_', ' ').title()}</h2>
            <p class="subtitle">{metadata.frequency.value.title()} Report</p>
            <p class="date">Generated: {metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="date">Period: {metadata.period_start.strftime('%Y-%m-%d')} to {metadata.period_end.strftime('%Y-%m-%d')}</p>
        </header>

        <div class="content">
"""

        # Add sections
        for section in sorted(sections, key=lambda x: x.order):
            html += f"""
            <section>
                <h3>{section.title}</h3>
                <div class="section-content">
                    {self._markdown_to_html(section.content)}
                </div>
            </section>
"""

        html += """
        </div>

        <footer>
            <p>Report ID: {report_id}</p>
            <p>{footer_text}</p>
        </footer>
    </div>
</body>
</html>
""".format(
            report_id=metadata.report_id,
            footer_text=self.config.report_footer or "Confidential - For Internal Use Only"
        )

        return html

    def _generate_json_report(
        self,
        metadata: ReportMetadata,
        sections: List[ReportSection]
    ) -> str:
        """Generate JSON report"""
        report_data = {
            'metadata': {
                'report_id': metadata.report_id,
                'report_type': metadata.report_type.value,
                'frequency': metadata.frequency.value,
                'format': metadata.format.value,
                'generated_at': metadata.generated_at.isoformat(),
                'period_start': metadata.period_start.isoformat(),
                'period_end': metadata.period_end.isoformat(),
                'generated_by': metadata.generated_by,
                'version': metadata.version
            },
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'order': section.order
                }
                for section in sections
            ]
        }

        return json.dumps(report_data, indent=2)

    def _generate_markdown_report(
        self,
        metadata: ReportMetadata,
        sections: List[ReportSection]
    ) -> str:
        """Generate Markdown report"""
        md = f"""# {self.config.company_name}
## {metadata.report_type.value.replace('_', ' ').title()}

**Frequency:** {metadata.frequency.value.title()}
**Generated:** {metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {metadata.period_start.strftime('%Y-%m-%d')} to {metadata.period_end.strftime('%Y-%m-%d')}

---

"""

        # Add sections
        for section in sorted(sections, key=lambda x: x.order):
            md += f"## {section.title}\n\n"
            md += f"{section.content}\n\n"
            md += "---\n\n"

        md += f"\n*Report ID: {metadata.report_id}*\n"
        md += f"\n*{self.config.report_footer or 'Confidential - For Internal Use Only'}*\n"

        return md

    def _convert_html_to_pdf(self, html_content: str) -> bytes:
        """
        Convert HTML to PDF (placeholder - requires external library)

        Args:
            html_content: HTML content to convert

        Returns:
            PDF bytes
        """
        # This is a placeholder. In production, use libraries like:
        # - weasyprint
        # - pdfkit
        # - reportlab
        # For now, return HTML as bytes with warning
        warning = "<!-- PDF generation requires weasyprint or pdfkit library -->\n"
        return (warning + html_content).encode('utf-8')

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert simple markdown to HTML"""
        html = markdown_text
        # Bold
        html = html.replace('**', '<strong>', 1)
        html = html.replace('**', '</strong>', 1)
        # Lists
        html = html.replace('\n- ', '\n<li>')
        html = html.replace('<li>', '<ul><li>', 1)
        html += '</ul>'
        # Paragraphs
        html = html.replace('\n\n', '</p><p>')
        html = '<p>' + html + '</p>'

        return html

    def _generate_report_id(
        self,
        report_type: ReportType,
        date: datetime
    ) -> str:
        """Generate unique report ID"""
        return f"{report_type.value}_{date.strftime('%Y%m%d_%H%M%S')}"

    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML reports"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        header h2 {
            margin: 10px 0;
            font-size: 1.8em;
            font-weight: 300;
        }
        .subtitle {
            font-size: 1.2em;
            margin: 5px 0;
        }
        .date {
            margin: 5px 0;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        section {
            margin-bottom: 40px;
            border-left: 4px solid #667eea;
            padding-left: 20px;
        }
        section h3 {
            color: #667eea;
            font-size: 1.5em;
            margin-top: 0;
        }
        .section-content {
            line-height: 1.6;
        }
        footer {
            background-color: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #667eea;
            color: white;
        }
        .metric {
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .metric-label {
            font-weight: bold;
            color: #666;
        }
        .metric-value {
            font-size: 1.5em;
            color: #333;
        }
        """

    def get_report_history(self) -> List[Dict]:
        """
        Get history of generated reports

        Returns:
            List of report metadata dictionaries
        """
        return [
            {
                'report_id': meta.report_id,
                'report_type': meta.report_type.value,
                'frequency': meta.frequency.value,
                'format': meta.format.value,
                'generated_at': meta.generated_at.isoformat(),
                'period_start': meta.period_start.isoformat(),
                'period_end': meta.period_end.isoformat()
            }
            for meta in self.generated_reports
        ]

    def schedule_report(
        self,
        frequency: ReportFrequency,
        report_type: ReportType,
        format: ReportFormat = ReportFormat.HTML
    ) -> Dict:
        """
        Schedule automatic report generation (placeholder for future implementation)

        Args:
            frequency: Report frequency
            report_type: Type of report
            format: Output format

        Returns:
            Schedule configuration
        """
        schedule_config = {
            'frequency': frequency.value,
            'report_type': report_type.value,
            'format': format.value,
            'next_run': None,  # Would be calculated based on frequency
            'enabled': True
        }

        return schedule_config

"""
Tear sheet generator for creating comprehensive performance reports.

This module provides pyfolio-style tear sheets with:
- Equity curve visualization
- Drawdown analysis
- Returns distribution
- Rolling metrics
- Monthly/annual returns heatmaps
- Benchmark comparison
- Factor exposure analysis
- HTML/PDF report generation
"""

from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from utils.logger import get_backtest_logger
from backtesting.performance.metrics_calculator import MetricsCalculator
from backtesting.performance.risk_metrics import RiskMetrics

log = get_backtest_logger()

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class TearSheetGenerator:
    """
    Generate comprehensive tear sheets for backtest analysis.

    Provides publication-quality visualizations and performance reports
    in the style of pyfolio and similar professional tools.
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        positions: Optional[pd.DataFrame] = None,
        trades: Optional[pd.DataFrame] = None,
        title: str = "Strategy Performance",
        risk_free_rate: float = 0.02
    ):
        """
        Initialize tear sheet generator.

        Args:
            returns: Series of portfolio returns
            benchmark_returns: Optional benchmark returns
            positions: Optional DataFrame of positions over time
            trades: Optional DataFrame of executed trades
            title: Title for the tear sheet
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.positions = positions
        self.trades = trades
        self.title = title
        self.risk_free_rate = risk_free_rate

        # Initialize calculators
        self.metrics_calc = MetricsCalculator(
            self.returns,
            self.benchmark_returns,
            risk_free_rate=risk_free_rate
        )
        self.risk_calc = RiskMetrics(self.returns, self.benchmark_returns)

        # Calculate metrics once
        self.metrics = self.metrics_calc.calculate_all_metrics()
        self.risk_metrics = self.risk_calc.calculate_all_risk_metrics()

        log.info(f"Initialized TearSheetGenerator for {title}")

    def generate_full_tear_sheet(
        self,
        output_dir: str = "reports",
        filename: Optional[str] = None,
        format: str = "pdf"
    ) -> str:
        """
        Generate complete tear sheet with all visualizations.

        Args:
            output_dir: Directory for output files
            filename: Output filename (auto-generated if None)
            format: Output format ('pdf', 'png', or 'html')

        Returns:
            Path to generated file
        """
        log.info(f"Generating full tear sheet in {format} format")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tear_sheet_{timestamp}.{format}"

        output_path = Path(output_dir) / filename

        if format == "html" and PLOTLY_AVAILABLE:
            return self._generate_html_tear_sheet(output_path)
        elif format == "pdf":
            return self._generate_pdf_tear_sheet(output_path)
        else:
            return self._generate_png_tear_sheet(output_path)

    def _generate_pdf_tear_sheet(self, output_path: Path) -> str:
        """Generate PDF tear sheet with all visualizations."""
        with PdfPages(output_path) as pdf:
            # Page 1: Overview
            self._plot_overview_page()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            # Page 2: Returns Analysis
            self._plot_returns_page()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            # Page 3: Risk Analysis
            self._plot_risk_page()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            # Page 4: Rolling Metrics
            self._plot_rolling_metrics_page()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            # Page 5: Trade Analysis (if available)
            if self.trades is not None and not self.trades.empty:
                self._plot_trade_analysis_page()
                pdf.savefig(bbox_inches='tight')
                plt.close()

            # Metadata
            d = pdf.infodict()
            d['Title'] = self.title
            d['Author'] = 'Personal Quant Desk'
            d['Subject'] = 'Backtest Performance Analysis'
            d['CreationDate'] = datetime.now()

        log.success(f"PDF tear sheet saved to {output_path}")
        return str(output_path)

    def _generate_png_tear_sheet(self, output_path: Path) -> str:
        """Generate PNG tear sheet."""
        self._plot_overview_page()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        log.success(f"PNG tear sheet saved to {output_path}")
        return str(output_path)

    def _generate_html_tear_sheet(self, output_path: Path) -> str:
        """Generate interactive HTML tear sheet using Plotly."""
        if not PLOTLY_AVAILABLE:
            log.warning("Plotly not available, falling back to PDF")
            return self._generate_pdf_tear_sheet(output_path.with_suffix('.pdf'))

        # Create HTML report with plotly
        html_components = []

        # Title
        html_components.append(f"<h1>{self.title}</h1>")

        # Summary metrics
        html_components.append(self._create_html_summary())

        # Equity curve
        fig = self._create_plotly_equity_curve()
        html_components.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # Drawdown plot
        fig = self._create_plotly_drawdown()
        html_components.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # Returns distribution
        fig = self._create_plotly_returns_distribution()
        html_components.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # Monthly returns heatmap
        fig = self._create_plotly_monthly_heatmap()
        html_components.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # Combine all components
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            {''.join(html_components)}
        </body>
        </html>
        """

        output_path.write_text(html_content)

        log.success(f"HTML tear sheet saved to {output_path}")
        return str(output_path)

    # ===========================
    # Page Layouts (Matplotlib)
    # ===========================

    def _plot_overview_page(self):
        """Plot overview page with key metrics and equity curve."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # 1. Summary statistics table
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_summary_table(ax1)

        # 2. Equity curve
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_equity_curve(ax2)

        # 3. Underwater plot (drawdown)
        ax3 = fig.add_subplot(gs[2, :])
        self._plot_underwater(ax3)

    def _plot_returns_page(self):
        """Plot returns analysis page."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle(f"{self.title} - Returns Analysis", fontsize=16, fontweight='bold')

        # 1. Returns distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_returns_distribution(ax1)

        # 2. Q-Q plot
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_qq(ax2)

        # 3. Monthly returns heatmap
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_monthly_heatmap(ax3)

        # 4. Annual returns bar chart
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_annual_returns(ax4)

    def _plot_risk_page(self):
        """Plot risk analysis page."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle(f"{self.title} - Risk Analysis", fontsize=16, fontweight='bold')

        # 1. Rolling volatility
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_rolling_volatility(ax1)

        # 2. Risk metrics table
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_risk_table(ax2)

        # 3. Drawdown periods
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_top_drawdowns(ax3)

        # 4. Best/Worst periods
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_best_worst_periods(ax4)

    def _plot_rolling_metrics_page(self):
        """Plot rolling metrics page."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

        fig.suptitle(f"{self.title} - Rolling Metrics", fontsize=16, fontweight='bold')

        # 1. Rolling Sharpe
        ax1 = fig.add_subplot(gs[0])
        self._plot_rolling_sharpe(ax1)

        # 2. Rolling Beta (if benchmark available)
        ax2 = fig.add_subplot(gs[1])
        if self.benchmark_returns is not None:
            self._plot_rolling_beta(ax2)
        else:
            self._plot_rolling_sortino(ax2)

        # 3. Rolling correlation
        ax3 = fig.add_subplot(gs[2])
        if self.benchmark_returns is not None:
            self._plot_rolling_correlation(ax3)
        else:
            self._plot_cumulative_returns_by_year(ax3)

    def _plot_trade_analysis_page(self):
        """Plot trade analysis page."""
        if self.trades is None or self.trades.empty:
            return

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle(f"{self.title} - Trade Analysis", fontsize=16, fontweight='bold')

        # 1. Trade PnL distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_trade_pnl_distribution(ax1)

        # 2. Cumulative PnL
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cumulative_pnl(ax2)

        # 3. Win/Loss analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_win_loss_analysis(ax3)

        # 4. Trade duration
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_trade_duration(ax4)

        # 5. Trade statistics table
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_trade_statistics(ax5)

    # ===========================
    # Individual Plot Functions
    # ===========================

    def _plot_summary_table(self, ax):
        """Plot summary statistics table."""
        ax.axis('off')

        # Prepare data
        data = [
            ['Total Return', f"{self.metrics.get('total_return', 0):.2%}"],
            ['CAGR', f"{self.metrics.get('cagr', 0):.2%}"],
            ['Sharpe Ratio', f"{self.metrics.get('sharpe_ratio', 0):.2f}"],
            ['Sortino Ratio', f"{self.metrics.get('sortino_ratio', 0):.2f}"],
            ['Calmar Ratio', f"{self.metrics.get('calmar_ratio', 0):.2f}"],
            ['Max Drawdown', f"{self.metrics.get('max_drawdown', 0):.2%}"],
            ['Volatility', f"{self.metrics.get('annualized_volatility', 0):.2%}"],
            ['Win Rate', f"{self.metrics.get('positive_periods', 0):.2%}"],
        ]

        if self.benchmark_returns is not None:
            data.extend([
                ['Alpha', f"{self.metrics.get('alpha', 0):.2%}"],
                ['Beta', f"{self.metrics.get('beta', 0):.2f}"],
                ['Information Ratio', f"{self.metrics.get('information_ratio', 0):.2f}"],
            ])

        # Create table
        table = ax.table(
            cellText=data,
            colLabels=['Metric', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    def _plot_equity_curve(self, ax):
        """Plot equity curve."""
        cum_returns = (1 + self.returns).cumprod()

        ax.plot(cum_returns.index, cum_returns.values, label='Strategy', linewidth=2)

        if self.benchmark_returns is not None:
            benchmark_cum = (1 + self.benchmark_returns).cumprod()
            ax.plot(benchmark_cum.index, benchmark_cum.values,
                   label='Benchmark', linewidth=2, alpha=0.7)

        ax.set_ylabel('Cumulative Returns')
        ax.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_underwater(self, ax):
        """Plot underwater (drawdown) chart."""
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        ax.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)

        ax.set_ylabel('Drawdown')
        ax.set_title('Underwater Plot', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(True, alpha=0.3)

    def _plot_returns_distribution(self, ax):
        """Plot returns distribution with normal overlay."""
        returns_pct = self.returns * 100

        # Histogram
        n, bins, patches = ax.hist(returns_pct, bins=50, density=True,
                                    alpha=0.7, color='skyblue', edgecolor='black')

        # Normal distribution overlay
        mu = returns_pct.mean()
        sigma = returns_pct.std()
        x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
               label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')

        ax.set_xlabel('Daily Returns (%)')
        ax.set_ylabel('Density')
        ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_qq(self, ax):
        """Plot Q-Q plot for normality test."""
        from scipy import stats

        stats.probplot(self.returns, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_monthly_heatmap(self, ax):
        """Plot monthly returns heatmap."""
        try:
            # Resample to monthly
            monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

            # Create pivot table
            monthly_returns = monthly_returns.to_frame('returns')
            monthly_returns['year'] = monthly_returns.index.year
            monthly_returns['month'] = monthly_returns.index.month

            pivot = monthly_returns.pivot_table(
                values='returns',
                index='year',
                columns='month',
                aggfunc='sum'
            )

            # Plot heatmap
            sns.heatmap(
                pivot * 100,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax,
                cbar_kws={'label': 'Return (%)'}
            )

            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            ax.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')

            # Month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_labels)

        except Exception as e:
            log.warning(f"Could not create monthly heatmap: {str(e)}")
            ax.text(0.5, 0.5, 'Insufficient data for monthly heatmap',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _plot_annual_returns(self, ax):
        """Plot annual returns bar chart."""
        try:
            # Resample to annual
            annual_returns = self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)

            # Create bar chart
            colors = ['green' if x > 0 else 'red' for x in annual_returns]
            ax.bar(annual_returns.index.year, annual_returns.values * 100,
                  color=colors, alpha=0.7, edgecolor='black')

            # Add benchmark if available
            if self.benchmark_returns is not None:
                benchmark_annual = self.benchmark_returns.resample('Y').apply(
                    lambda x: (1 + x).prod() - 1
                )
                ax.plot(benchmark_annual.index.year, benchmark_annual.values * 100,
                       marker='o', linewidth=2, label='Benchmark', color='blue')

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Year')
            ax.set_ylabel('Return (%)')
            ax.set_title('Annual Returns', fontsize=12, fontweight='bold')
            if self.benchmark_returns is not None:
                ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        except Exception as e:
            log.warning(f"Could not create annual returns chart: {str(e)}")
            ax.text(0.5, 0.5, 'Insufficient data for annual returns',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _plot_rolling_volatility(self, ax):
        """Plot rolling volatility."""
        window = min(252, len(self.returns) // 4)
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(252) * 100

        ax.plot(rolling_vol.index, rolling_vol.values, linewidth=2)
        ax.set_ylabel('Volatility (%)')
        ax.set_title(f'Rolling {window}-Day Volatility', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_risk_table(self, ax):
        """Plot risk metrics table."""
        ax.axis('off')

        data = [
            ['VaR (95%)', f"{self.risk_metrics.get('var_historical', 0):.2%}"],
            ['CVaR (95%)', f"{self.risk_metrics.get('cvar_historical', 0):.2%}"],
            ['CDaR', f"{self.risk_metrics.get('cdar', 0):.2%}"],
            ['Max Drawdown', f"{self.risk_metrics.get('max_drawdown', 0):.2%}"],
            ['Ulcer Index', f"{self.risk_metrics.get('ulcer_index', 0):.2f}"],
            ['Pain Index', f"{self.risk_metrics.get('pain_index', 0):.2%}"],
        ]

        table = ax.table(
            cellText=data,
            colLabels=['Risk Metric', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        for i in range(2):
            table[(0, i)].set_facecolor('#FF5722')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Risk Metrics', fontsize=12, fontweight='bold', pad=20)

    def _plot_top_drawdowns(self, ax):
        """Plot top drawdown periods."""
        # Calculate drawdowns
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_periods = []

        start = None
        for i, (date, in_dd) in enumerate(is_drawdown.items()):
            if in_dd and start is None:
                start = i
            elif not in_dd and start is not None:
                end = i - 1
                dd_values = drawdown.iloc[start:end+1]
                if len(dd_values) > 0:
                    drawdown_periods.append({
                        'start': drawdown.index[start],
                        'end': drawdown.index[end],
                        'max_dd': dd_values.min(),
                        'length': end - start + 1
                    })
                start = None

        # Get top 5
        if len(drawdown_periods) > 0:
            top_dds = sorted(drawdown_periods, key=lambda x: x['max_dd'])[:5]

            data = [[
                dd['start'].strftime('%Y-%m-%d'),
                dd['end'].strftime('%Y-%m-%d'),
                f"{dd['max_dd']:.2%}",
                f"{dd['length']}"
            ] for dd in top_dds]

            ax.axis('off')
            table = ax.table(
                cellText=data,
                colLabels=['Start', 'End', 'Depth', 'Length'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            for i in range(4):
                table[(0, i)].set_facecolor('#FF5722')
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax.set_title('Top 5 Drawdown Periods', fontsize=12, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'No significant drawdown periods',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _plot_best_worst_periods(self, ax):
        """Plot best and worst return periods."""
        best_days = self.returns.nlargest(10)
        worst_days = self.returns.nsmallest(10)

        x = np.arange(10)
        width = 0.35

        ax.barh(x, best_days.values * 100, width, label='Best Days',
               color='green', alpha=0.7)
        ax.barh(x + width, worst_days.values * 100, width, label='Worst Days',
               color='red', alpha=0.7)

        ax.set_yticks(x + width / 2)
        ax.set_yticklabels(range(1, 11))
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Rank')
        ax.set_title('Best and Worst Return Days', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_rolling_sharpe(self, ax):
        """Plot rolling Sharpe ratio."""
        window = min(252, len(self.returns) // 4)
        rolling_metrics = self.metrics_calc.rolling_metrics(window, metrics=['sharpe'])

        if not rolling_metrics.empty:
            ax.plot(rolling_metrics.index, rolling_metrics['sharpe_ratio'],
                   linewidth=2, label=f'{window}-day Sharpe')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for rolling metrics',
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_rolling_sortino(self, ax):
        """Plot rolling Sortino ratio."""
        window = min(252, len(self.returns) // 4)
        rolling_metrics = self.metrics_calc.rolling_metrics(window, metrics=['sortino'])

        if not rolling_metrics.empty:
            ax.plot(rolling_metrics.index, rolling_metrics['sortino_ratio'],
                   linewidth=2, label=f'{window}-day Sortino', color='orange')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax.set_ylabel('Sortino Ratio')
            ax.set_title(f'Rolling {window}-Day Sortino Ratio', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_rolling_beta(self, ax):
        """Plot rolling beta."""
        if self.benchmark_returns is None:
            return

        window = min(252, len(self.returns) // 4)
        rolling_beta = self.risk_calc.rolling_beta(window)

        if not rolling_beta.empty:
            ax.plot(rolling_beta.index, rolling_beta.values, linewidth=2, label=f'{window}-day Beta')
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Beta = 1')
            ax.set_ylabel('Beta')
            ax.set_title(f'Rolling {window}-Day Beta', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_rolling_correlation(self, ax):
        """Plot rolling correlation with benchmark."""
        if self.benchmark_returns is None:
            return

        window = min(252, len(self.returns) // 4)
        rolling_corr = self.returns.rolling(window).corr(self.benchmark_returns)

        ax.plot(rolling_corr.index, rolling_corr.values, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Correlation')
        ax.set_title(f'Rolling {window}-Day Correlation with Benchmark',
                    fontsize=12, fontweight='bold')
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)

    def _plot_cumulative_returns_by_year(self, ax):
        """Plot cumulative returns separated by year."""
        returns_by_year = {}

        for year in self.returns.index.year.unique():
            year_returns = self.returns[self.returns.index.year == year]
            cum_returns = (1 + year_returns).cumprod() - 1
            # Normalize to start of year
            day_of_year = year_returns.index.dayofyear
            returns_by_year[year] = pd.Series(cum_returns.values * 100, index=day_of_year)

        for year, returns in returns_by_year.items():
            ax.plot(returns.index, returns.values, label=str(year), alpha=0.7)

        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Cumulative Returns by Year', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_trade_pnl_distribution(self, ax):
        """Plot trade PnL distribution."""
        if 'pnl' not in self.trades.columns:
            ax.text(0.5, 0.5, 'No PnL data available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        pnl = self.trades['pnl'].dropna()

        ax.hist(pnl, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('PnL')
        ax.set_ylabel('Frequency')
        ax.set_title('Trade PnL Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_cumulative_pnl(self, ax):
        """Plot cumulative PnL from trades."""
        if 'pnl' not in self.trades.columns:
            return

        cum_pnl = self.trades['pnl'].cumsum()

        ax.plot(self.trades.index, cum_pnl.values, linewidth=2)
        ax.set_ylabel('Cumulative PnL')
        ax.set_title('Cumulative Trade PnL', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_win_loss_analysis(self, ax):
        """Plot win/loss analysis."""
        if 'pnl' not in self.trades.columns:
            return

        winning = self.trades[self.trades['pnl'] > 0]['pnl']
        losing = self.trades[self.trades['pnl'] < 0]['pnl']

        data = [len(winning), len(losing)]
        labels = ['Winning Trades', 'Losing Trades']
        colors = ['green', 'red']

        ax.pie(data, labels=labels, colors=colors, autopct='%1.1f%%',
              startangle=90, alpha=0.7)
        ax.set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')

    def _plot_trade_duration(self, ax):
        """Plot trade duration distribution."""
        if 'date' not in self.trades.columns or len(self.trades) < 2:
            ax.text(0.5, 0.5, 'Insufficient trade data',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Calculate durations (simplified - assumes paired buy/sell)
        ax.text(0.5, 0.5, 'Trade duration analysis\nrequires paired trades',
               ha='center', va='center', transform=ax.transAxes)

    def _plot_trade_statistics(self, ax):
        """Plot trade statistics table."""
        from backtesting.performance.metrics_calculator import calculate_trade_statistics

        stats = calculate_trade_statistics(self.trades)

        ax.axis('off')

        data = [
            ['Total Trades', f"{stats['total_trades']}"],
            ['Win Rate', f"{stats['win_rate']:.2%}"],
            ['Avg Win', f"{stats['avg_win']:.2f}"],
            ['Avg Loss', f"{stats['avg_loss']:.2f}"],
            ['Payoff Ratio', f"{stats['payoff_ratio']:.2f}"],
            ['Profit Factor', f"{stats['profit_factor']:.2f}"],
        ]

        table = ax.table(
            cellText=data,
            colLabels=['Statistic', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        for i in range(2):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Trade Statistics', fontsize=12, fontweight='bold', pad=20)

    # ===========================
    # Plotly Interactive Charts
    # ===========================

    def _create_html_summary(self) -> str:
        """Create HTML summary table."""
        metrics_html = "<table><tr><th>Metric</th><th>Value</th></tr>"

        key_metrics = [
            ('Total Return', 'total_return', '.2%'),
            ('CAGR', 'cagr', '.2%'),
            ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
            ('Sortino Ratio', 'sortino_ratio', '.2f'),
            ('Max Drawdown', 'max_drawdown', '.2%'),
            ('Volatility', 'annualized_volatility', '.2%'),
        ]

        for label, key, fmt in key_metrics:
            value = self.metrics.get(key, 0)
            metrics_html += f"<tr><td>{label}</td><td>{value:{fmt}}</td></tr>"

        metrics_html += "</table>"
        return metrics_html

    def _create_plotly_equity_curve(self):
        """Create interactive equity curve with Plotly."""
        cum_returns = (1 + self.returns).cumprod()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))

        if self.benchmark_returns is not None:
            benchmark_cum = (1 + self.benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=2, dash='dash')
            ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def _create_plotly_drawdown(self):
        """Create interactive drawdown chart."""
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red'),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))

        fig.update_layout(
            title='Underwater Plot',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def _create_plotly_returns_distribution(self):
        """Create interactive returns distribution."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=self.returns * 100,
            nbinsx=50,
            name='Returns',
            marker=dict(color='skyblue', line=dict(color='black', width=1))
        ))

        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Daily Returns (%)',
            yaxis_title='Frequency',
            template='plotly_white'
        )

        return fig

    def _create_plotly_monthly_heatmap(self):
        """Create interactive monthly returns heatmap."""
        try:
            monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns = monthly_returns.to_frame('returns')
            monthly_returns['year'] = monthly_returns.index.year
            monthly_returns['month'] = monthly_returns.index.month

            pivot = monthly_returns.pivot_table(
                values='returns',
                index='year',
                columns='month',
                aggfunc='sum'
            ) * 100

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=pivot.index,
                colorscale='RdYlGn',
                zmid=0,
                text=pivot.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10}
            ))

            fig.update_layout(
                title='Monthly Returns Heatmap (%)',
                xaxis_title='Month',
                yaxis_title='Year',
                template='plotly_white'
            )

            return fig

        except Exception as e:
            log.warning(f"Could not create monthly heatmap: {str(e)}")
            return go.Figure()


def create_tear_sheet(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
    output_path: str = "reports/tear_sheet.pdf",
    title: str = "Strategy Performance"
) -> str:
    """
    Convenience function to generate a tear sheet.

    Args:
        returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns
        trades: Optional trades DataFrame
        output_path: Output file path
        title: Report title

    Returns:
        Path to generated file
    """
    generator = TearSheetGenerator(
        returns=returns,
        benchmark_returns=benchmark_returns,
        trades=trades,
        title=title
    )

    format = Path(output_path).suffix[1:]  # Get extension without dot
    output_dir = Path(output_path).parent
    filename = Path(output_path).name

    return generator.generate_full_tear_sheet(
        output_dir=str(output_dir),
        filename=filename,
        format=format
    )

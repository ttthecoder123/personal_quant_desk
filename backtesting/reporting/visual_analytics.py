"""
Interactive visualization module using Plotly.

This module provides interactive dashboards and visualizations:
- Plotly dashboards for exploration
- 3D performance surfaces
- Heatmaps for parameter grids
- Correlation matrices with clustering
- Time series animations
- Drill-down capabilities
- HTML export for sharing
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger


class InteractiveVisualizer:
    """
    Create interactive visualizations using Plotly.

    Provides rich, interactive charts that can be explored in a web browser.
    """

    def __init__(self):
        """Initialize interactive visualizer."""
        logger.info("InteractiveVisualizer initialized")

    def create_equity_curve(
        self,
        equity_curve: pd.Series,
        title: str = "Interactive Equity Curve"
    ) -> go.Figure:
        """
        Create interactive equity curve with hover details.

        Args:
            equity_curve: Portfolio value time series
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='royalblue', width=2),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.2)',
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Value</b>: $%{y:,.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        return fig

    def create_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """
        Create interactive returns distribution histogram.

        Args:
            returns: Returns series
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='steelblue',
            hovertemplate='<b>Return Range</b>: %{x:.2f}%<br>' +
                         '<b>Count</b>: %{y}<extra></extra>'
        ))

        # Add normal distribution overlay
        mean = returns.mean() * 100
        std = returns.std() * 100

        x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)

        # Scale normal distribution to match histogram
        normal_dist *= len(returns) * (returns.max() - returns.min()) * 100 / 50

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            showlegend=True,
            height=500
        )

        return fig

    def create_drawdown_chart(
        self,
        returns: pd.Series,
        title: str = "Drawdown Analysis"
    ) -> go.Figure:
        """
        Create interactive drawdown chart.

        Args:
            returns: Returns series
            title: Plot title

        Returns:
            Plotly figure
        """
        # Calculate drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=2),
            fillcolor='rgba(255, 0, 0, 0.3)',
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Drawdown</b>: %{y:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        return fig

    def create_3d_surface(
        self,
        param_grid: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = 'sharpe_ratio',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create 3D surface plot for parameter optimization.

        Args:
            param_grid: Parameter grid DataFrame
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to visualize
            title: Plot title

        Returns:
            Plotly figure
        """
        # Create pivot table
        pivot = param_grid.pivot_table(
            values=metric,
            index=param2,
            columns=param1,
            aggfunc='mean'
        )

        fig = go.Figure(data=[go.Surface(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            hovertemplate='<b>' + param1 + '</b>: %{x}<br>' +
                         '<b>' + param2 + '</b>: %{y}<br>' +
                         '<b>' + metric + '</b>: %{z:.4f}<extra></extra>'
        )])

        fig.update_layout(
            title=title or f'3D Parameter Space: {metric}',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=metric
            ),
            height=600
        )

        return fig

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """
        Create interactive correlation heatmap.

        Args:
            correlation_matrix: Correlation matrix
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>' +
                         'Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Strategy',
            yaxis_title='Strategy',
            height=600,
            width=700
        )

        return fig


class PerformanceDashboard:
    """
    Create comprehensive performance dashboard.

    Combines multiple interactive visualizations into a single dashboard.
    """

    def __init__(self):
        """Initialize performance dashboard."""
        logger.info("PerformanceDashboard initialized")

    def create_dashboard(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard.

        Args:
            results: Backtest results dictionary
            output_path: Optional path to save HTML

        Returns:
            Plotly figure
        """
        logger.info("Creating performance dashboard")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Drawdown',
                'Returns Distribution',
                'Monthly Returns Heatmap',
                'Rolling Sharpe Ratio',
                'Cumulative Returns by Year'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        # 1. Equity Curve
        if 'equity_curve' in results:
            equity = results['equity_curve']
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    mode='lines',
                    name='Equity',
                    line=dict(color='royalblue', width=2)
                ),
                row=1, col=1
            )

        # 2. Drawdown
        if 'returns' in results:
            returns = results['returns']
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )

        # 3. Returns Distribution
        if 'returns' in results:
            returns = results['returns']
            fig.add_trace(
                go.Histogram(
                    x=returns * 100,
                    nbinsx=50,
                    name='Returns',
                    marker_color='steelblue'
                ),
                row=2, col=1
            )

        # 4. Monthly Returns Heatmap
        if 'returns' in results:
            monthly_returns = self._calculate_monthly_returns(results['returns'])

            if not monthly_returns.empty:
                fig.add_trace(
                    go.Heatmap(
                        z=monthly_returns.values,
                        x=monthly_returns.columns,
                        y=monthly_returns.index,
                        colorscale='RdYlGn',
                        zmid=0,
                        text=monthly_returns.values,
                        texttemplate='%{text:.1f}%',
                        textfont={"size": 8},
                        showscale=False
                    ),
                    row=2, col=2
                )

        # 5. Rolling Sharpe Ratio
        if 'returns' in results:
            rolling_sharpe = self._calculate_rolling_sharpe(results['returns'])

            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color='green', width=2)
                ),
                row=3, col=1
            )

        # 6. Annual Returns
        if 'returns' in results:
            annual_returns = self._calculate_annual_returns(results['returns'])

            colors = ['green' if x > 0 else 'red' for x in annual_returns.values]

            fig.add_trace(
                go.Bar(
                    x=annual_returns.index,
                    y=annual_returns.values * 100,
                    name='Annual Returns',
                    marker_color=colors
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="Performance Dashboard",
            title_font_size=20
        )

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.success(f"Dashboard saved to {output_path}")

        return fig

    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns for heatmap."""
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create year-month pivot
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100
        })

        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[i-1] for i in pivot.columns]

        return pivot

    def _calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        excess_returns = returns - (risk_free_rate / 252)
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        return rolling_sharpe

    def _calculate_annual_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate annual returns."""
        annual = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        annual.index = annual.index.year

        return annual


class ParameterExplorer:
    """
    Interactive parameter exploration tool.

    Provides drill-down capabilities for parameter optimization results.
    """

    def __init__(self):
        """Initialize parameter explorer."""
        logger.info("ParameterExplorer initialized")

    def create_parameter_explorer(
        self,
        param_grid: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create interactive parameter exploration dashboard.

        Args:
            param_grid: Parameter grid with results
            metrics: List of metrics to visualize
            output_path: Optional path to save HTML

        Returns:
            Plotly figure
        """
        logger.info("Creating parameter explorer")

        if metrics is None:
            metrics = ['sharpe_ratio', 'sortino_ratio', 'total_return', 'max_drawdown']

        # Filter available metrics
        available_metrics = [m for m in metrics if m in param_grid.columns]

        if not available_metrics:
            logger.warning("No metrics available in parameter grid")
            return go.Figure()

        # Get parameter columns
        param_cols = [col for col in param_grid.columns
                     if col not in available_metrics + ['iteration', 'fold']]

        # Create figure with dropdown menu
        fig = go.Figure()

        # Add traces for each metric
        for metric in available_metrics:
            for param in param_cols:
                visible = (metric == available_metrics[0] and param == param_cols[0])

                fig.add_trace(go.Scatter(
                    x=param_grid[param],
                    y=param_grid[metric],
                    mode='markers',
                    name=f'{param} vs {metric}',
                    visible=visible,
                    marker=dict(
                        size=10,
                        color=param_grid[metric],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=metric)
                    ),
                    hovertemplate=f'<b>{param}</b>: %{{x}}<br>' +
                                 f'<b>{metric}</b>: %{{y:.4f}}<extra></extra>'
                ))

        # Create dropdown buttons
        buttons = []

        for i, metric in enumerate(available_metrics):
            for j, param in enumerate(param_cols):
                trace_idx = i * len(param_cols) + j

                visible = [False] * len(fig.data)
                visible[trace_idx] = True

                buttons.append(
                    dict(
                        label=f'{param} - {metric}',
                        method='update',
                        args=[
                            {'visible': visible},
                            {'title': f'{metric} vs {param}',
                             'xaxis': {'title': param},
                             'yaxis': {'title': metric}}
                        ]
                    )
                )

        # Update layout with dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            title=f'{available_metrics[0]} vs {param_cols[0]}',
            xaxis_title=param_cols[0],
            yaxis_title=available_metrics[0],
            height=600
        )

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.success(f"Parameter explorer saved to {output_path}")

        return fig

    def create_parallel_coordinates(
        self,
        param_grid: pd.DataFrame,
        metric_col: str = 'sharpe_ratio',
        n_top: int = 50
    ) -> go.Figure:
        """
        Create parallel coordinates plot for parameter analysis.

        Args:
            param_grid: Parameter grid DataFrame
            metric_col: Metric column for color coding
            n_top: Number of top results to show

        Returns:
            Plotly figure
        """
        # Select top N results
        top_results = param_grid.nlargest(n_top, metric_col)

        # Get parameter columns
        param_cols = [col for col in top_results.columns
                     if col not in [metric_col, 'iteration', 'fold']]

        # Prepare dimensions
        dimensions = []

        for col in param_cols:
            dimensions.append(
                dict(
                    label=col,
                    values=top_results[col]
                )
            )

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=top_results[metric_col],
                colorscale='Viridis',
                showscale=True,
                cmin=top_results[metric_col].min(),
                cmax=top_results[metric_col].max(),
                colorbar=dict(title=metric_col)
            ),
            dimensions=dimensions
        ))

        fig.update_layout(
            title=f'Parameter Parallel Coordinates (Top {n_top} by {metric_col})',
            height=600
        )

        return fig

    def create_animated_optimization(
        self,
        param_grid: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = 'sharpe_ratio'
    ) -> go.Figure:
        """
        Create animated visualization of optimization progress.

        Args:
            param_grid: Parameter grid with iteration numbers
            param1: First parameter
            param2: Second parameter
            metric: Metric to visualize

        Returns:
            Plotly figure with animation
        """
        if 'iteration' not in param_grid.columns:
            logger.warning("No iteration column found for animation")
            return go.Figure()

        # Create frames for each iteration
        frames = []
        iterations = sorted(param_grid['iteration'].unique())

        for iteration in iterations:
            iter_data = param_grid[param_grid['iteration'] <= iteration]

            frame = go.Frame(
                data=[go.Scatter(
                    x=iter_data[param1],
                    y=iter_data[param2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=iter_data[metric],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=metric)
                    ),
                    text=iter_data[metric],
                    hovertemplate=f'<b>{param1}</b>: %{{x}}<br>' +
                                 f'<b>{param2}</b>: %{{y}}<br>' +
                                 f'<b>{metric}</b>: %{{text:.4f}}<extra></extra>'
                )],
                name=str(iteration)
            )
            frames.append(frame)

        # Initial data
        initial_data = param_grid[param_grid['iteration'] == iterations[0]]

        fig = go.Figure(
            data=[go.Scatter(
                x=initial_data[param1],
                y=initial_data[param2],
                mode='markers',
                marker=dict(
                    size=10,
                    color=initial_data[metric],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=metric)
                )
            )],
            frames=frames
        )

        # Add play and pause buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True}]),
                        dict(label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])
                    ],
                    x=0.1,
                    y=1.15
                )
            ],
            sliders=[dict(
                active=0,
                steps=[dict(
                    args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}],
                    label=f.name,
                    method="animate"
                ) for f in frames],
                x=0.1,
                len=0.9,
                xanchor="left",
                y=0,
                yanchor="top"
            )],
            title='Optimization Progress Animation',
            xaxis_title=param1,
            yaxis_title=param2,
            height=600
        )

        return fig

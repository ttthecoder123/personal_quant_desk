"""
Parameter optimization reporting module.

This module provides visualization and analysis tools for parameter optimization results:
- Parameter space visualization (2D/3D heatmaps)
- Convergence plots
- Parameter sensitivity analysis
- Walk-forward results
- In-sample vs out-of-sample comparison
- Parameter stability analysis
- Overfitting detection
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.interpolate import griddata


class OptimizationReporter:
    """
    Generate reports for parameter optimization results.

    Provides comprehensive visualization and analysis of optimization runs
    to understand parameter sensitivity and identify potential overfitting.
    """

    def __init__(self):
        """Initialize optimization reporter."""
        logger.info("OptimizationReporter initialized")

    def generate_optimization_report(
        self,
        optimization_results: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Generate comprehensive optimization report.

        Args:
            optimization_results: Dictionary containing:
                - parameter_grid: DataFrame with parameters and results
                - best_params: Dict of best parameter values
                - convergence_history: List of best scores over iterations
                - walk_forward_results: Optional walk-forward analysis results
            output_dir: Directory to save report figures

        Returns:
            Dictionary mapping report types to file paths
        """
        logger.info("Generating optimization report")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_files = {}

        # Parameter space visualization
        if 'parameter_grid' in optimization_results:
            grid_fig = self.plot_parameter_grid(
                optimization_results['parameter_grid']
            )
            path = output_dir / 'parameter_grid.png'
            grid_fig.savefig(path, dpi=300, bbox_inches='tight')
            report_files['parameter_grid'] = path
            plt.close(grid_fig)

        # Convergence plot
        if 'convergence_history' in optimization_results:
            conv_fig = self.plot_convergence(
                optimization_results['convergence_history']
            )
            path = output_dir / 'convergence.png'
            conv_fig.savefig(path, dpi=300, bbox_inches='tight')
            report_files['convergence'] = path
            plt.close(conv_fig)

        # Parameter sensitivity
        if 'parameter_grid' in optimization_results:
            sens_fig = self.plot_parameter_sensitivity(
                optimization_results['parameter_grid']
            )
            path = output_dir / 'parameter_sensitivity.png'
            sens_fig.savefig(path, dpi=300, bbox_inches='tight')
            report_files['parameter_sensitivity'] = path
            plt.close(sens_fig)

        # Walk-forward results
        if 'walk_forward_results' in optimization_results:
            wf_fig = self.plot_walk_forward_results(
                optimization_results['walk_forward_results']
            )
            path = output_dir / 'walk_forward.png'
            wf_fig.savefig(path, dpi=300, bbox_inches='tight')
            report_files['walk_forward'] = path
            plt.close(wf_fig)

        logger.success(f"Optimization report generated in {output_dir}")
        return report_files

    def plot_convergence(
        self,
        convergence_history: List[float],
        title: str = "Optimization Convergence"
    ) -> plt.Figure:
        """
        Plot optimization convergence over iterations.

        Args:
            convergence_history: List of best scores per iteration
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = range(1, len(convergence_history) + 1)
        ax.plot(iterations, convergence_history, linewidth=2, marker='o',
               markersize=4, alpha=0.7)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add final value annotation
        final_value = convergence_history[-1]
        ax.annotate(f'Final: {final_value:.4f}',
                   xy=(len(convergence_history), final_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        return fig

    def plot_parameter_grid(
        self,
        parameter_grid: pd.DataFrame,
        metric_col: str = 'sharpe_ratio'
    ) -> plt.Figure:
        """
        Plot parameter grid results.

        Args:
            parameter_grid: DataFrame with parameters and metric values
            metric_col: Column name for the metric to visualize

        Returns:
            Matplotlib figure
        """
        if metric_col not in parameter_grid.columns:
            logger.warning(f"Metric {metric_col} not found in parameter grid")
            return plt.figure()

        # Get parameter columns (exclude metric columns)
        param_cols = [col for col in parameter_grid.columns
                     if col not in [metric_col, 'iteration', 'fold', 'split']]

        if len(param_cols) == 0:
            logger.warning("No parameter columns found")
            return plt.figure()

        elif len(param_cols) == 1:
            # 1D plot
            fig = self._plot_1d_parameter_space(
                parameter_grid, param_cols[0], metric_col
            )

        elif len(param_cols) == 2:
            # 2D heatmap
            fig = self._plot_2d_parameter_space(
                parameter_grid, param_cols[0], param_cols[1], metric_col
            )

        else:
            # Multiple parameters - show pairwise plots
            fig = self._plot_pairwise_parameters(
                parameter_grid, param_cols[:4], metric_col
            )

        return fig

    def _plot_1d_parameter_space(
        self,
        df: pd.DataFrame,
        param_col: str,
        metric_col: str
    ) -> plt.Figure:
        """Plot 1D parameter space."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by parameter value and calculate mean metric
        grouped = df.groupby(param_col)[metric_col].agg(['mean', 'std']).reset_index()

        ax.plot(grouped[param_col], grouped['mean'], linewidth=2, marker='o')

        if 'std' in grouped.columns and grouped['std'].notna().any():
            ax.fill_between(
                grouped[param_col],
                grouped['mean'] - grouped['std'],
                grouped['mean'] + grouped['std'],
                alpha=0.3
            )

        ax.set_xlabel(param_col, fontsize=12)
        ax.set_ylabel(metric_col, fontsize=12)
        ax.set_title(f'{metric_col} vs {param_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_2d_parameter_space(
        self,
        df: pd.DataFrame,
        param1: str,
        param2: str,
        metric_col: str
    ) -> plt.Figure:
        """Plot 2D parameter space heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create pivot table
        pivot = df.pivot_table(
            values=metric_col,
            index=param2,
            columns=param1,
            aggfunc='mean'
        )

        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=pivot.mean().mean(),
            ax=ax,
            cbar_kws={'label': metric_col}
        )

        ax.set_title(f'{metric_col} - Parameter Space Heatmap',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)

        plt.tight_layout()
        return fig

    def _plot_pairwise_parameters(
        self,
        df: pd.DataFrame,
        param_cols: List[str],
        metric_col: str
    ) -> plt.Figure:
        """Plot pairwise parameter relationships."""
        n_params = len(param_cols)
        fig, axes = plt.subplots(n_params, n_params, figsize=(15, 15))

        for i, param1 in enumerate(param_cols):
            for j, param2 in enumerate(param_cols):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: distribution
                    ax.hist(df[param1], bins=20, alpha=0.7, edgecolor='black')
                    ax.set_ylabel('Frequency')
                else:
                    # Off-diagonal: scatter plot colored by metric
                    scatter = ax.scatter(
                        df[param2],
                        df[param1],
                        c=df[metric_col],
                        cmap='RdYlGn',
                        alpha=0.6,
                        s=30
                    )

                if i == n_params - 1:
                    ax.set_xlabel(param2, fontsize=10)
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(param1, fontsize=10)
                else:
                    ax.set_yticklabels([])

        # Add colorbar
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label=metric_col)

        fig.suptitle('Pairwise Parameter Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_parameter_sensitivity(
        self,
        parameter_grid: pd.DataFrame,
        metric_col: str = 'sharpe_ratio'
    ) -> plt.Figure:
        """
        Plot parameter sensitivity analysis.

        Shows how sensitive the metric is to each parameter.

        Args:
            parameter_grid: Parameter grid DataFrame
            metric_col: Metric column name

        Returns:
            Matplotlib figure
        """
        # Get parameter columns
        param_cols = [col for col in parameter_grid.columns
                     if col not in [metric_col, 'iteration', 'fold', 'split']]

        if not param_cols:
            logger.warning("No parameter columns found")
            return plt.figure()

        # Calculate sensitivity for each parameter
        sensitivities = {}

        for param in param_cols:
            # Group by parameter and calculate variance of metric
            grouped = parameter_grid.groupby(param)[metric_col].agg(['mean', 'std', 'var'])
            sensitivities[param] = {
                'range': grouped['mean'].max() - grouped['mean'].min(),
                'variance': grouped['var'].mean(),
                'cv': grouped['std'].mean() / grouped['mean'].mean() if grouped['mean'].mean() != 0 else 0
            }

        # Plot sensitivity
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['range', 'variance', 'cv']
        titles = ['Range of Metric', 'Variance', 'Coefficient of Variation']

        for ax, metric, title in zip(axes, metrics, titles):
            values = [sensitivities[param][metric] for param in param_cols]

            ax.barh(param_cols, values, alpha=0.7, edgecolor='black')
            ax.set_xlabel(title, fontsize=12)
            ax.set_title(f'Parameter Sensitivity - {title}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_walk_forward_results(
        self,
        walk_forward_results: Dict[str, Any]
    ) -> plt.Figure:
        """
        Plot walk-forward optimization results.

        Args:
            walk_forward_results: Dictionary containing walk-forward results

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        windows = walk_forward_results.get('windows', [])

        if not windows:
            logger.warning("No walk-forward windows found")
            return fig

        # Plot 1: In-sample vs Out-of-sample performance
        ax = axes[0, 0]
        is_returns = [w.get('in_sample_return', 0) for w in windows]
        oos_returns = [w.get('out_of_sample_return', 0) for w in windows]

        x = range(len(windows))
        ax.plot(x, is_returns, marker='o', label='In-Sample', linewidth=2)
        ax.plot(x, oos_returns, marker='s', label='Out-of-Sample', linewidth=2)

        ax.set_xlabel('Window', fontsize=12)
        ax.set_ylabel('Return', fontsize=12)
        ax.set_title('In-Sample vs Out-of-Sample Returns', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Degradation (IS - OOS)
        ax = axes[0, 1]
        degradation = [is_ret - oos_ret for is_ret, oos_ret in zip(is_returns, oos_returns)]

        ax.bar(x, degradation, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        ax.set_xlabel('Window', fontsize=12)
        ax.set_ylabel('Degradation (IS - OOS)', fontsize=12)
        ax.set_title('Performance Degradation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Sharpe ratios
        ax = axes[1, 0]
        is_sharpe = [w.get('in_sample_sharpe', 0) for w in windows]
        oos_sharpe = [w.get('out_of_sample_sharpe', 0) for w in windows]

        ax.plot(x, is_sharpe, marker='o', label='In-Sample', linewidth=2)
        ax.plot(x, oos_sharpe, marker='s', label='Out-of-Sample', linewidth=2)

        ax.set_xlabel('Window', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_title('In-Sample vs Out-of-Sample Sharpe Ratios', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Parameter stability (if available)
        ax = axes[1, 1]

        # Check if parameter values are tracked across windows
        if windows and 'best_params' in windows[0]:
            param_names = list(windows[0]['best_params'].keys())

            for param_name in param_names[:3]:  # Plot up to 3 parameters
                param_values = [w['best_params'].get(param_name, np.nan) for w in windows]
                ax.plot(x, param_values, marker='o', label=param_name, linewidth=2)

            ax.set_xlabel('Window', fontsize=12)
            ax.set_ylabel('Parameter Value', fontsize=12)
            ax.set_title('Parameter Stability Across Windows', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No parameter stability data available',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        return fig


class ParameterSpaceVisualizer:
    """
    Advanced parameter space visualization.

    Provides 3D visualization and interactive exploration of parameter space.
    """

    def __init__(self):
        """Initialize parameter space visualizer."""
        logger.info("ParameterSpaceVisualizer initialized")

    def plot_3d_surface(
        self,
        parameter_grid: pd.DataFrame,
        param1: str,
        param2: str,
        metric_col: str = 'sharpe_ratio'
    ) -> plt.Figure:
        """
        Create 3D surface plot of parameter space.

        Args:
            parameter_grid: Parameter grid DataFrame
            param1: First parameter name
            param2: Second parameter name
            metric_col: Metric column name

        Returns:
            Matplotlib figure with 3D plot
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Extract data
        x = parameter_grid[param1].values
        y = parameter_grid[param2].values
        z = parameter_grid[metric_col].values

        # Create grid for surface
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate z values
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # Plot surface
        surf = ax.plot_surface(
            xi, yi, zi,
            cmap='viridis',
            alpha=0.8,
            edgecolor='none'
        )

        # Plot original points
        ax.scatter(x, y, z, c='red', marker='o', s=50, alpha=0.6)

        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)
        ax.set_zlabel(metric_col, fontsize=12)
        ax.set_title(f'3D Parameter Space: {metric_col}',
                    fontsize=14, fontweight='bold')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        return fig

    def plot_contour(
        self,
        parameter_grid: pd.DataFrame,
        param1: str,
        param2: str,
        metric_col: str = 'sharpe_ratio',
        n_levels: int = 20
    ) -> plt.Figure:
        """
        Create contour plot of parameter space.

        Args:
            parameter_grid: Parameter grid DataFrame
            param1: First parameter name
            param2: Second parameter name
            metric_col: Metric column name
            n_levels: Number of contour levels

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract data
        x = parameter_grid[param1].values
        y = parameter_grid[param2].values
        z = parameter_grid[metric_col].values

        # Create grid
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # Plot contour
        contour = ax.contourf(xi, yi, zi, levels=n_levels, cmap='RdYlGn')
        ax.contour(xi, yi, zi, levels=n_levels, colors='black',
                  linewidths=0.5, alpha=0.3)

        # Plot points
        scatter = ax.scatter(x, y, c=z, cmap='RdYlGn',
                           edgecolors='black', s=50, alpha=0.8)

        # Mark best point
        best_idx = z.argmax()
        ax.scatter(x[best_idx], y[best_idx], c='gold', marker='*',
                  s=500, edgecolors='black', linewidths=2,
                  label='Best', zorder=5)

        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)
        ax.set_title(f'Parameter Space Contour: {metric_col}',
                    fontsize=14, fontweight='bold')

        fig.colorbar(contour, ax=ax, label=metric_col)
        ax.legend()

        plt.tight_layout()
        return fig


class WalkForwardAnalyzer:
    """
    Analyze walk-forward optimization results.

    Provides tools to assess strategy robustness and detect overfitting
    using walk-forward analysis.
    """

    def __init__(self):
        """Initialize walk-forward analyzer."""
        logger.info("WalkForwardAnalyzer initialized")

    def analyze_walk_forward(
        self,
        walk_forward_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive walk-forward analysis.

        Args:
            walk_forward_results: Walk-forward results dictionary

        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing walk-forward results")

        windows = walk_forward_results.get('windows', [])

        if not windows:
            logger.warning("No walk-forward windows found")
            return {}

        analysis = {
            'n_windows': len(windows),
            'performance_metrics': self._analyze_performance(windows),
            'degradation_analysis': self._analyze_degradation(windows),
            'parameter_stability': self._analyze_parameter_stability(windows),
            'overfitting_indicators': self._detect_overfitting(windows),
        }

        return analysis

    def _analyze_performance(self, windows: List[Dict]) -> Dict:
        """Analyze average performance across windows."""
        is_returns = [w.get('in_sample_return', 0) for w in windows]
        oos_returns = [w.get('out_of_sample_return', 0) for w in windows]

        is_sharpe = [w.get('in_sample_sharpe', 0) for w in windows]
        oos_sharpe = [w.get('out_of_sample_sharpe', 0) for w in windows]

        return {
            'mean_in_sample_return': np.mean(is_returns),
            'mean_out_of_sample_return': np.mean(oos_returns),
            'std_in_sample_return': np.std(is_returns),
            'std_out_of_sample_return': np.std(oos_returns),
            'mean_in_sample_sharpe': np.mean(is_sharpe),
            'mean_out_of_sample_sharpe': np.mean(oos_sharpe),
            'consistency': sum(1 for r in oos_returns if r > 0) / len(oos_returns),
        }

    def _analyze_degradation(self, windows: List[Dict]) -> Dict:
        """Analyze performance degradation from IS to OOS."""
        is_returns = [w.get('in_sample_return', 0) for w in windows]
        oos_returns = [w.get('out_of_sample_return', 0) for w in windows]

        degradation = [is_ret - oos_ret for is_ret, oos_ret in zip(is_returns, oos_returns)]

        return {
            'mean_degradation': np.mean(degradation),
            'median_degradation': np.median(degradation),
            'max_degradation': np.max(degradation),
            'degradation_ratio': np.mean(oos_returns) / np.mean(is_returns) if np.mean(is_returns) != 0 else 0,
        }

    def _analyze_parameter_stability(self, windows: List[Dict]) -> Dict:
        """Analyze parameter stability across windows."""
        if not windows or 'best_params' not in windows[0]:
            return {}

        param_names = list(windows[0]['best_params'].keys())
        stability = {}

        for param_name in param_names:
            values = [w['best_params'].get(param_name, np.nan) for w in windows]
            values = [v for v in values if not np.isnan(v)]

            if values:
                stability[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    'range': max(values) - min(values),
                }

        return stability

    def _detect_overfitting(self, windows: List[Dict]) -> Dict:
        """Detect overfitting indicators."""
        is_returns = [w.get('in_sample_return', 0) for w in windows]
        oos_returns = [w.get('out_of_sample_return', 0) for w in windows]

        degradation = [is_ret - oos_ret for is_ret, oos_ret in zip(is_returns, oos_returns)]

        indicators = {
            'high_degradation': np.mean(degradation) > 0.02,  # 2% threshold
            'inconsistent_oos': np.std(oos_returns) > 2 * np.std(is_returns),
            'negative_oos_rate': sum(1 for r in oos_returns if r < 0) / len(oos_returns) > 0.4,
            'degradation_trend': self._test_degradation_trend(degradation),
        }

        # Overall overfitting score (0-1, higher = more overfit)
        score = sum(indicators.values()) / len(indicators)
        indicators['overfitting_score'] = score

        return indicators

    def _test_degradation_trend(self, degradation: List[float]) -> bool:
        """Test if degradation is increasing over time."""
        if len(degradation) < 3:
            return False

        # Simple linear regression test
        x = np.arange(len(degradation))
        slope, _ = np.polyfit(x, degradation, 1)

        return slope > 0.01  # Positive trend indicates increasing degradation

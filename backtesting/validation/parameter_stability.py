"""
Parameter Stability Analysis Module

Implements comprehensive parameter stability analysis including:
- Parameter sensitivity analysis (gradient-based)
- Stability across sub-periods
- Parameter distribution analysis
- Robust parameter regions identification
- Parameter correlation analysis
- Out-of-sample stability metrics
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.optimize import approx_fprime


@dataclass
class StabilityResult:
    """Container for parameter stability results."""

    analysis_name: str
    is_stable: bool
    stability_score: float
    threshold: float
    details: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'analysis_name': self.analysis_name,
            'is_stable': self.is_stable,
            'stability_score': self.stability_score,
            'threshold': self.threshold,
            'details': self.details
        }


class ParameterStabilityAnalyzer:
    """
    Comprehensive parameter stability analysis for strategy validation.

    Analyzes how sensitive strategy performance is to parameter changes
    and whether parameters are stable across different time periods.
    """

    def __init__(self, stability_threshold: float = 0.3):
        """
        Initialize parameter stability analyzer.

        Args:
            stability_threshold: Threshold for stability (0-1, higher = more stable required)
        """
        self.stability_threshold = stability_threshold
        logger.info(f"Initialized ParameterStabilityAnalyzer with threshold {stability_threshold}")

    def sensitivity_analysis(
        self,
        performance_func: Callable,
        parameters: Dict[str, float],
        perturbation: float = 0.01
    ) -> StabilityResult:
        """
        Gradient-based sensitivity analysis.

        Calculates how much performance changes with small parameter perturbations.

        Args:
            performance_func: Function that takes parameters dict and returns performance
            parameters: Dictionary of parameter names and values
            perturbation: Relative perturbation size (default 1%)

        Returns:
            StabilityResult with sensitivity metrics
        """
        logger.debug(f"Running sensitivity analysis for {len(parameters)} parameters")

        param_names = list(parameters.keys())
        param_values = np.array([parameters[k] for k in param_names])

        # Baseline performance
        baseline_perf = performance_func(parameters)

        # Calculate gradients
        sensitivities = {}
        normalized_sensitivities = {}

        for i, name in enumerate(param_names):
            # Perturb parameter
            perturbed_params = parameters.copy()
            delta = param_values[i] * perturbation
            if delta == 0:
                delta = perturbation  # For zero parameters

            perturbed_params[name] = param_values[i] + delta
            perturbed_perf = performance_func(perturbed_params)

            # Calculate sensitivity (derivative)
            sensitivity = (perturbed_perf - baseline_perf) / delta
            sensitivities[name] = sensitivity

            # Normalized sensitivity (elasticity)
            normalized = sensitivity * param_values[i] / baseline_perf if baseline_perf != 0 else 0
            normalized_sensitivities[name] = normalized

        # Overall stability score (inverse of max normalized sensitivity)
        max_sensitivity = max(abs(v) for v in normalized_sensitivities.values())
        stability_score = 1 / (1 + max_sensitivity)

        is_stable = stability_score >= self.stability_threshold

        conclusion = (
            f"Parameter sensitivity: max elasticity = {max_sensitivity:.3f}, "
            f"stability score = {stability_score:.3f} "
            f"({'stable' if is_stable else 'unstable'})"
        )

        return StabilityResult(
            analysis_name="Parameter Sensitivity Analysis",
            is_stable=is_stable,
            stability_score=stability_score,
            threshold=self.stability_threshold,
            details={
                'sensitivities': sensitivities,
                'normalized_sensitivities': normalized_sensitivities,
                'max_sensitivity': max_sensitivity,
                'baseline_performance': baseline_perf,
                'conclusion': conclusion
            }
        )

    def subperiod_stability(
        self,
        returns: pd.Series,
        optimization_func: Callable,
        n_subperiods: int = 5,
        overlap: float = 0.0
    ) -> StabilityResult:
        """
        Analyze parameter stability across sub-periods.

        Optimizes parameters on different sub-periods and checks consistency.

        Args:
            returns: Time series of returns
            optimization_func: Function that takes returns and returns optimal parameters
            n_subperiods: Number of sub-periods to analyze
            overlap: Overlap fraction between periods (0-1)

        Returns:
            StabilityResult with sub-period stability metrics
        """
        logger.debug(f"Analyzing stability across {n_subperiods} sub-periods")

        n = len(returns)
        period_length = int(n / (n_subperiods * (1 - overlap) + overlap))

        optimal_params_list = []

        for i in range(n_subperiods):
            start = int(i * period_length * (1 - overlap))
            end = start + period_length

            if end > n:
                break

            subperiod_returns = returns.iloc[start:end]

            # Optimize on sub-period
            try:
                optimal_params = optimization_func(subperiod_returns)
                optimal_params_list.append(optimal_params)
            except Exception as e:
                logger.warning(f"Optimization failed for period {i}: {e}")
                continue

        if len(optimal_params_list) < 2:
            logger.warning("Insufficient sub-periods for stability analysis")
            return StabilityResult(
                analysis_name="Sub-Period Stability",
                is_stable=False,
                stability_score=0.0,
                threshold=self.stability_threshold,
                details={'error': 'Insufficient sub-periods'}
            )

        # Convert to DataFrame for analysis
        params_df = pd.DataFrame(optimal_params_list)

        # Calculate coefficient of variation for each parameter
        stability_scores = {}
        for col in params_df.columns:
            mean = params_df[col].mean()
            std = params_df[col].std()
            cv = std / abs(mean) if mean != 0 else np.inf
            stability_scores[col] = 1 / (1 + cv)  # Higher score = more stable

        # Overall stability: average of individual parameter stabilities
        overall_stability = np.mean(list(stability_scores.values()))

        is_stable = overall_stability >= self.stability_threshold

        conclusion = (
            f"Parameter stability across {len(optimal_params_list)} sub-periods: "
            f"score = {overall_stability:.3f} "
            f"({'stable' if is_stable else 'unstable'})"
        )

        return StabilityResult(
            analysis_name="Sub-Period Stability",
            is_stable=is_stable,
            stability_score=overall_stability,
            threshold=self.stability_threshold,
            details={
                'n_subperiods': len(optimal_params_list),
                'parameter_stability_scores': stability_scores,
                'parameter_means': params_df.mean().to_dict(),
                'parameter_stds': params_df.std().to_dict(),
                'optimal_params_by_period': optimal_params_list,
                'conclusion': conclusion
            }
        )

    def parameter_distribution_analysis(
        self,
        performance_func: Callable,
        parameter_name: str,
        parameter_range: Tuple[float, float],
        n_points: int = 50,
        optimal_value: Optional[float] = None
    ) -> Dict:
        """
        Analyze performance across parameter range.

        Creates performance surface to identify robust parameter regions.

        Args:
            performance_func: Function that takes parameter value and returns performance
            parameter_name: Name of parameter to analyze
            parameter_range: (min, max) range for parameter
            n_points: Number of points to sample
            optimal_value: Optional optimal parameter value to highlight

        Returns:
            Dictionary with distribution analysis results
        """
        logger.debug(f"Analyzing distribution for {parameter_name}")

        param_values = np.linspace(parameter_range[0], parameter_range[1], n_points)
        performances = []

        for val in param_values:
            try:
                perf = performance_func(val)
                performances.append(perf)
            except Exception as e:
                logger.warning(f"Performance calculation failed at {val}: {e}")
                performances.append(np.nan)

        performances = np.array(performances)

        # Find peak and robust region
        valid_idx = ~np.isnan(performances)
        if np.sum(valid_idx) == 0:
            logger.warning("No valid performance values")
            return {'error': 'No valid performance values'}

        best_idx = np.nanargmax(performances)
        best_value = param_values[best_idx]
        best_performance = performances[best_idx]

        # Define robust region: within 95% of best performance
        robust_threshold = best_performance * 0.95
        robust_mask = performances >= robust_threshold
        robust_region = param_values[robust_mask]

        if len(robust_region) > 0:
            robust_range = (robust_region.min(), robust_region.max())
            robust_width = robust_range[1] - robust_range[0]
            parameter_width = parameter_range[1] - parameter_range[0]
            robustness = robust_width / parameter_width
        else:
            robust_range = (best_value, best_value)
            robustness = 0.0

        # Check if optimal value is in robust region
        if optimal_value is not None:
            is_optimal_robust = robust_range[0] <= optimal_value <= robust_range[1]
        else:
            is_optimal_robust = None

        return {
            'parameter_name': parameter_name,
            'parameter_values': param_values,
            'performances': performances,
            'best_value': best_value,
            'best_performance': best_performance,
            'robust_region': robust_range,
            'robustness_score': robustness,
            'is_optimal_robust': is_optimal_robust
        }

    def robust_parameter_regions(
        self,
        performance_func: Callable,
        parameters: Dict[str, Tuple[float, float]],
        n_samples: int = 1000,
        performance_threshold: float = 0.9
    ) -> StabilityResult:
        """
        Identify robust parameter regions using Monte Carlo sampling.

        Args:
            performance_func: Function that takes parameters dict and returns performance
            parameters: Dict of parameter names to (min, max) ranges
            n_samples: Number of random samples
            performance_threshold: Threshold relative to best (0-1)

        Returns:
            StabilityResult with robust regions
        """
        logger.debug(f"Identifying robust regions with {n_samples} samples")

        param_names = list(parameters.keys())
        samples = []
        performances = []

        # Random sampling
        for _ in range(n_samples):
            sample = {}
            for name, (min_val, max_val) in parameters.items():
                sample[name] = np.random.uniform(min_val, max_val)

            samples.append(sample)

            try:
                perf = performance_func(sample)
                performances.append(perf)
            except Exception as e:
                logger.warning(f"Performance calculation failed: {e}")
                performances.append(np.nan)

        performances = np.array(performances)
        samples_df = pd.DataFrame(samples)

        # Find best performance
        valid_mask = ~np.isnan(performances)
        if np.sum(valid_mask) == 0:
            return StabilityResult(
                analysis_name="Robust Parameter Regions",
                is_stable=False,
                stability_score=0.0,
                threshold=self.stability_threshold,
                details={'error': 'No valid performance values'}
            )

        best_perf = np.nanmax(performances)
        threshold_perf = best_perf * performance_threshold

        # Robust region: samples above threshold
        robust_mask = performances >= threshold_perf
        robust_samples = samples_df[robust_mask]

        # Calculate robust ranges for each parameter
        robust_ranges = {}
        for name in param_names:
            if len(robust_samples) > 0:
                robust_ranges[name] = (
                    robust_samples[name].min(),
                    robust_samples[name].max()
                )
            else:
                robust_ranges[name] = parameters[name]

        # Stability score: average of (robust_range / total_range)
        stability_scores = []
        for name in param_names:
            total_range = parameters[name][1] - parameters[name][0]
            robust_range = robust_ranges[name][1] - robust_ranges[name][0]
            if total_range > 0:
                stability_scores.append(robust_range / total_range)

        stability_score = np.mean(stability_scores) if stability_scores else 0.0
        is_stable = stability_score >= self.stability_threshold

        conclusion = (
            f"{np.sum(robust_mask)} of {n_samples} samples ({np.sum(robust_mask)/n_samples:.1%}) "
            f"achieve {performance_threshold:.0%} of optimal performance"
        )

        return StabilityResult(
            analysis_name="Robust Parameter Regions",
            is_stable=is_stable,
            stability_score=stability_score,
            threshold=self.stability_threshold,
            details={
                'robust_ranges': robust_ranges,
                'parameter_ranges': parameters,
                'n_robust_samples': int(np.sum(robust_mask)),
                'best_performance': best_perf,
                'threshold_performance': threshold_perf,
                'conclusion': conclusion
            }
        )

    def parameter_correlation_analysis(
        self,
        parameters_history: pd.DataFrame,
        performance_history: pd.Series
    ) -> Dict:
        """
        Analyze correlation between parameters and performance.

        Args:
            parameters_history: DataFrame with parameter values over time
            performance_history: Series with performance values

        Returns:
            Dictionary with correlation analysis
        """
        logger.debug("Analyzing parameter-performance correlations")

        # Calculate correlations
        correlations = {}
        p_values = {}

        for col in parameters_history.columns:
            corr, p_val = stats.pearsonr(parameters_history[col], performance_history)
            correlations[col] = corr
            p_values[col] = p_val

        # Parameter-parameter correlations
        param_corr_matrix = parameters_history.corr()

        # Identify highly correlated parameter pairs
        high_corr_pairs = []
        for i in range(len(param_corr_matrix.columns)):
            for j in range(i+1, len(param_corr_matrix.columns)):
                corr_val = param_corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'param1': param_corr_matrix.columns[i],
                        'param2': param_corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        return {
            'performance_correlations': correlations,
            'p_values': p_values,
            'parameter_correlation_matrix': param_corr_matrix.to_dict(),
            'highly_correlated_pairs': high_corr_pairs
        }

    def out_of_sample_stability(
        self,
        in_sample_params: Dict[str, float],
        out_sample_returns: pd.Series,
        performance_func: Callable,
        reoptimize: bool = True
    ) -> StabilityResult:
        """
        Test parameter stability on out-of-sample data.

        Args:
            in_sample_params: Parameters optimized on in-sample data
            out_sample_returns: Out-of-sample returns
            performance_func: Function that takes (params, returns) and returns performance
            reoptimize: Whether to reoptimize on OOS data for comparison

        Returns:
            StabilityResult with OOS stability metrics
        """
        logger.debug("Testing out-of-sample parameter stability")

        # Performance with IS parameters on OOS data
        is_params_oos_perf = performance_func(in_sample_params, out_sample_returns)

        if reoptimize:
            # Optimize on OOS data (for comparison only)
            # Note: In practice, you wouldn't do this - just for stability analysis
            logger.debug("Reoptimizing on OOS data for comparison")
            # This would need the optimization function passed in
            # For now, we'll skip this part
            oos_optimal_perf = None
            degradation = None
        else:
            oos_optimal_perf = None
            degradation = None

        # Simple stability: if OOS performance is positive and reasonable
        is_stable = is_params_oos_perf > 0

        # Normalized stability score
        stability_score = max(0, min(1, is_params_oos_perf)) if is_params_oos_perf > 0 else 0

        conclusion = (
            f"In-sample parameters achieve {is_params_oos_perf:.3f} performance out-of-sample"
        )

        return StabilityResult(
            analysis_name="Out-of-Sample Stability",
            is_stable=is_stable,
            stability_score=stability_score,
            threshold=0.0,
            details={
                'is_params': in_sample_params,
                'oos_performance': is_params_oos_perf,
                'oos_optimal_performance': oos_optimal_perf,
                'degradation': degradation,
                'conclusion': conclusion
            }
        )

    def parameter_drift_detection(
        self,
        parameters_history: pd.DataFrame,
        window_size: int = 20
    ) -> Dict:
        """
        Detect drift in optimal parameters over time.

        Args:
            parameters_history: DataFrame with parameter values over time
            window_size: Window size for rolling statistics

        Returns:
            Dictionary with drift detection results
        """
        logger.debug(f"Detecting parameter drift with window size {window_size}")

        drift_detected = {}
        drift_statistics = {}

        for col in parameters_history.columns:
            series = parameters_history[col]

            # Calculate rolling mean and std
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()

            # Detect drift using CUSUM-like approach
            cumsum = np.cumsum(series - series.mean())
            max_cumsum = np.max(np.abs(cumsum))

            # Drift detected if cumsum exceeds threshold
            drift_threshold = 3 * series.std()
            has_drift = max_cumsum > drift_threshold

            drift_detected[col] = has_drift
            drift_statistics[col] = {
                'max_cumsum': max_cumsum,
                'threshold': drift_threshold,
                'mean': series.mean(),
                'std': series.std(),
                'rolling_mean': rolling_mean.tolist(),
                'rolling_std': rolling_std.tolist()
            }

        return {
            'drift_detected': drift_detected,
            'drift_statistics': drift_statistics,
            'n_parameters_with_drift': sum(drift_detected.values())
        }

    def run_stability_analysis(
        self,
        returns: pd.Series,
        parameters: Dict[str, float],
        performance_func: Callable,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, StabilityResult]:
        """
        Run comprehensive parameter stability analysis.

        Args:
            returns: Strategy returns
            parameters: Current parameter values
            performance_func: Function to evaluate performance
            parameter_ranges: Optional parameter ranges for analysis

        Returns:
            Dictionary of stability analysis results
        """
        logger.info("Running comprehensive parameter stability analysis")

        results = {}

        # Sensitivity analysis
        try:
            results['sensitivity'] = self.sensitivity_analysis(
                lambda p: performance_func(p, returns),
                parameters
            )
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")

        # Robust parameter regions (if ranges provided)
        if parameter_ranges is not None:
            try:
                results['robust_regions'] = self.robust_parameter_regions(
                    lambda p: performance_func(p, returns),
                    parameter_ranges
                )
            except Exception as e:
                logger.error(f"Robust regions analysis failed: {e}")

        logger.info(f"Completed {len(results)} stability analyses")
        return results

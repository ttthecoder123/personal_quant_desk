"""
Validation Framework for Backtesting System

Comprehensive validation framework implementing:
- Statistical tests for performance significance
- Overfitting detection (López de Prado methods)
- Parameter stability analysis
- Regime-specific performance analysis
- Monte Carlo validation methods

This framework provides rigorous statistical validation to ensure
strategy robustness and detect overfitting in backtested strategies.
"""

from .statistical_tests import StatisticalTestResult, StatisticalTests
from .overfitting_detection import OverfittingDetector, OverfittingResult
from .parameter_stability import ParameterStabilityAnalyzer, StabilityResult
from .regime_analysis import RegimeAnalyzer, RegimeResult
from .monte_carlo_validation import MonteCarloValidator, MonteCarloResult

__all__ = [
    # Statistical Tests
    'StatisticalTests',
    'StatisticalTestResult',

    # Overfitting Detection
    'OverfittingDetector',
    'OverfittingResult',

    # Parameter Stability
    'ParameterStabilityAnalyzer',
    'StabilityResult',

    # Regime Analysis
    'RegimeAnalyzer',
    'RegimeResult',

    # Monte Carlo Validation
    'MonteCarloValidator',
    'MonteCarloResult',
]

__version__ = '1.0.0'

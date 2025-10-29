"""
Backtest Orchestrator

Main entry point for running backtests. Coordinates all backtesting components:
- Engines (event-driven, vectorized, walk-forward)
- Market simulation
- Cost modeling
- Validation
- Performance analysis
- Reporting

Based on López de Prado, Jansen, and Carver methodologies.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# Engines
from .engines import (
    EventEngine,
    VectorizedEngine,
    SimulationEngine,
    WalkForwardEngine
)

# Market simulation
from .market_simulation import (
    OrderBookSimulator,
    MarketImpactModel,
    SlippageModel,
    FillSimulator,
    CorporateActionHandler
)

# Validation
from .validation import (
    StatisticalTests,
    OverfittingDetector,
    ParameterStabilityAnalyzer,
    RegimeAnalyzer,
    MonteCarloValidator
)

# Performance
from .performance import (
    MetricsCalculator,
    TearSheetGenerator,
    AttributionAnalyzer,
    RiskMetrics,
    BenchmarkComparison
)

# Optimization
from .optimization import (
    ParameterOptimizer,
    WalkForwardOptimizer,
    GeneticOptimizer,
    CombinatorialPurgedCV,
    HyperbandOptimizer
)

# Cost modeling
from .costs import (
    ComprehensiveCommissionModel,
    DynamicSpreadModel,
    ComprehensiveBorrowCostModel,
    ComprehensiveFundingModel,
    ComprehensiveTaxModel
)

# Data handling
from .data_handling import (
    DataLoader,
    DataAligner,
    SurvivorshipBiasHandler,
    PointInTimeDatabase,
    ComprehensiveDataQualityChecker
)

# Reporting
from .reporting import (
    BacktestReportGenerator,
    TradeAnalyzer,
    ComparisonReportGenerator,
    OptimizationReportGenerator,
    VisualAnalytics
)

# Scenarios
from .scenarios import (
    HistoricalScenarios,
    SyntheticScenarios,
    StressScenarios,
    RegimeScenarios
)


class BacktestOrchestrator:
    """
    Main orchestrator for backtesting operations.

    Coordinates all components and provides high-level interface for:
    - Running backtests
    - Parameter optimization
    - Walk-forward analysis
    - Validation
    - Reporting
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        cost_config_path: Optional[str] = None,
        validation_config_path: Optional[str] = None
    ):
        """
        Initialize backtest orchestrator.

        Args:
            config_path: Path to main backtest configuration
            cost_config_path: Path to cost configuration
            validation_config_path: Path to validation configuration
        """
        # Load configurations
        self.config = self._load_config(
            config_path or "backtesting/config/backtest_config.yaml"
        )
        self.cost_config = self._load_config(
            cost_config_path or "backtesting/config/cost_config.yaml"
        )
        self.validation_config = self._load_config(
            validation_config_path or "backtesting/config/validation_config.yaml"
        )

        # Initialize components
        self._initialize_components()

        logger.info("BacktestOrchestrator initialized")

    def _load_config(self, path: str) -> Dict:
        """Load YAML configuration."""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}, using defaults")
            return {}

    def _initialize_components(self):
        """Initialize all backtesting components."""
        # Data handling
        self.data_loader = DataLoader(self.config.get('data', {}))
        self.data_aligner = DataAligner()
        self.survivorship_handler = SurvivorshipBiasHandler()
        self.quality_checker = ComprehensiveDataQualityChecker()

        # Cost models
        self.commission_model = self._create_commission_model()
        self.spread_model = DynamicSpreadModel(
            **self.cost_config.get('spreads', {}).get('dynamic', {})
        )
        self.impact_model = MarketImpactModel(
            **self.cost_config.get('market_impact', {}).get('almgren_chriss', {})
        )

        # Validation
        self.statistical_tests = StatisticalTests(
            **self.validation_config.get('statistical_tests', {})
        )
        self.overfitting_detector = OverfittingDetector()
        self.stability_analyzer = ParameterStabilityAnalyzer()
        self.regime_analyzer = RegimeAnalyzer()
        self.mc_validator = MonteCarloValidator(
            n_simulations=self.validation_config.get('monte_carlo', {}).get('n_simulations', 10000)
        )

        # Performance analytics
        self.metrics_calculator = MetricsCalculator()
        self.tear_sheet_generator = TearSheetGenerator()
        self.attribution_analyzer = AttributionAnalyzer()

        # Reporting
        self.report_generator = BacktestReportGenerator(
            output_dir=self.config.get('reporting', {}).get('output_dir', 'backtesting/results')
        )
        self.trade_analyzer = TradeAnalyzer()
        self.visual_analytics = VisualAnalytics()

        # Scenarios
        self.historical_scenarios = HistoricalScenarios()
        self.stress_scenarios = StressScenarios()

        logger.info("All components initialized")

    def _create_commission_model(self):
        """Create commission model from config."""
        commission_config = self.cost_config.get('commissions', {})
        active_model = self.cost_config.get('active_models', {}).get('commission', 'interactive_brokers')

        if active_model == 'interactive_brokers':
            from .costs import InteractiveBrokersCommissionModel
            return InteractiveBrokersCommissionModel(
                **commission_config.get('interactive_brokers', {})
            )
        else:
            return ComprehensiveCommissionModel(**commission_config)

    def run_backtest(
        self,
        strategy: Any,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000.0,
        engine_type: str = 'event_driven'
    ) -> Dict[str, Any]:
        """
        Run single backtest.

        Args:
            strategy: Strategy instance
            symbols: List of symbols to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            engine_type: 'event_driven', 'vectorized', or 'simulation'

        Returns:
            Backtest results dictionary
        """
        logger.info(f"Running backtest: {strategy.name} ({start_date} to {end_date})")

        # Load and prepare data
        data = self._load_data(symbols, start_date, end_date)

        # Run quality checks
        quality_results = self._check_data_quality(data)
        if not all(r['passed'] for r in quality_results.values()):
            logger.warning("Data quality issues detected")

        # Select and run engine
        if engine_type == 'event_driven':
            engine = EventEngine(self.config.get('simulation', {}))
            results = engine.run_backtest(
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
                market_data=data,
                strategy=strategy,
                initial_capital=initial_capital
            )
        elif engine_type == 'vectorized':
            engine = VectorizedEngine(self.config.get('simulation', {}))
            signals = strategy.generate_signals(data)
            prices = data[[f"{s}_Close" for s in symbols]]
            results = engine.run_backtest(signals, prices)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        # Calculate comprehensive metrics
        returns = results['returns'] if 'returns' in results else results['equity_curve'].pct_change()
        metrics = self.metrics_calculator.calculate_all_metrics(
            returns=returns,
            prices=results['equity_curve'],
            trades=results.get('trades', pd.DataFrame())
        )

        # Add metrics to results
        results['metrics'] = metrics

        logger.info(f"Backtest complete. Sharpe: {metrics.sharpe_ratio:.2f}")

        return results

    def run_walk_forward(
        self,
        strategy_generator: callable,
        symbols: List[str],
        start_date: str,
        end_date: str,
        param_space: Optional[List] = None,
        initial_capital: float = 1000000.0
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis with parameter optimization.

        Args:
            strategy_generator: Function to create strategy from parameters
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            param_space: Parameter space for optimization
            initial_capital: Starting capital

        Returns:
            Walk-forward results
        """
        logger.info(f"Running walk-forward analysis ({start_date} to {end_date})")

        # Load data
        data = self._load_data(symbols, start_date, end_date)
        prices = data[[f"{s}_Close" for s in symbols]]

        # Create walk-forward optimizer
        wf_config = self.config.get('validation', {}).get('walk_forward', {})
        wf_optimizer = WalkForwardOptimizer(
            config=wf_config,
            param_space=param_space or [],
            n_jobs=self.config.get('optimization', {}).get('n_jobs', 4)
        )

        # Run walk-forward optimization
        results = wf_optimizer.run_walk_forward(
            data=data,
            prices=prices,
            strategy_generator=strategy_generator
        )

        # Analyze results
        logger.info(f"Walk-forward complete. Windows: {results['n_windows']}")
        logger.info(f"OOS Sharpe: {results['aggregated_metrics']['sharpe_ratio']:.2f}")

        return results

    def optimize_parameters(
        self,
        strategy_generator: callable,
        param_space: List,
        symbols: List[str],
        start_date: str,
        end_date: str,
        method: str = 'bayesian',
        objective_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.

        Args:
            strategy_generator: Function to create strategy from parameters
            param_space: Parameter space definition
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            method: Optimization method
            objective_metric: Metric to optimize

        Returns:
            Optimization results
        """
        logger.info(f"Optimizing parameters using {method}")

        # Load data
        data = self._load_data(symbols, start_date, end_date)

        # Create objective function
        def objective_function(params):
            strategy = strategy_generator(**params)
            results = self.run_backtest(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            return results['metrics'][objective_metric]

        # Select optimizer
        if method == 'bayesian':
            optimizer = ParameterOptimizer(
                objective_function=objective_function,
                param_space=param_space,
                method='bayesian'
            )
            result = optimizer.optimize(
                n_iter=self.config.get('optimization', {}).get('bayesian', {}).get('n_iter', 100)
            )
        elif method == 'genetic':
            param_bounds = {p.name: (p.lower, p.upper) for p in param_space}
            optimizer = GeneticOptimizer(
                objective_function=objective_function,
                param_bounds=param_bounds
            )
            result = optimizer.optimize()
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        logger.info(f"Optimization complete. Best {objective_metric}: {result.best_score:.2f}")

        return result

    def validate_strategy(
        self,
        strategy: Any,
        symbols: List[str],
        start_date: str,
        end_date: str,
        benchmark_symbol: str = 'SPY'
    ) -> Dict[str, Any]:
        """
        Comprehensive strategy validation.

        Args:
            strategy: Strategy instance
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            benchmark_symbol: Benchmark symbol

        Returns:
            Validation results
        """
        logger.info(f"Validating strategy: {strategy.name}")

        # Run backtest
        backtest_results = self.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        returns = backtest_results['returns']

        # Load benchmark
        benchmark_data = self._load_data([benchmark_symbol], start_date, end_date)
        benchmark_returns = benchmark_data[f'{benchmark_symbol}_Close'].pct_change().dropna()

        # Run statistical tests
        stat_results = self.statistical_tests.run_all_tests(
            returns=returns,
            benchmark_returns=benchmark_returns,
            trades=backtest_results.get('trades', pd.DataFrame())
        )

        # Overfitting detection
        sharpe = backtest_results['metrics']['sharpe_ratio']
        overfit_results = self.overfitting_detector.run_overfitting_analysis(
            returns=returns,
            sharpe_ratio=sharpe,
            n_trials=100  # Assume 100 parameter combinations tested
        )

        # Regime analysis
        regime_results = self.regime_analyzer.run_comprehensive_regime_analysis(
            returns=returns,
            market_returns=benchmark_returns
        )

        # Monte Carlo validation
        mc_results = self.mc_validator.run_monte_carlo_suite(
            returns=returns,
            benchmark_returns=benchmark_returns
        )

        validation_results = {
            'statistical_tests': stat_results,
            'overfitting_detection': overfit_results,
            'regime_analysis': regime_results,
            'monte_carlo': mc_results,
            'passed': self._check_validation_criteria(stat_results, overfit_results)
        }

        logger.info(f"Validation complete. Passed: {validation_results['passed']}")

        return validation_results

    def _check_validation_criteria(
        self,
        stat_results: Any,
        overfit_results: Any
    ) -> bool:
        """Check if strategy passes validation criteria."""
        criteria = self.validation_config.get('pass_fail_criteria', {})

        # Check Sharpe ratio
        if stat_results.sharpe_ratio < criteria.get('min_sharpe_ratio', 1.0):
            return False

        # Check PBO
        if overfit_results.pbo > criteria.get('max_pbo', 0.50):
            return False

        # Check deflated Sharpe
        if overfit_results.deflated_sharpe < criteria.get('min_deflated_sharpe', 0.50):
            return False

        return True

    def run_stress_tests(
        self,
        strategy: Any,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Run comprehensive stress tests.

        Args:
            strategy: Strategy instance
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            Stress test results
        """
        logger.info("Running stress tests")

        # Load data
        data = self._load_data(symbols, start_date, end_date)

        # Historical scenarios
        historical_results = {}
        for scenario in self.historical_scenarios.get_scenarios():
            logger.debug(f"Testing scenario: {scenario.name}")
            scenario_data = data.loc[scenario.start_date:scenario.end_date]
            if len(scenario_data) > 0:
                results = self.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=scenario.start_date.strftime('%Y-%m-%d'),
                    end_date=scenario.end_date.strftime('%Y-%m-%d')
                )
                historical_results[scenario.name] = results['metrics']

        # Stress scenarios
        stress_results = self.stress_scenarios.run_stress_tests(
            returns=data[[f"{s}_Close" for s in symbols]].pct_change(),
            strategy=strategy
        )

        stress_test_results = {
            'historical_scenarios': historical_results,
            'stress_scenarios': stress_results
        }

        logger.info("Stress tests complete")

        return stress_test_results

    def generate_report(
        self,
        backtest_results: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None,
        output_format: str = 'html',
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive backtest report.

        Args:
            backtest_results: Backtest results
            validation_results: Optional validation results
            output_format: 'html', 'pdf', 'json', or 'markdown'
            output_path: Optional output path

        Returns:
            Path to generated report
        """
        logger.info(f"Generating {output_format} report")

        report_path = self.report_generator.generate_report(
            backtest_results=backtest_results,
            validation_results=validation_results,
            format=output_format,
            output_path=output_path
        )

        logger.info(f"Report generated: {report_path}")

        return report_path

    def compare_strategies(
        self,
        strategies: List[Any],
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies.

        Args:
            strategies: List of strategy instances
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(strategies)} strategies")

        # Run backtests for all strategies
        results = {}
        for strategy in strategies:
            logger.debug(f"Running backtest for {strategy.name}")
            results[strategy.name] = self.run_backtest(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )

        # Generate comparison
        comparison = ComparisonReportGenerator().generate_comparison(
            strategy_results=results
        )

        logger.info("Strategy comparison complete")

        return comparison

    def _load_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Load and prepare data for backtesting."""
        logger.debug(f"Loading data for {symbols}")

        # Load data for each symbol
        data_dict = {}
        for symbol in symbols:
            symbol_data = self.data_loader.load(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            # Rename columns to include symbol
            for col in symbol_data.columns:
                data_dict[f"{symbol}_{col}"] = symbol_data[col]

        # Combine into single DataFrame
        data = pd.DataFrame(data_dict)

        # Handle survivorship bias if enabled
        if self.config.get('data', {}).get('survivorship_bias', {}).get('enabled', True):
            data = self.survivorship_handler.filter_data({s: data for s in symbols})

        return data

    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality."""
        quality_results = {}

        for col in data.columns:
            if '_Close' in col:
                symbol = col.replace('_Close', '')
                symbol_data = data[[c for c in data.columns if c.startswith(symbol)]]

                result = self.quality_checker.check(symbol_data, symbol)
                quality_results[symbol] = {
                    'quality_score': result.quality_score,
                    'passed': result.quality_score >= self.config.get('data', {}).get('min_quality_score', 70)
                }

        return quality_results


# Convenience functions
def run_quick_backtest(
    strategy: Any,
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000.0
) -> Dict[str, Any]:
    """Quick backtest with default settings."""
    orchestrator = BacktestOrchestrator()
    return orchestrator.run_backtest(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )


def run_full_validation(
    strategy: Any,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Full validation suite."""
    orchestrator = BacktestOrchestrator()
    return orchestrator.validate_strategy(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )

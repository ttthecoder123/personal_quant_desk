"""
Optimization Framework

Comprehensive parameter optimization framework with multiple algorithms:
- Grid search, random search, Bayesian optimization
- Particle swarm, differential evolution
- Multi-objective optimization
- Genetic algorithms
- Walk-forward optimization
- Combinatorial purged cross-validation (CPCV)
- Hyperband and ASHA
"""

# Parameter Optimizer
from .parameter_optimizer import (
    ParameterOptimizer,
    ParameterSpace,
    OptimizationResult,
    OptimizationMethod,
    create_objective_function
)

# Walk-Forward Optimizer
from .walk_forward_optimizer import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    WalkForwardWindow,
    WindowResult,
    WalkForwardResult
)

# Genetic Optimizer
from .genetic_optimizer import (
    GeneticOptimizer,
    GeneticConfig,
    GeneticResult,
    Individual,
    Generation,
    FitnessMetric,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod
)

# Combinatorial Cross-Validation
from .combinatorial_cv import (
    CombinatorialPurgedCV,
    CPCVConfig,
    CPCVSplit,
    CPCVResult
)

# Hyperband Optimizer
from .hyperband_optimizer import (
    HyperbandOptimizer,
    ASHAOptimizer,
    OptunaHyperbandOptimizer,
    HyperbandConfig,
    ASHAConfig,
    HyperbandResult,
    Configuration,
    Bracket,
    ResourceType,
    plot_hyperband_results
)

__all__ = [
    # Parameter Optimizer
    'ParameterOptimizer',
    'ParameterSpace',
    'OptimizationResult',
    'OptimizationMethod',
    'create_objective_function',

    # Walk-Forward Optimizer
    'WalkForwardOptimizer',
    'WalkForwardConfig',
    'WalkForwardWindow',
    'WindowResult',
    'WalkForwardResult',

    # Genetic Optimizer
    'GeneticOptimizer',
    'GeneticConfig',
    'GeneticResult',
    'Individual',
    'Generation',
    'FitnessMetric',
    'SelectionMethod',
    'CrossoverMethod',
    'MutationMethod',

    # Combinatorial CV
    'CombinatorialPurgedCV',
    'CPCVConfig',
    'CPCVSplit',
    'CPCVResult',

    # Hyperband
    'HyperbandOptimizer',
    'ASHAOptimizer',
    'OptunaHyperbandOptimizer',
    'HyperbandConfig',
    'ASHAConfig',
    'HyperbandResult',
    'Configuration',
    'Bracket',
    'ResourceType',
    'plot_hyperband_results',
]

__version__ = '1.0.0'

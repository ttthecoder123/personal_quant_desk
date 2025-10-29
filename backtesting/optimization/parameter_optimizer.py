"""
Parameter Optimizer

Comprehensive parameter optimization framework with multiple algorithms:
- Grid search with constraints
- Random search with smart sampling
- Bayesian optimization using Gaussian Process
- Particle swarm optimization
- Differential evolution
- Multi-objective optimization (NSGA-II)
- Robust optimization (minimax)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not available, Bayesian optimization disabled")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("optuna not available, some optimization features disabled")


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    MULTI_OBJECTIVE = "multi_objective"
    ROBUST = "robust"


@dataclass
class ParameterSpace:
    """Parameter space definition."""
    name: str
    type: str  # 'continuous', 'integer', 'categorical'
    lower: Optional[float] = None
    upper: Optional[float] = None
    values: Optional[List[Any]] = None
    log_scale: bool = False

    def __post_init__(self):
        """Validate parameter space."""
        if self.type in ['continuous', 'integer']:
            if self.lower is None or self.upper is None:
                raise ValueError(f"Parameter {self.name} requires lower and upper bounds")
        elif self.type == 'categorical':
            if not self.values:
                raise ValueError(f"Parameter {self.name} requires values list")

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from parameter space."""
        if self.type == 'continuous':
            if self.log_scale:
                samples = np.exp(np.random.uniform(
                    np.log(self.lower), np.log(self.upper), n
                ))
            else:
                samples = np.random.uniform(self.lower, self.upper, n)
        elif self.type == 'integer':
            samples = np.random.randint(self.lower, self.upper + 1, n)
        else:  # categorical
            samples = np.random.choice(self.values, n)

        return samples if n > 1 else samples[0]

    def to_skopt_space(self):
        """Convert to scikit-optimize space."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize not available")

        if self.type == 'continuous':
            if self.log_scale:
                return Real(self.lower, self.upper, prior='log-uniform', name=self.name)
            return Real(self.lower, self.upper, name=self.name)
        elif self.type == 'integer':
            return Integer(self.lower, self.upper, name=self.name)
        else:
            return Categorical(self.values, name=self.name)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    n_iterations: int
    method: str
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Get summary of optimization history."""
        return pd.DataFrame(self.optimization_history)

    def get_top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N parameter sets."""
        df = self.summary()
        return df.nlargest(n, 'score')


class ParameterOptimizer:
    """
    Comprehensive parameter optimization framework.

    Supports multiple optimization algorithms with parallel execution
    and progress tracking.
    """

    def __init__(
        self,
        objective_function: Callable,
        param_space: List[ParameterSpace],
        method: OptimizationMethod = OptimizationMethod.BAYESIAN,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize parameter optimizer.

        Args:
            objective_function: Function to optimize (higher is better)
            param_space: List of ParameterSpace objects
            method: Optimization method to use
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.param_space = param_space
        self.method = method
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.random_state = random_state

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

        # Optimization history
        self.history: List[Dict[str, Any]] = []
        self.iteration = 0

        logger.info(f"ParameterOptimizer initialized with {method.value} method")

    def optimize(
        self,
        n_iter: int = 100,
        constraints: Optional[List[Callable]] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run parameter optimization.

        Args:
            n_iter: Number of iterations
            constraints: List of constraint functions
            verbose: Print progress

        Returns:
            OptimizationResult object
        """
        logger.info(f"Starting optimization with {n_iter} iterations")

        # Route to appropriate optimization method
        if self.method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search(constraints, verbose)
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search(n_iter, constraints, verbose)
        elif self.method == OptimizationMethod.BAYESIAN:
            result = self._bayesian_optimization(n_iter, constraints, verbose)
        elif self.method == OptimizationMethod.PARTICLE_SWARM:
            result = self._particle_swarm_optimization(n_iter, constraints, verbose)
        elif self.method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = self._differential_evolution(n_iter, constraints, verbose)
        elif self.method == OptimizationMethod.MULTI_OBJECTIVE:
            result = self._multi_objective_optimization(n_iter, constraints, verbose)
        elif self.method == OptimizationMethod.ROBUST:
            result = self._robust_optimization(n_iter, constraints, verbose)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        logger.info(f"Optimization complete. Best score: {result.best_score:.6f}")
        return result

    def _grid_search(
        self,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """
        Grid search optimization.

        Exhaustively searches all parameter combinations.
        """
        logger.info("Running grid search")

        # Generate grid
        grid_points = []
        for param in self.param_space:
            if param.type == 'continuous':
                points = np.linspace(param.lower, param.upper, 10)
            elif param.type == 'integer':
                points = np.arange(param.lower, param.upper + 1)
            else:
                points = param.values
            grid_points.append(points)

        # Generate all combinations
        param_grid = np.array(np.meshgrid(*grid_points)).T.reshape(-1, len(self.param_space))

        logger.info(f"Testing {len(param_grid)} parameter combinations")

        # Evaluate all combinations
        results = self._parallel_evaluate(param_grid, constraints, verbose)

        # Find best
        best_idx = np.argmax([r['score'] for r in results])
        best_params = results[best_idx]['params']
        best_score = results[best_idx]['score']

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=results,
            n_iterations=len(results),
            method="grid_search"
        )

    def _random_search(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """
        Random search with smart sampling.

        Uses Latin Hypercube Sampling for better coverage.
        """
        logger.info(f"Running random search with {n_iter} iterations")

        # Use Latin Hypercube Sampling for better space coverage
        sampler = qmc.LatinHypercube(d=len(self.param_space), seed=self.random_state)
        unit_samples = sampler.random(n=n_iter)

        # Transform to parameter space
        param_samples = []
        for i in range(n_iter):
            params = {}
            for j, param in enumerate(self.param_space):
                if param.type == 'continuous':
                    if param.log_scale:
                        value = np.exp(
                            unit_samples[i, j] * (np.log(param.upper) - np.log(param.lower)) +
                            np.log(param.lower)
                        )
                    else:
                        value = unit_samples[i, j] * (param.upper - param.lower) + param.lower
                elif param.type == 'integer':
                    value = int(unit_samples[i, j] * (param.upper - param.lower + 1) + param.lower)
                else:  # categorical
                    idx = int(unit_samples[i, j] * len(param.values))
                    idx = min(idx, len(param.values) - 1)
                    value = param.values[idx]

                params[param.name] = value
            param_samples.append(params)

        # Evaluate samples
        results = []
        if self.n_jobs and self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self._evaluate_params, params, constraints): params
                    for params in param_samples
                }

                for i, future in enumerate(as_completed(futures)):
                    score = future.result()
                    params = futures[future]
                    results.append({'params': params, 'score': score, 'iteration': i})

                    if verbose and (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{n_iter} iterations")
        else:
            for i, params in enumerate(param_samples):
                score = self._evaluate_params(params, constraints)
                results.append({'params': params, 'score': score, 'iteration': i})

                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{n_iter} iterations")

        # Find best
        best_idx = np.argmax([r['score'] for r in results])
        best_params = results[best_idx]['params']
        best_score = results[best_idx]['score']

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=results,
            n_iterations=n_iter,
            method="random_search"
        )

    def _bayesian_optimization(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """
        Bayesian optimization using Gaussian Process.

        Uses scikit-optimize or optuna if available.
        """
        if OPTUNA_AVAILABLE:
            return self._bayesian_optuna(n_iter, constraints, verbose)
        elif SKOPT_AVAILABLE:
            return self._bayesian_skopt(n_iter, constraints, verbose)
        else:
            logger.warning("No Bayesian optimization library available, falling back to random search")
            return self._random_search(n_iter, constraints, verbose)

    def _bayesian_skopt(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """Bayesian optimization using scikit-optimize."""
        logger.info(f"Running Bayesian optimization (scikit-optimize) with {n_iter} iterations")

        # Convert parameter space
        space = [param.to_skopt_space() for param in self.param_space]

        # Define objective function
        @use_named_args(space)
        def objective(**params):
            # Return negative score (gp_minimize minimizes)
            score = self._evaluate_params(params, constraints)
            self.history.append({'params': params.copy(), 'score': score, 'iteration': len(self.history)})
            return -score

        # Run optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            random_state=self.random_state,
            verbose=verbose,
            n_jobs=self.n_jobs
        )

        # Extract best parameters
        best_params = {param.name: value for param, value in zip(self.param_space, result.x)}
        best_score = -result.fun

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=self.history,
            n_iterations=n_iter,
            method="bayesian_skopt"
        )

    def _bayesian_optuna(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """Bayesian optimization using Optuna."""
        logger.info(f"Running Bayesian optimization (Optuna) with {n_iter} iterations")

        # Suppress optuna logs if not verbose
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            for param in self.param_space:
                if param.type == 'continuous':
                    if param.log_scale:
                        params[param.name] = trial.suggest_float(
                            param.name, param.lower, param.upper, log=True
                        )
                    else:
                        params[param.name] = trial.suggest_float(
                            param.name, param.lower, param.upper
                        )
                elif param.type == 'integer':
                    params[param.name] = trial.suggest_int(
                        param.name, param.lower, param.upper
                    )
                else:  # categorical
                    params[param.name] = trial.suggest_categorical(
                        param.name, param.values
                    )

            score = self._evaluate_params(params, constraints)
            self.history.append({'params': params.copy(), 'score': score, 'iteration': len(self.history)})
            return score

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        # Run optimization
        study.optimize(objective, n_trials=n_iter, n_jobs=self.n_jobs)

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_history=self.history,
            n_iterations=n_iter,
            method="bayesian_optuna",
            additional_metrics={'study': study}
        )

    def _particle_swarm_optimization(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """
        Particle Swarm Optimization.

        Uses swarm intelligence to find optimal parameters.
        """
        logger.info(f"Running Particle Swarm Optimization with {n_iter} iterations")

        # PSO parameters
        n_particles = 30
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter

        # Filter continuous/integer parameters only
        continuous_params = [p for p in self.param_space if p.type in ['continuous', 'integer']]
        categorical_params = [p for p in self.param_space if p.type == 'categorical']

        if not continuous_params:
            logger.warning("PSO requires continuous parameters, falling back to random search")
            return self._random_search(n_iter, constraints, verbose)

        # Initialize particles
        particles = np.random.uniform(
            low=[p.lower for p in continuous_params],
            high=[p.upper for p in continuous_params],
            size=(n_particles, len(continuous_params))
        )

        velocities = np.zeros_like(particles)
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(n_particles, -np.inf)

        global_best_position = None
        global_best_score = -np.inf

        # Main PSO loop
        for iteration in range(n_iter):
            # Evaluate particles
            for i in range(n_particles):
                # Build parameter dict
                params = {}
                for j, param in enumerate(continuous_params):
                    value = particles[i, j]
                    if param.type == 'integer':
                        value = int(round(value))
                    params[param.name] = value

                # Add categorical parameters (random for now)
                for param in categorical_params:
                    params[param.name] = np.random.choice(param.values)

                # Evaluate
                score = self._evaluate_params(params, constraints)
                self.history.append({
                    'params': params.copy(),
                    'score': score,
                    'iteration': iteration
                })

                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()

                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()

            # Update velocities and positions
            r1, r2 = np.random.random(2)
            velocities = (
                w * velocities +
                c1 * r1 * (personal_best_positions - particles) +
                c2 * r2 * (global_best_position - particles)
            )
            particles += velocities

            # Apply bounds
            for j, param in enumerate(continuous_params):
                particles[:, j] = np.clip(particles[:, j], param.lower, param.upper)

            if verbose and (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{n_iter}, Best score: {global_best_score:.6f}")

        # Build best parameters
        best_params = {}
        for j, param in enumerate(continuous_params):
            value = global_best_position[j]
            if param.type == 'integer':
                value = int(round(value))
            best_params[param.name] = value

        return OptimizationResult(
            best_params=best_params,
            best_score=global_best_score,
            optimization_history=self.history,
            n_iterations=n_iter,
            method="particle_swarm"
        )

    def _differential_evolution(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """
        Differential Evolution optimization.

        Uses scipy's differential_evolution implementation.
        """
        logger.info(f"Running Differential Evolution with {n_iter} iterations")

        # Filter continuous parameters only
        continuous_params = [p for p in self.param_space if p.type in ['continuous', 'integer']]
        categorical_params = [p for p in self.param_space if p.type == 'categorical']

        if not continuous_params:
            logger.warning("DE requires continuous parameters, falling back to random search")
            return self._random_search(n_iter, constraints, verbose)

        # Define bounds
        bounds = [(p.lower, p.upper) for p in continuous_params]

        # Define objective function (minimize negative score)
        def objective(x):
            params = {}
            for i, param in enumerate(continuous_params):
                value = x[i]
                if param.type == 'integer':
                    value = int(round(value))
                params[param.name] = value

            # Add categorical parameters
            for param in categorical_params:
                params[param.name] = np.random.choice(param.values)

            score = self._evaluate_params(params, constraints)
            self.history.append({'params': params.copy(), 'score': score, 'iteration': len(self.history)})
            return -score  # Minimize negative score

        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=n_iter // 15,  # DE uses population-based iterations
            seed=self.random_state,
            workers=self.n_jobs if self.n_jobs else 1,
            disp=verbose
        )

        # Extract best parameters
        best_params = {}
        for i, param in enumerate(continuous_params):
            value = result.x[i]
            if param.type == 'integer':
                value = int(round(value))
            best_params[param.name] = value

        best_score = -result.fun

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=self.history,
            n_iterations=len(self.history),
            method="differential_evolution"
        )

    def _multi_objective_optimization(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """
        Multi-objective optimization using NSGA-II.

        Returns Pareto frontier of solutions.
        """
        logger.info(f"Running multi-objective optimization with {n_iter} iterations")
        logger.warning("Multi-objective optimization returning single best solution")

        # For now, use weighted sum approach
        # TODO: Implement proper NSGA-II
        return self._random_search(n_iter, constraints, verbose)

    def _robust_optimization(
        self,
        n_iter: int,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> OptimizationResult:
        """
        Robust optimization (minimax).

        Finds parameters that perform well under uncertainty.
        """
        logger.info(f"Running robust optimization with {n_iter} iterations")

        # Use random search with robustness testing
        candidate_results = self._random_search(n_iter // 2, constraints, verbose)

        # Test top candidates under perturbations
        top_candidates = candidate_results.get_top_n(min(20, n_iter // 10))

        robust_results = []
        for idx, row in top_candidates.iterrows():
            params = {k: v for k, v in row.items() if k not in ['score', 'iteration']}

            # Test with perturbations
            scores = []
            for _ in range(5):
                # Add noise to continuous parameters
                perturbed_params = params.copy()
                for param in self.param_space:
                    if param.type == 'continuous' and param.name in perturbed_params:
                        noise = np.random.uniform(-0.1, 0.1) * (param.upper - param.lower)
                        perturbed_params[param.name] += noise
                        perturbed_params[param.name] = np.clip(
                            perturbed_params[param.name], param.lower, param.upper
                        )

                score = self._evaluate_params(perturbed_params, constraints)
                scores.append(score)

            # Use worst-case score
            robust_score = min(scores)
            robust_results.append({
                'params': params,
                'score': robust_score,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            })

        # Find best robust solution
        best_idx = np.argmax([r['score'] for r in robust_results])
        best_result = robust_results[best_idx]

        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            optimization_history=self.history,
            n_iterations=len(self.history),
            method="robust",
            additional_metrics={
                'mean_score': best_result['mean_score'],
                'std_score': best_result['std_score']
            }
        )

    def _evaluate_params(
        self,
        params: Dict[str, Any],
        constraints: Optional[List[Callable]]
    ) -> float:
        """
        Evaluate parameters with constraints.

        Args:
            params: Parameter dictionary
            constraints: List of constraint functions

        Returns:
            Score (returns -inf if constraints violated)
        """
        # Check constraints
        if constraints:
            for constraint in constraints:
                if not constraint(params):
                    return -np.inf

        # Evaluate objective function
        try:
            score = self.objective_function(params)
            return score if not np.isnan(score) else -np.inf
        except Exception as e:
            logger.warning(f"Error evaluating parameters: {e}")
            return -np.inf

    def _parallel_evaluate(
        self,
        param_array: np.ndarray,
        constraints: Optional[List[Callable]],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """Evaluate parameters in parallel."""
        results = []

        # Convert array to list of dicts
        param_dicts = []
        for params_values in param_array:
            params = {
                param.name: params_values[i]
                for i, param in enumerate(self.param_space)
            }
            param_dicts.append(params)

        if self.n_jobs and self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self._evaluate_params, params, constraints): params
                    for params in param_dicts
                }

                for i, future in enumerate(as_completed(futures)):
                    score = future.result()
                    params = futures[future]
                    results.append({'params': params, 'score': score, 'iteration': i})

                    if verbose and (i + 1) % 10 == 0:
                        logger.info(f"Evaluated {i + 1}/{len(param_dicts)} parameter sets")
        else:
            for i, params in enumerate(param_dicts):
                score = self._evaluate_params(params, constraints)
                results.append({'params': params, 'score': score, 'iteration': i})

                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(param_dicts)} parameter sets")

        return results


def create_objective_function(
    backtest_engine: Any,
    data: pd.DataFrame,
    prices: pd.DataFrame,
    strategy_generator: Callable,
    metric: str = 'sharpe_ratio'
) -> Callable:
    """
    Create objective function for optimization.

    Args:
        backtest_engine: Backtest engine instance
        data: Feature data
        prices: Price data
        strategy_generator: Function to create strategy from parameters
        metric: Metric to optimize ('sharpe_ratio', 'sortino_ratio', 'calmar_ratio')

    Returns:
        Objective function
    """
    def objective(params: Dict[str, Any]) -> float:
        """Objective function to maximize."""
        try:
            # Create strategy with parameters
            strategy = strategy_generator(**params)

            # Generate signals
            signals = strategy.generate_signals(data, prices)

            # Calculate returns
            price_returns = prices.pct_change()
            if isinstance(signals, pd.DataFrame):
                strategy_returns = (signals.shift(1) * price_returns).mean(axis=1)
            else:
                strategy_returns = signals.shift(1) * price_returns

            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) < 10:
                return -np.inf

            # Calculate metric
            if metric == 'sharpe_ratio':
                if strategy_returns.std() > 0:
                    return (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
                return -np.inf
            elif metric == 'sortino_ratio':
                downside = strategy_returns[strategy_returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    return (strategy_returns.mean() / downside.std()) * np.sqrt(252)
                return -np.inf
            elif metric == 'calmar_ratio':
                cumulative = (1 + strategy_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(drawdown.min())
                if max_dd > 0:
                    annual_return = strategy_returns.mean() * 252
                    return annual_return / max_dd
                return -np.inf
            else:
                raise ValueError(f"Unknown metric: {metric}")

        except Exception as e:
            logger.debug(f"Error in objective function: {e}")
            return -np.inf

    return objective

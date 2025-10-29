"""
Hyperband and ASHA Optimizer

Implements Hyperband and Async Successive Halving Algorithm (ASHA)
for efficient hyperparameter search with early stopping.

Based on:
- "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
- "A System for Massively Parallel Hyperparameter Tuning"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from enum import Enum
import time
import warnings

import numpy as np
import pandas as pd
from loguru import logger

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("optuna not available, some Hyperband features disabled")


class ResourceType(Enum):
    """Type of resource for successive halving."""
    EPOCHS = "epochs"
    DATA_SIZE = "data_size"
    TIME = "time"


@dataclass
class HyperbandConfig:
    """Configuration for Hyperband."""
    max_resource: int = 81  # Maximum resource (e.g., epochs, data size)
    reduction_factor: int = 3  # Reduction factor (eta)
    resource_type: ResourceType = ResourceType.EPOCHS
    early_stopping: bool = True
    aggressive_pruning: bool = False  # More aggressive early stopping


@dataclass
class ASHAConfig:
    """Configuration for ASHA (Async Successive Halving)."""
    max_resource: int = 81
    reduction_factor: int = 3
    resource_type: ResourceType = ResourceType.EPOCHS
    grace_period: int = 10  # Minimum resource before pruning
    n_brackets: int = 4


@dataclass
class Configuration:
    """Single configuration to evaluate."""
    config_id: int
    params: Dict[str, Any]
    resource_allocated: int = 0
    score: float = -np.inf
    history: List[Tuple[int, float]] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, pruned


@dataclass
class Bracket:
    """Successive halving bracket."""
    bracket_id: int
    max_resource: int
    reduction_factor: int
    configurations: List[Configuration]
    rung_levels: List[int]
    current_rung: int = 0


@dataclass
class HyperbandResult:
    """Results from Hyperband optimization."""
    best_config: Configuration
    all_configurations: List[Configuration]
    brackets: List[Bracket]
    total_evaluations: int
    wall_time: float
    convergence_history: List[Tuple[int, float]]


class HyperbandOptimizer:
    """
    Hyperband optimizer for efficient hyperparameter search.

    Uses successive halving to efficiently allocate resources
    to promising configurations while early stopping poor ones.
    """

    def __init__(
        self,
        objective_function: Callable,
        param_sampler: Callable,
        config: Optional[HyperbandConfig] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize Hyperband optimizer.

        Args:
            objective_function: Function(params, resource) -> score
            param_sampler: Function() -> params dict
            config: Hyperband configuration
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.objective_function = objective_function
        self.param_sampler = param_sampler
        self.config = config or HyperbandConfig()
        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # State tracking
        self.total_evaluations = 0
        self.configuration_counter = 0
        self.convergence_history: List[Tuple[int, float]] = []

        logger.info(
            f"HyperbandOptimizer initialized: max_resource={self.config.max_resource}, "
            f"eta={self.config.reduction_factor}"
        )

    def optimize(
        self,
        n_configurations: Optional[int] = None,
        verbose: bool = True
    ) -> HyperbandResult:
        """
        Run Hyperband optimization.

        Args:
            n_configurations: Total number of configurations to try (None = auto)
            verbose: Print progress

        Returns:
            HyperbandResult object
        """
        start_time = time.time()
        logger.info("Starting Hyperband optimization")

        # Calculate Hyperband schedule
        brackets = self._generate_brackets(n_configurations)

        logger.info(f"Generated {len(brackets)} Hyperband brackets")

        # Run successive halving for each bracket
        all_configurations = []
        for bracket in brackets:
            if verbose:
                logger.info(
                    f"Processing bracket {bracket.bracket_id + 1}/{len(brackets)}: "
                    f"{len(bracket.configurations)} configs, {bracket.max_resource} max resource"
                )

            bracket_configs = self._run_successive_halving(bracket, verbose)
            all_configurations.extend(bracket_configs)

        # Find best configuration
        completed_configs = [c for c in all_configurations if c.status == "completed"]
        if not completed_configs:
            logger.warning("No configurations completed successfully")
            best_config = Configuration(
                config_id=-1,
                params={},
                score=-np.inf,
                status="failed"
            )
        else:
            best_config = max(completed_configs, key=lambda c: c.score)

        wall_time = time.time() - start_time

        logger.info(
            f"Hyperband optimization complete. Best score: {best_config.score:.6f}, "
            f"Time: {wall_time:.2f}s, Evaluations: {self.total_evaluations}"
        )

        return HyperbandResult(
            best_config=best_config,
            all_configurations=all_configurations,
            brackets=brackets,
            total_evaluations=self.total_evaluations,
            wall_time=wall_time,
            convergence_history=self.convergence_history
        )

    def _generate_brackets(
        self,
        n_configurations: Optional[int]
    ) -> List[Bracket]:
        """Generate Hyperband brackets with different resource allocations."""
        max_resource = self.config.max_resource
        eta = self.config.reduction_factor

        # Calculate number of brackets
        s_max = int(np.floor(np.log(max_resource) / np.log(eta)))
        B = (s_max + 1) * max_resource

        brackets = []

        for s in range(s_max, -1, -1):
            # Calculate resources and configurations for this bracket
            n = int(np.ceil(B / max_resource * (eta ** s) / (s + 1)))
            r = max_resource * (eta ** (-s))

            # Generate rung levels
            rung_levels = [int(r * (eta ** i)) for i in range(s + 1)]

            # Sample configurations
            configurations = []
            for _ in range(n):
                params = self.param_sampler()
                config = Configuration(
                    config_id=self.configuration_counter,
                    params=params
                )
                configurations.append(config)
                self.configuration_counter += 1

            bracket = Bracket(
                bracket_id=len(brackets),
                max_resource=max_resource,
                reduction_factor=eta,
                configurations=configurations,
                rung_levels=rung_levels
            )
            brackets.append(bracket)

            if n_configurations and self.configuration_counter >= n_configurations:
                break

        return brackets

    def _run_successive_halving(
        self,
        bracket: Bracket,
        verbose: bool
    ) -> List[Configuration]:
        """Run successive halving for a bracket."""
        active_configs = bracket.configurations.copy()

        for rung_idx, resource in enumerate(bracket.rung_levels):
            if not active_configs:
                break

            if verbose:
                logger.info(
                    f"  Rung {rung_idx + 1}/{len(bracket.rung_levels)}: "
                    f"Evaluating {len(active_configs)} configs at resource={resource}"
                )

            # Evaluate all configurations at this resource level
            self._evaluate_configurations(active_configs, resource)

            # Sort by performance
            active_configs.sort(key=lambda c: c.score, reverse=True)

            # Keep top 1/eta configurations for next rung
            if rung_idx < len(bracket.rung_levels) - 1:
                n_keep = max(1, int(len(active_configs) / bracket.reduction_factor))
                pruned_configs = active_configs[n_keep:]
                active_configs = active_configs[:n_keep]

                # Mark pruned configurations
                for config in pruned_configs:
                    config.status = "pruned"

                if verbose:
                    logger.info(f"    Keeping top {n_keep} configs, pruning {len(pruned_configs)}")
            else:
                # Final rung - mark as completed
                for config in active_configs:
                    config.status = "completed"

        return bracket.configurations

    def _evaluate_configurations(
        self,
        configurations: List[Configuration],
        resource: int
    ):
        """Evaluate configurations at given resource level."""
        if self.n_jobs and self.n_jobs > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_configuration,
                        config, resource
                    ): config
                    for config in configurations
                }

                for future in as_completed(futures):
                    config = futures[future]
                    score = future.result()
                    config.score = score
                    config.resource_allocated = resource
                    config.history.append((resource, score))
                    config.status = "evaluated"
                    self.total_evaluations += 1

                    # Track convergence
                    if score > -np.inf:
                        self.convergence_history.append((self.total_evaluations, score))
        else:
            # Sequential evaluation
            for config in configurations:
                score = self._evaluate_configuration(config, resource)
                config.score = score
                config.resource_allocated = resource
                config.history.append((resource, score))
                config.status = "evaluated"
                self.total_evaluations += 1

                # Track convergence
                if score > -np.inf:
                    self.convergence_history.append((self.total_evaluations, score))

    def _evaluate_configuration(
        self,
        config: Configuration,
        resource: int
    ) -> float:
        """Evaluate a single configuration at given resource."""
        try:
            score = self.objective_function(config.params, resource)
            return score if not np.isnan(score) else -np.inf
        except Exception as e:
            logger.debug(f"Error evaluating config {config.config_id}: {e}")
            return -np.inf


class ASHAOptimizer:
    """
    Asynchronous Successive Halving Algorithm (ASHA).

    Asynchronous version of Hyperband for parallel optimization.
    """

    def __init__(
        self,
        objective_function: Callable,
        param_sampler: Callable,
        config: Optional[ASHAConfig] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize ASHA optimizer.

        Args:
            objective_function: Function(params, resource) -> score
            param_sampler: Function() -> params dict
            config: ASHA configuration
            n_jobs: Number of parallel workers
            random_state: Random seed
        """
        self.objective_function = objective_function
        self.param_sampler = param_sampler
        self.config = config or ASHAConfig()
        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # State tracking
        self.total_evaluations = 0
        self.configuration_counter = 0
        self.rungs: Dict[int, List[Configuration]] = {}

        logger.info(f"ASHAOptimizer initialized with {n_jobs} workers")

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        verbose: bool = True
    ) -> HyperbandResult:
        """
        Run ASHA optimization.

        Args:
            n_trials: Number of configurations to try
            timeout: Maximum time in seconds (None = no limit)
            verbose: Print progress

        Returns:
            HyperbandResult object
        """
        start_time = time.time()
        logger.info(f"Starting ASHA optimization with {n_trials} trials")

        # Initialize rungs
        eta = self.config.reduction_factor
        max_resource = self.config.max_resource
        grace_period = self.config.grace_period

        # Create rung levels
        rung_levels = [grace_period]
        while rung_levels[-1] * eta <= max_resource:
            rung_levels.append(int(rung_levels[-1] * eta))

        for level in rung_levels:
            self.rungs[level] = []

        logger.info(f"ASHA rung levels: {rung_levels}")

        # Track all configurations
        all_configurations = []
        best_score = -np.inf
        best_config = None

        # Main optimization loop
        trial = 0
        while trial < n_trials:
            if timeout and (time.time() - start_time) > timeout:
                logger.info("Timeout reached")
                break

            # Sample new configuration
            params = self.param_sampler()
            config = Configuration(
                config_id=self.configuration_counter,
                params=params
            )
            self.configuration_counter += 1
            all_configurations.append(config)

            # Evaluate at grace period
            resource = grace_period
            score = self._evaluate_configuration(config, resource)
            config.score = score
            config.resource_allocated = resource
            config.history.append((resource, score))
            self.total_evaluations += 1

            # Add to rung
            self.rungs[resource].append(config)

            # Promote if better than median at this rung
            current_resource = resource
            while current_resource < max_resource:
                # Get next rung level
                next_idx = rung_levels.index(current_resource) + 1
                if next_idx >= len(rung_levels):
                    break

                next_resource = rung_levels[next_idx]

                # Check if should promote
                rung_scores = [c.score for c in self.rungs[current_resource]]
                if len(rung_scores) < eta:
                    break  # Wait for more configurations

                median_score = np.median(rung_scores)
                if config.score < median_score:
                    config.status = "pruned"
                    break

                # Promote to next rung
                score = self._evaluate_configuration(config, next_resource)
                config.score = score
                config.resource_allocated = next_resource
                config.history.append((next_resource, score))
                self.total_evaluations += 1

                self.rungs[next_resource].append(config)
                current_resource = next_resource

            # Mark as completed if reached max resource
            if config.resource_allocated == max_resource:
                config.status = "completed"
            elif config.status != "pruned":
                config.status = "evaluated"

            # Track best
            if config.score > best_score:
                best_score = config.score
                best_config = config

            trial += 1

            if verbose and trial % 10 == 0:
                logger.info(
                    f"Trial {trial}/{n_trials}: Best score={best_score:.4f}, "
                    f"Evaluations={self.total_evaluations}"
                )

        wall_time = time.time() - start_time

        logger.info(
            f"ASHA optimization complete. Best score: {best_score:.6f}, "
            f"Time: {wall_time:.2f}s, Evaluations: {self.total_evaluations}"
        )

        return HyperbandResult(
            best_config=best_config or Configuration(config_id=-1, params={}, score=-np.inf),
            all_configurations=all_configurations,
            brackets=[],  # ASHA doesn't use brackets
            total_evaluations=self.total_evaluations,
            wall_time=wall_time,
            convergence_history=[]
        )

    def _evaluate_configuration(
        self,
        config: Configuration,
        resource: int
    ) -> float:
        """Evaluate configuration at given resource."""
        try:
            score = self.objective_function(config.params, resource)
            return score if not np.isnan(score) else -np.inf
        except Exception as e:
            logger.debug(f"Error evaluating config {config.config_id}: {e}")
            return -np.inf


class OptunaHyperbandOptimizer:
    """
    Hyperband optimizer using Optuna.

    Provides integration with Optuna's optimized Hyperband implementation.
    """

    def __init__(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        config: Optional[HyperbandConfig] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize Optuna Hyperband optimizer.

        Args:
            objective_function: Function(trial) -> score (Optuna style)
            param_space: Parameter space definition
            config: Hyperband configuration
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna not available")

        self.objective_function = objective_function
        self.param_space = param_space
        self.config = config or HyperbandConfig()
        self.n_jobs = n_jobs
        self.random_state = random_state

        logger.info("OptunaHyperbandOptimizer initialized")

    def optimize(
        self,
        n_trials: int = 100,
        verbose: bool = True
    ) -> HyperbandResult:
        """
        Run Optuna Hyperband optimization.

        Args:
            n_trials: Number of trials
            verbose: Print progress

        Returns:
            HyperbandResult object
        """
        logger.info(f"Starting Optuna Hyperband optimization with {n_trials} trials")

        # Create pruner
        pruner = HyperbandPruner(
            min_resource=self.config.max_resource // (self.config.reduction_factor ** 3),
            max_resource=self.config.max_resource,
            reduction_factor=self.config.reduction_factor
        )

        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        # Suppress optuna logs if not verbose
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        study.optimize(
            self.objective_function,
            n_trials=n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=verbose
        )

        # Convert results
        best_config = Configuration(
            config_id=study.best_trial.number,
            params=study.best_params,
            score=study.best_value,
            status="completed"
        )

        all_configurations = []
        for trial in study.trials:
            config = Configuration(
                config_id=trial.number,
                params=trial.params,
                score=trial.value if trial.value is not None else -np.inf,
                status="completed" if trial.state == optuna.trial.TrialState.COMPLETE else "pruned"
            )
            all_configurations.append(config)

        logger.info(f"Optuna Hyperband complete. Best score: {study.best_value:.6f}")

        return HyperbandResult(
            best_config=best_config,
            all_configurations=all_configurations,
            brackets=[],
            total_evaluations=len(study.trials),
            wall_time=0.0,  # Not tracked by Optuna
            convergence_history=[]
        )


def plot_hyperband_results(result: HyperbandResult) -> None:
    """
    Plot Hyperband optimization results.

    Args:
        result: HyperbandResult object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    # Extract data
    completed = [c for c in result.all_configurations if c.status in ["completed", "evaluated"]]
    pruned = [c for c in result.all_configurations if c.status == "pruned"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Resource allocation vs score
    if completed:
        resources = [c.resource_allocated for c in completed]
        scores = [c.score for c in completed]

        axes[0].scatter(resources, scores, alpha=0.6, s=50, label='Completed')

    if pruned:
        pruned_resources = [c.resource_allocated for c in pruned]
        pruned_scores = [c.score for c in pruned]
        axes[0].scatter(pruned_resources, pruned_scores, alpha=0.3, s=30,
                       c='red', label='Pruned', marker='x')

    axes[0].set_xlabel('Resource Allocated')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Resource Allocation vs Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Convergence history
    if result.convergence_history:
        evals, scores = zip(*result.convergence_history)
        axes[1].plot(evals, scores, 'o-', alpha=0.7, markersize=3)
        axes[1].set_xlabel('Evaluation Number')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Convergence History')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

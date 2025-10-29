"""
Genetic Algorithm Optimizer

Implements genetic algorithm for parameter optimization with:
- Smart population initialization
- Multiple fitness functions
- Various selection mechanisms
- Crossover and mutation operations
- Elitism and diversity maintenance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
from loguru import logger


class FitnessMetric(Enum):
    """Available fitness metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    OMEGA_RATIO = "omega_ratio"
    CUSTOM = "custom"


class SelectionMethod(Enum):
    """Selection methods for genetic algorithm."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"


class CrossoverMethod(Enum):
    """Crossover methods."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"


class MutationMethod(Enum):
    """Mutation methods."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    POLYNOMIAL = "polynomial"


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    n_generations: int = 100
    fitness_metric: FitnessMetric = FitnessMetric.SHARPE_RATIO
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.ADAPTIVE
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1  # Top % to preserve
    tournament_size: int = 3
    diversity_threshold: float = 0.01  # Minimum diversity to maintain
    early_stopping_patience: int = 20  # Generations without improvement


@dataclass
class Individual:
    """Single individual in the population."""
    genes: Dict[str, Any]  # Parameter values
    fitness: float = -np.inf
    age: int = 0  # Generations survived


@dataclass
class Generation:
    """Single generation in evolution."""
    generation_id: int
    population: List[Individual]
    best_individual: Individual
    avg_fitness: float
    diversity: float


@dataclass
class GeneticResult:
    """Results from genetic algorithm."""
    best_individual: Individual
    final_population: List[Individual]
    generation_history: List[Generation]
    convergence_generation: Optional[int]
    total_evaluations: int


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for parameter tuning.

    Implements a comprehensive genetic algorithm with multiple
    selection, crossover, and mutation strategies.
    """

    def __init__(
        self,
        objective_function: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        config: Optional[GeneticConfig] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize genetic optimizer.

        Args:
            objective_function: Function to maximize (fitness function)
            param_bounds: Parameter bounds {param_name: (lower, upper)}
            config: Genetic algorithm configuration
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.objective_function = objective_function
        self.param_bounds = param_bounds
        self.config = config or GeneticConfig()
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

        # State tracking
        self.generation_history: List[Generation] = []
        self.total_evaluations = 0

        logger.info(f"GeneticOptimizer initialized: pop={self.config.population_size}")

    def optimize(
        self,
        constraints: Optional[List[Callable]] = None,
        verbose: bool = True
    ) -> GeneticResult:
        """
        Run genetic algorithm optimization.

        Args:
            constraints: List of constraint functions
            verbose: Print progress

        Returns:
            GeneticResult object
        """
        logger.info(f"Starting genetic algorithm optimization for {self.config.n_generations} generations")

        # Initialize population
        population = self._initialize_population()
        self._evaluate_population(population, constraints)

        # Track best solution
        best_ever = max(population, key=lambda ind: ind.fitness)
        generations_without_improvement = 0

        # Evolution loop
        for gen in range(self.config.n_generations):
            # Selection
            selected = self._selection(population)

            # Crossover
            offspring = self._crossover(selected)

            # Mutation
            offspring = self._mutation(offspring)

            # Evaluate offspring
            self._evaluate_population(offspring, constraints)

            # Combine with elites
            n_elites = max(1, int(self.config.population_size * self.config.elitism_rate))
            elites = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:n_elites]

            # Age elites
            for elite in elites:
                elite.age += 1

            # Create new population
            population = elites + offspring[:self.config.population_size - n_elites]

            # Maintain diversity if needed
            diversity = self._calculate_diversity(population)
            if diversity < self.config.diversity_threshold:
                population = self._inject_diversity(population)
                diversity = self._calculate_diversity(population)

            # Track best individual
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_ever.fitness:
                best_ever = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Record generation
            avg_fitness = np.mean([ind.fitness for ind in population if ind.fitness != -np.inf])
            generation = Generation(
                generation_id=gen,
                population=population.copy(),
                best_individual=current_best,
                avg_fitness=avg_fitness,
                diversity=diversity
            )
            self.generation_history.append(generation)

            # Logging
            if verbose and (gen + 1) % 10 == 0:
                logger.info(
                    f"Generation {gen + 1}/{self.config.n_generations}: "
                    f"Best={current_best.fitness:.4f}, Avg={avg_fitness:.4f}, "
                    f"Diversity={diversity:.4f}"
                )

            # Early stopping
            if generations_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at generation {gen + 1}")
                break

        logger.info(f"Optimization complete. Best fitness: {best_ever.fitness:.6f}")

        convergence_gen = None
        if generations_without_improvement >= self.config.early_stopping_patience:
            convergence_gen = gen + 1 - generations_without_improvement

        return GeneticResult(
            best_individual=best_ever,
            final_population=population,
            generation_history=self.generation_history,
            convergence_generation=convergence_gen,
            total_evaluations=self.total_evaluations
        )

    def _initialize_population(self) -> List[Individual]:
        """
        Initialize population with smart seeding.

        Combines random initialization with strategic points.
        """
        population = []

        # Random initialization for most individuals
        n_random = int(self.config.population_size * 0.8)
        for _ in range(n_random):
            genes = {}
            for param_name, (lower, upper) in self.param_bounds.items():
                genes[param_name] = np.random.uniform(lower, upper)
            population.append(Individual(genes=genes))

        # Strategic initialization (boundary points, center)
        n_strategic = self.config.population_size - n_random

        # Add center point
        if n_strategic > 0:
            center_genes = {
                param_name: (lower + upper) / 2
                for param_name, (lower, upper) in self.param_bounds.items()
            }
            population.append(Individual(genes=center_genes))
            n_strategic -= 1

        # Add corner points
        if n_strategic > 0:
            param_names = list(self.param_bounds.keys())
            n_dims = len(param_names)
            n_corners = min(n_strategic, 2 ** n_dims)

            for i in range(n_corners):
                corner_genes = {}
                for j, param_name in enumerate(param_names):
                    lower, upper = self.param_bounds[param_name]
                    # Use binary representation to select corner
                    if (i >> j) & 1:
                        corner_genes[param_name] = upper
                    else:
                        corner_genes[param_name] = lower
                population.append(Individual(genes=corner_genes))
                n_strategic -= 1
                if n_strategic == 0:
                    break

        # Fill remaining with Latin Hypercube Sampling
        while len(population) < self.config.population_size:
            genes = {}
            for param_name, (lower, upper) in self.param_bounds.items():
                genes[param_name] = np.random.uniform(lower, upper)
            population.append(Individual(genes=genes))

        logger.debug(f"Initialized population of {len(population)} individuals")
        return population

    def _evaluate_population(
        self,
        population: List[Individual],
        constraints: Optional[List[Callable]]
    ):
        """Evaluate fitness for all individuals in population."""
        if self.n_jobs and self.n_jobs > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self._evaluate_individual, ind, constraints): ind
                    for ind in population
                }

                for future in as_completed(futures):
                    ind = futures[future]
                    ind.fitness = future.result()
                    self.total_evaluations += 1
        else:
            # Sequential evaluation
            for ind in population:
                ind.fitness = self._evaluate_individual(ind, constraints)
                self.total_evaluations += 1

    def _evaluate_individual(
        self,
        individual: Individual,
        constraints: Optional[List[Callable]]
    ) -> float:
        """Evaluate fitness of a single individual."""
        # Check constraints
        if constraints:
            for constraint in constraints:
                if not constraint(individual.genes):
                    return -np.inf

        # Evaluate fitness
        try:
            fitness = self.objective_function(individual.genes)
            return fitness if not np.isnan(fitness) else -np.inf
        except Exception as e:
            logger.debug(f"Error evaluating individual: {e}")
            return -np.inf

    def _selection(self, population: List[Individual]) -> List[Individual]:
        """Select individuals for reproduction."""
        method = self.config.selection_method

        if method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population)
        elif method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection(population)
        elif method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection(population)
        elif method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            return self._stochastic_universal_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _tournament_selection(self, population: List[Individual]) -> List[Individual]:
        """Tournament selection."""
        selected = []
        n_select = self.config.population_size

        for _ in range(n_select):
            # Randomly select tournament participants
            tournament = np.random.choice(
                population,
                size=self.config.tournament_size,
                replace=False
            )
            # Select best from tournament
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected

    def _roulette_wheel_selection(self, population: List[Individual]) -> List[Individual]:
        """Roulette wheel (fitness proportionate) selection."""
        # Shift fitness to be positive
        fitnesses = np.array([ind.fitness for ind in population])
        min_fitness = fitnesses.min()
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1

        # Calculate selection probabilities
        total_fitness = fitnesses.sum()
        if total_fitness == 0:
            probabilities = np.ones(len(population)) / len(population)
        else:
            probabilities = fitnesses / total_fitness

        # Select individuals
        selected_indices = np.random.choice(
            len(population),
            size=self.config.population_size,
            p=probabilities
        )

        return [population[i] for i in selected_indices]

    def _rank_based_selection(self, population: List[Individual]) -> List[Individual]:
        """Rank-based selection."""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)

        # Assign selection probabilities based on rank
        ranks = np.arange(1, len(population) + 1)
        probabilities = ranks / ranks.sum()

        # Select individuals
        selected_indices = np.random.choice(
            len(sorted_pop),
            size=self.config.population_size,
            p=probabilities
        )

        return [sorted_pop[i] for i in selected_indices]

    def _stochastic_universal_selection(self, population: List[Individual]) -> List[Individual]:
        """Stochastic universal sampling."""
        # Calculate cumulative fitness
        fitnesses = np.array([ind.fitness for ind in population])
        min_fitness = fitnesses.min()
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1

        cumulative_fitness = np.cumsum(fitnesses)
        total_fitness = cumulative_fitness[-1]

        if total_fitness == 0:
            return list(np.random.choice(population, size=self.config.population_size))

        # Calculate pointers
        pointer_distance = total_fitness / self.config.population_size
        start = np.random.uniform(0, pointer_distance)
        pointers = [start + i * pointer_distance for i in range(self.config.population_size)]

        # Select individuals
        selected = []
        for pointer in pointers:
            for i, cum_fit in enumerate(cumulative_fitness):
                if pointer <= cum_fit:
                    selected.append(population[i])
                    break

        return selected

    def _crossover(self, population: List[Individual]) -> List[Individual]:
        """Apply crossover to population."""
        offspring = []
        method = self.config.crossover_method

        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]

            if np.random.random() < self.config.crossover_rate:
                if method == CrossoverMethod.SINGLE_POINT:
                    child1, child2 = self._single_point_crossover(parent1, parent2)
                elif method == CrossoverMethod.TWO_POINT:
                    child1, child2 = self._two_point_crossover(parent1, parent2)
                elif method == CrossoverMethod.UNIFORM:
                    child1, child2 = self._uniform_crossover(parent1, parent2)
                elif method == CrossoverMethod.ARITHMETIC:
                    child1, child2 = self._arithmetic_crossover(parent1, parent2)
                else:
                    raise ValueError(f"Unknown crossover method: {method}")

                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        return offspring

    def _single_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        param_names = list(parent1.genes.keys())
        crossover_point = np.random.randint(1, len(param_names))

        child1_genes = {}
        child2_genes = {}

        for i, param in enumerate(param_names):
            if i < crossover_point:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _two_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Two-point crossover."""
        param_names = list(parent1.genes.keys())
        points = sorted(np.random.choice(range(1, len(param_names)), size=2, replace=False))

        child1_genes = {}
        child2_genes = {}

        for i, param in enumerate(param_names):
            if points[0] <= i < points[1]:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]
            else:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _uniform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        child1_genes = {}
        child2_genes = {}

        for param in parent1.genes.keys():
            if np.random.random() < 0.5:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _arithmetic_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Arithmetic (blend) crossover."""
        alpha = np.random.random()

        child1_genes = {}
        child2_genes = {}

        for param in parent1.genes.keys():
            child1_genes[param] = alpha * parent1.genes[param] + (1 - alpha) * parent2.genes[param]
            child2_genes[param] = (1 - alpha) * parent1.genes[param] + alpha * parent2.genes[param]

            # Ensure bounds
            lower, upper = self.param_bounds[param]
            child1_genes[param] = np.clip(child1_genes[param], lower, upper)
            child2_genes[param] = np.clip(child2_genes[param], lower, upper)

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _mutation(self, population: List[Individual]) -> List[Individual]:
        """Apply mutation to population."""
        method = self.config.mutation_method

        for individual in population:
            if np.random.random() < self.config.mutation_rate:
                if method == MutationMethod.GAUSSIAN:
                    self._gaussian_mutation(individual)
                elif method == MutationMethod.UNIFORM:
                    self._uniform_mutation(individual)
                elif method == MutationMethod.ADAPTIVE:
                    self._adaptive_mutation(individual)
                elif method == MutationMethod.POLYNOMIAL:
                    self._polynomial_mutation(individual)
                else:
                    raise ValueError(f"Unknown mutation method: {method}")

        return population

    def _gaussian_mutation(self, individual: Individual):
        """Gaussian mutation."""
        for param, value in individual.genes.items():
            lower, upper = self.param_bounds[param]
            param_range = upper - lower

            # Add Gaussian noise
            noise = np.random.normal(0, 0.1 * param_range)
            individual.genes[param] = np.clip(value + noise, lower, upper)

    def _uniform_mutation(self, individual: Individual):
        """Uniform mutation."""
        # Select random parameter to mutate
        param = np.random.choice(list(individual.genes.keys()))
        lower, upper = self.param_bounds[param]
        individual.genes[param] = np.random.uniform(lower, upper)

    def _adaptive_mutation(self, individual: Individual):
        """Adaptive mutation based on individual's age and fitness."""
        # Mutation strength decreases with age
        mutation_strength = 0.2 / (1 + individual.age * 0.1)

        for param, value in individual.genes.items():
            lower, upper = self.param_bounds[param]
            param_range = upper - lower

            noise = np.random.normal(0, mutation_strength * param_range)
            individual.genes[param] = np.clip(value + noise, lower, upper)

    def _polynomial_mutation(self, individual: Individual):
        """Polynomial mutation."""
        eta = 20  # Distribution index

        for param, value in individual.genes.items():
            lower, upper = self.param_bounds[param]

            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

            mutated_value = value + delta * (upper - lower)
            individual.genes[param] = np.clip(mutated_value, lower, upper)

    def _calculate_diversity(self, population: List[Individual]) -> float:
        """
        Calculate population diversity.

        Uses average pairwise distance in parameter space.
        """
        if len(population) < 2:
            return 0.0

        # Convert population to array
        param_names = list(population[0].genes.keys())
        pop_array = np.array([
            [ind.genes[param] for param in param_names]
            for ind in population
        ])

        # Normalize by parameter ranges
        for i, param in enumerate(param_names):
            lower, upper = self.param_bounds[param]
            if upper > lower:
                pop_array[:, i] = (pop_array[:, i] - lower) / (upper - lower)

        # Calculate average pairwise distance
        distances = []
        for i in range(len(pop_array)):
            for j in range(i + 1, len(pop_array)):
                dist = np.linalg.norm(pop_array[i] - pop_array[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _inject_diversity(self, population: List[Individual]) -> List[Individual]:
        """Inject diversity by replacing worst individuals with new random ones."""
        n_replace = max(1, int(len(population) * 0.1))

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # Keep best individuals
        new_population = sorted_pop[:len(population) - n_replace]

        # Add random individuals
        for _ in range(n_replace):
            genes = {}
            for param_name, (lower, upper) in self.param_bounds.items():
                genes[param_name] = np.random.uniform(lower, upper)
            new_population.append(Individual(genes=genes))

        logger.debug(f"Injected diversity: replaced {n_replace} individuals")
        return new_population

    def plot_evolution(self, result: GeneticResult) -> None:
        """Plot evolution history."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        generations = [g.generation_id for g in result.generation_history]
        best_fitness = [g.best_individual.fitness for g in result.generation_history]
        avg_fitness = [g.avg_fitness for g in result.generation_history]
        diversity = [g.diversity for g in result.generation_history]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Fitness evolution
        axes[0].plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        axes[0].plot(generations, avg_fitness, 'g--', label='Average Fitness')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title('Fitness Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Diversity evolution
        axes[1].plot(generations, diversity, 'r-', linewidth=2)
        axes[1].axhline(
            y=self.config.diversity_threshold,
            color='k',
            linestyle='--',
            alpha=0.5,
            label='Diversity Threshold'
        )
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Diversity')
        axes[1].set_title('Population Diversity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PYGAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import get_population_size
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    PositiveFloat,
    PositiveInt,
    ProbabilityFloat,
    PyTree,
)


class ParentSelectionFunction(Protocol):
    """Protocol for user-defined parent selection functions.

    Args:
        fitness: Array of fitness values for all solutions in the population.
        num_parents: Number of parents to select.
        ga_instance: The PyGAD GA instance.

    Returns:
        Tuple of (selected_parents, parent_indices) where:
        - selected_parents: 2D array of selected parent solutions
        - parent_indices: 1D array of indices of selected parents

    """

    def __call__(
        self, fitness: NDArray[np.float64], num_parents: int, ga_instance: Any
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]: ...


class CrossoverFunction(Protocol):
    """Protocol for user-defined crossover functions.

    Args:
        parents: 2D array of parent solutions selected for mating.
        offspring_size: Tuple (num_offspring, num_genes) specifying
            the shape of the offspring population to be generated.
        ga_instance: The PyGAD GA instance.

    Returns:
        2D array of offspring solutions generated from the parents.

    """

    def __call__(
        self,
        parents: NDArray[np.float64],
        offspring_size: tuple[int, int],
        ga_instance: Any,
    ) -> NDArray[np.float64]: ...


class MutationFunction(Protocol):
    """Protocol for user-defined mutation functions.

    Args:
        offspring: 2D array of offspring solutions to be mutated.
        ga_instance: The PyGAD GA instance.

    Returns:
        2D array of mutated offspring solutions.

    """

    def __call__(
        self, offspring: NDArray[np.float64], ga_instance: Any
    ) -> NDArray[np.float64]: ...


class GeneConstraintFunction(Protocol):
    """Protocol for user-defined gene constraint functions.

    Gene constraint functions are applied to individual genes to enforce specific
    constraints on their values. Each function receives the current solution and
    a list of candidate values, then returns the constrained values.

    Args:
        solution: Current solution array containing all gene values.
        values: List or array of candidate values for the gene being constrained.

    Returns:
        Constrained values as a list or array, ensuring they satisfy the gene's
        specific constraints.

    """

    def __call__(
        self, solution: NDArray[np.float64], values: list[float] | NDArray[np.float64]
    ) -> list[float] | NDArray[np.float64]: ...


@mark.minimizer(
    name="pygad",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class Pygad(Algorithm):
    """Minimize a scalar function using the PyGAD genetic algorithm.

    This optimizer wraps the PyGAD genetic algorithm, a population-based evolutionary
    method for global optimization. It maintains a population of candidate solutions and
    evolves them over generations using biologically inspired operations: selection
    (choosing parents based on fitness), crossover (combining genes from parents), and
    mutation (introducing random variations).

    The algorithm is well-suited for global optimization problems with multiple local
    optima, black-box optimization where gradients are unavailable or difficult to
    compute.

    All variables must have finite bounds. Parallel fitness evaluation is supported via
    batch processing.

    For more details, see the
    `PyGAD documentation <https://pygad.readthedocs.io/en/latest/>`_.

    """

    population_size: PositiveInt | None = None
    """Number of solutions in each generation.

    Larger populations explore the search space more thoroughly but require more
    fitness evaluations per generation. If None, defaults to
    ``max(10, 10 * (problem_dimension + 1))``.

    """

    num_parents_mating: PositiveInt | None = 10
    """Number of parents selected for mating in each generation.

    Higher values can speed up convergence but may risk premature convergence. If None,
    defaults to half the population size.

    """

    num_generations: PositiveInt | None = 50
    """Number of generations to evolve the population."""

    initial_population: list[PyTree] | None = None
    """Optional initial population as a list of parameter PyTrees.

    If None, the population is initialized randomly within parameter bounds.

    """

    parent_selection_type: (
        Literal["sss", "rws", "sus", "rank", "random", "tournament"]
        | ParentSelectionFunction
    ) = "sss"
    """Parent selection strategy used to choose parents for crossover.

    Available methods:
    - "sss": Steady-State Selection (selects the best individuals to continue)
    - "rws": Roulette Wheel Selection (probabilistic, fitness-proportional)
    - "sus": Stochastic Universal Sampling (even sampling across the population)
    - "rank": Rank Selection (selects based on rank order)
    - "random": Random Selection
    - "tournament": Tournament Selection (best from K randomly chosen individuals)

    Alternatively, provide a custom function with signature
    ``(fitness, num_parents, ga_instance) -> tuple[NDArray, NDArray]``.

    """

    keep_parents: int = -1
    """Number of best parents to keep in the next generation.

    Only used if ``keep_elitism = 0``. Values:
    - -1: Keep all parents in the next generation (default)
    - 0: Keep no parents in the next generation
    - Positive integer: Keep the specified number of best parents

    """

    keep_elitism: PositiveInt = 1
    """Number of elite (best) solutions preserved each generation.

    Range: 0 to population_size.If nonzero, takes precedence over ``keep_parents``.

    """

    K_tournament: PositiveInt = 3
    """Tournament size for parent selection when
    ``parent_selection_type="tournament"``."""

    crossover_type: (
        Literal["single_point", "two_points", "uniform", "scattered"]
        | CrossoverFunction
        | None
    ) = "single_point"
    """Crossover operator for generating offspring.

    Available methods:
    - "single_point": Single-point crossover
    - "two_points": Two-point crossover
    - "uniform": Uniform crossover (randomly mixes genes)
    - "scattered": Scattered crossover (random mask)

    Or provide a custom function with signature
    ``(parents, offspring_size, ga_instance) -> NDArray``. Set to None to disable
    crossover.

    """

    crossover_probability: ProbabilityFloat | None = None
    """Probability of applying crossover to selected parents.

    Range [0, 1]. If None, uses PyGAD's default.

    """

    mutation_type: (
        Literal["random", "swap", "inversion", "scramble", "adaptive"]
        | MutationFunction
        | None
    ) = "random"
    """Mutation operator for introducing genetic diversity.

    Available methods:
    - "random": Replace with random values
    - "swap": Exchange two genes
    - "inversion": Reverse a sequence of genes
    - "scramble": Shuffle a subset of genes
    - "adaptive": Adaptively adjusts mutation rate

    Or provide a custom function with signature
    ``(offspring, ga_instance) -> NDArray``. Set to None to disable mutation.

    """
    mutation_probability: (
        ProbabilityFloat
        | list[ProbabilityFloat]
        | tuple[ProbabilityFloat, ProbabilityFloat]
        | NDArray[np.float64]
        | None
    ) = None
    """Probability of mutating each gene.

    - Scalar: Fixed probability for all genes (non-adaptive)
    - List/tuple/array of 2 values: Adaptive mutation;
      [prob_low_fitness, prob_high_fitness] (only with ``mutation_type="adaptive"``)

    When specified, takes precedence over ``mutation_percent_genes`` and
    ``mutation_num_genes``. Range [0, 1].

    """

    mutation_percent_genes: (
        PositiveFloat
        | str
        | list[PositiveFloat]
        | tuple[PositiveFloat, PositiveFloat]
        | NDArray[np.float64]
    ) = "default"
    """Percentage of genes to mutate in each solution.

    - "default": Uses 10% of genes (PyGAD default)
    - Scalar: Fixed percentage for all generations (0-100)
    - List/tuple/array of 2 values: Adaptive mutation;
      [percent_low_fitness, percent_high_fitness] (only with
      ``mutation_type="adaptive"``)

    Ignored if ``mutation_probability`` is specified.

    """

    mutation_num_genes: (
        PositiveInt
        | list[PositiveInt]
        | tuple[PositiveInt, PositiveInt]
        | NDArray[np.int_]
        | None
    ) = None
    """Number of genes to mutate per solution.

    - Scalar: Fixed number for all generations
    - List/tuple/array of 2 values: Adaptive;
      [count_low_fitness, count_high_fitness] (only with ``mutation_type="adaptive"``)

    Takes precedence over ``mutation_percent_genes`` but is ignored if
    ``mutation_probability`` is specified.

    """
    mutation_by_replacement: bool = False
    """If True, mutated gene values are replaced with random values.

    Only for ``mutation_type="random"``; if False, random values are added to the
    original.

    """

    random_mutation_min_val: float | list[float] | NDArray[np.float64] = -1.0

    random_mutation_max_val: float | list[float] | NDArray[np.float64] = 1.0
    """Minimum and maximum values used for random mutation.

    Can be scalars, arrays/lists (one per gene), or PyTrees matching the parameter
    structure. Only used with ``mutation_type="random"``.

    """

    allow_duplicate_genes: bool = True
    """If True, duplicate gene values are allowed within a solution."""

    gene_constraint: list[GeneConstraintFunction | None] | None = None
    """Optional list of per-gene constraint functions.

    Each with signature ``(solution, values) -> list[float] | NDArray``.

    """

    sample_size: PositiveInt = 100
    """Number of values to sample when enforcing uniqueness or gene constraints."""

    batch_size: PositiveInt | None = None
    """Number of solutions to evaluate in parallel batches.

    If None and ``n_cores > 1``, automatically set to ``n_cores``.

    """
    stop_criteria: str | list[str] | None = None
    """Stopping criteria for the genetic algorithm.

    Can be a string or list of strings.

    Supported criteria:
    - "reach_{value}": Stop when fitness reaches the specified value,
      e.g. "reach_0.01"
    - "saturate_{generations}": Stop if fitness doesn't improve for the given
      number of generations, e.g. "saturate_10"

    Can specify multiple criteria as a list.

    """

    n_cores: PositiveInt = 1
    """Number of CPU cores for parallel fitness evaluation."""

    seed: int | None = None
    """Random seed for reproducibility."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYGAD_INSTALLED:
            raise NotInstalledError(
                "The 'pygad' algorithm requires the pygad package to be "
                "installed. You can install it with 'pip install pygad'."
            )

        import pygad

        if (
            problem.bounds.lower is None
            or problem.bounds.upper is None
            or not np.isfinite(problem.bounds.lower).all()
            or not np.isfinite(problem.bounds.upper).all()
        ):
            raise ValueError("pygad requires finite bounds for all parameters.")

        # Determine effective batch_size for parallel processing
        effective_batch_size = determine_effective_batch_size(
            self.batch_size, self.n_cores
        )

        if (
            effective_batch_size is not None
            and effective_batch_size > 1
            and self.n_cores > 1
        ):

            def _fitness_func_batch(
                _ga_instance: Any,
                batch_solutions: NDArray[np.float64],
                _batch_indices: list[int] | NDArray[np.int_],
            ) -> list[float]:
                solutions_list: list[NDArray[np.float64]] = [
                    np.asarray(batch_solutions[i])
                    for i in range(batch_solutions.shape[0])
                ]
                batch_results = problem.batch_fun(
                    solutions_list,
                    n_cores=self.n_cores,
                    batch_size=effective_batch_size,
                )

                return [-float(result) for result in batch_results]

            fitness_function: Any = _fitness_func_batch
        else:

            def _fitness_func_single(
                _ga_instance: Any, solution: NDArray[np.float64], _solution_idx: int
            ) -> float:
                return -float(problem.fun(solution))

            fitness_function = _fitness_func_single

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        num_parents_mating = (
            self.num_parents_mating
            if self.num_parents_mating is not None
            else max(2, population_size // 2)
        )

        if self.initial_population is not None:
            initial_population = np.array(
                [
                    problem.converter.params_to_internal(params)
                    for params in self.initial_population
                ]
            )
        else:
            num_genes = len(x0)

            initial_population = np.random.uniform(
                problem.bounds.lower,
                problem.bounds.upper,
                size=(population_size, num_genes),
            )

            initial_population[0] = x0

        gene_space = [
            {"low": problem.bounds.lower[i], "high": problem.bounds.upper[i]}
            for i in range(len(x0))
        ]

        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_function,
            fitness_batch_size=effective_batch_size,
            initial_population=initial_population,
            gene_space=gene_space,
            parent_selection_type=self.parent_selection_type,
            keep_parents=self.keep_parents,
            keep_elitism=self.keep_elitism,
            K_tournament=self.K_tournament,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_probability,
            mutation_by_replacement=self.mutation_by_replacement,
            mutation_percent_genes=self.mutation_percent_genes,
            mutation_num_genes=self.mutation_num_genes,
            random_mutation_min_val=self.random_mutation_min_val,
            random_mutation_max_val=self.random_mutation_max_val,
            allow_duplicate_genes=self.allow_duplicate_genes,
            gene_constraint=self.gene_constraint,
            sample_size=self.sample_size,
            stop_criteria=self.stop_criteria,
            parallel_processing=None,
            random_seed=self.seed,
        )

        ga_instance.run()

        result = _process_pygad_result(ga_instance)

        return result


def determine_effective_batch_size(batch_size: int | None, n_cores: int) -> int | None:
    """Determine the effective batch_size for parallel processing.

    Behavior:
    - If `batch_size` is explicitly provided:
        - The value is returned unchanged.
        - A warning is issued if it is less than `n_cores`, as this may
        underutilize available cores.
    - If `batch_size` is `None`:
        - If `n_cores` > 1, defaults to `n_cores`.
        - Otherwise, returns None (i.e., single-threaded evaluation).
    Args:
        batch_size: User-specified batch size or None
        n_cores: Number of cores for parallel processing

    Returns:
        Effective batch size for PyGAD, or None for single-threaded processing

    """
    result = None

    if batch_size is not None:
        if batch_size < n_cores:
            warnings.warn(
                f"batch_size ({batch_size}) is smaller than "
                f"n_cores ({n_cores}). This may reduce parallel efficiency. "
                f"Consider setting batch_size >= n_cores."
            )
        result = batch_size
    elif n_cores > 1:
        result = n_cores

    return result


def _process_pygad_result(ga_instance: Any) -> InternalOptimizeResult:
    """Process PyGAD result into InternalOptimizeResult.

    Args:
        ga_instance: The PyGAD instance after running the optimization

    Returns:
        InternalOptimizeResult: Processed optimization results

    """
    best_solution, best_fitness, _ = ga_instance.best_solution()

    best_criterion = -best_fitness

    completed_generations = ga_instance.generations_completed
    success = ga_instance.run_completed
    if success:
        message = (
            "Optimization terminated successfully.\n"
            f"Generations completed: {completed_generations}"
        )
    else:
        message = (
            "Optimization failed to complete.\n"
            f"Generations completed: {completed_generations}"
        )

    return InternalOptimizeResult(
        x=best_solution,
        fun=best_criterion,
        success=success,
        message=message,
        n_fun_evals=ga_instance.generations_completed * ga_instance.pop_size[0],
    )

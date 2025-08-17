"""Implement PyGAD genetic algorithm optimizer."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

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
        offspring_size: Tuple (num_offspring, num_genes) specifying the shape
            of the offspring population to be generated.
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


@runtime_checkable
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

    Gene constraint functions are applied to individual genes to enforce
    specific constraints on their values. Each function receives the current
    solution and a list of candidate values, then returns the constrained
    values.

    Args:
        solution: Current solution array containing all gene values.
        values: List or array of candidate values for the gene being
            constrained.

    Returns:
        Constrained values as a list or array, ensuring they satisfy the
        gene's specific constraints.

    """

    def __call__(
        self,
        solution: NDArray[np.float64],
        values: list[float] | NDArray[np.float64],
    ) -> list[float] | NDArray[np.float64]: ...


@dataclass(frozen=True)
class _BuiltinMutation:
    """Base class for all built-in PyGAD mutation configurations.

    Note:
        This is an internal base class. Users should not inherit from it
        directly. To configure a built-in mutation, use one of its subclasses
        (e.g., `RandomMutation`, `AdaptiveMutation`). To define a custom
        mutation, provide a function that conforms to the `MutationFunction`
        protocol.

    """

    mutation_type: ClassVar[str] = "random"

    def to_pygad_params(self) -> dict[str, Any]:
        """Convert mutation configuration to PyGAD parameters.

        Default implementation that works for simple mutations. Complex
        mutations (RandomMutation, AdaptiveMutation) should override this.

        Returns:
            Dictionary of PyGAD mutation parameters.

        """
        return {
            "mutation_type": self.mutation_type,
            "mutation_probability": None,
            "mutation_percent_genes": "default",
            "mutation_num_genes": None,
            "mutation_by_replacement": False,
        }


@dataclass(frozen=True)
class RandomMutation(_BuiltinMutation):
    """Configuration for the random mutation in PyGAD.

    The random mutation selects a subset of genes in each solution and either
    replaces each selected gene with a new random value or adds a random value
    to it.

    The exact behavior depends on the `by_replacement` parameter: If
    `by_replacement` is True, the selected genes are replaced with new values;
    if False, random values are added to the existing gene values.

    The mutation rate is determined by the mutation probability, the number of
    genes, or the percentage of genes (with priority: probability > num_genes
    > percent_genes).

    """

    mutation_type: ClassVar[str] = "random"

    probability: ProbabilityFloat | None = None
    """Probability of mutating each gene.

    If specified, takes precedence over num_genes and percent_genes. Range [0, 1].

    """

    num_genes: PositiveInt | None = None
    """Number of genes to mutate per solution.

    Takes precedence over percent_genes but is ignored if probability is specified.

    """

    percent_genes: PositiveFloat | str = "default"
    """Percentage of genes to mutate in each solution.

    - "default": Uses 10% of genes (PyGAD default)
    - Numeric value: Percentage (0-100)

    Ignored if probability or num_genes are specified.

    """

    by_replacement: bool = False
    """If True, replace gene values with random values.

    If False, add random values to existing gene values.

    """

    def to_pygad_params(self) -> dict[str, Any]:
        """Convert RandomMutation configuration to PyGAD parameters."""
        return {
            "mutation_type": self.mutation_type,
            "mutation_probability": self.probability,
            "mutation_percent_genes": self.percent_genes,
            "mutation_num_genes": self.num_genes,
            "mutation_by_replacement": self.by_replacement,
        }


@dataclass(frozen=True)
class SwapMutation(_BuiltinMutation):
    """Configuration for the swap mutation in PyGAD.

    The swap mutation selects two random genes and exchanges their values. This
    operation maintains all gene values, altering only their positions within the
    chromosome.

    No additional parameters are required for this mutation type.

    """

    mutation_type: ClassVar[str] = "swap"


@dataclass(frozen=True)
class InversionMutation(_BuiltinMutation):
    """Configuration for the inversion mutation in PyGAD.

    The inversion mutation selects a contiguous segment of genes and reverses their
    order. All gene values remain unchanged; only the ordering within the selected
    segment is altered.

    No additional parameters are required for this mutation type.

    """

    mutation_type: ClassVar[str] = "inversion"


@dataclass(frozen=True)
class ScrambleMutation(_BuiltinMutation):
    """Configuration for the scramble mutation in PyGAD.

    The scramble mutation randomly shuffles the genes within a contiguous segment. This
    preserves gene values but changes their order within the chosen segment.

    No additional parameters are required for this mutation type.

    """

    mutation_type: ClassVar[str] = "scramble"


@dataclass(frozen=True)
class AdaptiveMutation(_BuiltinMutation):
    """Configuration for the adaptive mutation in PyGAD.

    The adaptive mutation dynamically adjusts the mutation rate based on
    solution quality. Solutions whose objective value is worse than the
    current population median receive a higher mutation rate to encourage
    exploration, while better-than-median solutions receive a lower rate
    to preserve promising traits.

    If no mutation rate parameters are specified, this mutation defaults to using
    probabilities, with a 10% rate for bad solutions (`probability_bad=0.1`)
    and a 5% rate for good solutions (`probability_good=0.05`).

    **Parameter Precedence:**
    The mutation rate is determined by the first set of parameters found, in the
    following order of priority:
    1. `probability_bad` and `probability_good`
    2. `num_genes_bad` and `num_genes_good`
    3. `percent_genes_bad` and `percent_genes_good`

    """

    mutation_type: ClassVar[str] = "adaptive"

    probability_bad: ProbabilityFloat | None = None
    """Probability of mutating each gene for below-average fitness solutions.

    If specified, takes precedence over num_genes_bad and percent_genes_bad. Range [0,
    1]. If no mutation rate parameters are provided at all, this defaults to
    0.1 (10% mutation rate for bad fitness solutions).

    """

    probability_good: ProbabilityFloat | None = None
    """Probability of mutating each gene for above-average fitness solutions.

    If specified, takes precedence over num_genes_good and percent_genes_good. Range [0,
    1]. If no mutation rate parameters are provided at all, this defaults to
    0.05 (5% mutation rate for good fitness solutions).

    """

    num_genes_bad: PositiveInt | None = None
    """Number of genes to mutate for below-average fitness solutions.

    Takes precedence over percent_genes_bad but is ignored if probability_bad is
    specified.

    """

    num_genes_good: PositiveInt | None = None
    """Number of genes to mutate for above-average fitness solutions.

    Takes precedence over percent_genes_good but is ignored if probability_good is
    specified.

    """

    percent_genes_bad: PositiveFloat | None = None
    """Percentage of genes to mutate for below-average fitness solutions.

    Ignored if probability_bad or num_genes_bad are specified.

    """

    percent_genes_good: PositiveFloat | None = None
    """Percentage of genes to mutate for above-average fitness solutions.

    Ignored if probability_good or num_genes_good are specified.

    """

    by_replacement: bool = False
    """If True, replace gene values with random values.

    If False, add random values to existing gene values.

    """

    def to_pygad_params(self) -> dict[str, Any]:
        """Convert AdaptiveMutation configuration to PyGAD parameters."""
        mutation_probability: list[float] | None = None
        mutation_num_genes: list[int] | None = None
        mutation_percent_genes: list[float] | str | None = None

        if self.probability_bad is not None and self.probability_good is not None:
            mutation_probability = [self.probability_bad, self.probability_good]
        elif self.num_genes_bad is not None and self.num_genes_good is not None:
            mutation_num_genes = [self.num_genes_bad, self.num_genes_good]
        elif self.percent_genes_bad is not None and self.percent_genes_good is not None:
            mutation_percent_genes = [self.percent_genes_bad, self.percent_genes_good]
        else:
            mutation_probability = [
                self.probability_bad or 0.1,
                self.probability_good or 0.05,
            ]

        return {
            "mutation_type": self.mutation_type,
            "mutation_probability": mutation_probability,
            "mutation_percent_genes": mutation_percent_genes,
            "mutation_num_genes": mutation_num_genes,
            "mutation_by_replacement": self.by_replacement,
        }


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

    Larger populations explore the search space more thoroughly but require
    more fitness evaluations per generation. If None, optimagic sets this to
    ``max(10, 10 * (problem_dimension + 1))``.

    """

    num_parents_mating: PositiveInt | None = 10
    """Number of parents selected for mating in each generation.

    Higher values can speed up convergence but may risk premature convergence.
    If None, defaults to ``max(2, population_size // 2)``.

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

    * ``"sss"``: Steady-State Selection (selects the best individuals to continue)
    * ``"rws"``: Roulette Wheel Selection (probabilistic, fitness-proportional)
    * ``"sus"``: Stochastic Universal Sampling (even sampling across population)
    * ``"rank"``: Rank Selection (selects based on rank order)
    * ``"random"``: Random Selection
    * ``"tournament"``: Tournament Selection (best from K randomly chosen individuals)

    Alternatively, provide a custom function with signature
    ``(fitness, num_parents, ga_instance) -> tuple[NDArray, NDArray]``.

    """

    keep_parents: int = -1
    """Number of best parents to keep in the next generation.

    Only used if ``keep_elitism = 0``. Values:

    * ``-1``: Keep all parents in the next generation (default)
    * ``0``: Keep no parents in the next generation
    * Positive integer: Keep the specified number of best parents

    """

    keep_elitism: int = 1
    """Number of elite (best) solutions preserved each generation.

    Range: 0 to population_size. If greater than 0, takes precedence over
    ``keep_parents``. When 0, elitism is disabled and ``keep_parents``
    controls parent retention.

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

    * ``"single_point"``: Single-point crossover
    * ``"two_points"``: Two-point crossover
    * ``"uniform"``: Uniform crossover (randomly mixes genes)
    * ``"scattered"``: Scattered crossover (random mask)

    Or provide a custom function with signature
    ``(parents, offspring_size, ga_instance) -> NDArray``.

    """

    crossover_probability: ProbabilityFloat | None = None
    """Probability of applying crossover to selected parents.

    Range [0, 1]. If None, uses PyGAD's default.

    """

    mutation: (
        Literal["random", "swap", "inversion", "scramble", "adaptive"]
        | type[_BuiltinMutation]
        | _BuiltinMutation
        | MutationFunction
        | None
    ) = "random"
    """Mutation operator for introducing genetic diversity.

    Available options:

    **String values for default configurations:**

    * ``"random"``: Random mutation with default parameters
    * ``"swap"``: Swap mutation with default parameters
    * ``"inversion"``: Inversion mutation with default parameters
    * ``"scramble"``: Scramble mutation with default parameters
    * ``"adaptive"``: Adaptive random mutation with default parameters

    **Mutation classes for default configurations:**

    * Any mutation class (e.g., ``RandomMutation``, ``SwapMutation``,
      ``AdaptiveMutation``, etc.)
    * All classes can be used without parameters for default behavior

    **Configured mutation instances:**

    * Any mutation instance (e.g., ``RandomMutation(...)``,
      ``SwapMutation()``, etc.)
    * All mutation classes inherit from ``_BuiltinMutation``

    **Custom function:**

    * Custom function with signature ``(offspring, ga_instance) -> NDArray``

    **Disable mutation:**

    * ``None`` to disable mutation

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

    * ``"reach_{value}"``: Stop when the objective value reaches the specified
      threshold, e.g. ``"reach_0.01"``
    * ``"saturate_{generations}"``: Stop if the objective value has not improved
      for the given number of generations, e.g. ``"saturate_10"``
    Multiple criteria can be specified as a list.

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
        effective_batch_size = _determine_effective_batch_size(
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

        # Convert mutation parameter to PyGAD parameters
        mutation_params = _convert_mutation_to_pygad_params(self.mutation)

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
            mutation_type=mutation_params["mutation_type"],
            mutation_probability=mutation_params["mutation_probability"],
            mutation_by_replacement=mutation_params["mutation_by_replacement"],
            mutation_percent_genes=mutation_params["mutation_percent_genes"],
            mutation_num_genes=mutation_params["mutation_num_genes"],
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


def _convert_mutation_to_pygad_params(mutation: Any) -> dict[str, Any]:
    """Convert the mutation parameter to PyGAD mutation parameters.

    Handles strings, classes, instances, and custom functions using the
    new mutation dataclass system with built-in conversion methods.

    Returns:
        Dictionary of PyGAD mutation parameters.

    """
    params: dict[str, Any]

    if mutation is None:
        params = _get_default_mutation_params(mutation_type=None)

    elif isinstance(mutation, str):
        mutation_instance = _create_mutation_from_string(mutation)
        params = mutation_instance.to_pygad_params()

    elif isinstance(mutation, type) and issubclass(mutation, _BuiltinMutation):
        mutation_instance = mutation()
        params = mutation_instance.to_pygad_params()

    elif isinstance(mutation, _BuiltinMutation):
        params = mutation.to_pygad_params()

    elif isinstance(mutation, MutationFunction):
        params = _get_default_mutation_params(mutation_type=mutation)

    else:
        raise ValueError(f"Unsupported mutation type: {type(mutation)}")

    return params


def _get_default_mutation_params(mutation_type: Any = "random") -> dict[str, Any]:
    """Get default PyGAD mutation parameters."""
    return {
        "mutation_type": mutation_type,
        "mutation_probability": None,
        "mutation_percent_genes": None if mutation_type is None else "default",
        "mutation_num_genes": None,
        "mutation_by_replacement": None if mutation_type is None else False,
    }


def _create_mutation_from_string(mutation_type: str) -> _BuiltinMutation:
    """Create a mutation instance from a string type.

    Args:
        mutation_type: String mutation type (e.g., "random", "swap", etc.)

    Returns:
        Appropriate mutation instance.

    Raises:
        ValueError: If mutation_type is not supported.

    """
    mutation_map = {
        "random": RandomMutation,
        "swap": SwapMutation,
        "inversion": InversionMutation,
        "scramble": ScrambleMutation,
        "adaptive": AdaptiveMutation,
    }

    if mutation_type not in mutation_map:
        raise ValueError(f"Unsupported mutation type: {mutation_type}")

    return mutation_map[mutation_type]()


def _determine_effective_batch_size(batch_size: int | None, n_cores: int) -> int | None:
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
        Effective batch size for PyGAD, or None for single-threaded
        processing

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

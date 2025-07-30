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
    NonNegativeFloat,
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
    """Protocol for user-defined gene constraint functions."""

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
    population_size: PositiveInt | None = None
    num_parents_mating: PositiveInt | None = 10
    num_generations: PositiveInt | None = 50

    initial_population: list[PyTree] | None = None

    parent_selection_type: (
        Literal["sss", "rws", "sus", "rank", "random", "tournament"]
        | ParentSelectionFunction
    ) = "sss"
    keep_parents: int = -1
    keep_elitism: PositiveInt = 1
    K_tournament: PositiveInt = 3

    crossover_type: (
        Literal["single_point", "two_points", "uniform", "scattered"]
        | CrossoverFunction
        | None
    ) = "single_point"
    crossover_probability: ProbabilityFloat | None = None

    mutation_type: (
        Literal["random", "swap", "inversion", "scramble", "adaptive"]
        | MutationFunction
        | None
    ) = "random"
    mutation_probability: (
        ProbabilityFloat
        | list[ProbabilityFloat]
        | tuple[ProbabilityFloat, ProbabilityFloat]
        | NDArray[np.float64]
        | None
    ) = None

    mutation_percent_genes: (
        PositiveFloat
        | str
        | list[PositiveFloat]
        | tuple[PositiveFloat, PositiveFloat]
        | NDArray[np.float64]
    ) = "default"

    mutation_num_genes: (
        PositiveInt
        | list[PositiveInt]
        | tuple[PositiveInt, PositiveInt]
        | NDArray[np.int_]
        | None
    ) = None

    mutation_by_replacement: bool = False
    random_mutation_min_val: float | list[float] | NDArray[np.float64] = -1.0
    random_mutation_max_val: float | list[float] | NDArray[np.float64] = 1.0

    allow_duplicate_genes: bool = True

    gene_constraint: list[GeneConstraintFunction | None] | None = None
    sample_size: PositiveInt = 100

    batch_size: PositiveInt | None = None
    stop_criteria: str | list[str] | None = None

    n_cores: PositiveInt = 1
    seed: int | None = None

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

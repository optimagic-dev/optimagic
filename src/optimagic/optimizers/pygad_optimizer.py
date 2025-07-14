import warnings
from dataclasses import dataclass
from typing import Any, Literal

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
)

if IS_PYGAD_INSTALLED:
    import pygad


@mark.minimizer(
    name="pygad",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class Pygad(Algorithm):
    population_size: PositiveInt | None = None
    num_parents_mating: PositiveInt | None = None
    num_generations: PositiveInt | None = None

    initial_population: NDArray[np.float64] | list[list[float]] | None = None

    parent_selection_type: Literal[
        "sss", "rws", "sus", "rank", "random", "tournament"
    ] = "sss"
    keep_parents: int = -1
    keep_elitism: PositiveInt = 1
    K_tournament: PositiveInt = 3

    crossover_type: (
        Literal["single_point", "two_points", "uniform", "scattered"] | None
    ) = "single_point"
    crossover_probability: NonNegativeFloat | None = None

    mutation_type: (
        Literal["random", "swap", "inversion", "scramble", "adaptive"] | None
    ) = "random"
    mutation_probability: (
        NonNegativeFloat
        | list[NonNegativeFloat]
        | tuple[NonNegativeFloat, NonNegativeFloat]
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

    fitness_batch_size: PositiveInt | None = None
    stop_criteria: str | list[str] | None = None

    n_cores: PositiveInt = 1
    random_seed: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYGAD_INSTALLED:
            raise NotInstalledError(
                "The 'pygad_pygad' algorithm requires the pygad package to be "
                "installed. You can install it with 'pip install pygad'."
            )

        if (
            problem.bounds.lower is None
            or problem.bounds.upper is None
            or not np.isfinite(problem.bounds.lower).all()
            or not np.isfinite(problem.bounds.upper).all()
        ):
            raise ValueError("pygad_pygad requires finite bounds for all parameters.")

        # Determine effective fitness_batch_size for parallel processing
        effective_fitness_batch_size = determine_effective_batch_size(
            self.fitness_batch_size, self.n_cores
        )

        if (
            effective_fitness_batch_size is not None
            and effective_fitness_batch_size > 1
            and self.n_cores > 1
        ):

            def _fitness_func_batch(
                _ga_instance: Any,
                batch_solutions: NDArray[np.float64],
                _batch_indices: list[int] | NDArray[np.int_],
            ) -> list[float]:
                batch_results = problem.batch_fun(
                    batch_solutions.tolist(), n_cores=self.n_cores
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

        if self.initial_population is not None:
            initial_population = np.array(self.initial_population)
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
            num_parents_mating=self.num_parents_mating,
            fitness_func=fitness_function,
            fitness_batch_size=effective_fitness_batch_size,
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
            stop_criteria=self.stop_criteria,
            parallel_processing=None,
            random_seed=self.random_seed,
        )

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        res = InternalOptimizeResult(
            x=solution,
            fun=-solution_fitness,
            success=True,
            message=(
                f"Optimization terminated successfully after "
                f"{ga_instance.generations_completed} generations."
            ),
            n_fun_evals=ga_instance.generations_completed * population_size,
        )

        return res


def determine_effective_batch_size(
    fitness_batch_size: int | None, n_cores: int
) -> int | None:
    if fitness_batch_size is not None:
        if fitness_batch_size < n_cores:
            warnings.warn(
                f"fitness_batch_size ({fitness_batch_size}) is smaller than "
                f"n_cores ({n_cores}). This may reduce parallel efficiency. "
                f"Consider setting fitness_batch_size >= n_cores."
            )
        return fitness_batch_size
    elif n_cores > 1:
        return n_cores
    else:
        return None

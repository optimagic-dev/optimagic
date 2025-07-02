from dataclasses import dataclass
from typing import Any, Literal, Union

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
    num_parents_mating: PositiveInt = 10
    num_generations: PositiveInt = 100

    initial_population: NDArray[np.float64] | list[list[float]] | None = None
    gene_type: (
        type[int]
        | type[float]
        | type[np.int8]
        | type[np.int16]
        | type[np.int32]
        | type[np.int64]
        | type[np.uint]
        | type[np.uint8]
        | type[np.uint16]
        | type[np.uint32]
        | type[np.uint64]
        | type[np.float16]
        | type[np.float32]
        | type[np.float64]
        | list[type]
        | list[list[type | None]]
    ) = float

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
    save_best_solutions: bool = False
    save_solutions: bool = False
    stop_criteria: str | list[str] | None = None

    random_seed: int | None = None
    parallel_processing: (
        int
        | tuple[Literal["process", "thread"], int | None]
        | list[Union[Literal["process", "thread"], int | None]]
        | None
    ) = None
    suppress_warnings: bool = True

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

        def fitness_func(
            ga_instance: Any, solution: NDArray[np.float64], solution_idx: int
        ) -> float:
            return -float(problem.fun(solution))

        if self.initial_population is not None:
            initial_population = self.initial_population
            population_size = len(initial_population)
        else:
            population_size = get_population_size(
                population_size=self.population_size, x=x0
            )
            initial_population = np.random.uniform(
                low=problem.bounds.lower,
                high=problem.bounds.upper,
                size=(population_size, len(x0)),
            )
            initial_population[0] = x0

        gene_space = [
            {"low": problem.bounds.lower[i], "high": problem.bounds.upper[i]}
            for i in range(len(x0))
        ]

        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=population_size,
            num_genes=len(x0),
            initial_population=initial_population,
            init_range_low=problem.bounds.lower,
            init_range_high=problem.bounds.upper,
            gene_space=gene_space,
            gene_type=self.gene_type,
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
            save_best_solutions=self.save_best_solutions,
            save_solutions=self.save_solutions,
            suppress_warnings=self.suppress_warnings,
            stop_criteria=self.stop_criteria,
            parallel_processing=self.parallel_processing,
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

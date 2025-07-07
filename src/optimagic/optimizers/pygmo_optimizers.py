"""Implement pygmo optimizers.

Notes for converting to the new algorithm interface:

- `create_algo_options` is not needed anymore because the only thing it did was mixing
  options that are supported by all optimizers (e.g. population_size, seed) with
  specific options. Then later they had to be taken appart again. Instead you need
  to pass  a few more arguments directly to `_minimize_pygmo`.
- Calling `_check_that_every_param_is_bounded` is not needed anymore. I do that check
  once in `_minimize_pygmo`.
- The documentation often just says float where I suspect PositiveFloats; Leave it at
  float for now and add todos where needed. Don't spend a lot of time on it.
- There are some specific type checks and type conversions that should not be needed
  anymore after switching to the new interface.
- Whenever we had batch_evaluator as `algo_option` we don't need it anymore but we
  should have `n_cores` in those algorithms.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PYGMO_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_XTOL_REL,
    STOPPING_MAXFUN_GLOBAL,
    get_population_size,
)
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

STOPPING_MAX_ITERATIONS_GENETIC = 250

try:
    import pygmo as pg
except ImportError:
    pass


@mark.minimizer(
    name="pygmo_gaco",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
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
class PygmoGaco(Algorithm):
    population_size: int | None = None
    n_cores: int = 1
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    kernel_size: PositiveInt = 63
    speed_parameter_q: PositiveFloat = 1.0
    oracle: float = 0.0
    accuracy: PositiveFloat = 0.01
    threshold: PositiveInt = 1
    speed_of_std_values_convergence: int = 7
    stopping_max_n_without_improvements: PositiveInt = 100000
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    focus: NonNegativeFloat = 0.0
    cache: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "ker": self.kernel_size,
            "q": self.speed_parameter_q,
            "oracle": self.oracle,
            "acc": self.accuracy,
            "threshold": self.threshold,
            "n_gen_mark": self.speed_of_std_values_convergence,
            "impstop": self.stopping_max_n_without_improvements,
            "evalstop": self.stopping_maxfun,
            "focus": self.focus,
            "memory": self.cache,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="gaco",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=self.n_cores,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )

        return res


@mark.minimizer(
    name="pygmo_bee_colony",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoBeeColony(Algorithm):
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    seed: int | None = None
    discard_start_params: bool = False
    max_n_trials: PositiveInt = 1
    population_size: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        algo_specific_options = {
            "limit": self.max_n_trials,
            "gen": self.stopping_maxiter,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="bee_colony",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_de",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoDe(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    # TODO: Refine type to fix range [0,2]
    weight_coefficient: NonNegativeFloat = 0.8
    # TODO: Probably refine type to fix range [0,1]
    crossover_probability: NonNegativeFloat = 0.9
    mutation_variant: Literal[
        "best/1/exp",
        "rand/1/exp",
        "rand-to-best/1/exp",
        "best/2/exp",
        "rand/2/exp",
        "best/1/bin",
        "rand/1/bin",
        "rand-to-best/1/bin",
        "best/2/bin",
        "rand/2/bin",
    ] = "rand/1/exp"
    convergence_criterion_tolerance: NonNegativeFloat = 1e-6
    convergence_relative_params_tolerance: NonNegativeFloat = CONVERGENCE_XTOL_REL

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )
        # support both integer and string specification of the mutation variant
        mutation_variant_str_to_int = {
            "best/1/exp": 1,
            "rand/1/exp": 2,
            "rand-to-best/1/exp": 3,
            "best/2/exp": 4,
            "rand/2/exp": 5,
            "best/1/bin": 6,
            "rand/1/bin": 7,
            "rand-to-best/1/bin": 8,
            "best/2/bin": 9,
            "rand/2/bin": 10,
        }
        mutation_variant = _convert_str_to_int(
            str_to_int=mutation_variant_str_to_int, value=self.mutation_variant
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "F": self.weight_coefficient,
            "CR": self.crossover_probability,
            "variant": mutation_variant,
            "ftol": self.convergence_criterion_tolerance,
            "xtol": self.convergence_relative_params_tolerance,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="de",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_sea",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSea(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = (
        10_000  # Each generation will compute the objective once
    )

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="sea",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_sga",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSga(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    # TODO: Refine type to fix range [0,1]
    crossover_probability: NonNegativeFloat = 0.9
    crossover_strategy: Literal[
        "exponential",
        "sbx",
        "single",
        "binomial",
    ] = "exponential"
    # TODO: Refine type to fix range [1,100]
    eta_c: PositiveFloat | None = None
    # TODO: Refine type to fix range [0,1]
    mutation_probability: NonNegativeFloat = 0.02
    mutation_strategy: Literal["uniform", "polynomial"] = "polynomial"
    # TODO: Refine type to fix range [0,1]
    mutation_polynomial_distribution_index: NonNegativeFloat | None = None
    # TODO: Refine type to fix range [0,1]
    mutation_gaussian_width: NonNegativeFloat | None = None
    selection_strategy: Literal["tournament", "truncated"] = "tournament"
    # TODO: Check if should be NonNegativeInt
    selection_truncated_n_best: int | None = None
    # TODO Check if should be NonNegativeInt
    selection_tournament_size: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        if self.eta_c is not None and self.crossover_strategy != "sbx":
            warnings.warn(
                f"You specified crossover strategy {self.crossover_strategy}"
                "and eta_c. However, eta_c is ignored because it is only used when "
                "the crossover_strategy is set to sbx."
            )
        eta_c = 1.0 if self.eta_c is None else self.eta_c

        if (
            self.mutation_polynomial_distribution_index is not None
        ) and self.mutation_strategy != "polynomial":
            warnings.warn(
                "You specified a mutation_polynomial_distribution_index but"
                "did not choose polynomial as your mutation_strategy. Thus, "
                "mutation_polynomial_distribution_index will be ignored."
            )
        if (
            self.mutation_gaussian_width is not None
            and self.mutation_strategy != "gaussian"
        ):
            warnings.warn(
                "You specified a mutation_gaussian_width but "
                "did not choose gaussion as your mutation_strategy. "
                "Thus, mutation_gaussian_width will be ignored."
            )
        if (
            self.selection_strategy != "truncated"
            and self.selection_truncated_n_best is not None
        ):
            warnings.warn(
                "You specified selection_truncated_n_best but "
                "did not specify truncated as your selection strategy. "
                "Therefore, selection_truncated_n_best is ignored."
            )
        if (
            self.selection_strategy != "tournament"
            and self.selection_tournament_size is not None
        ):
            warnings.warn(
                "You specified selection_tournament_size but "
                "did not specify tournament as your selection strategy. "
                "Therefore, selection_tournament_size is ignored."
            )

        if (
            self.mutation_strategy == "gaussian"
            and self.mutation_gaussian_width is not None
        ):
            param_m = self.mutation_gaussian_width
        elif (
            self.mutation_strategy == "polynomial"
            and self.mutation_polynomial_distribution_index is not None
        ):
            param_m = self.mutation_polynomial_distribution_index
        else:
            param_m = 1.0

        if (
            self.selection_strategy == "truncated"
            and self.selection_truncated_n_best is not None
        ):
            param_s = self.selection_truncated_n_best
        elif (
            self.selection_strategy == "tournament"
            and self.selection_tournament_size is not None
        ):
            param_s = self.selection_tournament_size
        else:
            param_s = 2

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "cr": self.crossover_probability,
            "eta_c": eta_c,
            "m": self.mutation_probability,
            "param_m": param_m,
            "crossover": self.crossover_strategy,
            "mutation": self.mutation_strategy,
            "selection": self.selection_strategy,
            "param_s": param_s,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="sga",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_sade",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSade(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    jde: bool = True
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    mutation_variant: Literal[
        "best/1/exp",
        "rand/1/exp",
        "rand-to-best/1/exp",
        "best/2/exp",
        "rand/2/exp",
        "best/1/bin",
        "rand/1/bin",
        "rand-to-best/1/bin",
        "best/2/bin",
        "rand/2/bin",
        "rand/3/exp",
        "rand/3/bin",
        "best/3/exp",
        "best/3/bin",
        "rand-to-current/2/exp",
        "rand-to-current/2/bin",
        "rand-to-best-and-current/2/exp",
        "rand-to-best-and-current/2/bin",
    ] = "rand/1/exp"
    keep_adapted_params: bool = False
    ftol: NonNegativeFloat = 1e-6
    xtol: NonNegativeFloat = 1e-6

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )
        mutation_variant_str_to_int = {
            "best/1/exp": 1,
            "rand/1/exp": 2,
            "rand-to-best/1/exp": 3,
            "best/2/exp": 4,
            "rand/2/exp": 5,
            "best/1/bin": 6,
            "rand/1/bin": 7,
            "rand-to-best/1/bin": 8,
            "best/2/bin": 9,
            "rand/2/bin": 10,
            "rand/3/exp": 11,
            "rand/3/bin": 12,
            "best/3/exp": 13,
            "best/3/bin": 14,
            "rand-to-current/2/exp": 15,
            "rand-to-current/2/bin": 16,
            "rand-to-best-and-current/2/exp": 17,
            "rand-to-best-and-current/2/bin": 18,
        }
        mutation_variant = _convert_str_to_int(
            str_to_int=mutation_variant_str_to_int, value=self.mutation_variant
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "variant": mutation_variant,
            "variant_adptv": 1 if self.jde else 2,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="sade",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_cmaes",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoCmaes(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    # TODO: Refine type to fix range [0,1]
    backward_horizon: NonNegativeFloat | None = None
    # TODO: Refine type to fix range [0,1]
    variance_loss_compensation: NonNegativeFloat | None = None
    # TODO: Refine type to fix range [0,1]
    learning_rate_rank_one_update: NonNegativeFloat | None = None
    # TODO: Refine type to fix range [0,1]
    learning_rate_rank_mu_update: NonNegativeFloat | None = None
    # TODO: Check if should be NonNegativeFloat
    initial_step_size: float = 0.5
    ftol: NonNegativeFloat = 1e-6
    xtol: NonNegativeFloat = 1e-6
    keep_adapted_params: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "cc": self.backward_horizon if self.backward_horizon is not None else -1.0,
            "cs": self.variance_loss_compensation
            if self.variance_loss_compensation is not None
            else -1.0,
            "c1": self.learning_rate_rank_one_update
            if self.learning_rate_rank_one_update is not None
            else -1.0,
            "cmu": self.learning_rate_rank_mu_update
            if self.learning_rate_rank_mu_update is not None
            else -1.0,
            "sigma0": self.initial_step_size,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
            "force_bounds": True,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="cmaes",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_simulated_annealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSimulatedAnnealing(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    start_temperature: PositiveFloat = 10.0
    # TODO: Check if type should be same as start_temperature
    end_temperature: float = 0.01
    # TODO: Check if type should be NonNegativeInt
    n_temp_adjustments: int = 10
    # TODO: Check if type should be NonNegativeInt
    n_range_adjustments: int = 10
    # TODO: Check if type should be NonNegativeInt
    bin_size: int = 10
    # TODO: Refine type to fix range [0,1]
    start_range: NonNegativeFloat = 1.0

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "Ts": self.start_temperature,
            "Tf": self.end_temperature,
            "n_T_adj": self.n_temp_adjustments,
            "n_range_adj": self.n_range_adjustments,
            "bin_size": self.bin_size,
            "start_range": self.start_range,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="simulated_annealing",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoPso(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    # TODO: Refine type to fix range [0,1]
    omega: NonNegativeFloat = 0.7298
    # TODO: Refine type to fix range [0,4]
    force_of_previous_best: NonNegativeFloat = 2.05
    # TODO: Refine type to fix range [0,4]
    force_of_best_in_neighborhood: NonNegativeFloat = 2.05
    # TODO: Refine type to fix range [0,1]
    max_velocity: NonNegativeFloat = 0.5
    algo_variant: Literal[
        "canonical_inertia",
        "social_and_cog_rand",
        "all_components_rand",
        "one_rand",
        "canonical_constriction",
        "fips",
    ] = "canonical_constriction"
    neighbor_definition: Literal[
        "gbest",
        "lbest",
        "Von Neumann",
        "Adaptive random",
    ] = "lbest"
    neighbor_param: int | None = None
    keep_velocities: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if (
            self.neighbor_definition in ["gbest", "Von Neumann"]
            and self.neighbor_param is not None
        ):
            warnings.warn(
                "You gave a neighbor parameter but selected a neighbor_definition "
                "that ignores this parameter."
            )

        neighbor_param = 4 if self.neighbor_param is None else self.neighbor_param

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        neighbor_definition_str_to_int = {
            "gbest": 1,
            "lbest": 2,
            "Von Neumann": 3,
            "Adaptive random": 4,
        }
        algo_variant_str_to_int = {
            "canonical_inertia": 1,
            "social_and_cog_rand": 2,
            "all_components_rand": 3,
            "one_rand": 4,
            "canonical_constriction": 5,
            "fips": 6,
        }

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "omega": self.omega,
            "eta1": self.force_of_previous_best,
            "eta2": self.force_of_best_in_neighborhood,
            "max_vel": self.max_velocity,
            "variant": _convert_str_to_int(algo_variant_str_to_int, self.algo_variant),
            "neighb_type": _convert_str_to_int(
                neighbor_definition_str_to_int, self.neighbor_definition
            ),
            "neighb_param": neighbor_param,
            "memory": self.keep_velocities,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="pso",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )

        return res


@mark.minimizer(
    name="pygmo_pso_gen",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
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
class PygmoPsoGen(Algorithm):
    population_size: int | None = None
    n_cores: PositiveInt = 1
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    # TODO: Refine type to fix range [0,1]
    omega: NonNegativeFloat = 0.7298
    # TODO: Refine type to fix range [0,4]
    force_of_previous_best: NonNegativeFloat = 2.05
    # TODO: Refine type to fix range [0,4]
    force_of_best_in_neighborhood: NonNegativeFloat = 2.05
    # TODO: Refine type to fix range [0,1]
    max_velocity: NonNegativeFloat = 0.5
    algo_variant: Literal[
        "canonical_inertia",
        "social_and_cog_rand",
        "all_components_rand",
        "one_rand",
        "canonical_constriction",
        "fips",
    ] = "canonical_constriction"
    neighbor_definition: Literal[
        "gbest",
        "lbest",
        "Von Neumann",
        "Adaptive random",
    ] = "lbest"
    neighbor_param: int | None = None
    keep_velocities: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if (
            self.neighbor_definition in ["gbest", "Von Neumann"]
            and self.neighbor_param is not None
        ):
            warnings.warn(
                "You gave a neighbor parameter but selected a neighbor_definition "
                "that ignores this parameter."
            )
        neighbor_param = 4 if self.neighbor_param is None else self.neighbor_param
        neighbor_str_to_int = {
            "gbest": 1,
            "lbest": 2,
            "Von Neumann": 3,
            "Adaptive random": 4,
        }
        neighbor_type = _convert_str_to_int(
            neighbor_str_to_int, self.neighbor_definition
        )
        algo_variant_str_to_int = {
            "canonical_inertia": 1,
            "social_and_cog_rand": 2,
            "all_components_rand": 3,
            "one_rand": 4,
            "canonical_constriction": 5,
            "fips": 6,
        }
        algo_variant = _convert_str_to_int(algo_variant_str_to_int, self.algo_variant)

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "omega": self.omega,
            "eta1": self.force_of_previous_best,
            "eta2": self.force_of_best_in_neighborhood,
            "max_vel": self.max_velocity,
            "variant": algo_variant,
            "neighb_type": neighbor_type,
            "neighb_param": neighbor_param,
            "memory": self.keep_velocities,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="pso_gen",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=self.n_cores,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_mbh",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoMbh(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    inner_algorithm: pg.algorithm | None = None
    # this is 30 instead of 5 in pygmo for our sum of squares test to pass
    stopping_max_inner_runs_without_improvement: PositiveInt = 30
    perturbation: float = 0.01

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        # the min default population size is this large to pass our sum of
        # squares tests.
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=250
        )

        algo_specific_options = {
            "algo": self.inner_algorithm,
            "stop": self.stopping_max_inner_runs_without_improvement,
            "perturb": self.perturbation,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="mbh",
            specific_options=algo_specific_options,
            population_size=population_size,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
            n_cores=1,
        )
        return res


@mark.minimizer(
    name="pygmo_xnes",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoXnes(Algorithm):
    population_size: float | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    # TODO: Refine type to fix range [0,1]
    learning_rate_mean_update: NonNegativeFloat | None = 1.0
    # TODO: Refine type to fix range [0,1]
    learning_rate_step_size_update: NonNegativeFloat | None = None
    # TODO: Refine type to fix range [0,1]
    learning_rate_cov_matrix_update: NonNegativeFloat | None = None
    # TODO: Refine type to fix range [0,1]
    initial_search_share: NonNegativeFloat | None = 1.0
    ftol: NonNegativeFloat = 1e-6
    xtol: NonNegativeFloat = 1e-6
    keep_adapted_params: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )
        eta_mu = (
            -1
            if self.learning_rate_mean_update is None
            else self.learning_rate_mean_update
        )
        eta_sigma = (
            -1
            if self.learning_rate_step_size_update is None
            else self.learning_rate_step_size_update
        )
        eta_b = (
            -1
            if self.learning_rate_cov_matrix_update is None
            else self.learning_rate_cov_matrix_update
        )
        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "eta_mu": eta_mu,
            "eta_sigma": eta_sigma,
            "eta_b": eta_b,
            "sigma0": self.initial_search_share,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
            "force_bounds": True,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="xnes",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_gwo",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoGwo(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )
        algo_specific_options = {
            "gen": self.stopping_maxiter,
        }
        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="gwo",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_compass_search",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoCompassSearch(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    # TODO: Refine type to fix range (0,1]
    start_range: PositiveFloat = 0.1
    # TODO?: mus be in (0,start_range]
    stop_range: PositiveFloat = 0.01
    # TODO: Refine type to fix range (0,1)
    reduction_coeff: PositiveFloat = 0.5

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.population_size is not None:
            warnings.warn(
                f"You specified population size {self.population_size}. "
                "compass_search does not have a population so this argument is ignored."
            )
            population_size = self.population_size
        else:
            # if discard_start_params is False population_size - 1
            # must still be positive
            population_size = 100

        algo_specific_options = {
            "max_fevals": self.stopping_maxfun,
            "start_range": self.start_range,
            "stop_range": self.stop_range,
            "reduction_coeff": self.reduction_coeff,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="compass_search",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_ihs",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoIhs(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    # TODO: Probably refine type to fix range [0,1]
    choose_from_memory_probability: NonNegativeFloat = 0.85
    # TODO: Refine type to fix range [0,1]
    min_pitch_adjustment_rate: NonNegativeFloat = 0.35
    # TODO: Refine type to fix range [0,1]
    max_pitch_adjustment_rate: NonNegativeFloat = 0.99
    min_distance_bandwidth: PositiveFloat = 1e-5
    max_distance_bandwidth: PositiveFloat = 1.0

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.population_size is not None:
            warnings.warn("The population size has no effect on IHS' performance.")

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=1
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "phmcr": self.choose_from_memory_probability,
            "ppar_min": self.min_pitch_adjustment_rate,
            "ppar_max": self.max_pitch_adjustment_rate,
            "bw_min": self.min_distance_bandwidth,
            "bw_max": self.max_distance_bandwidth,
        }
        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="ihs",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_de1220",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoDe1220(Algorithm):
    population_size: int | None = None
    seed: int | None = None
    discard_start_params: bool = False
    jde: bool = True
    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    allowed_variants: List[str] | None = None
    keep_adapted_params: bool = False
    ftol: NonNegativeFloat = 1e-6
    xtol: NonNegativeFloat = 1e-6

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        variant_str_to_int = {
            "best/1/exp": 1,
            "rand/1/exp": 2,
            "rand-to-best/1/exp": 3,
            "best/2/exp": 4,
            "rand/2/exp": 5,
            "best/1/bin": 6,
            "rand/1/bin": 7,
            "rand-to-best/1/bin": 8,
            "best/2/bin": 9,
            "rand/2/bin": 10,
            "rand/3/exp": 11,
            "rand/3/bin": 12,
            "best/3/exp": 13,
            "best/3/bin": 14,
            "rand-to-current/2/exp": 15,
            "rand-to-current/2/bin": 16,
            "rand-to-best-and-current/2/exp": 17,
            "rand-to-best-and-current/2/bin": 18,
        }
        if self.allowed_variants is None:
            allowed_variant_codes = [2, 3, 7, 10, 13, 14, 15, 16]
        else:
            allowed_variant_codes = [
                _convert_str_to_int(variant_str_to_int, variant)
                for variant in self.allowed_variants
            ]

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "variant_adptv": 1 if self.jde else 2,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
            "allowed_variants": allowed_variant_codes,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="de1220",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


# ====================================================================================


def _minimize_pygmo(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    method: str,
    specific_options: dict[str, Any],
    population_size: PositiveInt,
    n_cores: int,
    seed: int | None,
    discard_start_params: bool,
) -> InternalOptimizeResult:
    if not IS_PYGMO_INSTALLED:
        raise NotInstalledError(
            f"The {method} algorithm requires the pygmo package to be installed. "
            "You can install it with 'conda install -c conda-forge pygmo'. Visit "
            "https://esa.github.io/pygmo2/install.html for more detailed installation "
            "instructions."
        )

    bounds = problem.bounds
    if bounds is None or bounds.lower is None or bounds.upper is None:
        raise ValueError(f"{method} requires finitel bounds for all parameters.")
    elif not np.isfinite(bounds.lower).all() or not np.isfinite(bounds.upper).all():
        raise ValueError(f"{method} requires finite bounds for all parameters.")

    pygmo_problem = _create_pygmo_problem(problem, len(x0), n_cores)
    algo = _create_algorithm(method, specific_options, n_cores)
    pop = _create_population(
        problem=pygmo_problem,
        population_size=population_size,
        x=x0,
        seed=seed,
        discard_start_params=discard_start_params,
    )
    evolved = algo.evolve(pop)
    result = _process_pygmo_result(evolved)
    return result


def _create_pygmo_problem(
    problem: InternalOptimizationProblem, dim: int, n_cores: int
) -> pg.problem:
    class Problem:
        def fitness(self, x):
            return [problem.fun(x)]

        def get_bounds(self):
            return (problem.bounds.lower, problem.bounds.upper)

        def gradient(self, dv):  # noqa: ARG002
            raise ValueError("No pygmo optimizer should use a gradient.")

        def batch_fitness(self, dvs):
            x_list = list(dvs.reshape(-1, dim))
            eval_list = problem.batch_fun(x_list, n_cores=n_cores)
            evals = np.array(eval_list)
            return evals

    pygmo_problem = pg.problem(Problem())
    return pygmo_problem


def _create_algorithm(
    method: str, algo_options: dict[str, Any], n_cores: int
) -> pg.algorithm:
    """Create a pygmo algorithm."""
    pygmo_uda = getattr(pg, method)
    algo = pygmo_uda(**algo_options)
    try:
        algo.set_bfe(pg.bfe())
    except AttributeError:
        if n_cores >= 2:
            warnings.warn(
                f"Your specified algorithm {method} does not support parallelization. "
                "Choose another algorithm such as pygmo_gaco to parallelize."
            )
    out = pg.algorithm(algo)
    return out


def _create_population(
    problem: InternalOptimizationProblem,
    population_size: int,
    x: NDArray[np.float64],
    seed: int | None,
    discard_start_params: bool,
) -> pg.population:
    if not discard_start_params:
        population_size = population_size - 1

    pop = pg.population(
        problem,
        size=population_size,
        seed=seed,
        b=pg.bfe(),
    )
    if not discard_start_params:
        pop.push_back(x)
    return pop


def _process_pygmo_result(evolved: pg.population) -> InternalOptimizeResult:
    result = InternalOptimizeResult(
        x=evolved.champion_x,
        fun=evolved.champion_f[0],
        success=True,
        message="Number of generations reached.",
        n_fun_evals=evolved.problem.get_fevals(),
        n_jac_evals=evolved.problem.get_gevals(),
    )

    return result


def _convert_str_to_int(str_to_int, value):
    if value in str_to_int:
        out = str_to_int[value]
    elif value not in str_to_int.values():
        raise ValueError(
            f"You specified {value} as value. "
            f"It must be one of {', '.join(str_to_int.keys())}"
        )
    else:
        out = value
    return out

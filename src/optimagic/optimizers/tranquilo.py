from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_TRANQUILO_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

if IS_TRANQUILO_INSTALLED:
    from tranquilo.options import (
        AcceptanceOptions,
        FilterOptions,
        FitterOptions,
        NoiseAdaptationOptions,
        RadiusOptions,
        SamplerOptions,
        StagnationOptions,
        SubsolverOptions,
        VarianceEstimatorOptions,
    )
    from tranquilo.tranquilo import _tranquilo
else:
    AcceptanceOptions = Any
    FilterOptions = Any
    FitterOptions = Any
    NoiseAdaptationOptions = Any
    RadiusOptions = Any
    SamplerOptions = Any
    StagnationOptions = Any
    SubsolverOptions = Any
    VarianceEstimatorOptions = Any


@mark.minimizer(
    name="tranquilo",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_TRANQUILO_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=True,
)
@dataclass(frozen=True)
class Tranquilo(Algorithm):
    # function type
    functype: Literal["scalar"] = "scalar"
    # basic options
    noisy: bool = False
    # convergence options
    disable_convergence: bool = False
    convergence_ftol_abs: NonNegativeFloat = 0.0
    convergence_gtol_abs: NonNegativeFloat = 0.0
    convergence_xtol_abs: NonNegativeFloat = 0.0
    convergence_ftol_rel: NonNegativeFloat = 2e-9
    convergence_gtol_rel: NonNegativeFloat = 1e-8
    convergence_xtol_rel: NonNegativeFloat = 1e-8
    convergence_min_trust_region_radius: NonNegativeFloat = 0.0
    # stopping options
    stopping_maxfun: PositiveInt = 2_000
    stopping_maxiter: PositiveInt = 200
    stopping_maxtime: NonNegativeFloat = np.inf
    # single advanced options
    batch_evaluator: Literal[
        "joblib",
        "pathos",
    ] = "joblib"
    n_cores: PositiveInt = 1
    batch_size: PositiveInt | None = None
    sample_size: PositiveInt | None = None
    model_type: (
        Literal[
            "quadratic",
            "linear",
        ]
        | None
    ) = None
    search_radius_factor: PositiveFloat | None = None
    n_evals_per_point: NonNegativeInt | None = None
    n_evals_at_start: NonNegativeInt | None = None
    seed: int | None = 925408
    # bundled advanced options
    radius_options: RadiusOptions | None = None
    stagnation_options: StagnationOptions | None = None
    noise_adaptation_options: NoiseAdaptationOptions | None = None
    # component names and related options
    sampler: (
        Literal[
            "optimal_hull",
            "random_hull",
            "random_interior",
        ]
        | Callable
    ) = "optimal_hull"
    sampler_options: SamplerOptions | None = None
    sample_filter: (
        Literal[
            "discard_all",
            "keep_all",
            "clustering",
            "drop_excess",
        ]
        | Callable
        | None
    ) = None
    sample_filter_options: FilterOptions | None = None
    model_fitter: (
        Literal[
            "ols",
            "ridge",
            "powell",
            "tranquilo",
        ]
        | Callable
        | None
    ) = None
    model_fitter_options: FitterOptions | None = None
    cube_subsolver: (
        Literal[
            "bntr",
            "bntr_fast",
            "fallback_cube",
            "fallback_multistart",
        ]
        | Callable
    ) = "bntr_fast"
    sphere_subsolver: (
        Literal[
            "gqtpar",
            "gqtpar_fast",
            "fallback_reparametrized",
            "fallback_inscribed_cube",
            "fallback_norm_constraint",
        ]
        | Callable
    ) = "gqtpar_fast"
    retry_subproblem_with_fallback: bool = True
    subsolver_options: SubsolverOptions | None = None
    acceptance_decider: (
        Literal[
            "classic",
            "naive_noisy",
            "classic_line_search",
            "noisy",
        ]
        | Callable
        | None
    ) = None
    acceptance_decider_options: AcceptanceOptions | None = None
    variance_estimator: Literal["classic"] | Callable = "classic"
    variance_estimator_options: VarianceEstimatorOptions | None = None
    infinity_handler: Literal["relative"] | Callable = "relative"
    residualize: bool | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_TRANQUILO_INSTALLED:
            raise NotInstalledError(
                "The 'tranquilo-ls' algorithm requires the tranquilo package "
                "to be installed. You can install it with "
                "'conda install -c conda-forge tranquilo'."
            )
        raw_res = _tranquilo(
            functype="scalar",
            criterion=problem.fun,
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            noisy=self.noisy,
            disable_convergence=self.disable_convergence,
            convergence_absolute_criterion_tolerance=self.convergence_ftol_abs,
            convergence_absolute_gradient_tolerance=self.convergence_gtol_abs,
            convergence_absolute_params_tolerance=self.convergence_xtol_abs,
            convergence_relative_criterion_tolerance=self.convergence_ftol_rel,
            convergence_relative_gradient_tolerance=self.convergence_gtol_rel,
            convergence_relative_params_tolerance=self.convergence_xtol_rel,
            convergence_min_trust_region_radius=self.convergence_min_trust_region_radius,
            stopping_max_criterion_evaluations=self.stopping_maxfun,
            stopping_max_iterations=self.stopping_maxiter,
            stopping_max_time=self.stopping_maxtime,
            batch_evaluator=self.batch_evaluator,
            n_cores=self.n_cores,
            batch_size=self.batch_size,
            sample_size=self.sample_size,
            model_type=self.model_type,
            search_radius_factor=self.search_radius_factor,
            n_evals_per_point=self.n_evals_per_point,
            n_evals_at_start=self.n_evals_at_start,
            seed=self.seed,
            radius_options=self.radius_options,
            stagnation_options=self.stagnation_options,
            noise_adaptation_options=self.noise_adaptation_options,
            sampler=self.sampler,
            sampler_options=self.sampler_options,
            sample_filter=self.sample_filter,
            sample_filter_options=self.sample_filter_options,
            model_fitter=self.model_fitter,
            model_fitter_options=self.model_fitter_options,
            cube_subsolver=self.cube_subsolver,
            sphere_subsolver=self.sphere_subsolver,
            retry_subproblem_with_fallback=self.retry_subproblem_with_fallback,
            subsolver_options=self.subsolver_options,
            acceptance_decider=self.acceptance_decider,
            acceptance_decider_options=self.acceptance_decider_options,
            variance_estimator=self.variance_estimator,
            variance_estimator_options=self.variance_estimator_options,
            infinity_handler=self.infinity_handler,
            residualize=self.residualize,
        )

        res = InternalOptimizeResult(
            x=raw_res["solution_x"],
            fun=raw_res["solution_criterion"],
            message=raw_res["message"],
            info={"states": raw_res["states"]},
        )
        return res


@mark.minimizer(
    name="tranquilo_ls",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=IS_TRANQUILO_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=True,
)
@dataclass(frozen=True)
class TranquiloLS(Algorithm):
    # basic options
    noisy: bool = False
    # convergence options
    disable_convergence: bool = False
    convergence_ftol_abs: NonNegativeFloat = 0.0
    convergence_gtol_abs: NonNegativeFloat = 0.0
    convergence_xtol_abs: NonNegativeFloat = 0.0
    convergence_ftol_rel: NonNegativeFloat = 2e-9
    convergence_gtol_rel: NonNegativeFloat = 1e-8
    convergence_xtol_rel: NonNegativeFloat = 1e-8
    convergence_min_trust_region_radius: NonNegativeFloat = 0.0
    # stopping options
    stopping_maxfun: PositiveInt = 2_000
    stopping_maxiter: PositiveInt = 200
    stopping_maxtime: NonNegativeFloat = np.inf
    # single advanced options
    batch_evaluator: Literal[
        "joblib",
        "pathos",
    ] = "joblib"
    n_cores: PositiveInt = 1
    batch_size: PositiveInt | None = None
    sample_size: PositiveInt | None = None
    model_type: (
        Literal[
            "quadratic",
            "linear",
        ]
        | None
    ) = None
    search_radius_factor: PositiveFloat | None = None
    n_evals_per_point: NonNegativeInt | None = None
    n_evals_at_start: NonNegativeInt | None = None
    seed: int | None = 925408
    # bundled advanced options
    radius_options: RadiusOptions | None = None
    stagnation_options: StagnationOptions | None = None
    noise_adaptation_options: NoiseAdaptationOptions | None = None
    # component names and related options
    sampler: (
        Literal[
            "optimal_hull",
            "random_hull",
            "random_interior",
        ]
        | Callable
    ) = "optimal_hull"
    sampler_options: SamplerOptions | None = None
    sample_filter: (
        Literal[
            "discard_all",
            "keep_all",
            "clustering",
            "drop_excess",
        ]
        | Callable
        | None
    ) = None
    sample_filter_options: FilterOptions | None = None
    model_fitter: (
        Literal[
            "ols",
            "ridge",
            "powell",
            "tranquilo",
        ]
        | Callable
        | None
    ) = None
    model_fitter_options: FitterOptions | None = None
    cube_subsolver: (
        Literal[
            "bntr",
            "bntr_fast",
            "fallback_cube",
            "fallback_multistart",
        ]
        | Callable
    ) = "bntr_fast"
    sphere_subsolver: (
        Literal[
            "gqtpar",
            "gqtpar_fast",
            "fallback_reparametrized",
            "fallback_inscribed_cube",
            "fallback_norm_constraint",
        ]
        | Callable
    ) = "gqtpar_fast"
    retry_subproblem_with_fallback: bool = True
    subsolver_options: SubsolverOptions | None = None
    acceptance_decider: (
        Literal[
            "classic",
            "naive_noisy",
            "classic_line_search",
            "noisy",
        ]
        | Callable
        | None
    ) = None
    acceptance_decider_options: AcceptanceOptions | None = None
    variance_estimator: Literal["classic"] | Callable = "classic"
    variance_estimator_options: VarianceEstimatorOptions | None = None
    infinity_handler: Literal["relative"] | Callable = "relative"
    residualize: bool | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_TRANQUILO_INSTALLED:
            raise NotInstalledError(
                "The 'tranquilo-ls' algorithm requires the tranquilo package "
                "to be installed. You can install it with "
                "'conda install -c conda-forge tranquilo'."
            )
        raw_res = _tranquilo(
            functype="least_squares",
            criterion=problem.fun,
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            noisy=self.noisy,
            disable_convergence=self.disable_convergence,
            convergence_absolute_criterion_tolerance=self.convergence_ftol_abs,
            convergence_absolute_gradient_tolerance=self.convergence_gtol_abs,
            convergence_absolute_params_tolerance=self.convergence_xtol_abs,
            convergence_relative_criterion_tolerance=self.convergence_ftol_rel,
            convergence_relative_gradient_tolerance=self.convergence_gtol_rel,
            convergence_relative_params_tolerance=self.convergence_xtol_rel,
            convergence_min_trust_region_radius=self.convergence_min_trust_region_radius,
            stopping_max_criterion_evaluations=self.stopping_maxfun,
            stopping_max_iterations=self.stopping_maxiter,
            stopping_max_time=self.stopping_maxtime,
            batch_evaluator=self.batch_evaluator,
            n_cores=self.n_cores,
            batch_size=self.batch_size,
            sample_size=self.sample_size,
            model_type=self.model_type,
            search_radius_factor=self.search_radius_factor,
            n_evals_per_point=self.n_evals_per_point,
            n_evals_at_start=self.n_evals_at_start,
            seed=self.seed,
            radius_options=self.radius_options,
            stagnation_options=self.stagnation_options,
            noise_adaptation_options=self.noise_adaptation_options,
            sampler=self.sampler,
            sampler_options=self.sampler_options,
            sample_filter=self.sample_filter,
            sample_filter_options=self.sample_filter_options,
            model_fitter=self.model_fitter,
            model_fitter_options=self.model_fitter_options,
            cube_subsolver=self.cube_subsolver,
            sphere_subsolver=self.sphere_subsolver,
            retry_subproblem_with_fallback=self.retry_subproblem_with_fallback,
            subsolver_options=self.subsolver_options,
            acceptance_decider=self.acceptance_decider,
            acceptance_decider_options=self.acceptance_decider_options,
            variance_estimator=self.variance_estimator,
            variance_estimator_options=self.variance_estimator_options,
            infinity_handler=self.infinity_handler,
            residualize=self.residualize,
        )
        res = InternalOptimizeResult(
            x=raw_res["solution_x"],
            fun=raw_res["solution_criterion"],
            message=raw_res["message"],
            info={"states": raw_res["states"]},
        )
        return res

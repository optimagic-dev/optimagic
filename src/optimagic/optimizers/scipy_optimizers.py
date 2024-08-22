"""Implement scipy algorithms.

The following ``scipy`` algorithms are not supported because they
require the specification of the Hessian:

- dogleg
- trust-ncg
- trust-exact
- trust-krylov

The following arguments are not supported as part of ``algo_options``:

- ``disp``
    If set to True would print a convergence message.
    In optimagic it's always set to its default False.
    Refer to optimagic's result dictionary's "success" entry for the convergence
    message.
- ``return_all``
    If set to True, a list of the best solution at each iteration is returned.
    In optimagic it's always set to its default False.
- ``tol``
    This argument of minimize (not an options key) is passed as different types of
    tolerance (gradient, parameter or criterion, as well as relative or absolute)
    depending on the selected algorithm. We require the user to explicitely input
    the tolerance criteria or use our defaults instead.
- ``args``
    This argument of minimize (not an options key) is partialed into the function
    for the user. Specify ``criterion_kwargs`` in ``maximize`` or ``minimize`` to
    achieve the same behavior.
- ``callback``
    This argument would be called after each iteration and the algorithm would
    terminate if it returned True.

"""

import functools
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Tuple

import numpy as np
import scipy
import scipy.optimize
from numpy.typing import NDArray
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import OptimizeResult as ScipyOptimizeResult

from optimagic import mark
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_SECOND_BEST_FTOL_ABS,
    CONVERGENCE_SECOND_BEST_XTOL_ABS,
    CONVERGENCE_XTOL_ABS,
    CONVERGENCE_XTOL_REL,
    LIMITED_MEMORY_STORAGE_LENGTH,
    MAX_LINE_SEARCH_STEPS,
    STOPPING_MAXFUN,
    STOPPING_MAXFUN_GLOBAL,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
    vector_as_list_of_scalar_constraints,
)
from optimagic.typing import (
    AggregationLevel,
    BatchEvaluator,
    NegativeFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from optimagic.utilities import calculate_trustregion_initial_radius


@mark.minimizer(
    name="scipy_lbfgsb",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyLBFGSB(Algorithm):
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    limited_memory_storage_length: PositiveInt = LIMITED_MEMORY_STORAGE_LENGTH
    max_line_search_steps: PositiveInt = MAX_LINE_SEARCH_STEPS

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "maxcor": self.limited_memory_storage_length,
            "ftol": self.convergence_ftol_rel,
            "gtol": self.convergence_gtol_abs,
            "maxfun": self.stopping_maxfun,
            "maxiter": self.stopping_maxiter,
            "maxls": self.max_line_search_steps,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=_get_scipy_bounds(problem.bounds),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_slsqp",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipySLSQP(Algorithm):
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    stopping_maxiter: PositiveInt = STOPPING_MAXITER

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "maxiter": self.stopping_maxiter,
            "ftol": self.convergence_ftol_abs,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=_get_scipy_bounds(problem.bounds),
            constraints=problem.nonlinear_constraints,
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_neldermead",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyNelderMead(Algorithm):
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_XTOL_ABS
    adaptive: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "maxiter": self.stopping_maxiter,
            "maxfev": self.stopping_maxfun,
            "xatol": self.convergence_xtol_abs,
            "fatol": self.convergence_ftol_abs,
            # TODO: Benchmark if adaptive = True works better
            "adaptive": self.adaptive,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun,
            x0=x0,
            bounds=_get_scipy_bounds(problem.bounds),
            method="Nelder-Mead",
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_powell",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyPowell(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    stopping_maxiter: PositiveInt = STOPPING_MAXITER

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "xtol": self.convergence_xtol_rel,
            "ftol": self.convergence_ftol_rel,
            "maxfev": self.stopping_maxfun,
            "maxiter": self.stopping_maxiter,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun,
            x0=x0,
            method="Powell",
            bounds=_get_scipy_bounds(problem.bounds),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_bfgs",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyBFGS(Algorithm):
    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    norm: NonNegativeFloat = np.inf

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "gtol": self.convergence_gtol_abs,
            "maxiter": self.stopping_maxiter,
            "norm": self.norm,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac, x0=x0, method="BFGS", jac=True, options=options
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_conjugate_gradient",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyConjugateGradient(Algorithm):
    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    norm: NonNegativeFloat = np.inf

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "gtol": self.convergence_gtol_abs,
            "maxiter": self.stopping_maxiter,
            "norm": self.norm,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac, x0=x0, method="CG", jac=True, options=options
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_newton_cg",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyNewtonCG(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    stopping_maxiter: PositiveInt = STOPPING_MAXITER

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "xtol": self.convergence_xtol_rel,
            "maxiter": self.stopping_maxiter,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="Newton-CG",
            jac=True,
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_cobyla",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyCOBYLA(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    trustregion_initial_radius: PositiveFloat | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        # TODO: Maybe we should leave the radius at their default
        if self.trustregion_initial_radius is None:
            radius = calculate_trustregion_initial_radius(x0)
        else:
            radius = self.trustregion_initial_radius

        options = {
            "maxiter": self.stopping_maxiter,
            "rhobeg": radius,
        }

        # cannot handle equality constraints
        nonlinear_constraints = equality_as_inequality_constraints(
            problem.nonlinear_constraints
        )

        raw_res = scipy.optimize.minimize(
            fun=problem.fun,
            x0=x0,
            method="COBYLA",
            constraints=nonlinear_constraints,
            options=options,
            tol=self.convergence_xtol_rel,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_ls_trf",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyLSTRF(Algorithm):
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    relative_step_size_diff_approx: NonNegativeFloat | None = None
    tr_solver: Literal["exact", "lsmr"] | None = None
    tr_solver_options: dict[str, Any] | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.tr_solver_options is None:
            tr_solver_options = {}
        else:
            tr_solver_options = self.tr_solver_options

        raw_res = scipy.optimize.least_squares(
            fun=problem.fun,
            x0=x0,
            # This optimizer does not work with fun_and_jac
            jac=problem.jac,
            bounds=(problem.bounds.lower, problem.bounds.upper),
            method="trf",
            max_nfev=self.stopping_maxfun,
            ftol=self.convergence_ftol_rel,
            gtol=self.convergence_gtol_rel,
            diff_step=self.relative_step_size_diff_approx,
            tr_solver=self.tr_solver,
            tr_options=tr_solver_options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_ls_dogbox",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyLSDogbox(Algorithm):
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    relative_step_size_diff_approx: NonNegativeFloat | None = None
    tr_solver: Literal["exact", "lsmr"] | None = None
    tr_solver_options: dict[str, Any] | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.tr_solver_options is None:
            tr_solver_options = {}
        else:
            tr_solver_options = self.tr_solver_options

        raw_res = scipy.optimize.least_squares(
            fun=problem.fun,
            x0=x0,
            # This optimizer does not work with fun_and_jac
            jac=problem.jac,
            bounds=(problem.bounds.lower, problem.bounds.upper),
            method="dogbox",
            max_nfev=self.stopping_maxfun,
            ftol=self.convergence_ftol_rel,
            gtol=self.convergence_gtol_rel,
            diff_step=self.relative_step_size_diff_approx,
            tr_solver=self.tr_solver,
            tr_options=tr_solver_options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_ls_lm",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyLSLM(Algorithm):
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    relative_step_size_diff_approx: NonNegativeFloat | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        raw_res = scipy.optimize.least_squares(
            fun=problem.fun,
            x0=x0,
            # This optimizer does not work with fun_and_jac
            jac=problem.jac,
            method="lm",
            max_nfev=self.stopping_maxfun,
            ftol=self.convergence_ftol_rel,
            gtol=self.convergence_gtol_rel,
            diff_step=self.relative_step_size_diff_approx,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_truncated_newton",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyTruncatedNewton(Algorithm):
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    max_hess_evaluations_per_iteration: int = -1
    max_step_for_line_search: NonNegativeFloat = 0
    line_search_severity: float = -1
    finite_difference_precision: NonNegativeFloat = 0
    criterion_rescale_factor: float = -1
    # TODO: Check type hint for `func_min_estimate`
    func_min_estimate: float = 0

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "ftol": self.convergence_ftol_abs,
            "xtol": self.convergence_xtol_abs,
            "gtol": self.convergence_gtol_abs,
            "maxfun": self.stopping_maxfun,
            "maxCGit": self.max_hess_evaluations_per_iteration,
            "stepmx": self.max_step_for_line_search,
            "minfev": self.func_min_estimate,
            "eta": self.line_search_severity,
            "accuracy": self.finite_difference_precision,
            "rescale": self.criterion_rescale_factor,
        }

        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="TNC",
            jac=True,
            bounds=_get_scipy_bounds(problem.bounds),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_trust_constr",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyTrustConstr(Algorithm):
    # TODO: Check if can be set to CONVERGENCE_GTOL_ABS
    convergence_gtol_abs: NonNegativeFloat = 1e-08
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    trustregion_initial_radius: PositiveFloat | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.trustregion_initial_radius is None:
            trustregion_initial_radius = calculate_trustregion_initial_radius(x0)
        else:
            trustregion_initial_radius = self.trustregion_initial_radius

        options = {
            "gtol": self.convergence_gtol_abs,
            "maxiter": self.stopping_maxiter,
            "xtol": self.convergence_xtol_rel,
            "initial_tr_radius": trustregion_initial_radius,
        }

        # cannot handle equality constraints
        nonlinear_constraints = equality_as_inequality_constraints(
            problem.nonlinear_constraints
        )

        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            jac=True,
            x0=x0,
            method="trust-constr",
            bounds=_get_scipy_bounds(problem.bounds),
            constraints=_get_scipy_constraints(nonlinear_constraints),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


def process_scipy_result(scipy_res: ScipyOptimizeResult) -> InternalOptimizeResult:
    res = InternalOptimizeResult(
        x=scipy_res.x,
        fun=scipy_res.fun,
        success=bool(scipy_res.success),
        message=str(scipy_res.message),
        n_fun_evals=scipy_res.get("nfev"),
        n_jac_evals=scipy_res.get("njev"),
        n_hess_evals=scipy_res.get("nhev"),
        n_iterations=scipy_res.get("nit"),
        # TODO: Pass on more things once we can convert them to external
        status=None,
        jac=None,
        hess=None,
        hess_inv=None,
        max_constraint_violation=None,
        info=None,
        history=None,
    )
    return res


def _get_scipy_constraints(constraints):
    """Transform internal nonlinear constraints to scipy readable format.

    This format is currently only used by scipy_trust_constr.

    """
    scipy_constraints = [_internal_to_scipy_constraint(c) for c in constraints]
    return scipy_constraints


def _internal_to_scipy_constraint(c):
    new_constr = NonlinearConstraint(
        fun=c["fun"],
        lb=np.zeros(c["n_constr"]),
        ub=np.tile(np.inf, c["n_constr"]),
        jac=c["jac"],
    )
    return new_constr


@mark.minimizer(
    name="scipy_basinhopping",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyBasinhopping(Algorithm):
    local_algorithm: (
        Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        | Callable
    ) = "L-BFGS-B"
    n_local_optimizations: PositiveInt = 100
    temperature: NonNegativeFloat = 1.0
    stepsize: NonNegativeFloat = 0.5
    local_algo_options: dict[str, Any] | None = None
    take_step: Callable | None = None
    accept_test: Callable | None = None
    interval: PositiveInt = 50
    convergence_n_unchanged_iterations: PositiveInt | None = None
    seed: int | np.random.Generator | np.random.RandomState | None = None
    target_accept_rate: NonNegativeFloat = 0.5
    stepwise_factor: NonNegativeFloat = 0.9

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        n_local_optimizations = max(1, self.n_local_optimizations - 1)
        if self.local_algo_options is None:
            local_algo_options = {}
        else:
            local_algo_options = self.local_algo_options
        minimizer_kwargs = {
            "method": self.local_algorithm,
            "bounds": _get_scipy_bounds(problem.bounds),
            "jac": problem.jac,
        }
        minimizer_kwargs = {**minimizer_kwargs, **local_algo_options}

        res = scipy.optimize.basinhopping(
            func=problem.fun,
            x0=x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=n_local_optimizations,
            T=self.temperature,
            stepsize=self.stepsize,
            take_step=self.take_step,
            accept_test=self.accept_test,
            interval=self.interval,
            niter_success=self.convergence_n_unchanged_iterations,
            seed=self.seed,
            target_accept_rate=self.target_accept_rate,
            stepwise_factor=self.stepwise_factor,
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_brute",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=True,
)
@dataclass(frozen=True)
class ScipyBrute(Algorithm):
    n_grid_points: PositiveInt = 20
    polishing_function: Callable | None = None
    n_cores: PositiveInt = 1
    batch_evaluator: Literal["joblib", "pathos"] | BatchEvaluator = "joblib"

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        workers = _get_workers(self.n_cores, self.batch_evaluator)
        if problem.bounds.lower is None or problem.bounds.upper is None:
            raise ValueError(
                """Global algorithms like scipy_brute need finite bounds
                 for all parameters"""
            )
        raw_res = scipy.optimize.brute(
            func=problem.fun,
            ranges=tuple(
                map(
                    tuple, np.column_stack((problem.bounds.lower, problem.bounds.upper))
                )
            ),
            Ns=self.n_grid_points,
            full_output=True,
            finish=self.polishing_function,
            workers=workers,
        )
        res = InternalOptimizeResult(
            x=raw_res[0],
            fun=raw_res[1],
            n_fun_evals=raw_res[2].size,
            n_iterations=raw_res[2].size,
            success=True,
            message="brute force optimization terminated successfully",
        )

        return res


@mark.minimizer(
    name="scipy_differential_evolution",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=True,
)
@dataclass(frozen=True)
class ScipyDifferentialEvolution(Algorithm):
    strategy: (
        Literal[
            "best1bin",
            "best1exp",
            "rand1exp",
            "randtobest1exp",
            "currenttobest1exp",
            "best2exp",
            "rand2exp",
            "randtobest1bin",
            "currenttobest1bin",
            "best2bin",
            "rand2bin",
            "rand1bin",
        ]
        | Callable
    ) = "best1bin"
    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    population_size_multiplier: NonNegativeInt = 15
    convergence_ftol_rel: NonNegativeFloat = 0.01
    # TODO: Refine type to add ranges [0,2] if float.
    mutation_constant: NonNegativeFloat | Tuple[NonNegativeFloat, NonNegativeFloat] = (
        0.5,
        1,
    )
    # TODO: Refine type to add ranges [0,1].
    recombination_constant: NonNegativeFloat = 0.7
    seed: int | np.random.Generator | np.random.RandomState | None = None
    polish: bool = True
    sampling_method: (
        Literal["latinhypercube", "random", "sobol", "halton"] | NDArray[np.float64]
    ) = "latinhypercube"
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    n_cores: PositiveInt = 1
    batch_evaluator: Literal["joblib", "pathos"] | BatchEvaluator = "joblib"

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        workers = _get_workers(self.n_cores, self.batch_evaluator)
        res = scipy.optimize.differential_evolution(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            strategy=self.strategy,
            maxiter=self.stopping_maxiter,
            popsize=self.population_size_multiplier,
            tol=self.convergence_ftol_rel,
            mutation=self.mutation_constant,
            recombination=self.recombination_constant,
            seed=self.seed,
            polish=self.polish,
            init=self.sampling_method,
            atol=self.convergence_ftol_abs,
            updating="deferred",
            workers=workers,
            constraints=_get_scipy_constraints(problem.nonlinear_constraints),
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_shgo",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipySHGO(Algorithm):
    local_algorithm: (
        Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        | Callable
    ) = "L-BFGS-B"
    local_algo_options: dict[str, Any] | None = None
    n_sampling_points: PositiveInt = 128
    n_simplex_iterations: PositiveInt = 1
    sampling_method: Literal["simplicial", "halton", "sobol"] | Callable = "simplicial"
    max_sampling_evaluations: PositiveInt | None = None
    convergence_minimum_criterion_value: float | None = None
    convergence_minimum_criterion_tolerance: NonNegativeFloat = 1e-4
    stopping_maxiter: PositiveInt | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    stopping_max_processing_time: PositiveFloat | None = None
    minimum_homology_group_rank_differential: PositiveInt | None = None
    symmetry: List | bool = False
    minimize_every_iteration: bool = True
    max_local_minimizations_per_iteration: PositiveInt | bool = False
    infinity_constraints: bool = True

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.local_algorithm == "COBYLA":
            nonlinear_constraints = equality_as_inequality_constraints(
                problem.nonlinear_constraints
            )

        nonlinear_constraints = vector_as_list_of_scalar_constraints(
            problem.nonlinear_constraints
        )

        local_algo_options = (
            {} if self.local_algo_options is None else self.local_algo_options
        )
        default_minimizer_kwargs = {
            "method": self.local_algorithm,
            "bounds": _get_scipy_bounds(problem.bounds),
            "jac": problem.jac,
        }

        minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}
        options = {
            "maxfev": self.max_sampling_evaluations,
            "f_min": self.convergence_minimum_criterion_value,
            "f_tol": self.convergence_minimum_criterion_tolerance,
            "maxiter": self.stopping_maxiter,
            "maxev": self.stopping_maxfun,
            "maxtime": self.stopping_max_processing_time,
            "minhgrd": self.minimum_homology_group_rank_differential,
            "symmetry": self.symmetry,
            "jac": problem.jac,
            "minimize_every_iter": self.minimize_every_iteration,
            "local_iter": self.max_local_minimizations_per_iteration,
            "infty_constraints": self.infinity_constraints,
        }

        if any(options.values()) is False:
            options_used = None
        else:
            options_used = options

        res = scipy.optimize.shgo(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            constraints=nonlinear_constraints,
            minimizer_kwargs=minimizer_kwargs,
            n=self.n_sampling_points,
            iters=self.n_simplex_iterations,
            sampling_method=self.sampling_method,
            options=options_used,
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_dual_annealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyDualAnnealing(Algorithm):
    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    local_algorithm: (
        Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        | Callable
    ) = "L-BFGS-B"
    local_algo_options: dict[str, Any] | None = None
    # TODO: Refine type to add ranges (0.01, 5e4]
    initial_temperature: PositiveFloat = 5230.0
    # TODO: Refine type to add ranges (0,1)
    restart_temperature_ratio: PositiveFloat = 2e-05
    # TODO: Refine type to add ranges (1, 3]
    visit: PositiveFloat = 2.62
    # TODO: Refine type to add ranges (-1e4, -5]
    accept: NegativeFloat = -5.0
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    seed: int | np.random.Generator | np.random.RandomState | None = None
    no_local_search: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        local_algo_options = (
            {} if self.local_algo_options is None else self.local_algo_options
        )
        default_minimizer_kwargs = {
            "method": self.local_algorithm,
            "bounds": _get_scipy_bounds(problem.bounds),
            "jac": problem.jac,
        }

        minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}

        res = scipy.optimize.dual_annealing(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            maxiter=self.stopping_maxiter,
            minimizer_kwargs=minimizer_kwargs,
            initial_temp=self.initial_temperature,
            restart_temp_ratio=self.restart_temperature_ratio,
            visit=self.visit,
            accept=self.accept,
            maxfun=self.stopping_maxfun,
            seed=self.seed,
            no_local_search=self.no_local_search,
            x0=x0,
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_direct",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipyDirect(Algorithm):
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    locally_biased: bool = True
    convergence_minimum_criterion_value: float = -np.inf
    # TODO: must be between 0 and 1
    convergence_minimum_criterion_tolerance: NonNegativeFloat = 1e-4
    # TODO: must be between 0 and 1
    volume_hyperrectangle_tolerance: NonNegativeFloat = 1e-16
    # TODO: must be between 0 and 1
    length_hyperrectangle_tolerance: NonNegativeFloat = 1e-6

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = scipy.optimize.direct(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            eps=self.convergence_ftol_rel,
            maxfun=self.stopping_maxfun,
            maxiter=self.stopping_maxiter,
            locally_biased=self.locally_biased,
            f_min=self.convergence_minimum_criterion_value,
            f_min_rtol=self.convergence_minimum_criterion_tolerance,
            vol_tol=self.volume_hyperrectangle_tolerance,
            len_tol=self.length_hyperrectangle_tolerance,
        )

        return process_scipy_result(res)


def _get_workers(n_cores, batch_evaluator):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    out = functools.partial(
        batch_evaluator,
        n_cores=n_cores,
        error_handling="raise",
    )
    return out


def _get_scipy_bounds(bounds: InternalBounds) -> ScipyBounds:
    return ScipyBounds(lb=bounds.lower, ub=bounds.upper)


def process_scipy_result_old(scipy_results_obj):
    # using get with defaults to access dict elements is just a safety measure
    raw_res = {**scipy_results_obj}
    processed = {
        "solution_x": raw_res.get("x"),
        "solution_criterion": raw_res.get("fun"),
        "solution_derivative": raw_res.get("jac"),
        "solution_hessian": raw_res.get("hess"),
        "n_fun_evals": raw_res.get("nfev"),
        "n_jac_evals": raw_res.get("njac") or raw_res.get("njev"),
        "n_iterations": raw_res.get("nit"),
        "success": raw_res.get("success"),
        "reached_convergence_criterion": None,
        "message": raw_res.get("message"),
    }
    return processed

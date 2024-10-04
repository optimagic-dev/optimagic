"""Implement the fides optimizer."""

import logging
from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_FIDES_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_XTOL_ABS,
    STOPPING_MAXITER,
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

if IS_FIDES_INSTALLED:
    from fides import Optimizer, hessian_approximation


@mark.minimizer(
    name="fides",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_FIDES_INSTALLED,
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
class Fides(Algorithm):
    hessian_update_strategy: Literal[
        "bfgs",
        "bb",
        "bg",
        "dfp",
        "sr1",
    ] = "bfgs"
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    stopping_max_seconds: float = np.inf
    trustregion_initial_radius: PositiveFloat = 1.0
    trustregion_stepback_strategy: Literal[
        "truncate",
        "reflect",
        "reflect_single",
        "mixed",
    ] = "truncate"
    trustregion_subspace_dimension: Literal[
        "full",
        "2D",
        "scg",
    ] = "full"
    trustregion_max_stepback_fraction: float = 0.95
    trustregion_decrease_threshold: float = 0.25
    trustregion_increase_threshold: float = 0.75
    trustregion_decrease_factor: float = 0.25
    trustregion_increase_factor: float = 2.0

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = fides_internal(
            fun_and_jac=cast(
                Callable[[NDArray[np.float64]], NDArray[np.float64]],
                problem.fun_and_jac,
            ),
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            hessian_update_strategy=self.hessian_update_strategy,
            convergence_ftol_abs=self.convergence_ftol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_gtol_abs=self.convergence_gtol_abs,
            convergence_gtol_rel=self.convergence_gtol_rel,
            stopping_maxiter=self.stopping_maxiter,
            stopping_max_seconds=self.stopping_max_seconds,
            trustregion_initial_radius=self.trustregion_initial_radius,
            trustregion_stepback_strategy=self.trustregion_stepback_strategy,
            trustregion_subspace_dimension=self.trustregion_subspace_dimension,
            trustregion_max_stepback_fraction=self.trustregion_max_stepback_fraction,
            trustregion_decrease_threshold=self.trustregion_decrease_threshold,
            trustregion_increase_threshold=self.trustregion_increase_threshold,
            trustregion_decrease_factor=self.trustregion_decrease_factor,
            trustregion_increase_factor=self.trustregion_increase_factor,
        )

        return res


def fides_internal(
    fun_and_jac: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    lower_bounds: NDArray[np.float64] | None,
    upper_bounds: NDArray[np.float64] | None,
    hessian_update_strategy: Literal[
        "bfgs",
        "bb",
        "bg",
        "dfp",
        "sr1",
    ],
    convergence_ftol_abs: NonNegativeFloat,
    convergence_ftol_rel: NonNegativeFloat,
    convergence_xtol_abs: NonNegativeFloat,
    convergence_gtol_abs: NonNegativeFloat,
    convergence_gtol_rel: NonNegativeFloat,
    stopping_maxiter: PositiveInt,
    stopping_max_seconds: float,
    trustregion_initial_radius: PositiveFloat,
    trustregion_stepback_strategy: Literal[
        "truncate",
        "reflect",
        "reflect_single",
        "mixed",
    ],
    trustregion_subspace_dimension: Literal[
        "full",
        "2D",
        "scg",
    ],
    trustregion_max_stepback_fraction: float,
    trustregion_decrease_threshold: float,
    trustregion_increase_threshold: float,
    trustregion_decrease_factor: float,
    trustregion_increase_factor: float,
) -> InternalOptimizeResult:
    """Minimize a scalar function using the Fides Optimizer.

    For details see
    :ref: `fides_algorithm`.

    """
    if not IS_FIDES_INSTALLED:
        raise NotInstalledError(
            "The 'fides' algorithm requires the fides package to be installed. "
            "You can install it with `pip install fides>=0.7.4`."
        )

    fides_options = {
        "delta_init": trustregion_initial_radius,
        "eta": trustregion_increase_threshold,
        "fatol": convergence_ftol_abs,
        "frtol": convergence_ftol_rel,
        "gamma1": trustregion_decrease_factor,
        "gamma2": trustregion_increase_factor,
        "gatol": convergence_gtol_abs,
        "grtol": convergence_gtol_rel,
        "maxiter": stopping_maxiter,
        "maxtime": stopping_max_seconds,
        "mu": trustregion_decrease_threshold,
        "stepback_strategy": trustregion_stepback_strategy,
        "subspace_solver": trustregion_subspace_dimension,
        "theta_max": trustregion_max_stepback_fraction,
        "xtol": convergence_xtol_abs,
    }

    hessian_instance = _create_hessian_updater_from_user_input(hessian_update_strategy)

    opt = Optimizer(
        fun=fun_and_jac,
        lb=lower_bounds,
        ub=upper_bounds,
        verbose=logging.ERROR,
        options=fides_options,
        funargs=None,
        hessian_update=hessian_instance,
        resfun=False,
    )
    raw_res = opt.minimize(x)
    res = _process_fides_res(raw_res, opt)
    out = InternalOptimizeResult(
        x=res["solution_x"],
        fun=res["solution_criterion"],
        jac=res["solution_derivative"],
        hess=res["solution_hessian"],
        success=res["success"],
        message=res["message"],
        n_iterations=res["n_iterations"],
    )

    return out


def _process_fides_res(raw_res, opt):
    """Create an optimagic results dictionary from the Fides output.

    Args:
        raw_res (tuple): Tuple containing the Fides result
        opt (fides.Optimizer): Fides Optimizer after minimize has been called on it.

    """
    fval, x, grad, hess = raw_res
    res = {
        "solution_criterion": fval,
        "solution_x": x,
        "solution_derivative": grad,
        "solution_hessian": hess,
        "success": opt.converged,
        "n_iterations": opt.iteration,
        "message": _process_exitflag(opt.exitflag),
    }
    return res


def _process_exitflag(exitflag):
    messages = {
        "DID_NOT_RUN": "The optimizer did not run",
        "MAXITER": "Reached maximum number of allowed iterations",
        "MAXTIME": "Expected to reach maximum allowed time in next iteration",
        "NOT_FINITE": "Encountered non-finite fval/grad/hess",
        "EXCEEDED_BOUNDARY": "Exceeded specified boundaries",
        "DELTA_TOO_SMALL": "Trust Region Radius too small to proceed",
        "FTOL": "Converged according to fval difference",
        "XTOL": "Converged according to x difference",
        "GTOL": "Converged according to gradient norm",
    }

    out = messages.get(exitflag.name)

    return out


def _create_hessian_updater_from_user_input(hessian_update_strategy):
    hessians_needing_residuals = (
        hessian_approximation.FX,
        hessian_approximation.SSM,
        hessian_approximation.TSSM,
        hessian_approximation.GNSBFGS,
    )
    unsupported_hess_msg = (
        f"{hessian_update_strategy} not supported because it requires "
        "residuals. Choose one of 'BB', 'BFGS', 'BG', 'DFP' or 'SR1' or pass "
        "an instance of the fides.hessian_approximation.HessianApproximation "
        "class."
    )

    if hessian_update_strategy in ("broyden", "Broyden", "BROYDEN"):
        raise ValueError(
            "You cannot use the Broyden update strategy without specifying the "
            "interpolation parameter phi. Import the Broyden class from "
            "`fides.hessian_approximation`, create an instance of it with your "
            "desired value of phi and pass this instance instead."
        )
    elif isinstance(hessian_update_strategy, str):
        if hessian_update_strategy.lower() in ["fx", "ssm", "tssm", "gnsbfgs"]:
            raise NotImplementedError(unsupported_hess_msg)
        else:
            hessian_name = hessian_update_strategy.upper()
            hessian_class = getattr(hessian_approximation, hessian_name)
            hessian_instance = hessian_class()
    elif isinstance(
        hessian_update_strategy, hessian_approximation.HessianApproximation
    ):
        hessian_instance = hessian_update_strategy
        if isinstance(hessian_instance, hessians_needing_residuals):
            raise NotImplementedError(unsupported_hess_msg)
    else:
        raise TypeError(
            "You must provide a hessian_update_strategy that is either a string or an "
            "instance of the fides.hessian_approximation.HessianApproximation class."
        )
    return hessian_instance

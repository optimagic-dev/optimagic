"""Implement Berndt-Hall-Hall-Hausman (BHHH) algorithm."""

from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt


@mark.minimizer(
    name="bhhh",
    solver_type=AggregationLevel.LIKELIHOOD,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class BHHH(Algorithm):
    converence_gtol_abs: NonNegativeFloat = 1e-8
    # TODO: Why is this 200?
    stopping_maxiter: PositiveInt = 200

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = bhhh_internal(
            fun_and_jac=cast(
                Callable[[NDArray[np.float64]], NDArray[np.float64]],
                problem.fun_and_jac,
            ),
            x=x0,
            gtol_abs=self.converence_gtol_abs,
            maxiter=self.stopping_maxiter,
        )

        return res


def bhhh_internal(
    fun_and_jac: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    gtol_abs: NonNegativeFloat,
    maxiter: PositiveInt,
) -> InternalOptimizeResult:
    r"""Minimize a sum of contributions using the BHHH algorithm.

    With contributions :math:`f_i(x)`, Jacobian :math:`J` of shape (n_obs, n_params),
    gradient :math:`g = J^\top 1` and Hessian approximation :math:`H = J^\top J`, the
    search direction is the descent direction :math:`d = -H^{-1} g`.

    Args:
        fun_and_jac: Function that returns the contributions of shape (n_obs,) and
            their Jacobian of shape (n_obs, n_params). The sum of contributions is
            minimized.
        x: Start parameters of shape (n_params,).
        gtol_abs: Stop if :math:`g^\top H^{-1} g` falls below this value.
        maxiter: Maximum number of iterations.

    Returns:
        The result of the optimization.

    """
    criterion_accepted, jacobian = fun_and_jac(x)
    x_accepted = x
    direction, gtol = _get_direction(jacobian)

    initial_step_size = 1.0
    min_step_size = 1e-10
    step_size = initial_step_size
    success = gtol < gtol_abs
    message = "Converged." if success else "Maximum number of iterations reached."

    niter = 1
    while niter < maxiter and not success:
        niter += 1

        x_candidate = x_accepted + step_size * direction
        criterion_candidate, jacobian_candidate = fun_and_jac(x_candidate)

        # Line search: halve the step until the criterion does not increase. The
        # Jacobian at a rejected candidate is never used.
        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            step_size /= 2
            if step_size < min_step_size:
                message = "Line search did not find a decrease of the criterion."
                break

        else:
            x_accepted = x_candidate
            criterion_accepted = criterion_candidate
            direction, gtol = _get_direction(jacobian_candidate)
            step_size = initial_step_size
            if gtol < gtol_abs:
                success = True
                message = "Converged."

    res = InternalOptimizeResult(
        x=x_accepted,
        fun=criterion_accepted,
        success=success,
        message=message,
        n_iterations=niter,
    )

    return res


def _get_direction(
    jacobian: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    r"""Compute the BHHH search direction and the convergence criterion.

    Args:
        jacobian: Jacobian of the contributions of shape (n_obs, n_params).

    Returns:
        The descent direction :math:`d = -H^{-1} g` of shape (n_params,), with
        :math:`H = J^\top J` and :math:`g = J^\top 1`, and :math:`g^\top H^{-1} g`.

    """
    hessian_approx = jacobian.T @ jacobian
    gradient = jacobian.sum(axis=0)
    direction = -np.linalg.solve(hessian_approx, gradient)
    gtol = float(-gradient @ direction)
    return direction, gtol

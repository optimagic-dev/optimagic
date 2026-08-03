"""Implement Berndt-Hall-Hall-Hausman (BHHH) algorithm."""

from __future__ import annotations

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
    """Minimize a likelihood function using the BHHH algorithm.

    BHHH (:cite:`Berndt1974`) can - and should ONLY - be used for minimizing (or
    maximizing) a likelihood function. It is similar to the Newton-Raphson algorithm,
    but replaces the Hessian matrix with the outer product of the gradients of the
    likelihood contributions. This approximation is based on the information matrix
    equality (:cite:`Halbert1982`) and is thus only valid when minimizing (or
    maximizing) a likelihood function. In exchange, the approximated Hessian is
    always positive semidefinite and no second derivatives are needed.

    To use bhhh, the objective function must be a likelihood function, i.e. a
    function that is decorated with ``om.mark.likelihood`` and returns the
    likelihood contributions of each observation rather than their sum.

    The algorithm is a local optimizer that uses first derivatives of the
    likelihood contributions. It does not support bounds or constraints.

    .. note::
        This is a pure-Python implementation within optimagic. It is currently
        considered experimental.

    """

    converence_gtol_abs: NonNegativeFloat = 1e-8
    """Stopping criterion for the gradient tolerance.

    The algorithm converges when the inner product of the aggregated gradient and
    the candidate search direction falls below this value.

    """

    # TODO: Why is this 200?
    stopping_maxiter: PositiveInt = 200
    """Maximum number of iterations.

    If reached, terminate.

    """

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
    """Minimize a likelihood function using the BHHH algorithm.

    Args:
        criterion_and_derivative: The objective function to be minimized.
        x: Initial guess of the parameter vector (starting points).
        convergence_absolute_gradient_tolerance: Stopping criterion for the
            gradient tolerance.
        stopping_max_iterations: Maximum number of iterations. If reached,
            terminate.

    Returns:
        InternalOptimizeResult: The result of the optimization.

    """
    criterion_accepted, gradient = fun_and_jac(x)
    x_accepted = x

    hessian_approx = np.dot(gradient.T, gradient)
    gradient_sum = np.sum(gradient, axis=0)
    direction = np.linalg.solve(hessian_approx, gradient_sum)
    gtol = np.dot(gradient_sum, direction)

    initial_step_size = 1.0
    step_size = initial_step_size

    niter = 1
    while niter < maxiter:
        niter += 1

        x_candidate = x_accepted + step_size * direction
        criterion_candidate, gradient = fun_and_jac(x_candidate)

        # If previous step was accepted
        if step_size == initial_step_size:
            hessian_approx = np.dot(gradient.T, gradient)

        else:
            criterion_candidate, gradient = fun_and_jac(x_candidate)

        # Line search
        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            step_size /= 2

            if step_size <= 0.01:
                # Accept step
                x_accepted = x_candidate
                criterion_accepted = criterion_candidate

                # Reset step size
                step_size = initial_step_size

        # If decrease in likelihood, calculate new direction vector
        else:
            # Accept step
            x_accepted = x_candidate
            criterion_accepted = criterion_candidate

            gradient_sum = np.sum(gradient, axis=0)
            direction = np.linalg.solve(hessian_approx, gradient_sum)
            gtol = np.dot(gradient_sum, direction)

            if gtol < 0:
                hessian_approx = np.dot(gradient.T, gradient)
                direction = np.linalg.solve(hessian_approx, gradient_sum)

            # Reset stepsize
            step_size = initial_step_size

        if gtol < gtol_abs:
            break

    res = InternalOptimizeResult(
        x=x_accepted,
        fun=criterion_accepted,
        message="Under development",
        n_iterations=niter,
    )

    return res

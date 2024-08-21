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
    supports_parallelism=False,
    supports_bounds=False,
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

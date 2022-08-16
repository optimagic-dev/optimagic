"""Implement Berndt-Hall-Hall-Hausman (BHHH) algorithm."""
import numpy as np
from estimagic.decorators import mark_minimizer


@mark_minimizer(
    name="bhhh",
    primary_criterion_entry="contributions",
    needs_scaling=False,
    is_available=True,
)
def bhhh(
    criterion_and_derivative,
    x,
    *,
    convergence_absolute_gradient_tolerance=1e-8,
    stopping_max_iterations=200,
):
    """Minimize a likelihood function using the BHHH algorithm.

    For details, see :ref:`_own_algorithms`.
    """
    result_dict = bhhh_internal(
        criterion_and_derivative,
        x=x,
        convergence_absolute_gradient_tolerance=convergence_absolute_gradient_tolerance,
        stopping_max_iterations=stopping_max_iterations,
    )

    return result_dict


def bhhh_internal(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    convergence_absolute_gradient_tolerance,
    stopping_max_iterations,
):
    """Minimize a likelihood function using the BHHH algorithm.

    Args:
        criterion_and_derivative (callable): The objective function to be minimized.
        x (np.ndarray): Initial guess of the parameter vector x (starting points).
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        convergence_absolute_gradient_tolerance (float): Stopping criterion for the
            gradient tolerance.
        stopping_max_iterations (int): Maximum number of iterations. If reached,
            terminate.

    Returns:
        (dict) Result dictionary containing:

        - solution_x (np.ndarray): Solution vector of shape (n,).
        - solution_criterion (np.ndarray): Likelihood at the solution. Shape (n_obs,).
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution vector or reaching stopping_max_iterations.
        - message (str): Message to the user. Currently it says: "Under development."
    """
    criterion_accepted, gradient_candidate = criterion_and_derivative(x)
    x_accepted = x

    hessian_approx = gradient_candidate.T @ gradient_candidate
    direction, gtol = _calculate_new_direction_vector(
        gradient_candidate, hessian_approx
    )

    initial_step_size = 1
    step_size = initial_step_size

    for _n_iter in range(stopping_max_iterations):
        (
            x_candidate,
            criterion_candidate,
            gradient_candidate,
            hessian_approx,
        ) = find_new_candidates(
            x_accepted,
            direction,
            hessian_approx,
            lower_bounds,
            upper_bounds,
            step_size,
            initial_step_size,
            criterion_and_derivative,
        )

        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            x_accepted, criterion_accepted, step_size = determine_new_step_size(
                x_accepted,
                criterion_accepted,
                x_candidate,
                criterion_candidate,
                step_size,
                initial_step_size,
            )

        else:
            x_accepted = x_candidate
            criterion_accepted = criterion_candidate

            direction, hessian_approx, gtol = determine_new_search_direction(
                gradient_candidate,
                hessian_approx,
            )

            step_size = initial_step_size

        if gtol < convergence_absolute_gradient_tolerance:
            break

    result_dict = {
        "solution_x": x_accepted,
        "solution_criterion": criterion_accepted,
        "n_iterations": _n_iter,
        "message": "Under development",
    }

    return result_dict


def find_new_candidates(
    x_accepted,
    direction,
    hessian_approx,
    lower_bounds,
    upper_bounds,
    step_size,
    initial_step_size,
    criterion_and_derivative,
):
    """Find new candidates for x, criterion, gradient, and hessian."""
    x_candidate = x_accepted + step_size * direction
    x_candidate = _apply_bounds_to_x_candidate(x_candidate, lower_bounds, upper_bounds)

    criterion_candidate, gradient_candidate = criterion_and_derivative(x_candidate)

    if step_size == initial_step_size:
        hessian_approx = gradient_candidate.T @ gradient_candidate
    else:
        criterion_candidate, gradient_candidate = criterion_and_derivative(x_candidate)

    return x_candidate, criterion_candidate, gradient_candidate, hessian_approx


def determine_new_step_size(
    x_accepted,
    criterion_accepted,
    x_candidate,
    criterion_candidate,
    step_size,
    initial_step_size,
):
    """Determine new step size and accept candidates."""
    step_size /= 2

    if step_size <= 0.01:
        x_accepted = x_candidate
        criterion_accepted = criterion_candidate

        step_size = initial_step_size

    return x_accepted, criterion_accepted, step_size


def determine_new_search_direction(
    gradient_candidate,
    hessian_approx,
):
    """Determine new search direction and accept candidates."""
    direction, gtol = _calculate_new_direction_vector(
        gradient_candidate, hessian_approx
    )

    if gtol < 0:
        hessian_approx = gradient_candidate.T @ gradient_candidate
        direction, _ = _calculate_new_direction_vector(
            gradient_candidate, hessian_approx
        )

    return direction, hessian_approx, gtol


def _calculate_new_direction_vector(gradient_candidate, hessian_approx):
    """Calculate new direction vector."""
    gradient_sum = np.sum(gradient_candidate, axis=0)
    direction = np.linalg.solve(hessian_approx, gradient_sum)
    gtol = gradient_sum @ direction

    return direction, gtol


def _apply_bounds_to_x_candidate(x, lower_bounds, upper_bounds, bound_tol=0):
    """Apply upper and lower bounds to the candidate vector."""
    x = np.where(x <= lower_bounds + bound_tol, lower_bounds, x)
    x = np.where(x >= upper_bounds - bound_tol, upper_bounds, x)

    return x

"""Implement the Berndt-Hall-Hall-Hausman (BHHH) algorithm."""
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
    lower_bounds,
    upper_bounds,
    *,
    convergence_absolute_gradient_tolerance=1e-8,
    convergence_relative_gradient_tolerance=1e-8,
    stopping_max_iterations=200,
):
    """Minimize a likelihood function using the bounded BHHH algorithm.

    For details, see :ref:`_own_algorithms`.
    """
    result_dict = bhhh_internal(
        criterion_and_derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        convergence_absolute_gradient_tolerance=convergence_absolute_gradient_tolerance,
        convergence_relative_gradient_tolerance=convergence_relative_gradient_tolerance,
        stopping_max_iterations=stopping_max_iterations,
    )

    return result_dict


def bhhh_internal(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    convergence_absolute_gradient_tolerance,
    convergence_relative_gradient_tolerance,
    stopping_max_iterations,
):
    """Minimize a likelihood function using the bounded BHHH algorithm.

    Args:
        criterion_and_derivative (callable): A function returning the tuple:
            - criterion (np.ndarray): Likelihood contributions of shape (n_obs,)
            - derivative (np.ndarray): Jacobian matrix of shape (n_obs, n_params)
        x (np.ndarray): Initial guess of the parameter vector x (starting points).
        lower_bounds (np.ndarray): 1d array of shape (n_params,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n_params,) with upper bounds
            for the parameter vector x
        convergence_absolute_gradient_tolerance (float): Stopping criterion for the
            absolute gradient tolerance.
        convergence_relative_gradient_tolerance (float): Stopping criterion for the
            relative gradient tolerance.
        stopping_max_iterations (int): Maximum number of iterations. If reached,
            terminate.

    Returns:
        (dict) Result dictionary containing:

        - solution_x (np.ndarray): Solution vector of shape (n_params,).
        - solution_criterion (np.ndarray): Likelihood contributions at the solution.
            Shape (n_obs,).
        - solution_loglikelihood (float): Value of the log-likelihood at the solution.
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution vector or reaching stopping_max_iterations.
        - message (str): Message to the user. Currently it says: "Under development."
    """
    tol = {
        "abs": convergence_absolute_gradient_tolerance,
        "rel": convergence_relative_gradient_tolerance,
    }
    n_params = len(x)

    x_accepted = x
    criterion_accepted, jacobian = criterion_and_derivative(x_accepted)
    gradient = np.sum(jacobian, axis=0)

    norm_pg0 = np.linalg.norm(
        x_accepted - np.clip(x_accepted - gradient, lower_bounds, upper_bounds)
    )
    norm_pg = norm_pg0

    inactive_set = estimate_epsilon_inactive_set(
        x_accepted, norm_pg, lower_bounds, upper_bounds
    )

    gradient_reduced = gradient[inactive_set]
    hessian_reduced = jacobian[:, inactive_set].T @ jacobian[:, inactive_set]
    direction = determine_descent_direction(
        gradient, gradient_reduced, hessian_reduced, inactive_set, n_params
    )

    initial_step_size = 1
    step_size = initial_step_size

    for _n_iter in range(stopping_max_iterations):

        x_candidate = np.clip(
            x_accepted + step_size * direction, lower_bounds, upper_bounds
        )

        criterion_candidate, jacobian = criterion_and_derivative(x_candidate)

        if step_size == initial_step_size:
            hessian_reduced = jacobian[:, inactive_set].T @ jacobian[:, inactive_set]

        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            step_size /= 2

            if step_size <= 0.01:
                x_accepted = x_candidate
                criterion_accepted = criterion_candidate

                step_size = initial_step_size
        else:
            x_accepted = x_candidate
            criterion_accepted = criterion_candidate

            gradient = np.sum(jacobian, axis=0)
            gradient_reduced = gradient[inactive_set]

            direction = determine_descent_direction(
                gradient,
                gradient_reduced,
                hessian_reduced,
                inactive_set,
                n_params,
            )

            norm_pg = np.linalg.norm(
                x_accepted - np.clip(x_accepted - gradient, lower_bounds, upper_bounds)
            )
            inactive_set = estimate_epsilon_inactive_set(
                x_accepted, norm_pg, lower_bounds, upper_bounds
            )

            if norm_pg < 0:
                gradient_reduced = gradient[inactive_set]
                hessian_reduced = (
                    jacobian[:, inactive_set].T @ jacobian[:, inactive_set]
                )

                direction = determine_descent_direction(
                    gradient,
                    gradient_reduced,
                    hessian_reduced,
                    inactive_set,
                    n_params,
                )

            step_size = initial_step_size

        if norm_pg < tol["abs"] + tol["rel"] * norm_pg0:
            break

    result_dict = {
        "solution_x": x_accepted,
        "solution_criterion": criterion_accepted,
        "solution_loglikelihood": np.sum(criterion_accepted),
        "n_iterations": _n_iter,
        "message": "Under develpment",
    }

    return result_dict


def estimate_epsilon_inactive_set(x, norm_gradient, lower_bounds, upper_bounds):
    """Estimate the set of epsilon-inactive bound constraints up to a tolerance.

    The set of epsilon-inactive indices underestimates (overestimates) the actual
    set of inactive (active) indices.

        x (np.ndarray): Current candidate vector.
        norm_gradient (float): Norm of the projected gradient.
        lower_bounds (np.ndarray): 1d array of shape (n_params,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n_params,) with upper bounds
            for the parameter vector x
    """
    epsilon = min(np.min(upper_bounds - lower_bounds) / 2, norm_gradient)

    inactive_set = np.where(
        (lower_bounds + epsilon < x) & (x < upper_bounds - epsilon)
    )[0]

    return inactive_set


def determine_descent_direction(
    gradient_candidate, gradient_reduced, hessian_reduced, inactive_set, n_params
):
    """Determine the new descent serach direction."""
    direction = np.linalg.solve(hessian_reduced, gradient_reduced)

    direction_active = gradient_candidate.copy()
    direction_active[inactive_set] = 0

    direction_projected = np.zeros(n_params)
    direction_projected[inactive_set] = direction

    direction_all = direction_active + direction_projected

    return direction_all

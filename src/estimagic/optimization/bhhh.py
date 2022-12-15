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
    convergence_relative_params_tolerance=1e-8,
    stopping_max_iterations=200,
):
    """Minimize a likelihood function using the box-constrained BHHH algorithm.

    For details, see :ref:`_own_algorithms`.
    """
    if np.isinf(lower_bounds).all() and np.isinf(upper_bounds).all():
        result_dict = bhhh_unconstrained(
            criterion_and_derivative,
            x,
            convergence_relative_params_tolerance,
            convergence_absolute_gradient_tolerance,
            stopping_max_iterations,
        )
    else:
        result_dict = bhhh_box_constrained(
            criterion_and_derivative,
            x,
            lower_bounds,
            upper_bounds,
            convergence_relative_params_tolerance,
            convergence_absolute_gradient_tolerance,
            stopping_max_iterations,
        )

    return result_dict


def bhhh_unconstrained(
    criterion_and_derivative,
    x,
    convergence_relative_params_tolerance,
    convergence_absolute_gradient_tolerance,
    stopping_max_iterations,
):
    """Minimize a likelihood function using the unconstrained BHHH algorithm.

    This implementation is based on the Matlab code by Fedor Iskhakov, which can
    be found on his `Github page <https://github.com/fediskhakov/MPECvsNFXP/blob/
    8212346a0426f54977823bd754947f528ed749ba/nfxp_Rust87/nfxp.m#L315>`.

    Args:
        criterion_and_derivative (callable): A function returning the tuple:
            - criterion (np.ndarray): Likelihood contributions of shape (n_obs,)
            - derivative (np.ndarray): Jacobian matrix of shape (n_obs, n_params)
        x (np.ndarray): Initial guess of the parameter vector x (starting points).
        lower_bounds (np.ndarray): 1d array of shape (n_params,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n_params,) with upper bounds
            for the parameter vector x
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_absolute_gradient_tolerance (float): Stop if all elements of the
            gradient are smaller than this.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: Result dictionary containing:

        - solution_x (np.ndarray): Solution vector of shape (n_params,).
        - solution_criterion (np.ndarray): Likelihood contributions at the solution.
            Shape (n_obs,).
        - solution_loglikelihood (float): Value of the log-likelihood at the solution.
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution vector or reaching stopping_max_iterations.
        - message (str): Message to the user. Currently it says: "Under development."
    """
    _zero_threshold = np.finfo(float).eps

    criterion_accepted, jacobian = criterion_and_derivative(x)

    gradient = np.sum(jacobian, axis=0)
    hessian_approx = jacobian.T @ jacobian
    direction = np.linalg.solve(hessian_approx, gradient)

    norm_grad = gradient @ direction

    initial_step_size = 1
    step_size = initial_step_size

    x_candidate = x + step_size * direction

    converged = False
    for _n_iter in range(stopping_max_iterations):

        criterion_candidate, jacobian = criterion_and_derivative(x_candidate)

        if step_size == initial_step_size:
            hessian_approx = jacobian.T @ jacobian

        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            step_size /= 2

            if step_size <= 0.01:
                x = x_candidate
                criterion_accepted = criterion_candidate

                step_size = initial_step_size
        else:
            x = x_candidate
            criterion_accepted = criterion_candidate

            gradient = np.sum(jacobian, axis=0)
            direction = np.linalg.solve(hessian_approx, gradient)

            norm_grad = gradient @ direction

            if norm_grad < 0:
                hessian_approx = jacobian.T @ jacobian
                direction = np.linalg.solve(hessian_approx, gradient)

            step_size = initial_step_size

        x_candidate = x + step_size * direction

        relative_params_difference = np.max(
            np.abs(((x_candidate - x)) / (x + _zero_threshold)) / step_size
        )

        if relative_params_difference < convergence_relative_params_tolerance:
            converged = True
            break
        elif norm_grad < convergence_absolute_gradient_tolerance:
            converged = True
            break

    result_dict = {
        "solution_x": x,
        "solution_criterion": criterion_accepted,
        "solution_loglikelihood": np.sum(criterion_accepted),
        "converged": converged,
        "relative_params_difference": relative_params_difference,
        "absolute_gradient_norm": norm_grad,
        "n_iterations": _n_iter,
        "message": "Under develpment",
    }

    return result_dict


def bhhh_box_constrained(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    convergence_relative_params_tolerance,
    convergence_absolute_gradient_tolerance,
    stopping_max_iterations,
):
    """Minimize a likelihood function using the box-constrained BHHH algorithm.

    The (in)active constraints are identified via an epsilon-active-set method,
    similar to the approach used by the Projected BFGS-Armijo algorithm
    (see :cite:`Kelley1999`, p. 97).

    The unconstrained version of this algorithm is based on the Matlab implementation
    by Adam E. Theising. See `Assignment 4 <https://adamtheising.github.io/aae637>`
    on Maximum Likelihood Estimation on the course page of Applied Econometrics II.


    Args:
        criterion_and_derivative (callable): A function returning the tuple:
            - criterion (np.ndarray): Likelihood contributions of shape (n_obs,)
            - derivative (np.ndarray): Jacobian matrix of shape (n_obs, n_params)
        x (np.ndarray): Initial guess of the parameter vector x (starting points).
        lower_bounds (np.ndarray): 1d array of shape (n_params,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n_params,) with upper bounds
            for the parameter vector x
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_absolute_gradient_tolerance (float): Stop if all elements of the
            projected gradient are smaller than this.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: Result dictionary containing:

        - solution_x (np.ndarray): Solution vector of shape (n_params,).
        - solution_criterion (np.ndarray): Likelihood contributions at the solution.
            Shape (n_obs,).
        - solution_loglikelihood (float): Value of the log-likelihood at the solution.
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution vector or reaching stopping_max_iterations.
        - message (str): Message to the user. Currently it says: "Under development."
    """
    n_params = len(x)
    min_step_size = 1e-6
    _zero_threshold = np.finfo(float).eps

    jacobian = criterion_and_derivative(x, task="derivative")
    gradient = np.sum(jacobian, axis=0)

    norm_proj_grad = np.linalg.norm(
        x - np.clip(x - gradient, lower_bounds, upper_bounds)
    )
    inactive_set = _estimate_epsilon_inactive_set(
        x, norm_proj_grad, lower_bounds, upper_bounds
    )

    converged = False
    for _n_iter in range(stopping_max_iterations):
        jacobian = criterion_and_derivative(x, task="derivative")
        gradient = np.sum(jacobian, axis=0)

        gradient_reduced = gradient[inactive_set]
        hessian_approx = jacobian.T @ jacobian
        hessian_reduced = hessian_approx[inactive_set[:, np.newaxis], inactive_set]

        direction_projected = _determine_descent_direction(
            gradient, gradient_reduced, hessian_reduced, inactive_set, n_params
        )

        step_size = _find_optimal_step_size(
            x,
            direction_projected,
            lower_bounds,
            upper_bounds,
            min_step_size,
            criterion_and_derivative,
        )

        x_candidate = np.clip(
            x + step_size * direction_projected, lower_bounds, upper_bounds
        )
        relative_params_difference = np.max(
            np.abs(((x_candidate - x)) / (x + _zero_threshold)) / step_size
        )
        x = x_candidate

        norm_proj_grad = np.linalg.norm(
            x - np.clip(x - gradient, lower_bounds, upper_bounds)
        )

        if relative_params_difference < convergence_relative_params_tolerance:
            converged = True
            break
        elif norm_proj_grad < convergence_absolute_gradient_tolerance:
            converged = True
            break

        inactive_set = _estimate_epsilon_inactive_set(
            x, norm_proj_grad, lower_bounds, upper_bounds
        )

    solution_criterion = criterion_and_derivative(x, task="criterion")

    result_dict = {
        "solution_x": x,
        "solution_criterion": solution_criterion,
        "solution_loglikelihood": np.sum(solution_criterion),
        "converged": converged,
        "relative_params_difference": relative_params_difference,
        "absolute_gradient_norm": norm_proj_grad,
        "n_iterations": _n_iter,
        "message": "Under develpment",
    }

    return result_dict


def _estimate_epsilon_inactive_set(x, norm_grad, lower_bounds, upper_bounds):
    """Estimate the set of epsilon-inactive bound constraints.

    The set of epsilon-inactive indices underestimates (overestimates) the actual
    set of inactive (active) indices.

    Args:
        x (np.ndarray): Current parameter vector of shape (n_params,).
        norm_grad (float): Norm of the projected gradient.
        lower_bounds (np.ndarray): 1d array of shape (n_params,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n_params,) with upper bounds
            for the parameter vector x/

    Returns:
        np.ndarray: 1d array of shape (n_inactive_constraints,) containing the set
            of inactive constraints.
    """
    epsilon = min(np.min(upper_bounds - lower_bounds) / 2, norm_grad)

    inactive_set = np.where(
        (lower_bounds + epsilon < x) & (x < upper_bounds - epsilon)
    )[0]

    return inactive_set


def _determine_descent_direction(
    gradient_candidate, gradient_reduced, hessian_reduced, inactive_set, n_params
):
    """Determine the new descent (search) direction."""
    direction = np.linalg.solve(hessian_reduced, gradient_reduced)

    direction_active = gradient_candidate.copy()
    direction_active[inactive_set] = 0

    direction_projected = np.zeros(n_params)
    direction_projected[inactive_set] = direction

    direction_all = direction_active + direction_projected

    return direction_all


def _find_optimal_step_size(
    x,
    direction_projected,
    lower_bounds,
    upper_bounds,
    min_step_size,
    criterion_and_derivative,
):
    """Find optimal step length."""
    step_size_trial = 2
    step_size_optimal = 1

    loglike_full_step = 1
    loglike_half_step = 0

    while (loglike_full_step > loglike_half_step) & (step_size_trial >= min_step_size):
        step_size_trial /= 2

        criterion_full_step = criterion_and_derivative(
            np.clip(
                x + step_size_trial * direction_projected, lower_bounds, upper_bounds
            ),
            task="criterion",
        )
        criterion_half_step = criterion_and_derivative(
            np.clip(
                x + (step_size_trial / 2) * direction_projected,
                lower_bounds,
                upper_bounds,
            ),
            task="criterion",
        )

        loglike_full_step = np.sum(criterion_full_step)
        loglike_half_step = np.sum(criterion_half_step)

        step_size_optimal = step_size_trial

    return step_size_optimal
